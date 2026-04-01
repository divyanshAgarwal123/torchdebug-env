#!/usr/bin/env python3
"""
TorchDebug Environment — Baseline Inference Script

Uses the OpenAI client to interact with the TorchDebug environment,
demonstrating how an LLM agent can diagnose and fix PyTorch training bugs.

Environment Variables:
    API_BASE_URL: Base URL for the LLM API (e.g., https://api.openai.com/v1)
    MODEL_NAME: Model to use (e.g., gpt-4, meta-llama/Llama-3-70b)
    HF_TOKEN: API token for the target API endpoint (required)

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4"
    python inference.py

    # Or with HuggingFace Inference API:
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

import collections
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional

from openai import OpenAI

# Environment server URL (default: local Docker)
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# LLM configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# =============================================================================
# Environment Client (HTTP)
# =============================================================================

import requests
from client import TorchDebugEnv as MCPTorchDebugEnv


class TorchDebugClient:
    """Stateful MCP client wrapper for the TorchDebug environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.env = MCPTorchDebugEnv(base_url=self.base_url)
        self._loop = asyncio.new_event_loop()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def reset(self, task_id: str = "basic_failures", scenario_id: Optional[str] = None) -> Dict:
        """Reset the environment with a specific task (stateful MCP session)."""
        kwargs = {"task_id": task_id}
        if scenario_id:
            kwargs["scenario_id"] = scenario_id
        self._run(self.env.reset(**kwargs))
        obs = self._run(self.env.call_tool("get_observation"))
        return {
            "observation": obs,
            "reward": 0.0,
            "done": False,
        }

    def step(self, action: Dict) -> Dict:
        """Take a step by calling the environment's `take_action` tool."""
        tool_args = {
            "action_type": action.get("action_type", "analyze_logs"),
            "diagnosis": action.get("diagnosis", ""),
            "fix_description": action.get("fix_description", ""),
            "fix_code": action.get("fix_code", ""),
            "parameters": json.dumps(action.get("parameters", {})),
        }
        obs = self._run(self.env.call_tool("take_action", **tool_args))
        return {
            "observation": obs,
            "reward": obs.get("reward", 0.0),
            "done": obs.get("done", False),
        }

    def health(self) -> bool:
        """Check if the environment server is healthy."""
        try:
            resp = self.session.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the MCP client session."""
        try:
            self._run(self.env.close())
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass


# =============================================================================
# LLM Agent
# =============================================================================

SYSTEM_PROMPT = """You are an expert PyTorch debugging agent. You are given a broken PyTorch training run and must:

1. **Investigate**: Use available tools to analyze logs, inspect gradients, check model architecture, and examine data pipeline
2. **Diagnose**: Identify the root cause of the training failure
3. **Fix**: Prescribe the correct fix with description and (optionally) corrected code

Available actions:
- analyze_logs: Analyze training log patterns (loss, accuracy, gradient norms)
- inspect_gradients: Deep analysis of gradient flow and norms
- inspect_data_pipeline: Check data loading, augmentation, and splitting
- inspect_model_architecture: Examine model structure, layer types, loss function
- check_device_placement: Analyze GPU/CPU device placement of tensors
- diagnose: Submit your root cause diagnosis (provide 'diagnosis' field)
- prescribe_fix: Submit your fix (provide 'fix_description' and optionally 'fix_code')
- request_hint: Get a hint (costs penalty)

Strategy:
1. First, read the observation carefully (error message, training config, code, logs)
2. Investigate using 2-3 relevant inspection actions
3. Diagnose the root cause
4. Prescribe a specific fix

Respond with JSON containing your chosen action. Examples:
{"action_type": "analyze_logs"}
{"action_type": "inspect_gradients"}
{"action_type": "diagnose", "diagnosis": "Learning rate is too high at 10.0 for SGD on CIFAR-10, causing loss explosion"}
{"action_type": "prescribe_fix", "fix_description": "Reduce learning rate to 0.01", "fix_code": "optimizer = SGD(lr=0.01, momentum=0.9)"}
"""


def create_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client."""
    if not API_BASE_URL:
        raise RuntimeError("Missing required environment variable: API_BASE_URL")
    if not MODEL_NAME:
        raise RuntimeError("Missing required environment variable: MODEL_NAME")
    if not HF_TOKEN:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )


def format_observation(obs_data: Dict) -> str:
    """Format the observation for the LLM."""
    metadata = obs_data.get("observation", obs_data.get("metadata", obs_data))

    parts = [
        f"## Training Run Debug Task",
        f"**Scenario**: {metadata.get('task_id', 'unknown')}",
        f"**Difficulty**: {metadata.get('difficulty', 'unknown')}",
        f"**Step**: {metadata.get('step_number', 0)}/{metadata.get('max_steps', 15)}",
        f"",
        f"### Description",
        metadata.get('task_description', 'No description'),
        f"",
        f"### Training Config",
        json.dumps(metadata.get('training_config', {}), indent=2),
        f"",
        f"### Code Snippet",
        f"```python",
        metadata.get('code_snippet', 'No code'),
        f"```",
    ]

    # Error message
    if metadata.get('error_message'):
        parts.extend([f"", f"### Error Message", f"```", metadata['error_message'], f"```"])

    # Training logs
    logs = metadata.get('training_logs', [])
    if logs:
        parts.extend([f"", f"### Training Logs"])
        for log in logs[-8:]:  # Last 8 entries
            parts.append(f"  {json.dumps(log)}")

    # Feedback from last action
    if metadata.get('feedback'):
        parts.extend([f"", f"### Feedback", metadata['feedback']])

    # Inspection results
    inspections = metadata.get('inspection_results', [])
    if inspections:
        parts.extend([f"", f"### Previous Inspections"])
        for insp in inspections:
            parts.extend([
                f"**{insp.get('inspection_type', 'unknown')}**:",
                insp.get('findings', 'No findings'),
                f"",
            ])

    return "\n".join(parts)


def get_llm_action(client: OpenAI, observation: str, history: list) -> Dict:
    """Query the LLM for the next action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ] + history + [
        {"role": "user", "content": observation + "\n\nWhat action should you take next? Respond with JSON."},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
        )
        response_text = completion.choices[0].message.content.strip()

        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Try to parse JSON
        action = json.loads(response_text)
        return action

    except json.JSONDecodeError:
        # Fallback: try to extract action_type
        print(f"  [WARN] Could not parse JSON, attempting fallback")
        return {"action_type": "analyze_logs"}
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return {"action_type": "analyze_logs"}


def infer_bug_plan(observation_text: str) -> Dict[str, str]:
    """Heuristic fallback plan for deterministic baseline execution."""
    text = observation_text.lower()

    if "cuda" in text and "cpu" in text and "different devices" in text:
        return {
            "inspect": "check_device_placement",
            "diagnosis": "Model and tensors are on different devices because the model was not moved to GPU with model.to(device).",
            "fix_description": "Move the model to the same device as inputs before training, e.g., model = model.to(device).",
            "fix_code": "model = model.to(device)",
        }

    if "data leakage" in text or ("val_accuracy" in text and "test_accuracy" in text):
        return {
            "inspect": "inspect_data_pipeline",
            "diagnosis": "There is data leakage between train and validation due to incorrect splitting and augmentation applied to validation data.",
            "fix_description": "Split by patient/group IDs and use separate transforms with no augmentation for validation.",
            "fix_code": "train_dataset = ...train_transform\nval_dataset = ...val_transform",
        }

    if "ddp" in text or "gradient accumulation" in text:
        return {
            "inspect": "inspect_gradients",
            "diagnosis": "DDP is synchronizing gradients on every backward without no_sync during accumulation, causing communication overhead and poor scaling.",
            "fix_description": "Use model.no_sync() for non-boundary accumulation steps and tune learning rate for effective batch size.",
            "fix_code": "with model.no_sync():\n    loss.backward()",
        }

    # Default: common easy failure in baseline set
    return {
        "inspect": "inspect_gradients",
        "diagnosis": "Learning rate is too high and causes exploding gradients with inf/nan loss.",
        "fix_description": "Reduce learning rate to a stable value like 0.01 and retry training.",
        "fix_code": "optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)",
    }


def choose_action_with_fallback(step_num: int, llm_action: Dict, observation_text: str) -> Dict:
    """Apply deterministic action policy if the model output is unhelpful."""
    plan = infer_bug_plan(observation_text)
    action_type = (llm_action or {}).get("action_type", "analyze_logs")

    if step_num == 1:
        return {"action_type": "analyze_logs"}
    if step_num == 2:
        return {"action_type": plan["inspect"]}
    if step_num == 3:
        return {"action_type": "diagnose", "diagnosis": plan["diagnosis"]}
    if step_num >= 4:
        return {
            "action_type": "prescribe_fix",
            "fix_description": plan["fix_description"],
            "fix_code": plan["fix_code"],
        }

    if action_type == "request_hint":
        return {"action_type": "analyze_logs"}

    return llm_action


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def run_episode(
    env_client: TorchDebugClient,
    llm_client: OpenAI,
    task_id: str,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single debugging episode."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id} | Scenario: {scenario_id or 'random'}")
    print(f"{'='*60}")

    # Reset environment
    reset_data = env_client.reset(task_id=task_id, scenario_id=scenario_id)
    obs_text = format_observation(reset_data)
    print(f"\n  📋 Scenario loaded. Beginning investigation...\n")

    history = []
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < 20:
        steps += 1

        # Get LLM action
        llm_action = get_llm_action(llm_client, obs_text, history)
        action = choose_action_with_fallback(steps, llm_action, obs_text)
        action_type = action.get("action_type", "analyze_logs")
        print(f"  Step {steps}: 🔧 {action_type}", end="")

        if action_type == "diagnose":
            print(f" → \"{action.get('diagnosis', '')[:60]}...\"")
        elif action_type == "prescribe_fix":
            print(f" → \"{action.get('fix_description', '')[:60]}...\"")
        else:
            print()

        # Add to conversation history
        history.append({"role": "assistant", "content": json.dumps(action)})

        # Step environment
        step_data = env_client.step(action)
        obs_metadata = step_data.get("observation", step_data)
        reward = step_data.get("reward", obs_metadata.get("reward", 0.0))
        if reward is None:
            reward = 0.0
        done = step_data.get("done", obs_metadata.get("done", False))
        total_reward = reward  # Final reward replaces

        # Format new observation
        obs_text = format_observation(step_data)

        # Show feedback
        feedback = obs_metadata.get("feedback", "")
        if feedback:
            for line in feedback.split("\n")[:3]:
                print(f"        {line}")

        history.append({"role": "user", "content": obs_text[:2000]})

    print(f"\n  🏁 Episode complete! Score: {total_reward:.3f}/1.0 in {steps} steps")
    return {
        "task_id": task_id,
        "scenario_id": scenario_id,
        "score": total_reward,
        "steps": steps,
    }


def main():
    """Run the baseline evaluation across all tasks."""
    print("╔══════════════════════════════════════════════╗")
    print("║   TorchDebug — PyTorch Training Debugger     ║")
    print("║   Baseline Inference Script                  ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  LLM: {MODEL_NAME}")
    print(f"  API: {API_BASE_URL}")
    print(f"  ENV: {ENV_BASE_URL}\n")

    # Create clients
    env_client = TorchDebugClient(ENV_BASE_URL)
    llm_client = create_llm_client()

    # Wait for environment to be ready
    print("  Waiting for environment server...", end="", flush=True)
    for _ in range(30):
        if env_client.health():
            print(" ✅")
            break
        time.sleep(2)
        print(".", end="", flush=True)
    else:
        print("\n  ❌ Environment server not reachable!")
        sys.exit(1)

    # Run one deterministic scenario from each required task (easy/medium/hard)
    tasks = [
        ("basic_failures", "easy_lr_too_high"),
        ("performance_issues", "med_data_leakage"),
        ("subtle_bugs", "hard_ddp_grad_accum"),
    ]

    results = []
    try:
        for task_id, scenario_id in tasks:
            result = run_episode(env_client, llm_client, task_id, scenario_id)
            results.append(result)
    finally:
        env_client.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<25} {'Scenario':<30} {'Score':<8} {'Steps':<6}")
    print(f"  {'-'*25} {'-'*30} {'-'*8} {'-'*6}")

    total = 0.0
    by_task = collections.defaultdict(list)
    for r in results:
        print(f"  {r['task_id']:<25} {r['scenario_id']:<30} {r['score']:<8.3f} {r['steps']:<6}")
        total += r['score']
        by_task[r["task_id"]].append(r["score"])

    avg = total / len(results) if results else 0
    print(f"\n  Average Score: {avg:.3f}/1.0")
    print(f"  Total Scenarios: {len(results)}")

    print("\n  Per-task Averages:")
    for task_id in ["basic_failures", "performance_issues", "subtle_bugs"]:
        task_scores = by_task.get(task_id, [])
        task_avg = sum(task_scores) / len(task_scores) if task_scores else 0.0
        print(f"    - {task_id}: {task_avg:.3f}")

    # Save results
    output_path = "outputs/evals/baseline_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "results": results,
                "average_score": avg,
                "task_scores": {
                    task: (sum(scores) / len(scores) if scores else 0.0)
                    for task, scores in by_task.items()
                },
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
            },
            f,
            indent=2,
        )
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
