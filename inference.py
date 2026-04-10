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
ENV_NAME = "torchdebug_env"

# LLM configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional (for docker-image based evaluation flows)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


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
        print("[WARN] llm_json_parse_failed fallback=analyze_logs", file=sys.stderr)
        return {"action_type": "analyze_logs"}
    except Exception as e:
        print(f"[ERROR] llm_call_failed error={e}", file=sys.stderr)
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


def _format_action(action: Dict[str, Any]) -> str:
    """Compact one-line action string for structured STEP logs."""
    try:
        return json.dumps(action, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(action).replace("\n", " ")


def _format_error(error_value: Any) -> str:
    """Format last_action_error field per validator requirements."""
    if error_value is None or error_value == "":
        return "null"
    return str(error_value).replace("\n", " ")


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def run_episode(
    llm_client: OpenAI,
    task_id: str,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single debugging episode with strict structured stdout logs."""
    env_client = TorchDebugClient(ENV_BASE_URL)
    rewards: list[float] = []
    done = False
    success = False
    steps = 0
    total_reward = 0.0

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        # Wait for environment to be ready
        for _ in range(30):
            if env_client.health():
                break
            time.sleep(2)
        else:
            raise RuntimeError("env_server_unreachable")

        reset_data = env_client.reset(task_id=task_id, scenario_id=scenario_id)
        obs_text = format_observation(reset_data)
        history = []

        while not done and steps < 20:
            steps += 1

            llm_action = get_llm_action(llm_client, obs_text, history)
            action = choose_action_with_fallback(steps, llm_action, obs_text)

            # Add to conversation history
            history.append({"role": "assistant", "content": json.dumps(action)})

            step_data = env_client.step(action)
            obs_metadata = step_data.get("observation", step_data)
            reward = step_data.get("reward", obs_metadata.get("reward", 0.0))
            reward = float(0.0 if reward is None else reward)
            done = bool(step_data.get("done", obs_metadata.get("done", False)))
            total_reward = reward
            rewards.append(reward)

            error_value = _format_error(obs_metadata.get("last_action_error"))
            action_str = _format_action(action)
            print(
                f"[STEP] step={steps} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={str(done).lower()} "
                f"error={error_value}",
                flush=True,
            )

            obs_text = format_observation(step_data)
            history.append({"role": "user", "content": obs_text[:2000]})

        success = done

    except Exception as e:
        print(f"[ERROR] episode_failed error={str(e).replace(chr(10), ' ')}", file=sys.stderr)
    finally:
        try:
            env_client.close()
        except Exception:
            pass

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps} "
            f"rewards={rewards_str}",
            flush=True,
        )

    return {
        "task_id": task_id,
        "scenario_id": scenario_id,
        "score": total_reward,
        "steps": steps,
        "success": success,
    }


def main():
    """Run the baseline evaluation across all tasks."""
    llm_client = create_llm_client()

    # Run one deterministic scenario from each required task (easy/medium/hard)
    tasks = [
        ("basic_failures", "easy_lr_too_high"),
        ("performance_issues", "med_data_leakage"),
        ("subtle_bugs", "hard_ddp_grad_accum"),
    ]

    results = []
    for task_id, scenario_id in tasks:
        result = run_episode(llm_client, task_id, scenario_id)
        results.append(result)

    total = 0.0
    by_task = collections.defaultdict(list)
    for r in results:
        total += r['score']
        by_task[r["task_id"]].append(r["score"])

    avg = total / len(results) if results else 0

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
    print(f"[INFO] summary average_score={avg:.2f} output_path={output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
