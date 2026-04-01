"""
TorchDebug Environment — MCPEnvironment implementation.

A real-world OpenEnv environment that simulates debugging broken PyTorch
training runs. The agent must investigate, diagnose, and prescribe fixes.
"""

import math
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

try:
    from ..models import (
        TorchDebugObservation,
        TorchDebugAction,
        TorchDebugState,
        TrainingLogEntry,
        SystemInfo,
        InspectionResult,
    )
    from ..scenarios import (
        BugScenario,
        get_scenarios,
        get_scenario_by_id,
        get_all_scenarios,
    )
    # Import scenario registrations
    import torchdebug_env.scenarios.basic_failures
    import torchdebug_env.scenarios.performance_issues
    import torchdebug_env.scenarios.subtle_bugs

    from ..utils.reward import (
        compute_investigation_reward,
        grade_diagnosis,
        grade_fix,
        compute_episode_score,
        get_hint,
    )
except ImportError:
    from models import (
        TorchDebugObservation,
        TorchDebugAction,
        TorchDebugState,
        TrainingLogEntry,
        SystemInfo,
        InspectionResult,
    )
    from scenarios import (
        BugScenario,
        get_scenarios,
        get_scenario_by_id,
        get_all_scenarios,
    )
    # Import scenario registrations
    import scenarios.basic_failures
    import scenarios.performance_issues
    import scenarios.subtle_bugs

    from utils.reward import (
        compute_investigation_reward,
        grade_diagnosis,
        grade_fix,
        compute_episode_score,
        get_hint,
    )

# Task ID to difficulty mapping
TASK_DIFFICULTY_MAP = {
    "basic_failures": "easy",
    "performance_issues": "medium",
    "subtle_bugs": "hard",
}

MAX_STEPS = {
    "easy": 10,
    "medium": 15,
    "hard": 20,
}

AVAILABLE_ACTIONS = [
    "analyze_logs",
    "inspect_gradients",
    "inspect_data_pipeline",
    "inspect_model_architecture",
    "check_device_placement",
    "diagnose",
    "prescribe_fix",
    "request_hint",
]


class TorchDebugEnvironment(MCPEnvironment):
    """
    PyTorch Training Run Debugger Environment.

    The agent is presented with a broken PyTorch training run and must:
    1. Investigate using available inspection tools
    2. Diagnose the root cause
    3. Prescribe the correct fix

    Tasks:
    - basic_failures (easy): Common crashes and obvious bugs
    - performance_issues (medium): Subtle performance problems
    - subtle_bugs (hard): Compound bugs requiring deep PyTorch knowledge
    """

    def __init__(self):
        """Initialize the environment with MCP tools."""
        mcp = FastMCP("torchdebug_env")

        @mcp.tool
        def get_task_info() -> dict:
            """Get info about all available tasks and their scenarios."""
            all_scenarios = get_all_scenarios()
            info = {}
            for task_id, scenarios_list in all_scenarios.items():
                info[task_id] = {
                    "difficulty": TASK_DIFFICULTY_MAP[task_id],
                    "num_scenarios": len(scenarios_list),
                    "scenario_ids": [s.scenario_id for s in scenarios_list],
                }
            return info

        @mcp.tool
        def get_observation() -> dict:
            """Get the current observation (what the agent sees)."""
            if self._current_obs is None:
                return {"error": "Environment not initialized. Call reset first."}
            return self._current_obs.model_dump()

        @mcp.tool
        def take_action(
            action_type: str,
            diagnosis: str = "",
            fix_description: str = "",
            fix_code: str = "",
            parameters: str = "",
        ) -> dict:
            """
            Take a debugging action.

            action_type: One of: analyze_logs, inspect_gradients, inspect_data_pipeline,
                        inspect_model_architecture, check_device_placement,
                        diagnose, prescribe_fix, request_hint
            diagnosis: Root cause diagnosis (for 'diagnose' action)
            fix_description: Description of the fix (for 'prescribe_fix' action)
            fix_code: Corrected code (optional, for 'prescribe_fix' action)
            parameters: JSON string of additional parameters
            """
            import json
            params = {}
            if parameters:
                try:
                    params = json.loads(parameters)
                except json.JSONDecodeError:
                    params = {}

            action = TorchDebugAction(
                action_type=action_type,
                diagnosis=diagnosis or None,
                fix_description=fix_description or None,
                fix_code=fix_code or None,
                parameters=params,
            )
            obs = self._process_action(action)
            return obs.model_dump()

        super().__init__(mcp)
        self._state = TorchDebugState(episode_id=str(uuid4()), step_count=0)
        self._current_scenario: Optional[BugScenario] = None
        self._current_obs: Optional[TorchDebugObservation] = None
        self._investigation_reward: float = 0.0
        self._diagnosis_score: float = 0.0
        self._fix_score: float = 0.0
        self._current_task_id: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment with a new scenario.

        kwargs:
            task_id: Specific task ("basic_failures", "performance_issues", "subtle_bugs")
            scenario_id: Specific scenario ID (overrides task_id)
        """
        if seed is not None:
            random.seed(seed)

        # Determine which scenario to load
        scenario_id = kwargs.get("scenario_id")
        task_id = kwargs.get("task_id", "basic_failures")

        if scenario_id:
            scenario = get_scenario_by_id(scenario_id)
            if scenario is None:
                return Observation(
                    done=True,
                    reward=0.0,
                    metadata={"error": f"Scenario '{scenario_id}' not found"},
                )
        else:
            scenarios_list = get_scenarios(task_id)
            if not scenarios_list:
                return Observation(
                    done=True,
                    reward=0.0,
                    metadata={"error": f"No scenarios for task '{task_id}'"},
                )
            scenario = random.choice(scenarios_list)

        self._current_scenario = scenario
        self._current_task_id = scenario.task_id
        difficulty = TASK_DIFFICULTY_MAP[scenario.task_id]
        max_steps = MAX_STEPS[difficulty]

        # Reset state
        self._state = TorchDebugState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=scenario.task_id,
            difficulty=difficulty,
        )
        self._investigation_reward = 0.0
        self._diagnosis_score = 0.0
        self._fix_score = 0.0

        # Build training logs
        training_logs = []
        for log_data in scenario.training_logs_data:
            # Replace inf/nan with string representations for JSON safety
            safe_data = {}
            for k, v in log_data.items():
                if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                    safe_data[k] = str(v)
                else:
                    safe_data[k] = v
            training_logs.append(TrainingLogEntry(**safe_data))

        # Build initial observation
        system_info = SystemInfo(**(scenario.system_info or {}))

        self._current_obs = TorchDebugObservation(
            task_id=scenario.scenario_id,
            task_description=scenario.description,
            difficulty=difficulty,
            training_config=scenario.training_config,
            code_snippet=scenario.code_snippet,
            training_logs=training_logs,
            system_info=system_info,
            error_message=scenario.error_message,
            available_actions=AVAILABLE_ACTIONS,
            step_number=0,
            max_steps=max_steps,
            hints_used=0,
            inspection_results=[],
            feedback="Environment initialized. Analyze the training run and diagnose the issue.",
            done=False,
            reward=0.0,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata=self._current_obs.model_dump(),
        )

    def _process_action(self, action: TorchDebugAction) -> TorchDebugObservation:
        """Process an agent action and return updated observation."""
        scenario = self._current_scenario
        obs = self._current_obs

        if scenario is None or obs is None:
            return TorchDebugObservation(
                task_id="none",
                task_description="Environment not initialized",
                difficulty="easy",
                training_config={},
                code_snippet="",
                available_actions=AVAILABLE_ACTIONS,
                feedback="Error: Call reset() first.",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        self._state.actions_taken.append(action.action_type)
        step_reward = 0.0
        feedback = ""
        done = False

        difficulty = TASK_DIFFICULTY_MAP[scenario.task_id]
        max_steps = MAX_STEPS[difficulty]

        if action.action_type in [
            "analyze_logs", "inspect_gradients", "inspect_data_pipeline",
            "inspect_model_architecture", "check_device_placement",
        ]:
            # Investigation action
            reward = compute_investigation_reward(
                action.action_type, scenario, self._state.actions_taken[:-1]
            )
            self._investigation_reward += reward
            step_reward = reward

            # Return inspection data if available
            inspection_data = scenario.inspection_data.get(action.action_type)
            if inspection_data:
                new_result = InspectionResult(**inspection_data)
                feedback = f"[{action.action_type}] {inspection_data['findings']}"
            else:
                new_result = InspectionResult(
                    inspection_type=action.action_type,
                    findings=f"No specific findings from {action.action_type} inspection for this scenario.",
                    data={}
                )
                feedback = f"[{action.action_type}] No significant findings from this inspection type."

            # Update inspection results
            new_results = list(obs.inspection_results) + [new_result]

        elif action.action_type == "diagnose":
            # Diagnosis action
            self._diagnosis_score = grade_diagnosis(
                action.diagnosis or "", scenario
            )
            step_reward = self._diagnosis_score * 0.2  # Partial reward for diagnosis
            self._state.diagnosed = self._diagnosis_score > 0.5

            if self._diagnosis_score >= 0.8:
                feedback = "✅ Excellent diagnosis! You've correctly identified the root cause."
            elif self._diagnosis_score >= 0.5:
                feedback = "⚠️ Partial diagnosis. You're on the right track but missing some details."
            elif self._diagnosis_score > 0.0:
                feedback = "❌ Your diagnosis touches on some relevant aspects but doesn't identify the core issue."
            else:
                feedback = "❌ Diagnosis does not match the root cause. Try investigating more."
            new_results = list(obs.inspection_results)

        elif action.action_type == "prescribe_fix":
            # Fix prescription — can end episode
            self._fix_score = grade_fix(
                action.fix_description or "",
                action.fix_code,
                scenario,
            )
            self._state.fixed = self._fix_score > 0.5

            # Compute final episode score
            final_score = compute_episode_score(
                diagnosis_score=self._diagnosis_score,
                fix_score=self._fix_score,
                investigation_reward=self._investigation_reward,
                hints_used=self._state.hints_used,
                steps_taken=self._state.step_count,
                max_steps=max_steps,
                difficulty=difficulty,
                actions_taken=self._state.actions_taken,
                relevant_inspections=scenario.relevant_inspections,
            )
            step_reward = final_score
            self._state.cumulative_reward = final_score
            done = True

            if final_score >= 0.8:
                feedback = f"🎉 Excellent work! Score: {final_score:.2f}/1.0. You diagnosed and fixed the issue."
            elif final_score >= 0.5:
                feedback = f"✅ Good effort! Score: {final_score:.2f}/1.0. Partial credit for your analysis."
            else:
                feedback = f"❌ Score: {final_score:.2f}/1.0. The fix doesn't adequately address the root cause."

            feedback += f"\n\nGround truth: {scenario.root_cause_description}"
            feedback += f"\nCorrect fix: {scenario.correct_fix_description}"
            new_results = list(obs.inspection_results)

        elif action.action_type == "request_hint":
            # Hint request
            self._state.hints_used += 1
            hint = get_hint(scenario, self._state.hints_used)
            feedback = f"💡 Hint #{self._state.hints_used}: {hint}"
            step_reward = -0.02  # Small penalty for hints
            new_results = list(obs.inspection_results)

        else:
            feedback = f"Unknown action type: {action.action_type}"
            step_reward = 0.0
            new_results = list(obs.inspection_results)

        # Check step limit
        if self._state.step_count >= max_steps and not done:
            done = True
            final_score = compute_episode_score(
                diagnosis_score=self._diagnosis_score,
                fix_score=self._fix_score,
                investigation_reward=self._investigation_reward,
                hints_used=self._state.hints_used,
                steps_taken=self._state.step_count,
                max_steps=max_steps,
                difficulty=difficulty,
                actions_taken=self._state.actions_taken,
                relevant_inspections=scenario.relevant_inspections,
            )
            step_reward = final_score
            self._state.cumulative_reward = final_score
            feedback += f"\n\n⏰ Step limit reached ({max_steps} steps). Final score: {final_score:.2f}/1.0"
            feedback += f"\n\nGround truth: {scenario.root_cause_description}"

        # Build updated observation
        self._current_obs = TorchDebugObservation(
            task_id=scenario.scenario_id,
            task_description=scenario.description,
            difficulty=difficulty,
            training_config=scenario.training_config,
            code_snippet=scenario.code_snippet,
            training_logs=obs.training_logs,
            system_info=obs.system_info,
            error_message=scenario.error_message,
            available_actions=AVAILABLE_ACTIONS,
            step_number=self._state.step_count,
            max_steps=max_steps,
            hints_used=self._state.hints_used,
            inspection_results=new_results,
            feedback=feedback,
            done=done,
            reward=step_reward,
        )

        return self._current_obs

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (fallback)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction, CallToolAction, or the take_action MCP tool."
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
        )

