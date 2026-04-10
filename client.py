"""TorchDebug Environment client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import TorchDebugAction, TorchDebugObservation, TorchDebugState


class TorchDebugEnv(EnvClient[TorchDebugAction, TorchDebugObservation, TorchDebugState]):
    """Typed OpenEnv client for TorchDebug."""

    def _step_payload(self, action: TorchDebugAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "diagnosis": action.diagnosis,
            "fix_description": action.fix_description,
            "fix_code": action.fix_code,
            "parameters": action.parameters,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TorchDebugObservation]:
        obs_data = payload.get("observation", {})
        observation = TorchDebugObservation(
            task_id=obs_data.get("task_id", "unknown"),
            task_description=obs_data.get("task_description", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            training_config=obs_data.get("training_config", {}),
            code_snippet=obs_data.get("code_snippet", ""),
            training_logs=obs_data.get("training_logs", []),
            system_info=obs_data.get("system_info", {}),
            error_message=obs_data.get("error_message"),
            available_actions=obs_data.get("available_actions", []),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 15),
            hints_used=obs_data.get("hints_used", 0),
            inspection_results=obs_data.get("inspection_results", []),
            feedback=obs_data.get("feedback"),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TorchDebugState:
        return TorchDebugState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            difficulty=payload.get("difficulty"),
            diagnosed=payload.get("diagnosed", False),
            fixed=payload.get("fixed", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            hints_used=payload.get("hints_used", 0),
            actions_taken=payload.get("actions_taken", []),
        )
