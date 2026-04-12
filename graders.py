"""
Grader functions for the TorchDebug environment.

Each grader receives the agent's action and the environment observation,
and returns a score strictly in the open interval (0, 1).

These are referenced by openenv.yaml and called by the OpenEnv validator.
"""

from typing import Any


def _clamp_score(v: float) -> float:
    """Ensure score is strictly in (0, 1). Never returns 0.0 or 1.0."""
    v = float(v) if v is not None else 0.01
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.95
    return v


def grade_basic_failures(action: Any, observation: Any) -> float:
    """Grade performance on basic failure diagnosis tasks (easy difficulty).

    Evaluates the agent's ability to identify common PyTorch training
    failures like learning rate issues, loss explosions, etc.
    """
    try:
        # Extract reward from observation if available
        if hasattr(observation, "reward"):
            return _clamp_score(observation.reward)
        if isinstance(observation, dict):
            return _clamp_score(observation.get("reward", 0.5))
    except Exception:
        pass
    return 0.5


def grade_performance_issues(action: Any, observation: Any) -> float:
    """Grade performance on performance issue tasks (medium difficulty).

    Evaluates the agent's ability to diagnose data pipeline issues,
    data leakage, and training inefficiencies.
    """
    try:
        if hasattr(observation, "reward"):
            return _clamp_score(observation.reward)
        if isinstance(observation, dict):
            return _clamp_score(observation.get("reward", 0.5))
    except Exception:
        pass
    return 0.5


def grade_subtle_bugs(action: Any, observation: Any) -> float:
    """Grade performance on subtle bug tasks (hard difficulty).

    Evaluates the agent's ability to diagnose distributed training bugs,
    gradient accumulation issues, and other advanced problems.
    """
    try:
        if hasattr(observation, "reward"):
            return _clamp_score(observation.reward)
        if isinstance(observation, dict):
            return _clamp_score(observation.get("reward", 0.5))
    except Exception:
        pass
    return 0.5
