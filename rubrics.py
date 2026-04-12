"""
Rubrics for the TorchDebug environment.

Follows the OpenEnv Rubric system (RFC 004) to provide composable,
deterministic rewards for PyTorch debugging tasks.

Design: The rubric evaluates the agent's debugging trajectory —
investigation actions, diagnosis accuracy, and fix quality — and
returns a reward strictly in (0, 1).
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.01, hi: float = 0.99) -> float:
    """Clamp a value to the strict open interval (lo, hi)."""
    return max(lo, min(hi, float(value)))


# ---------------------------------------------------------------------------
# Component rubrics
# ---------------------------------------------------------------------------

class InvestigationRubric(Rubric):
    """Process rubric: small rewards for relevant investigation actions.

    Returns a small positive signal when the agent inspects logs,
    gradients, or the data pipeline — actions that demonstrate
    systematic debugging methodology.
    """

    def __init__(self, per_step_reward: float = 0.0) -> None:
        super().__init__()
        self.per_step_reward = per_step_reward

    def forward(self, action: Any, observation: Any) -> float:
        if getattr(observation, "done", False):
            return 0.0
        return self.per_step_reward

    def reset(self) -> None:
        pass


class DiagnosisRubric(Rubric):
    """Outcome rubric: evaluates diagnosis accuracy on terminal step.

    Compares the agent's diagnosis against the scenario's known root
    cause using keyword matching and returns partial credit.
    """

    def __init__(self) -> None:
        super().__init__()
        self._keywords: list[str] = []

    def set_keywords(self, keywords: list[str]) -> None:
        """Set the ground-truth diagnosis keywords from the scenario."""
        self._keywords = [k.lower() for k in keywords]

    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False):
            return 0.0
        # Check if we have diagnosis info in metadata
        metadata = getattr(observation, "metadata", {})
        diagnosis = metadata.get("agent_diagnosis", "")
        if not diagnosis or not self._keywords:
            return 0.0

        diagnosis_lower = diagnosis.lower()
        hits = sum(1 for kw in self._keywords if kw in diagnosis_lower)
        return hits / max(len(self._keywords), 1)

    def reset(self) -> None:
        self._keywords = []


class FixQualityRubric(Rubric):
    """Outcome rubric: evaluates the quality of the prescribed fix.

    Checks whether the agent's fix code addresses the identified root
    cause. Returns credit based on keyword overlap with expected fix.
    """

    def __init__(self) -> None:
        super().__init__()
        self._expected_keywords: list[str] = []

    def set_expected(self, keywords: list[str]) -> None:
        self._expected_keywords = [k.lower() for k in keywords]

    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False):
            return 0.0
        metadata = getattr(observation, "metadata", {})
        fix_code = metadata.get("agent_fix_code", "")
        if not fix_code or not self._expected_keywords:
            return 0.0

        fix_lower = fix_code.lower()
        hits = sum(1 for kw in self._expected_keywords if kw in fix_lower)
        return hits / max(len(self._expected_keywords), 1)

    def reset(self) -> None:
        self._expected_keywords = []


# ---------------------------------------------------------------------------
# Composite rubric
# ---------------------------------------------------------------------------

class TorchDebugRubric(Rubric):
    """Composite rubric for the TorchDebug environment.

    Combines process-based reward (investigation quality) with
    outcome-based reward (diagnosis accuracy + fix quality).

    On non-terminal steps: returns a small process reward.
    On terminal step: returns the weighted combination of diagnosis
    and fix quality scores, clamped to (0.01, 0.99).

    Weights:
        - Diagnosis accuracy: 40%
        - Fix quality: 40%
        - Investigation process: 20% (capped)
    """

    def __init__(
        self,
        investigation: Rubric | None = None,
        diagnosis: Rubric | None = None,
        fix_quality: Rubric | None = None,
        diagnosis_weight: float = 0.4,
        fix_weight: float = 0.4,
        process_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.investigation = investigation or InvestigationRubric()
        self.diagnosis = diagnosis or DiagnosisRubric()
        self.fix_quality = fix_quality or FixQualityRubric()
        self.diagnosis_weight = diagnosis_weight
        self.fix_weight = fix_weight
        self.process_weight = process_weight
        self._process_total = 0.0

    def forward(self, action: Any, observation: Any) -> float:
        done = getattr(observation, "done", False)

        if not done:
            # Non-terminal: accumulate process reward
            step_reward = self.investigation(action, observation)
            self._process_total += step_reward
            return _clamp(0.01)  # Tiny positive for intermediates

        # Terminal step: compute composite outcome score
        diagnosis_score = self.diagnosis(action, observation)
        fix_score = self.fix_quality(action, observation)
        process_score = min(self._process_total, 1.0)

        composite = (
            self.diagnosis_weight * diagnosis_score
            + self.fix_weight * fix_score
            + self.process_weight * process_score
        )
        return _clamp(composite)

    def reset(self) -> None:
        self._process_total = 0.0
        self.investigation.reset()
        self.diagnosis.reset()
        self.fix_quality.reset()
