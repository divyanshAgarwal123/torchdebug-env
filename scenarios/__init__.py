"""Scenario base class and registry for bug scenarios."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BugScenario:
    """Defines a single bug scenario for the training debugger."""

    # ── Required fields (no defaults) ──
    # Identity
    scenario_id: str
    title: str
    difficulty: str  # "easy", "medium", "hard"
    task_id: str  # "basic_failures", "performance_issues", "subtle_bugs"

    # What the agent sees
    description: str
    training_config: Dict[str, Any]
    code_snippet: str
    training_logs_data: List[Dict[str, Any]]

    # Ground truth for grading
    root_cause_category: str
    root_cause_description: str
    correct_fix_description: str

    # ── Optional / defaulted fields ──
    system_info: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    correct_fix_code: Optional[str] = None
    bug_location_hint: Optional[str] = None

    # For compound bugs (hard difficulty)
    secondary_bugs: List[Dict[str, str]] = field(default_factory=list)

    # Keywords for fuzzy matching diagnosis
    diagnosis_keywords: List[str] = field(default_factory=list)
    fix_keywords: List[str] = field(default_factory=list)

    # Relevant investigation actions
    relevant_inspections: List[str] = field(default_factory=list)

    # Inspection results that should be returned
    inspection_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# Global scenario registry
_SCENARIO_REGISTRY: Dict[str, List[BugScenario]] = {
    "basic_failures": [],
    "performance_issues": [],
    "subtle_bugs": [],
}


def register_scenario(scenario: BugScenario):
    """Register a bug scenario in the appropriate task list."""
    _SCENARIO_REGISTRY[scenario.task_id].append(scenario)


def get_scenarios(task_id: str) -> List[BugScenario]:
    """Get all scenarios for a given task ID."""
    return _SCENARIO_REGISTRY.get(task_id, [])


def get_all_scenarios() -> Dict[str, List[BugScenario]]:
    """Get all registered scenarios by task."""
    return _SCENARIO_REGISTRY


def get_scenario_by_id(scenario_id: str) -> Optional[BugScenario]:
    """Find a specific scenario by its ID."""
    for scenarios in _SCENARIO_REGISTRY.values():
        for s in scenarios:
            if s.scenario_id == scenario_id:
                return s
    return None
