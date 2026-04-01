import math

try:
    from torchdebug_env.scenarios import BugScenario
    from torchdebug_env.utils.reward import (
        compute_episode_score,
        grade_diagnosis,
        get_hint,
    )
except ImportError:
    from scenarios import BugScenario
    from utils.reward import (
        compute_episode_score,
        grade_diagnosis,
        get_hint,
    )


def _dummy_scenario() -> BugScenario:
    return BugScenario(
        scenario_id="t1",
        title="Dummy",
        difficulty="easy",
        task_id="basic_failures",
        description="dummy",
        training_config={},
        code_snippet="print('x')",
        training_logs_data=[],
        root_cause_category="learning_rate_too_high",
        root_cause_description="lr too high",
        correct_fix_description="lower lr",
        diagnosis_keywords=["learning rate", "too high", "nan"],
        fix_keywords=["lower", "learning rate", "0.01"],
        relevant_inspections=["analyze_logs", "inspect_gradients"],
    )


def test_grade_diagnosis_deterministic():
    scenario = _dummy_scenario()
    text = "The learning rate is too high and causes NaN divergence"
    s1 = grade_diagnosis(text, scenario)
    s2 = grade_diagnosis(text, scenario)
    assert math.isclose(s1, s2, rel_tol=0.0, abs_tol=1e-12)


def test_hint_progression_order():
    scenario = _dummy_scenario()
    h1 = get_hint(scenario, 1)
    h2 = get_hint(scenario, 2)
    h3 = get_hint(scenario, 3)
    assert h1 != h2
    assert h2 != h3


def test_exploit_cap_without_investigation():
    # High diagnosis+fix but no investigation actions should be capped.
    score = compute_episode_score(
        diagnosis_score=1.0,
        fix_score=1.0,
        investigation_reward=0.0,
        hints_used=0,
        steps_taken=2,
        max_steps=10,
        difficulty="easy",
        actions_taken=["diagnose", "prescribe_fix"],
        relevant_inspections=["analyze_logs", "inspect_gradients"],
    )
    assert score <= 0.60


def test_evidence_alignment_bonus():
    base = compute_episode_score(
        diagnosis_score=0.8,
        fix_score=0.8,
        investigation_reward=0.05,
        hints_used=0,
        steps_taken=5,
        max_steps=10,
        difficulty="medium",
        actions_taken=["analyze_logs", "diagnose", "prescribe_fix"],
        relevant_inspections=["analyze_logs", "inspect_gradients"],
    )
    better = compute_episode_score(
        diagnosis_score=0.8,
        fix_score=0.8,
        investigation_reward=0.10,
        hints_used=0,
        steps_taken=5,
        max_steps=10,
        difficulty="medium",
        actions_taken=["analyze_logs", "inspect_gradients", "diagnose", "prescribe_fix"],
        relevant_inspections=["analyze_logs", "inspect_gradients"],
    )
    assert better > base
