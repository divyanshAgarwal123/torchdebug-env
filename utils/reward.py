"""
Reward calculation and grading logic for the TorchDebug environment.

Provides partial credit for investigation actions, correct diagnosis,
and correct fix prescription.
"""

import re
from typing import List, Optional

try:
    from ..scenarios import BugScenario
except ImportError:
    from scenarios import BugScenario


def _strict_open_unit(value: float) -> float:
    """Clamp score to strict open interval (0, 1)."""
    v = float(value)
    if v <= 0.01:
        return 0.01
    if v >= 0.99:
        return 0.99
    return v


def compute_investigation_reward(
    action_type: str,
    scenario: BugScenario,
    actions_taken: List[str],
) -> float:
    """
    Compute reward for investigation actions.
    Relevant inspections get higher reward. Repeated inspections get less.
    """
    # Check if this inspection type has already been done
    same_type_count = actions_taken.count(action_type)

    if action_type in scenario.relevant_inspections:
        # First relevant inspection gives 0.05, subsequent give diminishing returns
        if same_type_count == 0:
            return 0.05
        elif same_type_count == 1:
            return 0.02
        else:
            return 0.005
    else:
        # Non-relevant but valid investigation
        if same_type_count == 0:
            return 0.02
        else:
            return 0.005


def compute_hint_penalty(hints_used: int) -> float:
    """Compute penalty for using hints. Increasing penalty per hint."""
    penalties = [0.05, 0.10, 0.15]  # 1st, 2nd, 3rd hint penalties
    if hints_used <= 0:
        return 0.0
    idx = min(hints_used - 1, len(penalties) - 1)
    return penalties[idx]


def fuzzy_keyword_match(text: str, keywords: List[str]) -> float:
    """
    Compute a fuzzy match score between text and a list of keywords.
    Returns a score between 0.0 and 1.0.
    """
    if not text or not keywords:
        return 0.0

    text_lower = text.lower().strip()
    matched = 0

    for kw in keywords:
        # Check for keyword presence (case-insensitive)
        if kw.lower() in text_lower:
            matched += 1
        else:
            # Check for partial word match (at least 4 chars)
            words = re.findall(r'\b\w+\b', kw.lower())
            for word in words:
                if len(word) >= 4 and word in text_lower:
                    matched += 0.5
                    break

    return min(matched / max(len(keywords) * 0.3, 1), 1.0)  # Need 30% of keywords for max


def grade_diagnosis(
    diagnosis: str,
    scenario: BugScenario,
) -> float:
    """
    Grade the agent's diagnosis against the ground truth.

    Returns a score between 0.0 and 1.0:
    - 1.0: Perfect match (root cause correctly identified)
    - 0.5-0.9: Partial match (some keywords matched)
    - 0.0: No match
    """
    if not diagnosis:
        return _strict_open_unit(0.0)

    # Check exact category match
    if scenario.root_cause_category.lower() in diagnosis.lower().replace(" ", "_"):
        return _strict_open_unit(1.0)

    # Fuzzy keyword matching
    return _strict_open_unit(fuzzy_keyword_match(diagnosis, scenario.diagnosis_keywords))


def grade_fix(
    fix_description: str,
    fix_code: Optional[str],
    scenario: BugScenario,
) -> float:
    """
    Grade the agent's prescribed fix.

    Returns a score between 0.0 and 1.0.
    """
    if not fix_description:
        return _strict_open_unit(0.0)

    score = 0.0

    # Grade fix description
    desc_score = fuzzy_keyword_match(fix_description, scenario.fix_keywords)
    score += desc_score * 0.6  # 60% weight on description

    # Grade fix code if provided
    if fix_code and scenario.correct_fix_code:
        code_score = fuzzy_keyword_match(fix_code, scenario.fix_keywords)
        score += code_score * 0.4  # 40% weight on code
    else:
        # If no code expected/provided, scale description score to 100%
        score = desc_score

    return _strict_open_unit(min(score, 1.0))


def compute_episode_score(
    diagnosis_score: float,
    fix_score: float,
    investigation_reward: float,
    hints_used: int,
    steps_taken: int,
    max_steps: int,
    difficulty: str,
    actions_taken: Optional[List[str]] = None,
    relevant_inspections: Optional[List[str]] = None,
) -> float:
    """
    Compute final episode score (0.0-1.0) used by the grader.

    Scoring breakdown:
    - Diagnosis: 40% weight
    - Fix: 40% weight
    - Investigation efficiency: 10% weight
    - Step efficiency: 10% weight
    - Minus hint penalties
    """
    # Diagnosis and fix (80% of score)
    score = diagnosis_score * 0.40 + fix_score * 0.40

    # Investigation efficiency (did agent inspect relevant things?)
    inv_score = min(investigation_reward / 0.15, 1.0)  # Cap at 0.15 total inv reward
    score += inv_score * 0.10

    # Step efficiency bonus (fewer steps = better)
    if max_steps > 0:
        efficiency = max(0, 1.0 - (steps_taken / max_steps))
        score += efficiency * 0.10

    # Hint penalties
    score -= compute_hint_penalty(hints_used)

    # Evidence alignment bonus (up to +0.05)
    if actions_taken is not None and relevant_inspections:
        inspections = {
            "analyze_logs",
            "inspect_gradients",
            "inspect_data_pipeline",
            "inspect_model_architecture",
            "check_device_placement",
        }
        inspected = [a for a in actions_taken if a in inspections]
        coverage = len(set(inspected).intersection(set(relevant_inspections))) / max(len(set(relevant_inspections)), 1)
        score += min(max(coverage, 0.0), 1.0) * 0.05

        # Anti-gaming penalty: if no investigation at all, cap final score.
        if len(inspected) == 0:
            score = min(score, 0.60)

        # Additional anti-gaming penalty: diagnose before any investigation.
        if "diagnose" in actions_taken:
            first_diag = actions_taken.index("diagnose")
            if not any(a in inspections for a in actions_taken[:first_diag]):
                score -= 0.10

    # Anti-shortcut guardrail:
    # High fix score with near-zero diagnosis should not receive top score.
    if fix_score >= 0.8 and diagnosis_score < 0.2:
        score = min(score, 0.55)

    # Difficulty multiplier for display (score is already 0-1 per task)
    bounded = max(0.0, min(1.0, score))

    # Hackathon validator requires strict open interval: 0 < score < 1.
    # Use visible margins so downstream 2-decimal formatting never becomes 0.00 or 1.00.
    if bounded <= 0.0:
        return 0.01
    if bounded >= 1.0:
        return 0.99
    return max(0.01, min(0.99, bounded))


def get_hint(scenario: BugScenario, hint_number: int) -> str:
    """Generate a progressive hint for the scenario."""
    hints = [
        # Hint 1: General direction
        f"Focus on the {scenario.relevant_inspections[0] if scenario.relevant_inspections else 'training logs'} "
        f"to find the root cause. The issue is related to {scenario.root_cause_category.replace('_', ' ')}.",
        # Hint 2: More specific
        f"Bug location: {scenario.bug_location_hint or 'Check the training configuration and code carefully'}. "
        f"Key symptom: {scenario.training_logs_data[-1] if scenario.training_logs_data else 'Check error message'}.",
        # Hint 3: Near-answer
        f"The root cause is: {scenario.root_cause_description[:100]}... "
        f"The fix involves: {scenario.correct_fix_description[:80]}...",
    ]
    idx = min(max(hint_number - 1, 0), len(hints) - 1)
    return hints[idx]
