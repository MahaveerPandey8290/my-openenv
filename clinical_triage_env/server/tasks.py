"""
Task definitions and step-level graders for ClinicalTriageEnv.

Each task has:
  - grade_step(action, patient, step_num) → float [0.0, 1.0]
  - get_feedback(action, patient) → str
  - max_steps: int

Grading philosophy:
  - Reward is SHAPED (not sparse): partial credit at each step.
  - Under-triage (calling immediate as non_urgent) is penalised heavily.
  - Correct tests and reasoning earn partial credit even before correct level.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict

from ..models import TriageAction


TRIAGE_ORDER = ["non_urgent", "less_urgent", "urgent", "immediate"]


def _level_distance(pred: str, true: str) -> int:
    """Signed distance: positive = over-triage, negative = under-triage."""
    return TRIAGE_ORDER.index(pred) - TRIAGE_ORDER.index(true)


def _test_overlap(ordered: list[str], correct: list[str]) -> float:
    ordered_lower = {t.lower() for t in ordered}
    correct_lower = {t.lower() for t in correct}
    if not correct_lower:
        return 0.0
    hits = sum(
        any(c in o or o in c for o in ordered_lower)
        for c in correct_lower
    )
    return hits / len(correct_lower)


def _condition_match(suspected: str, category: str) -> float:
    """Partial match: does suspected condition contain the condition category keyword?"""
    return 0.2 if category.lower() in suspected.lower() else 0.0


# ── EASY task grader ─────────────────────────────────────────────────────────

def _grade_vital_signs(action: TriageAction, patient: Dict[str, Any], step: int) -> float:
    reward = 0.0

    dist = _level_distance(action.triage_level, patient["triage_level"])

    if dist == 0:                        # exact match
        reward += 0.6
        if step <= 2:                    # efficiency bonus
            reward += 0.1
    elif abs(dist) == 1:                 # one level off
        reward += 0.2
    elif dist < 0:                       # under-triage — dangerous
        reward -= 0.3

    reward += _condition_match(action.suspected_condition, patient["condition_category"])

    if action.reasoning and len(action.reasoning) > 20:
        reward += 0.1                    # any meaningful reasoning

    return round(min(max(reward, 0.0), 1.0), 3)


def _feedback_vital_signs(action: TriageAction, patient: Dict[str, Any]) -> str:
    dist = _level_distance(action.triage_level, patient["triage_level"])
    if dist == 0:
        return f"✅ Correct triage level. Ground truth: {patient['triage_level']}."
    elif dist > 0:
        return f"⚠️ Over-triaged (you said {action.triage_level}, truth is {patient['triage_level']})."
    else:
        return f"🚨 UNDER-TRIAGE! You said {action.triage_level}, truth is {patient['triage_level']}. This could harm the patient."


# ── MEDIUM task grader ────────────────────────────────────────────────────────

def _grade_differential(action: TriageAction, patient: Dict[str, Any], step: int) -> float:
    reward = 0.0

    dist = _level_distance(action.triage_level, patient["triage_level"])
    if dist == 0:
        reward += 0.5
    elif abs(dist) == 1:
        reward += 0.15
    elif dist < 0:
        reward -= 0.25

    reward += _condition_match(action.suspected_condition, patient["condition_category"])

    test_score = _test_overlap(action.recommended_tests, patient["relevant_tests"])
    reward += 0.25 * test_score

    if step <= 2 and dist == 0:
        reward += 0.1

    if len(action.reasoning) > 40:
        reward += 0.05

    return round(min(max(reward, 0.0), 1.0), 3)


def _feedback_differential(action: TriageAction, patient: Dict[str, Any]) -> str:
    lines = []
    dist = _level_distance(action.triage_level, patient["triage_level"])
    if dist == 0:
        lines.append(f"✅ Correct level: {patient['triage_level']}")
    else:
        direction = "over-triaged" if dist > 0 else "UNDER-TRIAGED"
        lines.append(f"{'⚠️' if dist > 0 else '🚨'} {direction}: you said {action.triage_level}, truth={patient['triage_level']}")
    
    overlap = _test_overlap(action.recommended_tests, patient["relevant_tests"])
    lines.append(f"Tests: {overlap*100:.0f}% of key tests ordered. Key tests: {', '.join(patient['relevant_tests'][:3])}")
    return " | ".join(lines)


# ── HARD task grader ──────────────────────────────────────────────────────────

def _grade_polytrauma(action: TriageAction, patient: Dict[str, Any], step: int) -> float:
    """
    Hard task: reward is highest at later steps when more info is available.
    Earlier steps are partially rewarded; full reward requires integrating all info.
    """
    reward = 0.0
    dist = _level_distance(action.triage_level, patient["triage_level"])

    # Level reward: Give high base score for correct triage immediately.
    # Step 1-2 correct: 0.7 reward. Steps 3-5: 0.6 reward.
    if dist == 0:
        reward += 0.7 if step <= 2 else 0.6
    elif abs(dist) == 1:
        reward += 0.2  # partial credit for "urgent" vs "immediate"
    elif dist < 0:
        reward -= 0.4  # severe penalty for missing immediate in trauma

    reward += _condition_match(action.suspected_condition, patient["condition_category"])

    test_score = _test_overlap(action.recommended_tests, patient["relevant_tests"])
    reward += 0.2 * test_score

    # Reasoning quality bonus (longer + step-appropriate)
    if step >= 3 and len(action.reasoning) > 80:
        reward += 0.1
    elif step < 3 and len(action.reasoning) > 30:
        reward += 0.05

    return round(min(max(reward, 0.0), 1.0), 3)


def _feedback_polytrauma(action: TriageAction, patient: Dict[str, Any]) -> str:
    dist = _level_distance(action.triage_level, patient["triage_level"])
    parts = []
    if dist == 0:
        parts.append("✅ Correct level: immediate")
    elif dist < 0:
        parts.append(f"🚨 CRITICAL UNDER-TRIAGE: you said {action.triage_level}. This is a multi-system emergency!")
    else:
        parts.append(f"⚠️ Over-triaged: {action.triage_level}")
    
    parts.append(f"Condition hints: {patient['condition_category']} pattern")
    return " | ".join(parts)


# ── Task registry ─────────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    name: str
    difficulty: str
    description: str
    max_steps: int
    grade_step: Callable
    get_feedback: Callable


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "vital_signs_triage": TaskSpec(
        name="vital_signs_triage",
        difficulty="easy",
        description=(
            "Classify a patient's urgency level from vitals and chief complaint only. "
            "One step to decide. Tests recommended but not required."
        ),
        max_steps=3,
        grade_step=_grade_vital_signs,
        get_feedback=_feedback_vital_signs,
    ),
    "differential_diagnosis": TaskSpec(
        name="differential_diagnosis",
        difficulty="medium",
        description=(
            "Identify the most likely condition from a multi-symptom patient and "
            "recommend appropriate diagnostic tests. Up to 5 steps, more history revealed each step."
        ),
        max_steps=5,
        grade_step=_grade_differential,
        get_feedback=_feedback_differential,
    ),
    "polytrauma_cascade": TaskSpec(
        name="polytrauma_cascade",
        difficulty="hard",
        description=(
            "Multi-system trauma/complex medical case. New critical findings emerge each step. "
            "Agent must continuously update its assessment. Requires integrating all revealed info. "
            "Frontier-model challenge."
        ),
        max_steps=5,
        grade_step=_grade_polytrauma,
        get_feedback=_feedback_polytrauma,
    ),
}
