"""
Task definitions and step-level graders for ClinicalTriageEnv.

Scores are strictly in the open interval (0.0, 1.0) — exclusive.
The Meta OpenEnv validator rejects exactly 0.0 or exactly 1.0.
All return values go through _clamp() which enforces [0.01, 0.99].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict

from ..models import TriageAction


TRIAGE_ORDER = ["non_urgent", "less_urgent", "urgent", "immediate"]


def _clamp(reward: float) -> float:
    """
    Clamp reward to strictly open interval (0.0, 1.0).
    Validator requires: 0.0 < score < 1.0 (exclusive on both ends).
    Floor = 0.01, ceiling = 0.99.
    """
    return round(min(max(float(reward), 0.01), 0.99), 2)


def _level_distance(pred: str, true: str) -> int:
    """Signed distance: positive = over-triage, negative = under-triage."""
    try:
        return TRIAGE_ORDER.index(pred) - TRIAGE_ORDER.index(true)
    except ValueError:
        return 0


def _test_overlap(ordered: list, correct: list) -> float:
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
    return 0.2 if category.lower() in suspected.lower() else 0.0


# ── EASY task grader ──────────────────────────────────────────────────────────

def _grade_vital_signs(action: TriageAction, patient: Dict[str, Any], step: int) -> float:
    reward = 0.0
    dist = _level_distance(action.triage_level, patient["triage_level"])

    if dist == 0:
        reward += 0.55          # correct level — kept below 0.99 ceiling room
        if step <= 2:
            reward += 0.09      # efficiency bonus (total 0.64, still < 0.99)
    elif abs(dist) == 1:
        reward += 0.18          # one level off
    elif dist < 0:
        reward -= 0.25          # under-triage penalty

    reward += _condition_match(action.suspected_condition, patient["condition_category"])

    if action.reasoning and len(action.reasoning) > 20:
        reward += 0.09          # reasoning present

    return _clamp(reward)


def _feedback_vital_signs(action: TriageAction, patient: Dict[str, Any]) -> str:
    dist = _level_distance(action.triage_level, patient["triage_level"])
    if dist == 0:
        return f"[CORRECT] triage level. Ground truth: {patient['triage_level']}."
    elif dist > 0:
        return (
            f"[WARNING] Over-triaged (you said {action.triage_level}, "
            f"truth is {patient['triage_level']})."
        )
    else:
        return (
            f"[CRITICAL] UNDER-TRIAGE! You said {action.triage_level}, "
            f"truth is {patient['triage_level']}. This could harm the patient."
        )


# ── MEDIUM task grader ────────────────────────────────────────────────────────

def _grade_differential(action: TriageAction, patient: Dict[str, Any], step: int) -> float:
    reward = 0.0
    dist = _level_distance(action.triage_level, patient["triage_level"])

    if dist == 0:
        reward += 0.38
    elif abs(dist) == 1:
        reward += 0.13
    elif dist < 0:
        reward -= 0.22

    reward += _condition_match(action.suspected_condition, patient["condition_category"])

    test_score = _test_overlap(action.recommended_tests, patient["relevant_tests"])
    reward += 0.23 * test_score

    if step <= 2 and dist == 0:
        reward += 0.09          # early correct bonus

    if len(action.reasoning) > 40:
        reward += 0.05

    return _clamp(reward)


def _feedback_differential(action: TriageAction, patient: Dict[str, Any]) -> str:
    lines = []
    dist = _level_distance(action.triage_level, patient["triage_level"])
    if dist == 0:
        lines.append(f"[CORRECT] level: {patient['triage_level']}")
    else:
        direction = "over-triaged" if dist > 0 else "UNDER-TRIAGED"
        icon = "[WARNING]" if dist > 0 else "[CRITICAL]"
        lines.append(
            f"{icon} {direction}: you said {action.triage_level}, "
            f"truth={patient['triage_level']}"
        )
    overlap = _test_overlap(action.recommended_tests, patient["relevant_tests"])
    lines.append(
        f"Tests: {overlap * 100:.0f}% of key tests ordered. "
        f"Key tests: {', '.join(patient['relevant_tests'][:3])}"
    )
    return " | ".join(lines)


# ── HARD task grader ──────────────────────────────────────────────────────────

def _grade_polytrauma(action: TriageAction, patient: Dict[str, Any], step: int) -> float:
    reward = 0.0
    dist = _level_distance(action.triage_level, patient["triage_level"])

    level_weight = 0.28 + 0.05 * min(step, 4)   # step1→0.33 … step5→0.48, stays < 0.99
    if dist == 0:
        reward += level_weight
    elif abs(dist) == 1:
        reward += level_weight * 0.28
    elif dist < 0:
        reward -= 0.35

    reward += _condition_match(action.suspected_condition, patient["condition_category"])

    test_score = _test_overlap(action.recommended_tests, patient["relevant_tests"])
    reward += 0.18 * test_score

    if step >= 3 and len(action.reasoning) > 80:
        reward += 0.09
    elif step < 3 and len(action.reasoning) > 30:
        reward += 0.04

    return _clamp(reward)


def _feedback_polytrauma(action: TriageAction, patient: Dict[str, Any]) -> str:
    dist = _level_distance(action.triage_level, patient["triage_level"])
    parts = []
    if dist == 0:
        parts.append("[CORRECT] level: immediate")
    elif dist < 0:
        parts.append(
            f"CRITICAL UNDER-TRIAGE: you said {action.triage_level}. "
            "This is a multi-system emergency!"
        )
    else:
        parts.append(f"[WARNING] Over-triaged: {action.triage_level}")
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
            "Identify the most likely clinical condition and recommend appropriate "
            "diagnostic tests from a multi-symptom patient. Up to 5 steps, "
            "more history revealed each step."
        ),
        max_steps=5,
        grade_step=_grade_differential,
        get_feedback=_feedback_differential,
    ),
    "polytrauma_cascade": TaskSpec(
        name="polytrauma_cascade",
        difficulty="hard",
        description=(
            "Multi-system trauma or complex medical emergency. Critical findings "
            "cascade across 5 steps. Agent must continuously update its assessment. "
            "Designed to challenge frontier LLMs."
        ),
        max_steps=5,
        grade_step=_grade_polytrauma,
        get_feedback=_feedback_polytrauma,
    ),
}
