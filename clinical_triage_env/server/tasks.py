"""
Task specs and shaped reward graders for ClinicalTriageEnv v3.

Reward philosophy (Reasoning Gym style):
  - Every step gives signal — no sparse rewards
  - Ordering the RIGHT test rewards clinical reasoning
  - Under-triage is punished harder than over-triage (real medicine)
  - Efficiency bonus for earlier correct answers
  - Reasoning length rewarded (encourages chain-of-thought)
  - All scores strictly in (0.01, 0.99) — validator requirement
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from ..models import SubmitTriageAction

TRIAGE_ORDER = ["non_urgent", "less_urgent", "urgent", "immediate"]


def _clamp(r: float) -> float:
    return round(min(max(float(r), 0.01), 0.99), 2)


def _level_dist(pred: str, true: str) -> int:
    try:
        return TRIAGE_ORDER.index(pred) - TRIAGE_ORDER.index(true)
    except ValueError:
        return 0


def _condition_hit(suspected: str, category: str) -> float:
    return 0.18 if category.lower() in suspected.lower() else 0.0


def grade_order_test(is_relevant: bool, tests_used: int, max_tests: int) -> float:
    """
    Shaped reward for test ordering.
    Returns value around 0.5 — positive if relevant, negative if not.
    """
    base = 0.5
    delta = 0.15 if is_relevant else -0.05
    if tests_used > max_tests:
        delta -= 0.08 * (tests_used - max_tests)
    return _clamp(base + delta)


def _grade_easy(
    action: SubmitTriageAction,
    patient: Dict[str, Any],
    step: int,
    tests_used: int,
) -> float:
    r = 0.0
    dist = _level_dist(action.triage_level, patient["triage_level"])

    if dist == 0:
        r += 0.55
        if step <= 2:
            r += 0.08   # efficiency bonus
    elif abs(dist) == 1:
        r += 0.18
    elif dist < 0:
        r -= 0.30       # under-triage penalty

    r += _condition_hit(action.suspected_condition, patient["condition_category"])

    if len(action.reasoning) > 30:
        r += 0.06       # reasoning quality

    if tests_used >= 1:
        r += 0.05       # tool use bonus

    return _clamp(r)


def _grade_medium(
    action: SubmitTriageAction,
    patient: Dict[str, Any],
    step: int,
    tests_used: int,
) -> float:
    r = 0.0
    dist = _level_dist(action.triage_level, patient["triage_level"])

    if dist == 0:
        r += 0.42
        if step <= 3:
            r += 0.08
    elif abs(dist) == 1:
        r += 0.14
    elif dist < 0:
        r -= 0.28

    r += _condition_hit(action.suspected_condition, patient["condition_category"])

    if len(action.reasoning) > 50:
        r += 0.05

    # Tool use bonus scales with tests ordered
    r += min(tests_used, 2) * 0.06

    return _clamp(r)


def _grade_hard(
    action: SubmitTriageAction,
    patient: Dict[str, Any],
    step: int,
    tests_used: int,
) -> float:
    r = 0.0
    dist = _level_dist(action.triage_level, patient["triage_level"])

    weight = 0.28 + 0.04 * min(step, 5)
    if dist == 0:
        r += weight
    elif abs(dist) == 1:
        r += weight * 0.28
    elif dist < 0:
        r -= 0.38       # hardest penalty for hard task

    r += _condition_hit(action.suspected_condition, patient["condition_category"])

    if step >= 3 and len(action.reasoning) > 80:
        r += 0.08
    elif len(action.reasoning) > 30:
        r += 0.04

    r += min(tests_used, 3) * 0.05

    return _clamp(r)


def _feedback(action: SubmitTriageAction, patient: Dict[str, Any], tests_used: int) -> str:
    dist = _level_dist(action.triage_level, patient["triage_level"])
    tag = "[CORRECT]" if dist == 0 else ("[WARNING]" if dist > 0 else "[CRITICAL]")
    return (
        f"{tag} said={action.triage_level} truth={patient['triage_level']} "
        f"tests_used={tests_used}"
    )


@dataclass
class TaskSpec:
    name: str
    difficulty: str
    description: str
    max_steps: int
    max_tests: int
    grade_fn: Callable
    feedback_fn: Callable = _feedback


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "vital_signs_triage": TaskSpec(
        name="vital_signs_triage",
        difficulty="easy",
        description="Classify urgency from vitals and complaint. 1 test allowed.",
        max_steps=4,
        max_tests=1,
        grade_fn=_grade_easy,
    ),
    "differential_diagnosis": TaskSpec(
        name="differential_diagnosis",
        difficulty="medium",
        description="Identify condition + triage. 2 tests allowed. Relevant tests earn bonus.",
        max_steps=6,
        max_tests=2,
        grade_fn=_grade_medium,
    ),
    "polytrauma_cascade": TaskSpec(
        name="polytrauma_cascade",
        difficulty="hard",
        description="Multi-system emergency. 3 tests. Cascading findings. Frontier-model challenge.",
        max_steps=8,
        max_tests=3,
        grade_fn=_grade_hard,
    ),
}
