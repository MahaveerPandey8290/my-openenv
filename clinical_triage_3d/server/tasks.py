"""
3D Task specs and graders for ClinicalTriageEnv 3D.

Three tasks:
  1. single_patient_rescue   (easy)   — 1 critical patient, navigate and triage
  2. ward_prioritisation     (medium) — 4 patients, must prioritise correctly
  3. mass_casualty_incident  (hard)   — 4 patients, time limit 60s, MCI scenario

Reward components:
  - Navigation efficiency (spatial)
  - Examination quality (clinical reasoning)
  - Intervention correctness (clinical action)
  - Final triage accuracy (graded per patient)
  - Time bonus (faster = higher reward)
  - Under-triage penalty (safety-critical)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

TRIAGE_ORDER = ["non_urgent", "less_urgent", "urgent", "immediate", "deceased"]


def _clamp(r: float) -> float:
    return round(min(max(float(r), 0.01), 0.99), 2)


def _level_dist(pred: str, true: str) -> int:
    try:
        return TRIAGE_ORDER.index(pred) - TRIAGE_ORDER.index(true)
    except ValueError:
        return 0


def _grade_triage_assignments(
    assignments: Dict[str, str],
    ground_truth: Dict[str, str],
    time_elapsed: float,
    max_time: float,
) -> float:
    """
    Grade all triage assignments at episode end.
    Returns shaped scalar in (0.01, 0.99).
    """
    if not assignments:
        return 0.01

    score = 0.0
    n = len(ground_truth)

    for bed_id, true_level in ground_truth.items():
        pred_level = assignments.get(bed_id, "non_urgent")
        dist = _level_dist(pred_level, true_level)

        if dist == 0:
            score += 0.55 / n           # correct
        elif abs(dist) == 1:
            score += 0.22 / n           # one level off
        elif dist < 0:
            score -= 0.35 / n           # under-triage — dangerous

    # Time bonus — faster is better
    time_fraction = time_elapsed / max(max_time, 1.0)
    time_bonus = 0.12 * max(0.0, 1.0 - time_fraction)
    score += time_bonus

    return _clamp(score)


def _grade_easy(assignments, patients, time_elapsed, **kw) -> float:
    ground_truth = {bid: p["triage_level"] for bid, p in patients.items()}
    return _grade_triage_assignments(assignments, ground_truth, time_elapsed, 120.0)


def _grade_medium(assignments, patients, time_elapsed, **kw) -> float:
    ground_truth = {bid: p["triage_level"] for bid, p in patients.items()}
    return _grade_triage_assignments(assignments, ground_truth, time_elapsed, 120.0)


def _grade_hard(assignments, patients, time_elapsed, **kw) -> float:
    ground_truth = {bid: p["triage_level"] for bid, p in patients.items()}
    # Harder time limit for MCI
    return _grade_triage_assignments(assignments, ground_truth, time_elapsed, 60.0)


@dataclass
class TaskSpec3D:
    name: str
    difficulty: str
    description: str
    n_patients: int
    max_time_seconds: float
    max_steps: int
    grade_fn: Callable


TASK_REGISTRY_3D: Dict[str, TaskSpec3D] = {
    "single_patient_rescue": TaskSpec3D(
        name="single_patient_rescue",
        difficulty="easy",
        description=(
            "Navigate to the single critical patient, examine them, "
            "apply one intervention, submit correct triage. "
            "Time limit 120s."
        ),
        n_patients=1,
        max_time_seconds=120.0,
        max_steps=8,
        grade_fn=_grade_easy,
    ),
    "ward_prioritisation": TaskSpec3D(
        name="ward_prioritisation",
        difficulty="medium",
        description=(
            "4 patients with mixed severity. "
            "Agent must navigate the ward, examine each patient, "
            "and correctly prioritise all 4. Time limit 120s."
        ),
        n_patients=4,
        max_time_seconds=120.0,
        max_steps=20,
        grade_fn=_grade_medium,
    ),
    "mass_casualty_incident": TaskSpec3D(
        name="mass_casualty_incident",
        difficulty="hard",
        description=(
            "MCI scenario: 4 patients, all critical or urgent. "
            "Time limit 60s. Agent must triage rapidly with minimal "
            "examination. Tests speed vs accuracy tradeoff."
        ),
        n_patients=4,
        max_time_seconds=60.0,
        max_steps=15,
        grade_fn=_grade_hard,
    ),
}
