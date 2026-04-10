"""
Typed Pydantic models for ClinicalTriageEnv v3.
Two action types: OrderTestAction (tool use) and SubmitTriageAction (final).
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Union
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ── Actions ───────────────────────────────────────────────────────────────────

class OrderTestAction(Action):
    """
    Tool-use action: agent orders one diagnostic test.
    Environment returns realistic result in next observation.
    Reward: +0.15 if relevant, -0.05 if irrelevant.
    """
    action_type: Literal["order_test"] = "order_test"
    test_name: str = Field(
        ..., min_length=2, max_length=100,
        description="Test to order. E.g. ECG, troponin, CT head, FBC, D-dimer, ABG."
    )
    reasoning: str = Field(
        ..., min_length=5, max_length=500,
        description="Clinical reasoning for ordering this test."
    )


class SubmitTriageAction(Action):
    """
    Final action: agent submits triage level and diagnosis.
    This ends the episode and triggers full grading.
    """
    action_type: Literal["submit_triage"] = "submit_triage"
    triage_level: Literal["immediate", "urgent", "less_urgent", "non_urgent"] = Field(
        ...,
        description=(
            "immediate=life-threatening now | urgent=serious, assess in 15min | "
            "less_urgent=can wait 1hr | non_urgent=minor"
        )
    )
    suspected_condition: str = Field(..., min_length=2, max_length=200)
    reasoning: str = Field(..., min_length=10, max_length=1000)


# Union used by the server
ClinicalAction = Union[OrderTestAction, SubmitTriageAction]


# ── Observations ──────────────────────────────────────────────────────────────

class PatientObservation(Observation):
    """
    What the agent sees each step.
    test_results grows as agent orders tests.
    history_revealed grows each step.
    """
    patient_id: str
    chief_complaint: str
    vitals: Dict[str, str]
    visible_symptoms: List[str]
    history_revealed: List[str] = Field(default_factory=list)
    test_results: Dict[str, str] = Field(
        default_factory=dict,
        description="Results of tests ordered so far."
    )
    tests_remaining: int = Field(default=3)
    step_number: int = 0
    reward: float = 0.5
    done: bool = False
    feedback: str = ""
    cumulative_reward: float = 0.5
    available_actions: List[str] = Field(
        default_factory=lambda: ["order_test", "submit_triage"]
    )
    task_name: str = "vital_signs_triage"


# ── State ─────────────────────────────────────────────────────────────────────

class TriageState(State):
    """
    Full internal state — ground truth + progress tracking.
    Returned by GET /state.
    """
    episode_id: str
    task_name: str
    ground_truth_level: str
    ground_truth_condition: str
    condition_category: str
    relevant_tests: List[str]
    all_test_results: Dict[str, str] = Field(default_factory=dict)
    tests_ordered: List[str] = Field(default_factory=list)
    steps_taken: int = 0
    tests_used: int = 0
    max_tests: int = 3
    max_steps: int = 8
    cumulative_reward: float = 0.5
    solved: bool = False
    seed: Optional[int] = None
