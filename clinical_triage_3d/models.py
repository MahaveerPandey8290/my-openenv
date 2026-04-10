"""
Pydantic models for ClinicalTriageEnv 3D.
Observation includes base64 camera image — like CARLA env.
Actions are spatial + clinical — agent moves and acts in 3D space.
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Union
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ── Actions (5 types — genuine multi-action RL) ───────────────────────────────

class MoveToAction(Action):
    """Navigate agent to a location in the ED ward."""
    action_type: Literal["move_to"] = "move_to"
    target: Literal["bed_1", "bed_2", "bed_3", "bed_4",
                    "nurses_station", "equipment_cart", "exit"] = Field(
        ..., description="Target location to move to."
    )
    reasoning: str = Field(..., min_length=3, max_length=200)


class ExaminePatientAction(Action):
    """Examine a specific patient — reveals detailed clinical signs."""
    action_type: Literal["examine_patient"] = "examine_patient"
    bed_id: Literal["bed_1", "bed_2", "bed_3", "bed_4"] = Field(
        ..., description="Which bed to examine."
    )
    exam_type: Literal["visual", "vitals", "auscultation", "palpation"] = Field(
        ..., description="Type of examination."
    )
    reasoning: str = Field(..., min_length=3, max_length=200)


class OrderTestAction(Action):
    """Order a diagnostic test for the current patient."""
    action_type: Literal["order_test"] = "order_test"
    bed_id: Literal["bed_1", "bed_2", "bed_3", "bed_4"]
    test_name: str = Field(..., min_length=2, max_length=100)
    reasoning: str = Field(..., min_length=5, max_length=300)


class InterventionAction(Action):
    """Perform a physical intervention on a patient."""
    action_type: Literal["intervene"] = "intervene"
    bed_id: Literal["bed_1", "bed_2", "bed_3", "bed_4"]
    intervention: Literal[
        "oxygen_mask", "iv_access", "defib_pads",
        "cervical_collar", "tourniquet", "bag_valve_mask"
    ] = Field(..., description="Intervention to apply.")
    reasoning: str = Field(..., min_length=5, max_length=300)


class SubmitTriageAction(Action):
    """Submit final triage for one or all patients. Ends the episode."""
    action_type: Literal["submit_triage"] = "submit_triage"
    triage_assignments: Dict[str, Literal[
        "immediate", "urgent", "less_urgent", "non_urgent", "deceased"
    ]] = Field(
        ...,
        description="Dict mapping bed_id -> triage level for each patient."
    )
    reasoning: str = Field(..., min_length=10, max_length=1000)


# Union of all valid actions
ClinicalAction3D = Union[
    MoveToAction, ExaminePatientAction, OrderTestAction,
    InterventionAction, SubmitTriageAction
]


# ── Observations (visual + spatial) ──────────────────────────────────────────

class VisualObservation(Observation):
    """
    Full observation — camera frame + spatial + clinical context.
    Like CARLA env: image_base64 is the primary RL signal.
    """
    # Visual channel — rendered 3D scene as base64 JPEG
    image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded JPEG of current 3D camera view. "
            "84x84 pixels for training, 256x256 for evaluation."
        )
    )
    image_width: int = 84
    image_height: int = 84

    # Spatial state
    agent_location: str = Field(
        ..., description="Current agent position in the ward."
    )
    agent_facing: str = Field(
        default="north", description="Cardinal direction agent faces."
    )
    time_elapsed_seconds: float = Field(
        default=0.0, description="Episode time. Every second costs -0.01 reward."
    )
    time_remaining_seconds: float = Field(
        default=120.0, description="Remaining time before timeout."
    )

    # Clinical state
    beds_summary: Dict[str, Dict] = Field(
        default_factory=dict,
        description="High-level summary of each bed visible from current position."
    )
    nearby_beds: List[str] = Field(
        default_factory=list,
        description="Bed IDs within examination distance (<=2m)."
    )
    available_actions: List[str] = Field(
        default_factory=list
    )

    # Task context
    task_name: str = "multi_patient_triage"
    step_number: int = 0
    reward: float = 0.5
    done: bool = False
    feedback: str = ""
    cumulative_reward: float = 0.5
    n_patients: int = 4


class TriageState3D(State):
    """Full internal state — ground truth for all patients."""
    episode_id: str
    task_name: str
    patients: Dict[str, Dict]  # bed_id -> patient ground truth
    agent_position: str
    agent_position_xyz: List[float]
    steps_taken: int = 0
    time_elapsed: float = 0.0
    examinations_done: Dict[str, List[str]] = Field(default_factory=dict)
    tests_ordered: Dict[str, List[str]] = Field(default_factory=dict)
    interventions_done: Dict[str, List[str]] = Field(default_factory=dict)
    triage_submitted: bool = False
    cumulative_reward: float = 0.5
    seed: Optional[int] = None
