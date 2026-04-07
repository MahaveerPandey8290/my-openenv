"""
Pydantic typed models for ClinicalTriageEnv.
Action, Observation, State — all OpenEnv spec compliant.
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class TriageAction(Action):
    """
    Agent action: submit a triage assessment for the current patient.
    The agent must decide urgency level, suspected condition, tests, and reasoning.
    """
    triage_level: Literal["immediate", "urgent", "less_urgent", "non_urgent"] = Field(
        ...,
        description=(
            "Triage urgency level. "
            "'immediate'=life-threatening now, "
            "'urgent'=serious but stable, "
            "'less_urgent'=needs care soon, "
            "'non_urgent'=can wait."
        )
    )
    suspected_condition: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Primary diagnosis or condition the agent suspects."
    )
    recommended_tests: List[str] = Field(
        default_factory=list,
        description="List of diagnostic tests the agent recommends (e.g. ['ECG', 'troponin'])."
    )
    reasoning: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Agent's chain-of-thought justifying its assessment."
    )


class PatientObservation(Observation):
    """
    Partial patient info revealed to the agent each step.
    Info is revealed incrementally — more history appears as steps advance.
    """
    patient_id: str = Field(..., description="Unique episode/patient ID.")
    chief_complaint: str = Field(..., description="Patient's primary complaint in their own words.")
    vitals: Dict[str, str] = Field(
        ...,
        description="Current vital signs: HR, BP, SpO2, Temp, RR."
    )
    visible_symptoms: List[str] = Field(
        ...,
        description="Observable symptoms available at this step."
    )
    history_revealed: List[str] = Field(
        default_factory=list,
        description="Medical history items revealed so far (grows each step)."
    )
    step_number: int = Field(default=0, description="Current step in this episode.")
    reward: float = Field(default=0.0, description="Reward earned at this step.")
    done: bool = Field(default=False, description="True when the episode is finished.")
    feedback: str = Field(default="", description="Textual feedback from the grader.")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far.")


class TriageState(State):
    """
    Internal environment state — NOT sent to agent directly.
    Tracks ground truth and episode progress.
    """
    episode_id: str
    task_name: str
    ground_truth_level: str
    ground_truth_condition: str
    condition_category: str
    relevant_tests: List[str]
    steps_taken: int = 0
    cumulative_reward: float = 0.0
    max_steps: int = 5
    solved: bool = False
