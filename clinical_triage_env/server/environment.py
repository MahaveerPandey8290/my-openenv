"""
ClinicalTriageEnv — core Environment implementation.
Follows OpenEnv Environment ABC: reset() / step() / state property.
"""
from __future__ import annotations
import uuid
from typing import Optional

from openenv.core.env_server import Environment

from ..models import PatientObservation, TriageAction, TriageState
from .patient_generator import generate_patient
from .tasks import TASK_REGISTRY


class ClinicalTriageEnv(Environment):
    """
    A clinical triage environment where an AI agent evaluates synthetic
    patient presentations and decides on urgency level, diagnosis, and tests.

    Three tasks of increasing difficulty:
      1. vital_signs_triage  (easy)
      2. differential_diagnosis  (medium)
      3. polytrauma_cascade  (hard)
    """

    VALID_TASKS = list(TASK_REGISTRY.keys())

    def __init__(self, task_name: str = "vital_signs_triage") -> None:
        super().__init__()
        if task_name not in self.VALID_TASKS:
            raise ValueError(f"task_name must be one of {self.VALID_TASKS}")
        self.task_name = task_name
        self._state: Optional[TriageState] = None
        self._patient: Optional[dict] = None
        self._task_spec = TASK_REGISTRY[task_name]

    # ── OpenEnv required methods ──────────────────────────────────────────────

    def reset(self, task_name: Optional[str] = None) -> PatientObservation:
        """
        Start a new episode. Generates a fresh synthetic patient.
        Returns the initial observation (partial info only).
        """
        if task_name and task_name in self.VALID_TASKS:
            self.task_name = task_name
            self._task_spec = TASK_REGISTRY[task_name]

        self._patient = generate_patient(self.task_name)
        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            task_name=self.task_name,
            ground_truth_level=self._patient["triage_level"],
            ground_truth_condition=self._patient["condition"],
            condition_category=self._patient["condition_category"],
            relevant_tests=self._patient["relevant_tests"],
            max_steps=self._task_spec.max_steps,
        )

        return PatientObservation(
            patient_id=self._state.episode_id,
            chief_complaint=self._patient["complaint"],
            vitals=self._patient["vitals"],
            visible_symptoms=self._patient["initial_symptoms"],
            history_revealed=[],
            step_number=0,
            reward=0.0,
            done=False,
            feedback="New patient arrived. Assess and triage.",
            cumulative_reward=0.0,
        )

    def step(self, action: TriageAction) -> PatientObservation:
        """
        Execute one triage action. Reveals more patient info and returns shaped reward.
        """
        if self._state is None or self._patient is None:
            raise RuntimeError("Call reset() before step().")

        self._state.steps_taken += 1
        step_n = self._state.steps_taken

        # Reveal incremental symptoms and history
        extra = self._patient.get("extra_symptoms", [])
        revealed_extra = extra[: step_n - 1] if extra else []
        visible = self._patient["initial_symptoms"] + revealed_extra

        history_items = self._patient.get("history", [])
        # Parse "step N:" prefix format used in polytrauma patients
        revealed_history = []
        for item in history_items:
            if item.startswith("step "):
                parts = item.split(":", 1)
                try:
                    item_step = int(parts[0].replace("step ", "").strip())
                    if item_step <= step_n:
                        revealed_history.append(parts[1].strip())
                except (ValueError, IndexError):
                    revealed_history.append(item)
            else:
                revealed_history.append(item)

        # Compute shaped reward
        reward = self._task_spec.grade_step(action, self._patient, step_n)
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 3)

        # Determine if episode is done
        max_reached = step_n >= self._state.max_steps
        correct_level = action.triage_level == self._patient["triage_level"]
        done = max_reached or correct_level
        if done:
            self._state.solved = correct_level

        feedback = self._task_spec.get_feedback(action, self._patient)

        return PatientObservation(
            patient_id=self._state.episode_id,
            chief_complaint=self._patient["complaint"],
            vitals=self._patient["vitals"],
            visible_symptoms=visible,
            history_revealed=revealed_history,
            step_number=step_n,
            reward=reward,
            done=done,
            feedback=feedback,
            cumulative_reward=self._state.cumulative_reward,
        )

    @property
    def state(self) -> TriageState:
        """Return current internal state (ground truth + progress)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state
