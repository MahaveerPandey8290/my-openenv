"""
ClinicalTriageEnv v3 — true RL environment.

Stateful, multi-step, procedurally generated.
Supports OrderTestAction (tool use) and SubmitTriageAction (terminal).
Shaped reward at every step — no sparse end-only rewards.
"""
from __future__ import annotations
import uuid
from typing import Optional, Union

from openenv.core.env_server import Environment

from ..models import (
    OrderTestAction, SubmitTriageAction,
    PatientObservation, TriageState,
)
from .patient_generator import generate_patient
from .tasks import TASK_REGISTRY, grade_order_test
from .test_bank import get_test_result


def _clamp(r: float) -> float:
    return round(min(max(float(r), 0.01), 0.99), 2)


class ClinicalTriageEnv(Environment):
    """
    True RL environment for clinical triage with tool use.
    Follows OpenEnv Environment ABC — plugs into TRL GRPOTrainer.

    Episode flow:
      reset()                          -> initial observation
      step(OrderTestAction)            -> test result, partial reward
      step(OrderTestAction)            -> another test (up to max_tests)
      step(SubmitTriageAction)         -> final reward, done=True

    Key RL properties:
      - Procedural generation -> model cannot memorise patients
      - Shaped reward each step -> dense gradient signal for GRPO
      - Tool use -> genuine sequential decision problem
      - Under-triage penalty -> safety-aware reward function
    """

    VALID_TASKS = list(TASK_REGISTRY.keys())

    def __init__(self, task_name: str = "vital_signs_triage") -> None:
        super().__init__()
        self.task_name = task_name
        self._state: Optional[TriageState] = None
        self._patient: Optional[dict] = None
        self._episode_seed: Optional[int] = None

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> PatientObservation:
        """
        Start new episode. Procedurally generates a fresh patient.
        seed=None -> random (training). seed=int -> reproducible (eval).
        """
        if task_name and task_name in self.VALID_TASKS:
            self.task_name = task_name

        spec = TASK_REGISTRY[self.task_name]
        self._episode_seed = seed
        self._patient = generate_patient(self.task_name, seed=seed)

        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            task_name=self.task_name,
            ground_truth_level=self._patient["triage_level"],
            ground_truth_condition=self._patient["condition"],
            condition_category=self._patient["condition_category"],
            relevant_tests=self._patient["relevant_tests"],
            max_tests=spec.max_tests,
            max_steps=spec.max_steps,
            cumulative_reward=0.5,
            seed=seed,
        )

        return PatientObservation(
            patient_id=self._state.episode_id,
            chief_complaint=self._patient["complaint"],
            vitals=self._patient["vitals"],
            visible_symptoms=self._patient["initial_symptoms"],
            history_revealed=[],
            test_results={},
            tests_remaining=spec.max_tests,
            step_number=0,
            reward=0.5,
            done=False,
            feedback=(
                f"New patient. Task: {self.task_name}. "
                f"You may order up to {spec.max_tests} test(s) before submitting triage."
            ),
            cumulative_reward=0.5,
            available_actions=["order_test", "submit_triage"],
            task_name=self.task_name,
        )

    def step(
        self, action: Union[OrderTestAction, SubmitTriageAction]
    ) -> PatientObservation:
        if self._state is None or self._patient is None:
            raise RuntimeError("Call reset() before step().")

        self._state.steps_taken += 1
        step_n = self._state.steps_taken
        spec = TASK_REGISTRY[self.task_name]

        # Reveal incremental history
        revealed_history = self._build_history(step_n)
        visible = self._build_symptoms(step_n)

        # ── OrderTestAction ───────────────────────────────────────────────────
        if isinstance(action, OrderTestAction):
            self._state.tests_used += 1

            if self._state.tests_used > self._state.max_tests:
                reward = _clamp(0.01)
                feedback = f"[LIMIT] Test limit {self._state.max_tests} reached. Action ignored."
            else:
                result_str, is_relevant = get_test_result(
                    self._patient["condition_category"], action.test_name
                )
                self._state.all_test_results[action.test_name] = result_str
                self._state.tests_ordered.append(action.test_name)
                reward = grade_order_test(
                    is_relevant, self._state.tests_used, self._state.max_tests
                )
                tag = "[RELEVANT]" if is_relevant else "[LOW YIELD]"
                feedback = f"[TEST] {action.test_name}: {result_str} {tag}"

            self._update_cumulative(reward)
            tests_left = max(0, self._state.max_tests - self._state.tests_used)

            return PatientObservation(
                patient_id=self._state.episode_id,
                chief_complaint=self._patient["complaint"],
                vitals=self._patient["vitals"],
                visible_symptoms=visible,
                history_revealed=revealed_history,
                test_results=dict(self._state.all_test_results),
                tests_remaining=tests_left,
                step_number=step_n,
                reward=reward,
                done=False,
                feedback=feedback,
                cumulative_reward=self._state.cumulative_reward,
                available_actions=(
                    ["order_test", "submit_triage"] if tests_left > 0
                    else ["submit_triage"]
                ),
                task_name=self.task_name,
            )

        # ── SubmitTriageAction ────────────────────────────────────────────────
        elif isinstance(action, SubmitTriageAction):
            reward = spec.grade_fn(
                action, self._patient, step_n, self._state.tests_used
            )
            self._update_cumulative(reward)
            self._state.solved = (
                action.triage_level == self._patient["triage_level"]
            )
            feedback = spec.feedback_fn(action, self._patient, self._state.tests_used)

            return PatientObservation(
                patient_id=self._state.episode_id,
                chief_complaint=self._patient["complaint"],
                vitals=self._patient["vitals"],
                visible_symptoms=visible,
                history_revealed=revealed_history,
                test_results=dict(self._state.all_test_results),
                tests_remaining=0,
                step_number=step_n,
                reward=reward,
                done=True,
                feedback=feedback,
                cumulative_reward=self._state.cumulative_reward,
                available_actions=[],
                task_name=self.task_name,
            )

        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    @property
    def state(self) -> TriageState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def _build_history(self, step_n: int) -> list:
        items = self._patient.get("history", [])
        revealed = []
        for item in items:
            if item.startswith("step "):
                parts = item.split(":", 1)
                try:
                    if int(parts[0].replace("step ", "").strip()) <= step_n:
                        revealed.append(parts[1].strip())
                except (ValueError, IndexError):
                    revealed.append(item)
            else:
                revealed.append(item)
        return revealed

    def _build_symptoms(self, step_n: int) -> list:
        extra = self._patient.get("extra_symptoms", [])
        return self._patient["initial_symptoms"] + extra[: max(0, step_n - 1)]

    def _update_cumulative(self, reward: float) -> None:
        delta = reward - 0.5
        self._state.cumulative_reward = _clamp(
            self._state.cumulative_reward + delta
        )
