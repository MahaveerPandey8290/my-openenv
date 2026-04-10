"""
ClinicalTriageEnv 3D — core RL environment.
Combines 3D spatial simulation + clinical triage + visual observations.
Like CARLA: renders scene -> camera frame -> base64 JPEG -> agent observes.
"""
from __future__ import annotations
import math
import uuid
from typing import Dict, Optional, Union

from openenv.core.env_server import Environment

from ..models import (
    MoveToAction, ExaminePatientAction, OrderTestAction,
    InterventionAction, SubmitTriageAction,
    VisualObservation, TriageState3D,
)
from .patient_generator import generate_patient
from .renderer import WardRenderer
from .ward_state import WardState, POSITION_XYZ
from .tasks import TASK_REGISTRY_3D
from .test_bank import get_test_result


def _clamp(r: float) -> float:
    return round(min(max(float(r), 0.01), 0.99), 2)


class ClinicalTriageEnv3D(Environment):
    """
    3D Embodied Medical Agent environment.

    The agent operates inside a rendered 3D emergency department.
    Each step returns a base64 JPEG of the current camera view
    alongside structured clinical context.

    Action space:
      move_to          — navigate to a location (spatial cost)
      examine_patient  — examine a patient at nearby bed
      order_test       — order diagnostic test (tool use)
      intervene        — apply physical intervention
      submit_triage    — final triage for all patients (ends episode)

    Observation space:
      image_base64     — 84x84 JPEG of current 3D camera view
      beds_summary     — clinical state of each bed
      nearby_beds      — beds within examination distance
      time_remaining   — seconds left in episode

    Reward: shaped at every step (spatial + clinical + time)
    """

    VALID_TASKS = list(TASK_REGISTRY_3D.keys())

    def __init__(
        self,
        task_name: str = "ward_prioritisation",
        render_width: int = 84,
        render_height: int = 84,
    ) -> None:
        super().__init__()
        self.task_name = task_name
        self._renderer = WardRenderer(render_width, render_height)
        self._ward = WardState()
        self._state: Optional[TriageState3D] = None
        self._patients: Dict[str, dict] = {}
        self._exam_results: Dict[str, Dict[str, list]] = {}
        self._test_results: Dict[str, Dict[str, str]] = {}
        self._interventions: Dict[str, list] = {}

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> VisualObservation:
        if task_name and task_name in self.VALID_TASKS:
            self.task_name = task_name

        spec = TASK_REGISTRY_3D[self.task_name]

        # Generate patients
        self._patients = {}
        severity_map = {}
        bed_ids = [f"bed_{i+1}" for i in range(spec.n_patients)]

        for i, bed_id in enumerate(bed_ids):
            patient_seed = (seed * 100 + i) if seed is not None else None
            p = generate_patient(self.task_name, seed=patient_seed)
            self._patients[bed_id] = p
            # Map triage level -> visual alert level
            severity_map[bed_id] = {
                "immediate": "critical",
                "urgent": "warning",
                "less_urgent": "stable",
                "non_urgent": "stable",
            }.get(p["triage_level"], "unknown")

        self._exam_results = {b: {} for b in bed_ids}
        self._test_results = {b: {} for b in bed_ids}
        self._interventions = {b: [] for b in bed_ids}

        self._ward.reset(severity_map)

        self._state = TriageState3D(
            episode_id=str(uuid.uuid4()),
            task_name=self.task_name,
            patients=self._patients,
            agent_position="nurses_station",
            agent_position_xyz=[6.0, 0.0, 4.0],
            seed=seed,
        )

        image_b64 = self._renderer.render_frame(
            "nurses_station", self._build_patient_state_for_renderer()
        )

        return VisualObservation(
            image_base64=image_b64,
            image_width=self._renderer.width,
            image_height=self._renderer.height,
            agent_location="nurses_station",
            time_elapsed_seconds=0.0,
            time_remaining_seconds=spec.max_time_seconds,
            beds_summary=self._build_beds_summary(from_position="nurses_station"),
            nearby_beds=self._ward.get_nearby_beds(),
            available_actions=["move_to", "examine_patient", "order_test", "submit_triage"],
            task_name=self.task_name,
            step_number=0,
            reward=0.5,
            done=False,
            feedback=(
                f"You are in the ED ward. {spec.n_patients} patient(s) need triage. "
                f"Time limit: {spec.max_time_seconds}s. Prioritise the most critical."
            ),
            cumulative_reward=0.5,
            n_patients=spec.n_patients,
        )

    def step(
        self, action: Union[
            MoveToAction, ExaminePatientAction, OrderTestAction,
            InterventionAction, SubmitTriageAction
        ]
    ) -> VisualObservation:
        if self._state is None:
            raise RuntimeError("Call reset() first.")

        self._state.steps_taken += 1
        spec = TASK_REGISTRY_3D[self.task_name]
        reward, feedback, done = 0.5, "", False

        # ── MoveToAction ──────────────────────────────────────────────────────
        if isinstance(action, MoveToAction):
            reward, feedback = self._ward.move_to(action.target)
            self._state.agent_position = self._ward.agent_position

        # ── ExaminePatientAction ──────────────────────────────────────────────
        elif isinstance(action, ExaminePatientAction):
            reward, feedback, success = self._ward.examine(
                action.bed_id, action.exam_type
            )
            if success:
                patient = self._patients.get(action.bed_id, {})
                new_findings = self._get_exam_findings(
                    patient, action.exam_type
                )
                if action.bed_id not in self._exam_results:
                    self._exam_results[action.bed_id] = {}
                self._exam_results[action.bed_id][action.exam_type] = new_findings
                feedback += f" Findings: {', '.join(new_findings)}"

        # ── OrderTestAction ───────────────────────────────────────────────────
        elif isinstance(action, OrderTestAction):
            patient = self._patients.get(action.bed_id, {})
            result_str, is_relevant = get_test_result(
                patient.get("condition_category", "cardiac"), action.test_name
            )
            self._test_results[action.bed_id][action.test_name] = result_str
            reward = 0.65 if is_relevant else 0.4
            tag = "[RELEVANT]" if is_relevant else "[LOW YIELD]"
            feedback = f"[TEST] {action.bed_id}/{action.test_name}: {result_str[:80]} {tag}"
            self._ward.time_elapsed += 3.0

        # ── InterventionAction ────────────────────────────────────────────────
        elif isinstance(action, InterventionAction):
            reward, feedback = self._ward.intervene(
                action.bed_id, action.intervention
            )
            if reward > 0.5:
                self._interventions[action.bed_id].append(action.intervention)
                self._state.interventions_done = dict(self._interventions)

        # ── SubmitTriageAction ────────────────────────────────────────────────
        elif isinstance(action, SubmitTriageAction):
            reward = spec.grade_fn(
                assignments=action.triage_assignments,
                patients=self._patients,
                time_elapsed=self._ward.time_elapsed,
            )
            done = True
            correct = sum(
                1 for bid, level in action.triage_assignments.items()
                if self._patients.get(bid, {}).get("triage_level") == level
            )
            feedback = (
                f"Triage submitted. {correct}/{spec.n_patients} correct. "
                f"Time: {self._ward.time_elapsed:.1f}s. Score: {reward:.2f}"
            )

        # Timeout
        if self._ward.is_timed_out() and not done:
            done = True
            reward = _clamp(reward * 0.5)
            feedback = f"TIME OUT ({spec.max_time_seconds}s). Episode ended. " + feedback

        self._state.cumulative_reward = _clamp(
            self._state.cumulative_reward + (reward - 0.5)
        )
        self._state.time_elapsed = self._ward.time_elapsed

        # Render new frame
        image_b64 = self._renderer.render_frame(
            self._ward.agent_position,
            self._build_patient_state_for_renderer(),
        )

        avail = ["move_to", "examine_patient", "order_test", "intervene", "submit_triage"]
        if done:
            avail = []

        return VisualObservation(
            image_base64=image_b64,
            image_width=self._renderer.width,
            image_height=self._renderer.height,
            agent_location=self._ward.agent_position,
            time_elapsed_seconds=self._ward.time_elapsed,
            time_remaining_seconds=self._ward.time_remaining(),
            beds_summary=self._build_beds_summary(self._ward.agent_position),
            nearby_beds=self._ward.get_nearby_beds(),
            available_actions=avail,
            task_name=self.task_name,
            step_number=self._state.steps_taken,
            reward=reward,
            done=done,
            feedback=feedback,
            cumulative_reward=self._state.cumulative_reward,
            n_patients=spec.n_patients,
        )

    @property
    def state(self) -> TriageState3D:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def close(self):
        self._renderer.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_patient_state_for_renderer(self) -> Dict[str, Dict]:
        result = {}
        for bed_id, patient in self._patients.items():
            alert = {
                "immediate": "critical",
                "urgent": "warning",
                "less_urgent": "stable",
                "non_urgent": "stable",
            }.get(patient.get("triage_level", "non_urgent"), "unknown")
            result[bed_id] = {"alert_level": alert}
        return result

    def _build_beds_summary(self, from_position: str) -> Dict[str, Dict]:
        summary = {}
        for bed_id, patient in self._patients.items():
            pos_a = POSITION_XYZ.get(from_position, (6, 0, 4))
            pos_b = POSITION_XYZ.get(bed_id, (6, 0, 4))
            dist = math.sqrt(sum((a-b)**2 for a, b in zip(pos_a, pos_b)))
            summary[bed_id] = {
                "distance_metres": round(dist, 1),
                "alert_level": {
                    "immediate": "critical", "urgent": "warning",
                    "less_urgent": "stable", "non_urgent": "stable",
                }.get(patient.get("triage_level", ""), "unknown"),
                "chief_complaint_visible": dist <= 3.0,
                "chief_complaint": patient.get("complaint", "")[:60] if dist <= 3.0 else "",
                "exam_results": self._exam_results.get(bed_id, {}),
                "test_results": self._test_results.get(bed_id, {}),
                "interventions": self._interventions.get(bed_id, []),
            }
        return summary

    def _get_exam_findings(self, patient: dict, exam_type: str) -> list:
        """Return realistic findings based on exam type and patient condition."""
        symptoms = patient.get("initial_symptoms", [])
        vitals = patient.get("vitals", {})
        category = patient.get("condition_category", "")

        findings_map = {
            "visual": symptoms[:2] + [f"HR visible on monitor: {vitals.get('HR','?')}"],
            "vitals": [f"{k}: {v}" for k, v in vitals.items()],
            "auscultation": {
                "cardiac": ["Irregular heart rhythm", "Murmur grade III/VI"],
                "pulmonary_embol": ["Reduced air entry right base", "Pleural rub"],
                "sepsis": ["Coarse crepitations bilateral bases"],
            }.get(category, ["Normal breath sounds", "Normal heart sounds"]),
            "palpation": {
                "appendic": ["Rebound tenderness RLQ", "Guarding present", "Rovsing positive"],
                "trauma": ["Unstable pelvis on compression", "Rigid abdomen"],
            }.get(category, ["Soft abdomen", "No tenderness"]),
        }
        return findings_map.get(exam_type, symptoms[:1])
