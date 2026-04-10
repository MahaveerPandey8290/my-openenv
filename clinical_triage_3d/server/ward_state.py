"""
3D Ward State Engine — manages the spatial state of the ED ward.
Tracks agent position, patient states, time, and spatial rewards.
"""
from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Tuple


POSITION_XYZ = {
    "bed_1":          (2.0, 0.0, 2.0),
    "bed_2":          (2.0, 0.0, 6.0),
    "bed_3":          (10.0, 0.0, 2.0),
    "bed_4":          (10.0, 0.0, 6.0),
    "nurses_station": (6.0, 0.0, 4.0),
    "equipment_cart": (6.0, 0.0, 1.0),
    "exit":           (6.0, 0.0, 8.0),
}

EXAMINATION_DISTANCE = 2.5   # metres — must be within this to examine
TIME_COST_PER_SECOND = 0.01  # reward cost per second — encourages speed
MAX_EPISODE_SECONDS = 120.0  # 2 minutes per episode
STEP_TIME_SECONDS = 4.0      # each step costs 4 seconds of ward time


class WardState:
    """
    Manages the spatial simulation of the 3D emergency department.
    Computes movement costs, proximity checks, and time pressure.
    """

    def __init__(self, n_patients: int = 4):
        self.n_patients = n_patients
        self.agent_position = "nurses_station"
        self.time_elapsed = 0.0
        self.patient_alert_levels: Dict[str, str] = {}

    def reset(self, patient_severity_map: Dict[str, str]) -> None:
        self.agent_position = "nurses_station"
        self.time_elapsed = 0.0
        self.patient_alert_levels = dict(patient_severity_map)

    def move_to(self, target: str) -> Tuple[float, str]:
        """
        Move agent to target. Returns (movement_reward, feedback).
        Time cost = distance / walking_speed * TIME_COST_PER_SECOND.
        """
        if target not in POSITION_XYZ:
            return -0.05, f"Invalid target: {target}"

        dist = self._distance(self.agent_position, target)
        walk_time = dist / 1.4   # 1.4 m/s average walking speed
        self.time_elapsed += walk_time + 1.0   # +1s decision overhead

        old_pos = self.agent_position
        self.agent_position = target

        # Proximity reward — moving toward critical patient = small positive
        reward_delta = 0.0
        for bed_id, alert in self.patient_alert_levels.items():
            if target == bed_id and alert == "critical":
                reward_delta += 0.08
                break

        # Time cost
        time_cost = walk_time * TIME_COST_PER_SECOND
        total_reward = 0.5 - time_cost + reward_delta
        feedback = (
            f"Moved {old_pos} -> {target} "
            f"({dist:.1f}m, {walk_time:.1f}s). "
            f"Time elapsed: {self.time_elapsed:.1f}s"
        )
        return total_reward, feedback

    def examine(self, bed_id: str, exam_type: str) -> Tuple[float, str, bool]:
        """
        Examine a patient at a bed.
        Returns (reward, feedback, success).
        Must be within examination distance.
        """
        if bed_id not in POSITION_XYZ:
            return 0.2, f"Unknown bed: {bed_id}", False

        dist = self._distance(self.agent_position, bed_id)
        self.time_elapsed += STEP_TIME_SECONDS

        if dist > EXAMINATION_DISTANCE:
            return 0.2, (
                f"Too far from {bed_id} ({dist:.1f}m). "
                f"Move closer first."
            ), False

        alert = self.patient_alert_levels.get(bed_id, "unknown")

        # Examining a critical patient = more reward
        base = 0.58 if alert == "critical" else 0.52
        feedback = (
            f"Examined {bed_id} ({exam_type}). "
            f"Alert: {alert}. Time: {self.time_elapsed:.1f}s"
        )
        return base, feedback, True

    def intervene(self, bed_id: str, intervention: str) -> Tuple[float, str]:
        """
        Apply intervention. Must be at correct location with equipment.
        """
        dist = self._distance(self.agent_position, bed_id)
        self.time_elapsed += STEP_TIME_SECONDS * 1.5   # interventions take longer

        if dist > EXAMINATION_DISTANCE:
            return 0.2, f"Too far ({dist:.1f}m). Move to {bed_id} first."

        alert = self.patient_alert_levels.get(bed_id, "unknown")
        correct = {
            "critical": ["oxygen_mask", "iv_access", "defib_pads", "bag_valve_mask"],
            "warning": ["oxygen_mask", "iv_access", "cervical_collar"],
            "stable": ["iv_access"],
        }
        correct_for_level = correct.get(alert, [])

        if intervention in correct_for_level:
            reward = 0.72
            feedback = f"[CORRECT] {intervention} on {bed_id} (alert={alert}). +reward"
        else:
            reward = 0.35
            feedback = f"[LOW YIELD] {intervention} on {bed_id} (alert={alert})."

        return reward, feedback

    def get_nearby_beds(self) -> List[str]:
        """Return bed IDs within examination distance of current position."""
        nearby = []
        for bed_id in ["bed_1", "bed_2", "bed_3", "bed_4"]:
            if self._distance(self.agent_position, bed_id) <= EXAMINATION_DISTANCE:
                nearby.append(bed_id)
        return nearby

    def time_remaining(self) -> float:
        return max(0.0, MAX_EPISODE_SECONDS - self.time_elapsed)

    def is_timed_out(self) -> bool:
        return self.time_elapsed >= MAX_EPISODE_SECONDS

    def _distance(self, pos_a: str, pos_b: str) -> float:
        xyz_a = POSITION_XYZ.get(pos_a, (6.0, 0.0, 4.0))
        xyz_b = POSITION_XYZ.get(pos_b, (6.0, 0.0, 4.0))
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz_a, xyz_b)))
