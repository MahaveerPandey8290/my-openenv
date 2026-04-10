"""
OpenEnv Rubric reward class.
Wraps the shaped reward into the OpenEnv Rubric interface.
Used by GRPOTrainer when connecting via rollout_func.
"""
from __future__ import annotations
from typing import Any, Dict


class ClinicalRubric:
    """
    Rubric reward — called by OpenEnv to compute scalar reward
    from episode trajectory. Compatible with TRL GRPOTrainer.
    """

    def __call__(
        self,
        trajectory: list[Dict[str, Any]],
        ground_truth: Dict[str, Any],
    ) -> float:
        """
        Compute total normalised reward from a full episode trajectory.
        Each step in trajectory has: action, observation, reward.
        Returns scalar in (0.01, 0.99).
        """
        if not trajectory:
            return 0.5

        rewards = [step.get("reward", 0.5) for step in trajectory]
        raw = sum(rewards) / len(rewards)
        return round(min(max(float(raw), 0.01), 0.99), 2)

    def format_feedback(self, trajectory: list[Dict[str, Any]]) -> str:
        """Human-readable summary of the episode for debugging."""
        lines = []
        for i, step in enumerate(trajectory):
            lines.append(
                f"Step {i+1}: action={step.get('action_type','?')} "
                f"reward={step.get('reward', 0):.2f} "
                f"feedback={step.get('feedback','')}"
            )
        return "\n".join(lines)
