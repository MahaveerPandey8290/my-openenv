"""
HTTP client for ClinicalTriageEnv.
Used by inference.py and RL training loops.

Self-contained implementation; does NOT depend on openenv.core.http_env_client
so it works regardless of the installed openenv-core version.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from .models import TriageAction, PatientObservation, TriageState


@dataclass
class StepResult:
    observation: PatientObservation
    raw: dict


class ClinicalTriageEnvClient:
    """
    Typed HTTP client for ClinicalTriageEnv.
    Connects to a running ClinicalTriageEnv server (local or HF Space).

    Usage:
        with ClinicalTriageEnvClient(base_url="http://localhost:7860") as env:
            result = env.reset(task_name="differential_diagnosis")
            obs = result.observation
            result = env.step(TriageAction(
                triage_level="urgent",
                suspected_condition="appendicitis",
                recommended_tests=["FBC", "ultrasound"],
                reasoning="RLQ pain, fever, rebound tenderness."
            ))
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ── context manager support ──────────────────────────────────────────────
    def __enter__(self) -> "ClinicalTriageEnvClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ── environment API ──────────────────────────────────────────────────────
    def reset(self, task_name: Optional[str] = None) -> StepResult:
        """POST /reset and return initial observation."""
        # ResetRequest allows additionalProperties — task_name goes in top-level extras
        payload: dict = {}
        if task_name:
            payload["task_name"] = task_name
        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = PatientObservation(**data.get("observation", data))
        return StepResult(observation=obs, raw=data)

    def step(self, action: TriageAction) -> StepResult:
        """POST /step with a TriageAction wrapped in StepRequest format.

        The OpenEnv server expects: {"action": {...action fields...}}
        NOT a flat action dict.
        """
        payload = {"action": action.model_dump()}
        resp = self._client.post("/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = PatientObservation(**data.get("observation", data))
        return StepResult(observation=obs, raw=data)

    def health(self) -> dict:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def get_tasks(self) -> dict:
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        return resp.json()
