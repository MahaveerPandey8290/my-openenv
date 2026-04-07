"""
HTTP client for ClinicalTriageEnv.
Used by inference.py and RL training loops.
"""
from __future__ import annotations
from typing import Optional
import httpx
from .models import TriageAction, PatientObservation, TriageState


class ClinicalTriageEnvClient:
    """
    Typed HTTP client for ClinicalTriageEnv.
    Connects to a running ClinicalTriageEnv server (local or HF Space).

    Usage:
        client = ClinicalTriageEnvClient(base_url="http://localhost:7860")
        obs = client.reset(task_name="differential_diagnosis")
        result = client.step(TriageAction(
            triage_level="urgent",
            suspected_condition="appendicitis",
            recommended_tests=["FBC", "ultrasound"],
            reasoning="RLQ pain, fever, rebound tenderness."
        ))
        client.close()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def reset(self, task_name: Optional[str] = None) -> PatientObservation:
        payload = {}
        if task_name:
            payload["task_name"] = task_name
        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return PatientObservation(**data.get("observation", data))

    def step(self, action: TriageAction) -> PatientObservation:
        resp = self._client.post("/step", json={"action": action.model_dump()})
        resp.raise_for_status()
        data = resp.json()
        return PatientObservation(**data.get("observation", data))

    def health(self) -> dict:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
