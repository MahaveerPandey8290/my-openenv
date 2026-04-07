"""
HTTP client for ClinicalTriageEnv.
Used by inference.py and RL training loops.
"""
from __future__ import annotations
from openenv.core.http_env_client import HTTPEnvClient
from .models import TriageAction, PatientObservation, TriageState


class ClinicalTriageEnvClient(HTTPEnvClient):
    """
    Typed client for ClinicalTriageEnv.
    Connects to a running ClinicalTriageEnv server (local or HF Space).

    Usage:
        with ClinicalTriageEnvClient(base_url="http://localhost:7860") as env:
            obs = env.reset(task_name="differential_diagnosis")
            result = env.step(TriageAction(
                triage_level="urgent",
                suspected_condition="appendicitis",
                recommended_tests=["FBC", "ultrasound"],
                reasoning="RLQ pain, fever, rebound tenderness."
            ))
    """

    action_type = TriageAction
    observation_type = PatientObservation
    state_type = TriageState
