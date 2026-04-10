"""HTTP client for ClinicalTriageEnv v3.

Uses openenv's GenericEnvClient (async) wrapped in SyncEnvClient for synchronous use.
The GenericEnvClient works with raw dicts — we serialize/deserialize Pydantic models.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from openenv import GenericEnvClient
from openenv.core.sync_client import SyncEnvClient

from .models import SubmitTriageAction, OrderTestAction, PatientObservation, TriageState


class ClinicalTriageEnvClient:
    """
    Typed synchronous HTTP client for ClinicalTriageEnv v3.
    Used by inference.py and train_grpo.py.

    Example:
        with ClinicalTriageEnvClient(base_url="http://localhost:7860") as env:
            result = env.reset(task_name="differential_diagnosis")
            obs = result.observation
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self._async_client = GenericEnvClient(base_url=base_url)
        self._client: Optional[SyncEnvClient] = None

    def __enter__(self):
        self._client = self._async_client.sync()
        self._client.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()

    def reset(self, task_name: Optional[str] = None, seed: Optional[int] = None):
        """Reset environment and return initial observation."""
        kwargs: Dict[str, Any] = {}
        if task_name:
            kwargs["task_name"] = task_name
        if seed is not None:
            kwargs["seed"] = seed
        result = self._client.reset(**kwargs)
        return _wrap_result(result)

    def step(self, action):
        """Execute action and return step result."""
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        else:
            action_dict = dict(action)
        result = self._client.step(action_dict)
        return _wrap_result(result)


class _WrappedResult:
    """Thin wrapper to expose .observation with typed fields."""
    def __init__(self, raw):
        self._raw = raw
        obs_dict = {}
        if isinstance(raw, dict):
            obs_dict = raw.get("observation", raw)
        elif hasattr(raw, "observation"):
            obs_dict = raw.observation if isinstance(raw.observation, dict) else {}
        self.observation = _make_obs(obs_dict)
        self.reward = obs_dict.get("reward", 0.5) if isinstance(obs_dict, dict) else 0.5
        self.done = obs_dict.get("done", False) if isinstance(obs_dict, dict) else False


def _make_obs(obs_dict: dict) -> PatientObservation:
    """Coerce raw dict to PatientObservation."""
    try:
        return PatientObservation(**obs_dict)
    except Exception:
        # Fallback with minimum required fields
        return PatientObservation(
            patient_id=str(obs_dict.get("patient_id", "unknown")),
            chief_complaint=str(obs_dict.get("chief_complaint", "")),
            vitals=obs_dict.get("vitals", {}),
            visible_symptoms=obs_dict.get("visible_symptoms", []),
            reward=float(obs_dict.get("reward", 0.5)),
            done=bool(obs_dict.get("done", False)),
        )


def _wrap_result(result) -> _WrappedResult:
    """Wrap raw client result."""
    if isinstance(result, dict):
        return _WrappedResult(result)
    # result is a StepResult object
    obs_dict = {}
    if hasattr(result, "observation"):
        obs = result.observation
        obs_dict = obs if isinstance(obs, dict) else {}
    wrapped = _WrappedResult.__new__(_WrappedResult)
    wrapped._raw = result
    wrapped.observation = _make_obs(obs_dict)
    if hasattr(result, "reward"):
        wrapped.reward = result.reward
    else:
        wrapped.reward = obs_dict.get("reward", 0.5)
    if hasattr(result, "done"):
        wrapped.done = result.done
    else:
        wrapped.done = obs_dict.get("done", False)
    return wrapped
