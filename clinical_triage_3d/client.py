"""HTTP client for ClinicalTriageEnv 3D."""
from __future__ import annotations
from typing import Any, Dict, Optional

from openenv import GenericEnvClient
from openenv.core.sync_client import SyncEnvClient

from .models import (
    MoveToAction, ExaminePatientAction, OrderTestAction,
    InterventionAction, SubmitTriageAction,
    VisualObservation, TriageState3D
)


class ClinicalTriageEnv3DClient:
    """
    Typed synchronous HTTP client for ClinicalTriageEnv 3D.
    Used by 3D inference and training scripts.
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
        """Reset environment and return visual observation."""
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


class _WrappedResult3D:
    """Thin wrapper for 3D observations."""
    def __init__(self, raw):
        self._raw = raw
        obs_dict = {}
        if isinstance(raw, dict):
            obs_dict = raw.get("observation", raw)
        elif hasattr(raw, "observation"):
            obs_dict = raw.observation if isinstance(raw.observation, dict) else {}
        self.observation = _make_obs_3d(obs_dict)
        self.reward = obs_dict.get("reward", 0.5) if isinstance(obs_dict, dict) else 0.5
        self.done = obs_dict.get("done", False) if isinstance(obs_dict, dict) else False


def _make_obs_3d(obs_dict: dict) -> VisualObservation:
    """Coerce raw dict to VisualObservation."""
    try:
        return VisualObservation(**obs_dict)
    except Exception:
        # Fallback
        return VisualObservation(
            image_base64=obs_dict.get("image_base64", ""),
            agent_location=obs_dict.get("agent_location", "unknown"),
            reward=float(obs_dict.get("reward", 0.5)),
            done=bool(obs_dict.get("done", False)),
        )


def _wrap_result(result) -> _WrappedResult3D:
    if isinstance(result, dict):
        return _WrappedResult3D(result)
    obs_dict = {}
    if hasattr(result, "observation"):
        obs = result.observation
        obs_dict = obs if isinstance(obs, dict) else {}
    wrapped = _WrappedResult3D.__new__(_WrappedResult3D)
    wrapped._raw = result
    wrapped.observation = _make_obs_3d(obs_dict)
    wrapped.reward = getattr(result, "reward", obs_dict.get("reward", 0.5))
    wrapped.done = getattr(result, "done", obs_dict.get("done", False))
    return wrapped
