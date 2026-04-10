"""FastAPI server for ClinicalTriageEnv v3."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from ..models import OrderTestAction, SubmitTriageAction, PatientObservation, TriageState
from .environment import ClinicalTriageEnv

TASK_NAME = os.getenv("CLINICAL_TRIAGE_TASK", "vital_signs_triage")
env = ClinicalTriageEnv(task_name=TASK_NAME)

app = FastAPI(
    title="ClinicalTriageEnv",
    description="True RL environment for clinical triage with tool use and GRPO training.",
    version="3.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0", "env": "clinical_triage_env"}


@app.post("/reset")
def reset(body: Dict[str, Any] = {}):
    task_name = (body or {}).get("task_name", None)
    seed = (body or {}).get("seed", None)
    obs = env.reset(task_name=task_name, seed=seed)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.post("/step")
def step(body: Dict[str, Any]):
    raw = body.get("action", body)
    action_type = raw.get("action_type", "submit_triage")
    try:
        if action_type == "order_test":
            action = OrderTestAction(**raw)
        else:
            action = SubmitTriageAction(**raw)
    except (ValidationError, Exception) as e:
        raise HTTPException(status_code=422, detail=str(e))
    obs = env.step(action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def get_state():
    try:
        return env.state.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    from .tasks import TASK_REGISTRY
    return {
        name: {
            "difficulty": spec.difficulty,
            "description": spec.description,
            "max_steps": spec.max_steps,
            "max_tests": spec.max_tests,
        }
        for name, spec in TASK_REGISTRY.items()
    }
