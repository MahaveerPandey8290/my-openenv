"""FastAPI server for ClinicalTriageEnv 3D."""
from __future__ import annotations
import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from ..models import (
    MoveToAction, ExaminePatientAction, OrderTestAction,
    InterventionAction, SubmitTriageAction,
)
from .environment import ClinicalTriageEnv3D

TASK_NAME = os.getenv("CLINICAL_TRIAGE_TASK", "ward_prioritisation")
RENDER_W = int(os.getenv("RENDER_WIDTH", "84"))
RENDER_H = int(os.getenv("RENDER_HEIGHT", "84"))

env = ClinicalTriageEnv3D(task_name=TASK_NAME, render_width=RENDER_W, render_height=RENDER_H)

app = FastAPI(
    title="ClinicalTriageEnv 3D",
    description="3D embodied medical agent RL environment with visual observations.",
    version="4.0.0",
)

ACTION_MAP = {
    "move_to": MoveToAction,
    "examine_patient": ExaminePatientAction,
    "order_test": OrderTestAction,
    "intervene": InterventionAction,
    "submit_triage": SubmitTriageAction,
}


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0", "env": "clinical_triage_3d"}


@app.post("/reset")
def reset(body: Dict[str, Any] = {}):
    obs = env.reset(
        task_name=(body or {}).get("task_name"),
        seed=(body or {}).get("seed"),
    )
    d = obs.model_dump()
    d["has_image"] = True
    return {"observation": d, "reward": obs.reward, "done": obs.done}


@app.post("/step")
def step(body: Dict[str, Any]):
    raw = body.get("action", body)
    atype = raw.get("action_type", "submit_triage")
    cls = ACTION_MAP.get(atype)
    if not cls:
        raise HTTPException(status_code=422, detail=f"Unknown action_type: {atype}")
    try:
        action = cls(**raw)
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
    from .tasks import TASK_REGISTRY_3D
    return {
        n: {"difficulty": s.difficulty, "description": s.description,
            "n_patients": s.n_patients, "max_time_seconds": s.max_time_seconds}
        for n, s in TASK_REGISTRY_3D.items()
    }
