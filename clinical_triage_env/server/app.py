"""
FastAPI server for ClinicalTriageEnv.
Uses OpenEnv's create_fastapi_app helper.
Exposes POST /reset, POST /step, GET /state, GET /health
"""
import os
from typing import Any

from fastapi import Body, FastAPI
from openenv.core.env_server import create_fastapi_app

from ..models import TriageAction, PatientObservation, TriageState
from .environment import ClinicalTriageEnv

TASK_NAME = os.getenv("CLINICAL_TRIAGE_TASK", "vital_signs_triage")

# Create a singleton instance
env_instance = ClinicalTriageEnv(task_name=TASK_NAME)

# Build default OpenEnv app first
app: FastAPI = create_fastapi_app(lambda: env_instance, TriageAction, PatientObservation)

# Remove default /state and /reset routes so custom typed endpoints are authoritative.
app.router.routes = [
    route
    for route in app.router.routes
    if not (
        getattr(route, "path", None) in {"/state", "/reset"}
        and "GET" in getattr(route, "methods", set())
        or getattr(route, "path", None) in {"/state", "/reset"}
        and "POST" in getattr(route, "methods", set())
    )
]


@app.post("/reset")
def reset(payload: dict[str, Any] | None = Body(default=None)):
    task_name = payload.get("task_name") if payload else None
    obs = env_instance.reset(task_name=task_name)
    obs_dict = obs.model_dump()
    return {
        "observation": obs_dict,
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state", response_model=TriageState)
def get_state():
    return env_instance.state


@app.get("/")
def root():
    """Root endpoint — confirms the server is alive."""
    return {
        "name": "ClinicalTriageEnv",
        "version": "0.1.5",
        "status": "running",
        "description": "OpenEnv RL environment for clinical triage",
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset",
            "step": "POST /step",
            "tasks": "GET /tasks",
            "docs": "GET /docs",
        },
    }


@app.get("/tasks")
def list_tasks():
    """List all available tasks and their metadata."""
    from .tasks import TASK_REGISTRY
    return {
        name: {
            "difficulty": spec.difficulty,
            "description": spec.description,
            "max_steps": spec.max_steps,
        }
        for name, spec in TASK_REGISTRY.items()
    }


def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("clinical_triage_env.server.app:app", host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
