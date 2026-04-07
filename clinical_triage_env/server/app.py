"""
FastAPI server for ClinicalTriageEnv.
Uses OpenEnv's create_fastapi_app helper.
Exposes POST /reset, POST /step, GET /state, GET /health
"""
import os
from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app, Environment as BaseEnv

from ..models import TriageAction, PatientObservation
from .environment import ClinicalTriageEnv

TASK_NAME = os.getenv("CLINICAL_TRIAGE_TASK", "vital_signs_triage")

# Create a singleton instance
env_instance = ClinicalTriageEnv(task_name=TASK_NAME)

# Pass a factory function that returns the singleton
app: FastAPI = create_fastapi_app(
    lambda: env_instance, 
    TriageAction, 
    PatientObservation
)


@app.get("/")
def root():
    """Root endpoint — confirms the server is alive."""
    return {
        "name": "ClinicalTriageEnv",
        "version": "0.1.0",
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
