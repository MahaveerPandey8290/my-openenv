"""FastAPI server for ClinicalTriageEnv v3."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional

from openenv.core import create_fastapi_app

from ..models import OrderTestAction, SubmitTriageAction, PatientObservation, TriageState, ClinicalAction
from .environment import ClinicalTriageEnv

TASK_NAME = os.getenv("CLINICAL_TRIAGE_TASK", "vital_signs_triage")
env = ClinicalTriageEnv(task_name=TASK_NAME)

# Use openenv-core standard app creator
# This adds /metadata, /schema, /reset, /step, /state, /health endpoints automatically.
app = create_fastapi_app(
    env=lambda: env,
    action_cls=ClinicalAction,
    observation_cls=PatientObservation,
)

# Preserve original app metadata
app.title = "ClinicalTriageEnv"
app.description = "True RL environment for clinical triage with tool use and GRPO training."
app.version = "3.0.0"


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
