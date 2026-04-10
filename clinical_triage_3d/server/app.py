"""FastAPI server for ClinicalTriageEnv 3D."""
from __future__ import annotations
import os
from typing import Any, Dict

from openenv.core import create_fastapi_app

from ..models import ClinicalAction3D, VisualObservation, TriageState3D
from .environment import ClinicalTriageEnv3D

TASK_NAME = os.getenv("CLINICAL_TRIAGE_TASK", "ward_prioritisation")
RENDER_W = int(os.getenv("RENDER_WIDTH", "84"))
RENDER_H = int(os.getenv("RENDER_HEIGHT", "84"))

env = ClinicalTriageEnv3D(task_name=TASK_NAME, render_width=RENDER_W, render_height=RENDER_H)

app = create_fastapi_app(
    env=lambda: env,
    action_cls=ClinicalAction3D,
    observation_cls=VisualObservation,
)

app.title = "ClinicalTriageEnv 3D"
app.description = "3D embodied medical agent RL environment with visual observations."
app.version = "4.0.0"


@app.get("/tasks")
def list_tasks():
    from .tasks import TASK_REGISTRY_3D
    return {
        n: {"difficulty": s.difficulty, "description": s.description,
            "n_patients": s.n_patients, "max_time_seconds": s.max_time_seconds}
        for n, s in TASK_REGISTRY_3D.items()
    }
