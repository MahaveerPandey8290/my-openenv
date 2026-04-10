"""
Manual FastAPI implementation for ClinicalTriageEnv 3D.
Handles OpenEnv discovery (/metadata, /schema) and core RL loop for embodied agents.
"""
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import TypeAdapter

from .environment import ClinicalTriageEnv3D
from ..models import ClinicalAction3D, VisualObservation

app = FastAPI(title="ClinicalTriageEnv 3D Server")
env = ClinicalTriageEnv3D()

# ── Compliance Endpoints ──────────────────────────────────────────────────────

@app.get("/metadata")
async def get_metadata():
    """Discovery endpoint for OpenEnv / HF."""
    return {
        "name": "clinical_triage_3d",
        "version": "3.0.0",
        "description": "3D embodied medical agent environment with visual observations.",
        "author": "Mahaveer8290",
    }

@app.get("/schema")
async def get_schema():
    """Discovery endpoint for Action/Observation schemas."""
    try:
        action_adapter = TypeAdapter(ClinicalAction3D)
        obs_adapter = TypeAdapter(VisualObservation)
        return {
            "action": action_adapter.json_schema(),
            "observation": obs_adapter.json_schema(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Schema generation failed: {str(e)}"}
        )

@app.get("/tasks")
async def get_tasks():
    """List available tasks."""
    from .tasks import TASK_REGISTRY_3D
    return {"tasks": list(TASK_REGISTRY_3D.keys())}

# ── Core RL Endpoints ─────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: Request):
    """Start new episode."""
    body = await request.json()
    task_name = body.get("task_name")
    seed = body.get("seed")
    obs = env.reset(task_name=task_name, seed=seed)
    return obs.model_dump()

@app.post("/step")
async def step(request: Request):
    """Execute action."""
    body = await request.json()
    raw_action = body.get("action")
    if not raw_action:
        raise HTTPException(status_code=400, detail="Missing 'action' in request body")
    
    try:
        # Validate into Union type
        action = TypeAdapter(ClinicalAction3D).validate_python(raw_action)
        obs = env.step(action)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "env": "ClinicalTriageEnv 3D"}

@app.get("/state")
async def get_state():
    return env.state.model_dump()

# ── Landing Page ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>ClinicalTriageEnv 3D</title>
            <style>
                body { font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }
                .card { background: #1e293b; padding: 2rem; border-radius: 1rem; max-width: 600px; margin: auto; }
                h1 { color: #f472b6; }
                code { background: #334155; padding: 0.2rem 0.4rem; border-radius: 0.3rem; }
                .status { color: #4ade80; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🏥 ClinicalTriageEnv 3D</h1>
                <p>Status: <span class="status">RUNNING</span></p>
                <p>3D Embodied Medical Agent Environment is live.</p>
                <hr style="border: 0; border-top: 1px solid #334155; margin: 2rem 0;">
                <h3>Discovery</h3>
                <ul>
                    <li><code>/metadata</code> - Version & Description</li>
                    <li><code>/schema</code> - 3D Action/Visual Observation schemas</li>
                </ul>
            </div>
        </body>
    </html>
    """

def main():
    import uvicorn
    # 3D server usually runs on a different port if local, but on HF it's one Space per repo.
    # If the user has both in one repo, they might be using sub-paths or separate containers.
    # Assuming standard HF Space port 7860.
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
