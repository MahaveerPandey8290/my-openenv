"""Root entry point — re-exports the FastAPI app from the clinical_triage_env package."""
from clinical_triage_env.server.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
