# ── ClinicalTriageEnv — OpenEnv Hackathon Submission ──────────────────────────
# Hugging Face Spaces uses port 7860. Must bind to 0.0.0.0:7860.

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY clinical_triage_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy full project
COPY . /app/

# Install the env package itself
RUN pip install --no-cache-dir .

# Health check — HF Spaces pings /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port used by HF Spaces
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "clinical_triage_env.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]
