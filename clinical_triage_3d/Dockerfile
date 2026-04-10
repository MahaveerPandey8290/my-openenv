FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgles2-mesa \
    libegl1-mesa \
    mesa-utils \
    xvfb \
    freeglut3-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

ENV SDL_VIDEODRIVER=offscreen
ENV SDL_AUDIODRIVER=dummy
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=softpipe

WORKDIR /app

COPY clinical_triage_3d/server/requirements.txt /tmp/req3d.txt
RUN pip install --no-cache-dir -r /tmp/req3d.txt

COPY . /app/
RUN pip install --no-cache-dir -e .

HEALTHCHECK --interval=30s --timeout=15s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "clinical_triage_3d.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
