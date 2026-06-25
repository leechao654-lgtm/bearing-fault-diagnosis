# =============================================================
# Bearing Fault Diagnosis — Inference Service (local / edge deployment)
# =============================================================
# Serves the trained ML model as a REST API via FastAPI.
# Build context is the repo root (this file COPYs utils/ and mlruns/
# from there), so always build with -f pointing at this Dockerfile.
#
# Two local build modes:
#
#   Self-contained edge image (model baked in):
#     Run the training notebook first to populate mlruns/, then
#     COPY mlruns/ below embeds the model directly in the image —
#     no runtime dependency on the host filesystem.
#     docker build -f deployment/Dockerfile -t bearing-fault-api .
#
#   Local dev (model mounted at runtime):
#     docker compose -f deployment/docker-compose.yml up   ← volume mount
#     docker run -p 8000:8000 \
#       -v "$(pwd)/mlruns:/app/mlruns:ro" \
#       bearing-fault-api
# =============================================================

FROM python:3.11-slim

# --- System dependencies ---
# gcc is required to compile some Python C extensions (e.g. scipy wheels fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
# Copy requirements first to leverage Docker layer caching.
# Re-installing packages only happens when this file changes.
COPY deployment/requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# --- Application code ---
# Only copy the utils/ modules needed for inference; notebooks and
# training artefacts stay on the host.
COPY utils/ ./utils/

# --- Model artefacts ---
# Run BearingFault_Training.ipynb on the host first to populate mlruns/.
# This COPY embeds the trained model in the image for a self-contained
# edge-deployable artefact (fast cold starts, no host dependency at runtime).
# For local dev without baking the model in, use the docker-compose.yml
# volume mount instead.
COPY mlruns/ ./mlruns/

# --- Runtime ---
EXPOSE 8000

CMD ["uvicorn", "utils.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
