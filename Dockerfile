# =============================================================================
# AMD Hackathon Track 1 - Dockerfile
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): Install build deps, compile llama-cpp-python, download models
#   Stage 2 (runtime): Slim image with only runtime deps
#
# Target: linux/amd64 (required by grading harness)
# Budget: 4 GB RAM, 2 vCPU
# Run:    uvicorn src.app:app (FastAPI + lifespan task processing)
# =============================================================================

# -- Stage 1: Builder ----------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim AS builder

# Build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# -- Download GGUF models -------------------------------------------------------
# We do this FIRST so changing pyproject.toml doesn't trigger a 1.8GB re-download!
RUN pip install --no-cache-dir huggingface_hub
RUN mkdir -p /models
RUN python -c "import huggingface_hub; huggingface_hub.hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='/models'); print('Local model downloaded')"

# Copy project definition (pyproject.toml is the single source of truth)
COPY pyproject.toml .
# Minimal src stub so hatchling can resolve the package during install
COPY src/__init__.py ./src/__init__.py

# Install all project deps from pyproject.toml (except llama-cpp-python)
# llama-cpp-python needs special CMAKE_ARGS so we install it separately below
RUN pip install --no-cache-dir --prefix=/install \
    "hatchling" \
    && pip install --no-cache-dir --prefix=/install \
    ".[dev]" \
    --no-deps 2>/dev/null || true

# Install all runtime deps from pyproject.toml (excluding llama-cpp-python optional)
RUN pip install --no-cache-dir --prefix=/install .

# Install llama-cpp-python separately: CPU-only, no BLAS
# CMAKE_ARGS control: no native CPU extensions (portable across x86_64)
RUN CMAKE_ARGS="-DLLAMA_NATIVE=OFF -DLLAMA_BLAS=OFF" \
    pip install --no-cache-dir --prefix=/install "llama-cpp-python>=0.2.90"


# -- Stage 2: Runtime ----------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim AS runtime

# Runtime system deps: OpenMP (libgomp1) is required for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy models
COPY --from=builder /models /app/models

# Copy source code and project definition
COPY src/ ./src/
COPY pyproject.toml .
COPY entrypoint.sh .
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Install the package in editable-equivalent mode (no build, just src on path)
RUN pip install --no-cache-dir --no-deps -e .

# Create required directories and ensure they are world-writable
RUN mkdir -p /input /output && chmod 777 /input /output

# -- Environment defaults -------------------------------------------------------
# Harness overrides FIREWORKS_* and ALLOWED_MODELS at evaluation time.
ENV MODEL_DIR=/app/models \
    LOCAL_MODEL_FILENAME=qwen2.5-3b-instruct-q4_k_m.gguf \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    MAX_WORKERS=2 \
    MAX_RUNTIME_SECONDS=570 \
    LOCAL_N_CTX=2048 \
    LOCAL_N_THREADS=2 \
    LOCAL_MAX_TOKENS=512 \
    VALIDATOR_THRESHOLD=0.65 \
    LOG_LEVEL=INFO \
    HF_HOME=/tmp/hf_cache \
    TIKTOKEN_CACHE_DIR=/app/tiktoken_cache

# Pre-download tiktoken cache so it doesn't block or crash on container boot (60s rule)
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# -- Expose health port (internal only, not required by harness) ---------------
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
