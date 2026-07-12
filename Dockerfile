# =============================================================================
# AMD Hackathon Track 1 - Dockerfile
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): Install Python deps, download HF model weights
#   Stage 2 (runtime): Slim image with only runtime deps + baked model
#
# Target:  linux/amd64  (required by grading harness)
# Budget:  4 GB RAM, 2 vCPU
# Model:   Qwen/Qwen2.5-0.5B-Instruct  (~1 GB on disk, ~1.3 GB RAM)
#          Fits safely within 4 GB — leaves ~2.7 GB for agent code + Fireworks.
#          Model is baked into image at /app/models/hf_model (no runtime download).
# Run:     uvicorn src.app:app (FastAPI + lifespan task processing)
# =============================================================================

# -- Stage 1: Builder ----------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim AS builder

# Minimal build tools (no cmake/gcc needed — pure Python deps only now)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    g++ \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# -- Download HuggingFace model weights into /models/hf_model -----------------
# Done FIRST so changes to pyproject.toml do NOT invalidate the model download cache!
# Qwen2.5-3B-Instruct-GGUF: ~1.7 GB on disk.
# We set HF_HOME to /models so huggingface_hub caches directly there.
ENV HF_HOME=/models/hf_cache
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='/models/hf_model', local_dir_use_symlinks=False, ignore_patterns=['*.msgpack', '*.h5', '*.bin']); print('Model downloaded successfully')"

# Copy project definition
COPY pyproject.toml .
COPY src/__init__.py ./src/__init__.py

# Install all runtime deps
RUN pip install --no-cache-dir --prefix=/install --ignore-installed .

# Pre-download tiktoken encoding cache into the install tree
ENV TIKTOKEN_CACHE_DIR=/install/tiktoken_cache
RUN PYTHONPATH=/install/lib/python3.11/site-packages python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"



# -- Stage 2: Runtime ----------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim AS runtime

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy baked model weights
COPY --from=builder /models /app/models

# Copy source code and project definition
COPY src/ ./src/
COPY pyproject.toml .
COPY entrypoint.sh .
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Install the package itself (editable-equivalent, no rebuild of deps)
RUN pip install --no-cache-dir --no-deps -e .

# Create required I/O directories (world-writable for harness compatibility)
RUN mkdir -p /input /output && chmod 777 /input /output

# -- Environment defaults ------------------------------------------------------
# Harness overrides FIREWORKS_* and ALLOWED_MODELS at evaluation time.
ENV MODEL_DIR=/app/models \
    LOCAL_HF_MODEL_PATH=/app/models/hf_model \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    MAX_WORKERS=2 \
    MAX_RUNTIME_SECONDS=570 \
    LOCAL_MAX_TOKENS=128 \
    LOCAL_MAX_INPUT=1024 \
    VALIDATOR_THRESHOLD=0.65 \
    LOG_LEVEL=INFO \
    HF_HOME=/app/models/hf_cache \
    TIKTOKEN_CACHE_DIR=/usr/local/tiktoken_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# Pre-warm: verify model loads and tiktoken is cached (catches bad builds early)
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('tiktoken OK'); import transformers; print('transformers OK'); import torch; print('torch OK')"

# -- Expose health port --------------------------------------------------------
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
