# AMD Hackathon Track 1 Agent

A robust, production-ready, and highly token-optimized AI agent built for the AMD Hackathon Track 1. This agent is designed to execute LLM-powered tasks efficiently using LangGraph, minimize Fireworks API token usage via aggressive compression, safely handle fatal errors, and execute flawlessly within a grading harness environment (10GB limit, 4GB RAM, 2vCPU).

## Architecture

- **State Management & Orchestration:** Uses **LangGraph** to maintain agent state and route tasks through classification, local inference, prompt compression, and remote API calls.
- **Server/Lifespan:** Powered by **FastAPI** and **Uvicorn**, utilizing the ASGI lifespan to run background tasks while maintaining a responsive `/health` endpoint for the grading harness.
- **Hybrid LLM Execution:** 
  - **Local Tier (Zero Cost):** Easy tasks (factual, sentiment, summarization, NER) are routed to a local **Qwen2.5-0.5B-Instruct** model via **HuggingFace Transformers** running purely on CPU (`torch`). The model weights are baked into the Docker image, avoiding network latency and ensuring <60s startup.
  - **Remote Tier (Fireworks):** Hard tasks and code generation are routed to the remote Fireworks API.
- **Aggressive Token Optimization (Compressor):** Before hitting the Fireworks API, prompts pass through 7 layers of compression (including boilerplate stripping, generic stop-word removal, TF-IDF sentence pruning, and a **Free Local LLM Compression** pass) to strictly minimize scored tokens.
- **Crash Proof:** Bulletproof error handling that catches fatal exceptions (like missing input files), writes structured fallbacks to `results.json`, and elegantly terminates the Uvicorn process with `os.kill(SIGTERM)` to guarantee a strict `0` exit code.

## Building the Image

Because the grading harness strictly requires a `linux/amd64` architecture, the image MUST be built using Docker Buildx:

```bash
docker buildx build --platform linux/amd64 -t ghoshsoham71/amd-track1:v14 --push .
```

*Note: The HuggingFace model (`Qwen2.5-0.5B-Instruct`) and PyTorch CPU wheels are downloaded during the build step, resulting in a ~2.5GB image. If you run into severe caching issues where source code changes aren't being picked up, append `--no-cache` to the command.*

## Testing Locally

You can test the agent locally mimicking the exact behavior of the grading harness. 

1. **Setup Folders:** The project already comes with `input` and `output` folders.
2. **Mock Data:** Create a file named `input/tasks.json` with a JSON array of tasks (e.g. `[{"task_id": "test1", "prompt": "Say hello!"}]`).
3. **Run using Docker Compose:**

```bash
docker compose up --build
```
Alternatively, to run the container directly:

```bash
# Git Bash Users: Use $(pwd) instead of ~/ to ensure correct path translation
docker run --rm \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  -e FIREWORKS_API_KEY="your-api-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct" \
  -p 8080:8080 \
  ghoshsoham71/amd-track1:v14
```

The container will launch, execute the graph, output to `output/results.json`, and cleanly shut down.

## Directory Structure

- `/src`: Core application logic (`main.py`, `graph.py`, `router.py`, `compressor.py`, `local_model.py`, `prompts.py`, `token_counter.py`).
- `entrypoint.sh`: Rock-solid startup script that scrubs Windows CRLF line-endings, forces `chmod 777` on output volumes, and safely hands execution to Uvicorn.
- `pyproject.toml`: Modern dependency management using `hatchling`.
