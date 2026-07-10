# AMD Hackathon Track 1 Agent

A robust, production-ready AI agent built for the AMD Hackathon Track 1. This agent is designed to execute LLM-powered tasks efficiently using LangGraph, safely handle fatal errors, and execute flawlessly within a grading harness environment.

##  Architecture

- **State Management:** Uses **LangGraph** backed by an ephemeral local **Redis** instance to maintain agent state and reasoning execution safely.
- **Server/Lifespan:** Powered by **FastAPI** and **Uvicorn**, utilizing the ASGI lifespan to run background tasks while maintaining a responsive `/health` endpoint for the grading harness.
- **LLM Support:** Seamlessly handles both remote execution via the **Fireworks API** and local GGUF execution via **llama-cpp-python** (compiled specifically without AVX-512 flags to ensure compatibility across all AMD VMs).
- **Crash Proof:** Bulletproof error handling that catches fatal exceptions (like missing input files), writes an empty `results.json` fallback, and elegantly terminates the Uvicorn process with `os.kill(SIGTERM)` to guarantee a strict `0` exit code.

##  Building the Image

Because the grading harness strictly requires a `linux/amd64` architecture, the image MUST be built using Docker Buildx:

```bash
docker buildx build --platform linux/amd64 -t ghoshsoham71/amd-track1:latest --push .
```

*Note: If you run into severe caching issues where source code changes aren't being picked up, append `--no-cache` to the command.*

##  Testing Locally

You can test the agent locally mimicking the exact behavior of the grading harness. 

1. **Setup Folders:** Create an `input` and `output` folder in your project.
2. **Mock Data:** Create a file named `input/tasks.json` with a JSON array of tasks (e.g. `[{"task_id": "test1", "prompt": "Say hello!"}]`).
3. **Run the Container:**

```bash
# Git Bash Users: Use $(pwd) instead of ~/ to ensure correct path translation
docker run --rm \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  -e FIREWORKS_API_KEY="your-api-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct" \
  -p 8080:8080 \
  ghoshsoham71/amd-track1:latest
```

The container will launch, execute the graph, output to `output/results.json`, and cleanly shut down.

##  Directory Structure

- `/src`: Core application logic (`app.py`, `main.py`, `graph.py`, `router.py`, `token_counter.py`).
- `entrypoint.sh`: Rock-solid startup script that scrubs Windows CRLF line-endings, forces `chmod 777` on output volumes, and safely hands execution to Uvicorn.
- `pyproject.toml`: Modern dependency management using `hatchling`.

