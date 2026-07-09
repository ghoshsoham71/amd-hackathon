"""
FastAPI application — AMD Hackathon Track 1 Agent.

Why FastAPI + uvicorn?
  - Satisfies the "container ready within 60 seconds" requirement with an
    immediate /health endpoint (responds before task processing completes).
  - Enables async concurrent task processing via asyncio.
  - Provides structured /status and /metrics endpoints for local debugging.
  - Uvicorn's event loop is reused by all async components.

Startup flow:
  1. Uvicorn starts → FastAPI lifespan begins
  2. Redis server spawned as subprocess
  3. tasks.json loaded
  4. LangGraph pipeline runs tasks in background thread pool
  5. results.json written
  6. App signals uvicorn to shutdown → exit 0
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("app")

# ── Response models (Pydantic) — powers Swagger typed schemas ─────────────────

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Always 'ok' while process is alive")
    agent_status: str = Field(..., description="starting | processing | done | error")

    model_config = {"json_schema_extra": {"example": {"status": "ok", "agent_status": "processing"}}}


class StatusResponse(BaseModel):
    status: str        = Field(..., description="starting | processing | done | error")
    tasks_total: int   = Field(..., description="Total tasks loaded from tasks.json")
    tasks_done: int    = Field(..., description="Tasks completed so far")
    tasks_pending: int = Field(..., description="Tasks not yet completed")
    elapsed_s: float   = Field(..., description="Seconds since agent started")
    error: Optional[str] = Field(None, description="Error message if status=error")

    model_config = {"json_schema_extra": {"example": {
        "status": "processing",
        "tasks_total": 8,
        "tasks_done": 5,
        "tasks_pending": 3,
        "elapsed_s": 12.4,
        "error": None,
    }}}


class MetricsResponse(BaseModel):
    tasks: int                  = Field(0, description="Total tasks processed")
    fireworks_calls: int        = Field(0, description="Number of Fireworks API calls made")
    fireworks_total_tokens: int = Field(0, description="Total scored tokens (input + output)")
    fireworks_input_tokens: int = Field(0, description="Fireworks input tokens")
    fireworks_output_tokens: int= Field(0, description="Fireworks output tokens")
    local_calls: int            = Field(0, description="Local model inference calls (0 token cost)")
    local_tokens: int           = Field(0, description="Local tokens used (not scored)")

    model_config = {"json_schema_extra": {"example": {
        "tasks": 8,
        "fireworks_calls": 2,
        "fireworks_total_tokens": 312,
        "fireworks_input_tokens": 224,
        "fireworks_output_tokens": 88,
        "local_calls": 6,
        "local_tokens": 4821,
    }}}

# ── Shared state ──────────────────────────────────────────────────────────────
_state: dict = {
    "started_at":   None,
    "status":       "starting",   # starting | processing | done | error
    "tasks_total":  0,
    "tasks_done":   0,
    "error":        None,
}


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ASGI lifespan: runs startup logic, yields for request serving,
    then handles shutdown.
    """
    _state["started_at"] = time.time()
    logger.info("Agent lifespan starting")

    # Import here to avoid circular imports at module level
    from src.main import (
        load_tasks,
        run_all_tasks,
        validate_env,
        write_results,
    )

    # 1. Load env vars (non-fatal warnings only)
    validate_env()

    # 3. Load tasks
    from src.main import INPUT_PATH, OUTPUT_PATH
    tasks = load_tasks(INPUT_PATH)
    _state["tasks_total"] = len(tasks)
    _state["status"] = "processing"
    logger.info("Loaded %d tasks — beginning processing", len(tasks))

    # We need to run the pipeline in the background so FastAPI can actually start
    # and serve the /status and /docs endpoints immediately.
    loop = asyncio.get_event_loop()
    
    async def background_worker():
        try:
            results = await loop.run_in_executor(None, run_all_tasks, tasks)
            _state["tasks_done"] = len(results)

            write_results(results, OUTPUT_PATH)
            
            from src.token_counter import GLOBAL_TRACKER
            GLOBAL_TRACKER.log_summary()

            _state["status"] = "done"
            logger.info("All tasks complete. results.json written.")

        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
            _state["status"] = "error"
            _state["error"] = str(e)
            # ── Always write results.json even on error ──
            # Missing file = score zero AND may fail harness validation.
            # An empty array at least produces valid JSON and a clean exit.
            try:
                write_results([], OUTPUT_PATH)
                logger.warning("Wrote empty results.json after pipeline error")
            except Exception as write_err:
                logger.error("Failed to write fallback results.json: %s", write_err)
        finally:
            # ── Critical: shut down uvicorn so the container exits cleanly ──
            # Without this the server runs forever and the harness times out.
            logger.info("Sending SIGTERM to self to trigger clean shutdown...")
            signal.raise_signal(signal.SIGTERM)

    # Start the worker task in the background
    worker_task = asyncio.create_task(background_worker())

    # Yield control back to FastAPI so the HTTP server can start!
    yield

    # ── SHUTDOWN PHASE ──
    logger.info("Lifespan complete — shutting down")
    
    # Wait for the background task if it hasn't finished (with a short timeout)
    if not worker_task.done():
        logger.info("Waiting for background task to finish...")
        try:
            await asyncio.wait_for(worker_task, timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.warning("Background task did not finish in time — cancelling")
            worker_task.cancel()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AMD Track 1 — General-Purpose AI Agent",
    description=(
        "Token-efficient AI agent for AMD Hackathon Track 1.\n\n"
        "**Pipeline**: LangGraph waterfall → Local GGUF (Qwen2.5-1.5B → 3B) → "
        "Fireworks API (smallest permitted model).\n\n"
        "**Scoring**: accuracy gate first, then ranked by total Fireworks tokens used."
    ),
    version="1.0.0",
    contact={"name": "Track 1 Agent"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Always returns 200 while the process is alive. Responds immediately — before task processing completes.",
    tags=["ops"],
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", agent_status=_state["status"])


@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Processing progress",
    description="Returns task completion count, elapsed time, and any error details.",
    tags=["ops"],
)
async def status() -> StatusResponse:
    elapsed = time.time() - (_state["started_at"] or time.time())
    done  = _state["tasks_done"]
    total = _state["tasks_total"]
    return StatusResponse(
        status=_state["status"],
        tasks_total=total,
        tasks_done=done,
        tasks_pending=total - done,
        elapsed_s=round(elapsed, 1),
        error=_state["error"],
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Token usage metrics",
    description=(
        "Returns Fireworks API token counts (scored) and local model token counts (not scored). "
        "Use this to optimize routing during development."
    ),
    tags=["ops"],
)
async def metrics() -> MetricsResponse:
    try:
        from src.token_counter import GLOBAL_TRACKER
        data = GLOBAL_TRACKER.summary()
        return MetricsResponse(**data)
    except Exception as e:
        return MetricsResponse()
