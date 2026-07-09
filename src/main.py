"""
Main entry point — AMD Hackathon Track 1: General-Purpose AI Agent.

Startup sequence:
  1. Configure logging
  2. Validate environment variables
  3. Start Redis server (subprocess, non-blocking)
  4. Preload local models
  5. Read /input/tasks.json
  6. Run all tasks through the LangGraph pipeline (async, 4 workers)
  7. Write /output/results.json
  8. Log token summary and exit 0
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path(os.environ.get("INPUT_PATH",  "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))

# ── Parallelism ───────────────────────────────────────────────────────────────
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

# ── Global deadline (9.5 min — stay well under 10-min harness limit) ─────────
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", "570"))  # 9.5 min
_START_TIME: float = time.time()

def time_remaining() -> float:
    """Seconds remaining before we must stop processing."""
    return MAX_RUNTIME_SECONDS - (time.time() - _START_TIME)

def deadline_exceeded() -> bool:
    return time_remaining() <= 0


# ── Redis helpers ─────────────────────────────────────────────────────────────

def start_redis() -> "subprocess.Popen | None":
    """
    Start a Redis server as a subprocess if redis-server is available.
    Returns the process handle, or None if Redis is not installed.
    """
    try:
        proc = subprocess.Popen(
            ["redis-server", "--daemonize", "no", "--maxmemory", "128mb",
             "--maxmemory-policy", "allkeys-lru", "--save", ""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Redis started (pid=%d)", proc.pid)
        return proc
    except FileNotFoundError:
        logger.warning("redis-server not found — caching layer disabled")
        return None
    except Exception as e:
        logger.warning("Could not start Redis: %s — continuing without it", e)
        return None


def stop_redis(proc: "subprocess.Popen | None") -> None:
    """Gracefully terminate the Redis subprocess if it is running."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
        logger.info("Redis stopped")
    except Exception as e:
        logger.warning("Could not stop Redis cleanly: %s", e)
        try:
            proc.kill()
        except Exception:
            pass



# ── Environment validation ─────────────────────────────────────────────────────

def validate_env() -> bool:
    """
    Validate all harness-injected environment variables.

    Per the Track 1 spec, the harness injects:
      - FIREWORKS_API_KEY
      - FIREWORKS_BASE_URL
      - ALLOWED_MODELS  (comma-separated exact model IDs)

    We log the exact values (except key) so submission issues are debuggable.
    ALLOWED_MODELS is required — missing it means we can't route to Fireworks at all.
    """
    # FIREWORKS_API_KEY — warn if missing (local models can still run)
    fw_key = os.environ.get("FIREWORKS_API_KEY", "")
    if fw_key:
        logger.info("FIREWORKS_API_KEY: set (length=%d)", len(fw_key))
    else:
        logger.warning("FIREWORKS_API_KEY not set — Tier 3 (Fireworks) escalation will fail")

    # FIREWORKS_BASE_URL — all calls MUST go through this
    fw_url = os.environ.get("FIREWORKS_BASE_URL", "")
    if fw_url:
        logger.info("FIREWORKS_BASE_URL: %s", fw_url)
    else:
        logger.warning("FIREWORKS_BASE_URL not set — Fireworks calls will be skipped")

    # ALLOWED_MODELS — parsed exactly as: os.environ["ALLOWED_MODELS"].split(",")
    try:
        models = os.environ["ALLOWED_MODELS"].split(",")
        models = [m.strip() for m in models if m.strip()]
        logger.info("ALLOWED_MODELS: %d model(s) permitted", len(models))
        for m in models:
            logger.info("  - %s", m)
        # Pre-warm the router's model registry now
        from src.router import log_model_registry
        log_model_registry()
    except KeyError:
        logger.error(
            "ALLOWED_MODELS env var not set! "
            "Fireworks calls will be blocked to prevent MODEL_VIOLATION."
        )
        # Non-fatal: local model tiers can still run, results.json will be written

    logger.info("Environment validation complete")
    return True



# ── Task loading / output writing ─────────────────────────────────────────────

def load_tasks(path: Path) -> list[dict]:
    """Load and validate tasks from input JSON."""
    if not path.exists():
        # Raise instead of sys.exit() — inside the ASGI lifespan, sys.exit(1)
        # kills uvicorn before yield is reached, so /health never responds and
        # the harness marks the run as RUNTIME_ERROR.
        raise RuntimeError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise RuntimeError("tasks.json must be a JSON array")

    # Validate schema
    valid = []
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            logger.warning("Task %d is not an object, skipping", i)
            continue
        if "task_id" not in task or "prompt" not in task:
            logger.warning("Task %d missing task_id or prompt, skipping", i)
            continue
        valid.append(task)

    logger.info("Loaded %d tasks from %s", len(valid), path)
    return valid


def write_results(results: list[dict], path: Path) -> None:
    """Write results to output JSON. Strips internal _meta keys."""
    import stat as _stat
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Permission diagnostics — shows up in harness logs ─────────────────────
    try:
        st = out_dir.stat()
        logger.info(
            "Output dir %s: mode=%s uid=%d gid=%d",
            out_dir,
            oct(st.st_mode),
            st.st_uid,
            st.st_gid,
        )
    except Exception as e:
        logger.warning("Could not stat output dir: %s", e)

    if not os.access(out_dir, os.W_OK):
        logger.error(
            "Output dir %s is NOT writable — attempting chmod 777", out_dir
        )
        try:
            os.chmod(out_dir, 0o777)
        except Exception as chmod_err:
            logger.error("chmod failed: %s — write will likely fail", chmod_err)

    # Output schema: [{task_id, answer}]
    output = [
        {"task_id": r["task_id"], "answer": r["answer"]}
        for r in results
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Results written to %s (%d entries)", path, len(output))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_all_tasks(tasks: list[dict]) -> list[dict]:
    """
    Run all tasks through the LangGraph pipeline.

    Uses a thread pool (MAX_WORKERS) since each task is independent.
    Local models are not thread-safe, so we limit concurrency to 2 for
    the local inference tiers and allow full parallelism for Fireworks calls.

    Respects global 9.5-minute deadline — any task not started before deadline
    is answered with a fallback rather than missing from output entirely.
    """
    global _START_TIME
    # Reset deadline from NOW (actual processing start), not module import time.
    # Model loading can take 1-2 min and would otherwise eat into the budget.
    _START_TIME = time.time()

    from src.graph import run_task
    from src.token_counter import GLOBAL_TRACKER

    results: list[dict] = []
    total = len(tasks)

    # For the grading env (2 vCPU), use 2 workers max to avoid thrashing
    effective_workers = min(MAX_WORKERS, 2)
    logger.info("Processing %d tasks with %d workers", total, effective_workers)

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_task = {
            executor.submit(run_task, t["task_id"], t["prompt"]): t
            for t in tasks
        }

        for i, future in enumerate(as_completed(future_to_task), 1):
            # Check global deadline before processing each result
            remaining = time_remaining()
            if remaining <= 30:
                logger.warning(
                    "[%d/%d] Deadline approaching (%.0fs left) — using fallback for remaining tasks",
                    i, total, remaining,
                )
                # Add fallback answers for all tasks not yet in results.
                # Do NOT call executor.shutdown() here — calling it inside the
                # with block causes __exit__ to block again on running threads.
                # Just break; the with block exits and running threads finish
                # naturally (their results are already covered by fallbacks).
                done_ids = {r["task_id"] for r in results}
                for t in tasks:
                    if t["task_id"] not in done_ids:
                        results.append({"task_id": t["task_id"], "answer": "Processing time limit reached."})
                break

            task = future_to_task[future]
            try:
                # Per-task timeout: no single task should take more than 90s
                per_task_timeout = min(max(remaining - 10, 10), 90)
                result = future.result(timeout=per_task_timeout)
                results.append(result)

                meta = result.get("_meta", {})
                logger.info(
                    "[%d/%d] %s → %s | tier=%s cached=%s fw=%d+%d",
                    i, total,
                    result["task_id"],
                    meta.get("category", "?"),
                    meta.get("tier", "?"),
                    meta.get("cached", False),
                    meta.get("fw_in", 0),
                    meta.get("fw_out", 0),
                )
            except Exception as e:
                logger.error("Task %s failed: %s", task["task_id"], e)
                results.append({
                    "task_id": task["task_id"],
                    "answer": "Error: unable to process task.",
                })

    # Sort results in input order
    task_order = {t["task_id"]: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: task_order.get(r["task_id"], 999))

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("AMD Hackathon Track 1 Agent — Starting")
    logger.info("=" * 60)

    # 1. Validate environment
    validate_env()

    # 2. Start Redis
    redis_proc = start_redis()

    # 3. Preload local models (non-blocking: they lazy-load on first call)
    #    We do a non-blocking availability check here
    from src.local_model import is_available
    logger.info("Local model available: %s", is_available())

    # 4. Load tasks
    try:
        tasks = load_tasks(INPUT_PATH)
    except RuntimeError as e:
        logger.error("Failed to load tasks: %s", e)
        sys.exit(1)

    # 5. Warm up graph (compiles on first call — must happen before thread pool)
    from src.graph import get_graph
    get_graph()

    # 6. Process all tasks
    results = run_all_tasks(tasks)

    # 7. Write output
    write_results(results, OUTPUT_PATH)

    # 8. Log summary
    from src.token_counter import GLOBAL_TRACKER
    GLOBAL_TRACKER.log_summary()

    elapsed = time.time() - start_time
    logger.info("Completed in %.1fs", elapsed)

    # 9. Stop Redis
    stop_redis(redis_proc)

    # 10. Validate output exists
    if not OUTPUT_PATH.exists():
        logger.error("Output file was not created!")
        sys.exit(1)

    logger.info("SUCCESS — exiting 0")
    sys.exit(0)


if __name__ == "__main__":
    main()
    
