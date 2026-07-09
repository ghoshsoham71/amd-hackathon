#!/bin/bash
set -e

echo "=== AMD Track 1 Agent Starting ==="

# ── Validate required directories ─────────────────────────────────────────────
mkdir -p /input /output

# ── Fix permissions at runtime (after harness mounts override Dockerfile perms) ─
chmod 777 /output 2>/dev/null || echo "[entrypoint] WARNING: could not chmod /output"
chmod 755 /input  2>/dev/null || echo "[entrypoint] WARNING: could not chmod /input"
echo "[entrypoint] /output permissions: $(stat -c '%a %U:%G' /output)"
echo "[entrypoint] /input  permissions: $(stat -c '%a %U:%G' /input)"

# ── Launch via uvicorn ────────────────────────────────────────────────────────
# - src.app:app      → FastAPI application object
# - --host 0.0.0.0   → accessible for health checks inside container
# - --port 8080      → internal port (not exposed externally)
# - --workers 1      → single worker (task state is in-process)
# - --log-level info → structured logs to stdout
# - --timeout-graceful-shutdown 30 → allow cleanup on SIGTERM
echo "[entrypoint] Starting uvicorn..."
exec uvicorn src.app:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 1 \
  --log-level info \
  --timeout-graceful-shutdown 30 \
  --no-access-log
