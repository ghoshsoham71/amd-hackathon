#!/bin/bash

echo "=== AMD Track 1 Agent Starting ==="

# -- Validate required directories ---------------------------------------------
# We suppress errors here because if the grading harness runs as a non-root
# user with a read-only filesystem, this command will fail and `set -e`
# would immediately crash the container.
mkdir -p /input /output 2>/dev/null || echo "[entrypoint] WARNING: could not create /input or /output (likely read-only filesystem)"

# -- Fix permissions at runtime (after harness mounts override Dockerfile perms) -
chmod 777 /output 2>/dev/null || echo "[entrypoint] WARNING: could not chmod /output (likely insufficient permissions)"
chmod 755 /input  2>/dev/null || echo "[entrypoint] WARNING: could not chmod /input (likely insufficient permissions)"

# Only try to stat if they exist, otherwise ignore to prevent crash
if [ -d "/output" ]; then echo "[entrypoint] /output permissions: $(stat -c '%a %U:%G' /output 2>/dev/null || echo 'unknown')"; fi
if [ -d "/input" ]; then echo "[entrypoint] /input  permissions: $(stat -c '%a %U:%G' /input 2>/dev/null || echo 'unknown')"; fi

# -- Launch via uvicorn --------------------------------------------------------
# - src.app:app      -> FastAPI application object
# - --host 0.0.0.0   -> accessible for health checks inside container
# - --port 8080      -> internal port (not exposed externally)
# - --workers 1      -> single worker (task state is in-process)
# - --log-level info -> structured logs to stdout
# - --timeout-graceful-shutdown 30 -> allow cleanup on SIGTERM
echo "[entrypoint] Starting uvicorn..."

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec python -m uvicorn src.app:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 1 \
  --log-level info \
  --timeout-graceful-shutdown 30 \
  --no-access-log
