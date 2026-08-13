#!/bin/bash
# LaunchAgent wrapper: keeps admin.py alive via a restart loop.
# launchd itself only guards this wrapper; the wrapper guards admin.py.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR" || exit 1

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

# Ensure logs directory exists
mkdir -p logs

while true; do
    echo "[admin_wrapper] starting admin.py at $(date -Iseconds)"
    "$PYTHON_BIN" scripts/admin.py
    EXIT_CODE=$?
    echo "[admin_wrapper] admin.py exited with code $EXIT_CODE at $(date -Iseconds), restarting in 3s..."
    sleep 3
done
