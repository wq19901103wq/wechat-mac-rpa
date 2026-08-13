#!/bin/bash
# LaunchAgent wrapper: keeps admin.py alive via a restart loop.
# launchd itself only guards this wrapper; the wrapper guards admin.py.

cd /Users/yourname/wechat-mac-rpa

# Ensure logs directory exists
mkdir -p logs

while true; do
    echo "[admin_wrapper] starting admin.py at $(date -Iseconds)"
    /Users/yourname/anaconda3/bin/python3 scripts/admin.py
    EXIT_CODE=$?
    echo "[admin_wrapper] admin.py exited with code $EXIT_CODE at $(date -Iseconds), restarting in 3s..."
    sleep 3
done
