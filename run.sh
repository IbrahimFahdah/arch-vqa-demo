#!/usr/bin/env bash
# Start the app on http://localhost:7860 (Ctrl+C stops it and the model server).
set -euo pipefail
cd "$(dirname "$0")"
# If the app is killed without cleaning up (e.g. on Windows a child process outlives its parent),
# stop the llama-server it started, and only that one.
cleanup() {
  local f=logs/llama-server.pid
  [ -f "$f" ] || return 0
  local pid; pid=$(cat "$f"); rm -f "$f"
  if command -v taskkill > /dev/null; then taskkill //F //PID "$pid" > /dev/null 2>&1 || true
  else kill "$pid" 2> /dev/null || true; fi
}
trap cleanup EXIT
for py in python3 python py; do
  if command -v "$py" > /dev/null && "$py" -c "import sys; sys.exit(sys.version_info < (3, 8))" 2> /dev/null; then
    "$py" app/app.py
    exit
  fi
done
echo "Python 3.8 or newer is needed (https://www.python.org/downloads/)"; exit 1
