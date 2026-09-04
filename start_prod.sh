#!/bin/bash
# ==============================================================================
# Namma Route — Production Launcher
# ==============================================================================
set -euo pipefail

export PORT="${PORT:-7860}"

# The event simulator is OFF unless explicitly enabled. It used to start
# unconditionally here, which meant the public deployment continuously
# manufactured events and appended synthetic feedback to the learning log —
# inflating the very counters the dashboard presented as evidence of real use.
if [ "${ENABLE_SIMULATOR:-0}" = "1" ]; then
    echo "Simulator enabled — starting event stream in the background."
    python3 -u backend/simulate_stream.py &
else
    echo "Simulator disabled (set ENABLE_SIMULATOR=1 to enable)."
fi

echo "Launching API on port $PORT ..."
exec python3 -u -m uvicorn backend.api:app --host 0.0.0.0 --port "$PORT"
