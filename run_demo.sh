#!/bin/bash
# ==============================================================================
# Namma Route — Local Demo Launcher
# ==============================================================================
# Starts the API and, optionally, the event simulator.
#   ./run_demo.sh                    API only
#   ENABLE_SIMULATOR=1 ./run_demo.sh API + simulated event stream
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")"

export PORT="${PORT:-7860}"
API_PID=""
SIM_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Build any missing artefacts so a fresh clone works without manual steps.
if [ ! -f data/processed/events_clean.csv ] || [ ! -f data/processed/corridor_adjacency.csv ]; then
    echo "Building processed dataset..."
    python3 backend/data_cleaning.py || exit 1
fi
if [ ! -f data/processed/models/empirical_duration.pkl ]; then
    echo "Training models..."
    python3 backend/forecasting.py || exit 1
fi

echo "Launching API on port $PORT ..."
python3 -u -m uvicorn backend.api:app --host 0.0.0.0 --port "$PORT" > /tmp/namma_route_api.log 2>&1 &
API_PID=$!

echo "Waiting for API..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "API is healthy."
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "API process died. Log:"; tail -30 /tmp/namma_route_api.log; exit 1
    fi
    sleep 1
done

if [ "${ENABLE_SIMULATOR:-0}" = "1" ]; then
    echo "Starting event simulator..."
    ENABLE_SIMULATOR=1 python3 -u backend/simulate_stream.py &
    SIM_PID=$!
fi

echo "============================================================"
echo "  Namma Route is live:  http://localhost:$PORT/"
echo "  API log: /tmp/namma_route_api.log"
echo "  Ctrl+C to stop."
echo "============================================================"
wait $API_PID
