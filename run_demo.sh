#!/bin/bash

# ==============================================================================
# Gridlock Prototype — Demo Launcher
# ==============================================================================
# This script starts the FastAPI backend and the Real-Time Simulator stream.
# It handles graceful shutdown of both processes upon exit.
# ==============================================================================

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "🚦 Starting Gridlock 2.0 Prototype..."

# Function to clean up background processes on exit
cleanup() {
    echo -e "\n🛑 Shutting down Gridlock processes..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    if [ ! -z "$SIM_PID" ]; then
        kill $SIM_PID 2>/dev/null
    fi
    echo "✅ Shutdown complete."
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# 1. Start the API Server
echo "🚀 Launching FastAPI server on port 8000..."
python3 -m uvicorn backend.api:app --port 8000 > /dev/null 2>&1 &
API_PID=$!

# Wait for API to become healthy
echo "⏳ Waiting for API to become ready..."
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ API is online and healthy!"
        break
    fi
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ API failed to start. Check your python environment and logs."
    cleanup
fi

# 2. Start the Real-Time Simulation Stream
echo "🚀 Starting Real-Time Simulation Stream..."
python3 -u backend/simulate_stream.py &
SIM_PID=$!

echo "============================================================"
echo "🎯 GRIDLOCK IS LIVE!"
echo "👉 Open your browser to: http://localhost:8000/"
echo "⚙️  Press Ctrl+C to stop the demo."
echo "============================================================"

# Wait for processes
wait $SIM_PID
wait $API_PID
