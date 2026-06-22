#!/bin/bash

# ==============================================================================
# Gridlock Prototype — Production Launcher
# ==============================================================================

echo "🚦 Starting Gridlock 2.0 Prototype in Production Mode..."

# 1. Start the Real-Time Simulation Stream in the background
echo "🚀 Starting Real-Time Simulation Stream..."
python3 -u backend/simulate_stream.py &

# 2. Start the API Server in the foreground
# Use port 7860 by default for Hugging Face Spaces, or the PORT environment variable if provided by the host
export PORT="${PORT:-7860}"
echo "🚀 Launching FastAPI server on port $PORT..."
exec python3 -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT
