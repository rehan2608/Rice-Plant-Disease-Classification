#!/usr/bin/env bash
# run.sh — Run Rice Disease Classification Web App (F2.py)

set -e

# Change to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PORT=8501

# Determine Python binary (prefer Anaconda Python if present)
if [ -x "/opt/anaconda3/bin/python3" ]; then
    PYTHON="/opt/anaconda3/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
elif command -v python &>/dev/null; then
    PYTHON="$(command -v python)"
else
    echo "❌ Error: Python not found. Please install Python 3 or activate your environment."
    exit 1
fi

echo "============================================="
echo "🌾 Rice Plant Disease Classification App"
echo "============================================="
echo "📂 Working directory: $PROJECT_DIR"
echo "🐍 Python executable: $PYTHON"

# Free port 8501 if an existing instance is running
PID=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo "⚠️  Port $PORT is currently in use (PID: $PID). Stopping old process..."
    kill -9 $PID 2>/dev/null || true
    sleep 1
fi

# Automatically open the browser on macOS
if command -v open &>/dev/null; then
    (sleep 1.5 && open "http://localhost:$PORT" 2>/dev/null) &
fi

echo "🚀 Starting Streamlit server on http://localhost:$PORT ..."
exec "$PYTHON" -m streamlit run F2.py --server.port=$PORT --server.headless=true

