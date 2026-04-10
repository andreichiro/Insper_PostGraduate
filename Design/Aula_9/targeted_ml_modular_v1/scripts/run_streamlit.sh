#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
PORT="$2"
LOG_FILE="$3"
PYTHON_BIN="${4:-python}"

exec "$PYTHON_BIN" -m streamlit run \
  "$PROJECT_DIR/targeted_ml/apps/streamlit_app.py" \
  --server.port "$PORT" \
  --server.headless true >> "$LOG_FILE" 2>&1
