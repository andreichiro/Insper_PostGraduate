#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
PORT="$2"
PYTHON_BIN="${3:-python}"

exec "$PYTHON_BIN" -m http.server \
  "$PORT" \
  --bind 127.0.0.1 \
  --directory "$PROJECT_DIR/dbt_lineage/target"
