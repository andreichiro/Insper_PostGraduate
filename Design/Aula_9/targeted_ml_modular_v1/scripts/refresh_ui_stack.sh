#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/build}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Keep ports fixed to match the HTML links and avoid UI/config drift.
DBT_PORT="8081"
STREAMLIT_PORT="8501"
RUNTIME_DIR="$PROJECT_DIR/.runtime/ui_stack"
LOG_DIR="$RUNTIME_DIR/logs"
PID_DIR="$RUNTIME_DIR/pids"
SESSION_DIR="$RUNTIME_DIR/sessions"

mkdir -p "$LOG_DIR" "$PID_DIR" "$SESSION_DIR"
: > "$LOG_DIR/dbt_docs.log"
: > "$LOG_DIR/streamlit.log"

kill_if_running() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

kill_port_owner() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill $pids >/dev/null 2>&1 || true
    sleep 1
  fi
}

kill_screen_session() {
  local session_file="$1"
  if [[ -f "$session_file" ]]; then
    local session_name
    session_name="$(cat "$session_file")"
    if [[ -n "$session_name" ]]; then
      screen -S "$session_name" -X quit >/dev/null 2>&1 || true
    fi
    rm -f "$session_file"
  fi
}

spawn_detached() {
  local pid_file="$1"
  local log_file="$2"
  shift 2
  "$PYTHON_BIN" - "$pid_file" "$log_file" "$@" <<'PY'
import os
import subprocess
import sys

pid_file, log_file, *cmd = sys.argv[1:]
with open(log_file, "ab", buffering=0) as log_handle, open(os.devnull, "rb") as devnull:
    process = subprocess.Popen(
        cmd,
        stdin=devnull,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
    )
with open(pid_file, "w", encoding="utf-8") as handle:
    handle.write(str(process.pid))
PY
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local retries="${3:-30}"
  local sleep_seconds="${4:-1}"
  local attempt
  for attempt in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
  done
  printf '%s\n' "ERROR: $name did not become ready at $url" >&2
  return 1
}

kill_if_running "$PID_DIR/dbt_docs.pid"
kill_if_running "$PID_DIR/streamlit.pid"
kill_screen_session "$SESSION_DIR/dbt_docs.session"
kill_screen_session "$SESSION_DIR/streamlit.session"
screen -wipe >/dev/null 2>&1 || true
kill_port_owner "$DBT_PORT"
kill_port_owner "$STREAMLIT_PORT"

TARGETED_ML_BUILD_DUCKDB="$OUTPUT_ROOT/duckdb/build.duckdb" \
TARGETED_ML_MODELLED_DUCKDB="$OUTPUT_ROOT/modelled/duckdb/base_modelada_v2.duckdb" \
dbt docs generate --project-dir "$PROJECT_DIR/dbt_lineage" --profiles-dir "$PROJECT_DIR/dbt_lineage"

spawn_detached \
  "$PID_DIR/dbt_docs.pid" \
  "$LOG_DIR/dbt_docs.log" \
  /bin/bash "$PROJECT_DIR/scripts/run_dbt_docs.sh" \
  "$PROJECT_DIR" \
  "$DBT_PORT" \
  "$PYTHON_BIN"

if command -v screen >/dev/null 2>&1; then
  screen -DmS targeted_ml_streamlit /bin/bash "$PROJECT_DIR/scripts/run_streamlit.sh" \
    "$PROJECT_DIR" \
    "$STREAMLIT_PORT" \
    "$LOG_DIR/streamlit.log" \
    "$PYTHON_BIN" &
  printf '%s' "targeted_ml_streamlit" > "$SESSION_DIR/streamlit.session"
else
  spawn_detached \
    "$PID_DIR/streamlit.pid" \
    "$LOG_DIR/streamlit.log" \
    "$PYTHON_BIN" -m streamlit run "$PROJECT_DIR/targeted_ml/apps/streamlit_app.py" \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true
fi

sleep 2

wait_for_url "dbt docs" "http://localhost:$DBT_PORT/"
wait_for_url "Streamlit" "http://localhost:$STREAMLIT_PORT/"

printf '%s\n' "dbt docs: http://localhost:$DBT_PORT"
printf '%s\n' "Streamlit: http://localhost:$STREAMLIT_PORT"
printf '%s\n' "Logs: $LOG_DIR"
