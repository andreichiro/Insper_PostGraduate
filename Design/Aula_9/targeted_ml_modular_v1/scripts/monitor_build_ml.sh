#!/bin/zsh
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <build_pid> <output_dir> <poll_seconds>" >&2
  exit 2
fi

BUILD_PID="$1"
OUTPUT_DIR="$2"
POLL_SECONDS="$3"

METADATA_DIR="${OUTPUT_DIR}/metadata"
LOG_PATH="${METADATA_DIR}/build_ml_watch.log"
STATUS_PATH="${METADATA_DIR}/build_ml_watch_status.json"

mkdir -p "${METADATA_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

write_status() {
  local build_status="$1"
  local detail="$2"
  cat > "${STATUS_PATH}" <<EOF
{
  "status": "${build_status}",
  "detail": "${detail}",
  "build_pid": ${BUILD_PID},
  "updated_at": "$(timestamp)"
}
EOF
}

notify() {
  local title="$1"
  local message="$2"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${message}\" with title \"${title}\"" >/dev/null 2>&1 || true
  fi
}

echo "[$(timestamp)] monitor started for pid=${BUILD_PID} poll=${POLL_SECONDS}s" >> "${LOG_PATH}"
write_status "running" "monitor_started"

while true; do
  if kill -0 "${BUILD_PID}" >/dev/null 2>&1; then
    echo "[$(timestamp)] build still running" >> "${LOG_PATH}"
    write_status "running" "build_running"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if [[ -f "${METADATA_DIR}/build_summary_v1.json" ]]; then
    echo "[$(timestamp)] build completed successfully" >> "${LOG_PATH}"
    write_status "completed" "build_summary_present"
    notify "targeted_ml build" "build_same_month_entry concluido com sucesso"
    exit 0
  fi

  echo "[$(timestamp)] build stopped before build_summary_v1.json was written" >> "${LOG_PATH}"
  write_status "failed" "process_stopped_without_build_summary"
  notify "targeted_ml build" "build_same_month_entry parou antes do fim"
  exit 1
done
