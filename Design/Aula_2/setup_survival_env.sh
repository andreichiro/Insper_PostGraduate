#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements-survival.txt"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Missing requirements file: $REQ_FILE" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$REQ_FILE"

python - <<'PY'
import importlib
mods = ["xgboost", "sksurv", "sklearn", "duckdb", "pandas", "numpy", "matplotlib", "seaborn", "plotly"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"Missing modules after install: {missing}")
print("Environment ready. Installed modules verified.")
PY

echo "Done. Activate with: source $VENV_DIR/bin/activate"
