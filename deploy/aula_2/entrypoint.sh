#!/bin/sh
set -e

if [ ! -f "data/06_models/production_model.pkl" ]; then
    echo "No production model found — training…"
    uv run kedro run
fi

echo "Starting API server…"
exec uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
