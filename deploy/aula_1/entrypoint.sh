#!/bin/sh
set -e

PORT="${PORT:-8000}"

if [ "${KEDRO_ENV}" = "cloud" ]; then
    echo "Cloud environment — artifacts loaded from GCS at startup"
elif [ ! -f "data/06_models/production_model.pkl" ]; then
    echo "No production model found — training…"
    uv run kedro run
fi

echo "Starting API server on port ${PORT}…"
exec uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port "${PORT}"
