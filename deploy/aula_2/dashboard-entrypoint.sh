#!/bin/sh
set -e

echo "Esperando artefatos de produção…"
while [ ! -f "data/06_models/production_model.pkl" ]; do
    sleep 5
done

echo "Subindo dashboard Streamlit"
exec uv run streamlit run src/insper_deploy_kedro/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
