set -e

if [ ! -f "data/06_models/production_model.pkl" ]; then
    echo "Modelo de produção não encontrado; treinando…"
    uv run kedro run
fi

echo "Subindo servidor da API"
exec uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
