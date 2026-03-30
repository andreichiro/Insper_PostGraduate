"""Custom Kedro CLI commands. Auto-discovered by Kedro when placed next to settings.py."""

import click


@click.group(name="insper-deploy-kedro")
def cli():
    """Churn project commands."""


@cli.command()
def cheatsheet():
    """Print all project commands organized by workflow."""
    click.echo(
        """
╔══════════════════════════════════════════════════════════════════════╗
║                    CHURN PROJECT — COMMAND CHEATSHEET               ║
╚══════════════════════════════════════════════════════════════════════╝

─── Pipeline Execution ────────────────────────────────────────────────

  uv run kedro run
      Run the FULL pipeline (data engineering → modelling → refit → inference)

  uv run kedro run --pipeline data_engineering
      clean → features → split → encode → scale

  uv run kedro run --pipeline modelling
      optimize 3 models (Optuna) → evaluate → select best → test report

  uv run kedro run --pipeline refit
      refit winner on all data → calibrate probabilities

  uv run kedro run --pipeline inference
      transform inference CSV → predict with production model

─── API Server ────────────────────────────────────────────────────────

  uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
      Start the API locally

  open http://localhost:8000/docs
      Open Swagger docs (interactive API testing)

  curl http://localhost:8000/health
      Health check

  curl -X POST http://localhost:8000/inference \\
    -H "Content-Type: application/json" \\
    -H "X-API-Key: your-key" \\
    -d '{"instances": [{ ... }]}'
      Inference request (with API key)

─── Development ───────────────────────────────────────────────────────

  uv sync --all-extras
      Install all dependencies (main + dev + docs)

  uv run ruff check src/ tests/
      Lint (check for errors)

  uv run ruff check --fix src/ tests/
      Lint (auto-fix)

  uv run ruff format src/ tests/
      Format code

  uv run ruff format --check src/ tests/
      Check formatting (dry-run)

─── Testing ───────────────────────────────────────────────────────────

  uv run pytest
      Run all tests with coverage

  uv run pytest -v
      Verbose output

  uv run pytest tests/test_api.py
      API endpoint tests

  uv run pytest tests/test_nodes_data_engineering.py
      DE node tests

  uv run pytest tests/test_nodes_modelling.py
      Modelling node tests

  uv run pytest tests/test_nodes_inference.py
      Inference node tests

  uv run pytest tests/test_run.py
      Full pipeline integration test

  uv run pytest tests/test_nodes_modelling.py::TestOptimizeModel
      Specific test class

  uv run pytest --no-cov -x
      Skip coverage, stop on first failure

─── Docker ────────────────────────────────────────────────────────────

  docker compose up --build
      Build and run with docker-compose

  API_KEY=my-secret docker compose up --build
      Run with API key

  docker build -t churn-api .
      Build image only

  docker run -p 8000:8000 -e API_KEY=my-secret churn-api
      Run container manually

─── Cloud Run Deployment ──────────────────────────────────────────────

  export GCP_PROJECT_ID=your-project-id
  export GCS_BUCKET=your-bucket-name
  export API_KEY=your-api-key

  ./deploy.sh
      One-command deploy

  gsutil -m cp data/06_models/production_*.pkl gs://$GCS_BUCKET/churn/models/
      Upload artifacts to GCS

  gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/churn-api:latest
      Build container image

  gcloud run deploy churn-api \\
    --image gcr.io/$GCP_PROJECT_ID/churn-api:latest \\
    --set-env-vars KEDRO_ENV=cloud,GCS_BUCKET=$GCS_BUCKET,API_KEY=$API_KEY \\
    --allow-unauthenticated --memory 1Gi
      Deploy to Cloud Run

─── Kedro Utilities ───────────────────────────────────────────────────

  uv run kedro viz run
      Visualize the pipeline DAG in the browser

  uv run kedro registry list
      List all registered pipelines

  uv run kedro catalog list
      List all datasets in the catalog

  uv run kedro run --nodes optimize_baseline_node
      Run a single node by name

  uv run kedro run --from-nodes select_best_model_node
      Run from a specific node onward

  uv run kedro run --to-nodes evaluate_baseline_node
      Run up to a specific node

═══════════════════════════════════════════════════════════════════════
"""
    )
