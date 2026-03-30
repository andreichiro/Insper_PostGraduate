"""Custom Kedro CLI commands (auto-discovered via entry point)."""

import click


@click.group(name="custom")
def cli():
    """Diabetes prediction project commands."""


@cli.command()
def cheatsheet():
    """Print a quick-reference of every project command."""
    click.echo(
        """
╔═══════════════════════════════════════════════════════════════════╗
║                   DIABETES PROJECT — CHEATSHEET                   ║
╚═══════════════════════════════════════════════════════════════════╝

Pipeline Execution
──────────────────
  uv run kedro run                                    # full pipeline
  uv run kedro run --pipeline data_engineering        # clean → features → split → encode → scale
  uv run kedro run --pipeline modelling               # optimize 3 models (Optuna) → evaluate → select → test report
  uv run kedro run --pipeline refit                   # refit winner on all data → calibrate probabilities
  uv run kedro run --pipeline inference               # transform inference CSV → predict

API Server
──────────
  uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
  open http://localhost:8000/docs                     # Swagger docs
  curl http://localhost:8000/health                   # health check
  curl -X POST http://localhost:8000/inference \\
    -H "Content-Type: application/json" \\
    -H "X-API-Key: your-key" \\
    -d '{"instances": [{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}]}'

Development
───────────
  uv sync --all-extras                                # install all deps
  uv run ruff check src/ tests/                       # lint (check)
  uv run ruff check --fix src/ tests/                 # lint (auto-fix)
  uv run ruff format src/ tests/                      # format
  uv run ruff format --check src/ tests/              # format (dry-run)

Testing
───────
  uv run pytest                                       # all tests + coverage
  uv run pytest -v                                    # verbose
  uv run pytest tests/test_api.py                     # API tests
  uv run pytest tests/test_nodes_data_engineering.py  # DE node tests
  uv run pytest tests/test_nodes_modelling.py         # modelling node tests
  uv run pytest --no-cov -x                           # skip coverage, stop on 1st failure

Docker
──────
  docker compose up --build
  API_KEY=my-secret docker compose up --build
  docker build -t diabetes-api .
  docker run -p 8000:8000 -e API_KEY=my-secret diabetes-api

Kedro Utilities
───────────────
  uv run kedro viz run                                # visualize DAG
  uv run kedro registry list                          # list pipelines
  uv run kedro catalog list                           # list datasets
  uv run kedro run --nodes optimize_baseline_node     # run single node
  uv run kedro run --from-nodes select_best_model_node
  uv run kedro run --to-nodes evaluate_baseline_node
"""
    )
