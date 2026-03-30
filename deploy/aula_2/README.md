# Diabetes Prediction — Production ML Pipeline

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

End-to-end ML project that predicts diabetes outcome using a Kedro pipeline
and serves predictions via FastAPI.

## Quick Start

```bash
# 1. Install dependencies (requires Python 3.13+ and uv)
uv sync

# 2. Train the full pipeline (data engineering → modelling → refit)
uv run kedro run

# 3. Start the API server
uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000

# 4. Open Swagger docs
open http://localhost:8000/docs
```

### Docker (one command)

```bash
docker compose up --build
```

The container trains the model (if not already trained) and starts the API
on port 8000.

## Project Structure

```
├── conf/
│   ├── base/               # Shared config (committed)
│   │   ├── catalog.yml      # Data Catalog (I/O definitions)
│   │   └── parameters/      # Pipeline parameters (YAML)
│   └── local/               # Environment-specific (gitignored)
│       └── credentials.yml  # Secrets — NEVER committed
├── data/
│   └── 01_raw/              # Raw input CSVs
├── src/insper_deploy_kedro/
│   ├── api.py               # FastAPI serving layer
│   └── pipelines/
│       ├── data_engineering/ # clean → split → encode → scale
│       ├── modelling/        # train → evaluate → optimize
│       ├── inference/        # transform-only → predict
│       └── refit/            # refit on all data for production
├── tests/                   # Unit, integration, and e2e tests
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml           # Dependencies + tool config
```

## Development

### Linting and Formatting (Ruff)

```bash
uv run ruff check src/      # lint (F, I, UP, PL, T201 rules)
uv run ruff format src/     # format (replaces black + isort)
```

### Testing (pytest)

```bash
uv run pytest                # run all tests with coverage
uv run pytest -v             # verbose output
uv run pytest tests/test_api.py  # e2e API tests only
```

Test pyramid:
- **Unit tests** — individual node functions (`test_nodes_*.py`)
- **Integration tests** — pipeline DAG assembly
- **E2E tests** — FastAPI endpoints (`test_api.py`)

### Logging

All node functions use `logging.getLogger(__name__)` (never `print()`).
The logging configuration (`conf/logging.yml`) provides:
- **Rich console output** via Kedro's RichHandler
- **Rotating file handler** — `info.log`, 10 MB max, 20 backups

### Kedro Environments

| Directory     | Purpose                    | Git      |
|---------------|----------------------------|----------|
| `conf/base/`  | Shared config              | Committed |
| `conf/local/` | Machine-specific overrides | Ignored  |

Credentials go in `conf/local/credentials.yml` — **never committed**.

## Pipelines

| Pipeline           | Command                                        |
|--------------------|-------------------------------------------------|
| All (default)      | `uv run kedro run`                              |
| Data engineering   | `uv run kedro run --pipeline data_engineering`  |
| Modelling          | `uv run kedro run --pipeline modelling`         |
| Refit (production) | `uv run kedro run --pipeline refit`             |
| Inference          | `uv run kedro run --pipeline inference`         |

## Kedro Commands

### Visualization

```bash
uv run kedro viz run                              # open pipeline DAG in browser (localhost:4141)
```

### Registry & Catalog

```bash
uv run kedro registry list                        # list all registered pipelines
uv run kedro catalog list                         # list all datasets in the catalog
uv run kedro catalog resolve                      # show resolved catalog (with interpolation)
```

### Running Specific Nodes

```bash
uv run kedro run --nodes optimize_baseline_node   # run a single node by name
uv run kedro run --nodes "optimize_baseline_node,evaluate_baseline_node"  # multiple nodes
uv run kedro run --from-nodes select_best_model_node   # from a node onward
uv run kedro run --to-nodes evaluate_baseline_node     # up to a node (inclusive)
uv run kedro run --tags training                  # run nodes with a specific tag
```

### Pipeline-Level Execution

```bash
uv run kedro run                                  # full pipeline (all stages)
uv run kedro run --pipeline data_engineering      # clean → features → split → encode → scale
uv run kedro run --pipeline modelling             # optimize 3 models → evaluate → select → test report
uv run kedro run --pipeline refit                 # refit winner on all data → calibrate
uv run kedro run --pipeline inference             # transform inference CSV → predict
```

### Environment & Configuration

```bash
uv run kedro run --env cloud                      # use conf/cloud/ overlay
uv run kedro run --params "random_state:123"      # override a parameter at runtime
```

### Project Info

```bash
uv run kedro info                                 # project metadata (name, version, source dir)
uv run kedro cheatsheet                           # print all project commands (custom)
```

## API Endpoints

| Method | Path         | Description                           |
|--------|--------------|---------------------------------------|
| GET    | `/health`    | Liveness / readiness probe            |
| POST   | `/inference` | Batch diabetes predictions (JSON)     |
| GET    | `/docs`      | Interactive Swagger documentation     |
