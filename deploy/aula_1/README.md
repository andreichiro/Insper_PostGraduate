# Churn Prediction — Production ML Pipeline

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

End-to-end ML project that predicts customer churn using a Kedro pipeline
and serves predictions via FastAPI. Deploys to Google Cloud Run with a
single script.

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

## Deploy to Google Cloud Run

The API loads model artifacts via Kedro's **DataCatalog**. Switching from
local to cloud requires **zero code changes** — only a catalog overlay:

| Environment | Catalog                  | Artifacts live in        |
|-------------|--------------------------|--------------------------|
| `local`     | `conf/base/catalog.yml`  | `data/06_models/`        |
| `cloud`     | `conf/cloud/catalog.yml` | `gs://BUCKET/churn/models/` |

### Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/install) installed
2. Authenticated: `gcloud auth login`
3. Model trained locally: `uv run kedro run`

### One-script deploy

```bash
export GCP_PROJECT_ID=your-project-id
export GCS_BUCKET=your-bucket-name

./deploy.sh
```

This will:
1. Create the GCS bucket (if needed)
2. Upload production artifacts (`production_encoders.pkl`, `production_scalers.pkl`, `production_model.pkl`)
3. Build the container image with Cloud Build
4. Deploy to Cloud Run with `KEDRO_ENV=cloud`

### Manual deploy

```bash
# Upload artifacts
gsutil -m cp data/06_models/production_*.pkl gs://BUCKET/churn/models/

# Build image
gcloud builds submit --tag gcr.io/PROJECT/churn-api:latest

# Deploy
gcloud run deploy churn-api \
    --image gcr.io/PROJECT/churn-api:latest \
    --set-env-vars KEDRO_ENV=cloud,GCS_BUCKET=BUCKET \
    --allow-unauthenticated \
    --memory 1Gi
```

## Project Structure

```
├── conf/
│   ├── base/               # Shared config (committed)
│   │   ├── catalog.yml      # Data Catalog (local paths)
│   │   └── parameters/      # Pipeline parameters (YAML)
│   ├── cloud/               # Cloud overlay (committed)
│   │   └── catalog.yml      # Overrides → gs://BUCKET/...
│   └── local/               # Machine-specific (gitignored)
│       └── credentials.yml  # Secrets — NEVER committed
├── data/
│   └── 01_raw/              # Raw input CSVs
├── src/insper_deploy_kedro/
│   ├── api.py               # FastAPI serving layer (env-aware)
│   └── pipelines/
│       ├── data_engineering/ # clean → features → split → encode → scale
│       ├── modelling/        # optimize (Optuna) → evaluate → select → test report
│       ├── inference/        # transform-only → predict
│       └── refit/            # refit on all data → calibrate → production
├── tests/                   # Unit, integration, and e2e tests
├── Dockerfile
├── docker-compose.yml
├── deploy.sh                # One-command Cloud Run deployment
└── pyproject.toml           # Dependencies + tool config
```

## Reusing This Pipeline for a New Dataset

The pipeline is designed to be **modular** — adapting it to a different
classification problem (e.g. fraud detection, lead scoring) requires
changing only **configuration files**, not pipeline code.

### What to change

| Layer | File | What to change |
|-------|------|----------------|
| **Columns** | `conf/base/parameters/data_engineering.yml` | `raw_columns`, `columns` — list your dataset's categorical, numerical, and target columns |
| **Feature engineering** | `src/.../data_engineering/nodes.py` → `add_features()` | The only Python function with dataset-specific logic. Add/remove derived features here |
| **Models & search spaces** | `conf/base/parameters/modelling.yml` | Optuna search spaces, `init_args`, `n_trials`, `cv`, `scoring` for each model |
| **Class imbalance** | `conf/base/parameters/modelling.yml` | Adjust `scale_pos_weight`, `class_weight`, `auto_class_weights` |
| **Data catalog** | `conf/base/catalog.yml` | Point `raw_data` and `raw_data_inference` to your CSV/Parquet files |
| **API schema** | `src/.../api.py` → `ChurnFeatures` | Update the Pydantic model to match your feature columns |

### What stays the same (zero changes needed)

- `clean_data`, `add_split_column`, `fit_encoders`, `transform_encoders`,
  `fit_scalers`, `transform_scalers` — all driven by the column config
- `optimize_model` — generic Optuna optimizer, reads search space from YAML
- `evaluate_model`, `evaluate_all_on_test` — metric computation
- `select_best_model` — picks the winner by configurable metric
- `calibrate_model` — probability calibration (sigmoid/isotonic)
- `train_model` — dynamic class loading from `class_path`
- `predict`, `to_dataframe` — generic inference nodes
- Refit pipeline — reuses DE + modelling nodes on all data

### Kedro namespaces for multi-dataset projects

To run the **same pipeline logic** on multiple datasets simultaneously,
use Kedro's namespace feature:

```python
from kedro.pipeline import pipeline
from .base_pipeline import create_pipeline as base

def create_pipeline(**kwargs):
    return (
        pipeline(base(), namespace="churn", parameters={"params:columns": "params:churn_columns"})
        + pipeline(base(), namespace="fraud", parameters={"params:columns": "params:fraud_columns"})
    )
```

Each namespace gets isolated datasets (`churn.master_table`,
`fraud.master_table`) while sharing the same node functions.

## ML Pipeline Details

### Hyperparameter Optimization (Optuna)

All three models are optimized with Optuna's TPE sampler:

| Model | Searched Parameters | Trials | CV Folds |
|-------|-------------------|--------|----------|
| LogisticRegression | `C`, `solver` | 20 | 5 |
| CatBoostClassifier | `depth`, `iterations`, `learning_rate`, `l2_leaf_reg` | 30 | 5 |
| XGBClassifier | `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree` | 30 | 5 |

Search spaces are defined entirely in `conf/base/parameters/modelling.yml`.

### Feature Encoding

- **Categorical features**: `OrdinalEncoder` (handles unseen categories
  with `unknown_value=-1`)
- **Target**: `LabelEncoder` (binary: No=0, Yes=1)
- **Numerical features**: `StandardScaler` (zero mean, unit variance)

All encoders/scalers are **fit on the train split only**, then applied
to all splits (preventing data leakage).

### Probability Calibration

After refitting on all data, the production model is wrapped with
`CalibratedClassifierCV` (Platt scaling / sigmoid method). This ensures
predicted probabilities are well-calibrated for business thresholding
(e.g. "contact the top 20% most likely to churn").

### Test Evaluation Report

The `test_evaluation_node` evaluates all three candidate models on the
held-out test set and saves a comprehensive report including confusion
matrices. This runs as a pipeline node (not just a manual check).

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
| `conf/cloud/` | GCS artifact paths         | Committed |
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

| Method | Path         | Description                        |
|--------|--------------|------------------------------------|
| GET    | `/health`    | Liveness / readiness probe         |
| POST   | `/inference` | Batch churn predictions (JSON)     |
| GET    | `/docs`      | Interactive Swagger documentation  |
