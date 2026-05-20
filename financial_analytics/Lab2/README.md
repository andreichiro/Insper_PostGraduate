# PADS Forecasting Lab2

Kedro + MLflow forecasting project for the Distribuidora BR acquisition case.

The project is YAML-driven from `conf/base/`, uses Pydantic/custom contracts for leakage-safe validation, and writes durable outputs to `data/08_reporting/` plus the assignment file at `outputs/previsao.csv`.

Reporting is artifact-only: final decisions and QA checks are represented in Kedro Parquet outputs, SHAP explainability artifacts, a self-contained HTML report, and MLflow artifacts, with no notebook required as a deliverable. The one CSV kept intentionally is the assignment submission file, `outputs/previsao.csv`.

Model selection uses `common_mase` for cross-strategy ranking. The denominator is fixed per fold from the observed post-merger target process, so `post_only`, `proforma_sum`, and `calibrated_alpha` are compared on the same MASE scale. Local MASE remains in the artifacts as a diagnostic.

Core commands:

```bash
uv run kedro run --pipeline full
uv run kedro run --pipeline old_data_gate
uv run kedro run --pipeline model_comparison
uv run kedro run --pipeline final_forecast
uv run kedro run --pipeline reporting
```

Run checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Run the bounded Phase 10 clean-room QA pipeline:

```bash
uv run kedro run --env qa --pipeline full
```

The QA environment keeps every model lane enabled, writes to `data/99_qa/`, `outputs/previsao_qa.csv`, and `mlruns_qa/`, and does not create a notebook.

Key report artifacts:

```text
data/08_reporting/pads_forecasting_report.html
data/08_reporting/horizon_metrics.parquet
data/08_reporting/horizon_summary.parquet
data/08_reporting/mase_uncertainty.parquet
data/08_reporting/nested_selection_audit.parquet
data/08_reporting/rolling_origin_robustness.parquet
data/08_reporting/shap_feature_importance.parquet
data/08_reporting/shap_values_sample.parquet
data/08_reporting/figures/shap_feature_importance.png
outputs/previsao.csv
```

Open experiment tracking:

```bash
uv run mlflow ui --backend-store-uri mlruns
```
