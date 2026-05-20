"""Old-data usefulness gate nodes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from pads_forecasting.metrics import summarize_cv
from pads_forecasting.modeling import evaluate_cv, model_specs
from pads_forecasting.selection import old_data_gate_decision

FOLD_METRIC_COLUMNS = [
    "mae",
    "rmse",
    "mase",
    "mase_denominator",
    "common_mase",
    "common_mase_denominator",
    "bias",
    "relative_mae_vs_seasonal_naive",
    "train_mae",
    "validation_mae",
    "train_valid_mae_gap",
    "train_valid_mae_ratio",
    "train_residual_mean",
    "train_residual_abs_mean",
    "train_residual_std",
]

SUMMARY_METRIC_COLUMNS = [
    "mean_mae",
    "mean_rmse",
    "mean_mase",
    "normal_mean_mase",
    "mean_common_mase",
    "normal_mean_common_mase",
    "std_mase",
    "cv_mase",
    "max_mase",
    "std_common_mase",
    "cv_common_mase",
    "max_common_mase",
    "mean_bias",
    "mean_relative_mae_vs_seasonal_naive",
    "mean_train_valid_ratio",
    "mean_train_residual_mean",
    "mean_train_residual_abs_mean",
    "mean_train_residual_std",
    "folds",
]


def _old_data_gate_table(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    decision: pd.DataFrame,
    selected_alpha: float | None,
) -> pd.DataFrame:
    """Build a single auditable Stage A table with decision, summary, and fold rows."""

    decision_rows = decision.copy()
    decision_rows["record_type"] = "decision"
    summary_rows = summary.copy()
    summary_rows["record_type"] = "summary"
    fold_rows = fold_results.copy()
    fold_rows["record_type"] = "fold"
    fold_rows["passed"] = pd.NA
    fold_rows["decision"] = pd.NA
    out = pd.concat([decision_rows, summary_rows, fold_rows], ignore_index=True, sort=False)
    out["selected_alpha"] = selected_alpha
    return out


def _log_old_data_gate_to_mlflow(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    decision: pd.DataFrame,
    selected_alpha: float | None,
    gate_table: pd.DataFrame | None = None,
    selection_params: dict[str, Any] | None = None,
) -> None:
    """Log Stage A fold runs and gate decisions under the active kedro-mlflow run."""

    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        if gate_table is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "old_data_gate.parquet"
                gate_table.to_parquet(path, index=False)
                mlflow.log_artifact(str(path))

        if selection_params:
            for key, value in selection_params.items():
                mlflow.log_param(f"old_data_gate.selection.{key}", str(value))

        for _, row in decision.iterrows():
            prefix = f"old_data_gate.{row['target_strategy']}"
            mlflow.log_metric(
                f"{prefix}.passed",
                float(str(row.get("passed", "")).lower() in {"true", "1"}),
            )
            for metric in [
                "improvement_vs_post_only_pct",
                "normal_mean_mase",
                "normal_mean_common_mase",
                "cv_mase",
                "cv_common_mase",
                "train_valid_ratio",
                "residual_ratio_vs_post_only",
            ]:
                if metric in row and pd.notna(row[metric]):
                    mlflow.log_metric(f"{prefix}.{metric}", float(row[metric]))

        for _, row in summary.iterrows():
            run_name = f"old_data_gate/{row['target_strategy']}/{row['model_id']}/summary"
            with mlflow.start_run(run_name=run_name, nested=True):
                _log_row_params(
                    row,
                    _param_columns(
                        row,
                        [
                            "stage",
                            "target_strategy",
                            "model_id",
                            "model_family",
                            "model_params",
                            "covid_mode",
                            "complexity",
                        ],
                    ),
                    selected_alpha,
                )
                _log_row_metrics(row, SUMMARY_METRIC_COLUMNS)

        for _, row in fold_results.iterrows():
            run_name = (
                f"old_data_gate/{row['target_strategy']}/{row['model_id']}/{row['fold_name']}"
            )
            with mlflow.start_run(run_name=run_name, nested=True):
                _log_row_params(
                    row,
                    _param_columns(
                        row,
                        [
                            "stage",
                            "target_strategy",
                            "model_id",
                            "model_family",
                            "model_params",
                            "covid_mode",
                            "complexity",
                            "fold_name",
                            "fold_role",
                            "train_end",
                            "valid_start",
                            "valid_end",
                            "horizon",
                            "status",
                            "alpha",
                            "beta",
                            "alpha_selection_method",
                            "alpha_inner_fold_count",
                        ],
                    ),
                    selected_alpha,
                )
                _log_row_metrics(row, FOLD_METRIC_COLUMNS)
    except Exception:
        return


def _log_row_params(row: pd.Series, columns: list[str], selected_alpha: float | None) -> None:
    import mlflow

    for column in columns:
        if column in row and pd.notna(row[column]):
            mlflow.log_param(column, str(row[column]))
    if selected_alpha is not None:
        mlflow.log_param("selected_alpha", str(selected_alpha))


def _param_columns(row: pd.Series, columns: list[str]) -> list[str]:
    """Include explicit params plus flattened model hyperparameter columns."""

    return [*columns, *sorted(col for col in row.index if col.startswith("model_param_"))]


def _log_row_metrics(row: pd.Series, columns: list[str]) -> None:
    import mlflow

    for column in columns:
        if column in row and pd.notna(row[column]):
            mlflow.log_metric(column, float(row[column]))


def run_old_data_gate(
    target_strategies: dict[str, Any],
    folds_metadata: pd.DataFrame,
    project: dict[str, Any],
    validation: dict[str, Any],
    models: dict[str, Any],
    selection: dict[str, Any],
) -> pd.DataFrame:
    """Run Stage A: prove or disprove usefulness of old pre-merger data."""

    specs = model_specs(models, stage="old_data_gate", include_optional=False)
    fold_results = evaluate_cv(
        stage="old_data_gate",
        target_strategies=target_strategies,
        strategy_names=["post_only", "raw_full", "proforma_sum", "calibrated_alpha"],
        specs=specs,
        folds_metadata=folds_metadata,
        validation_params=validation,
        project_params=project,
    )
    ok = fold_results[fold_results["status"].eq("ok")].copy()
    summary = summarize_cv(ok) if not ok.empty else pd.DataFrame()
    decision = old_data_gate_decision(summary, selection)
    selected_alpha = target_strategies.get("selected_alpha")
    out = _old_data_gate_table(fold_results, summary, decision, selected_alpha)
    _log_old_data_gate_to_mlflow(fold_results, summary, decision, selected_alpha, out, selection)
    return out
