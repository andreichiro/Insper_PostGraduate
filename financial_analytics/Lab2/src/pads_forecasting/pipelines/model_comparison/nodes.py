"""Full model-comparison nodes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pads_forecasting.metrics import (
    bootstrap_mase_uncertainty,
    summarize_cv,
    summarize_horizon_metrics,
)
from pads_forecasting.modeling import evaluate_cv, model_specs
from pads_forecasting.pipelines.validation.nodes import build_folds_metadata
from pads_forecasting.selection import admissible_strategies, select_final_model

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
    "std_mae",
    "cv_mae",
    "max_mae",
    "std_rmse",
    "cv_rmse",
    "max_rmse",
    "std_mase",
    "cv_mase",
    "max_mase",
    "std_common_mase",
    "cv_common_mase",
    "max_common_mase",
    "mean_bias",
    "mean_relative_mae_vs_seasonal_naive",
    "std_relative_mae_vs_seasonal_naive",
    "cv_relative_mae_vs_seasonal_naive",
    "mean_train_valid_ratio",
    "mean_train_residual_mean",
    "mean_train_residual_abs_mean",
    "mean_train_residual_std",
    "folds",
]

SELECTION_METRIC_COLUMNS = [
    "mean_mae",
    "mean_rmse",
    "mean_mase",
    "normal_mean_mase",
    "mean_common_mase",
    "normal_mean_common_mase",
    "cv_mae",
    "cv_rmse",
    "std_common_mase",
    "cv_common_mase",
    "max_common_mase",
    "std_mase",
    "cv_mase",
    "max_mase",
    "mean_bias",
    "mean_relative_mae_vs_seasonal_naive",
    "mean_train_valid_ratio",
    "rank",
]

TRAIN_VALID_GAP_COLUMNS = [
    "stage",
    "target_strategy",
    "model_id",
    "model_family",
    "model_params",
    "covid_mode",
    "complexity",
    "alpha",
    "beta",
    "fold_name",
    "fold_role",
    "train_end",
    "valid_start",
    "valid_end",
    "horizon",
    "status",
    "train_mae",
    "validation_mae",
    "train_valid_mae_gap",
    "train_valid_mae_ratio",
]

MLFLOW_MAX_SUMMARY_NESTED_RUNS = 30
MLFLOW_MAX_FOLD_NESTED_CANDIDATES = 10

ARTIFACT_NAMES = {
    "fold_results": "cv_fold_results.parquet",
    "summary": "cv_summary.parquet",
    "train_valid_gap": "train_valid_gap.parquet",
    "model_selection": "model_selection.parquet",
    "horizon_metrics": "horizon_metrics.parquet",
    "horizon_summary": "horizon_summary.parquet",
    "mase_uncertainty": "mase_uncertainty.parquet",
    "nested_selection_audit": "nested_selection_audit.parquet",
    "nested_cv_results": "nested_cv_results.parquet",
    "nested_cv_summary": "nested_cv_summary.parquet",
    "rolling_origin_robustness": "rolling_origin_robustness.parquet",
    "robust_alpha_results": "robust_alpha_results.parquet",
    "robust_alpha_summary": "robust_alpha_summary.parquet",
    "selection_objective_audit": "selection_objective_audit.parquet",
    "covid_adjustment_coefficients": "covid_adjustment_coefficients.parquet",
    "covid_adjustment_audit": "covid_adjustment_audit.parquet",
    "covid_mode_comparison": "covid_mode_comparison.parquet",
}


def _param_columns(row: pd.Series, columns: list[str]) -> list[str]:
    """Include explicit params plus flattened model hyperparameter columns."""

    return [*columns, *sorted(col for col in row.index if col.startswith("model_param_"))]


def _log_row_params(row: pd.Series, columns: list[str]) -> None:
    import mlflow

    for column in columns:
        if column in row and pd.notna(row[column]):
            mlflow.log_param(column, str(row[column]))


def _log_row_metrics(row: pd.Series, columns: list[str]) -> None:
    import mlflow

    for column in columns:
        if column in row and pd.notna(row[column]):
            mlflow.log_metric(column, float(row[column]))


def _log_model_comparison_artifacts(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    train_valid_gap: pd.DataFrame,
    model_selection: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    mase_uncertainty: pd.DataFrame,
    nested_selection_audit: pd.DataFrame,
    nested_cv_results: pd.DataFrame,
    nested_cv_summary: pd.DataFrame,
    rolling_origin_robustness: pd.DataFrame,
    robust_alpha_results: pd.DataFrame,
    robust_alpha_summary: pd.DataFrame,
    selection_objective_audit: pd.DataFrame,
    covid_adjustment_coefficients: pd.DataFrame,
    covid_adjustment_audit: pd.DataFrame,
    covid_mode_comparison: pd.DataFrame,
) -> None:
    import mlflow

    artifacts = {
        ARTIFACT_NAMES["fold_results"]: fold_results,
        ARTIFACT_NAMES["summary"]: summary,
        ARTIFACT_NAMES["train_valid_gap"]: train_valid_gap,
        ARTIFACT_NAMES["model_selection"]: model_selection,
        ARTIFACT_NAMES["horizon_metrics"]: horizon_metrics,
        ARTIFACT_NAMES["horizon_summary"]: horizon_summary,
        ARTIFACT_NAMES["mase_uncertainty"]: mase_uncertainty,
        ARTIFACT_NAMES["nested_selection_audit"]: nested_selection_audit,
        ARTIFACT_NAMES["nested_cv_results"]: nested_cv_results,
        ARTIFACT_NAMES["nested_cv_summary"]: nested_cv_summary,
        ARTIFACT_NAMES["rolling_origin_robustness"]: rolling_origin_robustness,
        ARTIFACT_NAMES["robust_alpha_results"]: robust_alpha_results,
        ARTIFACT_NAMES["robust_alpha_summary"]: robust_alpha_summary,
        ARTIFACT_NAMES["selection_objective_audit"]: selection_objective_audit,
        ARTIFACT_NAMES["covid_adjustment_coefficients"]: covid_adjustment_coefficients,
        ARTIFACT_NAMES["covid_adjustment_audit"]: covid_adjustment_audit,
        ARTIFACT_NAMES["covid_mode_comparison"]: covid_mode_comparison,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, table in artifacts.items():
            path = Path(tmpdir) / name
            table.to_parquet(path, index=False)
            mlflow.log_artifact(str(path))


def _log_model_comparison_to_mlflow(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    train_valid_gap: pd.DataFrame,
    model_selection: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    mase_uncertainty: pd.DataFrame,
    nested_selection_audit: pd.DataFrame,
    nested_cv_results: pd.DataFrame,
    nested_cv_summary: pd.DataFrame,
    rolling_origin_robustness: pd.DataFrame,
    robust_alpha_results: pd.DataFrame,
    robust_alpha_summary: pd.DataFrame,
    selection_objective_audit: pd.DataFrame,
    covid_adjustment_coefficients: pd.DataFrame,
    covid_adjustment_audit: pd.DataFrame,
    covid_mode_comparison: pd.DataFrame,
    strategies: list[str],
    selection_params: dict[str, Any],
) -> None:
    """Log Stage B candidate folds, summaries, ranking, and artifacts."""

    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        _log_model_comparison_artifacts(
            fold_results,
            summary,
            train_valid_gap,
            model_selection,
            horizon_metrics,
            horizon_summary,
            mase_uncertainty,
            nested_selection_audit,
            nested_cv_results,
            nested_cv_summary,
            rolling_origin_robustness,
            robust_alpha_results,
            robust_alpha_summary,
            selection_objective_audit,
            covid_adjustment_coefficients,
            covid_adjustment_audit,
            covid_mode_comparison,
        )

        mlflow.log_param("model_comparison.admissible_strategies", ",".join(strategies))
        mlflow.log_param("model_comparison.candidate_rows", str(len(fold_results)))
        mlflow.log_param("model_comparison.summary_rows", str(len(summary)))
        for key, value in selection_params.items():
            mlflow.log_param(f"model_comparison.selection.{key}", str(value))

        selected = (
            model_selection[model_selection["selected"].astype(str).str.lower().isin(["true", "1"])]
            if "selected" in model_selection
            else pd.DataFrame()
        )
        if not selected.empty:
            selected_row = selected.iloc[0]
            mlflow.log_param(
                "model_comparison.selected.target_strategy", selected_row["target_strategy"]
            )
            mlflow.log_param("model_comparison.selected.model_id", selected_row["model_id"])
            for metric in SELECTION_METRIC_COLUMNS:
                if metric in selected_row and pd.notna(selected_row[metric]):
                    mlflow.log_metric(
                        f"model_comparison.selected.{metric}", float(selected_row[metric])
                    )

        summary_for_logging = _compact_summary_rows_for_mlflow(summary, model_selection)
        fold_results_for_logging = _compact_fold_rows_for_mlflow(
            fold_results,
            summary_for_logging,
        )
        mlflow.log_param("model_comparison.mlflow_logging_mode", "compact_full_csv_artifacts")
        mlflow.log_param(
            "model_comparison.summary_nested_rows_logged",
            str(len(summary_for_logging)),
        )
        mlflow.log_param(
            "model_comparison.fold_nested_rows_logged",
            str(len(fold_results_for_logging)),
        )
        mlflow.log_param(
            "model_comparison.fold_rows_available",
            str(len(fold_results)),
        )

        for _, row in summary_for_logging.iterrows():
            run_name = f"model_comparison/{row['target_strategy']}/{row['model_id']}/summary"
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
                )
                _log_row_metrics(row, SUMMARY_METRIC_COLUMNS)

        for _, row in fold_results_for_logging.iterrows():
            run_name = (
                f"model_comparison/{row['target_strategy']}/{row['model_id']}/{row['fold_name']}"
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
                        ],
                    ),
                )
                _log_row_metrics(row, FOLD_METRIC_COLUMNS)
    except Exception:
        return


def _sort_candidates_for_mlflow(df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates for compact MLflow nested logging.

    The complete candidate grid is preserved in CSV artifacts. MLflow file-store nested
    runs are intentionally capped to avoid hundreds of thousands of tiny param/metric
    files during full production runs.
    """

    if df.empty:
        return df.copy()
    order = [
        col
        for col in [
            "selected_sort",
            "rank",
            "normal_mean_common_mase",
            "mean_common_mase",
            "normal_mean_mase",
            "mean_mase",
        ]
        if col in df.columns
    ]
    ascending = [False if col == "selected_sort" else True for col in order]
    return df.sort_values(order, ascending=ascending, kind="mergesort") if order else df.copy()


def _compact_summary_rows_for_mlflow(
    summary: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Keep selected/top candidate summaries as nested MLflow runs."""

    if summary.empty:
        return summary.copy()
    out = summary.copy()
    out["selected_sort"] = False
    if not model_selection.empty and {"target_strategy", "model_id", "selected"}.issubset(
        model_selection.columns
    ):
        selected_keys = set(
            model_selection[
                model_selection["selected"].astype(str).str.lower().isin(["true", "1"])
            ][["target_strategy", "model_id"]].itertuples(index=False, name=None)
        )
        out["selected_sort"] = [
            (strategy, model_id) in selected_keys
            for strategy, model_id in out[["target_strategy", "model_id"]].itertuples(
                index=False,
                name=None,
            )
        ]
    compact = _sort_candidates_for_mlflow(out).head(MLFLOW_MAX_SUMMARY_NESTED_RUNS)
    return compact.drop(columns=["selected_sort"], errors="ignore").copy()


def _compact_fold_rows_for_mlflow(
    fold_results: pd.DataFrame,
    summary_for_logging: pd.DataFrame,
) -> pd.DataFrame:
    """Keep fold-level nested runs for selected/top logged candidates."""

    if fold_results.empty or summary_for_logging.empty:
        return pd.DataFrame(columns=fold_results.columns)
    candidate_keys = set(
        summary_for_logging[["target_strategy", "model_id"]]
        .head(MLFLOW_MAX_FOLD_NESTED_CANDIDATES)
        .itertuples(index=False, name=None)
    )
    mask = [
        (strategy, model_id) in candidate_keys
        for strategy, model_id in fold_results[["target_strategy", "model_id"]].itertuples(
            index=False,
            name=None,
        )
    ]
    return fold_results.loc[mask].copy()


def _candidate_key(row: pd.Series) -> tuple[str, str]:
    return str(row["target_strategy"]), str(row["model_id"])


def _candidate_columns(df: pd.DataFrame) -> list[str]:
    """Columns that define one model/hyperparameter candidate."""

    base_cols = [
        "target_strategy",
        "model_id",
        "model_family",
        "model_params",
        "covid_mode",
        "complexity",
    ]
    param_cols = sorted(col for col in df.columns if col.startswith("model_param_"))
    return [col for col in [*base_cols, *param_cols] if col in df.columns]


def _value_signature(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _add_candidate_signature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = _candidate_columns(out)
    out["candidate_signature"] = out[cols].apply(
        lambda row: "||".join(_value_signature(row[col]) for col in cols),
        axis=1,
    )
    return out


def _build_nested_cv_results(
    fold_results: pd.DataFrame,
    *,
    primary_metric: str = "common_mase",
) -> pd.DataFrame:
    """Run formal temporal nested selection over the full candidate grid.

    For each outer fold, all model families, target strategies, and hyperparameter
    candidates are ranked using only prior validation folds. The winner is then
    evaluated on the untouched outer fold already present in `fold_results`.
    """

    if fold_results.empty or primary_metric not in fold_results:
        return pd.DataFrame()
    ok = fold_results.copy()
    if "status" in ok:
        ok = ok[ok["status"].eq("ok")]
    if ok.empty:
        return pd.DataFrame()
    ok = _add_candidate_signature(ok)
    ok["valid_start_ts"] = pd.to_datetime(ok["valid_start"])
    ok["valid_end_ts"] = pd.to_datetime(ok["valid_end"])
    folds = (
        ok[["fold_name", "fold_role", "valid_start_ts"]]
        .drop_duplicates()
        .sort_values("valid_start_ts")
    )
    candidate_cols = _candidate_columns(ok)
    rows = []
    for _, outer in folds.iterrows():
        outer_fold = outer["fold_name"]
        inner = ok[ok["valid_end_ts"] < outer["valid_start_ts"]].copy()
        outer_rows = ok[ok["fold_name"].eq(outer_fold)].copy()
        if inner.empty:
            rows.append(
                {
                    "selection_scope": "global_full_candidate_grid",
                    "fold_name": outer_fold,
                    "fold_role": outer["fold_role"],
                    "status": "skipped_no_prior_inner_fold",
                    "inner_fold_count": 0,
                    "inner_candidate_count": int(ok["candidate_signature"].nunique()),
                }
            )
            continue

        candidate_scores = (
            inner.groupby("candidate_signature", dropna=False)
            .agg(
                inner_mean_common_mase=(primary_metric, "mean"),
                inner_std_common_mase=(primary_metric, lambda x: float(x.std(ddof=0))),
                inner_mean_mae=("mae", "mean"),
                inner_fold_count=("fold_name", "nunique"),
            )
            .reset_index()
        )
        representative = (
            inner.sort_values(["candidate_signature", "valid_start_ts"])
            .groupby("candidate_signature", as_index=False)
            .tail(1)[["candidate_signature", *candidate_cols]]
        )
        candidate_scores = candidate_scores.merge(
            representative,
            on="candidate_signature",
            how="left",
        )
        winner = candidate_scores.sort_values(
            [
                "inner_mean_common_mase",
                "inner_std_common_mase",
                "inner_mean_mae",
                "candidate_signature",
            ]
        ).iloc[0]
        outer_match = outer_rows[
            outer_rows["candidate_signature"].eq(winner["candidate_signature"])
        ]
        if outer_match.empty:
            rows.append(
                {
                    "selection_scope": "global_full_candidate_grid",
                    "fold_name": outer_fold,
                    "fold_role": outer["fold_role"],
                    "status": "selected_candidate_missing_on_outer_fold",
                    "inner_fold_count": int(inner["fold_name"].nunique()),
                    "inner_candidate_count": int(candidate_scores["candidate_signature"].nunique()),
                    **{col: winner.get(col, np.nan) for col in candidate_cols},
                }
            )
            continue
        outer_row = outer_match.iloc[0]
        rows.append(
            {
                "selection_scope": "global_full_candidate_grid",
                "fold_name": outer_fold,
                "fold_role": outer["fold_role"],
                "status": "ok",
                "inner_fold_count": int(inner["fold_name"].nunique()),
                "inner_candidate_count": int(candidate_scores["candidate_signature"].nunique()),
                "candidate_signature": winner["candidate_signature"],
                **{col: outer_row.get(col, winner.get(col, np.nan)) for col in candidate_cols},
                "alpha": outer_row.get("alpha", np.nan),
                "alpha_selection_method": outer_row.get("alpha_selection_method", ""),
                "alpha_inner_fold_count": outer_row.get("alpha_inner_fold_count", np.nan),
                "inner_mean_common_mase": float(winner["inner_mean_common_mase"]),
                "inner_std_common_mase": float(winner["inner_std_common_mase"]),
                "inner_mean_mae": float(winner["inner_mean_mae"]),
                "outer_common_mase": float(outer_row.get("common_mase", np.nan)),
                "outer_local_mase": float(outer_row.get("mase", np.nan)),
                "outer_mae": float(outer_row.get("mae", np.nan)),
                "outer_rmse": float(outer_row.get("rmse", np.nan)),
                "outer_bias": float(outer_row.get("bias", np.nan)),
                "outer_relative_mae_vs_seasonal_naive": float(
                    outer_row.get("relative_mae_vs_seasonal_naive", np.nan)
                ),
                "outer_train_valid_mae_ratio": float(
                    outer_row.get("train_valid_mae_ratio", np.nan)
                ),
                "common_mase_reference_strategy": outer_row.get(
                    "common_mase_reference_strategy",
                    "",
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_nested_cv(nested_cv_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize untouched outer-fold performance from nested selection."""

    if nested_cv_results.empty:
        return pd.DataFrame()
    ok = nested_cv_results[nested_cv_results["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = [
        {
            "summary_scope": "global_full_candidate_grid",
            "nested_outer_folds": int(ok["fold_name"].nunique()),
            "nested_mean_common_mase": float(ok["outer_common_mase"].mean()),
            "nested_mean_local_mase": float(ok["outer_local_mase"].mean()),
            "nested_mean_mae": float(ok["outer_mae"].mean()),
            "nested_mean_rmse": float(ok["outer_rmse"].mean()),
            "nested_mean_bias": float(ok["outer_bias"].mean()),
            "nested_mean_relative_mae_vs_seasonal_naive": float(
                ok["outer_relative_mae_vs_seasonal_naive"].mean()
            ),
            "nested_mean_train_valid_ratio": float(ok["outer_train_valid_mae_ratio"].mean()),
            "selected_candidate_count": int(ok["candidate_signature"].nunique()),
        }
    ]
    by_candidate = (
        ok.groupby(
            [
                "target_strategy",
                "model_id",
                "model_family",
                "model_params",
                "covid_mode",
                "complexity",
            ],
            dropna=False,
        )
        .agg(
            nested_outer_folds=("fold_name", "nunique"),
            nested_mean_common_mase=("outer_common_mase", "mean"),
            nested_mean_local_mase=("outer_local_mase", "mean"),
            nested_mean_mae=("outer_mae", "mean"),
            nested_mean_rmse=("outer_rmse", "mean"),
            nested_selection_count=("fold_name", "count"),
        )
        .reset_index()
        .sort_values(["nested_selection_count", "nested_mean_common_mase"], ascending=[False, True])
    )
    for _, row in by_candidate.iterrows():
        rows.append(
            {
                "summary_scope": "selected_candidate",
                **row.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _build_nested_selection_audit(
    fold_results: pd.DataFrame,
    *,
    primary_metric: str = "common_mase",
) -> pd.DataFrame:
    """Audit inner-fold-only model selection on untouched outer folds."""

    if fold_results.empty or primary_metric not in fold_results:
        return pd.DataFrame()
    ok = fold_results.copy()
    if "status" in ok:
        ok = ok[ok["status"].eq("ok")]
    if ok.empty:
        return pd.DataFrame()
    ok["valid_start_ts"] = pd.to_datetime(ok["valid_start"])
    ok["valid_end_ts"] = pd.to_datetime(ok["valid_end"])
    folds = (
        ok[["fold_name", "fold_role", "valid_start_ts"]]
        .drop_duplicates()
        .sort_values("valid_start_ts")
    )
    rows = []
    for _, outer in folds.iterrows():
        outer_fold = outer["fold_name"]
        inner = ok[ok["valid_end_ts"] < outer["valid_start_ts"]].copy()
        outer_rows = ok[ok["fold_name"].eq(outer_fold)].copy()
        if inner.empty:
            rows.append(
                {
                    "selection_scope": "global",
                    "fold_name": outer_fold,
                    "fold_role": outer["fold_role"],
                    "status": "skipped_no_prior_inner_fold",
                    "inner_fold_count": 0,
                }
            )
            continue
        scopes: list[tuple[str, list[str]]] = [
            ("global", []),
            ("target_strategy_model_family", ["target_strategy", "model_family"]),
        ]
        for scope_name, group_cols in scopes:
            grouped = inner.groupby(group_cols, dropna=False) if group_cols else [((), inner)]
            for group_key, group in grouped:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                selector = (
                    group.groupby(["target_strategy", "model_id"], dropna=False)[primary_metric]
                    .mean()
                    .reset_index(name="inner_mean_common_mase")
                    .sort_values("inner_mean_common_mase")
                    .iloc[0]
                )
                outer_match = outer_rows[
                    outer_rows["target_strategy"].eq(selector["target_strategy"])
                    & outer_rows["model_id"].eq(selector["model_id"])
                ]
                if outer_match.empty:
                    continue
                outer_row = outer_match.iloc[0]
                row = {
                    "selection_scope": scope_name,
                    "fold_name": outer_fold,
                    "fold_role": outer["fold_role"],
                    "status": "ok",
                    "target_strategy": selector["target_strategy"],
                    "model_id": selector["model_id"],
                    "model_family": outer_row.get("model_family"),
                    "alpha": outer_row.get("alpha", np.nan),
                    "alpha_selection_method": outer_row.get("alpha_selection_method", ""),
                    "inner_fold_count": int(inner["fold_name"].nunique()),
                    "inner_mean_common_mase": float(selector["inner_mean_common_mase"]),
                    "outer_common_mase": float(outer_row.get("common_mase", np.nan)),
                    "outer_local_mase": float(outer_row.get("mase", np.nan)),
                    "outer_mae": float(outer_row.get("mae", np.nan)),
                }
                for column, value in zip(group_cols, group_key, strict=False):
                    row[f"scope_{column}"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _build_robustness_folds(validation: dict[str, Any]) -> pd.DataFrame:
    """Generate extra rolling-origin folds for top-candidate robustness."""

    config = validation.get("robustness_rolling_origins", {})
    if not config.get("enabled", False):
        return pd.DataFrame()
    step_months = int(config.get("step_months", 3))
    train_end = pd.Timestamp(config["first_train_end"])
    last_train_end = pd.Timestamp(config["last_train_end"])
    normal_start = pd.Timestamp(config.get("normal_valid_start", "2022-01-01"))
    rows = []
    while train_end <= last_train_end:
        valid_start = train_end + pd.DateOffset(months=1)
        valid_end = valid_start + pd.DateOffset(months=int(validation["horizon"]) - 1)
        role = "normal" if valid_start >= normal_start else "stress"
        rows.append(
            {
                "name": f"fold_{train_end.strftime('%Y_%m')}_rolling",
                "train_end": train_end.strftime("%Y-%m-%d"),
                "valid_start": valid_start.strftime("%Y-%m-%d"),
                "valid_end": valid_end.strftime("%Y-%m-%d"),
                "role": role,
            }
        )
        train_end = train_end + pd.DateOffset(months=step_months)
    return build_folds_metadata({**validation, "folds": rows}) if rows else pd.DataFrame()


def _build_rolling_origin_robustness(
    *,
    target_strategies: dict[str, Any],
    model_selection: pd.DataFrame,
    validation: dict[str, Any],
    project: dict[str, Any],
    models: dict[str, Any],
) -> pd.DataFrame:
    """Re-evaluate selected/top challenger pairs on additional rolling origins."""

    robustness_folds = _build_robustness_folds(validation)
    if robustness_folds.empty or model_selection.empty:
        return pd.DataFrame()
    ranked = model_selection.copy()
    if "eligible_for_selection" in ranked:
        eligible = ranked[ranked["eligible_for_selection"].astype(bool)]
        if not eligible.empty:
            ranked = eligible
    ranked = ranked.sort_values(
        ["rank", "normal_mean_common_mase", "normal_mean_mase"],
        na_position="last",
    ).head(int(validation.get("robustness_top_n", 3)))
    if ranked.empty:
        return pd.DataFrame()
    spec_by_id = {
        spec["model_id"]: spec
        for spec in model_specs(models, stage="model_comparison", include_optional=True)
    }
    specs = [
        spec_by_id[str(model_id)] for model_id in ranked["model_id"] if str(model_id) in spec_by_id
    ]
    strategies = list(dict.fromkeys(ranked["target_strategy"].astype(str).tolist()))
    if not specs or not strategies:
        return pd.DataFrame()
    evaluated = evaluate_cv(
        stage="rolling_origin_robustness",
        target_strategies=target_strategies,
        strategy_names=strategies,
        specs=specs,
        folds_metadata=robustness_folds,
        validation_params=validation,
        project_params=project,
    )
    candidate_pairs = {_candidate_key(row) for _, row in ranked.iterrows()}
    evaluated = evaluated[
        evaluated.apply(lambda row: _candidate_key(row) in candidate_pairs, axis=1)
    ].copy()
    if evaluated.empty:
        return pd.DataFrame()
    ok = evaluated[evaluated["status"].eq("ok")]
    if ok.empty:
        return evaluated
    summary = summarize_cv(ok)
    summary["robustness_fold_count"] = int(robustness_folds["fold_name"].nunique())
    summary["robustness_step_months"] = int(
        validation.get("robustness_rolling_origins", {}).get("step_months", 3)
    )
    return summary


def _apply_rolling_origin_selection(
    model_selection: pd.DataFrame,
    rolling_origin_robustness: pd.DataFrame,
) -> pd.DataFrame:
    """Attach rolling-origin diagnostics and mark the robustness-screened winner."""

    if model_selection.empty or rolling_origin_robustness.empty:
        return model_selection
    out = model_selection.copy()
    robust_cols = [
        "target_strategy",
        "model_id",
        "normal_mean_common_mase",
        "mean_common_mase",
        "cv_common_mase",
        "robustness_fold_count",
        "robustness_step_months",
    ]
    robust = rolling_origin_robustness[
        [col for col in robust_cols if col in rolling_origin_robustness]
    ].copy()
    if robust.empty or "normal_mean_common_mase" not in robust:
        return out
    robust = robust.rename(
        columns={
            "normal_mean_common_mase": "robustness_normal_mean_common_mase",
            "mean_common_mase": "robustness_mean_common_mase",
            "cv_common_mase": "robustness_cv_common_mase",
        }
    )
    out = out.merge(robust, on=["target_strategy", "model_id"], how="left")
    if "selected_with_robustness" not in out:
        out["selected_with_robustness"] = False
    out["robustness_selection_reason"] = "not robustness-selected"
    evaluated = out[
        out["eligible_for_selection"].astype(bool)
        & out.get("stability_passed", pd.Series(True, index=out.index)).astype(bool)
        & out["robustness_normal_mean_common_mase"].notna()
    ].copy()
    if evaluated.empty:
        return out
    evaluated = evaluated.sort_values(
        [
            "robustness_normal_mean_common_mase",
            "normal_mean_common_mase",
            "cv_common_mase",
        ],
        na_position="last",
    )
    out["robustness_rank"] = np.nan
    out.loc[evaluated.index, "robustness_rank"] = np.arange(1, len(evaluated) + 1)
    robust_selected_idx = evaluated.index[0]
    out.loc[robust_selected_idx, "selected_with_robustness"] = True
    out.loc[robust_selected_idx, "robustness_selection_reason"] = (
        "selected by rolling-origin robustness diagnostic: lowest expanded "
        "fixed-target MASE among candidates that pass complete-CV, baseline, "
        "overfit, primary-COVID, and data-driven fold-variance gates"
    )
    return out.sort_values(
        ["selected", "selected_with_robustness", "rank", "robustness_rank"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _alpha_strategy_name(alpha: float) -> str:
    token = f"{float(alpha):.4g}".replace("-", "m").replace(".", "_")
    return f"alpha_{token}"


def _alpha_from_strategy_name(name: str) -> float:
    return float(name.removeprefix("alpha_").replace("m", "-").replace("_", "."))


def _top_specs_by_family_for_alpha(
    summary: pd.DataFrame,
    models: dict[str, Any],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Select a small, auditable model grid for alpha-as-hyperparameter checks."""

    all_specs = model_specs(models, stage="model_comparison", include_optional=True)
    spec_by_id = {spec["model_id"]: spec for spec in all_specs}
    if summary.empty or "model_id" not in summary:
        seen = set()
        fallback = []
        for spec in all_specs:
            if spec["family"] in seen:
                continue
            fallback.append(spec)
            seen.add(spec["family"])
        return fallback

    metric = (
        "normal_mean_common_mase"
        if "normal_mean_common_mase" in summary.columns
        else "normal_mean_mase"
    )
    ranked = summary.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.dropna(subset=[metric])
    if ranked.empty:
        return []
    ranked = ranked.sort_values(
        ["model_family", metric, "mean_common_mase", "mean_mae"],
        na_position="last",
    )
    selected_ids = (
        ranked.groupby("model_family", as_index=False, group_keys=False)
        .head(top_n)["model_id"]
        .astype(str)
        .tolist()
    )
    return [spec_by_id[model_id] for model_id in selected_ids if model_id in spec_by_id]


def _alpha_horizon_delta(
    horizon_metrics: pd.DataFrame,
    *,
    fold_name: str,
    model_id: str,
    selected_alpha: float,
) -> tuple[float, float]:
    """Measure whether alpha changes the forecast path versus alpha=1."""

    if horizon_metrics.empty:
        return np.nan, np.nan
    needed = {"fold_name", "model_id", "candidate_alpha", "horizon_index", "yhat"}
    if not needed.issubset(horizon_metrics.columns):
        return np.nan, np.nan
    selected = horizon_metrics[
        horizon_metrics["fold_name"].eq(fold_name)
        & horizon_metrics["model_id"].eq(model_id)
        & np.isclose(horizon_metrics["candidate_alpha"].astype(float), selected_alpha)
    ]
    alpha_one = horizon_metrics[
        horizon_metrics["fold_name"].eq(fold_name)
        & horizon_metrics["model_id"].eq(model_id)
        & np.isclose(horizon_metrics["candidate_alpha"].astype(float), 1.0)
    ]
    if selected.empty or alpha_one.empty:
        return np.nan, np.nan
    merged = selected.merge(
        alpha_one[["horizon_index", "yhat"]].rename(columns={"yhat": "yhat_alpha_one"}),
        on="horizon_index",
        how="inner",
    )
    if merged.empty:
        return np.nan, np.nan
    diff = (merged["yhat"].astype(float) - merged["yhat_alpha_one"].astype(float)).abs()
    scale = merged["yhat_alpha_one"].astype(float).abs().replace(0, np.nan)
    return float(diff.mean()), float((diff / scale).mean())


def _best_alpha_from_scores(
    fold_results: pd.DataFrame,
    *,
    model_family: str,
    shrinkage_lambda: float,
    normal_only: bool = False,
    stress_weight: float | None = None,
) -> float:
    subset = fold_results[fold_results["model_family"].eq(model_family)].copy()
    if normal_only:
        subset = subset[subset["fold_role"].eq("normal")]
    if subset.empty:
        return np.nan
    subset["weight"] = 1.0
    if stress_weight is not None:
        subset.loc[subset["fold_role"].eq("stress"), "weight"] = float(stress_weight)
    grouped = (
        subset.groupby("candidate_alpha", dropna=False)
        .apply(
            lambda group: np.average(
                group["common_mase"].astype(float),
                weights=group["weight"].astype(float),
            ),
            include_groups=False,
        )
        .reset_index(name="weighted_common_mase")
    )
    grouped["regularized_objective"] = (
        grouped["weighted_common_mase"]
        + float(shrinkage_lambda) * (grouped["candidate_alpha"].astype(float) - 1.0) ** 2
    )
    if grouped.empty:
        return np.nan
    return float(
        grouped.sort_values(
            ["regularized_objective", "weighted_common_mase", "candidate_alpha"]
        ).iloc[0]["candidate_alpha"]
    )


def _summarize_robust_alpha(
    robust_alpha_results: pd.DataFrame,
    alpha_fold_results: pd.DataFrame,
    *,
    shrinkage_lambda: float,
    stress_weight: float,
) -> pd.DataFrame:
    if robust_alpha_results.empty:
        return pd.DataFrame()
    ok = robust_alpha_results[robust_alpha_results["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for family, group in ok.groupby("model_family", dropna=False):
        normal = group[group["fold_role"].eq("normal")]
        selected_alphas = group["selected_alpha"].astype(float)
        alpha_counts = selected_alphas.value_counts()
        mode_alpha = float(alpha_counts.index[0]) if not alpha_counts.empty else np.nan
        delta_common = group["outer_common_mase"] - group["alpha_one_outer_common_mase"]
        delta_mae = group["outer_mae"] - group["alpha_one_outer_mae"]
        rows.append(
            {
                "model_family": family,
                "nested_outer_folds": int(group["fold_name"].nunique()),
                "selected_alpha_mode": mode_alpha,
                "selected_alpha_min": float(selected_alphas.min()),
                "selected_alpha_max": float(selected_alphas.max()),
                "selected_alpha_std": float(selected_alphas.std(ddof=0))
                if len(selected_alphas) > 1
                else 0.0,
                "alpha_selection_stable": bool(selected_alphas.nunique() <= 1),
                "mean_outer_common_mase": float(group["outer_common_mase"].mean()),
                "normal_outer_common_mase": float(normal["outer_common_mase"].mean())
                if not normal.empty
                else np.nan,
                "mean_outer_mae": float(group["outer_mae"].mean()),
                "normal_outer_mae": float(normal["outer_mae"].mean())
                if not normal.empty
                else np.nan,
                "mean_common_mase_delta_vs_alpha_one": float(delta_common.mean()),
                "normal_common_mase_delta_vs_alpha_one": float(
                    (normal["outer_common_mase"] - normal["alpha_one_outer_common_mase"]).mean()
                )
                if not normal.empty
                else np.nan,
                "mean_mae_delta_vs_alpha_one": float(delta_mae.mean()),
                "normal_mae_delta_vs_alpha_one": float(
                    (normal["outer_mae"] - normal["alpha_one_outer_mae"]).mean()
                )
                if not normal.empty
                else np.nan,
                "folds_beating_alpha_one_common_mase": int(
                    group["alpha_beats_one_by_common_mase"].sum()
                ),
                "folds_beating_alpha_one_mae": int(group["alpha_beats_one_by_mae"].sum()),
                "best_alpha_all_folds_grid": _best_alpha_from_scores(
                    alpha_fold_results,
                    model_family=str(family),
                    shrinkage_lambda=shrinkage_lambda,
                ),
                "best_alpha_normal_folds_only_grid": _best_alpha_from_scores(
                    alpha_fold_results,
                    model_family=str(family),
                    shrinkage_lambda=shrinkage_lambda,
                    normal_only=True,
                ),
                "best_alpha_stress_downweighted_grid": _best_alpha_from_scores(
                    alpha_fold_results,
                    model_family=str(family),
                    shrinkage_lambda=shrinkage_lambda,
                    stress_weight=stress_weight,
                ),
                "stress_downweight_used": float(stress_weight),
                "mean_abs_yhat_diff_vs_alpha_one": float(
                    group["mean_abs_yhat_diff_vs_alpha_one"].mean()
                ),
                "mean_abs_yhat_pct_diff_vs_alpha_one": float(
                    group["mean_abs_yhat_pct_diff_vs_alpha_one"].mean()
                ),
                "alpha_objective": "fixed_target_common_mase_plus_quadratic_shrinkage",
                "alpha_shrinkage_lambda": float(shrinkage_lambda),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["normal_outer_common_mase", "mean_outer_common_mase"],
        na_position="last",
    )


def _build_robust_alpha_artifacts(
    *,
    target_strategies: dict[str, Any],
    folds_metadata: pd.DataFrame,
    summary: pd.DataFrame,
    validation: dict[str, Any],
    project: dict[str, Any],
    models: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate alpha as a model-family hyperparameter with fixed-target MASE."""

    alpha_candidates = target_strategies.get("alpha_candidates", {})
    if not alpha_candidates:
        return pd.DataFrame(), pd.DataFrame()
    config = validation.get("robust_alpha", {})
    if config.get("enabled", True) is False:
        return pd.DataFrame(), pd.DataFrame()
    top_n = int(config.get("top_specs_per_family", 1))
    shrinkage_lambda = float(config.get("shrinkage_lambda", 0.0))
    stress_weight = float(config.get("stress_downweight", 0.5))
    specs = _top_specs_by_family_for_alpha(summary, models, top_n=top_n)
    if not specs:
        return pd.DataFrame(), pd.DataFrame()

    alpha_strategies = {"post_only": target_strategies["strategies"]["post_only"]}
    strategy_names = []
    for alpha, frame in sorted(alpha_candidates.items()):
        name = _alpha_strategy_name(float(alpha))
        alpha_strategies[name] = frame.copy()
        strategy_names.append(name)
    evaluated = evaluate_cv(
        stage="robust_alpha",
        target_strategies={**target_strategies, "strategies": alpha_strategies},
        strategy_names=strategy_names,
        specs=specs,
        folds_metadata=folds_metadata,
        validation_params=validation,
        project_params=project,
        collect_horizon_metrics=True,
    )
    if isinstance(evaluated, tuple):
        alpha_fold_results, alpha_horizon_metrics = evaluated
    else:
        alpha_fold_results = evaluated
        alpha_horizon_metrics = pd.DataFrame()
    if alpha_fold_results.empty:
        return pd.DataFrame(), pd.DataFrame()
    alpha_fold_results = alpha_fold_results.copy()
    alpha_fold_results["alpha_strategy_name"] = alpha_fold_results["target_strategy"]
    alpha_fold_results["candidate_alpha"] = alpha_fold_results["alpha_strategy_name"].map(
        _alpha_from_strategy_name
    )
    alpha_fold_results["target_strategy"] = "calibrated_alpha"
    if not alpha_horizon_metrics.empty:
        alpha_horizon_metrics = alpha_horizon_metrics.copy()
        alpha_horizon_metrics["alpha_strategy_name"] = alpha_horizon_metrics["target_strategy"]
        alpha_horizon_metrics["candidate_alpha"] = alpha_horizon_metrics["alpha_strategy_name"].map(
            _alpha_from_strategy_name
        )
        alpha_horizon_metrics["target_strategy"] = "calibrated_alpha"

    ok = alpha_fold_results[alpha_fold_results["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()
    ok["valid_start_ts"] = pd.to_datetime(ok["valid_start"])
    ok["valid_end_ts"] = pd.to_datetime(ok["valid_end"])
    folds = (
        ok[["fold_name", "fold_role", "valid_start_ts"]]
        .drop_duplicates()
        .sort_values("valid_start_ts")
    )
    rows = []
    for _, outer in folds.iterrows():
        outer_fold = outer["fold_name"]
        inner = ok[ok["valid_end_ts"] < outer["valid_start_ts"]].copy()
        outer_rows = ok[ok["fold_name"].eq(outer_fold)].copy()
        for family, family_outer in outer_rows.groupby("model_family", dropna=False):
            family_inner = inner[inner["model_family"].eq(family)].copy()
            if family_inner.empty:
                rows.append(
                    {
                        "selection_scope": "model_family_alpha_grid",
                        "fold_name": outer_fold,
                        "fold_role": outer["fold_role"],
                        "status": "skipped_no_prior_inner_fold",
                        "model_family": family,
                        "inner_fold_count": 0,
                        "inner_candidate_count": int(
                            family_outer[["model_id", "candidate_alpha"]].drop_duplicates().shape[0]
                        ),
                    }
                )
                continue
            candidate_scores = (
                family_inner.groupby(["model_id", "candidate_alpha"], dropna=False)
                .agg(
                    inner_mean_common_mase=("common_mase", "mean"),
                    inner_std_common_mase=("common_mase", lambda x: float(x.std(ddof=0))),
                    inner_mean_mae=("mae", "mean"),
                    inner_fold_count=("fold_name", "nunique"),
                )
                .reset_index()
            )
            candidate_scores["alpha_shrinkage_penalty"] = (
                float(shrinkage_lambda)
                * (candidate_scores["candidate_alpha"].astype(float) - 1.0) ** 2
            )
            candidate_scores["inner_regularized_objective"] = (
                candidate_scores["inner_mean_common_mase"]
                + candidate_scores["alpha_shrinkage_penalty"]
            )
            winner = candidate_scores.sort_values(
                [
                    "inner_regularized_objective",
                    "inner_mean_common_mase",
                    "inner_std_common_mase",
                    "inner_mean_mae",
                    "candidate_alpha",
                ]
            ).iloc[0]
            outer_match = family_outer[
                family_outer["model_id"].eq(winner["model_id"])
                & np.isclose(
                    family_outer["candidate_alpha"].astype(float),
                    float(winner["candidate_alpha"]),
                )
            ]
            alpha_one_match = family_outer[
                family_outer["model_id"].eq(winner["model_id"])
                & np.isclose(family_outer["candidate_alpha"].astype(float), 1.0)
            ]
            if outer_match.empty:
                continue
            outer_row = outer_match.iloc[0]
            alpha_one_row = alpha_one_match.iloc[0] if not alpha_one_match.empty else pd.Series()
            yhat_diff, yhat_pct_diff = _alpha_horizon_delta(
                alpha_horizon_metrics,
                fold_name=str(outer_fold),
                model_id=str(winner["model_id"]),
                selected_alpha=float(winner["candidate_alpha"]),
            )
            rows.append(
                {
                    "selection_scope": "model_family_alpha_grid",
                    "fold_name": outer_fold,
                    "fold_role": outer["fold_role"],
                    "status": "ok",
                    "model_family": family,
                    "model_id": winner["model_id"],
                    "selected_alpha": float(winner["candidate_alpha"]),
                    "alpha_selection_method": (
                        "inner_prior_folds_model_family_grid_fixed_target_mase"
                    ),
                    "alpha_objective": ("fixed_target_common_mase_plus_quadratic_shrinkage"),
                    "alpha_shrinkage_lambda": float(shrinkage_lambda),
                    "alpha_shrinkage_penalty": float(winner["alpha_shrinkage_penalty"]),
                    "inner_fold_count": int(family_inner["fold_name"].nunique()),
                    "inner_candidate_count": int(len(candidate_scores)),
                    "inner_mean_common_mase": float(winner["inner_mean_common_mase"]),
                    "inner_regularized_objective": float(winner["inner_regularized_objective"]),
                    "outer_common_mase": float(outer_row["common_mase"]),
                    "outer_local_mase": float(outer_row["mase"]),
                    "outer_mae": float(outer_row["mae"]),
                    "outer_rmse": float(outer_row["rmse"]),
                    "alpha_one_outer_common_mase": float(alpha_one_row.get("common_mase", np.nan)),
                    "alpha_one_outer_mae": float(alpha_one_row.get("mae", np.nan)),
                    "common_mase_delta_vs_alpha_one": float(
                        outer_row["common_mase"] - alpha_one_row.get("common_mase", np.nan)
                    ),
                    "mae_delta_vs_alpha_one": float(
                        outer_row["mae"] - alpha_one_row.get("mae", np.nan)
                    ),
                    "alpha_beats_one_by_common_mase": bool(
                        outer_row["common_mase"] < alpha_one_row.get("common_mase", np.inf)
                    ),
                    "alpha_beats_one_by_mae": bool(
                        outer_row["mae"] < alpha_one_row.get("mae", np.inf)
                    ),
                    "mean_abs_yhat_diff_vs_alpha_one": yhat_diff,
                    "mean_abs_yhat_pct_diff_vs_alpha_one": yhat_pct_diff,
                    "top_specs_per_family": int(top_n),
                }
            )
    robust_alpha_results = pd.DataFrame(rows)
    robust_alpha_summary = _summarize_robust_alpha(
        robust_alpha_results,
        ok,
        shrinkage_lambda=shrinkage_lambda,
        stress_weight=stress_weight,
    )
    return robust_alpha_results, robust_alpha_summary


def _build_selection_objective_audit(
    summary: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Record the one production objective and the selected candidate."""

    if summary.empty:
        return pd.DataFrame()
    rows = []
    selected = (
        model_selection[model_selection["selected"].astype(str).str.lower().isin(["true", "1"])]
        if not model_selection.empty and "selected" in model_selection
        else pd.DataFrame()
    )
    candidate_pool = (
        model_selection[model_selection["eligible_for_selection"].astype(bool)].copy()
        if not model_selection.empty and "eligible_for_selection" in model_selection
        else summary.copy()
    )
    if candidate_pool.empty:
        candidate_pool = summary.copy()
    metric = "normal_mean_common_mase"
    if metric in candidate_pool:
        ranked = candidate_pool.sort_values([metric, "normal_mean_mase", "mean_common_mase"])
        if not ranked.empty:
            row = ranked.iloc[0]
            rows.append(
                {
                    "objective": "hidden_2024_mase_proxy",
                    "objective_metric": metric,
                    "objective_description": (
                        "Single production objective: minimize expected MASE on the hidden "
                        "12 months of 2024 using the closest no-leakage post-merger CV proxy."
                    ),
                    "candidate_role": "best_for_objective",
                    "target_strategy": row.get("target_strategy", ""),
                    "model_family": row.get("model_family", ""),
                    "model_id": row.get("model_id", ""),
                    "normal_mean_common_mase": row.get("normal_mean_common_mase", np.nan),
                    "mean_common_mase": row.get("mean_common_mase", np.nan),
                    "normal_mean_mae": row.get("normal_mean_mae", np.nan),
                    "mean_mae": row.get("mean_mae", np.nan),
                }
            )
    if not selected.empty:
        row = selected.iloc[0]
        rows.append(
            {
                "objective": "hidden_2024_mase_proxy",
                "objective_metric": "normal_mean_common_mase",
                "objective_description": (
                    "Candidate marked selected by the production pipeline for the "
                    "single hidden-2024 MASE objective."
                ),
                "candidate_role": "selected_candidate",
                "target_strategy": row.get("target_strategy", ""),
                "model_family": row.get("model_family", ""),
                "model_id": row.get("model_id", ""),
                "normal_mean_common_mase": row.get("normal_mean_common_mase", np.nan),
                "mean_common_mase": row.get("mean_common_mase", np.nan),
                "normal_mean_mae": row.get("normal_mean_mae", np.nan),
                "mean_mae": row.get("mean_mae", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _build_covid_adjustment_artifacts(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create COVID fairness artifacts from fold-local adjustment metadata."""

    if fold_results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ok = fold_results.copy()
    if "status" in ok:
        ok = ok[ok["status"].eq("ok")]
    adjusted = ok[ok["covid_mode"].astype(str).eq("adjusted_target")].copy()
    beta_cols = sorted(column for column in ok.columns if column.startswith("covid_beta_"))
    id_cols = [
        "stage",
        "target_strategy",
        "model_id",
        "model_family",
        "covid_mode",
        "fold_name",
        "fold_role",
        "train_end",
        "valid_start",
        "valid_end",
        "alpha",
        "beta",
    ]
    meta_cols = [
        "covid_adjustment_estimator",
        "covid_adjustment_status",
        "covid_adjustment_train_rows",
        "covid_adjustment_feature_columns",
        "covid_adjustment_effect_mean",
        "covid_adjustment_effect_abs_mean",
        "covid_adjustment_effect_min",
        "covid_adjustment_effect_max",
    ]
    coefficient_cols = [column for column in [*id_cols, *meta_cols, *beta_cols] if column in ok]
    if adjusted.empty or not coefficient_cols:
        coefficients = pd.DataFrame(columns=coefficient_cols)
    else:
        coefficients = adjusted[coefficient_cols].drop_duplicates().reset_index(drop=True)

    if adjusted.empty:
        audit = pd.DataFrame()
    else:
        audit = adjusted[
            [
                column
                for column in [
                    *id_cols,
                    *meta_cols,
                    "mae",
                    "rmse",
                    "mase",
                    "common_mase",
                    "relative_mae_vs_seasonal_naive",
                    "train_valid_mae_ratio",
                ]
                if column in adjusted
            ]
        ].copy()
        audit["train_end_before_valid_start"] = pd.to_datetime(audit["train_end"]) < pd.to_datetime(
            audit["valid_start"]
        )
        audit["validation_compared_to_observed_target"] = True
        audit["future_covid_assumed_zero"] = True
        audit = audit.reset_index(drop=True)

    if summary.empty or "covid_mode" not in summary:
        comparison = pd.DataFrame()
    else:
        metric_cols = [
            "normal_mean_common_mase",
            "mean_common_mase",
            "normal_mean_mase",
            "mean_mase",
            "mean_mae",
            "mean_rmse",
            "cv_common_mase",
            "mean_relative_mae_vs_seasonal_naive",
        ]
        rows = []
        group_cols = ["target_strategy", "model_family"]
        for keys, group in summary.groupby(group_cols, dropna=False):
            strategy, family = keys
            best_by_mode = (
                group.dropna(subset=["normal_mean_common_mase"])
                .sort_values(["covid_mode", "normal_mean_common_mase", "mean_common_mase"])
                .groupby("covid_mode", as_index=False)
                .head(1)
            )
            none = best_by_mode[best_by_mode["covid_mode"].astype(str).eq("none")]
            adjusted_mode = best_by_mode[
                best_by_mode["covid_mode"].astype(str).eq("adjusted_target")
            ]
            row: dict[str, Any] = {
                "target_strategy": strategy,
                "model_family": family,
                "has_none": not none.empty,
                "has_adjusted_target": not adjusted_mode.empty,
            }
            if not none.empty:
                none_row = none.iloc[0]
                row["best_none_model_id"] = none_row["model_id"]
                for metric in metric_cols:
                    if metric in none_row:
                        row[f"none_{metric}"] = none_row.get(metric, np.nan)
            if not adjusted_mode.empty:
                adjusted_row = adjusted_mode.iloc[0]
                row["best_adjusted_target_model_id"] = adjusted_row["model_id"]
                for metric in metric_cols:
                    if metric in adjusted_row:
                        row[f"adjusted_target_{metric}"] = adjusted_row.get(metric, np.nan)
            none_mase = row.get("none_normal_mean_common_mase", np.nan)
            adjusted_mase = row.get("adjusted_target_normal_mean_common_mase", np.nan)
            if pd.notna(none_mase) and pd.notna(adjusted_mase):
                row["adjusted_minus_none_normal_common_mase"] = float(adjusted_mase - none_mase)
                row["adjusted_improvement_vs_none_pct"] = float(
                    (none_mase - adjusted_mase) / none_mase * 100
                    if float(none_mase) != 0
                    else np.nan
                )
                row["adjusted_target_beats_none"] = bool(adjusted_mase < none_mase)
            rows.append(row)
        comparison = pd.DataFrame(rows).sort_values(
            ["target_strategy", "model_family"],
            na_position="last",
        )
    return coefficients, audit, comparison


def run_model_comparison(
    target_strategies: dict[str, Any],
    folds_metadata: pd.DataFrame,
    old_data_gate: pd.DataFrame,
    project: dict[str, Any],
    validation: dict[str, Any],
    models: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run Stage B: full model competition on admissible target strategies."""

    strategies = admissible_strategies(old_data_gate)
    specs = model_specs(models, stage="model_comparison", include_optional=True)
    evaluated = evaluate_cv(
        stage="model_comparison",
        target_strategies=target_strategies,
        strategy_names=strategies,
        specs=specs,
        folds_metadata=folds_metadata,
        validation_params=validation,
        project_params=project,
        collect_horizon_metrics=True,
    )
    if isinstance(evaluated, tuple):
        fold_results, horizon_metrics = evaluated
    else:
        fold_results = evaluated
        horizon_metrics = pd.DataFrame()
    if "status" in fold_results:
        ok = fold_results[fold_results["status"].eq("ok")].copy()
    else:
        ok = pd.DataFrame()
    summary = summarize_cv(ok) if not ok.empty else pd.DataFrame()
    (
        covid_adjustment_coefficients,
        covid_adjustment_audit,
        covid_mode_comparison,
    ) = _build_covid_adjustment_artifacts(ok, summary)
    model_selection = (
        select_final_model(summary, selection) if not summary.empty else pd.DataFrame()
    )
    horizon_summary = summarize_horizon_metrics(horizon_metrics)
    mase_uncertainty = bootstrap_mase_uncertainty(
        horizon_metrics,
        n_bootstrap=int(validation.get("mase_uncertainty_bootstrap_samples", 1000)),
        seed=int(project.get("seed", 42)),
    )
    nested_selection_audit = _build_nested_selection_audit(ok)
    nested_cv_results = _build_nested_cv_results(ok)
    nested_cv_summary = _summarize_nested_cv(nested_cv_results)
    robust_alpha_results, robust_alpha_summary = _build_robust_alpha_artifacts(
        target_strategies=target_strategies,
        folds_metadata=folds_metadata,
        summary=summary,
        validation=validation,
        project=project,
        models=models,
    )
    rolling_origin_robustness = _build_rolling_origin_robustness(
        target_strategies=target_strategies,
        model_selection=model_selection,
        validation=validation,
        project=project,
        models=models,
    )
    model_selection = _apply_rolling_origin_selection(
        model_selection,
        rolling_origin_robustness,
    )
    selection_objective_audit = _build_selection_objective_audit(summary, model_selection)
    gap_cols = [
        col
        for col in [
            *TRAIN_VALID_GAP_COLUMNS,
            *sorted(col for col in ok.columns if col.startswith("model_param_")),
        ]
        if col in ok.columns
    ]
    train_valid_gap = ok[gap_cols].copy() if gap_cols else pd.DataFrame()
    _log_model_comparison_to_mlflow(
        fold_results,
        summary,
        train_valid_gap,
        model_selection,
        horizon_metrics,
        horizon_summary,
        mase_uncertainty,
        nested_selection_audit,
        nested_cv_results,
        nested_cv_summary,
        rolling_origin_robustness,
        robust_alpha_results,
        robust_alpha_summary,
        selection_objective_audit,
        covid_adjustment_coefficients,
        covid_adjustment_audit,
        covid_mode_comparison,
        strategies,
        selection,
    )
    return (
        fold_results,
        summary,
        train_valid_gap,
        model_selection,
        horizon_metrics,
        horizon_summary,
        mase_uncertainty,
        nested_selection_audit,
        nested_cv_results,
        nested_cv_summary,
        rolling_origin_robustness,
        robust_alpha_results,
        robust_alpha_summary,
        selection_objective_audit,
        covid_adjustment_coefficients,
        covid_adjustment_audit,
        covid_mode_comparison,
    )
