"""Treino, calibração, avaliação e retomada do pipeline modelled -> ml."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from targeted_ml.modeling.calibration import (
    build_temporal_calibration_holdout as canonical_build_temporal_calibration_holdout,
    build_temporal_calibrator,
)
from targeted_ml.modeling.model_specs import build_model_specs as build_canonical_model_specs
from targeted_ml.modeling.preprocessing import build_preprocessor_from_registry
from targeted_ml.modeling.splitters import ExpandingMonthSplit

from . import analysis_setup as setup
from .dataset_builder import select_active_features
from .storage import ModelTaskKey, TaskArtifactStore

MODEL_TASK_SIGNATURE_VERSION = "2026-03-30-tuning-grid-fix-v1"


@lru_cache(maxsize=1)
def build_model_task_code_fingerprint() -> str:
    digest = hashlib.sha256()
    file_paths = sorted(
        {
            str(Path(__file__).resolve()),
            str(Path(canonical_build_temporal_calibration_holdout.__code__.co_filename).resolve()),
            str(Path(build_preprocessor_from_registry.__code__.co_filename).resolve()),
            str(Path(build_canonical_model_specs.__code__.co_filename).resolve()),
        }
    )
    for file_path in file_paths:
        path = Path(file_path)
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()

def bootstrap_ci_width(values: np.ndarray, fn) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    samples: List[float] = []
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    for _ in range(setup.BOOTSTRAP_ITERATIONS):
        take = rng.integers(0, len(values), len(values))
        samples.append(float(fn(values[take])))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high), float(high - low)

def bootstrap_prevalence_ci_width_from_counts(n_rows: int, n_positive: int) -> tuple[float, float, float]:
    if n_rows <= 0:
        return float("nan"), float("nan"), float("nan")
    p = n_positive / n_rows
    z = 1.959963984540054
    denom = 1 + (z**2 / n_rows)
    center = (p + (z**2 / (2 * n_rows))) / denom
    margin = (z * math.sqrt((p * (1 - p) / n_rows) + (z**2 / (4 * n_rows**2)))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return float(low), float(high), float(high - low)

def pareto_front(df: pd.DataFrame, objectives: Dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.dropna(subset=list(objectives)).reset_index(drop=True).copy()
    if work.empty:
        return work
    dominated = np.zeros(len(work), dtype=bool)
    values = work[list(objectives)].to_dict(orient="records")
    objective_items = list(objectives.items())
    for i, row_i in enumerate(values):
        if dominated[i]:
            continue
        for j, row_j in enumerate(values):
            if i == j:
                continue
            better_or_equal = True
            strictly_better = False
            for metric, direction in objective_items:
                a = row_i[metric]
                b = row_j[metric]
                if direction == "max":
                    if b < a:
                        better_or_equal = False
                        break
                    if b > a:
                        strictly_better = True
                else:
                    if b > a:
                        better_or_equal = False
                        break
                    if b < a:
                        strictly_better = True
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    work["pareto_frontier_flag"] = (~dominated).astype(int)
    return work

def calibration_slope_intercept(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logits = np.log(score / (1 - score)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(logits, y)
    return float(model.coef_[0][0]), float(model.intercept_[0])

def probability_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    metrics = {
        "ap": float("nan"),
        "roc_auc": float("nan"),
        "brier": float("nan"),
        "log_loss": float("nan"),
        "calibration_slope": float("nan"),
        "calibration_intercept": float("nan"),
        "calibration_slope_error": float("nan"),
        "calibration_intercept_abs": float("nan"),
    }
    if len(np.unique(y)) < 2:
        return metrics
    metrics["ap"] = float(average_precision_score(y, score))
    metrics["roc_auc"] = float(roc_auc_score(y, score))
    metrics["brier"] = float(brier_score_loss(y, score))
    metrics["log_loss"] = float(log_loss(y, score))
    slope, intercept = calibration_slope_intercept(y, score)
    metrics["calibration_slope"] = slope
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope_error"] = abs(slope - 1.0) if pd.notna(slope) else float("nan")
    metrics["calibration_intercept_abs"] = abs(intercept) if pd.notna(intercept) else float("nan")
    return metrics


def generalization_gap(
    train_metric: float,
    test_metric: float,
    objective_direction: str,
) -> float:
    if pd.isna(train_metric) or pd.isna(test_metric):
        return float("nan")
    if objective_direction == "max":
        return float(train_metric - test_metric)
    return float(test_metric - train_metric)


def build_train_test_generalization_outputs(
    fold_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_columns = [
        "problem_key",
        "definition_name",
        "track_name",
        "model_name",
        "fold_id",
        "comparison_stage",
        "metric_name",
        "train_metric",
        "test_metric",
        "generalization_gap",
    ]
    summary_columns = [
        "problem_key",
        "definition_name",
        "track_name",
        "model_name",
        "comparison_stage",
        "metric_name",
        "valid_folds",
        "mean_train_metric",
        "mean_test_metric",
        "mean_generalization_gap",
        "ci_low_generalization_gap",
        "ci_high_generalization_gap",
        "ci_width_generalization_gap",
        "statistical_gap_flag",
    ]
    if fold_df.empty:
        return pd.DataFrame(columns=fold_columns), pd.DataFrame(columns=summary_columns)
    valid = fold_df[
        pd.to_numeric(fold_df.get("fold_valid_flag"), errors="coerce").fillna(0).astype(int) == 1
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=fold_columns), pd.DataFrame(columns=summary_columns)
    metric_objectives = {
        "ap": "max",
        "roc_auc": "max",
        "brier": "min",
        "log_loss": "min",
    }
    stage_sources = {
        "apparent_train": "apparent_train",
        "calibration_holdout": "calibration_holdout",
    }
    rows: list[dict[str, Any]] = []
    for row in valid.to_dict(orient="records"):
        for stage_name, prefix in stage_sources.items():
            for metric_name, direction in metric_objectives.items():
                train_metric = pd.to_numeric(row.get(f"{prefix}_{metric_name}"), errors="coerce")
                test_metric = pd.to_numeric(row.get(metric_name), errors="coerce")
                gap_value = generalization_gap(float(train_metric), float(test_metric), direction) if pd.notna(train_metric) and pd.notna(test_metric) else float("nan")
                rows.append(
                    {
                        "problem_key": row["problem_key"],
                        "definition_name": row["definition_name"],
                        "track_name": row["track_name"],
                        "model_name": row["model_name"],
                        "fold_id": row["fold_id"],
                        "comparison_stage": stage_name,
                        "metric_name": metric_name,
                        "train_metric": float(train_metric) if pd.notna(train_metric) else float("nan"),
                        "test_metric": float(test_metric) if pd.notna(test_metric) else float("nan"),
                        "generalization_gap": gap_value,
                    }
                )
    fold_audit = pd.DataFrame(rows, columns=fold_columns)
    if fold_audit.empty:
        return fold_audit, pd.DataFrame(columns=summary_columns)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in fold_audit.groupby(
        ["problem_key", "definition_name", "track_name", "model_name", "comparison_stage", "metric_name"],
        dropna=False,
    ):
        gaps = pd.to_numeric(group["generalization_gap"], errors="coerce").dropna().to_numpy(dtype=float)
        ci_low, ci_high, ci_width = bootstrap_ci_width(gaps, np.mean) if len(gaps) else (float("nan"), float("nan"), float("nan"))
        summary_rows.append(
            {
                "problem_key": keys[0],
                "definition_name": keys[1],
                "track_name": keys[2],
                "model_name": keys[3],
                "comparison_stage": keys[4],
                "metric_name": keys[5],
                "valid_folds": int(group["fold_id"].nunique()),
                "mean_train_metric": float(pd.to_numeric(group["train_metric"], errors="coerce").mean()),
                "mean_test_metric": float(pd.to_numeric(group["test_metric"], errors="coerce").mean()),
                "mean_generalization_gap": float(np.mean(gaps)) if len(gaps) else float("nan"),
                "ci_low_generalization_gap": float(ci_low),
                "ci_high_generalization_gap": float(ci_high),
                "ci_width_generalization_gap": float(ci_width),
                "statistical_gap_flag": int(pd.notna(ci_low) and ci_low > 0),
            }
        )
    return fold_audit, pd.DataFrame(summary_rows, columns=summary_columns)

def build_temporal_calibration_holdout(
    train: pd.DataFrame,
    month_col: str,
    target_col: str,
) -> tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    return canonical_build_temporal_calibration_holdout(train, month_col, target_col)

def build_preprocessor(feature_registry: pd.DataFrame, feature_names: list[str]) -> ColumnTransformer:
    return build_preprocessor_from_registry(feature_registry, feature_names)

def build_model_specs() -> list[dict[str, Any]]:
    return build_canonical_model_specs(setup.RUNTIME_CONFIG.model_family_scope)


def filter_official_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    if "fold_valid_flag" not in predictions.columns:
        return predictions.copy()
    return predictions[pd.to_numeric(predictions["fold_valid_flag"], errors="coerce").fillna(0).astype(int) == 1].copy()


def summarize_prediction_pools(
    predictions: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    keys = list(group_keys or ["problem_key", "definition_name", "track_name", "model_name"])
    summary_columns = [
        "pooled_rows",
        "pooled_positives",
        "pooled_negatives",
        "pooled_positive_rate",
        "pooled_ap",
        "pooled_roc_auc",
        "pooled_brier",
        "pooled_log_loss",
        "pooled_calibration_slope",
        "pooled_calibration_intercept",
        "pooled_calibration_slope_error",
        "pooled_calibration_intercept_abs",
    ]
    empty_columns = list(keys) + summary_columns
    official_predictions = filter_official_predictions(predictions)
    if official_predictions.empty:
        return pd.DataFrame(columns=empty_columns)
    rows: list[dict[str, Any]] = []
    for group_values, group in official_predictions.groupby(keys, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(keys, group_values))
        y_true = pd.to_numeric(group["y_true"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_score = pd.to_numeric(group["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        positives = int(y_true.sum())
        rows_count = int(len(group))
        negatives = int(rows_count - positives)
        metrics = probability_metrics(y_true, y_score)
        row.update(
            {
                "pooled_rows": rows_count,
                "pooled_positives": positives,
                "pooled_negatives": negatives,
                "pooled_positive_rate": float(positives / rows_count) if rows_count else float("nan"),
                "pooled_ap": metrics["ap"],
                "pooled_roc_auc": metrics["roc_auc"],
                "pooled_brier": metrics["brier"],
                "pooled_log_loss": metrics["log_loss"],
                "pooled_calibration_slope": metrics["calibration_slope"],
                "pooled_calibration_intercept": metrics["calibration_intercept"],
                "pooled_calibration_slope_error": metrics["calibration_slope_error"],
                "pooled_calibration_intercept_abs": metrics["calibration_intercept_abs"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=empty_columns)


def summarize_valid_model_folds(
    fold_df: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    summary_columns = [
        "valid_folds",
        "fold_mean_ap",
        "std_ap",
        "fold_mean_roc_auc",
        "std_roc_auc",
        "fold_mean_brier",
        "std_brier",
        "fold_mean_log_loss",
        "std_log_loss",
        "fold_mean_calibration_slope",
        "fold_mean_calibration_intercept",
        "fold_mean_calibration_slope_error",
        "fold_mean_calibration_intercept_abs",
    ]
    keys = list(group_keys or ["problem_key", "definition_name", "track_name", "model_name"])
    empty_columns = list(keys) + summary_columns
    valid_folds = fold_df[fold_df.get("fold_valid_flag", 0) == 1].copy()
    if valid_folds.empty:
        return pd.DataFrame(columns=empty_columns)
    return (
        valid_folds.groupby(list(keys), as_index=False)
        .agg(
            valid_folds=("fold_id", "nunique"),
            fold_mean_ap=("ap", "mean"),
            std_ap=("ap", "std"),
            fold_mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            fold_mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            fold_mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            fold_mean_calibration_slope=("calibration_slope", "mean"),
            fold_mean_calibration_intercept=("calibration_intercept", "mean"),
            fold_mean_calibration_slope_error=("calibration_slope_error", "mean"),
            fold_mean_calibration_intercept_abs=("calibration_intercept_abs", "mean"),
        )
    )


def summarize_model_performance(
    fold_df: pd.DataFrame,
    predictions: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    keys = list(group_keys or ["problem_key", "definition_name", "track_name", "model_name"])
    pooled = summarize_prediction_pools(predictions, group_keys=keys)
    fold_summary = summarize_valid_model_folds(fold_df, group_keys=keys)
    if pooled.empty and fold_summary.empty:
        return pd.DataFrame(
            columns=list(keys)
            + [
                "valid_folds",
                "pooled_rows",
                "pooled_positives",
                "pooled_negatives",
                "pooled_positive_rate",
                "mean_ap",
                "std_ap",
                "mean_roc_auc",
                "std_roc_auc",
                "mean_brier",
                "std_brier",
                "mean_log_loss",
                "std_log_loss",
                "mean_calibration_slope",
                "mean_calibration_intercept",
                "mean_calibration_slope_error",
                "mean_calibration_intercept_abs",
                "fold_mean_ap",
                "fold_mean_roc_auc",
                "fold_mean_brier",
                "fold_mean_log_loss",
                "fold_mean_calibration_slope",
                "fold_mean_calibration_intercept",
                "fold_mean_calibration_slope_error",
                "fold_mean_calibration_intercept_abs",
            ]
        )
    summary = pooled.merge(fold_summary, on=keys, how="outer")
    rename_map = {
        "pooled_ap": "mean_ap",
        "pooled_roc_auc": "mean_roc_auc",
        "pooled_brier": "mean_brier",
        "pooled_log_loss": "mean_log_loss",
        "pooled_calibration_slope": "mean_calibration_slope",
        "pooled_calibration_intercept": "mean_calibration_intercept",
        "pooled_calibration_slope_error": "mean_calibration_slope_error",
        "pooled_calibration_intercept_abs": "mean_calibration_intercept_abs",
    }
    summary = summary.rename(columns=rename_map)
    for col in ["std_ap", "std_roc_auc", "std_brier", "std_log_loss"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0.0)
    return summary


def build_temporal_tuning_splits(
    train: pd.DataFrame,
    month_col: str,
    target_col: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    splitter = ExpandingMonthSplit(
        month_col=month_col,
        min_train_periods=1,
        test_periods=1,
        max_splits=setup.TUNING_MAX_INNER_SPLITS,
    )
    audit_rows: list[dict[str, Any]] = []
    valid_splits: list[tuple[np.ndarray, np.ndarray]] = []
    months = pd.to_datetime(train[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    unique_months = np.array(sorted(months.dropna().unique()))
    if len(unique_months) < 2:
        audit_rows.append(
            {
                "inner_fold_id": 1,
                "split_strategy": "temporal_tuning_validation",
                "train_rows": int(len(train)),
                "test_rows": 0,
                "train_positives": int(pd.to_numeric(train[target_col], errors="coerce").fillna(0).sum()),
                "test_positives": 0,
                "valid_inner_split_flag": 0,
                "invalid_reason": "not_enough_months_for_tuning_validation",
            }
        )
        return [], pd.DataFrame(audit_rows)
    for inner_fold_id, (inner_train_idx, inner_validation_idx) in enumerate(splitter.split(train), start=1):
        inner_train = train.iloc[inner_train_idx].copy()
        inner_validation = train.iloc[inner_validation_idx].copy()
        validation_months = (
            pd.to_datetime(inner_validation[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp().dropna().unique().tolist()
        )
        invalid_reason = ""
        if inner_train[target_col].nunique() < 2:
            invalid_reason = "tuning_train_single_class"
        elif inner_validation[target_col].nunique() < 2:
            invalid_reason = "tuning_validation_single_class"
        audit_rows.append(
            {
                "inner_fold_id": inner_fold_id,
                "split_strategy": "temporal_tuning_validation",
                "validation_start_month": pd.Timestamp(validation_months[0]).strftime("%Y-%m-%d") if validation_months else "",
                "validation_month_count": int(len(validation_months)),
                "train_rows": int(len(inner_train_idx)),
                "test_rows": int(len(inner_validation_idx)),
                "train_positives": int(pd.to_numeric(inner_train[target_col], errors="coerce").fillna(0).sum()),
                "test_positives": int(pd.to_numeric(inner_validation[target_col], errors="coerce").fillna(0).sum()),
                "valid_inner_split_flag": int(not invalid_reason),
                "invalid_reason": invalid_reason,
            }
        )
        if not invalid_reason:
            valid_splits.append((inner_train_idx, inner_validation_idx))
    return valid_splits, pd.DataFrame(audit_rows)


def tune_temporal_estimator(
    estimator: Pipeline,
    model_spec: dict[str, Any],
    tuning_train: pd.DataFrame,
    feature_names: Sequence[str],
    target_col: str = "y_true",
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    default_meta = {
        "tuning_applied_flag": 0,
        "tuning_status": "disabled",
        "tuning_valid_splits": 0,
        "tuning_best_score": float("nan"),
        "best_params_json": setup.stable_json({}),
    }
    if not setup.TUNING_ENABLED:
        return clone(estimator), pd.DataFrame(), default_meta
    raw_param_distributions = model_spec.get("param_distributions", {})
    if isinstance(raw_param_distributions, list):
        param_distributions = [dict(space) for space in raw_param_distributions]
    elif isinstance(raw_param_distributions, dict):
        param_distributions = dict(raw_param_distributions)
    else:
        param_distributions = raw_param_distributions
    if not param_distributions:
        meta = default_meta.copy()
        meta["tuning_status"] = "no_param_space_registered"
        return clone(estimator), pd.DataFrame(), meta
    valid_splits, tuning_audit = build_temporal_tuning_splits(
        tuning_train,
        month_col="first_month",
        target_col=target_col,
    )
    if not valid_splits:
        meta = default_meta.copy()
        meta["tuning_status"] = "fallback_default_no_valid_inner_tuning_split"
        return clone(estimator), tuning_audit, meta
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=max(1, setup.TUNING_N_ITER),
        scoring=setup.TUNING_SCORING,
        n_jobs=1,
        refit=True,
        cv=valid_splits,
        random_state=42,
        error_score="raise",
        return_train_score=False,
    )
    X_tuning = tuning_train[list(feature_names) + ["first_month"]].copy()
    y_tuning = pd.to_numeric(tuning_train[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    search.fit(X_tuning, y_tuning)
    tuned_estimator = clone(search.best_estimator_)
    meta = {
        "tuning_applied_flag": 1,
        "tuning_status": "searched",
        "tuning_valid_splits": int(len(valid_splits)),
        "tuning_best_score": float(search.best_score_) if search.best_score_ is not None else float("nan"),
        "best_params_json": setup.stable_json(search.best_params_),
    }
    return tuned_estimator, tuning_audit, meta

def build_scoring_scenarios(
    frame: pd.DataFrame,
    feature_registry: pd.DataFrame,
    track_registry: pd.DataFrame,
    definition_frontier: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    # The model stage is intentionally decoupled from the 90-day definition study.
    # We always score the fixed 30-day business target (Definition B) and never let
    # the definition frontier determine the model outcome.
    official_definitions: list[str] = []
    if "definition_b_label" in frame.columns:
        official_definitions = ["definition_b_label"]
    for definition_name in official_definitions:
        label_col = definition_name
        if label_col not in frame.columns:
            continue
        for track in track_registry.to_dict(orient="records"):
            feature_names = [
                row["feature_name"]
                for row in feature_registry.to_dict(orient="records")
                if int(row[f'allowed_in_{track["track_name"]}']) == 1
            ]
            working = frame.dropna(subset=["first_month"]).copy()
            y = pd.to_numeric(working[label_col], errors="coerce")
            valid_mask = y.notna()
            working = working.loc[valid_mask].copy()
            y = y.loc[valid_mask].astype(int)
            rows.append(
                {
                    "problem_key": f'{definition_name}__{track["track_name"]}',
                    "definition_name": definition_name,
                    "label_col": label_col,
                    "track_name": track["track_name"],
                    "score_window_end_day": track["score_window_end_day"],
                    "feature_count": len(feature_names),
                    "feature_names_json": setup.stable_json(feature_names),
                    "rows": int(len(working)),
                    "positives": int(y.sum()),
                    "negatives": int((1 - y).sum()),
                    "months": int(working["first_month"].nunique()),
                }
            )
    return pd.DataFrame(rows)

def bootstrap_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = filter_official_predictions(predictions)
    rows: List[dict[str, Any]] = []
    for keys, group in predictions.groupby(["problem_key", "model_name"], dropna=False):
        y = group["y_true"].to_numpy(dtype=int)
        score = group["score"].to_numpy(dtype=float)
        rng = np.random.default_rng(42)
        stats = {"brier": [], "log_loss": [], "ap": [], "roc_auc": []}
        if len(np.unique(y)) < 2:
            rows.append(
                {
                    "problem_key": keys[0],
                    "model_name": keys[1],
                    "metric_name": "bootstrap_unavailable",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_width": np.nan,
                }
            )
            continue
        for _ in range(setup.BOOTSTRAP_ITERATIONS):
            take = rng.integers(0, len(group), len(group))
            y_sample = y[take]
            score_sample = score[take]
            if len(np.unique(y_sample)) < 2:
                continue
            stats["brier"].append(brier_score_loss(y_sample, score_sample))
            stats["log_loss"].append(log_loss(y_sample, np.clip(score_sample, 1e-6, 1 - 1e-6)))
            stats["ap"].append(average_precision_score(y_sample, score_sample))
            stats["roc_auc"].append(roc_auc_score(y_sample, score_sample))
        for metric_name, values in stats.items():
            if not values:
                ci_low = ci_high = ci_width = float("nan")
            else:
                ci_low, ci_high = np.percentile(values, [2.5, 97.5])
                ci_width = ci_high - ci_low
            rows.append(
                {
                    "problem_key": keys[0],
                    "model_name": keys[1],
                    "metric_name": metric_name,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "ci_width": float(ci_width),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["problem_key", "model_name", "metric_name", "ci_low", "ci_high", "ci_width"],
    )

def evaluate_single_problem_model(
    problem: dict[str, Any],
    model_spec: dict[str, Any],
    working: pd.DataFrame,
    feature_registry: pd.DataFrame,
    feature_names: Sequence[str],
    compute_feature_importance: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    inner_audit_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    post_model_output_status_rows: list[dict[str, Any]] = []
    # Outer folds:
    # - treino = meses acumulados ate aqui
    # - teste = mes seguinte
    # Isso respeita o tempo e deixa claro "o que o modelo saberia naquele mes".
    splitter = ExpandingMonthSplit(month_col="first_month", min_train_periods=1, test_periods=1, max_splits=setup.MAX_OUTER_TEST_MONTHS)
    print(f'[model]   estimator {model_spec["model_name"]}', flush=True)
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(working), start=1):
        print(f'[model]     fold {fold_id}', flush=True)
        train = working.iloc[train_idx].copy()
        test = working.iloc[test_idx].copy()
        test_rows = int(len(test))
        test_positives = int(pd.to_numeric(test["y_true"], errors="coerce").fillna(0).sum()) if "y_true" in test.columns else 0
        test_negatives = int(test_rows - test_positives)
        technical_fold_valid_flag = int(train["y_true"].nunique() >= 2 and test["y_true"].nunique() >= 2)
        if train["y_true"].nunique() < 2 or test["y_true"].nunique() < 2:
            fold_rows.append(
                {
                    "problem_key": problem["problem_key"],
                    "definition_name": problem["definition_name"],
                    "track_name": problem["track_name"],
                    "model_name": model_spec["model_name"],
                    "fold_id": fold_id,
                    "technical_fold_valid_flag": 0,
                    "fold_valid_flag": 0,
                    "rows": test_rows,
                    "positives": test_positives,
                    "negatives": test_negatives,
                    "test_positive_rate": float(test_positives / test_rows) if test_rows else float("nan"),
                    "invalid_reason": "single_class_fold",
                }
            )
            continue
        fit_idx, calibration_idx, inner_audit = build_temporal_calibration_holdout(train, month_col="first_month", target_col="y_true")
        if not inner_audit.empty:
            inner_audit = inner_audit.copy()
            inner_audit["problem_key"] = problem["problem_key"]
            inner_audit["definition_name"] = problem["definition_name"]
            inner_audit["track_name"] = problem["track_name"]
            inner_audit["model_name"] = model_spec["model_name"]
            inner_audit["outer_fold_id"] = fold_id
            inner_audit_rows.extend(inner_audit.to_dict(orient="records"))
        if fit_idx is None or calibration_idx is None:
            fold_rows.append(
                {
                    "problem_key": problem["problem_key"],
                    "definition_name": problem["definition_name"],
                    "track_name": problem["track_name"],
                    "model_name": model_spec["model_name"],
                    "fold_id": fold_id,
                    "technical_fold_valid_flag": technical_fold_valid_flag,
                    "fold_valid_flag": 0,
                    "rows": test_rows,
                    "positives": test_positives,
                    "negatives": test_negatives,
                    "test_positive_rate": float(test_positives / test_rows) if test_rows else float("nan"),
                    "invalid_reason": "no_valid_temporal_calibration_holdout",
                }
            )
            continue
        fit_train = train.iloc[fit_idx].copy()
        calibration_holdout = train.iloc[calibration_idx].copy()
        active_feature_names = select_active_features(fit_train, feature_names, calibration_holdout)
        if not active_feature_names:
            fold_rows.append(
                {
                    "problem_key": problem["problem_key"],
                    "definition_name": problem["definition_name"],
                    "track_name": problem["track_name"],
                    "model_name": model_spec["model_name"],
                    "fold_id": fold_id,
                    "technical_fold_valid_flag": technical_fold_valid_flag,
                    "fold_valid_flag": 0,
                    "rows": test_rows,
                    "positives": test_positives,
                    "negatives": test_negatives,
                    "test_positive_rate": float(test_positives / test_rows) if test_rows else float("nan"),
                    "invalid_reason": "all_train_features_missing_in_inner_splits",
                }
            )
            continue
        preprocessor = build_preprocessor(feature_registry, active_feature_names)
        estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model_spec["estimator"]),
            ]
        )
        train_for_calibration = train[active_feature_names + ["first_month", "y_true"]].copy()
        tuning_train = fit_train[active_feature_names + ["first_month", "y_true"]].copy()
        X_test = test[active_feature_names + ["first_month"]].copy()
        tuning_meta = {
            "tuning_applied_flag": 0,
            "tuning_status": "not_attempted",
            "tuning_valid_splits": 0,
            "tuning_best_score": float("nan"),
            "best_params_json": setup.stable_json({}),
        }
        try:
            tuned_estimator, tuning_audit, tuning_meta = tune_temporal_estimator(
                estimator=estimator,
                model_spec=model_spec,
                tuning_train=tuning_train,
                feature_names=active_feature_names,
                target_col="y_true",
            )
            if not tuning_audit.empty:
                tuning_audit = tuning_audit.copy()
                tuning_audit["problem_key"] = problem["problem_key"]
                tuning_audit["definition_name"] = problem["definition_name"]
                tuning_audit["track_name"] = problem["track_name"]
                tuning_audit["model_name"] = model_spec["model_name"]
                tuning_audit["outer_fold_id"] = fold_id
                inner_audit_rows.extend(tuning_audit.to_dict(orient="records"))
            calibrated = build_temporal_calibrator(
                estimator=tuned_estimator,
                train=train_for_calibration,
                target_col="y_true",
                fit_idx=fit_idx,
                calibration_idx=calibration_idx,
                method=setup.CALIBRATION_METHOD,
            )
            X_apparent_train = train_for_calibration[active_feature_names + ["first_month"]].copy()
            X_calibration_holdout = calibration_holdout[active_feature_names + ["first_month"]].copy()
            test_score = calibrated.predict_proba(X_test)[:, 1]
            apparent_train_score = calibrated.predict_proba(X_apparent_train)[:, 1]
            calibration_holdout_score = calibrated.predict_proba(X_calibration_holdout)[:, 1]
        except Exception as exc:
            exc_msg = setup.normalize_text(str(exc), default=type(exc).__name__).replace(" ", "_")[:120]
            fold_rows.append(
                {
                    "problem_key": problem["problem_key"],
                    "definition_name": problem["definition_name"],
                    "track_name": problem["track_name"],
                    "model_name": model_spec["model_name"],
                    "fold_id": fold_id,
                    "technical_fold_valid_flag": technical_fold_valid_flag,
                    "fold_valid_flag": 0,
                    "rows": test_rows,
                    "positives": test_positives,
                    "negatives": test_negatives,
                    "test_positive_rate": float(test_positives / test_rows) if test_rows else float("nan"),
                    **tuning_meta,
                    "invalid_reason": f"fit_exception:{type(exc).__name__}:{exc_msg}",
                }
            )
            continue
        apparent_train_metrics = probability_metrics(train_for_calibration["y_true"].to_numpy(), apparent_train_score)
        calibration_holdout_metrics = probability_metrics(calibration_holdout["y_true"].to_numpy(), calibration_holdout_score)
        metrics = probability_metrics(test["y_true"].to_numpy(), test_score)
        official_support_valid = int(
            test_rows >= setup.MIN_OFFICIAL_TEST_ROWS
            and test_positives >= setup.MIN_OFFICIAL_TEST_POSITIVES
            and test_negatives >= setup.MIN_OFFICIAL_TEST_NEGATIVES
        )
        invalid_reason = "" if official_support_valid == 1 else "insufficient_test_support"
        fold_rows.append(
            {
                "problem_key": problem["problem_key"],
                "definition_name": problem["definition_name"],
                "track_name": problem["track_name"],
                "model_name": model_spec["model_name"],
                "fold_id": fold_id,
                "technical_fold_valid_flag": technical_fold_valid_flag,
                "fold_valid_flag": official_support_valid,
                "inner_valid_splits": 1,
                "rows": test_rows,
                "positives": test_positives,
                "negatives": test_negatives,
                "test_positive_rate": float(test_positives / test_rows) if test_rows else float("nan"),
                **tuning_meta,
                "invalid_reason": invalid_reason,
                "apparent_train_rows": int(len(train_for_calibration)),
                "calibration_holdout_rows": int(len(calibration_holdout)),
                "apparent_train_positive_rate": float(pd.to_numeric(train_for_calibration["y_true"], errors="coerce").fillna(0).mean()),
                "calibration_holdout_positive_rate": float(pd.to_numeric(calibration_holdout["y_true"], errors="coerce").fillna(0).mean()),
                **{f"apparent_train_{key}": value for key, value in apparent_train_metrics.items()},
                **{f"calibration_holdout_{key}": value for key, value in calibration_holdout_metrics.items()},
                **metrics,
            }
        )
        pred_frame = test[["teacher_unique_id", "first_month", "y_true"]].copy()
        pred_frame["problem_key"] = problem["problem_key"]
        pred_frame["definition_name"] = problem["definition_name"]
        pred_frame["track_name"] = problem["track_name"]
        pred_frame["model_name"] = model_spec["model_name"]
        pred_frame["fold_id"] = fold_id
        pred_frame["score"] = test_score
        pred_frame["technical_fold_valid_flag"] = technical_fold_valid_flag
        pred_frame["fold_valid_flag"] = official_support_valid
        pred_frame["invalid_reason"] = invalid_reason
        prediction_rows.extend(pred_frame.to_dict(orient="records"))
        if compute_feature_importance:
            try:
                perm = permutation_importance(
                    calibrated,
                    X_test,
                    test["y_true"].to_numpy(),
                    scoring="neg_brier_score",
                    n_repeats=setup.FEATURE_IMPORTANCE_PERMUTATION_REPEATS,
                    random_state=42,
                )
                for feature_name, mean_value, std_value in zip(active_feature_names, perm.importances_mean, perm.importances_std):
                    importance_rows.append(
                        {
                            "problem_key": problem["problem_key"],
                            "definition_name": problem["definition_name"],
                            "track_name": problem["track_name"],
                            "model_name": model_spec["model_name"],
                            "fold_id": fold_id,
                            "feature_name": feature_name,
                            "importance_mean": float(mean_value),
                            "importance_std": float(std_value),
                        }
                    )
                post_model_output_status_rows.append(
                    {
                        "problem_key": problem["problem_key"],
                        "definition_name": problem["definition_name"],
                        "track_name": problem["track_name"],
                        "model_name": model_spec["model_name"],
                        "fold_id": fold_id,
                        "post_model_output_name": "feature_importance",
                        "post_model_output_status": "success",
                        "error_type": "",
                        "error_message": "",
                        "traceback_snippet": "",
                    }
                )
            except Exception as exc:
                post_model_output_status_rows.append(
                    {
                        "problem_key": problem["problem_key"],
                        "definition_name": problem["definition_name"],
                        "track_name": problem["track_name"],
                        "model_name": model_spec["model_name"],
                        "fold_id": fold_id,
                        "post_model_output_name": "feature_importance",
                        "post_model_output_status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                        "traceback_snippet": traceback.format_exc(limit=3)[:2000],
                    }
                )
    return fold_rows, prediction_rows, inner_audit_rows, importance_rows, post_model_output_status_rows

def run_or_load_model_task(
    problem: dict[str, Any],
    model_spec: dict[str, Any],
    working: pd.DataFrame,
    feature_registry: pd.DataFrame,
    feature_names: Sequence[str],
    compute_feature_importance: bool,
    task_store: TaskArtifactStore | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_scope = "post_model_feature_importance" if compute_feature_importance else "core_model_eval"
    task_key = ModelTaskKey(
        problem_key=str(problem["problem_key"]),
        model_name=str(model_spec["model_name"]),
        task_scope=task_scope,
    )
    task_signature = setup.stable_json(
        {
            "model_task_signature_version": MODEL_TASK_SIGNATURE_VERSION,
            "task_code_fingerprint": build_model_task_code_fingerprint(),
            "problem_key": problem["problem_key"],
            "model_name": model_spec["model_name"],
            "feature_names": list(feature_names),
            "compute_feature_importance": compute_feature_importance,
            "max_outer_test_months": setup.MAX_OUTER_TEST_MONTHS,
            "calibration_method": setup.CALIBRATION_METHOD,
            "min_official_test_rows": setup.MIN_OFFICIAL_TEST_ROWS,
            "min_official_test_positives": setup.MIN_OFFICIAL_TEST_POSITIVES,
            "min_official_test_negatives": setup.MIN_OFFICIAL_TEST_NEGATIVES,
            "tuning_enabled": setup.TUNING_ENABLED,
            "tuning_n_iter": setup.TUNING_N_ITER,
            "tuning_max_inner_splits": setup.TUNING_MAX_INNER_SPLITS,
            "tuning_scoring": setup.TUNING_SCORING,
            "param_distributions": model_spec.get("param_distributions", {}),
        }
    )
    if task_store and task_store.is_completed(task_key, expected_signature=task_signature):
        print(
            f'[cache] hit scope={task_scope} | problem={problem["problem_key"]} | model={model_spec["model_name"]}',
            flush=True,
        )
        staged = task_store.load_completed(task_key)
        return (
            staged.get("fold_metrics", pd.DataFrame()).to_dict(orient="records"),
            staged.get("predictions", pd.DataFrame()).to_dict(orient="records"),
            staged.get("inner_audit", pd.DataFrame()).to_dict(orient="records"),
            staged.get("importance", pd.DataFrame()).to_dict(orient="records"),
            staged.get("post_model_output_status", pd.DataFrame()).to_dict(orient="records"),
        )

    if task_store:
        print(
            f'[cache] miss scope={task_scope} | problem={problem["problem_key"]} | model={model_spec["model_name"]}',
            flush=True,
        )

    if task_store:
        task_store.write_running(
            task_key,
            metadata={
                "problem_key": problem["problem_key"],
                "model_name": model_spec["model_name"],
                "compute_feature_importance": compute_feature_importance,
                "task_signature": task_signature,
            },
        )

    try:
        result = evaluate_single_problem_model(
            problem=problem,
            model_spec=model_spec,
            working=working,
            feature_registry=feature_registry,
            feature_names=feature_names,
            compute_feature_importance=compute_feature_importance,
        )
        if task_store:
            fold_rows, prediction_rows, inner_rows, importance_rows, post_model_output_status_rows = result
            task_store.write_completed(
                task_key,
                tables={
                    "fold_metrics": pd.DataFrame(fold_rows),
                    "predictions": pd.DataFrame(prediction_rows),
                    "inner_audit": pd.DataFrame(inner_rows),
                    "importance": pd.DataFrame(importance_rows),
                    "post_model_output_status": pd.DataFrame(post_model_output_status_rows),
                },
                metadata={
                    "problem_key": problem["problem_key"],
                    "model_name": model_spec["model_name"],
                    "task_signature": task_signature,
                },
            )
        return result
    except Exception as exc:
        if task_store:
            task_store.write_failed(
                task_key,
                {
                    "problem_key": problem["problem_key"],
                    "model_name": model_spec["model_name"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "traceback_snippet": traceback.format_exc(limit=5)[:4000],
                },
            )
        raise


def build_definition_b_feature_block_gain_diagnostics(
    frame: pd.DataFrame,
    feature_registry: pd.DataFrame,
    scoring_scenarios: pd.DataFrame,
    task_store: TaskArtifactStore | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logistic_spec = next((spec for spec in build_model_specs() if spec["model_name"] == "logistic_regression"), None)
    if logistic_spec is None:
        return (
            pd.DataFrame(
                columns=[
                    "reference_problem_key",
                    "diagnostic_problem_key",
                    "definition_name",
                    "track_name",
                    "model_name",
                    "block_name",
                    "block_type",
                    "selected_feature_count",
                    "added_feature_count",
                    "selected_feature_names_json",
                    "added_feature_names_json",
                    "fold_id",
                    "fold_valid_flag",
                    "invalid_reason",
                    "rows",
                    "positives",
                    "ap",
                    "roc_auc",
                    "brier",
                    "log_loss",
                    "calibration_slope",
                    "calibration_intercept",
                    "calibration_slope_error",
                    "calibration_intercept_abs",
                ]
            ),
            pd.DataFrame(),
        )
    feature_meta = feature_registry.set_index("feature_name", drop=False)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    block_registry_rows: list[dict[str, Any]] = []
    definition_b_scenarios = scoring_scenarios[scoring_scenarios["definition_name"] == "definition_b_label"].copy()
    scenario_blocks: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for scenario in definition_b_scenarios.to_dict(orient="records"):
        allowed_feature_names = [
            feature_name
            for feature_name in json.loads(scenario["feature_names_json"])
            if feature_name in feature_meta.index
        ]
        if not allowed_feature_names:
            continue
        context_feature_names = [
            feature_name
            for feature_name in allowed_feature_names
            if str(feature_meta.loc[feature_name, "feature_class"]) == "context"
        ]
        if not context_feature_names:
            continue
        non_context_feature_names = [
            feature_name
            for feature_name in allowed_feature_names
            if feature_name not in context_feature_names
        ]
        block_specs: list[dict[str, Any]] = [
            {
                "block_name": "baseline_context_only",
                "block_type": "baseline",
                "selected_feature_names": sorted(context_feature_names),
                "added_feature_names": [],
            }
        ]
        for feature_class in sorted({str(feature_meta.loc[name, "feature_class"]) for name in non_context_feature_names}):
            block_feature_names = sorted(
                [
                    feature_name
                    for feature_name in non_context_feature_names
                    if str(feature_meta.loc[feature_name, "feature_class"]) == feature_class
                ]
            )
            if block_feature_names:
                block_specs.append(
                    {
                        "block_name": f"context_plus_feature_class::{feature_class}",
                        "block_type": "feature_class",
                        "selected_feature_names": sorted(set(context_feature_names + block_feature_names)),
                        "added_feature_names": block_feature_names,
                    }
                )
        for behavior_family in sorted({str(feature_meta.loc[name, "behavior_family"]) for name in non_context_feature_names}):
            block_feature_names = sorted(
                [
                    feature_name
                    for feature_name in non_context_feature_names
                    if str(feature_meta.loc[feature_name, "behavior_family"]) == behavior_family
                ]
            )
            if block_feature_names:
                block_specs.append(
                    {
                        "block_name": f"context_plus_behavior_family::{behavior_family}",
                        "block_type": "behavior_family",
                        "selected_feature_names": sorted(set(context_feature_names + block_feature_names)),
                        "added_feature_names": block_feature_names,
                    }
                )
        if non_context_feature_names:
            block_specs.append(
                {
                    "block_name": "full_allowed_features",
                    "block_type": "full_track_features",
                    "selected_feature_names": sorted(allowed_feature_names),
                    "added_feature_names": sorted(non_context_feature_names),
                }
            )
        unique_blocks: list[dict[str, Any]] = []
        seen_feature_sets: set[str] = set()
        for block in block_specs:
            feature_signature = setup.stable_json(block["selected_feature_names"])
            if feature_signature in seen_feature_sets:
                continue
            seen_feature_sets.add(feature_signature)
            unique_blocks.append(block)
        if unique_blocks:
            scenario_blocks.append((scenario, unique_blocks))

    total_blocks = sum(len(blocks) for _, blocks in scenario_blocks)
    completed_blocks = 0
    if progress_callback:
        progress_callback(
            "definition_b_feature_block_gain",
            0,
            total_blocks,
            "iniciando blocos da Definição B",
        )

    for scenario, unique_blocks in scenario_blocks:
        working = frame.dropna(subset=["first_month"]).copy()
        working["y_true"] = pd.to_numeric(working[scenario["label_col"]], errors="coerce")
        working = working[working["y_true"].notna()].copy()
        working["y_true"] = working["y_true"].astype(int)
        if working.empty:
            continue
        for block in unique_blocks:
            if progress_callback:
                progress_callback(
                    "definition_b_feature_block_gain",
                    completed_blocks,
                    total_blocks,
                    f'{scenario["problem_key"]} | {block["block_name"]}',
                )
            diagnostic_problem_key = f'{scenario["problem_key"]}::feature_block::{block["block_name"]}'
            block_registry_rows.append(
                {
                    "reference_problem_key": scenario["problem_key"],
                    "diagnostic_problem_key": diagnostic_problem_key,
                    "definition_name": scenario["definition_name"],
                    "track_name": scenario["track_name"],
                    "model_name": logistic_spec["model_name"],
                    "block_name": block["block_name"],
                    "block_type": block["block_type"],
                    "selected_feature_count": int(len(block["selected_feature_names"])),
                    "added_feature_count": int(len(block["added_feature_names"])),
                    "selected_feature_names_json": setup.stable_json(block["selected_feature_names"]),
                    "added_feature_names_json": setup.stable_json(block["added_feature_names"]),
                }
            )
            problem = {
                "problem_key": diagnostic_problem_key,
                "definition_name": scenario["definition_name"],
                "track_name": scenario["track_name"],
                "label_col": scenario["label_col"],
            }
            result = run_or_load_model_task(
                problem=problem,
                model_spec=logistic_spec,
                working=working,
                feature_registry=feature_registry,
                feature_names=block["selected_feature_names"],
                compute_feature_importance=False,
                task_store=task_store,
            )
            model_fold_rows, model_prediction_rows, _, _, _ = result
            for row in model_fold_rows:
                row["reference_problem_key"] = scenario["problem_key"]
                row["block_name"] = block["block_name"]
                row["block_type"] = block["block_type"]
                row["selected_feature_count"] = int(len(block["selected_feature_names"]))
                row["added_feature_count"] = int(len(block["added_feature_names"]))
                row["selected_feature_names_json"] = setup.stable_json(block["selected_feature_names"])
                row["added_feature_names_json"] = setup.stable_json(block["added_feature_names"])
                row["diagnostic_problem_key"] = diagnostic_problem_key
                fold_rows.append(row)
            for row in model_prediction_rows:
                row["reference_problem_key"] = scenario["problem_key"]
                row["diagnostic_problem_key"] = diagnostic_problem_key
                row["block_name"] = block["block_name"]
                row["block_type"] = block["block_type"]
                row["selected_feature_count"] = int(len(block["selected_feature_names"]))
                row["added_feature_count"] = int(len(block["added_feature_names"]))
                row["selected_feature_names_json"] = setup.stable_json(block["selected_feature_names"])
                row["added_feature_names_json"] = setup.stable_json(block["added_feature_names"])
                prediction_rows.append(row)
            completed_blocks += 1
            if progress_callback:
                progress_callback(
                    "definition_b_feature_block_gain",
                    completed_blocks,
                    total_blocks,
                    f'{scenario["problem_key"]} | {block["block_name"]} concluído',
                )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)
    if fold_df.empty:
        return fold_df, pd.DataFrame(block_registry_rows)

    summary = summarize_model_performance(
        fold_df,
        pred_df,
        group_keys=[
            "reference_problem_key",
            "diagnostic_problem_key",
            "definition_name",
            "track_name",
            "model_name",
            "block_name",
            "block_type",
            "selected_feature_count",
            "added_feature_count",
            "selected_feature_names_json",
            "added_feature_names_json",
        ],
    )
    if summary.empty:
        return fold_df, summary
    required_summary_columns = {"reference_problem_key", "model_name", "block_name"}
    missing_summary_columns = required_summary_columns.difference(summary.columns)
    if missing_summary_columns:
        raise KeyError(
            "definition_b_feature_block_gain_summary_missing_columns:"
            + ",".join(sorted(missing_summary_columns))
        )

    baseline = (
        summary[summary["block_name"] == "baseline_context_only"][
            [
                "reference_problem_key",
                "model_name",
                "mean_ap",
                "mean_roc_auc",
                "mean_brier",
                "mean_log_loss",
            ]
        ]
        .rename(
            columns={
                "mean_ap": "baseline_mean_ap",
                "mean_roc_auc": "baseline_mean_roc_auc",
                "mean_brier": "baseline_mean_brier",
                "mean_log_loss": "baseline_mean_log_loss",
            }
        )
        .drop_duplicates()
    )
    summary = summary.merge(
        baseline,
        on=["reference_problem_key", "model_name"],
        how="left",
    )
    summary["delta_ap_vs_context"] = summary["mean_ap"] - summary["baseline_mean_ap"]
    summary["delta_roc_auc_vs_context"] = summary["mean_roc_auc"] - summary["baseline_mean_roc_auc"]
    summary["brier_improvement_vs_context"] = summary["baseline_mean_brier"] - summary["mean_brier"]
    summary["log_loss_improvement_vs_context"] = summary["baseline_mean_log_loss"] - summary["mean_log_loss"]
    summary["uplift_metric_positive_count"] = (
        (summary["delta_ap_vs_context"] > 0).astype(int)
        + (summary["delta_roc_auc_vs_context"] > 0).astype(int)
        + (summary["brier_improvement_vs_context"] > 0).astype(int)
        + (summary["log_loss_improvement_vs_context"] > 0).astype(int)
    )

    uplift_metrics = [
        "delta_ap_vs_context",
        "delta_roc_auc_vs_context",
        "brier_improvement_vs_context",
        "log_loss_improvement_vs_context",
    ]
    summary["mean_uplift_percentile"] = np.nan
    summary["abnormal_uplift_flag"] = 0
    for (_, model_name), group in summary.groupby(["reference_problem_key", "model_name"], dropna=False):
        candidate_index = group.index[group["block_name"] != "baseline_context_only"].tolist()
        if not candidate_index:
            continue
        percentiles = []
        for metric_name in uplift_metrics:
            ranked = group.loc[candidate_index, metric_name].rank(method="average", pct=True)
            summary.loc[candidate_index, f"{metric_name}_percentile"] = ranked.to_numpy(dtype=float)
            percentiles.append(ranked.to_numpy(dtype=float))
        if percentiles:
            mean_percentile = np.nanmean(np.vstack(percentiles), axis=0)
            summary.loc[candidate_index, "mean_uplift_percentile"] = mean_percentile
            summary.loc[candidate_index, "abnormal_uplift_flag"] = (
                (mean_percentile >= 0.90)
                & (summary.loc[candidate_index, "valid_folds"].to_numpy(dtype=int) >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS)
                & (summary.loc[candidate_index, "uplift_metric_positive_count"].to_numpy(dtype=int) >= 3)
            ).astype(int)
    return fold_df, summary

def evaluate_model_problems(
    frame: pd.DataFrame,
    feature_registry: pd.DataFrame,
    scoring_scenarios: pd.DataFrame,
    allowed_problem_model_pairs: set[tuple[str, str]] | None = None,
    compute_feature_importance: bool = False,
    task_store: TaskArtifactStore | None = None,
    progress_stage_key: str = "model_evaluation",
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_specs = build_model_specs()
    scenario_records = scoring_scenarios.to_dict(orient="records")
    task_inputs: list[dict[str, Any]] = []
    for scenario in scenario_records:
        print(f'[model] scenario {scenario["problem_key"]}', flush=True)
        feature_names = json.loads(scenario["feature_names_json"])
        working = frame.dropna(subset=["first_month"]).copy()
        working["y_true"] = pd.to_numeric(working[scenario["label_col"]], errors="coerce")
        working = working[working["y_true"].notna()].copy()
        working["y_true"] = working["y_true"].astype(int)
        if working.empty:
            continue
        selected_specs = [
            spec for spec in model_specs
            if allowed_problem_model_pairs is None or (scenario["problem_key"], spec["model_name"]) in allowed_problem_model_pairs
        ]
        for model_spec in selected_specs:
            task_inputs.append(
                {
                    "problem": scenario,
                    "model_spec": model_spec,
                    "working": working,
                    "feature_names": feature_names,
                }
            )
    total_tasks = len(task_inputs)
    completed_tasks = 0
    if progress_callback:
        progress_callback(progress_stage_key, 0, total_tasks, "iniciando avaliação de modelos")
    fold_rows: List[dict[str, Any]] = []
    prediction_rows: List[dict[str, Any]] = []
    inner_audit_rows: List[dict[str, Any]] = []
    importance_rows: List[dict[str, Any]] = []
    post_model_output_status_rows: List[dict[str, Any]] = []
    max_workers = min(setup.MODEL_COMPARISON_WORKERS, max(1, total_tasks))
    if total_tasks == 0:
        max_workers = 0
    if progress_callback and task_inputs:
        progress_callback(
            progress_stage_key,
            0,
            total_tasks,
            f'{len(task_inputs)} tarefas globais | até {max_workers} workers',
        )
    if task_inputs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    run_or_load_model_task,
                    task["problem"],
                    task["model_spec"],
                    task["working"],
                    feature_registry,
                    task["feature_names"],
                    compute_feature_importance,
                    task_store,
                ): task
                for task in task_inputs
            }
            for future in as_completed(future_map):
                task = future_map[future]
                scenario = task["problem"]
                model_spec = task["model_spec"]
                model_fold_rows, model_prediction_rows, model_inner_rows, model_importance_rows, model_post_model_output_status_rows = future.result()
                fold_rows.extend(model_fold_rows)
                prediction_rows.extend(model_prediction_rows)
                inner_audit_rows.extend(model_inner_rows)
                importance_rows.extend(model_importance_rows)
                post_model_output_status_rows.extend(model_post_model_output_status_rows)
                completed_tasks += 1
                if progress_callback:
                    progress_callback(
                        progress_stage_key,
                        completed_tasks,
                        total_tasks,
                        f'{scenario["problem_key"]} | {model_spec["model_name"]} concluído',
                    )
    fold_df = pd.DataFrame(fold_rows)
    inner_audit_df = pd.DataFrame(
        inner_audit_rows,
        columns=[
            "problem_key",
            "definition_name",
            "track_name",
            "model_name",
            "outer_fold_id",
            "inner_fold_id",
            "split_strategy",
            "validation_start_month",
            "validation_month_count",
            "calibration_start_month",
            "calibration_month_count",
            "train_rows",
            "test_rows",
            "train_positives",
            "test_positives",
            "valid_inner_split_flag",
            "invalid_reason",
        ],
    )
    pred_df = pd.DataFrame(
        prediction_rows,
        columns=[
            "teacher_unique_id",
            "first_month",
            "y_true",
            "problem_key",
            "definition_name",
            "track_name",
            "model_name",
            "fold_id",
            "score",
            "technical_fold_valid_flag",
            "fold_valid_flag",
            "invalid_reason",
        ],
    )
    importance_df = pd.DataFrame(
        importance_rows,
        columns=[
            "problem_key",
            "definition_name",
            "track_name",
            "model_name",
            "fold_id",
            "feature_name",
            "importance_mean",
            "importance_std",
        ],
    )
    post_model_output_status_df = pd.DataFrame(
        post_model_output_status_rows,
        columns=[
            "problem_key",
            "definition_name",
            "track_name",
            "model_name",
            "fold_id",
            "post_model_output_name",
            "post_model_output_status",
            "error_type",
            "error_message",
            "traceback_snippet",
        ],
    )
    summary_df = summarize_model_performance(fold_df, pred_df)
    if summary_df.empty:
        return fold_df, pred_df, pd.DataFrame(
            columns=[
                "problem_key",
                "definition_name",
                "track_name",
                "model_name",
                "valid_folds",
                "pooled_rows",
                "pooled_positives",
                "pooled_negatives",
                "pooled_positive_rate",
                "mean_ap",
                "std_ap",
                "mean_roc_auc",
                "std_roc_auc",
                "mean_brier",
                "std_brier",
                "mean_log_loss",
                "std_log_loss",
                "mean_calibration_slope",
                "mean_calibration_intercept",
                "mean_calibration_slope_error",
                "mean_calibration_intercept_abs",
                "fold_mean_ap",
                "fold_mean_roc_auc",
                "fold_mean_brier",
                "fold_mean_log_loss",
                "fold_mean_calibration_slope",
                "fold_mean_calibration_intercept",
                "fold_mean_calibration_slope_error",
                "fold_mean_calibration_intercept_abs",
                "pareto_frontier_flag",
            ]
        ), inner_audit_df, importance_df, post_model_output_status_df
    summary_df = summary_df[summary_df["valid_folds"] >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS].copy()
    if summary_df.empty:
        return fold_df, pred_df, pd.DataFrame(
            columns=[
                "problem_key",
                "definition_name",
                "track_name",
                "model_name",
                "valid_folds",
                "pooled_rows",
                "pooled_positives",
                "pooled_negatives",
                "pooled_positive_rate",
                "mean_ap",
                "std_ap",
                "mean_roc_auc",
                "std_roc_auc",
                "mean_brier",
                "std_brier",
                "mean_log_loss",
                "std_log_loss",
                "mean_calibration_slope",
                "mean_calibration_intercept",
                "mean_calibration_slope_error",
                "mean_calibration_intercept_abs",
                "fold_mean_ap",
                "fold_mean_roc_auc",
                "fold_mean_brier",
                "fold_mean_log_loss",
                "fold_mean_calibration_slope",
                "fold_mean_calibration_intercept",
                "fold_mean_calibration_slope_error",
                "fold_mean_calibration_intercept_abs",
                "pareto_frontier_flag",
            ]
        ), inner_audit_df, importance_df, post_model_output_status_df
    frontier = pareto_front(summary_df, setup.OFFICIAL_METRIC_OBJECTIVES)
    return fold_df, pred_df, frontier, inner_audit_df, importance_df, post_model_output_status_df
