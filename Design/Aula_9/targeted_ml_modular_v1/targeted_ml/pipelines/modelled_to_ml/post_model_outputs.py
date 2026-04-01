"""Saídas pós-modelo: robustez, operação, cluster, heavy-user e navegação."""

from __future__ import annotations

import json
from typing import Any, List, Sequence

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from . import analysis_setup as setup
from .selection import select_serving_scope
from targeted_ml.modeling.preprocessing import FeatureSchema, build_column_transformer
from .modeling import filter_official_predictions

def regression_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2 or np.unique(x_arr[mask]).size < 2:
        return float("nan")
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    x_centered = x_use - x_use.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x_centered, y_use - y_use.mean()) / denom)

def permutation_slope_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    n_permutations: int = setup.PUBLISHED_PVALUE_PERMUTATIONS,
    random_state: int = 42,
) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3 or np.unique(x_arr[mask]).size < 2:
        return float("nan")
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    observed = abs(regression_slope(x_use, y_use))
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(random_state)
    exceed = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(y_use)
        slope = abs(regression_slope(x_use, permuted))
        if np.isfinite(slope) and slope >= observed:
            exceed += 1
    return float((exceed + 1) / (n_permutations + 1))

def resolve_registered_policy(scores: pd.Series, policy_name: str) -> dict[str, Any]:
    clean = pd.to_numeric(scores, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if clean.empty:
        return {"threshold": float("nan"), "low_edge": float("nan"), "high_edge": float("nan")}
    if policy_name == "top_10_percent":
        threshold = float(clean.quantile(0.90, interpolation="linear"))
        return {"threshold": threshold, "low_edge": float("nan"), "high_edge": threshold}
    if policy_name == "tercis":
        low_edge = float(clean.quantile(1 / 3, interpolation="linear"))
        high_edge = float(clean.quantile(2 / 3, interpolation="linear"))
        return {"threshold": high_edge, "low_edge": low_edge, "high_edge": high_edge}
    if policy_name == "score_ge_0_70":
        return {"threshold": 0.70, "low_edge": float("nan"), "high_edge": 0.70}
    raise ValueError(f"Unsupported registered policy: {policy_name}")

def compute_threshold_policy_outputs(
    group: pd.DataFrame,
    grouping_values: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for policy in setup.REGISTERED_BAND_POLICIES:
        resolved = resolve_registered_policy(group["risk_score"], policy["policy_name"])
        threshold = float(resolved["threshold"])
        y_true = group["y_risk_true"].to_numpy(dtype=int)
        y_pred = (group["risk_score"].to_numpy(dtype=float) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metric_rows.append(
            {
                **grouping_values,
                "policy_name": policy["policy_name"],
                "risk_threshold": threshold,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else float("nan"),
            }
        )
        for actual_name, predicted_name, value in [
            ("nao_realiza", "nao_realiza", tp),
            ("realiza", "nao_realiza", fp),
            ("nao_realiza", "realiza", fn),
            ("realiza", "realiza", tn),
        ]:
            confusion_rows.append(
                {
                    **grouping_values,
                    "policy_name": policy["policy_name"],
                    "actual_group": actual_name,
                    "predicted_group": predicted_name,
                    "rows": int(value),
                }
            )

        if policy["policy_name"] == "tercis":
            low_edge = resolved["low_edge"]
            high_edge = resolved["high_edge"]
            if pd.isna(low_edge) or pd.isna(high_edge) or low_edge >= high_edge:
                banded = pd.Series(np.where(group["risk_score"] >= threshold, "alto", "baixo"), index=group.index)
            else:
                banded = pd.cut(
                    group["risk_score"],
                    bins=[-np.inf, low_edge, high_edge, np.inf],
                    labels=["baixo", "medio", "alto"],
                    include_lowest=True,
                )
            for band_name, sub in group.assign(band_name=banded).groupby("band_name", dropna=False, observed=False):
                band_rows.append(
                    {
                        **grouping_values,
                        "policy_name": policy["policy_name"],
                        "band_name": str(band_name),
                        "rows": int(len(sub)),
                        "share": float(len(sub) / len(group)) if len(group) else float("nan"),
                    }
                )
        else:
            high_risk = group["risk_score"] >= threshold
            for band_name, mask in [("alto", high_risk), ("demais", ~high_risk)]:
                sub = group.loc[mask]
                band_rows.append(
                    {
                        **grouping_values,
                        "policy_name": policy["policy_name"],
                        "band_name": band_name,
                        "rows": int(len(sub)),
                        "share": float(len(sub) / len(group)) if len(group) else float("nan"),
                    }
                )
    return metric_rows, confusion_rows, band_rows

def build_threshold_post_model_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = filter_official_predictions(predictions)
    if predictions.empty:
        empty_threshold = pd.DataFrame(
            columns=[
                "problem_key",
                "model_name",
                "policy_name",
                "risk_threshold",
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "predicted_positive_rate",
            ]
        )
        empty_band = pd.DataFrame(columns=["problem_key", "model_name", "policy_name", "band_name", "rows", "share"])
        empty_month = pd.DataFrame(columns=["problem_key", "model_name", "monthly_r2", "monthly_mape_positive_months", "months_used"])
        return empty_threshold, empty_threshold.copy(), empty_band, empty_month

    work = predictions.copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["risk_score"] = 1.0 - work["score"]
    work["y_risk_true"] = 1 - pd.to_numeric(work["y_true"], errors="coerce").fillna(0).astype(int)

    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []

    for (problem_key, model_name), group in work.groupby(["problem_key", "model_name"], dropna=False):
        monthly = (
            group.groupby("first_month", as_index=False)
            .agg(realized_risk_rate=("y_risk_true", "mean"), predicted_risk_rate=("risk_score", "mean"))
            .dropna()
        )
        valid_monthly = monthly[monthly["realized_risk_rate"] > 0].copy()
        monthly_rows.append(
            {
                "problem_key": problem_key,
                "model_name": model_name,
                "monthly_r2": float(r2_score(monthly["realized_risk_rate"], monthly["predicted_risk_rate"])) if len(monthly) >= 2 else float("nan"),
                "monthly_mape_positive_months": float(mean_absolute_percentage_error(valid_monthly["realized_risk_rate"], valid_monthly["predicted_risk_rate"])) if len(valid_monthly) >= 1 else float("nan"),
                "months_used": int(len(monthly)),
            }
        )
        add_metric_rows, add_confusion_rows, add_band_rows = compute_threshold_policy_outputs(
            group,
            {"problem_key": problem_key, "model_name": model_name},
        )
        metric_rows.extend(add_metric_rows)
        confusion_rows.extend(add_confusion_rows)
        band_rows.extend(add_band_rows)

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(band_rows),
        pd.DataFrame(monthly_rows),
    )

def build_cv_score_robustness_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = filter_official_predictions(predictions)
    if predictions.empty:
        return (
            pd.DataFrame(columns=["problem_key", "model_name", "fold_id", "rows", "positives", "mean_score", "mean_risk_score", "realized_risk_rate", "score_std", "risk_score_std"]),
            pd.DataFrame(columns=["problem_key", "model_name", "metric_name", "valid_folds", "mean_value", "std_value", "min_value", "max_value", "value_range", "max_fold_to_fold_jump", "fold_order_slope", "fold_order_pvalue"]),
        )
    work = predictions.copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["risk_score"] = 1.0 - work["score"]
    work["y_risk_true"] = 1 - pd.to_numeric(work["y_true"], errors="coerce").fillna(0).astype(int)
    fold_df = (
        work.groupby(["problem_key", "model_name", "fold_id"], as_index=False)
        .agg(
            rows=("teacher_unique_id", "size"),
            positives=("y_true", "sum"),
            mean_score=("score", "mean"),
            mean_risk_score=("risk_score", "mean"),
            realized_risk_rate=("y_risk_true", "mean"),
            score_std=("score", "std"),
            risk_score_std=("risk_score", "std"),
        )
    )
    for col in ["score_std", "risk_score_std"]:
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce").fillna(0.0)
    metric_cols = ["mean_score", "mean_risk_score", "realized_risk_rate", "score_std", "risk_score_std"]
    summary_rows: list[dict[str, Any]] = []
    for (problem_key, model_name), group in fold_df.groupby(["problem_key", "model_name"], dropna=False):
        ordered = group.sort_values("fold_id")
        fold_ids = ordered["fold_id"].to_numpy(dtype=float)
        for metric_name in metric_cols:
            values = pd.to_numeric(ordered[metric_name], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() == 0:
                continue
            use_values = values[finite]
            use_folds = fold_ids[finite]
            jumps = np.abs(np.diff(use_values))
            summary_rows.append(
                {
                    "problem_key": problem_key,
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "valid_folds": int(len(use_values)),
                    "mean_value": float(np.mean(use_values)),
                    "std_value": float(np.std(use_values, ddof=0)) if len(use_values) > 1 else 0.0,
                    "min_value": float(np.min(use_values)),
                    "max_value": float(np.max(use_values)),
                    "value_range": float(np.max(use_values) - np.min(use_values)),
                    "max_fold_to_fold_jump": float(np.max(jumps)) if len(jumps) else 0.0,
                    "fold_order_slope": regression_slope(use_folds, use_values),
                    "fold_order_pvalue": permutation_slope_pvalue(use_folds, use_values),
                }
            )
    return fold_df, pd.DataFrame(summary_rows)

def build_cv_metric_robustness_outputs(model_fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = model_fold_metrics[model_fold_metrics.get("fold_valid_flag", 0) == 1].copy()
    if valid.empty:
        empty_cols = ["problem_key", "definition_name", "track_name", "model_name", "fold_id", "metric_name", "metric_value"]
        summary_cols = ["problem_key", "definition_name", "track_name", "model_name", "metric_name", "valid_folds", "mean_value", "std_value", "min_value", "max_value", "value_range", "max_fold_to_fold_jump", "fold_order_slope", "fold_order_pvalue"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=summary_cols)
    metric_cols = [
        "ap",
        "roc_auc",
        "brier",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
        "calibration_slope_error",
        "calibration_intercept_abs",
    ]
    fold_long = (
        valid[["problem_key", "definition_name", "track_name", "model_name", "fold_id"] + metric_cols]
        .melt(
            id_vars=["problem_key", "definition_name", "track_name", "model_name", "fold_id"],
            value_vars=metric_cols,
            var_name="metric_name",
            value_name="metric_value",
        )
        .dropna(subset=["metric_value"])
    )
    summary_rows: list[dict[str, Any]] = []
    for keys, group in fold_long.groupby(["problem_key", "definition_name", "track_name", "model_name", "metric_name"], dropna=False):
        ordered = group.sort_values("fold_id")
        values = pd.to_numeric(ordered["metric_value"], errors="coerce").to_numpy(dtype=float)
        folds = ordered["fold_id"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            continue
        use_values = values[finite]
        use_folds = folds[finite]
        jumps = np.abs(np.diff(use_values))
        problem_key, definition_name, track_name, model_name, metric_name = keys
        summary_rows.append(
            {
                "problem_key": problem_key,
                "definition_name": definition_name,
                "track_name": track_name,
                "model_name": model_name,
                "metric_name": metric_name,
                "valid_folds": int(len(use_values)),
                "mean_value": float(np.mean(use_values)),
                "std_value": float(np.std(use_values, ddof=0)) if len(use_values) > 1 else 0.0,
                "min_value": float(np.min(use_values)),
                "max_value": float(np.max(use_values)),
                "value_range": float(np.max(use_values) - np.min(use_values)),
                "max_fold_to_fold_jump": float(np.max(jumps)) if len(jumps) else 0.0,
                "fold_order_slope": regression_slope(use_folds, use_values),
                "fold_order_pvalue": permutation_slope_pvalue(use_folds, use_values),
            }
        )
    return fold_long, pd.DataFrame(summary_rows)

def build_cv_threshold_robustness_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = filter_official_predictions(predictions)
    if predictions.empty:
        metric_cols = [
            "problem_key",
            "model_name",
            "fold_id",
            "policy_name",
            "risk_threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "predicted_positive_rate",
        ]
        confusion_cols = ["problem_key", "model_name", "fold_id", "policy_name", "actual_group", "predicted_group", "rows"]
        summary_cols = ["problem_key", "model_name", "policy_name", "metric_name", "valid_folds", "mean_value", "std_value", "min_value", "max_value", "value_range", "max_fold_to_fold_jump", "fold_order_slope", "fold_order_pvalue"]
        confusion_summary_cols = ["problem_key", "model_name", "policy_name", "actual_group", "predicted_group", "valid_folds", "mean_rows", "std_rows", "min_rows", "max_rows", "max_fold_to_fold_jump", "fold_order_slope", "fold_order_pvalue"]
        return pd.DataFrame(columns=metric_cols), pd.DataFrame(columns=confusion_cols), pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=confusion_summary_cols)
    work = predictions.copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["risk_score"] = 1.0 - work["score"]
    work["y_risk_true"] = 1 - pd.to_numeric(work["y_true"], errors="coerce").fillna(0).astype(int)

    fold_metric_rows: list[dict[str, Any]] = []
    fold_confusion_rows: list[dict[str, Any]] = []
    for (problem_key, model_name, fold_id), group in work.groupby(["problem_key", "model_name", "fold_id"], dropna=False):
        add_metrics, add_confusion, _ = compute_threshold_policy_outputs(
            group,
            {"problem_key": problem_key, "model_name": model_name, "fold_id": int(fold_id)},
        )
        fold_metric_rows.extend(add_metrics)
        fold_confusion_rows.extend(add_confusion)
    fold_metric_df = pd.DataFrame(fold_metric_rows)
    fold_confusion_df = pd.DataFrame(fold_confusion_rows)

    metric_summary_rows: list[dict[str, Any]] = []
    metric_cols = ["risk_threshold", "precision", "recall", "f1", "accuracy", "predicted_positive_rate"]
    if not fold_metric_df.empty:
        for keys, group in fold_metric_df.groupby(["problem_key", "model_name", "policy_name"], dropna=False):
            ordered = group.sort_values("fold_id")
            fold_ids = ordered["fold_id"].to_numpy(dtype=float)
            for metric_name in metric_cols:
                values = pd.to_numeric(ordered[metric_name], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(values)
                if finite.sum() == 0:
                    continue
                use_values = values[finite]
                use_folds = fold_ids[finite]
                jumps = np.abs(np.diff(use_values))
                problem_key, model_name, policy_name = keys
                metric_summary_rows.append(
                    {
                        "problem_key": problem_key,
                        "model_name": model_name,
                        "policy_name": policy_name,
                        "metric_name": metric_name,
                        "valid_folds": int(len(use_values)),
                        "mean_value": float(np.mean(use_values)),
                        "std_value": float(np.std(use_values, ddof=0)) if len(use_values) > 1 else 0.0,
                        "min_value": float(np.min(use_values)),
                        "max_value": float(np.max(use_values)),
                        "value_range": float(np.max(use_values) - np.min(use_values)),
                        "max_fold_to_fold_jump": float(np.max(jumps)) if len(jumps) else 0.0,
                        "fold_order_slope": regression_slope(use_folds, use_values),
                        "fold_order_pvalue": permutation_slope_pvalue(use_folds, use_values),
                    }
                )
    confusion_summary_rows: list[dict[str, Any]] = []
    if not fold_confusion_df.empty:
        for keys, group in fold_confusion_df.groupby(["problem_key", "model_name", "policy_name", "actual_group", "predicted_group"], dropna=False):
            ordered = group.sort_values("fold_id")
            values = pd.to_numeric(ordered["rows"], errors="coerce").to_numpy(dtype=float)
            folds = ordered["fold_id"].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() == 0:
                continue
            use_values = values[finite]
            use_folds = folds[finite]
            jumps = np.abs(np.diff(use_values))
            problem_key, model_name, policy_name, actual_group, predicted_group = keys
            confusion_summary_rows.append(
                {
                    "problem_key": problem_key,
                    "model_name": model_name,
                    "policy_name": policy_name,
                    "actual_group": actual_group,
                    "predicted_group": predicted_group,
                    "valid_folds": int(len(use_values)),
                    "mean_rows": float(np.mean(use_values)),
                    "std_rows": float(np.std(use_values, ddof=0)) if len(use_values) > 1 else 0.0,
                    "min_rows": float(np.min(use_values)),
                    "max_rows": float(np.max(use_values)),
                    "max_fold_to_fold_jump": float(np.max(jumps)) if len(jumps) else 0.0,
                    "fold_order_slope": regression_slope(use_folds, use_values),
                    "fold_order_pvalue": permutation_slope_pvalue(use_folds, use_values),
                }
            )
    return fold_metric_df, fold_confusion_df, pd.DataFrame(metric_summary_rows), pd.DataFrame(confusion_summary_rows)


def build_definition_b_excessive_separation_outputs(model_frontier: pd.DataFrame) -> pd.DataFrame:
    summary = model_frontier.copy()
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "problem_key",
                "definition_name",
                "track_name",
                "model_name",
                "valid_folds",
                "mean_ap",
                "mean_roc_auc",
                "mean_brier",
                "mean_log_loss",
                "mean_calibration_slope_error",
                "mean_calibration_intercept_abs",
                "std_ap",
                "std_brier",
                "std_log_loss",
                "ap_good_percentile",
                "roc_auc_good_percentile",
                "brier_good_percentile",
                "log_loss_good_percentile",
                "calibration_good_percentile",
                "stability_good_percentile",
                "combined_separation_score",
                "combined_separation_percentile_within_track",
                "comparator_rows_in_track",
                "red_flag_eligible_flag",
                "excessive_separation_red_flag",
            ]
        )
    summary = summary.copy()
    summary["std_ap"] = pd.to_numeric(summary["std_ap"], errors="coerce").fillna(0.0)
    summary["std_brier"] = pd.to_numeric(summary["std_brier"], errors="coerce").fillna(0.0)
    summary["std_log_loss"] = pd.to_numeric(summary["std_log_loss"], errors="coerce").fillna(0.0)
    summary["calibration_quality"] = (
        pd.to_numeric(summary["mean_calibration_slope_error"], errors="coerce").fillna(np.nan)
        + pd.to_numeric(summary["mean_calibration_intercept_abs"], errors="coerce").fillna(np.nan)
    ) / 2.0
    summary["stability_quality"] = (
        summary["std_ap"].fillna(np.nan)
        + summary["std_brier"].fillna(np.nan)
        + summary["std_log_loss"].fillna(np.nan)
    ) / 3.0

    def assign_percentile(group: pd.DataFrame, column: str, higher_is_better: bool) -> pd.Series:
        series = pd.to_numeric(group[column], errors="coerce")
        if series.notna().sum() == 0:
            return pd.Series(np.nan, index=group.index)
        return series.rank(method="average", pct=True, ascending=higher_is_better)

    result_groups: list[pd.DataFrame] = []
    for _, group in summary.groupby("track_name", dropna=False):
        group = group.copy()
        comparator_count = int(len(group))
        group["ap_good_percentile"] = assign_percentile(group, "mean_ap", higher_is_better=True)
        group["roc_auc_good_percentile"] = assign_percentile(group, "mean_roc_auc", higher_is_better=True)
        group["brier_good_percentile"] = assign_percentile(group, "mean_brier", higher_is_better=False)
        group["log_loss_good_percentile"] = assign_percentile(group, "mean_log_loss", higher_is_better=False)
        group["calibration_good_percentile"] = assign_percentile(group, "calibration_quality", higher_is_better=False)
        group["stability_good_percentile"] = assign_percentile(group, "stability_quality", higher_is_better=False)
        group["combined_separation_score"] = group[
            [
                "ap_good_percentile",
                "roc_auc_good_percentile",
                "brier_good_percentile",
                "log_loss_good_percentile",
                "calibration_good_percentile",
                "stability_good_percentile",
            ]
        ].mean(axis=1)
        group["combined_separation_percentile_within_track"] = group["combined_separation_score"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        track_threshold = float(group["combined_separation_score"].quantile(0.95, interpolation="linear"))
        group["comparator_rows_in_track"] = comparator_count
        group["red_flag_eligible_flag"] = (
            (pd.to_numeric(group["valid_folds"], errors="coerce").fillna(0).astype(int) >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS)
            & (comparator_count >= 5)
        ).astype(int)
        group["excessive_separation_red_flag"] = (
            (group["definition_name"] == "definition_b_label")
            & (group["combined_separation_score"] >= track_threshold)
            & (group["red_flag_eligible_flag"] == 1)
        ).astype(int)
        result_groups.append(group)
    result = pd.concat(result_groups, ignore_index=True) if result_groups else summary.iloc[0:0].copy()
    return result[
        [
            "problem_key",
            "definition_name",
            "track_name",
            "model_name",
            "valid_folds",
            "mean_ap",
            "mean_roc_auc",
            "mean_brier",
            "mean_log_loss",
            "mean_calibration_slope_error",
            "mean_calibration_intercept_abs",
            "std_ap",
            "std_brier",
            "std_log_loss",
            "ap_good_percentile",
            "roc_auc_good_percentile",
            "brier_good_percentile",
            "log_loss_good_percentile",
            "calibration_good_percentile",
            "stability_good_percentile",
            "combined_separation_score",
            "combined_separation_percentile_within_track",
            "comparator_rows_in_track",
            "red_flag_eligible_flag",
            "excessive_separation_red_flag",
        ]
    ]

def build_cluster_outputs(
    conn: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_ready = conn.execute("SELECT * FROM mart_teacher_cluster_ready WHERE COALESCE(cluster_analysis_eligible_flag, 0) = 1").fetchdf()
    if cluster_ready.empty:
        return (
            pd.DataFrame(columns=["teacher_unique_id", "cluster_id", "cluster_name", "cluster_k", "cluster_silhouette"]),
            pd.DataFrame(columns=["cluster_name", "feature_name", "feature_mean", "cluster_rows"]),
            pd.DataFrame(columns=["problem_key", "model_name", "cluster_name", "rows", "mean_score", "mean_risk_score", "realized_inactivity_rate", "share"]),
            pd.DataFrame(columns=["iteration_id", "cluster_k", "silhouette", "stability_ari_vs_full", "selected_cluster_k", "selected_cluster_silhouette"]),
        )
    feature_cols = [col for col in cluster_ready.columns if col not in {"teacher_unique_id", "cluster_analysis_eligible_flag"}]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(cluster_ready[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    preprocessor = build_column_transformer(
        FeatureSchema(
            numeric_features=numeric_cols,
            categorical_features=categorical_cols,
        )
    )
    transformed = preprocessor.fit_transform(cluster_ready[feature_cols])
    silhouette_sample_size = min(setup.CLUSTER_SAMPLE_SIZE, len(cluster_ready))
    best_k = None
    best_labels = None
    best_silhouette = float("-inf")
    for k in setup.CLUSTER_K_CANDIDATES:
        if len(cluster_ready) <= k:
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(transformed)
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(transformed, labels, sample_size=silhouette_sample_size, random_state=42))
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k
            best_labels = labels
    if best_labels is None or best_k is None:
        return (
            pd.DataFrame(columns=["teacher_unique_id", "cluster_id", "cluster_name", "cluster_k", "cluster_silhouette"]),
            pd.DataFrame(columns=["cluster_name", "feature_name", "feature_mean", "cluster_rows"]),
            pd.DataFrame(columns=["problem_key", "model_name", "cluster_name", "rows", "mean_score", "mean_risk_score", "realized_inactivity_rate", "share"]),
            pd.DataFrame(columns=["iteration_id", "cluster_k", "silhouette", "stability_ari_vs_full", "selected_cluster_k", "selected_cluster_silhouette"]),
        )

    assignment = cluster_ready[["teacher_unique_id"]].copy()
    assignment["cluster_id"] = best_labels.astype(int)
    assignment["cluster_name"] = assignment["cluster_id"].map(lambda value: f"cluster_{int(value) + 1}")
    assignment["cluster_k"] = int(best_k)
    assignment["cluster_silhouette"] = float(best_silhouette)

    bootstrap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for iteration in range(1, setup.CLUSTER_BOOTSTRAP_ITERATIONS + 1):
        take_size = min(len(cluster_ready), max(best_k + 1, silhouette_sample_size))
        take = np.sort(rng.choice(len(cluster_ready), size=take_size, replace=False))
        if len(take) <= best_k:
            continue
        sample_transformed = transformed.iloc[take] if isinstance(transformed, pd.DataFrame) else transformed[take]
        sample_model = KMeans(n_clusters=best_k, random_state=42 + iteration, n_init=20)
        sample_labels = sample_model.fit_predict(sample_transformed)
        full_labels_on_take = best_labels[take]
        bootstrap_rows.append(
            {
                "iteration_id": iteration,
                "cluster_k": int(best_k),
                "silhouette": float(silhouette_score(sample_transformed, sample_labels, sample_size=min(setup.CLUSTER_SAMPLE_SIZE, len(sample_transformed)), random_state=42 + iteration))
                if len(np.unique(sample_labels)) > 1
                else float("nan"),
                "stability_ari_vs_full": float(adjusted_rand_score(full_labels_on_take, sample_labels)),
            }
        )

    validation = pd.DataFrame(bootstrap_rows)
    if not validation.empty:
        validation["selected_cluster_k"] = int(best_k)
        validation["selected_cluster_silhouette"] = float(best_silhouette)

    profile = (
        assignment.merge(cluster_ready, on="teacher_unique_id", how="left")
        .groupby(["cluster_name"], as_index=False)[numeric_cols]
        .mean()
        .melt(id_vars=["cluster_name"], var_name="feature_name", value_name="feature_mean")
    )
    profile = profile.merge(
        assignment.groupby("cluster_name", as_index=False).size().rename(columns={"size": "cluster_rows"}),
        on="cluster_name",
        how="left",
    )
    if predictions.empty:
        return (
            assignment,
            profile,
            pd.DataFrame(columns=["problem_key", "model_name", "cluster_name", "rows", "mean_score", "mean_risk_score", "realized_inactivity_rate", "share"]),
            validation,
        )
    scored = predictions.copy()
    scored["risk_score"] = 1.0 - pd.to_numeric(scored["score"], errors="coerce").fillna(0.0)
    summary = (
        scored.merge(assignment, on="teacher_unique_id", how="left")
        .dropna(subset=["cluster_name"])
        .groupby(["problem_key", "model_name", "cluster_name"], as_index=False)
        .agg(
            rows=("teacher_unique_id", "size"),
            mean_score=("score", "mean"),
            mean_risk_score=("risk_score", "mean"),
            realized_inactivity_rate=("y_true", lambda y: float((1 - pd.to_numeric(y, errors="coerce").fillna(0)).mean())),
        )
    )
    if not summary.empty:
        summary["share"] = summary.groupby(["problem_key", "model_name"])["rows"].transform(lambda values: values / values.sum())
    return assignment, profile, summary, validation

def resolve_percentile_cutoff(scores: pd.Series, top_share: float) -> float:
    clean = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    if clean.empty:
        return float("nan")
    return float(clean.quantile(max(0.0, 1.0 - top_share), interpolation="linear"))

def build_heavy_user_outputs(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "future_business_active_weeks",
        "future_sessions",
        "future_session_minutes",
        "future_active_days",
        "future_distinct_actions",
        "future_activity_events",
        "future_downloads",
        "future_content_views",
        "future_mapped_lessons",
        "future_formation_events",
    ]
    available = [col for col in metric_cols if col in frame.columns]
    base = frame[["teacher_unique_id"] + available].drop_duplicates("teacher_unique_id").copy()
    if base.empty or not available:
        return (
            pd.DataFrame(columns=["teacher_unique_id", "heavy_intensity_raw", "heavy_intensity_score", "heavy_intensity_pc_explained_variance", "policy_name", "heavy_cutoff", "heavy_user_flag"]),
            pd.DataFrame(columns=["heavy_user_flag", "metric_name", "metric_mean", "policy_name", "heavy_cutoff"]),
            pd.DataFrame(columns=["problem_key", "model_name", "policy_name", "heavy_user_flag", "rows", "mean_score", "mean_risk_score", "realized_inactivity_rate", "share"]),
        )
    matrix = np.log1p(base[available].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    pca = PCA(n_components=1, random_state=42)
    component = pca.fit_transform(scaled).reshape(-1)
    direction = np.corrcoef(component, matrix.sum(axis=1).to_numpy(dtype=float))[0, 1]
    if np.isfinite(direction) and direction < 0:
        component = -component
    percentile = pd.Series(component).rank(method="average", pct=True).to_numpy(dtype=float)
    heavy_base = base[["teacher_unique_id"]].copy()
    heavy_base["heavy_intensity_raw"] = component
    heavy_base["heavy_intensity_score"] = percentile
    heavy_base["heavy_intensity_pc_explained_variance"] = float(pca.explained_variance_ratio_[0])

    profile_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    summary_rows: list[pd.DataFrame] = []
    for policy in setup.HEAVY_USER_PERCENTILE_POLICIES:
        top_share = json.loads(policy["parameter_json"])["top_share"]
        cutoff = resolve_percentile_cutoff(heavy_base["heavy_intensity_score"], top_share)
        heavy_flag = (heavy_base["heavy_intensity_score"] >= cutoff).astype(int)
        policy_frame = heavy_base.copy()
        policy_frame["policy_name"] = policy["policy_name"]
        policy_frame["heavy_cutoff"] = cutoff
        policy_frame["heavy_user_flag"] = heavy_flag
        score_rows.extend(policy_frame.to_dict(orient="records"))
        if not predictions.empty:
            merged = predictions.merge(policy_frame[["teacher_unique_id", "policy_name", "heavy_user_flag"]], on="teacher_unique_id", how="left")
            merged["heavy_user_flag"] = merged["heavy_user_flag"].fillna(0).astype(int)
            merged["risk_score"] = 1.0 - pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
            summary = (
                merged.groupby(["problem_key", "model_name", "policy_name", "heavy_user_flag"], as_index=False)
                .agg(
                    rows=("teacher_unique_id", "size"),
                    mean_score=("score", "mean"),
                    mean_risk_score=("risk_score", "mean"),
                    realized_inactivity_rate=("y_true", lambda y: float((1 - pd.to_numeric(y, errors="coerce").fillna(0)).mean())),
                )
            )
            summary["share"] = summary.groupby(["problem_key", "model_name", "policy_name"])["rows"].transform(lambda values: values / values.sum())
            summary_rows.append(summary)
        metric_profile = (
            base.assign(heavy_user_flag=heavy_flag)
            .groupby("heavy_user_flag", as_index=False)[available]
            .mean()
            .melt(id_vars=["heavy_user_flag"], var_name="metric_name", value_name="metric_mean")
        )
        metric_profile["policy_name"] = policy["policy_name"]
        metric_profile["heavy_cutoff"] = cutoff
        profile_rows.extend(metric_profile.to_dict(orient="records"))
    return (
        pd.DataFrame(
            score_rows,
            columns=["teacher_unique_id", "heavy_intensity_raw", "heavy_intensity_score", "heavy_intensity_pc_explained_variance", "policy_name", "heavy_cutoff", "heavy_user_flag"],
        ),
        pd.DataFrame(
            profile_rows,
            columns=["heavy_user_flag", "metric_name", "metric_mean", "policy_name", "heavy_cutoff"],
        ),
        pd.concat(summary_rows, ignore_index=True)
        if summary_rows
        else pd.DataFrame(columns=["problem_key", "model_name", "policy_name", "heavy_user_flag", "rows", "mean_score", "mean_risk_score", "realized_inactivity_rate", "share"]),
    )

def select_reference_scope_for_post_model_outputs(
    frontier: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    definition_selection: pd.DataFrame | None = None,
    definition_frontier: pd.DataFrame | None = None,
    scoring_scenarios: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame(columns=["problem_key", "model_name", "selection_reason"])
    selected_scope, _, _ = select_serving_scope(
        model_frontier=frontier,
        model_predictions=predictions if predictions is not None else pd.DataFrame(columns=["problem_key", "model_name"]),
        definition_selection=definition_selection,
        definition_frontier=definition_frontier,
        scoring_scenarios=scoring_scenarios,
    )
    return selected_scope.drop_duplicates(subset=["problem_key", "model_name"])

def build_navigation_outputs(journey: pd.DataFrame, frame: pd.DataFrame, official_definition_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = journey.merge(frame[["teacher_unique_id"] + official_definition_names], on="teacher_unique_id", how="inner")
    sequence_rows: List[dict[str, Any]] = []
    transition_rows: List[dict[str, Any]] = []
    for definition_name in official_definition_names:
        for label_value, subset in merged.groupby(definition_name):
            seq = (
                subset.groupby("step_sequence_observed_first5", dropna=False)
                .agg(teachers=("teacher_unique_id", "nunique"))
                .reset_index()
                .sort_values("teachers", ascending=False)
                .head(20)
            )
            seq = seq.rename(columns={"step_sequence_observed_first5": "step_sequence_first5"})
            seq["definition_name"] = definition_name
            seq["label_value"] = int(label_value)
            sequence_rows.extend(seq.to_dict(orient="records"))
            tokens = subset[["teacher_unique_id", "step_1_token", "step_2_token", "step_3_token", "step_4_token", "step_5_token"]].copy()
            for _, row in tokens.iterrows():
                values = [setup.normalize_text(row[f"step_{idx}_token"]) for idx in range(1, 6)]
                values = [value for value in values if value != "missing"]
                for left, right in zip(values[:-1], values[1:]):
                    transition_rows.append(
                        {
                            "definition_name": definition_name,
                            "label_value": int(label_value),
                            "from_token": left,
                            "to_token": right,
                            "teacher_unique_id": row["teacher_unique_id"],
                        }
                    )
    transition_df = pd.DataFrame(transition_rows)
    if not transition_df.empty:
        transition_df = (
            transition_df.groupby(["definition_name", "label_value", "from_token", "to_token"], as_index=False)
            .agg(teachers=("teacher_unique_id", "nunique"))
            .sort_values(["definition_name", "label_value", "teachers"], ascending=[True, True, False])
        )
    return pd.DataFrame(sequence_rows), transition_df
