from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def normalize_definition_group(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("definition_a"):
        return "definition_a"
    if text.startswith("definition_b"):
        return "definition_b"
    return str(value or "").strip()


def candidate_definition_group(problem_key: Any, definition_name: Any) -> str:
    problem_text = str(problem_key or "")
    if problem_text.startswith("definition_a::"):
        return "definition_a"
    if problem_text.startswith("definition_b"):
        return "definition_b"
    return normalize_definition_group(definition_name)


def rank_primary_definition_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    ranked = candidates.copy()
    if "pareto_frontier_flag" in ranked.columns:
        pareto_flag = pd.to_numeric(ranked["pareto_frontier_flag"], errors="coerce").fillna(0).astype(int)
        if pareto_flag.any():
            ranked = ranked.loc[pareto_flag == 1].copy()
    if ranked.empty:
        return ranked
    numeric_defaults = {
        "folds": (False, float("-inf")),
        "test_gap_returned_active_post_label_m1": (False, float("-inf")),
        "test_gap_returned_active_post_label_m2": (False, float("-inf")),
        "test_gap_returned_active_post_label_m3": (False, float("-inf")),
        "test_gap_active_days_post_label_3m": (False, float("-inf")),
        "test_gap_sustained_active_2of3_post_label": (False, float("-inf")),
        "test_gap_sustained_active_2of3_post_label_ci_width": (True, float("inf")),
        "test_prevalence_entropy": (False, float("-inf")),
        "test_bootstrap_prevalence_ci_width": (True, float("inf")),
        "test_monthly_prevalence_std": (True, float("inf")),
        "rule_size": (True, float("inf")),
        "threshold": (True, float("inf")),
    }
    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col, (asc, default) in numeric_defaults.items():
        if col in ranked.columns:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce").fillna(default)
            sort_cols.append(col)
            ascending.append(asc)
    text_cols = [col for col in ["metric_name", "rule_operator", "rule_text", "rule_json"] if col in ranked.columns]
    for col in text_cols:
        ranked[col] = ranked[col].astype(str)
        sort_cols.append(col)
        ascending.append(True)
    if not sort_cols:
        ranked = ranked.reset_index(drop=True)
        ranked["primary_selection_rank"] = np.arange(1, len(ranked) + 1)
        return ranked
    ranked = ranked.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ranked["primary_selection_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def select_primary_definition_group(
    definition_selection: pd.DataFrame | None,
    definition_frontier: pd.DataFrame | None,
) -> tuple[str | None, str]:
    if definition_selection is not None and not definition_selection.empty and "winner_flag" in definition_selection.columns:
        winners = definition_selection[
            pd.to_numeric(definition_selection["winner_flag"], errors="coerce").fillna(0).astype(int) == 1
        ].copy()
        if not winners.empty and "definition_group" in winners.columns:
            groups = winners["definition_group"].map(normalize_definition_group).dropna().unique().tolist()
            if len(groups) == 1:
                return str(groups[0]), f"definition_selection_winner_flag::{groups[0]}"
    if definition_frontier is not None and not definition_frontier.empty:
        candidates = definition_frontier.copy()
        if "pareto_frontier_flag" in candidates.columns and pd.to_numeric(candidates["pareto_frontier_flag"], errors="coerce").fillna(0).astype(int).any():
            candidates = candidates[pd.to_numeric(candidates["pareto_frontier_flag"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if not candidates.empty:
            candidates["definition_group"] = candidates["definition_name"].map(normalize_definition_group)
            sort_cols = [
                "test_gap_returned_active_post_label_m1",
                "test_gap_returned_active_post_label_m2",
                "test_gap_returned_active_post_label_m3",
                "test_gap_active_days_post_label_3m",
                "test_gap_sustained_active_2of3_post_label",
                "test_gap_sustained_active_2of3_post_label_ci_width",
                "folds",
                "test_prevalence_entropy",
                "test_bootstrap_prevalence_ci_width",
                "test_monthly_prevalence_std",
                "rule_size",
            ]
            available = [col for col in sort_cols if col in candidates.columns]
            if not available:
                first_group = candidates["definition_group"].iloc[0]
                return str(first_group), f"definition_frontier_fallback::{first_group}"
            ascending = [False, False, False, False, False, True, False, False, True, True, True][: len(available)]
            chosen = candidates.sort_values(available, ascending=ascending, kind="mergesort").iloc[0]
            return str(chosen["definition_group"]), "definition_frontier_external_validation_lexicographic"
    return None, "definition_selection_unavailable"


def build_serving_operational_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "problem_key",
                "model_name",
                "max_operational_metric_std",
                "max_operational_metric_jump",
                "max_confusion_share_std",
                "max_confusion_share_jump",
            ]
        )
    # Lazy import avoids a module cycle because post_model_outputs also imports this module.
    from targeted_ml.pipelines.modelled_to_ml.post_model_outputs import build_cv_threshold_robustness_outputs

    _, _, threshold_summary, confusion_summary = build_cv_threshold_robustness_outputs(predictions)
    metric_keep = {"precision", "recall", "f1", "accuracy", "predicted_positive_rate"}
    threshold_work = threshold_summary[threshold_summary["metric_name"].isin(metric_keep)].copy()
    threshold_agg = (
        threshold_work.groupby(["problem_key", "model_name"], as_index=False)
        .agg(
            max_operational_metric_std=("std_value", "max"),
            max_operational_metric_jump=("max_fold_to_fold_jump", "max"),
        )
        if not threshold_work.empty
        else pd.DataFrame(columns=["problem_key", "model_name", "max_operational_metric_std", "max_operational_metric_jump"])
    )
    if not confusion_summary.empty:
        totals = confusion_summary.groupby(["problem_key", "model_name", "policy_name"], as_index=False).agg(total_mean_rows=("mean_rows", "sum"))
        confusion_work = confusion_summary.merge(totals, on=["problem_key", "model_name", "policy_name"], how="left")
        denom = pd.to_numeric(confusion_work["total_mean_rows"], errors="coerce").fillna(0.0).clip(lower=1.0)
        confusion_work["confusion_share_std"] = pd.to_numeric(confusion_work["std_rows"], errors="coerce").fillna(0.0) / denom
        confusion_work["confusion_share_jump"] = pd.to_numeric(confusion_work["max_fold_to_fold_jump"], errors="coerce").fillna(0.0) / denom
        confusion_agg = confusion_work.groupby(["problem_key", "model_name"], as_index=False).agg(
            max_confusion_share_std=("confusion_share_std", "max"),
            max_confusion_share_jump=("confusion_share_jump", "max"),
        )
    else:
        confusion_agg = pd.DataFrame(columns=["problem_key", "model_name", "max_confusion_share_std", "max_confusion_share_jump"])
    merged = threshold_agg.merge(confusion_agg, on=["problem_key", "model_name"], how="outer")
    for col in [
        "max_operational_metric_std",
        "max_operational_metric_jump",
        "max_confusion_share_std",
        "max_confusion_share_jump",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(float("inf"))
    return merged


def _rank_serving_candidates(
    candidates: pd.DataFrame,
    model_predictions: pd.DataFrame,
    scoring_scenarios: pd.DataFrame | None,
) -> pd.DataFrame:
    working = candidates.copy()
    candidate_predictions = model_predictions.merge(
        working[["problem_key", "model_name"]].drop_duplicates(),
        on=["problem_key", "model_name"],
        how="inner",
    )
    operational = build_serving_operational_summary(candidate_predictions)
    if scoring_scenarios is not None and not scoring_scenarios.empty and "score_window_end_day" in scoring_scenarios.columns:
        scenario_meta = scoring_scenarios[["problem_key", "score_window_end_day"]].drop_duplicates()
        working = working.merge(scenario_meta, on="problem_key", how="left")
    working = working.merge(operational, on=["problem_key", "model_name"], how="left")
    working["max_probability_metric_std"] = working[["std_ap", "std_roc_auc", "std_brier", "std_log_loss"]].max(axis=1)
    for col in [
        "max_operational_metric_std",
        "max_operational_metric_jump",
        "max_confusion_share_std",
        "max_confusion_share_jump",
    ]:
        if col not in working.columns:
            working[col] = np.nan
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(float("inf"))
    if "score_window_end_day" not in working.columns:
        working["score_window_end_day"] = np.nan
    working["score_window_end_day"] = pd.to_numeric(working["score_window_end_day"], errors="coerce").fillna(float("-inf"))
    sort_cols = [
        "mean_brier",
        "mean_log_loss",
        "mean_calibration_slope_error",
        "mean_calibration_intercept_abs",
        "max_probability_metric_std",
        "max_operational_metric_std",
        "max_operational_metric_jump",
        "max_confusion_share_std",
        "max_confusion_share_jump",
        "mean_ap",
        "mean_roc_auc",
        "score_window_end_day",
        "problem_key",
        "model_name",
    ]
    ascending = [True, True, True, True, True, True, True, True, True, False, False, False, True, True]
    ordered = working.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ordered["serving_rank"] = np.arange(1, len(ordered) + 1)
    return ordered


def select_serving_scope(
    model_frontier: pd.DataFrame,
    model_predictions: pd.DataFrame,
    definition_selection: pd.DataFrame | None = None,
    definition_frontier: pd.DataFrame | None = None,
    scoring_scenarios: pd.DataFrame | None = None,
    problem_keys: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = model_frontier.copy()
    if "pareto_frontier_flag" in candidates.columns and pd.to_numeric(candidates["pareto_frontier_flag"], errors="coerce").fillna(0).astype(int).any():
        candidates = candidates[pd.to_numeric(candidates["pareto_frontier_flag"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if problem_keys:
        candidates = candidates[candidates["problem_key"].isin(list(problem_keys))].copy()
    if model_names:
        candidates = candidates[candidates["model_name"].isin(list(model_names))].copy()
    if candidates.empty:
        empty = pd.DataFrame(columns=list(model_frontier.columns) + ["definition_group", "serving_rank"])
        return (
            empty.iloc[0:0].copy(),
            empty,
            {
                "definition_group_context": None,
                "definition_context_reason": "no_model_candidates_after_requested_filters",
                "selected_primary_definition_group": None,
                "candidate_pool_size": 0,
                "available_model_groups": [],
                "selection_scope": "requested_filters_empty",
                "serving_candidate_found": 0,
                "serving_status": "no_model_candidates_after_requested_filters",
                "selection_policy": [],
            },
        )
    candidates["definition_group"] = [
        candidate_definition_group(problem_key, definition_name)
        for problem_key, definition_name in zip(candidates["problem_key"], candidates.get("definition_name", pd.Series([""] * len(candidates))))
    ]
    available_groups = sorted(candidates["definition_group"].dropna().astype(str).unique().tolist())
    if len(available_groups) == 1:
        chosen_definition_group = str(available_groups[0])
        definition_reason = f"model_frontier_unique_definition_group::{chosen_definition_group}"
    else:
        chosen_definition_group, definition_reason = select_primary_definition_group(definition_selection, definition_frontier)
    selection_scope = "all_pareto_frontier_candidates"
    if chosen_definition_group:
        selection_scope = "definition_group_matched_frontier_candidates"
        matching_group = candidates[candidates["definition_group"] == chosen_definition_group].copy()
        if matching_group.empty:
            ordered = _rank_serving_candidates(candidates, model_predictions, scoring_scenarios)
            empty_selected = ordered.iloc[0:0].copy()
            return (
                empty_selected,
                ordered,
                {
                    "definition_group_context": chosen_definition_group,
                    "definition_context_reason": definition_reason,
                    "selected_primary_definition_group": None,
                    "candidate_pool_size": 0,
                    "available_model_groups": available_groups,
                    "selection_scope": selection_scope,
                    "serving_candidate_found": 0,
                    "serving_status": "no_valid_model_for_selected_definition_group",
                    "selection_policy": [
                        "mean_brier asc",
                        "mean_log_loss asc",
                        "mean_calibration_slope_error asc",
                        "mean_calibration_intercept_abs asc",
                        "max_probability_metric_std asc",
                        "max_operational_metric_std asc",
                        "max_operational_metric_jump asc",
                        "max_confusion_share_std asc",
                        "max_confusion_share_jump asc",
                        "mean_ap desc",
                        "mean_roc_auc desc",
                        "score_window_end_day desc",
                    ],
                },
            )
        candidates = matching_group
    ordered = _rank_serving_candidates(candidates, model_predictions, scoring_scenarios)
    selected = ordered.head(1).copy()
    selected["selection_reason"] = f"serving_primary::{selection_scope}::probability_then_variability_then_information"
    selected_definition_group = (
        str(selected["definition_group"].iloc[0])
        if not selected.empty and "definition_group" in selected.columns
        else None
    )
    selection_meta = {
        "definition_group_context": chosen_definition_group,
        "definition_context_reason": definition_reason,
        "selected_primary_definition_group": selected_definition_group,
        "candidate_pool_size": int(len(ordered)),
        "available_model_groups": available_groups,
        "selection_scope": selection_scope,
        "serving_candidate_found": int(not selected.empty),
        "serving_status": "selected_primary_model" if not selected.empty else "no_selected_primary_model",
        "selection_policy": [
            "mean_brier asc",
            "mean_log_loss asc",
            "mean_calibration_slope_error asc",
            "mean_calibration_intercept_abs asc",
            "max_probability_metric_std asc",
            "max_operational_metric_std asc",
            "max_operational_metric_jump asc",
            "max_confusion_share_std asc",
            "max_confusion_share_jump asc",
            "mean_ap desc",
            "mean_roc_auc desc",
            "score_window_end_day desc",
        ],
    }
    return selected, ordered, selection_meta
