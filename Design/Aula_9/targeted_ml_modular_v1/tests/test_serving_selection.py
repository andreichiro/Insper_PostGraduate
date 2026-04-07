from __future__ import annotations

import pandas as pd
from targeted_ml.pipelines.modelled_to_ml.selection import select_serving_scope


def test_select_serving_scope_restricts_candidates_to_selected_definition_group() -> None:
    model_frontier = pd.DataFrame(
        [
            {
                "problem_key": "definition_a::rule__S7",
                "definition_name": "definition_a::rule",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.032,
                "mean_log_loss": 0.120,
                "mean_calibration_slope_error": 0.08,
                "mean_calibration_intercept_abs": 0.02,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.003,
                "std_log_loss": 0.004,
                "mean_ap": 0.21,
                "mean_roc_auc": 0.80,
            },
            {
                "problem_key": "definition_a::rule__S1_PLUS_S7",
                "definition_name": "definition_a::rule",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.025,
                "mean_log_loss": 0.105,
                "mean_calibration_slope_error": 0.05,
                "mean_calibration_intercept_abs": 0.01,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.002,
                "std_log_loss": 0.003,
                "mean_ap": 0.24,
                "mean_roc_auc": 0.82,
            },
            {
                "problem_key": "definition_b_label__S1_PLUS_S7",
                "definition_name": "definition_b",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.024,
                "mean_log_loss": 0.106,
                "mean_calibration_slope_error": 0.06,
                "mean_calibration_intercept_abs": 0.02,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.002,
                "std_log_loss": 0.003,
                "mean_ap": 0.23,
                "mean_roc_auc": 0.81,
            },
        ]
    )
    model_predictions = pd.DataFrame(columns=["problem_key", "model_name"])
    definition_selection = pd.DataFrame(
        [
            {"definition_group": "definition_a", "winner_flag": 1},
            {"definition_group": "definition_b", "winner_flag": 0},
        ]
    )
    definition_frontier = pd.DataFrame()
    scoring_scenarios = pd.DataFrame(
        [
            {"problem_key": "definition_a::rule__S7", "score_window_end_day": 7},
            {"problem_key": "definition_a::rule__S1_PLUS_S7", "score_window_end_day": 7},
            {"problem_key": "definition_b_label__S1_PLUS_S7", "score_window_end_day": 7},
        ]
    )

    selected, ordered, meta = select_serving_scope(
        model_frontier=model_frontier,
        model_predictions=model_predictions,
        definition_selection=definition_selection,
        definition_frontier=definition_frontier,
        scoring_scenarios=scoring_scenarios,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["problem_key"] == "definition_a::rule__S1_PLUS_S7"
    assert ordered.iloc[0]["problem_key"] == "definition_a::rule__S1_PLUS_S7"
    assert meta["definition_group_context"] == "definition_a"
    assert meta["definition_context_reason"] == "definition_selection_winner_flag::definition_a"
    assert meta["selected_primary_definition_group"] == "definition_a"
    assert meta["selection_scope"] == "definition_group_matched_frontier_candidates"
    assert meta["candidate_pool_size"] == 2


def test_select_serving_scope_treats_missing_score_window_as_worse_than_present_information() -> None:
    model_frontier = pd.DataFrame(
        [
            {
                "problem_key": "definition_a::rule__S7",
                "definition_name": "definition_a::rule",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.030,
                "mean_log_loss": 0.110,
                "mean_calibration_slope_error": 0.05,
                "mean_calibration_intercept_abs": 0.01,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.002,
                "std_log_loss": 0.003,
                "mean_ap": 0.20,
                "mean_roc_auc": 0.80,
            },
            {
                "problem_key": "definition_a::rule__S1_PLUS_S7",
                "definition_name": "definition_a::rule",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.030,
                "mean_log_loss": 0.110,
                "mean_calibration_slope_error": 0.05,
                "mean_calibration_intercept_abs": 0.01,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.002,
                "std_log_loss": 0.003,
                "mean_ap": 0.20,
                "mean_roc_auc": 0.80,
            },
        ]
    )
    selected, _, _ = select_serving_scope(
        model_frontier=model_frontier,
        model_predictions=pd.DataFrame(columns=["problem_key", "model_name"]),
        definition_selection=pd.DataFrame([{"definition_group": "definition_a", "winner_flag": 1}]),
        definition_frontier=pd.DataFrame(),
        scoring_scenarios=pd.DataFrame([{"problem_key": "definition_a::rule__S1_PLUS_S7", "score_window_end_day": 7}]),
    )

    assert selected.iloc[0]["problem_key"] == "definition_a::rule__S1_PLUS_S7"


def test_select_serving_scope_prefers_unique_model_frontier_group_when_model_stage_is_decoupled_from_definition_study() -> None:
    model_frontier = pd.DataFrame(
        [
            {
                "problem_key": "definition_b_label__S7",
                "definition_name": "definition_b",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.030,
                "mean_log_loss": 0.110,
                "mean_calibration_slope_error": 0.05,
                "mean_calibration_intercept_abs": 0.01,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.002,
                "std_log_loss": 0.003,
                "mean_ap": 0.20,
                "mean_roc_auc": 0.80,
            },
        ]
    )
    selected, ordered, meta = select_serving_scope(
        model_frontier=model_frontier,
        model_predictions=pd.DataFrame(columns=["problem_key", "model_name"]),
        definition_selection=pd.DataFrame([{"definition_group": "definition_a", "winner_flag": 1}]),
        definition_frontier=pd.DataFrame(),
        scoring_scenarios=pd.DataFrame([{"problem_key": "definition_b_label__S7", "score_window_end_day": 7}]),
    )

    assert not selected.empty
    assert ordered.iloc[0]["problem_key"] == "definition_b_label__S7"
    assert meta["definition_group_context"] == "definition_b"
    assert meta["definition_context_reason"] == "model_frontier_unique_definition_group::definition_b"
    assert meta["serving_candidate_found"] == 1
    assert meta["serving_status"] == "selected_primary_model"
    assert meta["available_model_groups"] == ["definition_b"]
    assert meta["selection_scope"] == "definition_group_matched_frontier_candidates"


def test_select_serving_scope_prefers_official_definition_selection_winner_over_definition_frontier_fallback() -> None:
    model_frontier = pd.DataFrame(
        [
            {
                "problem_key": "definition_a::rule__S7",
                "definition_name": "definition_a::rule",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.031,
                "mean_log_loss": 0.111,
                "mean_calibration_slope_error": 0.05,
                "mean_calibration_intercept_abs": 0.01,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.002,
                "std_log_loss": 0.003,
                "mean_ap": 0.19,
                "mean_roc_auc": 0.79,
            },
            {
                "problem_key": "definition_b_label__S7",
                "definition_name": "definition_b",
                "model_name": "catboost",
                "pareto_frontier_flag": 1,
                "mean_brier": 0.030,
                "mean_log_loss": 0.110,
                "mean_calibration_slope_error": 0.05,
                "mean_calibration_intercept_abs": 0.01,
                "std_ap": 0.01,
                "std_roc_auc": 0.01,
                "std_brier": 0.002,
                "std_log_loss": 0.003,
                "mean_ap": 0.20,
                "mean_roc_auc": 0.80,
            },
        ]
    )
    definition_frontier = pd.DataFrame(
        [
            {
                "definition_name": "definition_b_label",
                "pareto_frontier_flag": 1,
                "test_gap_returned_active_post_label_m1": 0.4,
                "test_gap_returned_active_post_label_m2": 0.3,
                "test_gap_returned_active_post_label_m3": 0.2,
                "test_gap_active_days_post_label_3m": 2.0,
                "test_gap_sustained_active_2of3_post_label": 0.2,
                "folds": 6,
                "test_prevalence_entropy": 0.4,
                "test_bootstrap_prevalence_ci_width": 0.02,
                "test_monthly_prevalence_std": 0.01,
                "rule_size": 1,
            }
        ]
    )
    selected, _, meta = select_serving_scope(
        model_frontier=model_frontier,
        model_predictions=pd.DataFrame(columns=["problem_key", "model_name"]),
        definition_selection=pd.DataFrame([{"definition_group": "definition_a", "winner_flag": 1}]),
        definition_frontier=definition_frontier,
        scoring_scenarios=pd.DataFrame([{"problem_key": "definition_b_label__S7", "score_window_end_day": 7}]),
    )

    assert selected.iloc[0]["problem_key"] == "definition_a::rule__S7"
    assert meta["definition_group_context"] == "definition_a"
    assert meta["definition_context_reason"] == "definition_selection_winner_flag::definition_a"
