from __future__ import annotations

import json

import pandas as pd

from targeted_ml.pipelines.modelled_to_ml import analysis_setup as setup
from targeted_ml.pipelines.modelled_to_ml.modeling import (
    build_scoring_scenarios,
    build_train_test_generalization_outputs,
    evaluate_model_problems,
)
from targeted_ml.pipelines.modelled_to_ml.selection import rank_primary_definition_candidates


def test_rank_primary_definition_candidates_orders_frontier_deterministically() -> None:
    candidates = pd.DataFrame(
        [
            {
                "definition_name": "definition_a",
                "metric_name": "future_sessions",
                "rule_operator": ">=",
                "rule_text": "future_sessions >= 24",
                "rule_json": '{"metric":"sessions"}',
                "folds": 4,
                "test_gap_returned_active_post_label_m1": 0.70,
                "test_gap_returned_active_post_label_m2": 0.60,
                "test_gap_returned_active_post_label_m3": 0.50,
                "test_gap_active_days_post_label_3m": 11.0,
                "test_gap_sustained_active_2of3_post_label": 0.71,
                "test_prevalence_entropy": 0.01,
                "test_bootstrap_prevalence_ci_width": 0.0006,
                "test_monthly_prevalence_std": 0.003,
                "rule_size": 1,
                "threshold": 24.0,
                "pareto_frontier_flag": 1,
            },
            {
                "definition_name": "definition_a",
                "metric_name": "future_active_days",
                "rule_operator": ">=",
                "rule_text": "future_active_days >= 9",
                "rule_json": '{"metric":"active_days"}',
                "folds": 4,
                "test_gap_returned_active_post_label_m1": 0.74,
                "test_gap_returned_active_post_label_m2": 0.56,
                "test_gap_returned_active_post_label_m3": 0.47,
                "test_gap_active_days_post_label_3m": 8.7,
                "test_gap_sustained_active_2of3_post_label": 0.64,
                "test_prevalence_entropy": 0.027,
                "test_bootstrap_prevalence_ci_width": 0.0010,
                "test_monthly_prevalence_std": 0.006,
                "rule_size": 1,
                "threshold": 9.0,
                "pareto_frontier_flag": 1,
            },
        ]
    )

    ranked = rank_primary_definition_candidates(candidates)

    assert ranked["primary_selection_rank"].tolist() == [1, 2]
    assert ranked.iloc[0]["metric_name"] == "future_active_days"


def test_build_scoring_scenarios_uses_definition_b_only_even_when_definition_a_frontier_exists() -> None:
    frame = pd.DataFrame(
        {
            "teacher_unique_id": ["a", "b"],
            "first_month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "feature_a": [1.0, 2.0],
            "definition_a::rule": [0, 1],
            "definition_b_label": [1, 0],
        }
    )
    feature_registry = pd.DataFrame(
        [
            {
                "feature_name": "feature_a",
                "feature_class": "behavior",
                "allowed_in_S1": 1,
                "allowed_in_S7": 1,
                "allowed_in_S1_PLUS_S7": 1,
                "allowed_in_STRICT_CONTEXT": 1,
            }
        ]
    )
    track_registry = pd.DataFrame(
        [
            {"track_name": "S1", "score_window_end_day": 0},
            {"track_name": "S7", "score_window_end_day": 7},
        ]
    )
    definition_frontier = pd.DataFrame(
        [
            {"definition_name": "definition_a::rule"},
            {"definition_name": "definition_b_label"},
        ]
    )

    scenarios = build_scoring_scenarios(frame, feature_registry, track_registry, definition_frontier)

    assert sorted(scenarios["definition_name"].unique().tolist()) == ["definition_b_label"]
    assert sorted(scenarios["problem_key"].tolist()) == ["definition_b_label__S1", "definition_b_label__S7"]


def test_evaluate_model_problems_runs_global_problem_model_grid(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "teacher_unique_id": ["a", "b", "c", "d"],
            "first_month": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "definition_a_label": [0, 1, 0, 1],
            "definition_b_label": [1, 0, 1, 0],
        }
    )
    scoring_scenarios = pd.DataFrame(
        [
            {
                "problem_key": "definition_a_label__S1",
                "definition_name": "definition_a_label",
                "label_col": "definition_a_label",
                "track_name": "S1",
                "feature_names_json": json.dumps(["feature_a"]),
            },
            {
                "problem_key": "definition_b_label__S7",
                "definition_name": "definition_b_label",
                "label_col": "definition_b_label",
                "track_name": "S7",
                "feature_names_json": json.dumps(["feature_a"]),
            },
        ]
    )
    feature_registry = pd.DataFrame(
        [
            {
                "feature_name": "feature_a",
                "feature_class": "behavior",
            }
        ]
    )

    model_specs = [
        {"model_name": "model_alpha", "estimator": object(), "param_distributions": {}},
        {"model_name": "model_beta", "estimator": object(), "param_distributions": {}},
    ]
    seen_tasks: list[tuple[str, str]] = []

    monkeypatch.setattr("targeted_ml.pipelines.modelled_to_ml.modeling.build_model_specs", lambda: model_specs)

    def _fake_run(problem, model_spec, working, feature_registry, feature_names, compute_feature_importance, task_store):
        seen_tasks.append((problem["problem_key"], model_spec["model_name"]))
        return (
                [
                    {
                        "problem_key": problem["problem_key"],
                        "definition_name": problem["definition_name"],
                        "track_name": problem["track_name"],
                        "model_name": model_spec["model_name"],
                        "fold_id": 1,
                        "fold_valid_flag": 1,
                        "ap": 0.5,
                        "roc_auc": 0.6,
                        "brier": 0.2,
                        "log_loss": 0.4,
                        "calibration_slope": 1.0,
                        "calibration_intercept": 0.0,
                        "calibration_slope_error": 0.0,
                        "calibration_intercept_abs": 0.0,
                    }
                ],
            [
                {
                    "teacher_unique_id": working["teacher_unique_id"].iloc[0],
                    "first_month": working["first_month"].iloc[0],
                    "y_true": int(working["y_true"].iloc[0]),
                    "problem_key": problem["problem_key"],
                    "definition_name": problem["definition_name"],
                    "track_name": problem["track_name"],
                    "model_name": model_spec["model_name"],
                    "fold_id": 1,
                    "score": 0.5,
                    "technical_fold_valid_flag": 1,
                    "fold_valid_flag": 1,
                    "invalid_reason": "",
                }
            ],
            [],
            [],
            [],
        )

    monkeypatch.setattr("targeted_ml.pipelines.modelled_to_ml.modeling.run_or_load_model_task", _fake_run)
    previous_workers = setup.MODEL_COMPARISON_WORKERS
    setup.MODEL_COMPARISON_WORKERS = 6
    try:
        fold_df, pred_df, _, _, _, _ = evaluate_model_problems(
            frame=frame,
            feature_registry=feature_registry,
            scoring_scenarios=scoring_scenarios,
            compute_feature_importance=False,
            task_store=None,
        )
    finally:
        setup.MODEL_COMPARISON_WORKERS = previous_workers

    assert sorted(seen_tasks) == [
        ("definition_a_label__S1", "model_alpha"),
        ("definition_a_label__S1", "model_beta"),
        ("definition_b_label__S7", "model_alpha"),
        ("definition_b_label__S7", "model_beta"),
    ]
    assert len(fold_df) == 4
    assert len(pred_df) == 4


def test_build_train_test_generalization_outputs_flags_positive_gap_when_train_beats_test() -> None:
    fold_df = pd.DataFrame(
        [
            {
                "problem_key": "definition_b_label__S7",
                "definition_name": "definition_b_label",
                "track_name": "S7",
                "model_name": "catboost",
                "fold_id": 1,
                "fold_valid_flag": 1,
                "ap": 0.35,
                "roc_auc": 0.71,
                "brier": 0.12,
                "log_loss": 0.40,
                "apparent_train_ap": 0.70,
                "apparent_train_roc_auc": 0.90,
                "apparent_train_brier": 0.04,
                "apparent_train_log_loss": 0.12,
                "calibration_holdout_ap": 0.55,
                "calibration_holdout_roc_auc": 0.80,
                "calibration_holdout_brier": 0.09,
                "calibration_holdout_log_loss": 0.25,
            },
            {
                "problem_key": "definition_b_label__S7",
                "definition_name": "definition_b_label",
                "track_name": "S7",
                "model_name": "catboost",
                "fold_id": 2,
                "fold_valid_flag": 1,
                "ap": 0.30,
                "roc_auc": 0.69,
                "brier": 0.13,
                "log_loss": 0.43,
                "apparent_train_ap": 0.67,
                "apparent_train_roc_auc": 0.88,
                "apparent_train_brier": 0.05,
                "apparent_train_log_loss": 0.15,
                "calibration_holdout_ap": 0.50,
                "calibration_holdout_roc_auc": 0.77,
                "calibration_holdout_brier": 0.10,
                "calibration_holdout_log_loss": 0.28,
            },
        ]
    )
    gap_folds, gap_summary = build_train_test_generalization_outputs(fold_df)

    assert not gap_folds.empty
    assert not gap_summary.empty
    apparent_ap = gap_summary[
        (gap_summary["comparison_stage"] == "apparent_train")
        & (gap_summary["metric_name"] == "ap")
    ].iloc[0]
    assert apparent_ap["mean_generalization_gap"] > 0
    assert int(apparent_ap["statistical_gap_flag"]) == 1
