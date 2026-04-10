from __future__ import annotations

import numpy as np
import pandas as pd

from targeted_ml.pipelines.modelled_to_ml import analysis_setup as setup
from targeted_ml.pipelines.modelled_to_ml.analysis_setup import (
    RUNTIME_CONFIG,
    RuntimeBuildConfig,
    apply_runtime_config,
    split_definition_workflow_frame,
)
from targeted_ml.pipelines.modelled_to_ml.definitions import (
    aggregate_definition_test_eval,
    build_definition_evaluability_audit,
    build_definition_search,
    build_definition_search_stage_audit,
    choose_final_definition_a_from_lock,
    compute_candidate_diagnostics,
    summarize_lock_neighbor_sensitivity,
)


def _runtime_payload() -> dict[str, object]:
    return {
        "analysis_kind": "activity",
        "official_population_filter": "all_observed_first_use",
        "enabled_tracks": ["S1", "S7", "S1_PLUS_S7", "STRICT_CONTEXT"],
        "label_window_days": 30,
        "post_label_block_days": 30,
        "post_label_block_count": 3,
        "external_validators": [
            "returned_active_post_label_m1",
            "returned_active_post_label_m2",
            "returned_active_post_label_m3",
            "active_days_post_label_3m",
            "sustained_active_2of3_post_label",
        ],
        "definition_a_enabled": True,
        "definition_a_strategy": "univariate_exact",
        "definition_a_candidate_metrics": ["future_business_active_weeks"],
        "definition_a_promoted_candidate_limit": 3,
        "definition_a_sql_file": "",
        "definition_a_python_strategy": "",
        "definition_b": {
            "definition_name": "definition_b",
            "metric_name": "future_business_active_weeks",
            "operator": ">=",
            "threshold": 1.0,
        },
        "definition_b_sql_file": "",
        "definition_b_python_strategy": "",
        "max_outer_test_months": 2,
        "definition_selection_holdout_months": 2,
        "definition_lock_months": 2,
        "min_official_valid_outer_folds": 2,
        "min_official_test_rows": 1,
        "min_official_test_positives": 1,
        "min_official_test_negatives": 1,
        "definition_lock_bootstrap_gate": {
            "column_name": "lock_gap_sustained_active_2of3_post_label_ci_low",
            "operator": ">",
            "threshold": 0.0,
        },
        "tuning_enabled": True,
        "tuning_n_iter": 2,
        "tuning_max_inner_splits": 2,
        "tuning_scoring": "neg_brier_score",
        "model_family_scope": ["logistic_regression"],
        "model_comparison_workers": 1,
        "calibration_method": "sigmoid",
        "feature_importance_permutation_repeats": 1,
        "cluster_k_candidates": [2, 3],
        "cluster_bootstrap_iterations": 2,
        "cluster_sample_size": 100,
        "registered_band_policies": [],
        "heavy_user_percentile_policies": [],
    }


def _runtime_payload_compound() -> dict[str, object]:
    payload = _runtime_payload()
    payload["definition_a_strategy"] = "screened_pairwise_compound_weighted"
    payload["definition_a_candidate_metrics"] = ["future_sessions", "future_active_days"]
    payload["definition_a_promoted_candidate_limit"] = 2
    return payload


def _make_metrics() -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=10, freq="MS")
    rows: list[dict[str, object]] = []
    for month_idx, month in enumerate(months):
        for teacher_offset in range(6):
            metric_value = float(teacher_offset + (1 if month_idx % 2 else 0))
            positive = int(metric_value >= 3.0)
            rows.append(
                {
                    "teacher_unique_id": f"{month.strftime('%Y%m')}_{teacher_offset}",
                    "first_month": month,
                    "full_followup_observed_flag": 1,
                    "months_after_entry": 0,
                    "future_business_active_weeks": metric_value,
                    "returned_active_post_label_m1": positive,
                    "returned_active_post_label_m2": positive,
                    "returned_active_post_label_m3": positive,
                    "active_days_post_label_3m": 9 if positive else 1,
                    "sustained_active_2of3_post_label": positive,
                }
            )
    return pd.DataFrame(rows)


def _make_metrics_compound() -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=10, freq="MS")
    rows: list[dict[str, object]] = []
    for month_idx, month in enumerate(months):
        for teacher_offset in range(8):
            sessions = float(teacher_offset + (1 if month_idx % 2 else 0))
            active_days = float(max(0, teacher_offset - 1))
            score = 0.60 * (sessions / 8.0) + 0.40 * (active_days / 6.0)
            positive = int(score >= 0.45)
            rows.append(
                {
                    "teacher_unique_id": f"{month.strftime('%Y%m')}_{teacher_offset}",
                    "first_month": month,
                    "full_followup_observed_flag": 1,
                    "months_after_entry": 0,
                    "future_sessions": sessions,
                    "future_active_days": active_days,
                    "returned_active_post_label_m1": positive,
                    "returned_active_post_label_m2": positive,
                    "returned_active_post_label_m3": positive,
                    "active_days_post_label_3m": 12 if positive else 1,
                    "sustained_active_2of3_post_label": positive,
                }
            )
    return pd.DataFrame(rows)


def test_split_definition_workflow_frame_reserves_lock_and_final_eval_months() -> None:
    previous = RUNTIME_CONFIG
    apply_runtime_config(RuntimeBuildConfig.from_payload(_runtime_payload()))
    try:
        metrics = _make_metrics()
        development, lock, final_eval, development_months, lock_months, final_eval_months = split_definition_workflow_frame(metrics)
        assert [month.strftime("%Y-%m") for month in development_months] == [
            "2024-01",
            "2024-02",
            "2024-03",
            "2024-04",
            "2024-05",
            "2024-06",
        ]
        assert [month.strftime("%Y-%m") for month in lock_months] == ["2024-07", "2024-08"]
        assert [month.strftime("%Y-%m") for month in final_eval_months] == ["2024-09", "2024-10"]
        assert development["first_month"].nunique() == 6
        assert lock["first_month"].nunique() == 2
        assert final_eval["first_month"].nunique() == 2
    finally:
        apply_runtime_config(previous)


def test_definition_search_uses_development_then_locks_one_definition_without_model_metrics() -> None:
    previous = setup.RUNTIME_CONFIG
    apply_runtime_config(RuntimeBuildConfig.from_payload(_runtime_payload()))
    try:
        metrics = _make_metrics()
        candidate_metric_registry = pd.DataFrame(
            [
                {
                    "metric_name": "future_business_active_weeks",
                    "definition_a_candidate_flag": 1,
                }
            ]
        )
        candidate_df, candidate_test_df, definition_lock_df, selection_df = build_definition_search(metrics, candidate_metric_registry)
        assert not candidate_df.empty
        assert sorted(candidate_test_df["fold_id"].unique().tolist()) == [1, 2, 3, 4, 5]
        test_univariate = candidate_test_df[candidate_test_df["candidate_type"] == "univariate_exact_threshold"].copy()
        assert not test_univariate.empty
        assert test_univariate["threshold_candidate_rank"].notna().all()
        assert test_univariate["threshold_candidate_count"].notna().all()
        winners = selection_df[
            (selection_df["definition_group"] == "definition_a")
            & (pd.to_numeric(selection_df["winner_flag"], errors="coerce").fillna(0).astype(int) == 1)
        ].copy()
        assert len(winners) == 1
        winner = winners.iloc[0]
        assert winner["official_status"] == "official_winner"
        assert float(winner["lock_selection_rank"]) == 1.0
        assert "future_business_active_weeks" in str(winner["rule_text"])
        assert int(winner["lock_months"]) == 2
        assert pd.notna(winner["lock_min_label_jaccard"])
        assert pd.notna(winner["lock_max_neighbor_gap_delta"])
        assert pd.notna(winner["lock_max_neighbor_prevalence_delta"])
        assert "ap" not in selection_df.columns
        assert "brier" not in selection_df.columns
        assert not definition_lock_df.empty
        assert {"lock_min_label_jaccard", "lock_max_neighbor_gap_delta", "lock_max_neighbor_prevalence_delta"}.issubset(
            set(definition_lock_df.columns)
        )
    finally:
        apply_runtime_config(previous)


def test_definition_candidate_requires_support_not_only_two_classes() -> None:
    previous = setup.RUNTIME_CONFIG
    payload = _runtime_payload()
    payload["min_official_test_rows"] = 50
    payload["min_official_test_positives"] = 5
    payload["min_official_test_negatives"] = 20
    apply_runtime_config(RuntimeBuildConfig.from_payload(payload))
    try:
        frame = pd.DataFrame(
            {
                "teacher_unique_id": [f"teacher_{idx}" for idx in range(23)],
                "first_month": pd.to_datetime(["2024-01-01"] * 23),
                "returned_active_post_label_m1": [1] * 4 + [0] * 19,
                "returned_active_post_label_m2": [1] * 4 + [0] * 19,
                "returned_active_post_label_m3": [1] * 4 + [0] * 19,
                "active_days_post_label_3m": [8] * 4 + [1] * 19,
                "sustained_active_2of3_post_label": [1] * 4 + [0] * 19,
            }
        )
        label = np.array([1] * 4 + [0] * 19, dtype=int)
        diagnostics = compute_candidate_diagnostics(frame, label)
        assert int(diagnostics["technical_candidate_valid_flag"]) == 1
        assert int(diagnostics["support_valid_flag"]) == 0
        assert int(diagnostics["candidate_valid_flag"]) == 0
        assert diagnostics["invalid_reason"] == "insufficient_definition_support"
    finally:
        apply_runtime_config(previous)


def test_lock_selection_requires_positive_primary_gap_bootstrap_ci() -> None:
    lock_summary = pd.DataFrame(
        [
            {
                "rule_json": '{"rule":"a"}',
                "metric_name": "rule_a",
                "rule_text": "rule_a",
                "rule_operator": ">=",
                "candidate_type": "univariate_exact_threshold",
                "candidate_group_key": "g::a",
                "definition_name": "definition_a",
                "threshold": 1.0,
                "rule_size": 1,
                "threshold_source": "observed_train_value",
                "lock_months": 6,
                "lock_gap_returned_active_post_label_m1": 0.20,
                "lock_gap_returned_active_post_label_m2": 0.20,
                "lock_gap_returned_active_post_label_m3": 0.20,
                "lock_gap_active_days_post_label_3m": 2.0,
                "lock_gap_sustained_active_2of3_post_label": 0.25,
                "lock_gap_sustained_active_2of3_post_label_ci_low": -0.01,
                "lock_gap_sustained_active_2of3_post_label_ci_high": 0.40,
                "lock_gap_sustained_active_2of3_post_label_ci_width": 0.41,
                "lock_max_gap_std": 0.01,
                "lock_max_gap_jump": 0.02,
                "lock_min_label_jaccard": 0.90,
                "lock_max_neighbor_gap_delta": 0.03,
                "lock_max_neighbor_prevalence_delta": 0.02,
                "lock_prevalence_entropy": 0.65,
                "lock_bootstrap_prevalence_ci_width": 0.10,
                "lock_prevalence_std": 0.02,
                "development_rank": 1,
            },
            {
                "rule_json": '{"rule":"b"}',
                "metric_name": "rule_b",
                "rule_text": "rule_b",
                "rule_operator": ">=",
                "candidate_type": "univariate_exact_threshold",
                "candidate_group_key": "g::b",
                "definition_name": "definition_a",
                "threshold": 2.0,
                "rule_size": 1,
                "threshold_source": "observed_train_value",
                "lock_months": 6,
                "lock_gap_returned_active_post_label_m1": 0.18,
                "lock_gap_returned_active_post_label_m2": 0.18,
                "lock_gap_returned_active_post_label_m3": 0.18,
                "lock_gap_active_days_post_label_3m": 1.8,
                "lock_gap_sustained_active_2of3_post_label": 0.22,
                "lock_gap_sustained_active_2of3_post_label_ci_low": 0.05,
                "lock_gap_sustained_active_2of3_post_label_ci_high": 0.32,
                "lock_gap_sustained_active_2of3_post_label_ci_width": 0.27,
                "lock_max_gap_std": 0.01,
                "lock_max_gap_jump": 0.02,
                "lock_min_label_jaccard": 0.90,
                "lock_max_neighbor_gap_delta": 0.03,
                "lock_max_neighbor_prevalence_delta": 0.02,
                "lock_prevalence_entropy": 0.65,
                "lock_bootstrap_prevalence_ci_width": 0.10,
                "lock_prevalence_std": 0.02,
                "development_rank": 2,
            },
        ]
    )
    ranked = choose_final_definition_a_from_lock(lock_summary)
    assert len(ranked) == 1
    assert ranked.iloc[0]["rule_json"] == '{"rule":"b"}'
    assert int(ranked.iloc[0]["lock_primary_gap_ci_positive_flag"]) == 1


def test_lock_selection_can_return_empty_when_all_candidates_fail_primary_gap_ci() -> None:
    lock_summary = pd.DataFrame(
        [
            {
                "rule_json": '{"rule":"a"}',
                "metric_name": "rule_a",
                "rule_text": "rule_a",
                "rule_operator": ">=",
                "candidate_type": "univariate_exact_threshold",
                "candidate_group_key": "g::a",
                "definition_name": "definition_a",
                "threshold": 1.0,
                "rule_size": 1,
                "threshold_source": "observed_train_value",
                "lock_months": 6,
                "lock_gap_returned_active_post_label_m1": 0.20,
                "lock_gap_returned_active_post_label_m2": 0.20,
                "lock_gap_returned_active_post_label_m3": 0.20,
                "lock_gap_active_days_post_label_3m": 2.0,
                "lock_gap_sustained_active_2of3_post_label": 0.25,
                "lock_gap_sustained_active_2of3_post_label_ci_low": 0.0,
                "lock_gap_sustained_active_2of3_post_label_ci_high": 0.40,
                "lock_gap_sustained_active_2of3_post_label_ci_width": 0.40,
                "lock_max_gap_std": 0.01,
                "lock_max_gap_jump": 0.02,
                "lock_min_label_jaccard": 0.90,
                "lock_max_neighbor_gap_delta": 0.03,
                "lock_max_neighbor_prevalence_delta": 0.02,
                "lock_prevalence_entropy": 0.65,
                "lock_bootstrap_prevalence_ci_width": 0.10,
                "lock_prevalence_std": 0.02,
                "development_rank": 1,
            }
        ]
    )
    ranked = choose_final_definition_a_from_lock(lock_summary)
    assert ranked.empty


def test_lock_selection_can_use_configured_ci_width_rule() -> None:
    previous = setup.RUNTIME_CONFIG
    payload = _runtime_payload()
    payload["definition_lock_bootstrap_gate"] = {
        "column_name": "lock_gap_sustained_active_2of3_post_label_ci_width",
        "operator": "<",
        "threshold": 0.30,
    }
    apply_runtime_config(RuntimeBuildConfig.from_payload(payload))
    try:
        lock_summary = pd.DataFrame(
            [
                {
                    "rule_json": '{"rule":"a"}',
                    "metric_name": "rule_a",
                    "rule_text": "rule_a",
                    "rule_operator": ">=",
                    "candidate_type": "univariate_exact_threshold",
                    "candidate_group_key": "g::a",
                    "definition_name": "definition_a",
                    "threshold": 1.0,
                    "rule_size": 1,
                    "threshold_source": "observed_train_value",
                    "lock_months": 6,
                    "lock_gap_returned_active_post_label_m1": 0.20,
                    "lock_gap_returned_active_post_label_m2": 0.20,
                    "lock_gap_returned_active_post_label_m3": 0.20,
                    "lock_gap_active_days_post_label_3m": 2.0,
                    "lock_gap_sustained_active_2of3_post_label": 0.25,
                    "lock_gap_sustained_active_2of3_post_label_ci_low": 0.01,
                    "lock_gap_sustained_active_2of3_post_label_ci_high": 0.40,
                    "lock_gap_sustained_active_2of3_post_label_ci_width": 0.20,
                    "lock_max_gap_std": 0.01,
                    "lock_max_gap_jump": 0.02,
                    "lock_min_label_jaccard": 0.90,
                    "lock_max_neighbor_gap_delta": 0.03,
                    "lock_max_neighbor_prevalence_delta": 0.02,
                    "lock_prevalence_entropy": 0.65,
                    "lock_bootstrap_prevalence_ci_width": 0.10,
                    "lock_prevalence_std": 0.02,
                    "development_rank": 1.0,
                },
                {
                    "rule_json": '{"rule":"b"}',
                    "metric_name": "rule_b",
                    "rule_text": "rule_b",
                    "rule_operator": ">=",
                    "candidate_type": "univariate_exact_threshold",
                    "candidate_group_key": "g::b",
                    "definition_name": "definition_a",
                    "threshold": 2.0,
                    "rule_size": 1,
                    "threshold_source": "observed_train_value",
                    "lock_months": 6,
                    "lock_gap_returned_active_post_label_m1": 0.30,
                    "lock_gap_returned_active_post_label_m2": 0.30,
                    "lock_gap_returned_active_post_label_m3": 0.30,
                    "lock_gap_active_days_post_label_3m": 2.5,
                    "lock_gap_sustained_active_2of3_post_label": 0.35,
                    "lock_gap_sustained_active_2of3_post_label_ci_low": 0.15,
                    "lock_gap_sustained_active_2of3_post_label_ci_high": 0.60,
                    "lock_gap_sustained_active_2of3_post_label_ci_width": 0.45,
                    "lock_max_gap_std": 0.01,
                    "lock_max_gap_jump": 0.02,
                    "lock_min_label_jaccard": 0.90,
                    "lock_max_neighbor_gap_delta": 0.03,
                    "lock_max_neighbor_prevalence_delta": 0.02,
                    "lock_prevalence_entropy": 0.65,
                    "lock_bootstrap_prevalence_ci_width": 0.10,
                    "lock_prevalence_std": 0.02,
                    "development_rank": 2.0,
                },
            ]
        )
        ranked = choose_final_definition_a_from_lock(lock_summary)
        assert len(ranked) == 1
        row = ranked.iloc[0]
        assert row["rule_text"] == "rule_a"
        assert row["lock_primary_gate_column_name"] == "lock_gap_sustained_active_2of3_post_label_ci_width"
        assert row["lock_primary_gate_operator"] == "<"
        assert float(row["lock_primary_gate_threshold"]) == 0.30
        assert int(row["lock_primary_gate_pass_flag"]) == 1
    finally:
        apply_runtime_config(previous)


def test_definition_search_generates_compound_and_weighted_candidates_with_structural_lock_sensitivity() -> None:
    previous = setup.RUNTIME_CONFIG
    apply_runtime_config(RuntimeBuildConfig.from_payload(_runtime_payload_compound()))
    try:
        metrics = _make_metrics_compound()
        candidate_metric_registry = pd.DataFrame(
            [
                {"metric_name": "future_sessions", "definition_a_candidate_flag": 1},
                {"metric_name": "future_active_days", "definition_a_candidate_flag": 1},
            ]
        )
        candidate_df, candidate_test_df, definition_lock_df, selection_df = build_definition_search(metrics, candidate_metric_registry)
        assert {"compound_pairwise_and", "compound_pairwise_or", "weighted_pairwise_percentile_threshold"}.issubset(
            set(candidate_df["candidate_type"].dropna().unique().tolist())
        )
        assert {"compound_pairwise_and", "compound_pairwise_or", "weighted_pairwise_percentile_threshold"}.issubset(
            set(candidate_test_df["candidate_type"].dropna().unique().tolist())
        )
        weighted_test = candidate_test_df[
            candidate_test_df["candidate_type"] == "weighted_pairwise_percentile_threshold"
        ].copy()
        assert not weighted_test.empty
        assert weighted_test["threshold_candidate_rank"].notna().all()
        assert weighted_test["threshold_candidate_count"].notna().all()
        assert {"lock_threshold_neighbor_count", "lock_structural_neighbor_count", "lock_weight_neighbor_count"}.issubset(
            set(definition_lock_df.columns)
        )
        search_audit = build_definition_search_stage_audit(
            candidate_df,
            candidate_test_df,
            definition_lock_df,
            selection_df,
        )
        assert not search_audit.empty
        weighted_train_stage = search_audit[
            (search_audit["stage_name"] == "train_candidates")
            & (search_audit["candidate_type"] == "weighted_pairwise_percentile_threshold")
        ]
        weighted_test_stage = search_audit[
            (search_audit["stage_name"] == "test_candidates")
            & (search_audit["candidate_type"] == "weighted_pairwise_percentile_threshold")
        ]
        assert len(weighted_train_stage) == 1
        assert len(weighted_test_stage) == 1
        assert int(weighted_train_stage.iloc[0]["rows"]) > 0
        assert int(weighted_test_stage.iloc[0]["rows"]) > 0
        winners = selection_df[
            (selection_df["definition_group"] == "definition_a")
            & (pd.to_numeric(selection_df["winner_flag"], errors="coerce").fillna(0).astype(int) == 1)
        ].copy()
        assert len(winners) == 1
        assert winners.iloc[0]["selection_basis"].startswith("atomic_screening_on_development_outer_tests")
    finally:
        apply_runtime_config(previous)


def test_definition_test_aggregation_adds_bootstrap_ci_for_primary_gap() -> None:
    grouped = pd.DataFrame(
        {
            "definition_name": ["definition_a", "definition_a", "definition_a"],
            "candidate_type": ["univariate_exact_threshold"] * 3,
            "candidate_group_key": ["g1"] * 3,
            "metric_name": ["future_business_active_weeks"] * 3,
            "threshold": [3.0, 3.0, 3.0],
            "rule_json": ['{"kind":"atomic"}'] * 3,
            "rule_text": ["future_business_active_weeks >= 3"] * 3,
            "rule_size": [1, 1, 1],
            "rule_operator": [">="] * 3,
            "threshold_source": ["observed_train_metric_value"] * 3,
            "fold_id": [1, 2, 3],
            "gap_returned_active_post_label_m1": [0.1, 0.2, 0.3],
            "gap_returned_active_post_label_m2": [0.1, 0.2, 0.3],
            "gap_returned_active_post_label_m3": [0.1, 0.2, 0.3],
            "gap_active_days_post_label_3m": [1.0, 2.0, 3.0],
            "gap_sustained_active_2of3_post_label": [0.2, 0.3, 0.4],
            "prevalence_entropy": [0.6, 0.6, 0.6],
            "monthly_prevalence_std": [0.1, 0.1, 0.1],
            "bootstrap_prevalence_ci_width": [0.05, 0.06, 0.07],
            "threshold_candidate_rank": [1, 1, 1],
            "threshold_candidate_count": [10, 10, 10],
        }
    )
    aggregated = aggregate_definition_test_eval(grouped)
    assert "test_gap_sustained_active_2of3_post_label_ci_low" in aggregated.columns
    assert "test_gap_sustained_active_2of3_post_label_ci_high" in aggregated.columns
    assert "test_gap_sustained_active_2of3_post_label_ci_width" in aggregated.columns
    row = aggregated.iloc[0]
    assert pd.notna(row["test_gap_sustained_active_2of3_post_label_ci_width"])
    assert float(row["test_gap_sustained_active_2of3_post_label_ci_width"]) >= 0.0


def test_lock_neighbor_sensitivity_handles_compound_and_weighted_rules() -> None:
    lock_frame = _make_metrics_compound()
    promoted_candidates = pd.DataFrame(
        [
            {
                "rule_json": setup.stable_json(
                    {
                        "kind": "compound",
                        "combiner": "AND",
                        "rules": [
                            {"kind": "atomic", "metric_name": "future_sessions", "operator": ">=", "threshold": 4.0},
                            {"kind": "atomic", "metric_name": "future_active_days", "operator": ">=", "threshold": 3.0},
                        ],
                    }
                ),
                "candidate_type": "compound_pairwise_and",
            },
            {
                "rule_json": setup.stable_json(
                    {
                        "kind": "weighted",
                        "components": [
                            {"metric_name": "future_active_days", "weight": 0.25},
                            {"metric_name": "future_sessions", "weight": 0.75},
                        ],
                        "operator": ">=",
                        "threshold": 0.50,
                        "normalization": "empirical_percentile",
                    }
                ),
                "candidate_type": "weighted_pairwise_percentile_threshold",
            },
        ]
    )
    threshold_pool = pd.DataFrame(
        [
            {
                "metric_name": "future_sessions",
                "threshold": 3.0,
                "candidate_group_key": "univariate_exact_threshold::future_sessions::>=",
            },
            {
                "metric_name": "future_sessions",
                "threshold": 5.0,
                "candidate_group_key": "univariate_exact_threshold::future_sessions::>=",
            },
            {
                "metric_name": "future_active_days",
                "threshold": 2.0,
                "candidate_group_key": "univariate_exact_threshold::future_active_days::>=",
            },
            {
                "metric_name": "future_active_days",
                "threshold": 4.0,
                "candidate_group_key": "univariate_exact_threshold::future_active_days::>=",
            },
            {
                "metric_name": "future_active_days + future_sessions",
                "threshold": 0.40,
                "candidate_group_key": "weighted_pairwise_percentile_threshold::future_active_days:0.25|future_sessions:0.75",
            },
            {
                "metric_name": "future_active_days + future_sessions",
                "threshold": 0.60,
                "candidate_group_key": "weighted_pairwise_percentile_threshold::future_active_days:0.25|future_sessions:0.75",
            },
        ]
    )

    sensitivity = summarize_lock_neighbor_sensitivity(
        lock_frame=lock_frame,
        promoted_candidates=promoted_candidates,
        threshold_pool=threshold_pool,
        reference_frame=lock_frame,
    )

    assert len(sensitivity) == 2
    compound_row = sensitivity.iloc[0]
    weighted_row = sensitivity.iloc[1]
    assert compound_row["lock_structural_neighbor_count"] >= 1
    assert weighted_row["lock_weight_neighbor_count"] >= 1


def test_definition_evaluability_audit_summarizes_lock_and_final_period_support() -> None:
    previous = setup.RUNTIME_CONFIG
    apply_runtime_config(RuntimeBuildConfig.from_payload(_runtime_payload()))
    try:
        metrics = _make_metrics()
        selection_df = pd.DataFrame(
            [
                {
                    "definition_group": "definition_a",
                    "official_status": "official_winner",
                    "rule_json": setup.stable_json(
                        {"kind": "atomic", "metric_name": "future_business_active_weeks", "operator": ">=", "threshold": 3.0}
                    ),
                },
                {
                    "definition_group": "definition_b",
                    "official_status": "official_fixed_literal",
                    "rule_json": setup.stable_json(
                        {"kind": "atomic", "metric_name": "future_business_active_weeks", "operator": ">=", "threshold": 1.0}
                    ),
                },
            ]
        )
        audit, summary = build_definition_evaluability_audit(metrics, selection_df)

        assert not audit.empty
        assert not summary.empty
        assert {"definition_lock_holdout", "official_model_evaluation_holdout"}.issubset(set(audit["period_role"].unique().tolist()))
        assert {"months_with_two_classes", "months_meeting_current_official_support"}.issubset(set(summary.columns))
    finally:
        apply_runtime_config(previous)
