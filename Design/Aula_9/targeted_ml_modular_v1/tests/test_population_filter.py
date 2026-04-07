from __future__ import annotations

import pandas as pd

from targeted_ml.pipelines.modelled_to_ml.analysis_setup import RuntimeBuildConfig, apply_official_population_filter, apply_runtime_config
from targeted_ml.pipelines.modelled_to_ml.runner import build_population_sensitivity_summary


def _runtime_payload(official_population_filter: str) -> dict[str, object]:
    return {
        "analysis_kind": "activity",
        "official_population_filter": official_population_filter,
        "enabled_tracks": ["S1", "S7", "S1_PLUS_S7", "STRICT_CONTEXT"],
        "label_window_days": 30,
        "post_label_block_days": 30,
        "post_label_block_count": 3,
        "external_validators": [],
        "definition_a_enabled": True,
        "definition_a_strategy": "univariate_exact",
        "definition_a_candidate_metrics": [],
        "definition_a_sql_file": "",
        "definition_a_python_strategy": "",
        "definition_b": {"definition_name": "definition_b", "metric_name": "future_business_active_weeks", "operator": ">=", "threshold": 1.0},
        "definition_b_sql_file": "",
        "definition_b_python_strategy": "",
        "max_outer_test_months": 6,
        "definition_selection_holdout_months": 6,
        "min_official_valid_outer_folds": 2,
        "min_official_test_rows": 50,
        "min_official_test_positives": 5,
        "min_official_test_negatives": 20,
        "tuning_enabled": True,
        "tuning_n_iter": 8,
        "tuning_max_inner_splits": 3,
        "tuning_scoring": "neg_brier_score",
        "model_family_scope": ["logistic_regression", "random_forest", "catboost"],
        "model_comparison_workers": 1,
        "calibration_method": "sigmoid",
        "feature_importance_permutation_repeats": 5,
        "cluster_k_candidates": [2, 3],
        "cluster_bootstrap_iterations": 2,
        "cluster_sample_size": 100,
        "registered_band_policies": [],
        "heavy_user_percentile_policies": [],
    }


def test_apply_official_population_filter_same_month_only() -> None:
    previous = RuntimeBuildConfig.from_payload(_runtime_payload("all_observed_first_use"))
    apply_runtime_config(RuntimeBuildConfig.from_payload(_runtime_payload("same_month_entry_only")))
    try:
        frame = pd.DataFrame(
            {
                "teacher_unique_id": ["a", "b", "c"],
                "months_after_entry": [0, 2, 0],
            }
        )
        filtered = apply_official_population_filter(frame)
        assert filtered["teacher_unique_id"].tolist() == ["a", "c"]
    finally:
        apply_runtime_config(previous)


def test_population_sensitivity_summary_separates_same_month_and_delayed_entry() -> None:
    previous = RuntimeBuildConfig.from_payload(_runtime_payload("all_observed_first_use"))
    apply_runtime_config(RuntimeBuildConfig.from_payload(_runtime_payload("all_observed_first_use")))
    try:
        frame = pd.DataFrame(
            {
                "teacher_unique_id": ["a", "b", "c", "d"],
                "first_month": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
                "months_after_entry": [0, 2, 0, 3],
                "definition_b_label": [1, 0, 1, 0],
                "returned_active_post_label_m1": [1, 0, 1, 0],
                "returned_active_post_label_m2": [1, 0, 1, 0],
                "returned_active_post_label_m3": [1, 0, 1, 0],
                "active_days_post_label_3m": [3, 0, 3, 0],
                "sustained_active_2of3_post_label": [1, 0, 1, 0],
            }
        )
        summary = build_population_sensitivity_summary(frame, ["definition_b_label"])
        assert set(summary["population_group"]) == {
            "all_observed_first_use",
            "same_month_entry_only",
            "delayed_entry_observed",
        }
        same_month = summary[summary["population_group"] == "same_month_entry_only"].iloc[0]
        delayed = summary[summary["population_group"] == "delayed_entry_observed"].iloc[0]
        assert int(same_month["rows"]) == 2
        assert int(delayed["rows"]) == 2
        assert float(same_month["months_after_entry_mean"]) == 0.0
        assert float(delayed["months_after_entry_mean"]) == 2.5
    finally:
        apply_runtime_config(previous)
