import numpy as np
import pandas as pd
import pytest

from pads_forecasting.leakage import LeakageError, assert_fold_datasets_no_leakage
from pads_forecasting.metrics import score_forecast, summarize_cv, utilsforecast_point_metrics
from pads_forecasting.modeling import evaluate_cv
from pads_forecasting.pipelines.validation.nodes import build_folds_metadata, make_fold_slices
from pads_forecasting.selection import (
    admissible_strategies,
    old_data_gate_decision,
    select_final_model,
)


def _validation_config():
    return {
        "horizon": 12,
        "season_length": 12,
        "folds": [
            {
                "name": "fold_2021_stress",
                "train_end": "2020-12-01",
                "valid_start": "2021-01-01",
                "valid_end": "2021-12-01",
                "role": "stress",
            },
            {
                "name": "fold_2022_normal",
                "train_end": "2021-12-01",
                "valid_start": "2022-01-01",
                "valid_end": "2022-12-01",
                "role": "normal",
            },
        ],
    }


def _strategy_frame(target_source: str = "observed_consolidated") -> pd.DataFrame:
    dates = pd.date_range("2019-07-01", "2022-12-01", freq="MS")
    return pd.DataFrame(
        {
            "data": dates,
            "y": np.arange(len(dates), dtype=float) + 100.0,
            "target_strategy": "post_only",
            "strategy_family": "post_only",
            "alpha": 1.0,
            "beta": 1.0,
            "target_source": target_source,
            "covid_shock": 0,
            "covid_recovery": 0,
            "month": dates.month,
            "trend_index": range(len(dates)),
        }
    )


def test_validation_metadata_enforces_rolling_origin_horizon():
    folds = build_folds_metadata(_validation_config())

    assert list(folds["fold_name"]) == ["fold_2021_stress", "fold_2022_normal"]
    assert folds["horizon"].eq(12).all()
    assert folds["expected_horizon"].eq(12).all()
    assert folds["season_length"].eq(12).all()


def test_validation_metadata_rejects_wrong_horizon():
    config = _validation_config()
    config["folds"][0] = {
        **config["folds"][0],
        "valid_end": "2021-11-01",
    }

    with pytest.raises(LeakageError):
        build_folds_metadata(config)


def test_make_fold_slices_creates_fold_local_train_and_validation_windows():
    fold = build_folds_metadata(_validation_config()).iloc[0]
    train, valid = make_fold_slices(_strategy_frame(), fold)

    assert train["data"].max() == pd.Timestamp("2020-12-01")
    assert valid["data"].min() == pd.Timestamp("2021-01-01")
    assert valid["data"].max() == pd.Timestamp("2021-12-01")
    assert set(train["data"]).isdisjoint(set(valid["data"]))


def test_fold_dataset_leakage_checks_reject_overlapping_train_validation_dates():
    frame = _strategy_frame()
    train = frame.query("data <= '2021-01-01'")
    valid = frame.query("'2021-01-01' <= data <= '2021-12-01'")

    with pytest.raises(LeakageError):
        assert_fold_datasets_no_leakage(
            train,
            valid,
            train_end="2020-12-01",
            valid_start="2021-01-01",
            valid_end="2021-12-01",
        )


def test_fold_dataset_leakage_checks_reject_reconstructed_validation_target():
    train = _strategy_frame().query("data <= '2020-12-01'")
    valid = _strategy_frame("reconstructed_or_raw_pre_acquisition").query(
        "'2021-01-01' <= data <= '2021-12-01'"
    )

    with pytest.raises(LeakageError):
        assert_fold_datasets_no_leakage(
            train,
            valid,
            train_end="2020-12-01",
            valid_start="2021-01-01",
            valid_end="2021-12-01",
        )


def test_evaluate_cv_fails_loudly_on_leakage():
    folds = build_folds_metadata(_validation_config()).head(1)
    target_strategies = {"strategies": {"post_only": _strategy_frame("bad_source")}}
    specs = [
        {
            "model_id": "seasonal_naive",
            "family": "seasonal_naive",
            "params": {"season_length": 12},
            "covid_mode": "none",
            "complexity": "simple",
        }
    ]

    with pytest.raises(LeakageError):
        evaluate_cv(
            stage="phase3_test",
            target_strategies=target_strategies,
            strategy_names=["post_only"],
            specs=specs,
            folds_metadata=folds,
            validation_params=_validation_config(),
            project_params={"seed": 42},
        )


def test_score_forecast_reports_fold_local_mase_denominator_and_utilsforecast_metrics():
    train = pd.Series([10.0] * 12 + [20.0] * 12)
    valid = pd.Series([30.0, 40.0])
    pred = pd.Series([32.0, 37.0])

    utils_metrics = utilsforecast_point_metrics(valid, pred)
    scores = score_forecast(valid, pred, train, season_length=12)

    assert scores["mase_denominator"] == 10.0
    assert scores["mae"] == utils_metrics["mae"]
    assert scores["rmse"] == utils_metrics["rmse"]
    assert scores["bias"] == utils_metrics["bias"]


def test_summarize_cv_and_old_data_gate_apply_stability_and_overfit_rules():
    rows = []
    for strategy, mase_value, rel_mae, ratio in [
        ("post_only", 1.00, 0.95, 1.1),
        ("raw_full", 0.90, 0.90, 1.1),
        ("proforma_sum", 0.80, 0.80, 1.2),
    ]:
        for fold_name, role in [("fold_2021_stress", "stress"), ("fold_2022_normal", "normal")]:
            rows.append(
                {
                    "stage": "old_data_gate",
                    "target_strategy": strategy,
                    "model_id": "seasonal_naive",
                    "fold_name": fold_name,
                    "fold_role": role,
                    "mae": mase_value * 10,
                    "rmse": mase_value * 12,
                    "mase": mase_value,
                    "bias": 0.0,
                    "relative_mae_vs_seasonal_naive": rel_mae,
                    "train_valid_mae_ratio": ratio,
                }
            )

    summary = summarize_cv(pd.DataFrame(rows))
    decision = old_data_gate_decision(
        summary,
        {
            "old_data_min_improvement_pct": 3.0,
            "max_cv_mase": 0.40,
            "train_valid_ratio_reject": 3.0,
        },
    )
    proforma = decision[decision["target_strategy"].eq("proforma_sum")].iloc[0]

    assert bool(proforma["passed"]) is True
    assert bool(proforma["beats_raw_full"]) is True
    assert bool(proforma["beats_seasonal_naive"]) is True


def test_admissible_strategies_exclude_old_data_without_stage_a_pass():
    gate = pd.DataFrame(
        [
            {"record_type": "decision", "target_strategy": "proforma_sum", "passed": False},
            {"record_type": "decision", "target_strategy": "calibrated_alpha", "passed": True},
            {"record_type": "summary", "target_strategy": "raw_full", "passed": np.nan},
        ]
    )

    assert admissible_strategies(gate) == ["post_only", "calibrated_alpha"]


def test_select_final_model_prefers_best_predictive_cv_after_data_driven_stability():
    selection_params = {
        "old_data_min_improvement_pct": 0.0,
        "stability_iqr_multiplier": 1.5,
        "stability_min_fold_variation_epsilon": 1e-12,
        "train_valid_ratio_reject": 3.0,
    }
    summary = pd.DataFrame(
        [
            {
                "stage": "model_comparison",
                "target_strategy": "calibrated_alpha",
                "model_id": "ets_add",
                "model_family": "ets",
                "complexity": "simple",
                "normal_mean_mase": 0.98,
                "mean_mase": 0.98,
                "cv_mae": 0.20,
                "cv_rmse": 0.20,
                "cv_mase": 0.20,
                "cv_relative_mae_vs_seasonal_naive": 0.20,
                "max_mase": 1.1,
                "mean_relative_mae_vs_seasonal_naive": 0.8,
                "mean_train_valid_ratio": 1.2,
                "folds": 3,
            },
            {
                "stage": "model_comparison",
                "target_strategy": "proforma_sum",
                "model_id": "ets_add",
                "model_family": "ets",
                "complexity": "simple",
                "normal_mean_mase": 1.00,
                "mean_mase": 1.00,
                "cv_mae": 0.20,
                "cv_rmse": 0.20,
                "cv_mase": 0.20,
                "cv_relative_mae_vs_seasonal_naive": 0.20,
                "max_mase": 1.1,
                "mean_relative_mae_vs_seasonal_naive": 0.85,
                "mean_train_valid_ratio": 1.2,
                "folds": 3,
            },
            {
                "stage": "model_comparison",
                "target_strategy": "post_only",
                "model_id": "lightgbm_d2",
                "model_family": "lightgbm",
                "complexity": "complex",
                "normal_mean_mase": 0.50,
                "mean_mase": 0.50,
                "cv_mae": 0.20,
                "cv_rmse": 0.20,
                "cv_mase": 0.20,
                "cv_relative_mae_vs_seasonal_naive": 0.20,
                "max_mase": 0.8,
                "mean_relative_mae_vs_seasonal_naive": 0.7,
                "mean_train_valid_ratio": 1.2,
                "folds": 2,
            },
        ]
    )

    selected = select_final_model(summary, selection_params)
    selected_row = selected[selected["selected"]].iloc[0]
    incomplete_row = selected[selected["model_id"].eq("lightgbm_d2")].iloc[0]

    assert selected_row["target_strategy"] == "calibrated_alpha"
    assert bool(incomplete_row["eligible_for_selection"]) is False


def test_select_final_model_audits_stability_without_overriding_mase_objective():
    selection_params = {
        "old_data_min_improvement_pct": 0.0,
        "stability_iqr_multiplier": 1.5,
        "stability_min_fold_variation_epsilon": 1e-12,
        "train_valid_ratio_reject": 3.0,
    }
    rows = []
    for idx, cv_mase in enumerate([0.18, 0.19, 0.20, 0.21, 3.00]):
        rows.append(
            {
                "stage": "model_comparison",
                "target_strategy": "calibrated_alpha",
                "model_id": f"model_{idx}",
                "model_family": "ets",
                "complexity": "simple",
                "normal_mean_mase": 0.40 if idx == 4 else 0.50 + idx / 100,
                "mean_mase": 0.50,
                "cv_mae": cv_mase,
                "cv_rmse": cv_mase,
                "cv_mase": cv_mase,
                "cv_relative_mae_vs_seasonal_naive": cv_mase,
                "max_mase": 1.1,
                "mean_relative_mae_vs_seasonal_naive": 0.8,
                "mean_train_valid_ratio": 1.2,
                "folds": 3,
            }
        )

    selected = select_final_model(pd.DataFrame(rows), selection_params)
    unstable = selected[selected["model_id"].eq("model_4")].iloc[0]
    selected_row = selected[selected["selected"]].iloc[0]

    assert bool(unstable["eligible_for_selection"]) is True
    assert bool(unstable["stability_passed"]) is False
    assert "high fold-variation outlier" in unstable["stability_reason"]
    assert selected_row["model_id"] == "model_4"
