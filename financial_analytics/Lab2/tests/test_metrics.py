import numpy as np
import pandas as pd

from pads_forecasting.metrics import (
    bootstrap_mase_uncertainty,
    mase,
    mase_denominator,
    score_forecast,
    summarize_horizon_metrics,
)


def test_mase_denominator_uses_training_series_only():
    train = pd.Series([10.0] * 12 + [20.0] * 12)
    valid = pd.Series([1_000.0] * 12)
    pred = pd.Series([900.0] * 12)

    denom_before = mase_denominator(train, season_length=12)
    score = mase(valid, pred, train, season_length=12)

    assert denom_before == 10.0
    assert score == 10.0


def test_common_mase_uses_supplied_fixed_denominator():
    train = pd.Series([10.0] * 12 + [20.0] * 12)
    valid = pd.Series([30.0, 40.0])
    pred = pd.Series([32.0, 37.0])

    scores = score_forecast(
        valid,
        pred,
        train,
        season_length=12,
        common_mase_denominator=5.0,
    )

    assert scores["mase_denominator"] == 10.0
    assert scores["common_mase_denominator"] == 5.0
    assert scores["mase"] == 0.25
    assert scores["common_mase"] == 0.5


def test_horizon_summary_and_bootstrap_uncertainty_use_common_mase():
    rows = []
    for model_id, values in {
        "seasonal_naive": [1.0, 1.0, 1.0, 1.0],
        "ets_add": [0.7, 0.8, 0.9, 0.6],
    }.items():
        for index, common_mase in enumerate(values, start=1):
            rows.append(
                {
                    "stage": "model_comparison",
                    "target_strategy": "proforma_sum",
                    "model_id": model_id,
                    "model_family": model_id,
                    "fold_name": f"fold_{1 + (index > 2)}",
                    "horizon_index": 1 + ((index - 1) % 2),
                    "abs_error": common_mase * 10,
                    "squared_error": (common_mase * 10) ** 2,
                    "error": common_mase * 10,
                    "local_mase": common_mase,
                    "common_mase": common_mase,
                    "status": "ok",
                }
            )
    horizon = pd.DataFrame(rows)

    summary = summarize_horizon_metrics(horizon)
    uncertainty = bootstrap_mase_uncertainty(horizon, n_bootstrap=100, seed=123)
    ets_uncertainty = uncertainty[uncertainty["model_id"].eq("ets_add")].iloc[0]

    assert set(summary["horizon_index"]) == {1, 2}
    assert np.isclose(ets_uncertainty["candidate_mean_common_mase"], 0.75)
    assert ets_uncertainty["bootstrap_probability_beats_seasonal_naive"] > 0.95
