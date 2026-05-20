import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pads_forecasting.modeling import _make_model, model_specs
from pads_forecasting.models.baseline import SeasonalNaiveForecaster
from pads_forecasting.models.bvar import LatentComponentBVARForecaster
from pads_forecasting.models.ets import ETSForecaster
from pads_forecasting.models.ml_lags import RecursiveLagForecaster
from pads_forecasting.models.prophet_model import ProphetForecaster
from pads_forecasting.models.sarima import SARIMAXForecaster


def _phase4_model_instances():
    return [
        SeasonalNaiveForecaster(season_length=12),
        ETSForecaster(season_length=12),
        SARIMAXForecaster(season_length=12),
        ProphetForecaster(),
        RecursiveLagForecaster(
            model_type="ridge",
            lags=[1, 2, 12],
            season_length=12,
        ),
        RecursiveLagForecaster(
            model_type="elasticnet",
            lags=[1, 2, 12],
            season_length=12,
        ),
        RecursiveLagForecaster(
            model_type="lightgbm",
            lags=[1, 2, 12],
            rolling_windows=[3, 6, 12],
            season_length=12,
        ),
        LatentComponentBVARForecaster(season_length=12),
    ]


def _monthly_frame(n: int = 48) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    dates = pd.Series(pd.date_range("2018-01-01", periods=n, freq="MS"))
    seasonal = 8 * np.sin(2 * np.pi * dates.dt.month / 12)
    trend = np.arange(n) * 0.4
    y = pd.Series(100 + trend + seasonal, dtype=float)
    exog = pd.DataFrame(
        {
            "covid_shock": np.where((dates >= "2020-03-01") & (dates <= "2020-06-01"), 1, 0),
            "covid_recovery": np.where((dates >= "2020-07-01") & (dates <= "2020-12-01"), 1, 0),
        }
    )
    return y, dates, exog


def _future_frame(last_date: pd.Timestamp, horizon: int = 6) -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.Series(
        pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    )
    exog = pd.DataFrame({"covid_shock": 0, "covid_recovery": 0}, index=range(horizon))
    return dates, exog


def _assert_payload_and_intervals(model, prediction: np.ndarray, intervals: pd.DataFrame) -> None:
    assert len(prediction) == 6
    assert np.isfinite(prediction).all()
    assert set(intervals.columns) == {"lo_80", "hi_80", "lo_95", "hi_95"}
    assert len(intervals) == 6
    params = model.serializable_params()
    payload = model.mlflow_log_payload()
    assert "model_family" in params
    assert {"params", "metrics", "artifacts"} == set(payload)
    assert payload["params"]["model_family"] == params["model_family"]


def test_phase4_all_dev_lanes_expose_required_shared_interface():
    for model in _phase4_model_instances():
        fit_params = list(inspect.signature(model.fit).parameters)
        predict_params = list(inspect.signature(model.predict).parameters)
        interval_params = list(inspect.signature(model.prediction_intervals).parameters)

        assert fit_params[:3] == ["y", "exog", "config"]
        assert predict_params[:2] == ["horizon", "future_exog"]
        assert interval_params[:3] == ["horizon", "future_exog", "levels"]
        assert callable(model.fitted_values)
        assert callable(model.residuals)
        assert callable(model.serializable_params)
        assert callable(model.mlflow_log_payload)


def test_phase4_track_a_seasonal_naive_and_ets_shared_interface():
    y, dates, exog = _monthly_frame()
    _, future_exog = _future_frame(dates.iloc[-1])
    for model in [
        SeasonalNaiveForecaster(season_length=12),
        ETSForecaster(season_length=12, trend="add", seasonal="add", damped_trend=False),
    ]:
        model.fit(y, exog, {"dates": dates})
        prediction = model.predict(6, future_exog)
        intervals = model.prediction_intervals(6, future_exog)
        _assert_payload_and_intervals(model, prediction, intervals)
        assert len(model.fitted_values()) == len(y)
        assert len(model.residuals()) == len(y)


def test_phase4_track_b_sarimax_shared_interface_and_native_intervals():
    y, dates, exog = _monthly_frame()
    model = SARIMAXForecaster(
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        trend="n",
        season_length=12,
        use_exog=True,
    )
    _, future_exog = _future_frame(dates.iloc[-1])

    model.fit(y, exog, {"dates": dates})
    prediction = model.predict(6, future_exog)
    intervals = model.prediction_intervals(6, future_exog)

    _assert_payload_and_intervals(model, prediction, intervals)


def test_phase4_track_c_prophet_shared_interface_and_native_intervals():
    pytest.importorskip("prophet")
    y, dates, exog = _monthly_frame()
    _, future_exog = _future_frame(dates.iloc[-1])
    model = ProphetForecaster(
        yearly_seasonality=3,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5.0,
        use_covid_regressors=True,
    )

    model.fit(y, exog, {"dates": dates})
    prediction = model.predict(6, future_exog)
    intervals = model.prediction_intervals(6, future_exog)

    _assert_payload_and_intervals(model, prediction, intervals)


def test_phase4_track_d_ridge_elasticnet_and_lightgbm_mlforecast_interface():
    y, dates, exog = _monthly_frame()
    _, future_exog = _future_frame(dates.iloc[-1])
    models = [
        RecursiveLagForecaster(
            model_type="ridge",
            lags=[1, 2, 3, 12],
            model_params={"alpha": 1.0},
            season_length=12,
        ),
        RecursiveLagForecaster(
            model_type="elasticnet",
            lags=[1, 2, 3, 12],
            model_params={"alpha": 0.1, "l1_ratio": 0.5},
            season_length=12,
        ),
        RecursiveLagForecaster(
            model_type="lightgbm",
            lags=[1, 2, 3, 12],
            rolling_windows=[3, 6],
            model_params={
                "max_depth": 2,
                "num_leaves": 4,
                "n_estimators": 10,
                "learning_rate": 0.05,
                "min_data_in_leaf": 8,
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
            },
            seed=42,
            season_length=12,
        ),
    ]

    for model in models:
        model.fit(y, exog, {"dates": dates})
        prediction = model.predict(6, future_exog)
        intervals = model.prediction_intervals(6, future_exog)
        _assert_payload_and_intervals(model, prediction, intervals)

    assert models[-1].serializable_params()["fitted_engine"] == "mlforecast_lightgbm"


def test_phase4_track_e_bvar_pymc_arviz_interface():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    y, dates, exog = _monthly_frame(30)
    train_frame = pd.DataFrame(
        {
            "data": dates,
            "y": y,
            "br_component_observed": y * 0.82,
            "acquired_component_observed": y * 0.18,
        }
    )
    train_frame.loc[train_frame.index >= 20, "br_component_observed"] = pd.NA
    train_frame.loc[train_frame.index >= 20, "acquired_component_observed"] = pd.NA
    model = LatentComponentBVARForecaster(
        lags=1,
        minnesota_lambda=0.2,
        cross_lag_shrinkage=0.1,
        covid_exog=True,
        draws=8,
        tune=8,
        seed=42,
        season_length=12,
    )

    model.fit(y, exog, {"dates": dates, "train_frame": train_frame})
    prediction = model.predict(6)
    intervals = model.prediction_intervals(6, None)
    summary = model.arviz_summary()

    _assert_payload_and_intervals(model, prediction, intervals)
    assert model.serializable_params()["design"] == "component_bvar"
    assert not summary.empty


def test_phase4_model_specs_cover_all_dev_lanes_and_grid_params():
    params = {
        "models": {
            "seasonal_naive": {"enabled": True, "season_length": 12},
            "ets": {
                "enabled": True,
                "grid": {
                    "trend": ["add"],
                    "seasonal": ["add"],
                    "damped_trend": [False, True],
                    "use_boxcox": [False, True],
                },
            },
            "sarimax": {
                "enabled": True,
                "grid": {
                    "p": [0],
                    "d": [1],
                    "q": [0, 1],
                    "P": [0],
                    "D": [1],
                    "Q": [0, 1],
                    "m": [12],
                    "trend": ["n", "c"],
                },
                "covid_modes": ["none", "covid"],
            },
            "prophet": {
                "enabled": True,
                "optional": False,
                "grid": {
                    "yearly_seasonality": [3, 5],
                    "seasonality_mode": ["additive", "multiplicative"],
                    "changepoint_prior_scale": [0.01, 0.05],
                    "seasonality_prior_scale": [1.0, 5.0],
                    "covid_regressors": [False, True],
                },
            },
            "ridge": {"enabled": True, "alpha_grid": [1.0], "lags": [1, 2, 12]},
            "elasticnet": {
                "enabled": True,
                "alpha_grid": [0.1],
                "l1_ratio_grid": [0.5],
                "lags": [1, 2, 12],
            },
            "lightgbm": {
                "enabled": True,
                "grid": {
                    "lags": [[1, 2, 12]],
                    "rolling_windows": [[3]],
                    "max_depth": [2],
                    "num_leaves": [4],
                    "n_estimators": [25],
                    "learning_rate": [0.05],
                    "min_data_in_leaf": [8, 12],
                    "lambda_l1": [0.0, 0.1],
                    "lambda_l2": [0.0, 0.5],
                    "forecast_strategy": ["recursive"],
                },
            },
            "bvar": {
                "enabled": True,
                "optional": False,
                "grid": {
                    "lags": [1, 2],
                    "minnesota_lambda": [0.2],
                    "cross_lag_shrinkage": [0.1],
                    "covid_exog": [True],
                    "draws": [20, 40],
                    "tune": [20],
                },
            },
        }
    }
    specs = model_specs(params, stage="model_comparison", include_optional=True)
    families = {spec["family"] for spec in specs}
    lightgbm_specs = [spec for spec in specs if spec["family"] == "lightgbm"]
    prophet_specs = [spec for spec in specs if spec["family"] == "prophet"]
    sarimax_specs = [spec for spec in specs if spec["family"] == "sarimax"]
    bvar_specs = [spec for spec in specs if spec["family"] == "bvar"]

    assert {
        "seasonal_naive",
        "ets",
        "sarimax",
        "prophet",
        "ridge",
        "elasticnet",
        "lightgbm",
        "bvar",
    }.issubset(families)
    assert len(lightgbm_specs) == 24
    assert len(prophet_specs) == 48
    assert len(sarimax_specs) == 24
    assert len(bvar_specs) == 12
    for family in families:
        family_modes = {spec["covid_mode"] for spec in specs if spec["family"] == family}
        assert {"none", "adjusted_target"}.issubset(family_modes)
    assert any(spec["params"]["lambda_l1"] == 0.1 for spec in lightgbm_specs)
    assert {spec["params"]["seasonality_mode"] for spec in prophet_specs} == {
        "additive",
        "multiplicative",
    }
    assert {spec["params"]["trend"] for spec in sarimax_specs} == {"n", "c"}
    assert {spec["params"]["draws"] for spec in bvar_specs} == {20, 40}

    for spec in specs:
        model = _make_model(spec, season_length=12, seed=42)
        assert hasattr(model, "serializable_params")


def test_phase4_yaml_grids_match_final_plan_exactly():
    params_path = Path("conf/base/parameters_models.yml")
    models = yaml.safe_load(params_path.read_text())["models"]

    assert models["ets"]["grid"]["trend"] == ["add"]
    assert models["ets"]["grid"]["seasonal"] == ["add"]
    assert models["ets"]["grid"]["damped_trend"] == [False, True]
    assert models["ets"]["grid"]["use_boxcox"] == [False, True]

    assert models["sarimax"]["grid"] == {
        "p": [0, 1, 2],
        "d": [0, 1],
        "q": [0, 1, 2],
        "P": [0, 1],
        "D": [1],
        "Q": [0, 1],
        "m": [12],
        "trend": ["n", "c"],
    }
    assert models["sarimax"]["covid_modes"] == ["none", "adjusted_target", "covid"]

    assert models["prophet"]["grid"] == {
        "yearly_seasonality": [3, 5, 8],
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_prior_scale": [0.01, 0.05, 0.1],
        "seasonality_prior_scale": [1.0, 5.0, 10.0],
        "covid_regressors": [False, True],
    }
    assert models["prophet"]["covid_modes"] == ["none", "adjusted_target", "regressors"]

    assert models["lightgbm"]["grid"]["max_depth"] == [2, 3]
    assert models["lightgbm"]["grid"]["num_leaves"] == [4, 7, 15]
    assert models["lightgbm"]["grid"]["n_estimators"] == [25, 50, 100]
    assert models["lightgbm"]["grid"]["learning_rate"] == [0.03, 0.05, 0.1]
    assert models["lightgbm"]["grid"]["min_data_in_leaf"] == [8, 12, 18]
    assert models["lightgbm"]["grid"]["lambda_l1"] == [0.0, 0.1]
    assert models["lightgbm"]["grid"]["lambda_l2"] == [0.0, 0.5, 1.0]
    assert models["lightgbm"]["grid"]["forecast_strategy"] == ["recursive"]
    assert models["lightgbm"]["covid_modes"] == ["none", "adjusted_target", "features"]

    assert models["bvar"]["grid"] == {
        "lags": [1, 2],
        "minnesota_lambda": [0.1, 0.2, 0.5],
        "cross_lag_shrinkage": [0.05, 0.1],
        "covid_exog": [True],
        "draws": [1000],
        "tune": [1000],
    }


def test_phase4_full_config_specs_have_expected_plan_counts():
    models = yaml.safe_load(Path("conf/base/parameters_models.yml").read_text())
    specs = model_specs(models, stage="model_comparison", include_optional=True)
    counts = pd.Series([spec["family"] for spec in specs]).value_counts().to_dict()

    assert counts == {
        "lightgbm": 2916,
        "sarimax": 432,
        "prophet": 162,
        "bvar": 36,
        "elasticnet": 27,
        "ets": 8,
        "ridge": 12,
        "seasonal_naive": 2,
    }


def test_phase4_old_data_gate_keeps_controlled_subset_without_ets_boxcox():
    models = yaml.safe_load(Path("conf/base/parameters_models.yml").read_text())
    specs = model_specs(models, stage="old_data_gate", include_optional=False)
    ets_specs = [spec for spec in specs if spec["family"] == "ets"]

    assert {spec["family"] for spec in specs} == {
        "seasonal_naive",
        "ets",
        "ridge",
        "elasticnet",
        "lightgbm",
    }
    assert len(ets_specs) == 4
    assert all(spec["params"]["use_boxcox"] is False for spec in ets_specs)
