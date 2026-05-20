import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pads_forecasting.modeling import model_specs
from pads_forecasting.pipelines.final_forecast.nodes import run_final_forecast


def _reduced_all_lane_models() -> dict[str, Any]:
    return {
        "models": {
            "seasonal_naive": {"enabled": True, "season_length": 12},
            "ets": {
                "enabled": True,
                "grid": {
                    "trend": ["add"],
                    "seasonal": ["add"],
                    "damped_trend": [False],
                    "use_boxcox": [False],
                },
            },
            "sarimax": {
                "enabled": True,
                "grid": {
                    "p": [0],
                    "d": [1],
                    "q": [1],
                    "P": [0],
                    "D": [1],
                    "Q": [1],
                    "m": [12],
                    "trend": ["n"],
                },
                "covid_modes": ["none"],
            },
            "prophet": {
                "enabled": True,
                "optional": True,
                "grid": {
                    "yearly_seasonality": [3],
                    "seasonality_mode": ["additive"],
                    "changepoint_prior_scale": [0.05],
                    "seasonality_prior_scale": [5.0],
                    "covid_regressors": [False],
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
                    "min_data_in_leaf": [8],
                    "lambda_l1": [0.0],
                    "lambda_l2": [0.0],
                    "forecast_strategy": ["recursive"],
                },
            },
            "bvar": {
                "enabled": True,
                "optional": True,
                "grid": {
                    "lags": [1],
                    "minnesota_lambda": [0.2],
                    "cross_lag_shrinkage": [0.1],
                    "covid_exog": [True],
                    "draws": [8],
                    "tune": [8],
                },
            },
        }
    }


def _target_strategies() -> dict[str, Any]:
    dates = pd.Series(pd.date_range("2014-01-01", "2023-12-01", freq="MS"))
    seasonal = 10 * np.sin(2 * np.pi * dates.dt.month / 12)
    y = 100 + np.arange(len(dates)) * 0.5 + seasonal
    frame = pd.DataFrame(
        {
            "data": dates,
            "y": y,
            "target_strategy": "proforma_sum",
            "strategy_family": "proforma_sum",
            "alpha": 1.0,
            "beta": 1.0,
            "covid_shock": 0,
            "covid_recovery": 0,
            "month": dates.dt.month,
            "trend_index": np.arange(len(dates)),
        }
    )
    return {
        "strategies": {
            "proforma_sum": frame,
            "calibrated_alpha": frame.assign(target_strategy="calibrated_alpha", alpha=1.5),
        },
        "alpha_candidates": {
            1.0: frame.assign(target_strategy="calibrated_alpha", alpha=1.0),
            1.5: frame.assign(target_strategy="calibrated_alpha", alpha=1.5),
        },
    }


def _model_selection(models: dict[str, Any], selected_family: str) -> pd.DataFrame:
    rows = []
    selected_seen = False
    for rank, spec in enumerate(
        model_specs(models, stage="model_comparison", include_optional=True),
        start=1,
    ):
        model_id = spec["model_id"]
        is_selected = spec["family"] == selected_family and not selected_seen
        if is_selected:
            selected_seen = True
            if spec["family"] == "prophet":
                model_id = f"prophet_y{spec['params']['yearly_seasonality']}_none"
        rows.append(
            {
                "stage": "model_comparison",
                "target_strategy": "proforma_sum",
                "model_id": model_id,
                "model_family": spec["family"],
                "selected": is_selected,
                "rank": rank,
                "normal_mean_mase": 0.5 + rank / 100,
                "mean_mase": 0.6 + rank / 100,
                "selection_reason": "selected for phase 8 test" if is_selected else "not selected",
            }
        )
    assert selected_seen
    return pd.DataFrame(rows)


class _FakeForecaster:
    def __init__(self, family: str) -> None:
        self.family = family

    def fit(self, y, exog=None, config=None):
        self.y = pd.Series(y).astype(float).reset_index(drop=True)
        return self

    def predict(self, horizon, future_exog=None, config=None):
        base = float(self.y.iloc[-1])
        return base + np.arange(1, horizon + 1)

    def fitted_values(self):
        return self.y - 1

    def residuals(self):
        return self.y - self.fitted_values()

    def prediction_intervals(self, horizon, future_exog=None, levels=(80, 95), config=None):
        point = self.predict(horizon, future_exog, config)
        return pd.DataFrame(
            {
                "lo_80": point - 10,
                "hi_80": point + 10,
                "lo_95": point - 20,
                "hi_95": point + 20,
            }
        )

    def serializable_params(self):
        return {"model_family": self.family}

    def mlflow_log_payload(self):
        return {"params": self.serializable_params(), "metrics": {}, "artifacts": {}}


class _FakeRun:
    def __init__(self, mlflow, run: dict) -> None:
        self.mlflow = mlflow
        self.run = run
        self.previous = None

    def __enter__(self):
        self.previous = self.mlflow.current
        self.mlflow.current = self.run
        self.mlflow.runs.append(self.run)
        return self.run

    def __exit__(self, exc_type, exc, tb):
        self.mlflow.current = self.previous
        return False


class _FakeMlflow(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.current = None
        self.runs: list[dict] = []
        self.artifacts: list[str] = []

    def active_run(self):
        return object()

    def start_run(self, run_name: str, nested: bool = False):
        return _FakeRun(self, {"run_name": run_name, "nested": nested, "params": {}, "metrics": {}})

    def log_param(self, key: str, value: object) -> None:
        self.current["params"][key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.current["metrics"][key] = value

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(Path(path).name)


def test_phase8_final_forecast_refits_all_dev_lanes_exports_challengers_and_mlflow(
    monkeypatch,
    tmp_path,
):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(
        "pads_forecasting.pipelines.final_forecast.nodes._make_model",
        lambda spec, **kwargs: _FakeForecaster(spec["family"]),
    )

    models = _reduced_all_lane_models()
    expected_columns = ["data", "previsao", "lo_80", "hi_80", "lo_95", "hi_95"]
    for family in {
        "seasonal_naive",
        "ets",
        "sarimax",
        "prophet",
        "ridge",
        "elasticnet",
        "lightgbm",
        "bvar",
    }:
        forecast, previsao, challengers, metadata = run_final_forecast(
            target_strategies=_target_strategies(),
            model_selection=_model_selection(models, family),
            robust_alpha_summary=pd.DataFrame(),
            project={"seed": 42},
            data={
                "acquisition_date": "2019-07-01",
                "final_forecast_start": "2024-01-01",
                "horizon": 12,
            },
            validation={"season_length": 12},
            interventions={"covid": {"future_value": 0}},
            models=models,
            outputs={"figures_dir": str(tmp_path / "figures")},
        )

        assert list(forecast.columns) == expected_columns
        assert list(previsao.columns) == ["data", "previsao"]
        assert len(forecast) == 12
        assert len(previsao) == 12
        assert forecast["data"].iloc[0] == "2024-01-01"
        assert forecast["data"].iloc[-1] == "2024-12-01"
        assert challengers["candidate_role"].eq("challenger").all()
        assert challengers["model_id"].nunique() == 2
        assert len(challengers) == 24
        assert metadata["future_covid_shock_sum"].iloc[0] == 0
        assert metadata["future_covid_recovery_sum"].iloc[0] == 0
        assert "selected_final_alpha_source" in metadata
        assert metadata["train_start"].iloc[0] == "2014-01-01"
        assert metadata["train_end"].iloc[0] == "2023-12-01"
        if family == "prophet":
            assert metadata["diagnostic_spec_resolution"].iloc[0] == "compatible_model_family"

    assert (tmp_path / "figures/final_forecast_intervals.png").exists()
    assert "forecast_intervals.parquet" in fake_mlflow.artifacts
    assert "previsao.csv" in fake_mlflow.artifacts
    assert "challenger_forecasts.parquet" in fake_mlflow.artifacts
    assert "final_model_metadata.parquet" in fake_mlflow.artifacts
    assert "selected_final_model.pkl" in fake_mlflow.artifacts


def test_phase8_calibrated_alpha_final_refit_uses_robust_alpha_not_screening_alpha(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "pads_forecasting.pipelines.final_forecast.nodes._make_model",
        lambda spec, **kwargs: _FakeForecaster(spec["family"]),
    )
    models = _reduced_all_lane_models()
    model_selection = _model_selection(models, "ets")
    model_selection.loc[model_selection["selected"], "target_strategy"] = "calibrated_alpha"
    robust_alpha_summary = pd.DataFrame(
        [
            {
                "model_family": "ets",
                "best_alpha_normal_folds_only_grid": 1.5,
                "mean_common_mase_delta_vs_alpha_one": -0.1,
                "normal_common_mase_delta_vs_alpha_one": -0.1,
            }
        ]
    )

    forecast, _previsao, _challengers, metadata = run_final_forecast(
        target_strategies=_target_strategies(),
        model_selection=model_selection,
        robust_alpha_summary=robust_alpha_summary,
        project={"seed": 42},
        data={
            "acquisition_date": "2019-07-01",
            "final_forecast_start": "2024-01-01",
            "horizon": 12,
        },
        validation={
            "season_length": 12,
            "robust_alpha": {"final_alpha_objective": "normal_folds"},
        },
        interventions={"covid": {"future_value": 0}},
        models=models,
        outputs={"figures_dir": str(tmp_path / "figures")},
    )

    assert len(forecast) == 12
    assert metadata["selected_final_alpha"].iloc[0] == 1.5
    assert metadata["selected_final_alpha_source"].iloc[0] == (
        "robust_alpha_best_alpha_normal_folds_only_grid"
    )
