import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pads_forecasting.modeling import model_specs
from pads_forecasting.pipelines.diagnostics.nodes import (
    DIAGNOSTIC_METRIC_COLUMNS,
    INTERVAL_COVERAGE_COLUMNS,
    INTERVAL_PREDICTION_COLUMNS,
    run_residual_diagnostics,
)
from pads_forecasting.pipelines.validation.nodes import build_folds_metadata


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


def _fold_metadata() -> pd.DataFrame:
    return build_folds_metadata(
        {
            "horizon": 12,
            "season_length": 12,
            "folds": [
                {
                    "name": "fold_2021_stress",
                    "role": "stress",
                    "train_end": "2020-12-01",
                    "valid_start": "2021-01-01",
                    "valid_end": "2021-12-01",
                },
                {
                    "name": "fold_2022_normal",
                    "role": "normal",
                    "train_end": "2021-12-01",
                    "valid_start": "2022-01-01",
                    "valid_end": "2022-12-01",
                },
                {
                    "name": "fold_2023_normal",
                    "role": "normal",
                    "train_end": "2022-12-01",
                    "valid_start": "2023-01-01",
                    "valid_end": "2023-12-01",
                },
            ],
        }
    )


def _target_strategies() -> dict[str, Any]:
    dates = pd.Series(pd.date_range("2014-01-01", "2023-12-01", freq="MS"))
    seasonal = 10 * np.sin(2 * np.pi * dates.dt.month / 12)
    y = 100 + np.arange(len(dates)) * 0.4 + seasonal
    frame = pd.DataFrame(
        {
            "data": dates,
            "y": y,
            "target_strategy": "proforma_sum",
            "strategy_family": "proforma_sum",
            "alpha": 1.0,
            "beta": 1.0,
            "target_source": "observed_consolidated",
            "covid_shock": 0,
            "covid_recovery": 0,
            "month": dates.dt.month,
            "trend_index": np.arange(len(dates)),
        }
    )
    return {"strategies": {"proforma_sum": frame}}


def _model_selection(models: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rank, spec in enumerate(
        model_specs(models, stage="model_comparison", include_optional=True),
        start=1,
    ):
        model_id = spec["model_id"]
        if spec["family"] == "prophet":
            model_id = f"prophet_y{spec['params']['yearly_seasonality']}_none"
        rows.append(
            {
                "target_strategy": "proforma_sum",
                "model_id": model_id,
                "model_family": spec["family"],
                "selected": spec["family"] == "lightgbm",
                "rank": rank,
                "normal_mean_mase": 0.5 + rank / 100,
                "mean_mase": 0.6 + rank / 100,
            }
        )
    return pd.DataFrame(rows)


class _FakeForecaster:
    def fit(self, y, exog=None, config=None):
        self.y = pd.Series(y).astype(float).reset_index(drop=True)
        self.dates = pd.Series(config["dates"]).reset_index(drop=True)
        return self

    def _offset(self) -> pd.Series:
        return pd.Series(1.0 + 0.2 * np.sin(np.arange(len(self.y))), dtype=float)

    def fitted_values(self) -> pd.Series:
        return self.y - self._offset()

    def residuals(self) -> pd.Series:
        return self.y - self.fitted_values()

    def predict(self, horizon, future_exog=None, config=None):
        return np.repeat(float(self.y.iloc[-1]), horizon)

    def prediction_intervals(self, horizon, future_exog=None, levels=(80, 95), config=None):
        point = self.predict(horizon, future_exog, config)
        return pd.DataFrame(
            {
                "lo_80": point - 25.0,
                "hi_80": point + 25.0,
                "lo_95": point - 40.0,
                "hi_95": point + 40.0,
            }
        )


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
        return _FakeRun(
            self,
            {
                "run_name": run_name,
                "nested": nested,
                "params": {},
                "metrics": {},
            },
        )

    def log_param(self, key: str, value: object) -> None:
        self.current["params"][key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.current["metrics"][key] = value

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(Path(path).name)


def test_phase7_diagnostics_outputs_residual_interval_and_mlflow_artifacts(
    monkeypatch,
    tmp_path,
):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(
        "pads_forecasting.pipelines.diagnostics.nodes._make_model",
        lambda *args, **kwargs: _FakeForecaster(),
    )
    monkeypatch.setattr(
        "pads_forecasting.pipelines.diagnostics.nodes._selected_rows",
        lambda model_selection, n=3: model_selection,
    )

    models = _reduced_all_lane_models()
    diagnostics, coverage, interval_predictions = run_residual_diagnostics(
        target_strategies=_target_strategies(),
        model_selection=_model_selection(models),
        folds_metadata=_fold_metadata(),
        project={"seed": 42},
        validation={"season_length": 12},
        models=models,
        outputs={"figures_dir": str(tmp_path / "figures")},
    )

    assert set(diagnostics["model_family"]) == {
        "seasonal_naive",
        "ets",
        "sarimax",
        "prophet",
        "ridge",
        "elasticnet",
        "lightgbm",
        "bvar",
    }
    assert diagnostics["status"].eq("ok").all()
    assert set(DIAGNOSTIC_METRIC_COLUMNS).issubset(diagnostics.columns)
    assert set(INTERVAL_COVERAGE_COLUMNS).issubset(coverage.columns)
    assert set(INTERVAL_PREDICTION_COLUMNS).issubset(interval_predictions.columns)
    assert len(coverage) == len(diagnostics) * 3
    assert len(interval_predictions) == len(diagnostics) * 3 * 12
    assert coverage["status"].eq("ok").all()
    assert interval_predictions["status"].eq("ok").all()
    assert interval_predictions[["lo_80", "hi_80", "lo_95", "hi_95"]].notna().all().all()
    assert diagnostics["interval_fold_count"].eq(3).all()
    assert diagnostics["interval_observation_count"].eq(36).all()
    prophet = diagnostics[diagnostics["model_family"].eq("prophet")].iloc[0]
    assert prophet["diagnostic_spec_resolution"] == "compatible_model_family"
    assert prophet["diagnostic_spec_id"].startswith("prophet_y3_additive")

    figures = tmp_path / "figures"
    assert (figures / "residual_acf.png").exists()
    assert (figures / "residual_time.png").exists()
    assert (figures / "residual_histogram.png").exists()
    assert (figures / "interval_coverage_proxy.png").exists()

    assert "residual_diagnostics.parquet" in fake_mlflow.artifacts
    assert "interval_coverage_proxy.parquet" in fake_mlflow.artifacts
    assert "interval_validation_predictions.parquet" in fake_mlflow.artifacts
    assert "residual_acf.png" in fake_mlflow.artifacts
    assert "residual_histogram.png" in fake_mlflow.artifacts
    assert len(fake_mlflow.runs) == len(diagnostics)
    first_run = fake_mlflow.runs[0]
    assert first_run["nested"] is True
    assert "ljung_box_p_lag_12" in first_run["metrics"]
    assert "interval_coverage_80" in first_run["metrics"]
