"""Baseline forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pads_forecasting.intervals import residual_bootstrap_intervals
from pads_forecasting.metrics import seasonal_naive_forecast


class SeasonalNaiveForecaster:
    """Monthly seasonal naive baseline."""

    def __init__(self, season_length: int = 12) -> None:
        self.season_length = season_length
        self.y_train: pd.Series | None = None

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> SeasonalNaiveForecaster:
        self.y_train = pd.Series(y).astype(float).reset_index(drop=True)
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before predict.")
        return seasonal_naive_forecast(self.y_train, horizon, self.season_length)

    def fitted_values(self) -> pd.Series:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before fitted_values.")
        vals = [np.nan] * len(self.y_train)
        if len(self.y_train) > self.season_length:
            vals[self.season_length :] = self.y_train.iloc[: -self.season_length].to_list()
        return pd.Series(vals)

    def residuals(self) -> pd.Series:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before residuals.")
        return self.y_train - self.fitted_values()

    def prediction_intervals(
        self,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
        levels: tuple[int, ...] = (80, 95),
    ) -> pd.DataFrame:
        forecast = self.predict(horizon)
        return residual_bootstrap_intervals(forecast, self.residuals(), levels=levels)

    def serializable_params(self) -> dict[str, object]:
        return {
            "model_family": "seasonal_naive",
            "season_length": self.season_length,
        }

    def mlflow_log_payload(self) -> dict[str, dict[str, object]]:
        return {
            "params": self.serializable_params(),
            "metrics": {},
            "artifacts": {},
        }
