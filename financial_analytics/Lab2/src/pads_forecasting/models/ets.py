"""ETS model wrapper."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pads_forecasting.intervals import residual_bootstrap_intervals
from pads_forecasting.models.baseline import SeasonalNaiveForecaster


class ETSForecaster:
    """Statsmodels Holt-Winters ETS wrapper with safe fallback."""

    def __init__(
        self,
        *,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool = False,
        use_boxcox: bool = False,
        season_length: int = 12,
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.use_boxcox = use_boxcox
        self.season_length = season_length
        self.y_train: pd.Series | None = None
        self.result = None
        self.fallback: SeasonalNaiveForecaster | None = None

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> ETSForecaster:
        self.y_train = pd.Series(y).astype(float).reset_index(drop=True)
        if len(self.y_train) < 2 * self.season_length:
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
            return self
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    self.y_train,
                    trend=self.trend,
                    damped_trend=self.damped_trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.season_length,
                    initialization_method="estimated",
                    use_boxcox=self.use_boxcox,
                )
                self.result = model.fit(optimized=True)
        except Exception:
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        if self.fallback is not None:
            return self.fallback.predict(horizon)
        if self.result is None:
            raise RuntimeError("Model must be fit before predict.")
        return np.asarray(self.result.forecast(horizon), dtype=float)

    def fitted_values(self) -> pd.Series:
        if self.fallback is not None:
            return self.fallback.fitted_values()
        if self.result is None:
            raise RuntimeError("Model must be fit before fitted_values.")
        return pd.Series(self.result.fittedvalues)

    def residuals(self) -> pd.Series:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before residuals.")
        fitted = self.fitted_values().reset_index(drop=True)
        return self.y_train - fitted

    def prediction_intervals(
        self,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
        levels: tuple[int, ...] = (80, 95),
    ) -> pd.DataFrame:
        forecast = self.predict(horizon, future_exog)
        return residual_bootstrap_intervals(forecast, self.residuals(), levels=levels)

    def serializable_params(self) -> dict[str, object]:
        return {
            "model_family": "ets",
            "trend": self.trend,
            "seasonal": self.seasonal,
            "damped_trend": self.damped_trend,
            "use_boxcox": self.use_boxcox,
            "season_length": self.season_length,
            "fitted_engine": "seasonal_naive_fallback"
            if self.fallback is not None
            else "statsmodels",
        }

    def mlflow_log_payload(self) -> dict[str, dict[str, object]]:
        return {
            "params": self.serializable_params(),
            "metrics": {},
            "artifacts": {},
        }
