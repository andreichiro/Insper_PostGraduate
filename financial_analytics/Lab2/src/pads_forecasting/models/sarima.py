"""SARIMA/SARIMAX wrapper."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pads_forecasting.intervals import residual_bootstrap_intervals
from pads_forecasting.models.baseline import SeasonalNaiveForecaster


class SARIMAXForecaster:
    """Statsmodels SARIMAX wrapper with fallback."""

    def __init__(
        self,
        *,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 12),
        trend: str = "n",
        season_length: int = 12,
        use_exog: bool = False,
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.season_length = season_length
        self.use_exog = use_exog
        self.y_train: pd.Series | None = None
        self.result = None
        self.fallback: SeasonalNaiveForecaster | None = None

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> SARIMAXForecaster:
        self.y_train = pd.Series(y).astype(float).reset_index(drop=True)
        if len(self.y_train) < 2 * self.season_length:
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
            return self
        exog_fit = exog if self.use_exog else None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    self.y_train,
                    exog=exog_fit,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.result = model.fit(disp=False, maxiter=80)
        except Exception:
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        if self.fallback is not None:
            return self.fallback.predict(horizon)
        if self.result is None:
            raise RuntimeError("Model must be fit before predict.")
        exog_future = future_exog if self.use_exog else None
        return np.asarray(
            self.result.get_forecast(steps=horizon, exog=exog_future).predicted_mean, dtype=float
        )

    def prediction_intervals(
        self,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
        levels: tuple[int, ...] = (80, 95),
    ) -> pd.DataFrame:
        if self.fallback is not None:
            forecast = self.fallback.predict(horizon)
            return residual_bootstrap_intervals(forecast, self.fallback.residuals(), levels=levels)
        if self.result is None:
            raise RuntimeError("Model must be fit before prediction_intervals.")
        exog_future = future_exog if self.use_exog else None
        forecast_result = self.result.get_forecast(steps=horizon, exog=exog_future)
        out: dict[str, np.ndarray] = {}
        for level in levels:
            alpha = 1 - level / 100
            frame = pd.DataFrame(forecast_result.conf_int(alpha=alpha)).reset_index(drop=True)
            out[f"lo_{level}"] = frame.iloc[:, 0].to_numpy(dtype=float)
            out[f"hi_{level}"] = frame.iloc[:, 1].to_numpy(dtype=float)
        return pd.DataFrame(out)

    def fitted_values(self) -> pd.Series:
        if self.fallback is not None:
            return self.fallback.fitted_values()
        if self.result is None:
            raise RuntimeError("Model must be fit before fitted_values.")
        return pd.Series(self.result.fittedvalues)

    def residuals(self) -> pd.Series:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before residuals.")
        return self.y_train - self.fitted_values().reset_index(drop=True)

    def serializable_params(self) -> dict[str, object]:
        return {
            "model_family": "sarimax",
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "season_length": self.season_length,
            "use_exog": self.use_exog,
            "fitted_engine": "seasonal_naive_fallback"
            if self.fallback is not None
            else "statsmodels",
            "aic": float(self.result.aic) if self.result is not None else np.nan,
            "bic": float(self.result.bic) if self.result is not None else np.nan,
        }

    def mlflow_log_payload(self) -> dict[str, dict[str, object]]:
        params = self.serializable_params()
        metrics = {
            key: params[key]
            for key in ["aic", "bic"]
            if isinstance(params.get(key), float) and np.isfinite(params[key])
        }
        return {
            "params": {key: value for key, value in params.items() if key not in metrics},
            "metrics": metrics,
            "artifacts": {},
        }
