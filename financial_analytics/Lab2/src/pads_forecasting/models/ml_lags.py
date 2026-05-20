"""Lag-based ML forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from pads_forecasting.intervals import residual_bootstrap_intervals
from pads_forecasting.models.baseline import SeasonalNaiveForecaster

COVID_COLUMNS = ["covid_shock", "covid_recovery", "covid_aftershock_2021"]


class RecursiveLagForecaster:
    """Recursive lag forecaster for Ridge and LightGBM."""

    def __init__(
        self,
        *,
        model_type: str,
        lags: list[int],
        rolling_windows: list[int] | None = None,
        model_params: dict | None = None,
        seed: int = 42,
        season_length: int = 12,
    ) -> None:
        self.model_type = model_type
        self.lags = sorted(lags)
        self.rolling_windows = rolling_windows or []
        self.model_params = model_params or {}
        self.seed = seed
        self.season_length = season_length
        self.y_train: pd.Series | None = None
        self.model = None
        self.mlforecast = None
        self.fallback: SeasonalNaiveForecaster | None = None
        self.feature_columns: list[str] = []
        self.fitted_pred: pd.Series | None = None
        self.fitted_engine = "unfit"

    def _make_model(self):
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                params = {
                    "objective": "regression",
                    "random_state": self.seed,
                    "verbosity": -1,
                    **self.model_params,
                }
                return LGBMRegressor(**params)
            except Exception:
                return Ridge(alpha=10.0)
        if self.model_type == "elasticnet":
            return ElasticNet(
                alpha=float(self.model_params.get("alpha", 1.0)),
                l1_ratio=float(self.model_params.get("l1_ratio", 0.5)),
                random_state=self.seed,
                max_iter=10_000,
            )
        return Ridge(alpha=float(self.model_params.get("alpha", 1.0)))

    def _default_dates(self, y: pd.Series) -> pd.Series:
        return pd.Series(pd.date_range("2000-01-01", periods=len(y), freq="MS"))

    def _future_dates_from_training(self, horizon: int) -> pd.Series:
        if not hasattr(self, "train_dates") or self.train_dates is None:
            return pd.Series(pd.date_range("2000-01-01", periods=horizon, freq="MS"))
        start = pd.Timestamp(self.train_dates.iloc[-1]) + pd.offsets.MonthBegin(1)
        return pd.Series(pd.date_range(start, periods=horizon, freq="MS"))

    def _features_for_position(
        self,
        history: list[float],
        date: pd.Timestamp,
        exog_row: pd.Series | None,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        for lag in self.lags:
            features[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
        for window in self.rolling_windows:
            values = history[-window:] if len(history) >= window else history
            features[f"rolling_mean_{window}"] = float(np.mean(values)) if values else np.nan
            features[f"rolling_std_{window}"] = (
                float(np.std(values, ddof=0)) if len(values) > 1 else 0.0
            )
        month = int(date.month)
        features["month_sin"] = float(np.sin(2 * np.pi * month / 12))
        features["month_cos"] = float(np.cos(2 * np.pi * month / 12))
        features["trend_index"] = float(len(history))
        if exog_row is not None:
            for col in COVID_COLUMNS:
                features[col] = float(exog_row.get(col, 0.0))
        else:
            for col in COVID_COLUMNS:
                features[col] = 0.0
        return features

    def _training_matrix(
        self,
        y: pd.Series,
        dates: pd.Series,
        exog: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.Series, list[int]]:
        history: list[float] = []
        rows = []
        targets = []
        positions = []
        for idx, value in enumerate(y.astype(float).to_list()):
            date = pd.Timestamp(dates.iloc[idx])
            exog_row = exog.iloc[idx] if exog is not None and len(exog) > idx else None
            if idx >= max(self.lags):
                rows.append(self._features_for_position(history, date, exog_row))
                targets.append(float(value))
                positions.append(idx)
            history.append(float(value))
        X = pd.DataFrame(rows)
        y_target = pd.Series(targets)
        return X, y_target, positions

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> RecursiveLagForecaster:
        dates = None
        if config is not None and not isinstance(config, dict):
            dates = exog
            exog = config
            config = {}
        elif config is not None:
            dates = config.get("dates")
        elif exog is not None and not isinstance(exog, pd.DataFrame):
            dates = exog
            exog = None
        self.y_train = pd.Series(y).astype(float).reset_index(drop=True)
        self.train_dates = (
            pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
            if dates is not None
            else self._default_dates(self.y_train)
        )
        if len(self.y_train) <= max(self.lags) + 3:
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
            self.fitted_engine = "seasonal_naive_fallback"
            return self
        if self.model_type == "lightgbm" and self._fit_with_mlforecast(exog):
            return self
        X, y_target, positions = self._training_matrix(self.y_train, self.train_dates, exog)
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        y_target = y_target.loc[X.index]
        if len(X) < 6:
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
            self.fitted_engine = "seasonal_naive_fallback"
            return self
        self.feature_columns = X.columns.to_list()
        self.model = self._make_model()
        self.model.fit(X, y_target)
        fitted = pd.Series(np.nan, index=range(len(self.y_train)), dtype=float)
        fitted_positions = [positions[i] for i in X.index]
        fitted.loc[fitted_positions] = self.model.predict(X)
        self.fitted_pred = fitted
        self.fitted_engine = "sklearn_recursive"
        return self

    def _mlforecast_frame(self, exog: pd.DataFrame | None) -> pd.DataFrame:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before building MLForecast frame.")
        frame = pd.DataFrame(
            {
                "unique_id": "series",
                "ds": pd.to_datetime(self.train_dates),
                "y": self.y_train.astype(float).to_numpy(),
            }
        )
        return self._add_exogenous_features(frame, exog, start_index=0)

    def _future_exog_frame(
        self,
        future_dates: pd.Series,
        future_exog: pd.DataFrame | None,
    ) -> pd.DataFrame:
        frame = pd.DataFrame({"unique_id": "series", "ds": pd.to_datetime(future_dates)})
        start_index = len(self.y_train) if self.y_train is not None else 0
        return self._add_exogenous_features(frame, future_exog, start_index=start_index)

    def _add_exogenous_features(
        self,
        frame: pd.DataFrame,
        exog: pd.DataFrame | None,
        *,
        start_index: int,
    ) -> pd.DataFrame:
        out = frame.copy()
        dates = pd.to_datetime(out["ds"])
        out["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
        out["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
        out["trend_index"] = np.arange(start_index, start_index + len(out), dtype=float)
        for col in COVID_COLUMNS:
            out[col] = exog[col].to_numpy(dtype=float) if exog is not None and col in exog else 0.0
        return out

    def _fit_with_mlforecast(self, exog: pd.DataFrame | None) -> bool:
        try:
            from lightgbm import LGBMRegressor
            from mlforecast import MLForecast
            from mlforecast.lag_transforms import RollingMean, RollingStd

            lag_transforms = {
                1: [
                    transform(window_size=window, min_samples=1)
                    for window in self.rolling_windows
                    for transform in (RollingMean, RollingStd)
                ]
            }
            params = {
                "objective": "regression",
                "random_state": self.seed,
                "verbosity": -1,
                **self.model_params,
            }
            self.model = LGBMRegressor(**params)
            self.mlforecast = MLForecast(
                models={"lightgbm": self.model},
                freq="MS",
                lags=self.lags,
                lag_transforms=lag_transforms,
                date_features=[],
                num_threads=1,
            )
            train_frame = self._mlforecast_frame(exog)
            self.mlforecast.fit(train_frame, static_features=[], fitted=True)
            fitted = self.mlforecast.forecast_fitted_values()
            fitted_series = pd.Series(np.nan, index=range(len(self.y_train)), dtype=float)
            fitted_map = fitted.set_index("ds")["lightgbm"]
            for idx, date in enumerate(self.train_dates):
                if pd.Timestamp(date) in fitted_map.index:
                    fitted_series.iloc[idx] = float(fitted_map.loc[pd.Timestamp(date)])
            self.fitted_pred = fitted_series
            self.fitted_engine = "mlforecast_lightgbm"
            return True
        except Exception:
            self.mlforecast = None
            self.model = None
            return False

    def predict(
        self,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> np.ndarray:
        future_dates = None
        if config is not None and not isinstance(config, dict):
            future_dates = future_exog
            future_exog = config
            config = {}
        elif config is not None:
            future_dates = config.get("dates")
        elif future_exog is not None and not isinstance(future_exog, pd.DataFrame):
            future_dates = future_exog
            future_exog = None
        if self.fallback is not None:
            return self.fallback.predict(horizon)
        if future_dates is None:
            future_dates = self._future_dates_from_training(horizon)
        else:
            future_dates = pd.Series(pd.to_datetime(future_dates)).reset_index(drop=True)
        if self.mlforecast is not None:
            future = self._future_exog_frame(future_dates, future_exog)
            forecast = self.mlforecast.predict(horizon, X_df=future)
            return forecast["lightgbm"].to_numpy(dtype=float)
        if self.y_train is None or self.model is None:
            raise RuntimeError("Model must be fit before predict.")
        history = self.y_train.astype(float).to_list()
        preds = []
        for step in range(horizon):
            exog_row = (
                future_exog.iloc[step]
                if future_exog is not None and len(future_exog) > step
                else None
            )
            features = pd.DataFrame(
                [
                    self._features_for_position(
                        history, pd.Timestamp(future_dates.iloc[step]), exog_row
                    )
                ]
            )[self.feature_columns]
            pred = float(self.model.predict(features)[0])
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)

    def fitted_values(self) -> pd.Series:
        if self.fallback is not None:
            return self.fallback.fitted_values()
        if self.fitted_pred is None:
            raise RuntimeError("Model must be fit before fitted_values.")
        return self.fitted_pred

    def residuals(self) -> pd.Series:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before residuals.")
        return self.y_train - self.fitted_values().reset_index(drop=True)

    def prediction_intervals(
        self,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
        levels: tuple[int, ...] = (80, 95),
        config: dict | None = None,
    ) -> pd.DataFrame:
        future_dates = None
        if isinstance(levels, pd.DataFrame):
            future_dates = future_exog
            future_exog = levels
            levels = (80, 95)
        elif config is not None and not isinstance(config, dict):
            future_dates = future_exog
            future_exog = config
            config = {}
        elif config is not None:
            future_dates = config.get("dates")
        elif future_exog is not None and not isinstance(future_exog, pd.DataFrame):
            future_dates = future_exog
            future_exog = None
        forecast = self.predict(horizon, future_exog, {"dates": future_dates})
        return residual_bootstrap_intervals(forecast, self.residuals(), levels=levels)

    def serializable_params(self) -> dict[str, object]:
        return {
            "model_family": self.model_type,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "model_params": self.model_params,
            "seed": self.seed,
            "season_length": self.season_length,
            "fitted_engine": self.fitted_engine,
            "feature_columns": self.feature_columns,
        }

    def mlflow_log_payload(self) -> dict[str, dict[str, object]]:
        return {
            "params": self.serializable_params(),
            "metrics": {},
            "artifacts": {},
        }
