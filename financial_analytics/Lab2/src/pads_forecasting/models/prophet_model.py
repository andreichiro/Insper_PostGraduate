"""Prophet wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pads_forecasting.intervals import residual_bootstrap_intervals

COVID_COLUMNS = ["covid_shock", "covid_recovery", "covid_aftershock_2021"]


class ProphetForecaster:
    """Thin Prophet wrapper, optional because Prophet may be unavailable."""

    def __init__(
        self,
        *,
        yearly_seasonality: int = 5,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 5.0,
        use_covid_regressors: bool = False,
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.use_covid_regressors = use_covid_regressors
        self.model = None
        self.train_df: pd.DataFrame | None = None
        self.fitted_pred: pd.Series | None = None

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> ProphetForecaster:
        try:
            from prophet import Prophet
        except Exception as exc:
            raise RuntimeError(
                "Prophet is not installed; install optional dependency `prophet`."
            ) from exc

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
        if dates is None:
            dates = pd.Series(pd.date_range("2000-01-01", periods=len(y), freq="MS"))
        train = pd.DataFrame(
            {"ds": pd.to_datetime(dates), "y": pd.Series(y).astype(float).to_numpy()}
        )
        if self.use_covid_regressors and exog is not None:
            for column in COVID_COLUMNS:
                train[column] = (
                    exog[column].to_numpy(dtype=float) if column in exog else np.zeros(len(train))
                )
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=0.95,
        )
        if self.use_covid_regressors:
            for column in COVID_COLUMNS:
                self.model.add_regressor(column)
        # LBFGS is materially faster and more stable than Newton for the many
        # short rolling-origin fits used in this project.
        self.model.fit(train, algorithm="LBFGS")
        fitted = self.model.predict(train)
        self.train_df = train
        self.fitted_pred = pd.Series(fitted["yhat"].to_numpy())
        return self

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
        if self.model is None:
            raise RuntimeError("Model must be fit before predict.")
        if future_dates is None:
            future_dates = self._default_future_dates(horizon)
        future = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
        if self.use_covid_regressors:
            for column in COVID_COLUMNS:
                future[column] = (
                    future_exog[column].to_numpy(dtype=float)
                    if future_exog is not None and column in future_exog
                    else np.zeros(horizon)
                )
        forecast = self.model.predict(future)
        return forecast["yhat"].to_numpy(dtype=float)

    def prediction_frame(
        self,
        future_dates: pd.Series,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction_frame.")
        future = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
        if self.use_covid_regressors:
            for column in COVID_COLUMNS:
                future[column] = (
                    future_exog[column].to_numpy(dtype=float)
                    if future_exog is not None and column in future_exog
                    else np.zeros(len(future))
                )
        return self.model.predict(future)

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
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction_intervals.")
        if future_dates is None:
            future_dates = self._default_future_dates(horizon)

        out: dict[str, np.ndarray] = {}
        original_width = getattr(self.model, "interval_width", None)
        try:
            for level in levels:
                self.model.interval_width = level / 100
                frame = self.prediction_frame(future_dates, future_exog)
                out[f"lo_{level}"] = frame["yhat_lower"].to_numpy(dtype=float)
                out[f"hi_{level}"] = frame["yhat_upper"].to_numpy(dtype=float)
        except Exception:
            point = self.predict(horizon, future_exog, {"dates": future_dates})
            return residual_bootstrap_intervals(point, self.residuals(), levels=levels)
        finally:
            if original_width is not None:
                self.model.interval_width = original_width
        return pd.DataFrame(out)

    def _default_future_dates(self, horizon: int) -> pd.Series:
        if self.train_df is None:
            raise RuntimeError("Model must be fit before generating future dates.")
        start = pd.Timestamp(self.train_df["ds"].max()) + pd.offsets.MonthBegin(1)
        return pd.Series(pd.date_range(start, periods=horizon, freq="MS"))

    def fitted_values(self) -> pd.Series:
        if self.fitted_pred is None:
            raise RuntimeError("Model must be fit before fitted_values.")
        return self.fitted_pred

    def residuals(self) -> pd.Series:
        if self.train_df is None:
            raise RuntimeError("Model must be fit before residuals.")
        return pd.Series(self.train_df["y"].to_numpy()) - self.fitted_values().reset_index(
            drop=True
        )

    def serializable_params(self) -> dict[str, object]:
        return {
            "model_family": "prophet",
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "seasonality_mode": self.seasonality_mode,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "use_covid_regressors": self.use_covid_regressors,
            "fitted_engine": "prophet",
        }

    def mlflow_log_payload(self) -> dict[str, dict[str, object]]:
        return {
            "params": self.serializable_params(),
            "metrics": {},
            "artifacts": {},
        }
