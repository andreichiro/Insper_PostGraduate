"""PyMC/ArviZ Bayesian VAR challenger."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from pads_forecasting.models.baseline import SeasonalNaiveForecaster

COVID_COLUMNS = ["covid_shock", "covid_recovery", "covid_aftershock_2021"]


class LatentComponentBVARForecaster:
    """Component-aware Bayesian VAR wrapper with practical fallbacks."""

    def __init__(
        self,
        *,
        lags: int = 1,
        minnesota_lambda: float = 0.2,
        cross_lag_shrinkage: float = 0.1,
        covid_exog: bool = True,
        draws: int = 1000,
        tune: int = 1000,
        requested_draws: int | None = None,
        requested_tune: int | None = None,
        seed: int = 42,
        season_length: int = 12,
    ) -> None:
        self.lags = int(lags)
        self.minnesota_lambda = float(minnesota_lambda)
        self.cross_lag_shrinkage = float(cross_lag_shrinkage)
        self.covid_exog = bool(covid_exog)
        self.draws = int(draws)
        self.tune = int(tune)
        self.requested_draws = int(requested_draws) if requested_draws is not None else self.draws
        self.requested_tune = int(requested_tune) if requested_tune is not None else self.tune
        self.seed = seed
        self.season_length = season_length
        self.y_train: pd.Series | None = None
        self.idata = None
        self.fallback: SeasonalNaiveForecaster | None = None
        self.fitted_pred: pd.Series | None = None
        self.design = "unfit"
        self.mean_: float = 0.0
        self.std_: float = 1.0
        self.component_mean_: np.ndarray | None = None
        self.component_std_: np.ndarray | None = None
        self.component_history_: np.ndarray | None = None
        self.feature_columns_: list[str] = []

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        config: dict[str, Any] | None = None,
    ) -> LatentComponentBVARForecaster:
        self.y_train = pd.Series(y).astype(float).reset_index(drop=True)
        if len(self.y_train) < max(24, self.lags + 8):
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
            self.design = "seasonal_naive_fallback"
            return self

        component_matrix = self._component_matrix_from_config(config)
        if component_matrix is not None:
            try:
                self._fit_component_bvar(component_matrix, exog)
                return self
            except Exception:
                self.idata = None
                self.fitted_pred = None

        try:
            self._fit_univariate_bayesian_ar()
        except Exception:
            self.fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
            self.design = "seasonal_naive_fallback"
        return self

    def _component_matrix_from_config(self, config: dict[str, Any] | None) -> np.ndarray | None:
        if self.y_train is None or not config:
            return None
        frame = config.get("train_frame")
        if frame is None:
            return None
        frame = pd.DataFrame(frame).reset_index(drop=True)
        br_col = (
            "br_component_observed"
            if "br_component_observed" in frame
            else "br_publicado"
            if "br_publicado" in frame
            else None
        )
        acquired_col = (
            "acquired_component_observed"
            if "acquired_component_observed" in frame
            else "adquirida_separada"
            if "adquirida_separada" in frame
            else None
        )
        if br_col is None or acquired_col is None:
            return None

        br = pd.to_numeric(frame[br_col], errors="coerce").reset_index(drop=True)
        acquired = pd.to_numeric(frame[acquired_col], errors="coerce").reset_index(drop=True)
        target = self.y_train.reset_index(drop=True)
        observed_components = br.notna() & acquired.notna() & (br + acquired).gt(0)
        if observed_components.sum() < max(8, self.lags + 4):
            return None

        share = br[observed_components] / (br[observed_components] + acquired[observed_components])
        br_share = float(np.clip(share.median(), 0.05, 0.95))
        components = pd.DataFrame({"br": br, "acquired": acquired})
        missing = components.isna().any(axis=1)
        components.loc[missing, "br"] = target.loc[missing] * br_share
        components.loc[missing, "acquired"] = target.loc[missing] * (1.0 - br_share)
        if components.isna().any().any():
            return None
        return components.to_numpy(dtype=float)

    def _fit_component_bvar(self, components: np.ndarray, exog: pd.DataFrame | None) -> None:
        import pymc as pm

        self.design = "component_bvar"
        self.component_mean_ = components.mean(axis=0)
        self.component_std_ = components.std(axis=0)
        self.component_std_[self.component_std_ == 0] = 1.0
        scaled = (components - self.component_mean_) / self.component_std_
        self.component_history_ = scaled.copy()
        X, target, prior_scale = self._component_design_matrix(scaled, exog)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pm.Model():
                intercept = pm.Normal("intercept", mu=0.0, sigma=1.0, shape=2)
                beta = pm.Normal(
                    "beta",
                    mu=0.0,
                    sigma=prior_scale,
                    shape=(X.shape[1], 2),
                )
                sigma = pm.HalfNormal("sigma", sigma=1.0, shape=2)
                mu = intercept + pm.math.dot(X, beta)
                pm.Normal("obs", mu=mu, sigma=sigma, observed=target)
                self.idata = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=1,
                    cores=1,
                    random_seed=self.seed,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
        self._compute_component_fitted(X)

    def _component_design_matrix(
        self,
        scaled_components: np.ndarray,
        exog: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = []
        targets = []
        self.feature_columns_ = []
        prior_rows = []
        for lag in range(1, self.lags + 1):
            self.feature_columns_.extend([f"br_lag_{lag}", f"acquired_lag_{lag}"])
            own = self.minnesota_lambda / lag
            cross = self.cross_lag_shrinkage / lag
            prior_rows.extend([[own, cross], [cross, own]])
        if self.covid_exog:
            self.feature_columns_.extend(COVID_COLUMNS)
            prior_rows.extend([[0.5, 0.5] for _ in COVID_COLUMNS])
        exog = self._aligned_exog(exog, len(scaled_components))
        for idx in range(self.lags, len(scaled_components)):
            row = []
            for lag in range(1, self.lags + 1):
                row.extend(scaled_components[idx - lag].tolist())
            if self.covid_exog:
                row.extend(exog.iloc[idx][COVID_COLUMNS].to_list())
            rows.append(row)
            targets.append(scaled_components[idx])
        return (
            np.asarray(rows, dtype=float),
            np.asarray(targets, dtype=float),
            np.asarray(prior_rows, dtype=float),
        )

    def _aligned_exog(self, exog: pd.DataFrame | None, n: int) -> pd.DataFrame:
        if exog is None:
            return pd.DataFrame({column: np.zeros(n) for column in COVID_COLUMNS})
        out = pd.DataFrame(index=range(n))
        for col in COVID_COLUMNS:
            out[col] = (
                pd.to_numeric(exog[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                if col in exog
                else np.zeros(n)
            )
        return out

    def _fit_univariate_bayesian_ar(self) -> None:
        import pymc as pm

        self.design = "bayesian_ar_fallback"
        self.mean_ = float(self.y_train.mean())
        self.std_ = float(self.y_train.std(ddof=0) or 1.0)
        scaled = ((self.y_train - self.mean_) / self.std_).to_numpy()
        X = np.column_stack([scaled[self.lags - lag : -lag] for lag in range(1, self.lags + 1)])
        target = scaled[self.lags :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pm.Model():
                intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
                coefs = pm.Normal("coefs", mu=0.0, sigma=self.minnesota_lambda, shape=self.lags)
                sigma = pm.HalfNormal("sigma", sigma=1.0)
                mu = intercept + pm.math.dot(X, coefs)
                pm.Normal("obs", mu=mu, sigma=sigma, observed=target)
                self.idata = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=1,
                    cores=1,
                    random_seed=self.seed,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
        self._compute_univariate_fitted(X)

    def _component_posterior_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        posterior = self.idata.posterior
        intercept = posterior["intercept"].values.reshape(-1, 2)
        beta = posterior["beta"].values.reshape(-1, len(self.feature_columns_), 2)
        sigma = posterior["sigma"].values.reshape(-1, 2)
        return intercept, beta, sigma

    def _univariate_posterior_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        posterior = self.idata.posterior
        intercept = posterior["intercept"].values.reshape(-1)
        coefs = posterior["coefs"].values.reshape(-1, self.lags)
        sigma = posterior["sigma"].values.reshape(-1)
        return intercept, coefs, sigma

    def _compute_component_fitted(self, X: np.ndarray) -> None:
        if self.idata is None or self.y_train is None or self.component_mean_ is None:
            return
        intercept, beta, _ = self._component_posterior_arrays()
        fitted_scaled = intercept.mean(axis=0) + X @ beta.mean(axis=0)
        fitted_components = fitted_scaled * self.component_std_ + self.component_mean_
        fitted = fitted_components.sum(axis=1)
        self.fitted_pred = pd.Series(np.nan, index=range(len(self.y_train)), dtype=float)
        self.fitted_pred.iloc[self.lags :] = fitted

    def _compute_univariate_fitted(self, X: np.ndarray) -> None:
        if self.idata is None or self.y_train is None:
            return
        intercept, coefs, _ = self._univariate_posterior_arrays()
        fitted_scaled = float(intercept.mean()) + X @ coefs.mean(axis=0)
        fitted = fitted_scaled * self.std_ + self.mean_
        self.fitted_pred = pd.Series(np.nan, index=range(len(self.y_train)), dtype=float)
        self.fitted_pred.iloc[self.lags :] = fitted

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        if self.fallback is not None:
            return self.fallback.predict(horizon)
        if self.y_train is None or self.idata is None:
            raise RuntimeError("Model must be fit before predict.")
        if self.design == "component_bvar":
            return np.median(
                self._component_paths(horizon, future_exog, include_noise=False), axis=0
            )
        return np.median(self._univariate_paths(horizon, include_noise=False), axis=0)

    def _component_paths(
        self,
        horizon: int,
        future_exog: pd.DataFrame | None,
        *,
        include_noise: bool,
    ) -> np.ndarray:
        if self.component_history_ is None or self.component_mean_ is None:
            raise RuntimeError("Component BVAR must be fit before prediction.")
        intercept, beta, sigma = self._component_posterior_arrays()
        rng = np.random.default_rng(self.seed)
        sample_idx = rng.choice(len(intercept), size=min(400, len(intercept)), replace=True)
        exog = self._aligned_exog(future_exog, horizon)
        paths = []
        for sample in sample_idx:
            history = self.component_history_.tolist()
            path = []
            for step in range(horizon):
                features = self._component_future_features(history, exog.iloc[step])
                pred_scaled = intercept[sample] + np.asarray(features) @ beta[sample]
                if include_noise:
                    pred_scaled = rng.normal(pred_scaled, sigma[sample])
                history.append(pred_scaled.tolist())
                component_pred = pred_scaled * self.component_std_ + self.component_mean_
                path.append(float(component_pred.sum()))
            paths.append(path)
        return np.asarray(paths, dtype=float)

    def _component_future_features(
        self, history: list[list[float]], exog_row: pd.Series
    ) -> list[float]:
        features = []
        for lag in range(1, self.lags + 1):
            features.extend(history[-lag])
        if self.covid_exog:
            features.extend([float(exog_row.get(column, 0.0)) for column in COVID_COLUMNS])
        return features

    def _univariate_paths(self, horizon: int, *, include_noise: bool) -> np.ndarray:
        intercept, coefs, sigma = self._univariate_posterior_arrays()
        rng = np.random.default_rng(self.seed)
        sample_idx = rng.choice(len(intercept), size=min(400, len(intercept)), replace=True)
        paths = []
        for sample in sample_idx:
            history = ((self.y_train - self.mean_) / self.std_).to_list()
            path = []
            for _step in range(horizon):
                lag_vec = np.asarray([history[-lag] for lag in range(1, self.lags + 1)])
                pred_scaled = float(intercept[sample] + lag_vec @ coefs[sample])
                if include_noise:
                    pred_scaled = float(rng.normal(pred_scaled, sigma[sample]))
                history.append(pred_scaled)
                path.append(pred_scaled * self.std_ + self.mean_)
            paths.append(path)
        return np.asarray(paths, dtype=float)

    def posterior_interval_frame(
        self,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.fallback is not None:
            point = self.fallback.predict(horizon)
            intervals = self.fallback.prediction_intervals(horizon)
            return pd.concat([pd.DataFrame({"previsao": point}), intervals], axis=1)
        if self.y_train is None or self.idata is None:
            raise RuntimeError("Model must be fit before posterior intervals.")
        paths = (
            self._component_paths(horizon, future_exog, include_noise=True)
            if self.design == "component_bvar"
            else self._univariate_paths(horizon, include_noise=True)
        )
        return pd.DataFrame(
            {
                "previsao": np.median(paths, axis=0),
                "lo_80": np.percentile(paths, 10, axis=0),
                "hi_80": np.percentile(paths, 90, axis=0),
                "lo_95": np.percentile(paths, 2.5, axis=0),
                "hi_95": np.percentile(paths, 97.5, axis=0),
            }
        )

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
    ) -> pd.DataFrame:
        frame = self.posterior_interval_frame(horizon, future_exog)
        columns = [f"{side}_{level}" for level in levels for side in ("lo", "hi")]
        missing = [col for col in columns if col not in frame]
        if missing:
            fallback = SeasonalNaiveForecaster(self.season_length).fit(self.y_train)
            return fallback.prediction_intervals(horizon, levels=levels)
        return frame[columns].copy()

    def arviz_summary(self) -> pd.DataFrame:
        if self.idata is None:
            return pd.DataFrame(
                [{"status": "seasonal_naive_fallback" if self.fallback is not None else "unfit"}]
            )
        try:
            import arviz as az

            return az.summary(self.idata).reset_index(names="parameter")
        except Exception as exc:
            return pd.DataFrame([{"status": f"arviz_summary_failed: {exc}"}])

    def serializable_params(self) -> dict[str, object]:
        return {
            "model_family": "bvar",
            "design": self.design,
            "lags": self.lags,
            "minnesota_lambda": self.minnesota_lambda,
            "cross_lag_shrinkage": self.cross_lag_shrinkage,
            "covid_exog": self.covid_exog,
            "draws": self.draws,
            "tune": self.tune,
            "requested_draws": self.requested_draws,
            "requested_tune": self.requested_tune,
            "seed": self.seed,
            "season_length": self.season_length,
            "feature_columns": self.feature_columns_,
            "fitted_engine": "seasonal_naive_fallback"
            if self.fallback is not None
            else "pymc_arviz",
        }

    def mlflow_log_payload(self) -> dict[str, dict[str, object]]:
        metrics: dict[str, object] = {}
        summary = self.arviz_summary()
        if "r_hat" in summary:
            metrics["max_r_hat"] = float(pd.to_numeric(summary["r_hat"], errors="coerce").max())
        if "ess_bulk" in summary:
            metrics["min_ess_bulk"] = float(
                pd.to_numeric(summary["ess_bulk"], errors="coerce").min()
            )
        return {
            "params": self.serializable_params(),
            "metrics": metrics,
            "artifacts": {"arviz_summary": summary},
        }
