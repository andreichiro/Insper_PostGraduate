"""Native fold-local COVID intervention adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovidAdjustmentResult:
    """Result of a fold-local COVID target adjustment."""

    adjusted_y: pd.Series
    covid_effect: pd.Series
    coefficients: dict[str, float]
    estimator: str
    status: str
    feature_columns: list[str]
    train_rows: int

    def metadata(self) -> dict[str, Any]:
        """Return flat metadata for fold/model artifacts."""

        effect = pd.Series(self.covid_effect, dtype=float)
        out: dict[str, Any] = {
            "covid_adjustment_estimator": self.estimator,
            "covid_adjustment_status": self.status,
            "covid_adjustment_train_rows": self.train_rows,
            "covid_adjustment_feature_columns": ",".join(self.feature_columns),
            "covid_adjustment_effect_mean": float(effect.mean()) if len(effect) else np.nan,
            "covid_adjustment_effect_abs_mean": float(effect.abs().mean())
            if len(effect)
            else np.nan,
            "covid_adjustment_effect_min": float(effect.min()) if len(effect) else np.nan,
            "covid_adjustment_effect_max": float(effect.max()) if len(effect) else np.nan,
        }
        for column in self.feature_columns:
            out[f"covid_beta_{column}"] = float(self.coefficients.get(column, 0.0))
        return out


def covid_feature_columns(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve declared COVID dummy columns that exist in the frame."""

    configured = (config or {}).get(
        "dummies",
        ["covid_shock", "covid_recovery", "covid_aftershock_2021"],
    )
    return [column for column in configured if column in frame.columns]


def zero_future_covid_exog(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Build future exogenous frame with COVID intervention values forced to zero."""

    return pd.DataFrame({column: np.zeros(len(frame), dtype=float) for column in columns})


def adjust_training_target(
    train: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    y_col: str = "y",
) -> CovidAdjustmentResult:
    """Estimate and remove COVID intervention effects using training data only.

    Primary estimator is statsmodels UnobservedComponents with COVID dummies as
    intervention regressors. A small sklearn fallback is used only when the state
    space fit is numerically unavailable for a short fold.
    """

    config = config or {}
    y = pd.to_numeric(train[y_col], errors="coerce").astype(float).reset_index(drop=True)
    columns = covid_feature_columns(train, config)
    if not columns or not any(
        pd.to_numeric(train[col], errors="coerce").fillna(0).sum() for col in columns
    ):
        zero_effect = pd.Series(np.zeros(len(y), dtype=float))
        return CovidAdjustmentResult(
            adjusted_y=y.copy(),
            covid_effect=zero_effect,
            coefficients={column: 0.0 for column in columns},
            estimator="none_no_training_intervention",
            status="skipped_no_active_covid_dummies",
            feature_columns=columns,
            train_rows=len(y),
        )

    exog = (
        train[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
        .reset_index(drop=True)
    )
    try:
        return _adjust_with_unobserved_components(y, exog, config)
    except Exception as exc:
        return _adjust_with_sklearn_fallback(y, exog, config, fallback_reason=str(exc))


def _adjust_with_unobserved_components(
    y: pd.Series,
    exog: pd.DataFrame,
    config: dict[str, Any],
) -> CovidAdjustmentResult:
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    level = str(config.get("level", "local linear trend")).replace("_", " ")
    seasonal = int(config.get("seasonal", 12))
    maxiter = int(config.get("maxiter", 120))
    model = UnobservedComponents(
        y,
        level=level,
        seasonal=seasonal,
        exog=exog,
        mle_regression=True,
    )
    result = model.fit(disp=False, maxiter=maxiter)
    params = pd.Series(result.params, index=result.param_names, dtype=float)
    coefficients = _extract_regression_coefficients(params, exog.columns)
    effect = _effect_from_coefficients(exog, coefficients)
    adjusted = y - effect
    return CovidAdjustmentResult(
        adjusted_y=adjusted.reset_index(drop=True),
        covid_effect=effect.reset_index(drop=True),
        coefficients=coefficients,
        estimator="statsmodels_unobserved_components",
        status="ok",
        feature_columns=list(exog.columns),
        train_rows=len(y),
    )


def _extract_regression_coefficients(
    params: pd.Series,
    columns: pd.Index,
) -> dict[str, float]:
    coefficients = {}
    for column in columns:
        candidates = [
            f"beta.{column}",
            f"beta_{column}",
            f"exog.{column}",
            str(column),
        ]
        value = 0.0
        for name in candidates:
            if name in params.index and np.isfinite(params.loc[name]):
                value = float(params.loc[name])
                break
        coefficients[str(column)] = value
    return coefficients


def _adjust_with_sklearn_fallback(
    y: pd.Series,
    exog: pd.DataFrame,
    config: dict[str, Any],
    *,
    fallback_reason: str,
) -> CovidAdjustmentResult:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import HuberRegressor, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

    dates = pd.date_range("2000-01-01", periods=len(y), freq="MS")
    fourier_order = int(config.get("fallback_fourier_order", 2))
    dp = DeterministicProcess(
        index=dates,
        constant=True,
        order=1,
        additional_terms=[CalendarFourier(freq="A", order=fourier_order)],
        drop=True,
    )
    deterministic = dp.in_sample().reset_index(drop=True)
    design = pd.concat([deterministic, exog.reset_index(drop=True)], axis=1)
    covid_columns = list(exog.columns)
    try:
        model = make_pipeline(
            ColumnTransformer(
                [("scale", StandardScaler(), list(design.columns))],
                remainder="drop",
                verbose_feature_names_out=False,
            ),
            HuberRegressor(alpha=float(config.get("fallback_huber_alpha", 0.01))),
        )
        model.fit(design, y)
        estimator = model.named_steps["huberregressor"]
        transformer = model.named_steps["columntransformer"]
        scaled = transformer.transform(design)
        feature_names = list(transformer.get_feature_names_out())
        coef = pd.Series(estimator.coef_, index=feature_names)
        scaled_df = pd.DataFrame(scaled, columns=feature_names)
        coefficients = {column: float(coef.get(column, 0.0)) for column in covid_columns}
        effect = pd.Series(
            sum(scaled_df[column] * coefficients[column] for column in covid_columns),
            dtype=float,
        )
        status = f"fallback_huber_after_ucm_failure: {fallback_reason[:120]}"
        estimator_name = "sklearn_huber_with_statsmodels_deterministic_process"
    except Exception:
        model = RidgeCV(alphas=np.asarray(config.get("fallback_ridge_alphas", [0.1, 1.0, 10.0])))
        model.fit(design, y)
        coef = pd.Series(model.coef_, index=design.columns)
        coefficients = {column: float(coef.get(column, 0.0)) for column in covid_columns}
        effect = _effect_from_coefficients(exog, coefficients)
        status = f"fallback_ridge_after_ucm_failure: {fallback_reason[:120]}"
        estimator_name = "sklearn_ridgecv_with_statsmodels_deterministic_process"

    adjusted = y - effect
    return CovidAdjustmentResult(
        adjusted_y=adjusted.reset_index(drop=True),
        covid_effect=effect.reset_index(drop=True),
        coefficients=coefficients,
        estimator=estimator_name,
        status=status,
        feature_columns=covid_columns,
        train_rows=len(y),
    )


def _effect_from_coefficients(
    exog: pd.DataFrame,
    coefficients: dict[str, float],
) -> pd.Series:
    effect = pd.Series(np.zeros(len(exog), dtype=float))
    for column, coefficient in coefficients.items():
        if column in exog:
            effect = effect + pd.to_numeric(exog[column], errors="coerce").fillna(0.0) * float(
                coefficient
            )
    return effect.astype(float).reset_index(drop=True)
