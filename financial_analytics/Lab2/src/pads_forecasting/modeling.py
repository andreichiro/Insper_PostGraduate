"""Shared model fitting/evaluation utilities for Kedro nodes."""

from __future__ import annotations

import json
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from pads_forecasting.covid_adjustment import adjust_training_target, zero_future_covid_exog
from pads_forecasting.leakage import LeakageError
from pads_forecasting.metrics import (
    mae,
    mase,
    mase_denominator,
    score_forecast,
    seasonal_naive_forecast,
)
from pads_forecasting.models.baseline import SeasonalNaiveForecaster
from pads_forecasting.models.bvar import LatentComponentBVARForecaster
from pads_forecasting.models.ets import ETSForecaster
from pads_forecasting.models.ml_lags import RecursiveLagForecaster
from pads_forecasting.models.prophet_model import ProphetForecaster
from pads_forecasting.models.sarima import SARIMAXForecaster
from pads_forecasting.pipelines.validation.nodes import make_fold_slices

EXOG_COLUMNS = ["covid_shock", "covid_recovery", "covid_aftershock_2021"]
PRIMARY_COVID_MODES = {"none", "adjusted_target"}
NATIVE_COVID_MODES = {"covid", "regressors", "features", "exog", "native_dummies"}


def _available_exog_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in EXOG_COLUMNS if column in frame.columns]


def _model_id_with_covid_mode(base_id: str, covid_mode: str, *, default_mode: str) -> str:
    """Keep historical ids for the default mode and suffix primary alternatives."""

    return base_id if covid_mode == default_mode else f"{base_id}_{covid_mode}"


def _configured_covid_modes(
    family_config: dict[str, Any],
    *,
    default: list[str],
) -> list[str]:
    modes = family_config.get("covid_modes", default)
    return list(dict.fromkeys(["none", "adjusted_target", *(str(mode) for mode in modes)]))


def _native_covid_enabled(covid_mode: str) -> bool:
    return covid_mode in NATIVE_COVID_MODES


def _serialize_model_param(value: Any) -> str | int | float | bool:
    """Keep scalar params searchable and encode structured params for CSV/MLflow."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, default=str)


def future_month_frame(
    start: str | pd.Timestamp, horizon: int, covid_value: int = 0
) -> pd.DataFrame:
    """Create future dates and known covariates."""

    dates = pd.date_range(start=pd.Timestamp(start), periods=horizon, freq="MS")
    return pd.DataFrame(
        {
            "data": dates,
            "covid_shock": covid_value,
            "covid_recovery": covid_value,
            "covid_aftershock_2021": covid_value,
        }
    )


def model_specs(
    models_params: dict[str, Any], *, stage: str, include_optional: bool = False
) -> list[dict[str, Any]]:
    """Expand model specifications from YAML grids."""

    specs: list[dict[str, Any]] = []
    models = models_params.get("models", models_params)

    if models["seasonal_naive"].get("enabled", False):
        for covid_mode in _configured_covid_modes(
            models["seasonal_naive"],
            default=["none", "adjusted_target"],
        ):
            specs.append(
                {
                    "model_id": _model_id_with_covid_mode(
                        "seasonal_naive",
                        covid_mode,
                        default_mode="none",
                    ),
                    "family": "seasonal_naive",
                    "params": {"season_length": models["seasonal_naive"]["season_length"]},
                    "covid_mode": covid_mode,
                    "complexity": "simple",
                }
            )

    if models["ets"].get("enabled", False):
        grid = models["ets"]["grid"]
        use_boxcox_values = [False] if stage == "old_data_gate" else grid["use_boxcox"]
        for trend, seasonal, damped_trend, use_boxcox, covid_mode in product(
            grid.get("trend", ["add"]),
            grid.get("seasonal", ["add"]),
            grid["damped_trend"],
            use_boxcox_values,
            _configured_covid_modes(models["ets"], default=["none", "adjusted_target"]),
        ):
            model_id = f"ets_{'damped' if damped_trend else trend}"
            if seasonal != "add":
                model_id += f"_{seasonal}"
            if use_boxcox:
                model_id += "_boxcox"
            model_id = _model_id_with_covid_mode(model_id, covid_mode, default_mode="none")
            specs.append(
                {
                    "model_id": model_id,
                    "family": "ets",
                    "params": {
                        "trend": trend,
                        "seasonal": seasonal,
                        "damped_trend": bool(damped_trend),
                        "use_boxcox": bool(use_boxcox),
                    },
                    "covid_mode": covid_mode,
                    "complexity": "simple",
                }
            )

    if stage == "old_data_gate":
        if models["ridge"].get("enabled", False):
            for covid_mode in _configured_covid_modes(
                models["ridge"],
                default=["none", "adjusted_target", "features"],
            ):
                specs.append(
                    {
                        "model_id": _model_id_with_covid_mode(
                            "ridge_lags",
                            covid_mode,
                            default_mode="features",
                        ),
                        "family": "ridge",
                        "params": {"alpha": 10.0, "lags": models["ridge"]["lags"]},
                        "covid_mode": covid_mode,
                        "complexity": "moderate",
                    }
                )
        if models.get("elasticnet", {}).get("enabled", False):
            for covid_mode in _configured_covid_modes(
                models["elasticnet"],
                default=["none", "adjusted_target", "features"],
            ):
                specs.append(
                    {
                        "model_id": _model_id_with_covid_mode(
                            "elasticnet_lags",
                            covid_mode,
                            default_mode="features",
                        ),
                        "family": "elasticnet",
                        "params": {
                            "alpha": models["elasticnet"]["alpha_grid"][0],
                            "l1_ratio": models["elasticnet"]["l1_ratio_grid"][0],
                            "lags": models["elasticnet"]["lags"],
                        },
                        "covid_mode": covid_mode,
                        "complexity": "moderate",
                    }
                )
        if models["lightgbm"].get("enabled", False):
            for covid_mode in _configured_covid_modes(
                models["lightgbm"],
                default=["none", "adjusted_target", "features"],
            ):
                specs.append(
                    {
                        "model_id": _model_id_with_covid_mode(
                            "lightgbm_shallow",
                            covid_mode,
                            default_mode="features",
                        ),
                        "family": "lightgbm",
                        "params": {
                            "lags": models["lightgbm"]["grid"]["lags"][0],
                            "rolling_windows": models["lightgbm"]["grid"]["rolling_windows"][0],
                            "max_depth": 2,
                            "num_leaves": 4,
                            "n_estimators": 50,
                            "learning_rate": 0.05,
                            "min_data_in_leaf": 12,
                            "lambda_l1": 0.1,
                            "lambda_l2": 0.5,
                        },
                        "covid_mode": covid_mode,
                        "complexity": "complex",
                    }
                )
        return specs

    if models["sarimax"].get("enabled", False):
        grid = models["sarimax"]["grid"]
        orders = grid.get("orders")
        if orders is None:
            orders = list(product(grid["p"], grid["d"], grid["q"]))
        seasonal_orders = grid.get("seasonal_orders")
        if seasonal_orders is None:
            seasonal_orders = list(product(grid["P"], grid["D"], grid["Q"], grid["m"]))
        for order, seasonal_order, trend, covid_mode in product(
            orders,
            seasonal_orders,
            grid["trend"],
            _configured_covid_modes(
                models["sarimax"],
                default=["none", "adjusted_target", "covid"],
            ),
        ):
            specs.append(
                {
                    "model_id": f"sarimax_{tuple(order)}_{tuple(seasonal_order)}_{trend}_{covid_mode}",
                    "family": "sarimax",
                    "params": {
                        "order": tuple(order),
                        "seasonal_order": tuple(seasonal_order),
                        "trend": trend,
                        "use_exog": _native_covid_enabled(covid_mode),
                    },
                    "covid_mode": covid_mode,
                    "complexity": "simple",
                }
            )

    if models["prophet"].get("enabled", False) and (
        include_optional or not models["prophet"].get("optional", False)
    ):
        grid = models["prophet"]["grid"]
        covid_modes = _configured_covid_modes(
            models["prophet"],
            default=[
                "none",
                "adjusted_target",
                *(
                    ["regressors"]
                    if any(bool(value) for value in grid.get("covid_regressors", []))
                    else []
                ),
            ],
        )
        for (
            yearly,
            seasonality_mode,
            changepoint_prior_scale,
            seasonality_prior_scale,
            covid_mode,
        ) in product(
            grid["yearly_seasonality"],
            grid["seasonality_mode"],
            grid["changepoint_prior_scale"],
            grid["seasonality_prior_scale"],
            covid_modes,
        ):
            covid_regressors = _native_covid_enabled(covid_mode)
            suffix = "covid" if covid_mode == "regressors" else covid_mode
            specs.append(
                {
                    "model_id": (
                        f"prophet_y{yearly}_{seasonality_mode}"
                        f"_cp{changepoint_prior_scale}_sp{seasonality_prior_scale}"
                        f"_{suffix}"
                    ),
                    "family": "prophet",
                    "params": {
                        "yearly_seasonality": yearly,
                        "seasonality_mode": seasonality_mode,
                        "changepoint_prior_scale": changepoint_prior_scale,
                        "seasonality_prior_scale": seasonality_prior_scale,
                        "use_covid_regressors": covid_regressors,
                    },
                    "covid_mode": covid_mode,
                    "complexity": "complex",
                }
            )

    if models["ridge"].get("enabled", False):
        for alpha, covid_mode in product(
            models["ridge"]["alpha_grid"],
            _configured_covid_modes(
                models["ridge"],
                default=["none", "adjusted_target", "features"],
            ),
        ):
            base_id = f"ridge_lags_alpha_{alpha}"
            specs.append(
                {
                    "model_id": _model_id_with_covid_mode(
                        base_id,
                        covid_mode,
                        default_mode="features",
                    ),
                    "family": "ridge",
                    "params": {"alpha": alpha, "lags": models["ridge"]["lags"]},
                    "covid_mode": covid_mode,
                    "complexity": "moderate",
                }
            )

    if models.get("elasticnet", {}).get("enabled", False):
        for alpha, l1_ratio, covid_mode in product(
            models["elasticnet"]["alpha_grid"],
            models["elasticnet"]["l1_ratio_grid"],
            _configured_covid_modes(
                models["elasticnet"],
                default=["none", "adjusted_target", "features"],
            ),
        ):
            base_id = f"elasticnet_lags_alpha_{alpha}_l1_{l1_ratio}"
            specs.append(
                {
                    "model_id": _model_id_with_covid_mode(
                        base_id,
                        covid_mode,
                        default_mode="features",
                    ),
                    "family": "elasticnet",
                    "params": {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "lags": models["elasticnet"]["lags"],
                    },
                    "covid_mode": covid_mode,
                    "complexity": "moderate",
                }
            )

    if models["lightgbm"].get("enabled", False):
        grid = models["lightgbm"]["grid"]
        for (
            max_depth,
            num_leaves,
            n_estimators,
            learning_rate,
            min_data_in_leaf,
            lambda_l1,
            lambda_l2,
            covid_mode,
        ) in product(
            grid["max_depth"],
            grid["num_leaves"],
            grid["n_estimators"],
            grid["learning_rate"],
            grid["min_data_in_leaf"],
            grid["lambda_l1"],
            grid["lambda_l2"],
            _configured_covid_modes(
                models["lightgbm"],
                default=["none", "adjusted_target", "features"],
            ),
        ):
            base_id = (
                f"lightgbm_d{max_depth}_l{num_leaves}_n{n_estimators}"
                f"_lr{learning_rate}_leaf{min_data_in_leaf}"
                f"_l1{lambda_l1}_l2{lambda_l2}"
            )
            specs.append(
                {
                    "model_id": _model_id_with_covid_mode(
                        base_id,
                        covid_mode,
                        default_mode="features",
                    ),
                    "family": "lightgbm",
                    "params": {
                        "lags": grid["lags"][0],
                        "rolling_windows": grid["rolling_windows"][0],
                        "max_depth": max_depth,
                        "num_leaves": num_leaves,
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "min_data_in_leaf": min_data_in_leaf,
                        "lambda_l1": lambda_l1,
                        "lambda_l2": lambda_l2,
                        "forecast_strategy": grid.get("forecast_strategy", ["recursive"])[0],
                    },
                    "covid_mode": covid_mode,
                    "complexity": "complex",
                }
            )

    if models["bvar"].get("enabled", False) and (
        include_optional or not models["bvar"].get("optional", False)
    ):
        grid = models["bvar"]["grid"]
        cv_draw_cap = models["bvar"].get("cv_draw_cap")
        cv_tune_cap = models["bvar"].get("cv_tune_cap")
        covid_modes = _configured_covid_modes(
            models["bvar"],
            default=[
                "none",
                "adjusted_target",
                *(["exog"] if any(bool(value) for value in grid.get("covid_exog", [])) else []),
            ],
        )
        for (
            lags,
            minnesota_lambda,
            cross_lag_shrinkage,
            covid_mode,
            draws,
            tune,
        ) in product(
            grid["lags"],
            grid["minnesota_lambda"],
            grid["cross_lag_shrinkage"],
            covid_modes,
            grid["draws"],
            grid["tune"],
        ):
            covid_exog = _native_covid_enabled(covid_mode)
            suffix = "covid" if covid_mode == "exog" else covid_mode
            specs.append(
                {
                    "model_id": (
                        f"bvar_l{lags}_lambda{minnesota_lambda}"
                        f"_cross{cross_lag_shrinkage}_{suffix}"
                        f"_draws{draws}_tune{tune}"
                    ),
                    "family": "bvar",
                    "params": {
                        "lags": lags,
                        "minnesota_lambda": minnesota_lambda,
                        "cross_lag_shrinkage": cross_lag_shrinkage,
                        "covid_exog": covid_exog,
                        "draws": min(int(draws), int(cv_draw_cap)) if cv_draw_cap else draws,
                        "tune": min(int(tune), int(cv_tune_cap)) if cv_tune_cap else tune,
                        "requested_draws": draws,
                        "requested_tune": tune,
                    },
                    "covid_mode": covid_mode,
                    "complexity": "complex",
                }
            )

    return specs


def _make_model(spec: dict[str, Any], *, season_length: int, seed: int):
    family = spec["family"]
    params = spec["params"]
    if family == "seasonal_naive":
        return SeasonalNaiveForecaster(season_length=season_length)
    if family == "ets":
        return ETSForecaster(**params, season_length=season_length)
    if family == "sarimax":
        return SARIMAXForecaster(**params, season_length=season_length)
    if family == "ridge":
        return RecursiveLagForecaster(
            model_type="ridge",
            lags=params["lags"],
            rolling_windows=[],
            model_params={"alpha": params["alpha"]},
            seed=seed,
            season_length=season_length,
        )
    if family == "elasticnet":
        return RecursiveLagForecaster(
            model_type="elasticnet",
            lags=params["lags"],
            rolling_windows=[],
            model_params={"alpha": params["alpha"], "l1_ratio": params["l1_ratio"]},
            seed=seed,
            season_length=season_length,
        )
    if family == "lightgbm":
        lgbm_params = {
            key: value
            for key, value in params.items()
            if key not in {"lags", "rolling_windows", "forecast_strategy"}
        }
        return RecursiveLagForecaster(
            model_type="lightgbm",
            lags=params["lags"],
            rolling_windows=params.get("rolling_windows", []),
            model_params=lgbm_params,
            seed=seed,
            season_length=season_length,
        )
    if family == "prophet":
        return ProphetForecaster(**params)
    if family == "bvar":
        return LatentComponentBVARForecaster(**params, seed=seed, season_length=season_length)
    raise ValueError(f"Unknown model family: {family}")


def fit_predict(
    spec: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    season_length: int,
    seed: int,
    common_mase_denominator: float | None = None,
    covid_adjustment_config: dict[str, Any] | None = None,
    precomputed_covid_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one model and forecast one validation fold."""

    covid_mode = str(spec.get("covid_mode", "none"))
    train_for_model = train_df.copy().reset_index(drop=True)
    covid_metadata: dict[str, Any] = {}
    if covid_mode == "adjusted_target" and precomputed_covid_metadata is not None:
        covid_metadata = precomputed_covid_metadata
    elif covid_mode == "adjusted_target":
        adjustment = adjust_training_target(
            train_for_model,
            covid_adjustment_config,
            y_col="y",
        )
        train_for_model["y"] = adjustment.adjusted_y.to_numpy(dtype=float)
        covid_metadata = adjustment.metadata()
    else:
        covid_metadata = {
            "covid_adjustment_estimator": "not_applicable",
            "covid_adjustment_status": "not_applicable",
            "covid_adjustment_train_rows": len(train_for_model),
            "covid_adjustment_feature_columns": "",
            "covid_adjustment_effect_mean": 0.0,
            "covid_adjustment_effect_abs_mean": 0.0,
            "covid_adjustment_effect_min": 0.0,
            "covid_adjustment_effect_max": 0.0,
        }

    train_y = train_for_model["y"].astype(float).reset_index(drop=True)
    valid_y = valid_df["y"].astype(float).reset_index(drop=True)
    train_dates = train_for_model["data"].reset_index(drop=True)
    exog_columns = _available_exog_columns(train_for_model)
    train_exog_actual = (
        train_for_model[exog_columns].reset_index(drop=True) if exog_columns else None
    )
    valid_exog_zero = (
        zero_future_covid_exog(valid_df, exog_columns).reset_index(drop=True)
        if exog_columns
        else None
    )
    uses_native_covid = _native_covid_enabled(covid_mode)
    train_exog = train_exog_actual if uses_native_covid else None
    valid_exog = valid_exog_zero
    model = _make_model(spec, season_length=season_length, seed=seed)

    model.fit(
        train_y,
        train_exog,
        {
            "dates": train_dates,
            "train_frame": train_df.reset_index(drop=True),
        },
    )
    yhat = model.predict(len(valid_y), valid_exog)

    fitted = model.fitted_values()
    residuals = model.residuals()
    train_mask = fitted.notna()
    train_mae = mae(train_y.loc[train_mask], fitted.loc[train_mask]) if train_mask.any() else np.nan
    residual_values = pd.Series(residuals).dropna().astype(float)
    sn_pred = seasonal_naive_forecast(train_y, len(valid_y), season_length)
    sn_mae = mae(valid_y, sn_pred)
    scores = score_forecast(
        valid_y,
        yhat,
        train_y,
        season_length=season_length,
        seasonal_naive_mae=sn_mae,
        common_mase_denominator=common_mase_denominator,
    )
    scores["train_mae"] = float(train_mae) if np.isfinite(train_mae) else np.nan
    scores["validation_mae"] = scores["mae"]
    scores["train_valid_mae_gap"] = (
        scores["validation_mae"] - scores["train_mae"] if np.isfinite(train_mae) else np.nan
    )
    scores["train_valid_mae_ratio"] = (
        scores["validation_mae"] / scores["train_mae"]
        if np.isfinite(train_mae) and train_mae > 0
        else np.nan
    )
    scores["train_residual_mean"] = (
        float(residual_values.mean()) if not residual_values.empty else np.nan
    )
    scores["train_residual_abs_mean"] = (
        float(residual_values.abs().mean()) if not residual_values.empty else np.nan
    )
    scores["train_residual_std"] = (
        float(residual_values.std(ddof=0)) if len(residual_values) > 1 else np.nan
    )
    scores.update(covid_metadata)
    return {
        "model": model,
        "yhat": yhat,
        "fitted": fitted,
        "residuals": residuals,
        "scores": scores,
    }


def _score_alpha_candidate_on_folds(
    strategy: pd.DataFrame,
    folds: list[dict[str, Any]],
    season_length: int,
    *,
    common_denominators: dict[str, float] | None = None,
) -> float:
    """Score one alpha candidate using only supplied inner folds.

    Alpha changes the pre-merger target scale, so fixed-target MASE is preferred
    whenever common denominators are available. This keeps alpha selection
    comparable across target reconstructions.
    """

    values = []
    for fold in folds:
        train, valid = make_fold_slices(strategy, fold)
        yhat = seasonal_naive_forecast(train["y"], len(valid), season_length)
        denominator = (
            common_denominators.get(fold["fold_name"]) if common_denominators is not None else None
        )
        if denominator is None or not np.isfinite(denominator) or denominator <= 0:
            values.append(mase(valid["y"], yhat, train["y"], season_length))
        else:
            values.append(mae(valid["y"], yhat) / denominator)
    return float(np.mean(values)) if values else np.nan


def _select_alpha_for_outer_fold(
    alpha_candidates: dict[float, pd.DataFrame],
    folds_metadata: pd.DataFrame,
    outer_fold: dict[str, Any],
    season_length: int,
    *,
    common_denominators: dict[str, float] | None = None,
    shrinkage_lambda: float = 0.0,
) -> tuple[float, pd.DataFrame | None, dict[str, Any]]:
    """Choose calibrated alpha with inner folds strictly before the outer validation."""

    if not alpha_candidates:
        return (
            np.nan,
            None,
            {
                "alpha_selection_method": "not_applicable",
                "alpha_inner_fold_count": 0,
                "alpha_inner_score": np.nan,
            },
        )

    valid_start = pd.Timestamp(outer_fold["valid_start"])
    inner_folds = [
        fold
        for fold in folds_metadata.to_dict("records")
        if pd.Timestamp(fold["valid_end"]) < valid_start
    ]
    if not inner_folds:
        alpha = 1.0 if 1.0 in alpha_candidates else sorted(alpha_candidates)[0]
        return (
            float(alpha),
            alpha_candidates[float(alpha)],
            {
                "alpha_selection_method": "fallback_alpha_1_no_prior_inner_folds",
                "alpha_inner_fold_count": 0,
                "alpha_inner_score": np.nan,
            },
        )

    scored = []
    for alpha, candidate in alpha_candidates.items():
        raw_score = _score_alpha_candidate_on_folds(
            candidate,
            inner_folds,
            season_length,
            common_denominators=common_denominators,
        )
        shrinkage_penalty = float(shrinkage_lambda) * (float(alpha) - 1.0) ** 2
        scored.append(
            {
                "alpha": float(alpha),
                "raw_score": raw_score,
                "shrinkage_penalty": shrinkage_penalty,
                "score": raw_score + shrinkage_penalty,
            }
        )
    scored_df = pd.DataFrame(scored).replace([np.inf, -np.inf], np.nan).dropna()
    if scored_df.empty:
        alpha = 1.0 if 1.0 in alpha_candidates else sorted(alpha_candidates)[0]
        return (
            float(alpha),
            alpha_candidates[float(alpha)],
            {
                "alpha_selection_method": "fallback_alpha_1_no_finite_inner_score",
                "alpha_inner_fold_count": len(inner_folds),
                "alpha_inner_score": np.nan,
            },
        )
    scored_df["distance_to_one"] = (scored_df["alpha"] - 1.0).abs()
    winner = scored_df.sort_values(["score", "distance_to_one", "alpha"]).iloc[0]
    alpha = float(winner["alpha"])
    return (
        alpha,
        alpha_candidates[alpha],
        {
            "alpha_selection_method": "nested_prior_fold_fixed_target_mase_with_shrinkage",
            "alpha_inner_fold_count": len(inner_folds),
            "alpha_inner_score": float(winner["score"]),
            "alpha_inner_raw_score": float(winner["raw_score"]),
            "alpha_shrinkage_penalty": float(winner["shrinkage_penalty"]),
            "alpha_selection_metric": "seasonal_naive_common_mase_proxy",
        },
    )


def _common_mase_denominators_by_fold(
    strategies: dict[str, pd.DataFrame],
    folds_metadata: pd.DataFrame,
    *,
    reference_strategy: str,
    season_length: int,
) -> dict[str, float]:
    """Compute one common MASE denominator per fold from one target process.

    The preferred reference is the observed post-merger consolidated process:
    the `post_only` strategy contains only observed C_t rows from the acquisition
    onward. This avoids giving any target strategy a MASE advantage from changing
    the pre-merger reconstruction scale.
    """

    observed_aliases = {
        "observed_post_merger",
        "post_merger_observed",
        "observed_consolidated",
        "post_only_observed",
    }
    if reference_strategy in observed_aliases:
        reference_strategy = "post_only"
    if reference_strategy not in strategies:
        reference_strategy = "post_only" if "post_only" in strategies else next(iter(strategies))
    reference = strategies[reference_strategy].copy()
    reference["data"] = pd.to_datetime(reference["data"])
    denominators = {}
    for fold in folds_metadata.to_dict("records"):
        train, _valid = make_fold_slices(reference, fold)
        denominators[fold["fold_name"]] = mase_denominator(train["y"], season_length)
    return denominators


def _covid_adjustment_cache_key(
    strategy_name: str,
    fold: dict[str, Any],
    train: pd.DataFrame,
) -> tuple[str, str, str, str, str, str, str, int]:
    """Identify one fold-local COVID adjustment training window."""

    alpha = str(float(train["alpha"].iloc[0])) if "alpha" in train and len(train) else "nan"
    beta = str(float(train["beta"].iloc[0])) if "beta" in train and len(train) else "nan"
    return (
        str(strategy_name),
        str(fold["fold_name"]),
        alpha,
        beta,
        pd.Timestamp(train["data"].min()).strftime("%Y-%m-%d"),
        pd.Timestamp(train["data"].max()).strftime("%Y-%m-%d"),
        str(len(train)),
        int(train[_available_exog_columns(train)].sum().sum())
        if _available_exog_columns(train)
        else 0,
    )


def _adjusted_training_from_cache(
    *,
    cache: dict[tuple[str, str, str, str, str, str, str, int], tuple[pd.DataFrame, dict[str, Any]]],
    strategy_name: str,
    fold: dict[str, Any],
    train: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the COVID intervention model once per strategy/fold training window."""

    key = _covid_adjustment_cache_key(strategy_name, fold, train)
    if key not in cache:
        adjustment = adjust_training_target(train, config, y_col="y")
        adjusted = train.copy().reset_index(drop=True)
        adjusted["y"] = adjustment.adjusted_y.to_numpy(dtype=float)
        cache[key] = (adjusted, adjustment.metadata())
    return cache[key]


def _horizon_metric_rows(
    base: dict[str, Any],
    valid: pd.DataFrame,
    yhat: np.ndarray,
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Create forecast-horizon metric rows for h=1..H."""

    y_true = valid["y"].astype(float).to_numpy()
    y_pred = np.asarray(yhat, dtype=float)
    local_denom = float(scores["mase_denominator"])
    common_denom = float(scores["common_mase_denominator"])
    rows = []
    for idx, (actual, pred) in enumerate(zip(y_true, y_pred, strict=True), start=1):
        error = float(pred - actual)
        abs_error = abs(error)
        rows.append(
            {
                **base,
                "horizon_index": idx,
                "data": pd.Timestamp(valid["data"].iloc[idx - 1]).strftime("%Y-%m-%d"),
                "y_true": float(actual),
                "yhat": float(pred),
                "error": error,
                "abs_error": float(abs_error),
                "squared_error": float(error**2),
                "local_mase": float(abs_error / local_denom) if local_denom else np.nan,
                "common_mase": float(abs_error / common_denom) if common_denom else np.nan,
                "local_mase_denominator": local_denom,
                "common_mase_denominator": common_denom,
            }
        )
    return rows


def evaluate_cv(
    *,
    stage: str,
    target_strategies: dict[str, Any],
    strategy_names: list[str],
    specs: list[dict[str, Any]],
    folds_metadata: pd.DataFrame,
    validation_params: dict[str, Any],
    project_params: dict[str, Any],
    collect_horizon_metrics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a set of target strategies, model specs, and folds."""

    strategies = target_strategies["strategies"]
    alpha_candidates = target_strategies.get("alpha_candidates", {})
    season_length = validation_params["season_length"]
    seed = int(project_params["seed"])
    common_reference = validation_params.get(
        "common_mase_reference_strategy",
        "observed_post_merger",
    )
    common_denominators = _common_mase_denominators_by_fold(
        strategies,
        folds_metadata,
        reference_strategy=common_reference,
        season_length=season_length,
    )
    alpha_selection_config = validation_params.get("robust_alpha", {})
    alpha_shrinkage_lambda = float(alpha_selection_config.get("shrinkage_lambda", 0.0))
    covid_adjustment_config = validation_params.get("covid_adjustment", {})
    rows = []
    horizon_rows = []
    covid_adjustment_cache: dict[
        tuple[str, str, str, str, str, str, str, int],
        tuple[pd.DataFrame, dict[str, Any]],
    ] = {}
    for strategy_name in strategy_names:
        if strategy_name not in strategies:
            continue
        base_strategy_df = strategies[strategy_name].copy()
        base_strategy_df["data"] = pd.to_datetime(base_strategy_df["data"])
        for spec in specs:
            for fold in folds_metadata.to_dict("records"):
                train = pd.DataFrame()
                valid = pd.DataFrame()
                strategy_df = base_strategy_df
                alpha_info = {
                    "alpha_selection_method": "fixed_strategy_alpha",
                    "alpha_inner_fold_count": np.nan,
                    "alpha_inner_score": np.nan,
                }
                if strategy_name == "calibrated_alpha" and alpha_candidates:
                    _alpha, candidate, alpha_info = _select_alpha_for_outer_fold(
                        alpha_candidates,
                        folds_metadata,
                        fold,
                        season_length,
                        common_denominators=common_denominators,
                        shrinkage_lambda=alpha_shrinkage_lambda,
                    )
                    if candidate is not None:
                        strategy_df = candidate.copy()
                        strategy_df["data"] = pd.to_datetime(strategy_df["data"])
                base = {
                    "stage": stage,
                    "target_strategy": strategy_name,
                    "alpha": float(train["alpha"].iloc[0])
                    if not train.empty and "alpha" in train
                    else np.nan,
                    "beta": float(train["beta"].iloc[0])
                    if not train.empty and "beta" in train
                    else np.nan,
                    "model_id": spec["model_id"],
                    "model_family": spec["family"],
                    "model_params": json.dumps(
                        spec.get("params", {}),
                        sort_keys=True,
                        default=str,
                    ),
                    "covid_mode": spec.get("covid_mode", "none"),
                    "complexity": spec.get("complexity", "unknown"),
                    "fold_name": fold["fold_name"],
                    "fold_role": fold["fold_role"],
                    "train_end": fold["train_end"],
                    "valid_start": fold["valid_start"],
                    "valid_end": fold["valid_end"],
                    "horizon": fold.get("horizon", fold.get("expected_horizon", np.nan)),
                    "common_mase_reference_strategy": common_reference,
                    **alpha_info,
                }
                for key, value in spec.get("params", {}).items():
                    base[f"model_param_{key}"] = _serialize_model_param(value)
                try:
                    train, valid = make_fold_slices(strategy_df, fold)
                    base["alpha"] = float(train["alpha"].iloc[0]) if "alpha" in train else np.nan
                    base["beta"] = float(train["beta"].iloc[0]) if "beta" in train else np.nan
                    precomputed_covid_metadata = None
                    if str(spec.get("covid_mode", "none")) == "adjusted_target":
                        train, precomputed_covid_metadata = _adjusted_training_from_cache(
                            cache=covid_adjustment_cache,
                            strategy_name=strategy_name,
                            fold=fold,
                            train=train,
                            config=covid_adjustment_config,
                        )
                    result = fit_predict(
                        spec,
                        train,
                        valid,
                        season_length=season_length,
                        seed=seed,
                        common_mase_denominator=common_denominators.get(
                            fold["fold_name"],
                            np.nan,
                        ),
                        covid_adjustment_config=covid_adjustment_config,
                        precomputed_covid_metadata=precomputed_covid_metadata,
                    )
                    rows.append({**base, "status": "ok", **result["scores"]})
                    if collect_horizon_metrics:
                        horizon_rows.extend(
                            _horizon_metric_rows(
                                {**base, "status": "ok"},
                                valid,
                                result["yhat"],
                                result["scores"],
                            )
                        )
                except LeakageError:
                    raise
                except Exception as exc:
                    rows.append({**base, "status": f"failed: {exc}"})
    fold_results = pd.DataFrame(rows)
    horizon_results = pd.DataFrame(horizon_rows)
    if collect_horizon_metrics:
        return fold_results, horizon_results
    return fold_results
