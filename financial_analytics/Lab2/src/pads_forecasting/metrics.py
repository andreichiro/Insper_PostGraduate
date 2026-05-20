"""Forecast metrics with local and common-denominator MASE support."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean(errors**2)))


def bias(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def mase_denominator(y_train: pd.Series, season_length: int = 12) -> float:
    """Compute the seasonal MASE denominator from training data only."""

    y = pd.Series(y_train).astype(float).reset_index(drop=True)
    if len(y) <= season_length:
        diffs = y.diff().dropna().abs()
    else:
        diffs = y.iloc[season_length:].to_numpy() - y.iloc[:-season_length].to_numpy()
        diffs = pd.Series(np.abs(diffs))
    denom = float(diffs.mean()) if len(diffs) else float("nan")
    if not np.isfinite(denom) or denom <= 1e-12:
        return 1.0
    return denom


def mase(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: pd.Series,
    season_length: int = 12,
) -> float:
    return mae(y_true, y_pred) / mase_denominator(y_train, season_length)


def seasonal_naive_forecast(
    y_train: pd.Series, horizon: int, season_length: int = 12
) -> np.ndarray:
    """Seasonal naive forecast with robust fallback for short series."""

    y = pd.Series(y_train).astype(float).reset_index(drop=True)
    if len(y) >= season_length:
        seasonal_values = y.iloc[-season_length:].to_numpy()
        reps = int(np.ceil(horizon / season_length))
        return np.tile(seasonal_values, reps)[:horizon].astype(float)
    return np.repeat(float(y.iloc[-1]), horizon)


def utilsforecast_point_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """Compute point metrics through utilsforecast, with local fallback."""

    try:
        from utilsforecast.losses import bias as uf_bias
        from utilsforecast.losses import mae as uf_mae
        from utilsforecast.losses import rmse as uf_rmse

        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        frame = pd.DataFrame(
            {
                "unique_id": "series",
                "ds": pd.date_range("2000-01-01", periods=len(y_true_arr), freq="MS"),
                "cutoff": pd.Timestamp("1999-12-01"),
                "y": y_true_arr,
                "model": y_pred_arr,
            }
        )
        return {
            "mae": float(uf_mae(frame, models=["model"])["model"].iloc[0]),
            "rmse": float(uf_rmse(frame, models=["model"])["model"].iloc[0]),
            "bias": float(uf_bias(frame, models=["model"])["model"].iloc[0]),
        }
    except Exception:
        return {
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "bias": bias(y_true, y_pred),
        }


def score_forecast(
    y_true: pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: pd.Series,
    *,
    season_length: int,
    seasonal_naive_mae: float | None = None,
    common_mase_denominator: float | None = None,
) -> dict[str, float]:
    """Compute fold metrics."""

    y_pred_arr = np.asarray(y_pred, dtype=float)
    point_metrics = utilsforecast_point_metrics(y_true, y_pred_arr)
    fold_mae = point_metrics["mae"]
    fold_mase_denominator = mase_denominator(y_train, season_length)
    fold_common_denominator = (
        float(common_mase_denominator)
        if common_mase_denominator is not None and np.isfinite(common_mase_denominator)
        else fold_mase_denominator
    )
    if fold_common_denominator <= 1e-12:
        fold_common_denominator = 1.0
    if seasonal_naive_mae is None:
        sn_pred = seasonal_naive_forecast(y_train, len(y_true), season_length)
        seasonal_naive_mae = mae(y_true, sn_pred)
    rel = fold_mae / seasonal_naive_mae if seasonal_naive_mae and seasonal_naive_mae > 0 else np.nan
    return {
        "mae": fold_mae,
        "rmse": point_metrics["rmse"],
        "mase": fold_mae / fold_mase_denominator,
        "mase_denominator": fold_mase_denominator,
        "common_mase": fold_mae / fold_common_denominator,
        "common_mase_denominator": fold_common_denominator,
        "bias": point_metrics["bias"],
        "relative_mae_vs_seasonal_naive": float(rel),
    }


def summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level CV metrics."""

    optional_group_cols = [
        col
        for col in ["model_family", "covid_mode", "complexity", "model_params"]
        if col in df.columns
    ]
    optional_group_cols.extend(sorted(col for col in df.columns if col.startswith("model_param_")))
    group_cols = ["stage", "target_strategy", "model_id", *optional_group_cols]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = dict(zip(group_cols, key_values, strict=True))
        mae_values = grp["mae"].astype(float)
        rmse_values = grp["rmse"].astype(float)
        mase_values = grp["mase"].astype(float)
        common_mase_values = (
            grp["common_mase"].astype(float) if "common_mase" in grp else mase_values
        )
        rel_values = grp["relative_mae_vs_seasonal_naive"].astype(float)
        mean_mae = float(mae_values.mean())
        mean_rmse = float(rmse_values.mean())
        mean_mase = float(mase_values.mean())
        mean_common_mase = float(common_mase_values.mean())
        mean_rel = float(rel_values.mean())
        std_mae = float(mae_values.std(ddof=0))
        std_rmse = float(rmse_values.std(ddof=0))
        std_mase = float(mase_values.std(ddof=0))
        std_common_mase = float(common_mase_values.std(ddof=0))
        std_rel = float(rel_values.std(ddof=0))
        normal = grp[grp["fold_role"].eq("normal")]
        normal_common = normal["common_mase"] if "common_mase" in normal else normal["mase"]
        rows.append(
            {
                **key_map,
                "mean_mae": mean_mae,
                "mean_rmse": mean_rmse,
                "mean_mase": mean_mase,
                "normal_mean_mase": float(normal["mase"].mean()) if len(normal) else mean_mase,
                "mean_common_mase": mean_common_mase,
                "normal_mean_common_mase": float(normal_common.mean())
                if len(normal)
                else mean_common_mase,
                "std_mae": std_mae,
                "cv_mae": std_mae / mean_mae if mean_mae else np.nan,
                "max_mae": float(mae_values.max()),
                "std_rmse": std_rmse,
                "cv_rmse": std_rmse / mean_rmse if mean_rmse else np.nan,
                "max_rmse": float(rmse_values.max()),
                "std_mase": std_mase,
                "cv_mase": std_mase / mean_mase if mean_mase else np.nan,
                "max_mase": float(mase_values.max()),
                "std_common_mase": std_common_mase,
                "cv_common_mase": std_common_mase / mean_common_mase
                if mean_common_mase
                else np.nan,
                "max_common_mase": float(common_mase_values.max()),
                "mean_bias": float(grp["bias"].mean()),
                "mean_relative_mae_vs_seasonal_naive": mean_rel,
                "std_relative_mae_vs_seasonal_naive": std_rel,
                "cv_relative_mae_vs_seasonal_naive": std_rel / mean_rel if mean_rel else np.nan,
                "mean_train_valid_ratio": float(grp["train_valid_mae_ratio"].mean()),
                "folds": int(grp["fold_name"].nunique()),
            }
        )
        for residual_col in [
            "train_residual_mean",
            "train_residual_abs_mean",
            "train_residual_std",
        ]:
            if residual_col in grp:
                rows[-1][f"mean_{residual_col}"] = float(grp[residual_col].mean())
    sort_cols = ["normal_mean_common_mase", "mean_common_mase"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def summarize_horizon_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate month-ahead errors by forecast horizon."""

    if df.empty:
        return pd.DataFrame()
    optional_group_cols = [
        col
        for col in ["model_family", "covid_mode", "complexity", "model_params"]
        if col in df.columns
    ]
    optional_group_cols.extend(sorted(col for col in df.columns if col.startswith("model_param_")))
    group_cols = [
        "stage",
        "target_strategy",
        "model_id",
        *optional_group_cols,
        "horizon_index",
    ]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = dict(zip(group_cols, key_values, strict=True))
        abs_error = grp["abs_error"].astype(float)
        squared_error = grp["squared_error"].astype(float)
        local_mase = grp["local_mase"].astype(float)
        common_mase = grp["common_mase"].astype(float)
        rows.append(
            {
                **key_map,
                "horizon_mae": float(abs_error.mean()),
                "horizon_rmse": float(np.sqrt(squared_error.mean())),
                "horizon_local_mase": float(local_mase.mean()),
                "horizon_common_mase": float(common_mase.mean()),
                "horizon_bias": float(grp["error"].astype(float).mean()),
                "folds": int(grp["fold_name"].nunique()),
                "observations": int(len(grp)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["target_strategy", "model_id", "horizon_index"])
        .reset_index(drop=True)
    )


def bootstrap_mase_uncertainty(
    horizon_metrics: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate paired uncertainty around common-MASE gains vs SeasonalNaive."""

    if horizon_metrics.empty or "common_mase" not in horizon_metrics:
        return pd.DataFrame()
    ok = horizon_metrics.copy()
    if "status" in ok:
        ok = ok[ok["status"].eq("ok")]
    baseline_mask = ok["model_id"].eq("seasonal_naive")
    if "model_family" in ok:
        baseline_mask = baseline_mask | ok["model_family"].eq("seasonal_naive")
    baseline = ok[baseline_mask].copy()
    if baseline.empty:
        return pd.DataFrame()

    pair_cols = ["stage", "target_strategy", "fold_name", "horizon_index"]
    baseline = baseline[pair_cols + ["common_mase"]].rename(
        columns={"common_mase": "baseline_common_mase"}
    )
    optional_group_cols = [
        col
        for col in ["model_family", "covid_mode", "complexity", "model_params"]
        if col in ok.columns
    ]
    optional_group_cols.extend(sorted(col for col in ok.columns if col.startswith("model_param_")))
    group_cols = ["stage", "target_strategy", "model_id", *optional_group_cols]
    rows = []
    rng = np.random.default_rng(seed)
    for keys, grp in ok.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = dict(zip(group_cols, key_values, strict=True))
        paired = grp.merge(baseline, on=pair_cols, how="inner")
        if paired.empty:
            continue
        diff = (
            paired["common_mase"].astype(float) - paired["baseline_common_mase"].astype(float)
        ).to_numpy()
        if len(diff) == 0:
            continue
        boot = rng.choice(diff, size=(int(n_bootstrap), len(diff)), replace=True).mean(axis=1)
        candidate_mean = float(paired["common_mase"].mean())
        baseline_mean = float(paired["baseline_common_mase"].mean())
        rows.append(
            {
                **key_map,
                "candidate_mean_common_mase": candidate_mean,
                "baseline_mean_common_mase": baseline_mean,
                "mean_common_mase_diff_vs_seasonal_naive": float(diff.mean()),
                "mean_common_mase_ratio_vs_seasonal_naive": candidate_mean / baseline_mean
                if baseline_mean
                else np.nan,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "bootstrap_probability_beats_seasonal_naive": float(np.mean(boot < 0.0)),
                "bootstrap_samples": int(n_bootstrap),
                "paired_observations": int(len(diff)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["candidate_mean_common_mase", "mean_common_mase_diff_vs_seasonal_naive"])
        .reset_index(drop=True)
    )
