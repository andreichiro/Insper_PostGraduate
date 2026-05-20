"""Residual diagnostics and interval validation nodes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from pads_forecasting.covid_adjustment import adjust_training_target, zero_future_covid_exog
from pads_forecasting.metrics import mae
from pads_forecasting.modeling import EXOG_COLUMNS, _make_model, model_specs
from pads_forecasting.pipelines.validation.nodes import make_fold_slices

DIAGNOSTIC_METRIC_COLUMNS = [
    "residual_mean",
    "residual_std",
    "residual_bias",
    "forecast_bias",
    "acf_lag_1",
    "acf_lag_12",
    "acf_abs_max_lag_24",
    "ljung_box_p_lag_12",
    "ljung_box_p_lag_24",
    "interval_coverage_80",
    "interval_coverage_95",
    "interval_mean_width_80",
    "interval_mean_width_95",
    "interval_fold_count",
    "interval_observation_count",
    "n_residuals",
]

INTERVAL_COVERAGE_COLUMNS = [
    "target_strategy",
    "model_id",
    "model_family",
    "diagnostic_spec_id",
    "diagnostic_spec_resolution",
    "fold_name",
    "fold_role",
    "train_end",
    "valid_start",
    "valid_end",
    "horizon",
    "coverage_80",
    "coverage_95",
    "covered_80_count",
    "covered_95_count",
    "observation_count",
    "mean_width_80",
    "mean_width_95",
    "forecast_bias",
    "mae",
    "status",
]

INTERVAL_PREDICTION_COLUMNS = [
    "target_strategy",
    "model_id",
    "model_family",
    "diagnostic_spec_id",
    "diagnostic_spec_resolution",
    "fold_name",
    "fold_role",
    "data",
    "y_true",
    "yhat",
    "lo_80",
    "hi_80",
    "lo_95",
    "hi_95",
    "covered_80",
    "covered_95",
    "status",
]


def _selected_rows(model_selection: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if model_selection.empty:
        return pd.DataFrame()
    selected_mask = model_selection["selected"].astype(str).str.lower().isin(["true", "1"])
    chosen = model_selection[selected_mask]
    if chosen.empty:
        chosen = model_selection.head(1)
    challengers = model_selection.head(n)
    return pd.concat([chosen, challengers], ignore_index=True).drop_duplicates(
        ["target_strategy", "model_id"]
    )


def _resolve_spec(
    item: dict[str, Any],
    specs_by_id: dict[str, dict[str, Any]],
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None, str]:
    """Resolve exact specs, with compatibility fallback for stale candidate ids."""

    model_id = str(item.get("model_id", ""))
    if model_id in specs_by_id:
        return specs_by_id[model_id], model_id, "exact"

    family = str(item.get("model_family", ""))
    candidates = [spec for spec in specs if spec["family"] == family]
    if not candidates:
        return None, None, "missing"

    if family == "prophet":
        parsed_yearly = _legacy_prophet_yearly(model_id)
        if parsed_yearly is not None:
            yearly_matches = [
                spec
                for spec in candidates
                if int(spec["params"].get("yearly_seasonality", -1)) == parsed_yearly
            ]
            if yearly_matches:
                candidates = yearly_matches
        if model_id.endswith("_none"):
            non_covid = [
                spec
                for spec in candidates
                if not spec["params"].get("use_covid_regressors", False)
                and spec.get("covid_mode") == "none"
            ]
            if non_covid:
                candidates = non_covid
        elif model_id.endswith("_covid"):
            covid = [
                spec for spec in candidates if spec["params"].get("use_covid_regressors", False)
            ]
            if covid:
                candidates = covid

    chosen = sorted(candidates, key=lambda spec: spec["model_id"])[0]
    return chosen, chosen["model_id"], "compatible_model_family"


def _legacy_prophet_yearly(model_id: str) -> int | None:
    """Parse legacy ids like prophet_y5_none."""

    if not model_id.startswith("prophet_y"):
        return None
    yearly = model_id.removeprefix("prophet_y").split("_", maxsplit=1)[0]
    return int(yearly) if yearly.isdigit() else None


def _exog(df: pd.DataFrame) -> pd.DataFrame | None:
    columns = [column for column in EXOG_COLUMNS if column in df.columns]
    return df[columns].reset_index(drop=True) if columns else None


def _uses_native_covid(spec: dict[str, Any]) -> bool:
    return str(spec.get("covid_mode", "none")) in {
        "covid",
        "regressors",
        "features",
        "exog",
        "native_dummies",
    }


def _prepare_train_for_spec(
    spec: dict[str, Any],
    train: pd.DataFrame,
    validation: dict[str, Any],
) -> pd.DataFrame:
    out = train.copy().reset_index(drop=True)
    if str(spec.get("covid_mode", "none")) == "adjusted_target":
        adjustment = adjust_training_target(
            out,
            validation.get("covid_adjustment", {}),
            y_col="y",
        )
        out["y"] = adjustment.adjusted_y.to_numpy(dtype=float)
    return out


def _fit_model(
    spec: dict[str, Any],
    train: pd.DataFrame,
    *,
    season_length: int,
    seed: int,
    validation: dict[str, Any],
):
    train = _prepare_train_for_spec(spec, train, validation)
    model = _make_model(spec, season_length=season_length, seed=seed)
    train_exog = _exog(train) if _uses_native_covid(spec) else None
    model.fit(
        train["y"].astype(float).reset_index(drop=True),
        train_exog,
        {
            "dates": train["data"].reset_index(drop=True),
            "train_frame": train.reset_index(drop=True),
        },
    )
    return model


def _future_zero_exog(valid: pd.DataFrame) -> pd.DataFrame | None:
    exog = _exog(valid)
    return zero_future_covid_exog(valid, list(exog.columns)) if exog is not None else None


def _predict(model: Any, valid: pd.DataFrame) -> np.ndarray:
    valid_exog = _future_zero_exog(valid)
    future_dates = valid["data"].reset_index(drop=True)
    try:
        return np.asarray(
            model.predict(len(valid), valid_exog, {"dates": future_dates}),
            dtype=float,
        )
    except TypeError:
        return np.asarray(model.predict(len(valid), valid_exog), dtype=float)


def _prediction_intervals(model: Any, valid: pd.DataFrame) -> pd.DataFrame:
    valid_exog = _future_zero_exog(valid)
    future_dates = valid["data"].reset_index(drop=True)
    try:
        intervals = model.prediction_intervals(
            len(valid),
            valid_exog,
            levels=(80, 95),
            config={"dates": future_dates},
        )
    except TypeError:
        intervals = model.prediction_intervals(len(valid), valid_exog, levels=(80, 95))
    return intervals.reset_index(drop=True)


def _ljung_box_pvalues(residuals: pd.Series) -> dict[str, float]:
    values = residuals.dropna().astype(float)
    available_lags = [lag for lag in [12, 24] if len(values) > lag]
    out = {"ljung_box_p_lag_12": np.nan, "ljung_box_p_lag_24": np.nan}
    if not available_lags:
        return out
    lb = acorr_ljungbox(values, lags=available_lags, return_df=True)
    for lag in available_lags:
        out[f"ljung_box_p_lag_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    return out


def _acf_summary(residuals: pd.Series) -> dict[str, float]:
    values = residuals.dropna().astype(float)
    if len(values) < 2:
        return {
            "acf_lag_1": np.nan,
            "acf_lag_12": np.nan,
            "acf_abs_max_lag_24": np.nan,
        }
    acf_values = acf(values, nlags=min(24, len(values) - 1), fft=False)
    lag_1 = float(acf_values[1]) if len(acf_values) > 1 else np.nan
    lag_12 = float(acf_values[12]) if len(acf_values) > 12 else np.nan
    max_abs = float(np.max(np.abs(acf_values[1:]))) if len(acf_values) > 1 else np.nan
    return {
        "acf_lag_1": lag_1,
        "acf_lag_12": lag_12,
        "acf_abs_max_lag_24": max_abs,
    }


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _plot_residuals(
    residuals: pd.Series,
    dates: pd.Series,
    *,
    figures_dir: Path,
    model_id: str,
    write_primary: bool,
) -> list[Path]:
    paths = []
    suffix = _safe_filename(model_id)
    residual_dates = pd.to_datetime(dates).iloc[-len(residuals) :]

    acf_values = acf(residuals, nlags=min(24, len(residuals) - 1), fft=False)
    for path in [
        figures_dir / f"residual_acf_{suffix}.png",
        *([figures_dir / "residual_acf.png"] if write_primary else []),
    ]:
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(acf_values)), acf_values)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(f"Residual ACF: {model_id}")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    for path in [
        figures_dir / f"residual_time_{suffix}.png",
        *([figures_dir / "residual_time.png"] if write_primary else []),
    ]:
        plt.figure(figsize=(9, 4))
        plt.plot(residual_dates, residuals.to_numpy())
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(f"Residual time plot: {model_id}")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    for path in [
        figures_dir / f"residual_histogram_{suffix}.png",
        *([figures_dir / "residual_histogram.png"] if write_primary else []),
    ]:
        plt.figure(figsize=(7, 4))
        plt.hist(residuals.to_numpy(), bins=min(20, max(5, int(np.sqrt(len(residuals))))))
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"Residual histogram: {model_id}")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    return paths


def _interval_coverage_for_candidate(
    spec: dict[str, Any],
    strategy: pd.DataFrame,
    folds_metadata: pd.DataFrame,
    *,
    season_length: int,
    seed: int,
    item: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_rows = []
    for fold in folds_metadata.to_dict("records"):
        base = {
            "target_strategy": item["target_strategy"],
            "model_id": item["model_id"],
            "model_family": spec["family"],
            "diagnostic_spec_id": item.get("diagnostic_spec_id", spec["model_id"]),
            "diagnostic_spec_resolution": item.get("diagnostic_spec_resolution", "exact"),
            "fold_name": fold["fold_name"],
            "fold_role": fold["fold_role"],
            "train_end": fold["train_end"],
            "valid_start": fold["valid_start"],
            "valid_end": fold["valid_end"],
            "horizon": fold["horizon"],
        }
        try:
            train, valid = make_fold_slices(strategy, fold)
            model = _fit_model(
                spec,
                train,
                season_length=season_length,
                seed=seed,
                validation=validation,
            )
            y_true = valid["y"].astype(float).reset_index(drop=True)
            yhat = pd.Series(_predict(model, valid))
            intervals = _prediction_intervals(model, valid)
            for column in ["lo_80", "hi_80", "lo_95", "hi_95"]:
                if column not in intervals:
                    raise ValueError(f"Missing interval column: {column}")
            covered_80 = y_true.between(intervals["lo_80"], intervals["hi_80"])
            covered_95 = y_true.between(intervals["lo_95"], intervals["hi_95"])
            prediction_rows.extend(
                pd.DataFrame(
                    {
                        "target_strategy": item["target_strategy"],
                        "model_id": item["model_id"],
                        "model_family": spec["family"],
                        "diagnostic_spec_id": base["diagnostic_spec_id"],
                        "diagnostic_spec_resolution": base["diagnostic_spec_resolution"],
                        "fold_name": fold["fold_name"],
                        "fold_role": fold["fold_role"],
                        "data": valid["data"].reset_index(drop=True),
                        "y_true": y_true,
                        "yhat": yhat,
                        "lo_80": intervals["lo_80"].to_numpy(dtype=float),
                        "hi_80": intervals["hi_80"].to_numpy(dtype=float),
                        "lo_95": intervals["lo_95"].to_numpy(dtype=float),
                        "hi_95": intervals["hi_95"].to_numpy(dtype=float),
                        "covered_80": covered_80.to_numpy(dtype=bool),
                        "covered_95": covered_95.to_numpy(dtype=bool),
                        "status": "ok",
                    }
                ).to_dict("records")
            )
            rows.append(
                {
                    **base,
                    "coverage_80": float(covered_80.mean()),
                    "coverage_95": float(covered_95.mean()),
                    "covered_80_count": int(covered_80.sum()),
                    "covered_95_count": int(covered_95.sum()),
                    "observation_count": int(len(y_true)),
                    "mean_width_80": float((intervals["hi_80"] - intervals["lo_80"]).mean()),
                    "mean_width_95": float((intervals["hi_95"] - intervals["lo_95"]).mean()),
                    "forecast_bias": float((yhat - y_true).mean()),
                    "mae": mae(y_true, yhat),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "coverage_80": np.nan,
                    "coverage_95": np.nan,
                    "covered_80_count": 0,
                    "covered_95_count": 0,
                    "observation_count": 0,
                    "mean_width_80": np.nan,
                    "mean_width_95": np.nan,
                    "forecast_bias": np.nan,
                    "mae": np.nan,
                    "status": f"failed: {exc}",
                }
            )
            prediction_rows.append(
                {
                    "target_strategy": item["target_strategy"],
                    "model_id": item["model_id"],
                    "model_family": spec["family"],
                    "diagnostic_spec_id": base["diagnostic_spec_id"],
                    "diagnostic_spec_resolution": base["diagnostic_spec_resolution"],
                    "fold_name": fold["fold_name"],
                    "fold_role": fold["fold_role"],
                    "data": pd.NaT,
                    "y_true": np.nan,
                    "yhat": np.nan,
                    "lo_80": np.nan,
                    "hi_80": np.nan,
                    "lo_95": np.nan,
                    "hi_95": np.nan,
                    "covered_80": False,
                    "covered_95": False,
                    "status": f"failed: {exc}",
                }
            )
    return (
        pd.DataFrame(rows, columns=INTERVAL_COVERAGE_COLUMNS),
        pd.DataFrame(prediction_rows, columns=INTERVAL_PREDICTION_COLUMNS),
    )


def _aggregate_interval_coverage(coverage: pd.DataFrame) -> dict[str, float]:
    ok = coverage[coverage["status"].eq("ok")].copy()
    if ok.empty:
        return {
            "interval_coverage_80": np.nan,
            "interval_coverage_95": np.nan,
            "interval_mean_width_80": np.nan,
            "interval_mean_width_95": np.nan,
            "interval_fold_count": 0,
            "interval_observation_count": 0,
        }
    observations = int(ok["observation_count"].sum())
    return {
        "interval_coverage_80": float(ok["covered_80_count"].sum() / observations)
        if observations
        else np.nan,
        "interval_coverage_95": float(ok["covered_95_count"].sum() / observations)
        if observations
        else np.nan,
        "interval_mean_width_80": float(
            np.average(ok["mean_width_80"], weights=ok["observation_count"])
        ),
        "interval_mean_width_95": float(
            np.average(ok["mean_width_95"], weights=ok["observation_count"])
        ),
        "interval_fold_count": int(ok["fold_name"].nunique()),
        "interval_observation_count": observations,
    }


def _plot_interval_coverage(
    interval_coverage_proxy: pd.DataFrame,
    *,
    figures_dir: Path,
) -> Path | None:
    ok = interval_coverage_proxy[interval_coverage_proxy["status"].eq("ok")]
    if ok.empty:
        return None
    summary = (
        ok.groupby(["target_strategy", "model_id"], dropna=False)
        .agg(
            covered_80_count=("covered_80_count", "sum"),
            covered_95_count=("covered_95_count", "sum"),
            observation_count=("observation_count", "sum"),
        )
        .reset_index()
    )
    summary["coverage_80"] = summary["covered_80_count"] / summary["observation_count"]
    summary["coverage_95"] = summary["covered_95_count"] / summary["observation_count"]
    labels = summary["target_strategy"] + "\n" + summary["model_id"].astype(str).str[:24]
    x = np.arange(len(summary))
    width = 0.35
    path = figures_dir / "interval_coverage_proxy.png"
    plt.figure(figsize=(max(8, len(summary) * 2.2), 4))
    plt.bar(x - width / 2, summary["coverage_80"], width, label="80%")
    plt.bar(x + width / 2, summary["coverage_95"], width, label="95%")
    plt.axhline(0.8, color="tab:blue", linestyle="--", linewidth=1)
    plt.axhline(0.95, color="tab:orange", linestyle="--", linewidth=1)
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("CV coverage proxy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _log_to_mlflow(
    diagnostics: pd.DataFrame,
    interval_coverage_proxy: pd.DataFrame,
    interval_validation_predictions: pd.DataFrame,
    artifact_paths: list[Path],
) -> None:
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics_path = Path(tmpdir) / "residual_diagnostics.parquet"
            coverage_path = Path(tmpdir) / "interval_coverage_proxy.parquet"
            prediction_path = Path(tmpdir) / "interval_validation_predictions.parquet"
            diagnostics.to_parquet(diagnostics_path, index=False)
            interval_coverage_proxy.to_parquet(coverage_path, index=False)
            interval_validation_predictions.to_parquet(prediction_path, index=False)
            mlflow.log_artifact(str(diagnostics_path))
            mlflow.log_artifact(str(coverage_path))
            mlflow.log_artifact(str(prediction_path))
        for path in artifact_paths:
            if path.exists():
                mlflow.log_artifact(str(path))
        for _, row in diagnostics.iterrows():
            run_name = f"diagnostics/{row['target_strategy']}/{row['model_id']}"
            with mlflow.start_run(run_name=run_name, nested=True):
                for param in [
                    "target_strategy",
                    "model_id",
                    "model_family",
                    "diagnostic_spec_id",
                    "diagnostic_spec_resolution",
                    "selected",
                    "rank",
                    "status",
                ]:
                    if param in row and pd.notna(row[param]):
                        mlflow.log_param(param, str(row[param]))
                for metric in DIAGNOSTIC_METRIC_COLUMNS:
                    if metric in row and pd.notna(row[metric]):
                        mlflow.log_metric(metric, float(row[metric]))
    except Exception:
        return


def run_residual_diagnostics(
    target_strategies: dict[str, Any],
    model_selection: pd.DataFrame,
    folds_metadata: pd.DataFrame,
    project: dict[str, Any],
    validation: dict[str, Any],
    models: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute residual diagnostics and CV interval coverage proxies for top candidates."""

    spec_list = model_specs(models, stage="model_comparison", include_optional=True)
    specs = {spec["model_id"]: spec for spec in spec_list}
    figures_dir = Path(outputs["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed = int(project["seed"])
    season_length = int(validation["season_length"])

    diagnostics_rows = []
    interval_rows = []
    interval_prediction_rows = []
    artifact_paths: list[Path] = []
    plotted_primary = False

    for item in _selected_rows(model_selection).to_dict("records"):
        spec, diagnostic_spec_id, diagnostic_spec_resolution = _resolve_spec(
            item,
            specs,
            spec_list,
        )
        base = {
            "target_strategy": item["target_strategy"],
            "model_id": item["model_id"],
            "model_family": item.get("model_family", spec["family"] if spec else "unknown"),
            "diagnostic_spec_id": diagnostic_spec_id,
            "diagnostic_spec_resolution": diagnostic_spec_resolution,
            "selected": bool(str(item.get("selected", "")).lower() in {"true", "1"}),
            "rank": item.get("rank", np.nan),
        }
        if spec is None:
            diagnostics_rows.append(
                {
                    **base,
                    "residual_mean": np.nan,
                    "residual_std": np.nan,
                    "residual_bias": np.nan,
                    "forecast_bias": np.nan,
                    "acf_lag_1": np.nan,
                    "acf_lag_12": np.nan,
                    "acf_abs_max_lag_24": np.nan,
                    "ljung_box_p_lag_12": np.nan,
                    "ljung_box_p_lag_24": np.nan,
                    "interval_coverage_80": np.nan,
                    "interval_coverage_95": np.nan,
                    "interval_mean_width_80": np.nan,
                    "interval_mean_width_95": np.nan,
                    "interval_fold_count": 0,
                    "interval_observation_count": 0,
                    "n_residuals": 0,
                    "status": "failed: model spec not found",
                }
            )
            continue

        strategy = target_strategies["strategies"][item["target_strategy"]].copy()
        strategy["data"] = pd.to_datetime(strategy["data"])
        try:
            model = _fit_model(
                spec,
                strategy,
                season_length=season_length,
                seed=seed,
                validation=validation,
            )
            residuals = model.residuals().dropna().astype(float).reset_index(drop=True)
            if residuals.empty:
                raise ValueError("No residuals available for diagnostics.")
            coverage, interval_predictions = _interval_coverage_for_candidate(
                spec,
                strategy,
                folds_metadata,
                season_length=season_length,
                seed=seed,
                item=item,
                validation=validation,
            )
            coverage["diagnostic_spec_id"] = diagnostic_spec_id
            coverage["diagnostic_spec_resolution"] = diagnostic_spec_resolution
            interval_predictions["diagnostic_spec_id"] = diagnostic_spec_id
            interval_predictions["diagnostic_spec_resolution"] = diagnostic_spec_resolution
            interval_rows.append(coverage)
            interval_prediction_rows.append(interval_predictions)
            residual_mean = float(residuals.mean())
            diagnostics_rows.append(
                {
                    **base,
                    "residual_mean": residual_mean,
                    "residual_std": float(residuals.std(ddof=0)),
                    "residual_bias": residual_mean,
                    "forecast_bias": -residual_mean,
                    **_acf_summary(residuals),
                    **_ljung_box_pvalues(residuals),
                    **_aggregate_interval_coverage(coverage),
                    "n_residuals": int(len(residuals)),
                    "status": "ok",
                }
            )
            artifact_paths.extend(
                _plot_residuals(
                    residuals,
                    strategy["data"],
                    figures_dir=figures_dir,
                    model_id=item["model_id"],
                    write_primary=not plotted_primary,
                )
            )
            plotted_primary = True
        except Exception as exc:
            diagnostics_rows.append(
                {
                    **base,
                    "residual_mean": np.nan,
                    "residual_std": np.nan,
                    "residual_bias": np.nan,
                    "forecast_bias": np.nan,
                    "acf_lag_1": np.nan,
                    "acf_lag_12": np.nan,
                    "acf_abs_max_lag_24": np.nan,
                    "ljung_box_p_lag_12": np.nan,
                    "ljung_box_p_lag_24": np.nan,
                    "interval_coverage_80": np.nan,
                    "interval_coverage_95": np.nan,
                    "interval_mean_width_80": np.nan,
                    "interval_mean_width_95": np.nan,
                    "interval_fold_count": 0,
                    "interval_observation_count": 0,
                    "n_residuals": 0,
                    "status": f"failed: {exc}",
                }
            )

    diagnostics = pd.DataFrame(diagnostics_rows)
    interval_coverage_proxy = (
        pd.concat(interval_rows, ignore_index=True)
        if interval_rows
        else pd.DataFrame(columns=INTERVAL_COVERAGE_COLUMNS)
    )
    interval_validation_predictions = (
        pd.concat(interval_prediction_rows, ignore_index=True)
        if interval_prediction_rows
        else pd.DataFrame(columns=INTERVAL_PREDICTION_COLUMNS)
    )
    coverage_plot = _plot_interval_coverage(
        interval_coverage_proxy,
        figures_dir=figures_dir,
    )
    if coverage_plot is not None:
        artifact_paths.append(coverage_plot)
    _log_to_mlflow(
        diagnostics,
        interval_coverage_proxy,
        interval_validation_predictions,
        artifact_paths,
    )
    return diagnostics, interval_coverage_proxy, interval_validation_predictions
