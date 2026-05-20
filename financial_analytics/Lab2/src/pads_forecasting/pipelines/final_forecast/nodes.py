"""Final 2024 forecast nodes."""

from __future__ import annotations

import json
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pads_forecasting.contracts import validate_previsao_shape
from pads_forecasting.covid_adjustment import adjust_training_target, zero_future_covid_exog
from pads_forecasting.modeling import EXOG_COLUMNS, _make_model, future_month_frame, model_specs

FORECAST_INTERVAL_COLUMNS = ["data", "previsao", "lo_80", "hi_80", "lo_95", "hi_95"]
CHALLENGER_COLUMNS = [
    "candidate_role",
    "target_strategy",
    "model_id",
    "model_family",
    "diagnostic_spec_id",
    "diagnostic_spec_resolution",
    "rank",
    "data",
    "previsao",
    "lo_80",
    "hi_80",
    "lo_95",
    "hi_95",
    "status",
]


def _choose_selected(model_selection: pd.DataFrame) -> pd.Series:
    selected_mask = model_selection["selected"].astype(str).str.lower().isin(["true", "1"])
    if selected_mask.any():
        return model_selection[selected_mask].iloc[0]
    sort_cols = [
        col
        for col in ["normal_mean_common_mase", "mean_common_mase", "normal_mean_mase", "mean_mase"]
        if col in model_selection.columns
    ]
    return model_selection.sort_values(sort_cols).iloc[0]


def _legacy_prophet_yearly(model_id: str) -> int | None:
    if not model_id.startswith("prophet_y"):
        return None
    yearly = model_id.removeprefix("prophet_y").split("_", maxsplit=1)[0]
    return int(yearly) if yearly.isdigit() else None


def _resolve_spec(
    item: pd.Series | dict[str, Any],
    specs_by_id: dict[str, dict[str, Any]],
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    """Resolve selected model ids, including legacy compact Prophet ids."""

    model_id = str(item.get("model_id", ""))
    if model_id in specs_by_id:
        return specs_by_id[model_id], model_id, "exact"

    family = str(item.get("model_family", ""))
    candidates = [spec for spec in specs if spec["family"] == family]
    if not candidates:
        raise ValueError(f"Could not resolve model spec for {model_id!r}.")

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


def _predict(
    model: Any, horizon: int, future: pd.DataFrame, future_exog: pd.DataFrame
) -> np.ndarray:
    try:
        return np.asarray(
            model.predict(horizon, future_exog, {"dates": future["data"].reset_index(drop=True)}),
            dtype=float,
        )
    except TypeError:
        return np.asarray(model.predict(horizon, future_exog), dtype=float)


def _prediction_intervals(
    model: Any,
    horizon: int,
    future: pd.DataFrame,
    future_exog: pd.DataFrame,
) -> pd.DataFrame:
    try:
        intervals = model.prediction_intervals(
            horizon,
            future_exog,
            levels=(80, 95),
            config={"dates": future["data"].reset_index(drop=True)},
        )
    except TypeError:
        intervals = model.prediction_intervals(horizon, future_exog, levels=(80, 95))
    return intervals.reset_index(drop=True)


def _fit_candidate(
    *,
    item: pd.Series,
    spec: dict[str, Any],
    diagnostic_spec_id: str,
    diagnostic_spec_resolution: str,
    strategy: pd.DataFrame,
    future: pd.DataFrame,
    future_exog: pd.DataFrame,
    horizon: int,
    season_length: int,
    seed: int,
    candidate_role: str,
    covid_adjustment_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, Any | None, dict[str, Any]]:
    """Fit one final candidate on 2014-2023 observed data and forecast 2024."""

    strategy = strategy.copy().reset_index(drop=True)
    covid_metadata: dict[str, Any] = {}
    if str(spec.get("covid_mode", "none")) == "adjusted_target":
        adjustment = adjust_training_target(
            strategy,
            config=covid_adjustment_config,
            y_col="y",
        )
        strategy["y"] = adjustment.adjusted_y.to_numpy(dtype=float)
        covid_metadata = adjustment.metadata()
    else:
        covid_metadata = {"covid_adjustment_status": "not_applicable"}

    model = _make_model(spec, season_length=season_length, seed=seed)
    y = strategy["y"].astype(float).reset_index(drop=True)
    dates = strategy["data"].reset_index(drop=True)
    exog_columns = [column for column in EXOG_COLUMNS if column in strategy.columns]
    uses_native_covid = str(spec.get("covid_mode", "none")) in {
        "covid",
        "regressors",
        "features",
        "exog",
        "native_dummies",
    }
    exog = strategy[exog_columns].reset_index(drop=True) if uses_native_covid else None
    predict_exog = zero_future_covid_exog(future, exog_columns) if exog_columns else None
    model.fit(y, exog, {"dates": dates, "train_frame": strategy.reset_index(drop=True)})
    yhat = _predict(model, horizon, future, predict_exog)
    intervals = _prediction_intervals(model, horizon, future, predict_exog)
    for column in ["lo_80", "hi_80", "lo_95", "hi_95"]:
        if column not in intervals:
            raise ValueError(f"Missing final interval column: {column}")

    frame = pd.concat(
        [
            future[["data"]].reset_index(drop=True),
            pd.DataFrame({"previsao": yhat}),
            intervals[["lo_80", "hi_80", "lo_95", "hi_95"]].reset_index(drop=True),
        ],
        axis=1,
    )
    frame["data"] = pd.to_datetime(frame["data"]).dt.strftime("%Y-%m-%d")
    frame.insert(0, "rank", item.get("rank", np.nan))
    frame.insert(0, "diagnostic_spec_resolution", diagnostic_spec_resolution)
    frame.insert(0, "diagnostic_spec_id", diagnostic_spec_id)
    frame.insert(0, "model_family", spec["family"])
    frame.insert(0, "model_id", item["model_id"])
    frame.insert(0, "target_strategy", item["target_strategy"])
    frame.insert(0, "candidate_role", candidate_role)
    frame["status"] = "ok"

    payload = {
        "model_id": item["model_id"],
        "target_strategy": item["target_strategy"],
        "model_family": spec["family"],
        "diagnostic_spec_id": diagnostic_spec_id,
        "diagnostic_spec_resolution": diagnostic_spec_resolution,
        "rank": item.get("rank", np.nan),
        "serializable_params": model.serializable_params(),
        "mlflow_payload": model.mlflow_log_payload(),
        "covid_adjustment": covid_metadata,
    }
    return frame, model, payload


def _top_challenger_rows(
    model_selection: pd.DataFrame, selected: pd.Series, n: int = 2
) -> list[pd.Series]:
    selected_key = (selected["target_strategy"], selected["model_id"])
    sort_cols = [
        col
        for col in ["rank", "normal_mean_common_mase", "mean_common_mase", "normal_mean_mase"]
        if col in model_selection.columns
    ]
    ranked = model_selection.sort_values(sort_cols, na_position="last")
    challengers = []
    for _, row in ranked.iterrows():
        key = (row["target_strategy"], row["model_id"])
        if key == selected_key:
            continue
        challengers.append(row)
        if len(challengers) == n:
            break
    return challengers


def _failure_forecast_row(
    item: pd.Series,
    *,
    candidate_role: str,
    status: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {column: np.nan for column in CHALLENGER_COLUMNS}
            | {
                "candidate_role": candidate_role,
                "target_strategy": item.get("target_strategy"),
                "model_id": item.get("model_id"),
                "model_family": item.get("model_family"),
                "rank": item.get("rank", np.nan),
                "status": status,
            }
        ],
        columns=CHALLENGER_COLUMNS,
    )


def _nearest_alpha_candidate(
    alpha_candidates: dict[float, pd.DataFrame],
    desired_alpha: float,
) -> tuple[float, pd.DataFrame]:
    """Resolve a configured alpha candidate without depending on float exactness."""

    if not alpha_candidates:
        raise ValueError("No alpha candidates available for calibrated_alpha final refit.")
    alpha = min(alpha_candidates, key=lambda value: abs(float(value) - float(desired_alpha)))
    return float(alpha), alpha_candidates[alpha].copy()


def _final_alpha_choice(
    item: pd.Series,
    robust_alpha_summary: pd.DataFrame,
    validation: dict[str, Any],
) -> tuple[float | None, str, str]:
    """Choose final alpha with the same explicit objective used in reporting."""

    if str(item.get("target_strategy", "")) != "calibrated_alpha":
        return None, "not_applicable", "not_applicable"
    objective = (
        validation.get("robust_alpha", {})
        .get("final_alpha_objective", "normal_folds")
        .strip()
        .lower()
    )
    objective_columns = {
        "normal_folds": "best_alpha_normal_folds_only_grid",
        "all_folds": "best_alpha_all_folds_grid",
        "stress_downweighted": "best_alpha_stress_downweighted_grid",
    }
    alpha_column = objective_columns.get(objective, "best_alpha_normal_folds_only_grid")
    family = str(item.get("model_family", ""))
    if robust_alpha_summary.empty or "model_family" not in robust_alpha_summary:
        return None, "fallback_configured_calibrated_alpha_no_robust_summary", objective
    match = robust_alpha_summary[robust_alpha_summary["model_family"].astype(str).eq(family)]
    if match.empty or alpha_column not in match:
        return None, "fallback_configured_calibrated_alpha_no_family_row", objective
    row = match.iloc[0]
    delta_column = (
        "normal_common_mase_delta_vs_alpha_one"
        if objective == "normal_folds"
        else "mean_common_mase_delta_vs_alpha_one"
    )
    delta = pd.to_numeric(pd.Series([row.get(delta_column, np.nan)]), errors="coerce").iloc[0]
    if pd.notna(delta) and float(delta) >= 0.0:
        return 1.0, "robust_alpha_reverted_to_one_no_predictive_gain", objective
    alpha = pd.to_numeric(pd.Series([row.get(alpha_column, np.nan)]), errors="coerce").iloc[0]
    if pd.isna(alpha):
        return None, "fallback_configured_calibrated_alpha_alpha_missing", objective
    return float(alpha), f"robust_alpha_{alpha_column}", objective


def _strategy_for_final_item(
    item: pd.Series,
    target_strategies: dict[str, Any],
    robust_alpha_summary: pd.DataFrame,
    validation: dict[str, Any],
) -> tuple[pd.DataFrame, float | None, str, str]:
    """Return the exact target representation for final refit/challengers."""

    strategy_name = str(item["target_strategy"])
    if strategy_name != "calibrated_alpha":
        strategy = target_strategies["strategies"][strategy_name].copy()
        return strategy, None, "fixed_strategy", "not_applicable"

    alpha, source, objective = _final_alpha_choice(item, robust_alpha_summary, validation)
    if alpha is None:
        strategy = target_strategies["strategies"][strategy_name].copy()
        return strategy, float(strategy["alpha"].iloc[0]), source, objective
    resolved_alpha, strategy = _nearest_alpha_candidate(
        target_strategies.get("alpha_candidates", {}),
        alpha,
    )
    return strategy, resolved_alpha, source, objective


def _plot_final_forecast(
    strategy: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    acquisition_date: str,
    figures_dir: Path,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_dates = pd.to_datetime(strategy["data"])
    forecast_dates = pd.to_datetime(forecast["data"])
    path = figures_dir / "final_forecast_intervals.png"

    plt.figure(figsize=(10, 4))
    plt.plot(plot_dates, strategy["y"], label="training target")
    plt.plot(forecast_dates, forecast["previsao"], label="2024 forecast", color="tab:red")
    plt.fill_between(
        forecast_dates,
        forecast["lo_95"].astype(float).to_numpy(),
        forecast["hi_95"].astype(float).to_numpy(),
        color="tab:red",
        alpha=0.12,
        label="95%",
    )
    plt.fill_between(
        forecast_dates,
        forecast["lo_80"].astype(float).to_numpy(),
        forecast["hi_80"].astype(float).to_numpy(),
        color="tab:red",
        alpha=0.22,
        label="80%",
    )
    plt.axvline(pd.Timestamp(acquisition_date), color="black", linestyle="--", linewidth=1)
    plt.title("Final 2024 forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _log_final_forecast_to_mlflow(
    *,
    selected_payload: dict[str, Any],
    forecast: pd.DataFrame,
    previsao: pd.DataFrame,
    challenger_forecasts: pd.DataFrame,
    final_model_metadata: pd.DataFrame,
    figure_path: Path,
    selected_model: Any,
) -> None:
    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        with mlflow.start_run(
            run_name=f"final_forecast/{selected_payload['target_strategy']}/{selected_payload['model_id']}",
            nested=True,
        ):
            mlflow.log_param("target_strategy", selected_payload["target_strategy"])
            mlflow.log_param("model_id", selected_payload["model_id"])
            mlflow.log_param("model_family", selected_payload["model_family"])
            mlflow.log_param("diagnostic_spec_id", selected_payload["diagnostic_spec_id"])
            mlflow.log_param(
                "diagnostic_spec_resolution",
                selected_payload["diagnostic_spec_resolution"],
            )
            mlflow.log_metric("forecast_rows", float(len(forecast)))
            mlflow.log_metric("forecast_mean", float(forecast["previsao"].mean()))
            mlflow.log_metric(
                "mean_interval_width_80",
                float((forecast["hi_80"] - forecast["lo_80"]).mean()),
            )
            mlflow.log_metric(
                "mean_interval_width_95",
                float((forecast["hi_95"] - forecast["lo_95"]).mean()),
            )
            for key, value in selected_payload["serializable_params"].items():
                mlflow.log_param(f"model_param.{key}", str(value))

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                forecast_path = tmp / "forecast_intervals.parquet"
                previsao_path = tmp / "previsao.csv"
                challenger_path = tmp / "challenger_forecasts.parquet"
                metadata_path = tmp / "final_model_metadata.parquet"
                params_path = tmp / "selected_model_params.json"
                model_path = tmp / "selected_final_model.pkl"
                forecast.to_parquet(forecast_path, index=False)
                previsao.to_csv(previsao_path, index=False)
                challenger_forecasts.to_parquet(challenger_path, index=False)
                final_model_metadata.to_parquet(metadata_path, index=False)
                params_path.write_text(
                    json.dumps(selected_payload["serializable_params"], indent=2, default=str),
                    encoding="utf-8",
                )
                for path in [
                    forecast_path,
                    previsao_path,
                    challenger_path,
                    metadata_path,
                    params_path,
                ]:
                    mlflow.log_artifact(str(path))
                try:
                    with model_path.open("wb") as handle:
                        pickle.dump(selected_model, handle)
                    mlflow.log_artifact(str(model_path))
                except Exception:
                    pass
                if hasattr(selected_model, "arviz_summary"):
                    try:
                        summary_path = tmp / "bvar_arviz_summary.parquet"
                        selected_model.arviz_summary().to_parquet(summary_path, index=False)
                        mlflow.log_artifact(str(summary_path))
                    except Exception:
                        pass
            if figure_path.exists():
                mlflow.log_artifact(str(figure_path))
    except Exception:
        return


def run_final_forecast(
    target_strategies: dict[str, Any],
    model_selection: pd.DataFrame,
    robust_alpha_summary: pd.DataFrame,
    project: dict[str, Any],
    data: dict[str, Any],
    validation: dict[str, Any],
    interventions: dict[str, Any],
    models: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit selected model on all observed 2014-2023 data and forecast 2024."""

    future_covid_value = int(interventions["covid"]["future_value"])
    if future_covid_value != 0:
        raise ValueError("Final forecast requires future COVID covariates equal to 0.")

    selected = _choose_selected(model_selection)
    spec_list = model_specs(models, stage="model_comparison", include_optional=True)
    specs = {spec["model_id"]: spec for spec in spec_list}
    selected_spec, diagnostic_spec_id, diagnostic_spec_resolution = _resolve_spec(
        selected,
        specs,
        spec_list,
    )
    strategy, final_alpha, final_alpha_source, final_alpha_objective = _strategy_for_final_item(
        selected,
        target_strategies,
        robust_alpha_summary,
        validation,
    )
    strategy["data"] = pd.to_datetime(strategy["data"])
    future = future_month_frame(
        data["final_forecast_start"],
        data["horizon"],
        covid_value=future_covid_value,
    )
    future_exog = future[EXOG_COLUMNS]
    if not future_exog.eq(0).all().all():
        raise ValueError("Final forecast future COVID covariates must be zero.")

    forecast_with_meta, selected_model, selected_payload = _fit_candidate(
        item=selected,
        spec=selected_spec,
        diagnostic_spec_id=diagnostic_spec_id,
        diagnostic_spec_resolution=diagnostic_spec_resolution,
        strategy=strategy,
        future=future,
        future_exog=future_exog,
        horizon=int(data["horizon"]),
        season_length=int(validation["season_length"]),
        seed=int(project["seed"]),
        candidate_role="selected",
        covid_adjustment_config=validation.get("covid_adjustment", {}),
    )
    forecast = forecast_with_meta[FORECAST_INTERVAL_COLUMNS].copy()
    validate_previsao_shape(forecast[["data", "previsao"]])
    previsao = forecast[["data", "previsao"]].copy()
    validate_previsao_shape(previsao)

    challenger_rows = _top_challenger_rows(model_selection, selected, n=2)
    challenger_frames: list[pd.DataFrame] = []

    def fit_challenger(row: pd.Series) -> pd.DataFrame:
        try:
            spec, resolved_id, resolution = _resolve_spec(row, specs, spec_list)
            challenger_strategy, _alpha, _alpha_source, _alpha_objective = _strategy_for_final_item(
                row,
                target_strategies,
                robust_alpha_summary,
                validation,
            )
            challenger_strategy["data"] = pd.to_datetime(challenger_strategy["data"])
            frame, _, _payload = _fit_candidate(
                item=row,
                spec=spec,
                diagnostic_spec_id=resolved_id,
                diagnostic_spec_resolution=resolution,
                strategy=challenger_strategy,
                future=future,
                future_exog=future_exog,
                horizon=int(data["horizon"]),
                season_length=int(validation["season_length"]),
                seed=int(project["seed"]),
                candidate_role="challenger",
                covid_adjustment_config=validation.get("covid_adjustment", {}),
            )
            return frame[CHALLENGER_COLUMNS]
        except Exception as exc:
            return _failure_forecast_row(
                row,
                candidate_role="challenger",
                status=f"failed: {exc}",
            )

    if challenger_rows:
        with ThreadPoolExecutor(max_workers=min(2, len(challenger_rows))) as executor:
            challenger_frames.extend(executor.map(fit_challenger, challenger_rows))
    challenger_forecasts = (
        pd.concat(challenger_frames, ignore_index=True)
        if challenger_frames
        else pd.DataFrame(columns=CHALLENGER_COLUMNS)
    )

    figure_path = _plot_final_forecast(
        strategy,
        forecast,
        acquisition_date=data["acquisition_date"],
        figures_dir=Path(outputs["figures_dir"]),
    )
    final_model_metadata = pd.DataFrame(
        [
            {
                "selected_target_strategy": selected["target_strategy"],
                "selected_final_alpha": final_alpha,
                "selected_final_alpha_source": final_alpha_source,
                "selected_final_alpha_objective": final_alpha_objective,
                "selected_model_id": selected["model_id"],
                "selected_model_family": selected_spec["family"],
                "diagnostic_spec_id": diagnostic_spec_id,
                "diagnostic_spec_resolution": diagnostic_spec_resolution,
                "selected_rank": selected.get("rank", np.nan),
                "selection_reason": selected.get("selection_reason", ""),
                "train_start": strategy["data"].min().strftime("%Y-%m-%d"),
                "train_end": strategy["data"].max().strftime("%Y-%m-%d"),
                "train_rows": len(strategy),
                "forecast_start": pd.Timestamp(data["final_forecast_start"]).strftime("%Y-%m-%d"),
                "forecast_end": future["data"].max().strftime("%Y-%m-%d"),
                "forecast_rows": len(forecast),
                "future_covid_shock_sum": int(future["covid_shock"].sum()),
                "future_covid_recovery_sum": int(future["covid_recovery"].sum()),
                "future_covid_aftershock_2021_sum": int(
                    future.get("covid_aftershock_2021", pd.Series(dtype=int)).sum()
                ),
                "selected_covid_mode": selected_spec.get("covid_mode", "none"),
                "selected_covid_adjustment_status": selected_payload.get(
                    "covid_adjustment",
                    {},
                ).get("covid_adjustment_status", "not_applicable"),
                "selected_covid_adjustment_estimator": selected_payload.get(
                    "covid_adjustment",
                    {},
                ).get("covid_adjustment_estimator", "not_applicable"),
                "challenger_count": int(
                    challenger_forecasts["model_id"].nunique()
                    if not challenger_forecasts.empty
                    else 0
                ),
                "figure_path": str(figure_path),
                "model_params": json.dumps(selected_spec["params"], sort_keys=True, default=str),
                "serializable_params": json.dumps(
                    selected_payload["serializable_params"],
                    sort_keys=True,
                    default=str,
                ),
            }
        ]
    )
    _log_final_forecast_to_mlflow(
        selected_payload=selected_payload,
        forecast=forecast,
        previsao=previsao,
        challenger_forecasts=challenger_forecasts,
        final_model_metadata=final_model_metadata,
        figure_path=figure_path,
        selected_model=selected_model,
    )
    return forecast, previsao, challenger_forecasts, final_model_metadata
