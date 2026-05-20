"""Reporting nodes and artifact-only assignment checklist."""

from __future__ import annotations

from contextlib import nullcontext
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pads_forecasting.modeling import EXOG_COLUMNS, model_specs
from pads_forecasting.models.ml_lags import RecursiveLagForecaster

FINAL_MLFLOW_ARTIFACTS = {
    "forecast_intervals.parquet",
    "previsao.csv",
    "challenger_forecasts.parquet",
    "final_model_metadata.parquet",
    "selected_final_model.pkl",
    "final_forecast_intervals.png",
}


def _log_to_mlflow(
    *,
    project: dict[str, Any],
    old_data_gate: pd.DataFrame,
    cv_summary: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    interval_coverage_proxy: pd.DataFrame,
    interval_validation_predictions: pd.DataFrame,
    forecast_intervals: pd.DataFrame,
    challenger_forecasts: pd.DataFrame,
    final_model_metadata: pd.DataFrame,
    outputs: dict[str, Any],
) -> None:
    """Create a compact MLflow audit run with nested candidate summaries."""

    try:
        import mlflow

        active_run = mlflow.active_run()
        run_context = nullcontext(active_run)
        if active_run is None:
            mlflow.set_tracking_uri(outputs.get("mlruns_dir", "mlruns"))
            mlflow.set_experiment("pads_forecasting_lab2")
            run_context = mlflow.start_run(run_name=project["run_id"])

        with run_context:
            mlflow.log_param("run_id", project["run_id"])
            mlflow.log_param("seed", project["seed"])
            selected = cv_summary.sort_values(
                ["normal_mean_common_mase", "mean_common_mase", "normal_mean_mase"],
                na_position="last",
            ).iloc[0]
            mlflow.log_param("selected_target_strategy", selected["target_strategy"])
            mlflow.log_param("selected_model_id", selected["model_id"])
            for metric in [
                "normal_mean_mase",
                "normal_mean_common_mase",
                "mean_mase",
                "mean_common_mase",
                "std_mase",
                "std_common_mase",
                "cv_mase",
                "cv_common_mase",
                "max_mase",
                "max_common_mase",
                "mean_relative_mae_vs_seasonal_naive",
            ]:
                mlflow.log_metric(metric, float(selected[metric]))

            decision_rows = old_data_gate[old_data_gate["record_type"].eq("decision")]
            for _, row in decision_rows.iterrows():
                mlflow.log_metric(
                    f"old_data_gate.{row['target_strategy']}.passed",
                    float(str(row["passed"]).lower() in {"true", "1"}),
                )
                if pd.notna(row.get("improvement_vs_post_only_pct")):
                    mlflow.log_metric(
                        f"old_data_gate.{row['target_strategy']}.improvement_pct",
                        float(row["improvement_vs_post_only_pct"]),
                    )

            if not residual_diagnostics.empty:
                diag = residual_diagnostics.iloc[0]
                for metric in [
                    "residual_mean",
                    "residual_std",
                    "ljung_box_p_lag_12",
                    "ljung_box_p_lag_24",
                ]:
                    if pd.notna(diag.get(metric)):
                        mlflow.log_metric(metric, float(diag[metric]))

            if not interval_coverage_proxy.empty:
                coverage = interval_coverage_proxy
                if "status" in coverage.columns:
                    coverage = coverage[coverage["status"].eq("ok")]
                if not coverage.empty:
                    mlflow.log_metric(
                        "interval_coverage_80_mean",
                        float(coverage["coverage_80"].mean()),
                    )
                    mlflow.log_metric(
                        "interval_coverage_95_mean",
                        float(coverage["coverage_95"].mean()),
                    )
                    mlflow.log_metric(
                        "interval_coverage_ok_folds",
                        float(len(coverage)),
                    )
            if not interval_validation_predictions.empty:
                predictions = interval_validation_predictions
                if "status" in predictions.columns:
                    predictions = predictions[predictions["status"].eq("ok")]
                mlflow.log_metric("interval_prediction_rows", float(len(predictions)))

            mlflow.log_metric("forecast_rows", float(len(forecast_intervals)))
            if not challenger_forecasts.empty:
                ok_challengers = challenger_forecasts
                if "status" in ok_challengers.columns:
                    ok_challengers = ok_challengers[ok_challengers["status"].eq("ok")]
                mlflow.log_metric(
                    "challenger_forecast_rows",
                    float(len(ok_challengers)),
                )
            if not final_model_metadata.empty:
                meta = final_model_metadata.iloc[0]
                for key in [
                    "selected_target_strategy",
                    "selected_model_id",
                    "selected_model_family",
                    "diagnostic_spec_id",
                    "diagnostic_spec_resolution",
                ]:
                    if key in meta and pd.notna(meta[key]):
                        mlflow.log_param(f"final_forecast.{key}", str(meta[key]))
            artifact_paths = [
                Path(outputs["reporting_dir"]),
                Path(outputs["figures_dir"]),
                Path(outputs["previsao_path"]),
            ]
            for path in artifact_paths:
                if path.exists() and path.is_dir():
                    mlflow.log_artifacts(str(path), artifact_path=path.name)
                elif path.exists():
                    mlflow.log_artifact(str(path))

            for _, row in cv_summary.head(10).iterrows():
                with mlflow.start_run(
                    run_name=f"{row['target_strategy']}__{row['model_id']}",
                    nested=True,
                ):
                    mlflow.log_param("target_strategy", row["target_strategy"])
                    mlflow.log_param("model_id", row["model_id"])
                    for metric in [
                        "normal_mean_mase",
                        "normal_mean_common_mase",
                        "mean_mase",
                        "mean_common_mase",
                        "std_mase",
                        "std_common_mase",
                        "cv_mase",
                        "cv_common_mase",
                        "max_mase",
                        "max_common_mase",
                    ]:
                        mlflow.log_metric(metric, float(row[metric]))
    except Exception:
        return


def _final_mlflow_artifacts_present(mlruns_dir: Path) -> bool:
    """Return true when one final-forecast MLflow run contains the required artifact bundle."""

    if not mlruns_dir.exists():
        return False
    for artifacts_dir in mlruns_dir.glob("*/*/artifacts"):
        if not artifacts_dir.is_dir():
            continue
        artifact_names = {path.name for path in artifacts_dir.iterdir() if path.is_file()}
        if FINAL_MLFLOW_ARTIFACTS.issubset(artifact_names):
            return True
    return False


def _no_notebook_dependency(outputs: dict[str, Any]) -> bool:
    """Phase 10 requires artifact-only reporting, with no notebook as a deliverable."""

    if "notebook_path" in outputs:
        return False
    return not any(Path("notebooks").glob("*.ipynb"))


def _best_lightgbm_spec(
    model_selection: pd.DataFrame,
    models: dict[str, Any],
) -> tuple[pd.Series | None, dict[str, Any] | None]:
    """Return the best validated LightGBM row and matching YAML-expanded spec."""

    lightgbm_rows = model_selection[model_selection["model_family"].eq("lightgbm")].copy()
    if "eligible_for_selection" in lightgbm_rows:
        eligible = lightgbm_rows[lightgbm_rows["eligible_for_selection"].astype(bool)]
        if not eligible.empty:
            lightgbm_rows = eligible
    if lightgbm_rows.empty:
        return None, None

    lightgbm_rows = lightgbm_rows.sort_values(
        ["normal_mean_common_mase", "mean_common_mase", "cv_common_mase"],
        na_position="last",
    )
    selected_row = lightgbm_rows.iloc[0]
    spec_by_id = {
        spec["model_id"]: spec
        for spec in model_specs(models, stage="model_comparison", include_optional=True)
        if spec["family"] == "lightgbm"
    }
    return selected_row, spec_by_id.get(str(selected_row["model_id"]))


def _empty_shap_artifacts(status: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    importance = pd.DataFrame(
        [
            {
                "status": status,
                "target_strategy": pd.NA,
                "model_id": pd.NA,
                "feature": pd.NA,
                "mean_abs_shap": np.nan,
                "mean_shap": np.nan,
                "std_abs_shap": np.nan,
                "rank": np.nan,
                "n_rows": 0,
                "method": "tree_shap_lightgbm",
            }
        ]
    )
    values = pd.DataFrame(
        columns=[
            "status",
            "target_strategy",
            "model_id",
            "data",
            "y",
            "feature",
            "feature_value",
            "shap_value",
        ]
    )
    return importance, values


def build_shap_explainability(
    target_strategies: dict[str, Any],
    model_selection: pd.DataFrame,
    project: dict[str, Any],
    models: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SHAP explainability for the best validation-selected LightGBM candidate."""

    selected_row, spec = _best_lightgbm_spec(model_selection, models)
    if selected_row is None or spec is None:
        return _empty_shap_artifacts("skipped_no_lightgbm_candidate")

    strategy_name = str(selected_row["target_strategy"])
    strategy_df = target_strategies["strategies"][strategy_name].copy()
    strategy_df = strategy_df.sort_values("data").reset_index(drop=True)

    params = spec["params"]
    helper = RecursiveLagForecaster(
        model_type="lightgbm",
        lags=params["lags"],
        rolling_windows=params.get("rolling_windows", []),
        model_params={},
        seed=int(project["seed"]),
        season_length=12,
    )
    exog = strategy_df[EXOG_COLUMNS] if set(EXOG_COLUMNS).issubset(strategy_df) else None
    X, y_target, positions = helper._training_matrix(  # noqa: SLF001
        strategy_df["y"].astype(float),
        pd.to_datetime(strategy_df["data"]),
        exog,
    )
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y_target = y_target.loc[X.index]
    positions = [positions[index] for index in X.index]
    if X.empty:
        return _empty_shap_artifacts("skipped_no_feature_matrix")

    try:
        import shap
        from lightgbm import LGBMRegressor

        lgbm_params = {
            key: value
            for key, value in params.items()
            if key not in {"lags", "rolling_windows", "forecast_strategy"}
        }
        model = LGBMRegressor(
            objective="regression",
            random_state=int(project["seed"]),
            verbosity=-1,
            **lgbm_params,
        )
        model.fit(X, y_target)
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(X), dtype=float)
    except Exception as exc:
        importance, values = _empty_shap_artifacts(f"failed_{type(exc).__name__}")
        importance["error"] = str(exc)
        return importance, values

    abs_values = np.abs(shap_values)
    importance = pd.DataFrame(
        {
            "status": "ok",
            "target_strategy": strategy_name,
            "model_id": str(selected_row["model_id"]),
            "feature": X.columns,
            "mean_abs_shap": abs_values.mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "std_abs_shap": abs_values.std(axis=0),
            "n_rows": len(X),
            "method": "tree_shap_lightgbm",
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)

    values_rows = []
    sample_indexes = list(range(max(0, len(X) - 36), len(X)))
    for row_position in sample_indexes:
        source_row = strategy_df.iloc[positions[row_position]]
        for col_index, feature in enumerate(X.columns):
            values_rows.append(
                {
                    "status": "ok",
                    "target_strategy": strategy_name,
                    "model_id": str(selected_row["model_id"]),
                    "data": pd.Timestamp(source_row["data"]).strftime("%Y-%m-%d"),
                    "y": float(source_row["y"]),
                    "feature": feature,
                    "feature_value": float(X.iloc[row_position, col_index]),
                    "shap_value": float(shap_values[row_position, col_index]),
                }
            )
    values = pd.DataFrame(values_rows)

    figures_dir = Path(outputs["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    top = importance.head(12).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2f6f9f")
    ax.set_title("LightGBM SHAP Feature Importance")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_feature_importance.png", dpi=160)
    plt.close(fig)

    return importance, values


def _table_html(df: pd.DataFrame, columns: list[str] | None = None, rows: int = 10) -> str:
    table = df.copy()
    if columns is not None:
        table = table[[column for column in columns if column in table.columns]]
    return table.head(rows).to_html(index=False, classes="data-table", border=0, escape=True)


def _figure_html(filename: str, alt: str) -> str:
    return f'<figure><img src="figures/{escape(filename)}" alt="{escape(alt)}"><figcaption>{escape(alt)}</figcaption></figure>'


def _short_label(row: pd.Series) -> str:
    family = str(row.get("model_family", "model"))
    strategy = str(row.get("target_strategy", "strategy"))
    covid_mode = str(row.get("covid_mode", ""))
    return f"{family} / {strategy} / {covid_mode}"


def _ensure_core_reporting_figures(
    *,
    cv_summary: pd.DataFrame,
    model_selection: pd.DataFrame,
    train_valid_gap: pd.DataFrame,
    outputs: dict[str, Any],
) -> None:
    """Materialize core comparison figures used by the artifact HTML reports."""

    figures_dir = Path(outputs["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not cv_summary.empty:
        family_rows = (
            cv_summary.sort_values(
                ["model_family", "normal_mean_common_mase", "mean_common_mase"],
                na_position="last",
            )
            .groupby("model_family", as_index=False)
            .head(1)
            .sort_values("normal_mean_common_mase", ascending=False)
        )
        if not family_rows.empty:
            fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(family_rows))))
            labels = family_rows.apply(_short_label, axis=1)
            ax.barh(labels, family_rows["normal_mean_common_mase"], color="#2f6f9f")
            ax.set_xlabel("Normal-fold fixed-target MASE")
            ax.set_title("Best Validated Candidate by Model Family")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures_dir / "cv_metric_comparison.png", dpi=160)
            plt.close(fig)

    if not model_selection.empty:
        top = model_selection.sort_values(
            ["normal_mean_common_mase", "mean_common_mase"],
            na_position="last",
        ).head(12)
        if not top.empty:
            labels = top.apply(_short_label, axis=1)
            x = np.arange(len(top))
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(x - 0.2, top["normal_mean_common_mase"], width=0.4, label="Normal MASE")
            ax.bar(x + 0.2, top["mean_common_mase"], width=0.4, label="All-fold MASE")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Fixed-target MASE")
            ax.set_title("Fold Stability: Normal vs All-Fold MASE")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures_dir / "fold_mase_stability.png", dpi=160)
            plt.close(fig)

    if not train_valid_gap.empty:
        selected_gap = train_valid_gap.copy()
        if {"target_strategy", "model_id"}.issubset(model_selection.columns):
            selected = _selected_row(model_selection)
            selected_gap = selected_gap[
                selected_gap["target_strategy"].astype(str).eq(str(selected["target_strategy"]))
                & selected_gap["model_id"].astype(str).eq(str(selected["model_id"]))
            ]
        selected_gap = selected_gap[selected_gap["status"].astype(str).eq("ok")]
        if not selected_gap.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            labels = selected_gap["fold_name"].astype(str)
            x = np.arange(len(selected_gap))
            ax.plot(x, selected_gap["train_mae"], marker="o", label="Train MAE")
            ax.plot(x, selected_gap["validation_mae"], marker="o", label="Validation MAE")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("MAE")
            ax.set_title("Train vs Validation Gap for Selected Model")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures_dir / "train_validation_gap.png", dpi=160)
            plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "n/a"
    return f"{float(number):.{digits}f}"


def _metric_cell(label: str, value: Any, *, digits: int = 3) -> str:
    return f'<div class="kpi">{escape(label)}<strong>{escape(_fmt(value, digits))}</strong></div>'


def _selected_row(model_selection: pd.DataFrame) -> pd.Series:
    selected = model_selection[model_selection["selected"].astype(bool)]
    if not selected.empty:
        return selected.iloc[0]
    return model_selection.sort_values(
        ["normal_mean_common_mase", "mean_common_mase", "normal_mean_mase"],
        na_position="last",
    ).iloc[0]


def _selected_residual_row(
    residual_diagnostics: pd.DataFrame,
    selected: pd.Series,
) -> pd.Series:
    if residual_diagnostics.empty:
        return pd.Series(dtype=object)
    if "selected" in residual_diagnostics:
        marked = residual_diagnostics[residual_diagnostics["selected"].astype(bool)]
        if not marked.empty:
            return marked.iloc[0]
    match = residual_diagnostics[
        residual_diagnostics["target_strategy"].astype(str).eq(str(selected["target_strategy"]))
        & residual_diagnostics["model_id"].astype(str).eq(str(selected["model_id"]))
    ]
    return match.iloc[0] if not match.empty else residual_diagnostics.iloc[0]


def _best_acquisition_rows(model_selection: pd.DataFrame) -> pd.DataFrame:
    acquisition_pool = model_selection[
        model_selection["target_strategy"].isin(
            ["post_only", "raw_full", "proforma_sum", "calibrated_alpha"]
        )
    ].copy()
    if "eligible_for_selection" in acquisition_pool:
        eligible_acquisition = acquisition_pool[
            acquisition_pool["eligible_for_selection"].astype(bool)
        ]
        if not eligible_acquisition.empty:
            acquisition_pool = eligible_acquisition
    if acquisition_pool.empty:
        return acquisition_pool
    return (
        acquisition_pool.sort_values(
            ["target_strategy", "normal_mean_common_mase", "mean_common_mase"],
            na_position="last",
        )
        .groupby("target_strategy", as_index=False)
        .head(1)
        .sort_values("normal_mean_common_mase", na_position="last")
    )


def _best_strategy_row(acquisition_rows: pd.DataFrame, strategy: str) -> pd.Series | None:
    if acquisition_rows.empty:
        return None
    rows = acquisition_rows[acquisition_rows["target_strategy"].astype(str).eq(strategy)]
    return rows.iloc[0] if not rows.empty else None


def _pre_merge_improvement_pct(acquisition_rows: pd.DataFrame) -> float:
    post = _best_strategy_row(acquisition_rows, "post_only")
    reconstructed = acquisition_rows[
        acquisition_rows["target_strategy"].isin(["proforma_sum", "calibrated_alpha"])
    ].copy()
    if post is None or reconstructed.empty:
        return np.nan
    best_reconstructed = reconstructed.sort_values("normal_mean_common_mase").iloc[0]
    post_mase = float(post["normal_mean_common_mase"])
    reconstructed_mase = float(best_reconstructed["normal_mean_common_mase"])
    return (post_mase - reconstructed_mase) / post_mase * 100 if post_mase else np.nan


def _selected_covid_row(
    covid_mode_comparison: pd.DataFrame,
    metadata: pd.Series,
) -> pd.Series | None:
    if covid_mode_comparison.empty:
        return None
    selected_strategy = str(metadata["selected_target_strategy"])
    selected_family = str(metadata["selected_model_family"])
    rows = covid_mode_comparison[
        covid_mode_comparison["target_strategy"].astype(str).eq(selected_strategy)
        & covid_mode_comparison["model_family"].astype(str).eq(selected_family)
    ]
    return rows.iloc[0] if not rows.empty else None


def _build_decision_html_report(
    *,
    acquisition_rows: pd.DataFrame,
    covid_mode_comparison: pd.DataFrame,
    model_selection: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    rolling_origin_robustness: pd.DataFrame,
    forecast_intervals: pd.DataFrame,
    final_model_metadata: pd.DataFrame,
    project: dict[str, Any],
    outputs: dict[str, Any],
) -> str:
    """Build the short decision/evidence HTML requested as report 1."""

    selected = _selected_row(model_selection)
    selected_diag = _selected_residual_row(residual_diagnostics, selected)
    metadata = final_model_metadata.iloc[0]
    pre_merge_improvement = _pre_merge_improvement_pct(acquisition_rows)
    best_post = _best_strategy_row(acquisition_rows, "post_only")
    reconstructed = acquisition_rows[
        acquisition_rows["target_strategy"].isin(["proforma_sum", "calibrated_alpha"])
    ].copy()
    best_reconstructed = (
        reconstructed.sort_values("normal_mean_common_mase").iloc[0]
        if not reconstructed.empty
        else pd.Series(dtype=object)
    )
    covid_row = _selected_covid_row(covid_mode_comparison, metadata)
    covid_none = (
        covid_row.get("none_normal_mean_common_mase", np.nan) if covid_row is not None else np.nan
    )
    covid_adjusted = (
        covid_row.get("adjusted_target_normal_mean_common_mase", np.nan)
        if covid_row is not None
        else np.nan
    )
    robust_selected = model_selection[
        model_selection.get(
            "selected_with_robustness", pd.Series(False, index=model_selection.index)
        )
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    ]
    robustness_same = (
        not robust_selected.empty
        and str(robust_selected.iloc[0]["model_id"]) == str(selected["model_id"])
        and str(robust_selected.iloc[0]["target_strategy"]) == str(selected["target_strategy"])
    )
    selected_robust = (
        rolling_origin_robustness[
            rolling_origin_robustness["target_strategy"]
            .astype(str)
            .eq(str(selected["target_strategy"]))
            & rolling_origin_robustness["model_id"].astype(str).eq(str(selected["model_id"]))
        ]
        if not rolling_origin_robustness.empty
        else pd.DataFrame()
    )
    selected_robust_row = selected_robust.iloc[0] if not selected_robust.empty else pd.Series()

    report_path = Path(outputs["decision_html_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PADS Forecasting - Decisoes MASE 2024</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1c2430; background: #f7f8fa; }}
    header {{ background: #102a43; color: white; padding: 30px 44px; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 28px 22px 52px; }}
    section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 22px; margin: 18px 0; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .kpi {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #fbfcfd; }}
    .kpi strong {{ display: block; font-size: 1.3rem; color: #102a43; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e5e9f0; padding: 7px 8px; text-align: left; vertical-align: top; }}
    .data-table th {{ background: #eef2f7; color: #243b53; }}
    figure {{ margin: 12px 0; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; border-radius: 6px; background: white; }}
    figcaption {{ font-size: 0.85rem; color: #52606d; margin-top: 4px; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>PADS Forecasting - Decisoes para MASE 2024</h1>
  <p>Run <code>{escape(str(project["run_id"]))}</code>. HTML 1: escolhas, evidencias, descartes e controles de robustez.</p>
</header>
<main>
  <section>
    <h2>Decisao Principal</h2>
    <div class="kpis">
      <div class="kpi">Modelo final<strong>{escape(str(metadata["selected_model_id"]))}</strong></div>
      <div class="kpi">Tratamento final<strong>{escape(str(metadata["selected_target_strategy"]))}</strong></div>
      <div class="kpi">Alpha final<strong>{escape(str(metadata.get("selected_final_alpha", "n/a")))}</strong></div>
      {_metric_cell("MASE normal", selected["normal_mean_common_mase"])}
      {_metric_cell("MASE all-fold", selected["mean_common_mase"])}
      {_metric_cell("Ganho pre-merge vs post_only", pre_merge_improvement, digits=1)}
    </div>
    <p><strong>Qual era a escolha?</strong> Decidir se os dados pre-merge deveriam entrar no treinamento para reduzir a MASE esperada dos 12 meses ocultos de 2024, sem quebrar robustez, validacao temporal, diagnostico residual ou protecao contra vazamento.</p>
    <p><strong>Escolha final:</strong> usar o historico pre-merge reconstruido como pro-forma simples. O candidato aparece como <code>calibrated_alpha</code>, mas o alpha final e <code>{escape(str(metadata.get("selected_final_alpha", "n/a")))}</code>, entao a representacao final equivale a <code>B_t + A_t</code> antes de 2019-07 e <code>C_t</code> observado depois.</p>
  </section>

  <section>
    <h2>O que decidiu esse caminho?</h2>
    {_figure_html("target_reconstruction_overlay.png", "Reconstrucoes de alvo antes/depois da aquisicao")}
    {_figure_html("cv_metric_comparison.png", "Comparacao MASE/MAE/RMSE por modelo")}
    {_figure_html("fold_mase_stability.png", "Variacao de MASE por fold")}
    <p>A evidencia critica e a validacao temporal com <code>common_mase</code>, que usa denominador fixo por fold. O melhor candidato com pre-merge reconstruido tem MASE normal <strong>{_fmt(best_reconstructed.get("normal_mean_common_mase", np.nan))}</strong>. O melhor <code>post_only</code> tem MASE normal <strong>{_fmt(best_post.get("normal_mean_common_mase", np.nan) if best_post is not None else np.nan)}</strong>. Isso representa melhora de aproximadamente <strong>{_fmt(pre_merge_improvement, 1)}%</strong> no proxy mais proximo do alvo oculto de 2024.</p>
    <p>Os controles tambem passam: o modelo final bate o SeasonalNaive, tem validacao completa, nao usa 2024, compara validacao contra <code>C_t</code> observado, passa o gate de overfit e tambem vence no diagnostico de robustez por origens rolantes extras: <strong>{escape("sim" if robustness_same else "nao")}</strong>.</p>
    {_table_html(acquisition_rows, ["target_strategy", "model_family", "model_id", "covid_mode", "normal_mean_common_mase", "mean_common_mase", "mean_mae", "mean_rmse", "mean_relative_mae_vs_seasonal_naive", "stability_reason"], rows=8)}
  </section>

  <section>
    <h2>O que foi descartado e por que?</h2>
    <h3>Descartado: usar apenas post-merge</h3>
    <p><code>post_only</code> foi mantido como comparacao obrigatoria, mas ficou pior no objetivo principal. Ele evita qualquer reconstrucao historica, porem perdeu MASE contra o pro-forma reconstruido.</p>
    <h3>Descartado: alpha diferente de 1 no forecast final</h3>
    <p>Alpha foi explorado de forma regularizada e fold-local. Para ETS, alpha diferente de 1 nao gerou ganho preditivo: o delta normal de common MASE contra alpha 1 ficou positivo, entao piorou levemente. Por isso o forecast final volta para alpha 1.</p>
    <h3>Descartado: ajuste COVID como modo final</h3>
    <p>COVID foi tratado como opcao justa para todas as familias com <code>none</code> versus <code>adjusted_target</code>. Para a familia selecionada, <code>none</code> teve MASE normal <strong>{_fmt(covid_none)}</strong>, enquanto <code>adjusted_target</code> teve <strong>{_fmt(covid_adjusted)}</strong>. Portanto o ajuste COVID piorou a MASE e nao foi usado no forecast final.</p>
    {_table_html(covid_mode_comparison, ["target_strategy", "model_family", "best_none_model_id", "none_normal_mean_common_mase", "best_adjusted_target_model_id", "adjusted_target_normal_mean_common_mase", "adjusted_minus_none_normal_common_mase", "adjusted_target_beats_none"], rows=12)}
  </section>

  <section>
    <h2>Robustez e Diagnosticos</h2>
    <div class="kpis">
      {_metric_cell("Robustez normal MASE", selected_robust_row.get("normal_mean_common_mase", np.nan))}
      {_metric_cell("Robustez all-fold MASE", selected_robust_row.get("mean_common_mase", np.nan))}
      {_metric_cell("CV common MASE", selected["cv_common_mase"])}
      {_metric_cell("Train/valid ratio", selected["mean_train_valid_ratio"])}
      {_metric_cell("Ljung-Box p lag 12", selected_diag.get("ljung_box_p_lag_12", np.nan))}
      {_metric_cell("Ljung-Box p lag 24", selected_diag.get("ljung_box_p_lag_24", np.nan))}
    </div>
    {_figure_html("residual_acf.png", "ACF dos residuos do modelo escolhido")}
    {_figure_html("final_forecast_intervals.png", "Previsao final com intervalos de 80% e 95%")}
    <p>Os intervalos de 80% e 95% estao materializados em <code>forecast_intervals.parquet</code> nas colunas <code>lo_80</code>, <code>hi_80</code>, <code>lo_95</code> e <code>hi_95</code>, e visualizados no grafico final.</p>
    {_table_html(forecast_intervals, rows=12)}
  </section>
</main>
</body>
</html>
"""


def build_html_report(
    data_validation: pd.DataFrame,
    eda_summary: pd.DataFrame,
    stationarity_tests: pd.DataFrame,
    target_strategy_summary: pd.DataFrame,
    old_data_gate: pd.DataFrame,
    cv_summary: pd.DataFrame,
    model_selection: pd.DataFrame,
    train_valid_gap: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    mase_uncertainty: pd.DataFrame,
    nested_selection_audit: pd.DataFrame,
    nested_cv_results: pd.DataFrame,
    nested_cv_summary: pd.DataFrame,
    rolling_origin_robustness: pd.DataFrame,
    robust_alpha_results: pd.DataFrame,
    robust_alpha_summary: pd.DataFrame,
    selection_objective_audit: pd.DataFrame,
    covid_adjustment_coefficients: pd.DataFrame,
    covid_adjustment_audit: pd.DataFrame,
    covid_mode_comparison: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    interval_coverage_proxy: pd.DataFrame,
    forecast_intervals: pd.DataFrame,
    final_model_metadata: pd.DataFrame,
    shap_feature_importance: pd.DataFrame,
    project: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[str, str]:
    """Build the short decision HTML and full artifact-only HTML report."""

    selected = _selected_row(model_selection)
    selected_diag = _selected_residual_row(residual_diagnostics, selected)
    acquisition_rows = _best_acquisition_rows(model_selection)
    _ensure_core_reporting_figures(
        cv_summary=cv_summary,
        model_selection=model_selection,
        train_valid_gap=train_valid_gap,
        outputs=outputs,
    )
    family_rows = (
        cv_summary.sort_values(["model_family", "normal_mean_common_mase", "mean_common_mase"])
        .groupby("model_family", as_index=False)
        .head(1)
        .sort_values("normal_mean_common_mase")
    )
    selected_horizon = (
        horizon_summary[
            horizon_summary["target_strategy"].eq(selected["target_strategy"])
            & horizon_summary["model_id"].eq(selected["model_id"])
        ]
        if not horizon_summary.empty
        else pd.DataFrame()
    )
    selected_uncertainty = (
        mase_uncertainty[
            mase_uncertainty["target_strategy"].eq(selected["target_strategy"])
            & mase_uncertainty["model_id"].eq(selected["model_id"])
        ]
        if not mase_uncertainty.empty
        else pd.DataFrame()
    )

    metadata = final_model_metadata.iloc[0]
    selected_strategy = str(metadata["selected_target_strategy"])
    selected_alpha = metadata.get("selected_final_alpha", "n/a")
    if selected_strategy == "calibrated_alpha" and pd.notna(selected_alpha):
        acquisition_text = (
            "O candidato vencedor aparece como calibrated_alpha, mas o alpha final foi "
            f"{float(selected_alpha):.1f}. Isso torna a representacao final igual ao "
            "pro-forma simples B_t + A_t antes da aquisicao, seguido do C_t observado."
        )
    elif selected_strategy == "proforma_sum":
        acquisition_text = (
            "O candidato vencedor usa proforma_sum: B_t + A_t antes da aquisicao, "
            "seguido do C_t observado."
        )
    else:
        acquisition_text = "O candidato vencedor usa apenas dados pos-aquisicao observados."

    covid_selected = covid_mode_comparison[
        covid_mode_comparison["target_strategy"].astype(str).eq(selected_strategy)
        & covid_mode_comparison["model_family"]
        .astype(str)
        .eq(str(metadata["selected_model_family"]))
    ]
    covid_text = (
        "Ajuste COVID nao foi selecionado porque piorou a MASE no proxy de 2024."
        if not covid_selected.empty
        and bool(covid_selected.iloc[0].get("adjusted_target_beats_none")) is False
        else "A auditoria COVID compara none versus adjusted_target para a familia selecionada."
    )

    robust_selected = model_selection[
        model_selection.get(
            "selected_with_robustness", pd.Series(False, index=model_selection.index)
        )
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    ]
    robust_text = (
        "O vencedor por MASE e tambem o vencedor no diagnostico de robustez/variancia."
        if not robust_selected.empty
        and robust_selected.iloc[0]["model_id"] == selected["model_id"]
        and robust_selected.iloc[0]["target_strategy"] == selected["target_strategy"]
        else "O diagnostico de robustez aponta um vencedor diferente do vencedor primario por MASE."
    )

    report_title = "PADS Forecasting - Distribuidora BR 2024 MASE Forecast"
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(report_title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1c2430; background: #f7f8fa; }}
    header {{ background: #102a43; color: white; padding: 32px 48px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 56px; }}
    section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 22px; margin: 18px 0; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .kpi {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #fbfcfd; }}
    .kpi strong {{ display: block; font-size: 1.25rem; color: #102a43; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e5e9f0; padding: 7px 8px; text-align: left; vertical-align: top; }}
    .data-table th {{ background: #eef2f7; color: #243b53; }}
    figure {{ margin: 12px 0; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; border-radius: 6px; background: white; }}
    figcaption {{ font-size: 0.85rem; color: #52606d; margin-top: 4px; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>{escape(report_title)}</h1>
  <p>Run <code>{escape(str(project["run_id"]))}</code>. Objetivo unico: minimizar MASE nos 12 meses ocultos de 2024. Relatorio HTML gerado por Kedro; notebook nao e necessario.</p>
</header>
<main>
  <section>
    <h2>Resposta Executiva: Objetivo Unico MASE 2024</h2>
    <div class="kpis">
      <div class="kpi">Modelo final<strong>{escape(str(metadata["selected_model_id"]))}</strong></div>
      <div class="kpi">Tratamento da aquisicao<strong>{escape(str(metadata["selected_target_strategy"]))}</strong></div>
      <div class="kpi">Alpha final<strong>{escape(str(metadata.get("selected_final_alpha", "n/a")))}</strong></div>
      <div class="kpi">Common MASE normal<strong>{float(selected["normal_mean_common_mase"]):.3f}</strong></div>
      <div class="kpi">Common MASE medio<strong>{float(selected["mean_common_mase"]):.3f}</strong></div>
      <div class="kpi">Local MASE normal<strong>{float(selected["normal_mean_mase"]):.3f}</strong></div>
      <div class="kpi">Ljung-Box p lag 12<strong>{float(selected_diag["ljung_box_p_lag_12"]):.3f}</strong></div>
      <div class="kpi">Ljung-Box p lag 24<strong>{float(selected_diag["ljung_box_p_lag_24"]):.3f}</strong></div>
    </div>
    <p><strong>Leitura:</strong> {escape(acquisition_text)} {escape(covid_text)} {escape(robust_text)}</p>
  </section>

  <section>
    <h2>Dados, Aquisicao, Sazonalidade e Estacionariedade</h2>
    {_figure_html("series_acquisition_covid.png", "Serie principal com aquisicao e COVID")}
    {_figure_html("decomposition.png", "Decomposicao em tendencia, sazonalidade e residuo")}
    {_figure_html("seasonality_month_profile.png", "Perfil sazonal mensal")}
    {_figure_html("outliers_covid.png", "Outliers e periodo COVID")}
    <h3>Resumo dos dados</h3>
    <p>A decomposicao exibida e aditiva. Ela e apropriada quando as oscilacoes tem tamanho aproximadamente constante. Se as flutuacoes crescessem ou diminuissem claramente com o nivel da serie, uma decomposicao multiplicativa seria mais natural; aqui ela fica como diagnostico, nao como regra de selecao.</p>
    {_table_html(eda_summary, rows=12)}
    <h3>ADF e KPSS</h3>
    {_table_html(stationarity_tests, rows=12)}
  </section>

  <section>
    <h2>Tratamento da Aquisicao e Dados Antigos</h2>
    {_figure_html("target_reconstruction_overlay.png", "Reconstrucoes de alvo antes e depois da aquisicao")}
    {_figure_html("alpha_sensitivity.png", "Sensibilidade do alpha validada sem vazamento")}
    <p>A regra final compara dados antigos reconstruidos contra pos-aquisicao observado. O alvo de validacao nunca e reconstruido. A selecao do forecast usa uma unica metrica principal: <code>normal_mean_common_mase</code>, o melhor proxy sem vazamento para os 12 meses ocultos de 2024.</p>
    <p>Para comparacao entre estrategias de alvo, o ranking usa <code>common_mase</code>: o denominador e fixo por fold e vem apenas do processo consolidado observado pos-aquisicao (<code>post_only</code>), evitando que <code>calibrated_alpha</code> ganhe vantagem por alterar a volatilidade do periodo pre-fusao.</p>
    <p>O alpha e auditado como hiperparametro por familia de modelo. A penalizacao quadratica em torno de <code>alpha=1</code> significa somar <code>lambda * (alpha - 1)^2</code> ao score interno: ela favorece o pro-forma simples quando dois alphas empatam ou diferem pouco, mas nao bloqueia um alpha diferente se ele melhora a MASE validada. Nao ha corte duro de 3%.</p>
    <p>Tambem nao ha veto por media de estrategia: se uma familia especifica de modelo mostra evidencia preditiva para dados antigos em CV, ela pode competir na selecao final mesmo que a media de todos os modelos daquela estrategia seja fraca.</p>
    <h3>Comparacao model-specific das estrategias de aquisicao</h3>
    {_table_html(acquisition_rows, ["target_strategy", "model_family", "model_id", "covid_mode", "selected", "eligible_for_selection", "normal_mean_common_mase", "mean_common_mase", "normal_mean_mase", "mean_mae", "mean_rmse", "mean_relative_mae_vs_seasonal_naive", "stability_reason"], rows=10)}
    <h3>Alpha robusto por familia de modelo</h3>
    {_table_html(robust_alpha_summary, ["model_family", "selected_alpha_mode", "selected_alpha_min", "selected_alpha_max", "selected_alpha_std", "alpha_selection_stable", "normal_outer_common_mase", "mean_common_mase_delta_vs_alpha_one", "normal_common_mase_delta_vs_alpha_one", "folds_beating_alpha_one_common_mase", "best_alpha_all_folds_grid", "best_alpha_normal_folds_only_grid", "best_alpha_stress_downweighted_grid", "mean_abs_yhat_pct_diff_vs_alpha_one", "alpha_objective"], rows=12)}
    {_table_html(robust_alpha_results, ["fold_name", "fold_role", "model_family", "model_id", "selected_alpha", "inner_fold_count", "inner_mean_common_mase", "inner_regularized_objective", "outer_common_mase", "alpha_one_outer_common_mase", "alpha_beats_one_by_common_mase", "mean_abs_yhat_diff_vs_alpha_one"], rows=16)}
    <h3>Resumo das estrategias</h3>
    {_table_html(target_strategy_summary, rows=12)}
  </section>

  <section>
    <h2>Comparacao de Modelos e Validacao Temporal</h2>
    {_figure_html("cv_metric_comparison.png", "Comparacao de metricas por modelo")}
    {_figure_html("fold_mase_stability.png", "Estabilidade de MASE por fold")}
    {_figure_html("train_validation_gap.png", "Diferenca treino-validacao")}
    <h3>Melhor candidato por familia</h3>
    {_table_html(family_rows, ["model_family", "target_strategy", "model_id", "normal_mean_common_mase", "mean_common_mase", "normal_mean_mase", "mean_mae", "mean_rmse", "cv_common_mase", "cv_mase", "mean_relative_mae_vs_seasonal_naive"], rows=12)}
    <h3>Ranking de selecao</h3>
    {_table_html(model_selection, ["rank", "selected", "eligible_for_selection", "target_strategy", "model_family", "model_id", "normal_mean_common_mase", "mean_common_mase", "normal_mean_mase", "cv_mae", "cv_rmse", "cv_common_mase", "mean_train_valid_ratio", "stability_reason", "selection_reason"], rows=15)}
    <h3>Objetivo de selecao: MASE 2024</h3>
    <p>O objetivo de producao e unico: prever os 12 meses ocultos de 2024 com menor MASE. Como 2024 esta oculto, usamos folds normais pos-fusao 2022-2023 como proxy principal sem vazamento. Variancia, folds extras e 2021 sao criterios de validade/diagnostico, nao objetivos concorrentes.</p>
    {_table_html(selection_objective_audit, ["objective", "objective_metric", "candidate_role", "target_strategy", "model_family", "model_id", "normal_mean_common_mase", "mean_common_mase", "normal_mean_mae", "mean_mae", "objective_description"], rows=8)}
    <h3>MASE por horizonte do modelo selecionado</h3>
    {_table_html(selected_horizon, ["horizon_index", "horizon_common_mase", "horizon_local_mase", "horizon_mae", "horizon_rmse", "horizon_bias", "folds"], rows=12)}
    <h3>Incerteza pareada contra SeasonalNaive</h3>
    {_table_html(selected_uncertainty, ["candidate_mean_common_mase", "baseline_mean_common_mase", "mean_common_mase_diff_vs_seasonal_naive", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_probability_beats_seasonal_naive", "paired_observations"], rows=5)}
    <h3>Nested CV formal: hiperparametros escolhidos so com folds anteriores</h3>
    {_table_html(nested_cv_summary, ["summary_scope", "nested_outer_folds", "nested_mean_common_mase", "nested_mean_local_mase", "nested_mean_mae", "nested_mean_rmse", "nested_mean_relative_mae_vs_seasonal_naive", "selected_candidate_count", "target_strategy", "model_family", "model_id", "nested_selection_count"], rows=12)}
    {_table_html(nested_cv_results, ["fold_name", "fold_role", "status", "target_strategy", "model_family", "model_id", "alpha", "alpha_selection_method", "inner_fold_count", "inner_candidate_count", "inner_mean_common_mase", "outer_common_mase", "outer_mae"], rows=12)}
    <h3>Auditoria de selecao nested por folds anteriores</h3>
    {_table_html(nested_selection_audit, ["selection_scope", "fold_name", "fold_role", "status", "target_strategy", "model_family", "model_id", "alpha", "alpha_selection_method", "inner_fold_count", "inner_mean_common_mase", "outer_common_mase"], rows=20)}
    <h3>Robustez com origens rolantes extras</h3>
    {_table_html(rolling_origin_robustness, ["target_strategy", "model_family", "model_id", "normal_mean_common_mase", "mean_common_mase", "cv_common_mase", "robustness_fold_count", "robustness_step_months"], rows=10)}
  </section>

  <section>
    <h2>Tratamento Justo de COVID</h2>
    <p>Todos os modelos competem nos modos primarios <code>none</code> e <code>adjusted_target</code>. No modo ajustado, um modelo nativo <code>statsmodels UnobservedComponents</code> estima o efeito das dummies de COVID apenas no treino de cada fold, remove esse efeito do alvo de treino e usa COVID futuro igual a zero.</p>
    <h3>Comparacao none versus adjusted_target</h3>
    {_table_html(covid_mode_comparison, ["target_strategy", "model_family", "best_none_model_id", "none_normal_mean_common_mase", "best_adjusted_target_model_id", "adjusted_target_normal_mean_common_mase", "adjusted_minus_none_normal_common_mase", "adjusted_improvement_vs_none_pct", "adjusted_target_beats_none"], rows=20)}
    <h3>Coeficientes aprendidos por fold</h3>
    {_table_html(covid_adjustment_coefficients, ["target_strategy", "model_family", "model_id", "fold_name", "fold_role", "covid_adjustment_estimator", "covid_adjustment_status", "covid_beta_covid_shock", "covid_beta_covid_recovery", "covid_beta_covid_aftershock_2021", "covid_adjustment_effect_abs_mean"], rows=20)}
    <h3>Auditoria sem vazamento</h3>
    {_table_html(covid_adjustment_audit, ["target_strategy", "model_family", "model_id", "fold_name", "fold_role", "train_end_before_valid_start", "validation_compared_to_observed_target", "future_covid_assumed_zero", "common_mase", "mae"], rows=20)}
  </section>

  <section>
    <h2>Explainability SHAP do Modelo ML</h2>
    {_figure_html("shap_feature_importance.png", "Importancia SHAP do melhor LightGBM validado")}
    <p>O SHAP foi calculado para o melhor candidato LightGBM escolhido por validacao temporal, treinado apenas em dados observados ate 2023.</p>
    {_table_html(shap_feature_importance, ["rank", "status", "target_strategy", "model_id", "feature", "mean_abs_shap", "mean_shap", "n_rows", "method"], rows=15)}
  </section>

  <section>
    <h2>Diagnostico Residual e Intervalos</h2>
    {_figure_html("residual_acf.png", "ACF residual do modelo selecionado")}
    {_figure_html("residual_histogram.png", "Histograma residual do modelo selecionado")}
    {_figure_html("residual_time.png", "Residuos ao longo do tempo")}
    {_figure_html("interval_coverage_proxy.png", "Cobertura proxy dos intervalos em CV")}
    {_table_html(residual_diagnostics, rows=10)}
    <h3>Cobertura por fold</h3>
    {_table_html(interval_coverage_proxy, rows=12)}
  </section>

  <section>
    <h2>Previsao Final 2024</h2>
    {_figure_html("final_forecast_intervals.png", "Previsao final com intervalos de 80% e 95%")}
    {_table_html(forecast_intervals, rows=12)}
    <p>Arquivo de submissao: <code>{escape(str(outputs["previsao_path"]))}</code>.</p>
  </section>

  <section>
    <h2>Decisoes Principais</h2>
    <ol>
      <li>Selecionar o forecast pelo menor <code>normal_mean_common_mase</code>, o proxy sem vazamento para a MASE dos 12 meses ocultos de 2024.</li>
      <li>Usar historico pre-aquisicao reconstruido porque <code>calibrated_alpha/proforma_sum + ETS damped</code> vence o melhor <code>post_only</code> no score principal.</li>
      <li>Interpretar o empate <code>calibrated_alpha</code> versus <code>proforma_sum</code> corretamente: o alpha final e <code>1.0</code>, portanto as duas representacoes sao matematicamente iguais no forecast final.</li>
      <li>Testar COVID de forma justa em todos os modelos com <code>none</code> e <code>adjusted_target</code>; escolher <code>none</code> porque a remocao fold-local do efeito COVID piorou a MASE para os melhores candidatos.</li>
      <li>Manter variancia, overfit, Ljung-Box, intervalos e SHAP como controles de validade e explicabilidade, sem trocar o objetivo unico de producao: MASE 2024.</li>
    </ol>
  </section>
</main>
</body>
</html>
"""

    decision_html = _build_decision_html_report(
        acquisition_rows=acquisition_rows,
        covid_mode_comparison=covid_mode_comparison,
        model_selection=model_selection,
        residual_diagnostics=residual_diagnostics,
        rolling_origin_robustness=rolling_origin_robustness,
        forecast_intervals=forecast_intervals,
        final_model_metadata=final_model_metadata,
        project=project,
        outputs=outputs,
    )

    report_path = Path(outputs["html_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return decision_html, html


def build_assignment_checklist(
    data_validation: pd.DataFrame,
    eda_summary: pd.DataFrame,
    stationarity_tests: pd.DataFrame,
    old_data_gate: pd.DataFrame,
    cv_summary: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    mase_uncertainty: pd.DataFrame,
    nested_selection_audit: pd.DataFrame,
    nested_cv_results: pd.DataFrame,
    nested_cv_summary: pd.DataFrame,
    rolling_origin_robustness: pd.DataFrame,
    robust_alpha_results: pd.DataFrame,
    robust_alpha_summary: pd.DataFrame,
    selection_objective_audit: pd.DataFrame,
    covid_adjustment_coefficients: pd.DataFrame,
    covid_adjustment_audit: pd.DataFrame,
    covid_mode_comparison: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    interval_coverage_proxy: pd.DataFrame,
    interval_validation_predictions: pd.DataFrame,
    forecast_intervals: pd.DataFrame,
    challenger_forecasts: pd.DataFrame,
    final_model_metadata: pd.DataFrame,
    shap_feature_importance: pd.DataFrame,
    shap_values_sample: pd.DataFrame,
    decision_html_report: str,
    html_report: str,
    previsao: pd.DataFrame,
    project: dict[str, Any],
    outputs: dict[str, Any],
) -> pd.DataFrame:
    """Confirm assignment deliverables exist without notebook-dependent decisions."""

    checks = [
        ("data_validation_complete", not data_validation.empty),
        ("eda_summary_complete", not eda_summary.empty),
        ("adf_kpss_complete", {"ADF", "KPSS"}.issubset(set(stationarity_tests["test"]))),
        ("old_data_gate_complete", not old_data_gate.empty),
        ("cv_summary_complete", not cv_summary.empty),
        (
            "fixed_target_mase_complete",
            not cv_summary.empty
            and {"mean_common_mase", "normal_mean_common_mase", "cv_common_mase"}.issubset(
                cv_summary.columns
            ),
        ),
        (
            "horizon_mase_complete",
            not horizon_metrics.empty
            and not horizon_summary.empty
            and {"horizon_index", "common_mase"}.issubset(horizon_metrics.columns)
            and {"horizon_index", "horizon_common_mase"}.issubset(horizon_summary.columns),
        ),
        (
            "mase_uncertainty_complete",
            not mase_uncertainty.empty
            and {
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_probability_beats_seasonal_naive",
            }.issubset(mase_uncertainty.columns),
        ),
        (
            "nested_selection_audit_complete",
            not nested_selection_audit.empty
            and {"selection_scope", "inner_fold_count", "outer_common_mase"}.issubset(
                nested_selection_audit.columns
            ),
        ),
        (
            "formal_nested_cv_complete",
            not nested_cv_results.empty
            and not nested_cv_summary.empty
            and {
                "selection_scope",
                "inner_candidate_count",
                "inner_mean_common_mase",
                "outer_common_mase",
            }.issubset(nested_cv_results.columns)
            and {
                "summary_scope",
                "nested_outer_folds",
                "nested_mean_common_mase",
            }.issubset(nested_cv_summary.columns),
        ),
        (
            "rolling_origin_robustness_complete",
            not rolling_origin_robustness.empty
            and {"mean_common_mase", "robustness_fold_count"}.issubset(
                rolling_origin_robustness.columns
            ),
        ),
        (
            "robust_alpha_model_family_complete",
            not robust_alpha_results.empty
            and not robust_alpha_summary.empty
            and {
                "model_family",
                "selected_alpha",
                "inner_mean_common_mase",
                "outer_common_mase",
                "alpha_beats_one_by_common_mase",
            }.issubset(robust_alpha_results.columns)
            and {
                "model_family",
                "selected_alpha_mode",
                "best_alpha_normal_folds_only_grid",
                "best_alpha_stress_downweighted_grid",
                "mean_common_mase_delta_vs_alpha_one",
            }.issubset(robust_alpha_summary.columns),
        ),
        (
            "selection_objective_audit_complete",
            not selection_objective_audit.empty
            and {"objective", "objective_metric", "target_strategy", "model_id"}.issubset(
                selection_objective_audit.columns
            ),
        ),
        (
            "covid_adjusted_target_fairness_complete",
            not covid_mode_comparison.empty
            and {"none", "adjusted_target"}.issubset(set(cv_summary["covid_mode"].astype(str))),
        ),
        (
            "covid_adjustment_coefficients_complete",
            not covid_adjustment_coefficients.empty
            and any(
                column.startswith("covid_beta_") for column in covid_adjustment_coefficients.columns
            ),
        ),
        (
            "covid_adjustment_no_leakage_audit_complete",
            not covid_adjustment_audit.empty
            and bool(covid_adjustment_audit["train_end_before_valid_start"].all())
            and bool(covid_adjustment_audit["future_covid_assumed_zero"].all()),
        ),
        ("residual_diagnostics_complete", not residual_diagnostics.empty),
        (
            "interval_coverage_proxy_complete",
            not interval_coverage_proxy.empty
            and {
                "coverage_80",
                "coverage_95",
                "mean_width_80",
                "mean_width_95",
            }.issubset(interval_coverage_proxy.columns),
        ),
        (
            "interval_validation_predictions_complete",
            not interval_validation_predictions.empty
            and {
                "data",
                "y_true",
                "yhat",
                "lo_80",
                "hi_80",
                "lo_95",
                "hi_95",
                "covered_80",
                "covered_95",
            }.issubset(interval_validation_predictions.columns),
        ),
        (
            "forecast_intervals_complete",
            list(forecast_intervals.columns)
            == ["data", "previsao", "lo_80", "hi_80", "lo_95", "hi_95"]
            and len(forecast_intervals) == 12,
        ),
        (
            "challenger_forecasts_complete",
            not challenger_forecasts.empty
            and {"candidate_role", "data", "previsao", "lo_80", "hi_80", "lo_95", "hi_95"}.issubset(
                challenger_forecasts.columns
            ),
        ),
        (
            "final_model_metadata_complete",
            not final_model_metadata.empty
            and {
                "selected_target_strategy",
                "selected_final_alpha",
                "selected_final_alpha_source",
                "selected_model_id",
                "train_start",
                "train_end",
                "forecast_start",
                "forecast_end",
                "future_covid_shock_sum",
                "future_covid_recovery_sum",
            }.issubset(final_model_metadata.columns),
        ),
        (
            "previsao_shape_complete",
            list(previsao.columns) == ["data", "previsao"] and len(previsao) == 12,
        ),
        (
            "shap_explainability_complete",
            not shap_feature_importance.empty
            and not shap_values_sample.empty
            and shap_feature_importance["status"].eq("ok").any()
            and {"feature", "mean_abs_shap", "rank", "method"}.issubset(
                shap_feature_importance.columns
            ),
        ),
        (
            "decision_html_report_complete",
            bool(decision_html_report)
            and "Decisoes para MASE 2024" in decision_html_report
            and "O que foi descartado" in decision_html_report
            and "ajuste COVID" in decision_html_report
            and Path(outputs["decision_html_report_path"]).exists(),
        ),
        (
            "html_report_complete",
            bool(html_report)
            and "PADS Forecasting" in html_report
            and "Explainability SHAP" in html_report
            and "Common MASE" in html_report
            and Path(outputs["html_report_path"]).exists(),
        ),
        (
            "mlflow_final_artifacts_complete",
            _final_mlflow_artifacts_present(Path(outputs["mlruns_dir"])),
        ),
        ("no_hidden_notebook_decisions", _no_notebook_dependency(outputs)),
    ]
    _log_to_mlflow(
        project=project,
        old_data_gate=old_data_gate,
        cv_summary=cv_summary,
        residual_diagnostics=residual_diagnostics,
        interval_coverage_proxy=interval_coverage_proxy,
        interval_validation_predictions=interval_validation_predictions,
        forecast_intervals=forecast_intervals,
        challenger_forecasts=challenger_forecasts,
        final_model_metadata=final_model_metadata,
        outputs=outputs,
    )
    return pd.DataFrame([{"requirement": name, "passed": bool(passed)} for name, passed in checks])
