"""Nodes for acquisition-aware target reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pads_forecasting.metrics import mase, seasonal_naive_forecast
from pads_forecasting.tracking import mlflow_log_artifacts


def _strategy_frame(
    panel: pd.DataFrame,
    *,
    name: str,
    acquisition_date: str,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    df = panel.copy()
    acquisition_ts = pd.Timestamp(acquisition_date)
    pre = df["data"] < acquisition_ts
    post = df["data"] >= acquisition_ts

    if name == "raw_full":
        df["y"] = df["br_publicado"]
        df["strategy_family"] = "raw_full"
    elif name == "post_only":
        df = df.loc[post].copy()
        df["y"] = df["br_publicado"]
        df["strategy_family"] = "post_only"
    elif name in {"proforma_sum", "calibrated_alpha"}:
        df["y"] = df["br_publicado"]
        acquired_pre = pd.to_numeric(df.loc[pre, "adquirida_separada"], errors="coerce").fillna(0.0)
        br_pre = pd.to_numeric(df.loc[pre, "br_publicado"], errors="coerce")
        df.loc[pre, "y"] = beta * br_pre + alpha * acquired_pre
        df["strategy_family"] = name
    else:
        df["y"] = df["br_publicado"]
        acquired_pre = pd.to_numeric(df.loc[pre, "adquirida_separada"], errors="coerce").fillna(0.0)
        br_pre = pd.to_numeric(df.loc[pre, "br_publicado"], errors="coerce")
        df.loc[pre, "y"] = beta * br_pre + alpha * acquired_pre
        df["strategy_family"] = "two_weight_sensitivity"

    df["target_strategy"] = name
    df["alpha"] = alpha
    df["beta"] = beta
    pre_current = df["data"] < acquisition_ts
    post_current = df["data"] >= acquisition_ts
    df["br_component_observed"] = pd.NA
    df["acquired_component_observed"] = pd.NA
    br_pre = pd.to_numeric(df.loc[pre_current, "br_publicado"], errors="coerce")
    acquired_pre = pd.to_numeric(df.loc[pre_current, "adquirida_separada"], errors="coerce").fillna(
        0.0
    )
    if name == "raw_full":
        df.loc[pre_current, "br_component_observed"] = br_pre
        df.loc[pre_current, "acquired_component_observed"] = 0.0
    elif name != "post_only":
        df.loc[pre_current, "br_component_observed"] = beta * br_pre
        df.loc[pre_current, "acquired_component_observed"] = alpha * acquired_pre
    df["consolidated_observed"] = pd.NA
    df.loc[post_current, "consolidated_observed"] = pd.to_numeric(
        df.loc[post_current, "br_publicado"], errors="coerce"
    )
    df.loc[df["data"] >= acquisition_ts, "target_source"] = "observed_consolidated"
    df.loc[df["data"] < acquisition_ts, "target_source"] = "reconstructed_or_raw_pre_acquisition"
    columns = [
        "data",
        "y",
        "target_strategy",
        "strategy_family",
        "alpha",
        "beta",
        "target_source",
        "br_component_observed",
        "acquired_component_observed",
        "consolidated_observed",
        "covid_shock",
        "covid_recovery",
        "covid_aftershock_2021",
        "month",
        "trend_index",
    ]
    return df[[column for column in columns if column in df.columns]].reset_index(drop=True)


def _alpha_cv_score(
    strategy: pd.DataFrame, folds: list[dict[str, Any]], season_length: int
) -> tuple[float, float]:
    values = []
    for fold in folds:
        train = strategy[strategy["data"] <= pd.Timestamp(fold["train_end"])]
        valid = strategy[
            (strategy["data"] >= pd.Timestamp(fold["valid_start"]))
            & (strategy["data"] <= pd.Timestamp(fold["valid_end"]))
        ]
        if train.empty or valid.empty:
            continue
        yhat = seasonal_naive_forecast(train["y"], len(valid), season_length)
        values.append(mase(valid["y"], yhat, train["y"], season_length))
    if not values:
        return np.inf, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=0))


def build_target_strategies(
    panel: pd.DataFrame,
    data: dict[str, Any],
    reconstruction: dict[str, Any],
    validation: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all YAML-defined target strategies and alpha evidence tables."""

    acquisition_date = data["acquisition_date"]
    season_length = validation["season_length"]
    folds = validation["folds"]

    alpha_rows = []
    alpha_candidates = {}
    for alpha in reconstruction["alpha_grid"]:
        candidate = _strategy_frame(
            panel,
            name="calibrated_alpha",
            acquisition_date=acquisition_date,
            alpha=float(alpha),
        )
        mean_mase, std_mase = _alpha_cv_score(candidate, folds, season_length)
        penalized = mean_mase + reconstruction["alpha_penalty_lambda"] * (float(alpha) - 1.0) ** 2
        alpha_rows.append(
            {
                "alpha": float(alpha),
                "beta": 1.0,
                "candidate_role": "calibrated_alpha",
                "mean_mase_seasonal_naive": mean_mase,
                "std_mase_seasonal_naive": std_mase,
                "penalized_mase": penalized,
            }
        )
        alpha_candidates[float(alpha)] = candidate
    alpha_sensitivity = (
        pd.DataFrame(alpha_rows).sort_values("penalized_mase").reset_index(drop=True)
    )
    selected_alpha = float(alpha_sensitivity.iloc[0]["alpha"])
    alpha_one = alpha_sensitivity[alpha_sensitivity["alpha"].eq(1.0)]
    if not alpha_one.empty:
        best = alpha_sensitivity.iloc[0]["penalized_mase"]
        one = float(alpha_one.iloc[0]["penalized_mase"])
        improvement = (one - best) / one * 100 if one else 0.0
        if improvement < reconstruction["alpha_prefer_one_margin_pct"]:
            selected_alpha = 1.0

    strategies = {
        "raw_full": _strategy_frame(panel, name="raw_full", acquisition_date=acquisition_date),
        "post_only": _strategy_frame(panel, name="post_only", acquisition_date=acquisition_date),
        "proforma_sum": _strategy_frame(
            panel,
            name="proforma_sum",
            acquisition_date=acquisition_date,
            alpha=1.0,
        ),
        "calibrated_alpha": _strategy_frame(
            panel,
            name="calibrated_alpha",
            acquisition_date=acquisition_date,
            alpha=selected_alpha,
        ),
    }
    two_weight_candidates = {}
    for beta in reconstruction["beta_grid"]:
        for alpha in reconstruction["alpha_grid"]:
            key = f"beta_{float(beta):.2f}__alpha_{float(alpha):.2f}"
            two_weight_candidates[key] = _strategy_frame(
                panel,
                name="two_weight_sensitivity",
                acquisition_date=acquisition_date,
                alpha=float(alpha),
                beta=float(beta),
            )

    summary_rows = []
    for name, strategy in strategies.items():
        pre = strategy[strategy["data"] < pd.Timestamp(acquisition_date)]
        post = strategy[strategy["data"] >= pd.Timestamp(acquisition_date)]
        summary_rows.append(
            {
                "target_strategy": name,
                "candidate_role": "primary"
                if name in {"raw_full", "post_only", "proforma_sum"}
                else "challenger",
                "alpha": float(strategy["alpha"].iloc[0]),
                "beta": float(strategy["beta"].iloc[0]),
                "rows": len(strategy),
                "pre_mean": float(pre["y"].mean()) if len(pre) else np.nan,
                "post_mean": float(post["y"].mean()) if len(post) else np.nan,
                "pre_std": float(pre["y"].std(ddof=0)) if len(pre) else np.nan,
                "post_std": float(post["y"].std(ddof=0)) if len(post) else np.nan,
            }
        )
    for key, strategy in two_weight_candidates.items():
        pre = strategy[strategy["data"] < pd.Timestamp(acquisition_date)]
        post = strategy[strategy["data"] >= pd.Timestamp(acquisition_date)]
        summary_rows.append(
            {
                "target_strategy": "two_weight_sensitivity",
                "candidate_id": key,
                "candidate_role": "sensitivity",
                "alpha": float(strategy["alpha"].iloc[0]),
                "beta": float(strategy["beta"].iloc[0]),
                "rows": len(strategy),
                "pre_mean": float(pre["y"].mean()) if len(pre) else np.nan,
                "post_mean": float(post["y"].mean()) if len(post) else np.nan,
                "pre_std": float(pre["y"].std(ddof=0)) if len(pre) else np.nan,
                "post_std": float(post["y"].std(ddof=0)) if len(post) else np.nan,
            }
        )

    leave_rows = []
    for fold in folds:
        subset = [item for item in folds if item["name"] != fold["name"]]
        best_alpha = selected_alpha
        best_score = np.inf
        for alpha, candidate in alpha_candidates.items():
            score, _ = _alpha_cv_score(candidate, subset, season_length)
            if score < best_score:
                best_score = score
                best_alpha = alpha
        leave_rows.append(
            {
                "held_out_fold": fold["name"],
                "selected_alpha": best_alpha,
                "mean_mase_without_fold": best_score,
            }
        )

    figures_dir = Path(outputs["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(alpha_sensitivity["alpha"], alpha_sensitivity["penalized_mase"], marker="o")
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.axvline(selected_alpha, color="tab:red", linestyle=":", linewidth=1)
    plt.title("Alpha sensitivity")
    plt.xlabel("alpha")
    plt.ylabel("penalized seasonal-naive MASE")
    plt.tight_layout()
    plt.savefig(figures_dir / "alpha_sensitivity.png", dpi=160)
    plt.close()

    target_strategy_summary = pd.DataFrame(summary_rows)
    leave_one_fold_alpha = pd.DataFrame(leave_rows)
    target_summary_path = Path(outputs["reporting_dir"]) / "target_strategy_summary.parquet"
    alpha_path = Path(outputs["reporting_dir"]) / "alpha_sensitivity.parquet"
    leave_path = Path(outputs["reporting_dir"]) / "leave_one_fold_alpha.parquet"
    target_strategy_summary.to_parquet(target_summary_path, index=False)
    alpha_sensitivity.to_parquet(alpha_path, index=False)
    leave_one_fold_alpha.to_parquet(leave_path, index=False)
    mlflow_log_artifacts(
        [
            target_summary_path,
            alpha_path,
            leave_path,
            figures_dir / "alpha_sensitivity.png",
        ]
    )

    return (
        {
            "strategies": strategies,
            "alpha_candidates": alpha_candidates,
            "two_weight_candidates": two_weight_candidates,
            "selected_alpha": selected_alpha,
        },
        target_strategy_summary,
        alpha_sensitivity,
        leave_one_fold_alpha,
    )
