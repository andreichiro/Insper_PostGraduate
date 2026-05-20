"""EDA nodes and artifact-ready figures/tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from pads_forecasting.tracking import mlflow_log_artifacts


def _safe_test(name: str, series: pd.Series, test: str) -> dict[str, Any]:
    y = series.dropna().astype(float)
    try:
        if test == "adf":
            stat, pvalue, usedlag, nobs, *_ = adfuller(y, autolag="AIC")
            return {
                "series": name,
                "test": "ADF",
                "statistic": stat,
                "p_value": pvalue,
                "used_lag": usedlag,
                "n_obs": nobs,
                "conclusion": "reject unit root" if pvalue < 0.05 else "fail to reject unit root",
            }
        stat, pvalue, usedlag, *_ = kpss(y, regression="c", nlags="auto")
        return {
            "series": name,
            "test": "KPSS",
            "statistic": stat,
            "p_value": pvalue,
            "used_lag": usedlag,
            "n_obs": len(y),
            "conclusion": "reject stationarity" if pvalue < 0.05 else "fail to reject stationarity",
        }
    except Exception as exc:
        return {
            "series": name,
            "test": test.upper(),
            "statistic": np.nan,
            "p_value": np.nan,
            "used_lag": np.nan,
            "n_obs": len(y),
            "conclusion": f"test failed: {exc}",
        }


def generate_eda_outputs(
    panel: pd.DataFrame,
    target_strategies: dict[str, Any],
    data: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create EDA tables and figures."""

    figures_dir = Path(outputs["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    acquisition_date = pd.Timestamp(data["acquisition_date"])

    plt.figure(figsize=(10, 4))
    plt.plot(panel["data"], panel["br_publicado"], label="main published series")
    plt.axvline(acquisition_date, color="black", linestyle="--", label="acquisition")
    plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"), color="tab:red", alpha=0.12)
    plt.axvspan(
        pd.Timestamp("2020-07-01"), pd.Timestamp("2020-12-01"), color="tab:orange", alpha=0.10
    )
    plt.title("Distribuidora BR series with acquisition and COVID windows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "series_acquisition_covid.png", dpi=160)
    plt.close()

    acquired = panel[panel["data"] < acquisition_date]
    plt.figure(figsize=(9, 4))
    plt.plot(acquired["data"], acquired["adquirida_separada"], color="tab:green")
    plt.title("Acquired company standalone series")
    plt.tight_layout()
    plt.savefig(figures_dir / "acquired_company_series.png", dpi=160)
    plt.close()

    proforma = target_strategies["strategies"]["proforma_sum"]
    raw = target_strategies["strategies"]["raw_full"]
    plt.figure(figsize=(10, 4))
    plt.plot(raw["data"], raw["y"], label="raw_full")
    plt.plot(proforma["data"], proforma["y"], label="proforma_sum", alpha=0.85)
    plt.axvline(acquisition_date, color="black", linestyle="--")
    plt.title("Raw vs pro-forma reconstructed target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "target_reconstruction_overlay.png", dpi=160)
    plt.close()

    seasonal = panel.assign(month=panel["data"].dt.month)
    plt.figure(figsize=(9, 4))
    seasonal.boxplot(column="br_publicado", by="month", grid=False)
    plt.suptitle("")
    plt.title("Month-of-year profile")
    plt.tight_layout()
    plt.savefig(figures_dir / "seasonality_month_profile.png", dpi=160)
    plt.close()

    stl_series = proforma.set_index("data")["y"].asfreq("MS")
    result = STL(stl_series, period=12, robust=True).fit()
    fig = result.plot()
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    fig.savefig(figures_dir / "decomposition.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    plt.plot(panel["data"], panel["br_publicado"], label="main")
    covid_mask = panel["covid_shock"].eq(1) | panel["covid_recovery"].eq(1)
    plt.scatter(
        panel.loc[covid_mask, "data"],
        panel.loc[covid_mask, "br_publicado"],
        color="tab:red",
        label="COVID window",
    )
    plt.title("Visible shock/outlier months")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "outliers_covid.png", dpi=160)
    plt.close()

    stationarity_rows = []
    for name in ["raw_full", "post_only", "proforma_sum"]:
        series = target_strategies["strategies"][name]["y"]
        stationarity_rows.append(_safe_test(name, series, "adf"))
        stationarity_rows.append(_safe_test(name, series, "kpss"))

    eda_summary = pd.DataFrame(
        [
            {"item": "main_rows", "value": len(panel)},
            {"item": "main_start", "value": str(panel["data"].min().date())},
            {"item": "main_end", "value": str(panel["data"].max().date())},
            {
                "item": "pre_acquisition_mean",
                "value": panel.loc[panel["data"] < acquisition_date, "br_publicado"].mean(),
            },
            {
                "item": "post_acquisition_mean",
                "value": panel.loc[panel["data"] >= acquisition_date, "br_publicado"].mean(),
            },
            {"item": "covid_shock_months", "value": int(panel["covid_shock"].sum())},
            {"item": "covid_recovery_months", "value": int(panel["covid_recovery"].sum())},
        ]
    )
    eda_summary["value"] = eda_summary["value"].astype(str)
    stationarity_tests = pd.DataFrame(stationarity_rows)
    eda_summary_path = Path(outputs["reporting_dir"]) / "eda_summary.parquet"
    stationarity_path = Path(outputs["reporting_dir"]) / "stationarity_tests.parquet"
    eda_summary.to_parquet(eda_summary_path, index=False)
    stationarity_tests.to_parquet(stationarity_path, index=False)
    mlflow_log_artifacts(
        [
            figures_dir / "series_acquisition_covid.png",
            figures_dir / "acquired_company_series.png",
            figures_dir / "target_reconstruction_overlay.png",
            figures_dir / "seasonality_month_profile.png",
            figures_dir / "decomposition.png",
            figures_dir / "outliers_covid.png",
            eda_summary_path,
            stationarity_path,
        ]
    )
    return eda_summary, stationarity_tests
