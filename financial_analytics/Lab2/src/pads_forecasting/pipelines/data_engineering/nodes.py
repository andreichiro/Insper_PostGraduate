"""Nodes for raw loading, contracts, canonical panel, and resolved config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pads_forecasting.contracts import (
    assert_validation_passed,
    ensure_dir,
    normalize_monthly_frame,
    validate_monthly_series,
    validation_frame,
)
from pads_forecasting.leakage import assert_no_2024_used
from pads_forecasting.schemas import validate_parameter_groups
from pads_forecasting.tracking import (
    config_hash,
    dataframe_fingerprint,
    mlflow_log_artifacts,
    mlflow_log_metrics,
    mlflow_log_params,
)


def build_canonical_panel(
    main_raw: pd.DataFrame,
    acquired_raw: pd.DataFrame,
    project: dict[str, Any],
    data: dict[str, Any],
    interventions: dict[str, Any],
    reconstruction: dict[str, Any],
    validation: dict[str, Any],
    selection: dict[str, Any],
    outputs: dict[str, Any],
    models: dict[str, Any],
    hpo: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate raw data and build the canonical modeling panel."""

    validate_parameter_groups(
        project,
        data,
        interventions,
        reconstruction,
        validation,
        selection,
        outputs,
        models,
        hpo,
        metrics,
    )
    ensure_dir(outputs["reporting_dir"])
    ensure_dir(outputs["figures_dir"])
    ensure_dir("outputs")

    main = normalize_monthly_frame(
        main_raw,
        date_col=data["date_col"],
        value_col=data["value_col"],
        value_name="br_publicado",
    )
    acquired = normalize_monthly_frame(
        acquired_raw,
        date_col=data["date_col"],
        value_col=data["value_col"],
        value_name="adquirida_separada",
    )

    rows = []
    rows.extend(
        validate_monthly_series(
            main,
            name="main",
            value_col="br_publicado",
            expected_start=data["expected_main_start"],
            expected_end=data["expected_main_end"],
            expected_rows=data["expected_main_rows"],
        )
    )
    rows.extend(
        validate_monthly_series(
            acquired,
            name="acquired",
            value_col="adquirida_separada",
            expected_start=data["expected_acquired_start"],
            expected_end=data["expected_acquired_end"],
            expected_rows=data["expected_acquired_rows"],
        )
    )
    assert_validation_passed(rows)
    assert_no_2024_used(main)
    assert_no_2024_used(acquired)

    panel = main.merge(acquired, on="data", how="left")
    acquisition_date = pd.Timestamp(data["acquisition_date"])
    panel["is_pre_acquisition"] = panel["data"] < acquisition_date
    panel["is_post_acquisition"] = panel["data"] >= acquisition_date
    panel["consolidado_observado"] = panel["br_publicado"].where(panel["is_post_acquisition"])
    panel["target_source"] = panel["is_post_acquisition"].map(
        {True: "observed_consolidated", False: "observed_br_standalone"}
    )
    panel["month"] = panel["data"].dt.month
    panel["trend_index"] = range(len(panel))

    covid = interventions["covid"]
    shock_start, shock_end = map(pd.Timestamp, covid["shock_window"])
    recovery_start, recovery_end = map(pd.Timestamp, covid["recovery_window"])
    panel["covid_shock"] = (
        covid["enabled"] & panel["data"].between(shock_start, shock_end)
    ).astype(int)
    panel["covid_recovery"] = (
        covid["enabled"] & panel["data"].between(recovery_start, recovery_end)
    ).astype(int)
    aftershock_window = covid.get("aftershock_window")
    if aftershock_window:
        aftershock_start, aftershock_end = map(pd.Timestamp, aftershock_window)
        panel["covid_aftershock_2021"] = (
            covid["enabled"] & panel["data"].between(aftershock_start, aftershock_end)
        ).astype(int)
    else:
        panel["covid_aftershock_2021"] = 0

    # Standalone acquired-company values are intentionally unavailable after acquisition.
    panel.loc[panel["is_post_acquisition"], "adquirida_separada"] = pd.NA

    resolved_config = {
        "project": project,
        "data": data,
        "interventions": interventions,
        "reconstruction": reconstruction,
        "validation": validation,
        "selection": selection,
        "outputs": outputs,
        "models": models,
        "hpo": hpo,
        "metrics": metrics,
    }
    resolved_path = Path(outputs["reporting_dir"]) / "config_resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    resolved_config_hash = config_hash(resolved_config)
    main_fingerprint = dataframe_fingerprint(main)
    acquired_fingerprint = dataframe_fingerprint(acquired)

    mlflow_log_metrics(
        {
            "main_rows": len(main),
            "acquired_rows": len(acquired),
        },
        prefix="data",
    )
    mlflow_log_params(
        {
            "run_id": project["run_id"],
            "config_hash": resolved_config_hash,
            "acquisition_date": data["acquisition_date"],
            "forecast_horizon": data["horizon"],
        },
        prefix="run",
    )
    mlflow_log_params(
        {
            "main_fingerprint": main_fingerprint,
            "acquired_fingerprint": acquired_fingerprint,
        },
        prefix="data",
    )
    mlflow_log_artifacts([resolved_path])
    validation = validation_frame(rows)
    validation.loc[len(validation)] = {
        "check": "config_hash",
        "passed": True,
        "observed": resolved_config_hash,
        "expected": "stable hash for idempotence",
    }
    validation.loc[len(validation)] = {
        "check": "main_fingerprint",
        "passed": True,
        "observed": main_fingerprint,
        "expected": "stable fingerprint",
    }
    validation.loc[len(validation)] = {
        "check": "acquired_fingerprint",
        "passed": True,
        "observed": acquired_fingerprint,
        "expected": "stable fingerprint",
    }
    return panel, validation
