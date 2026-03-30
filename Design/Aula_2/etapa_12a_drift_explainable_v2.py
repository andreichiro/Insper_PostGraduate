#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from explainable_drift_prediction_common_v2 import (
    PUBLIC_TABLE_MAP,
    RECENT_WINDOW_MONTHS,
    attach_reference,
    build_config,
    build_feature_candidates,
    build_input_map,
    build_output_reference,
    categorical_drift_level,
    connect_output,
    connect_source,
    drift_rank,
    normalize_text,
    numeric_drift_level,
    persist_table,
    prepare_model_population,
    psi_numeric,
    strip_reference_cols,
    write_json,
    write_markdown,
    load_public_tables,
    Config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 1: explainable drift review from the relevant base-modelada tables.")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_assumptions_table() -> pd.DataFrame:
    rows = [
        {
            "step_number": 1,
            "topic": "population",
            "assumption": "Drift is measured on the exact modeling population that will later be used for prediction.",
            "why_this_is_sound": "This keeps the drift story decision-relevant. We are not mixing modeled and non-modeled populations.",
            "what_changes_if_it_fails": "If we widen the population, measured drift can be diluted by rows that are never used by the model.",
        },
        {
            "step_number": 2,
            "topic": "time_window",
            "assumption": f"`old` means the first {RECENT_WINDOW_MONTHS} modeling months and `recent` means the last {RECENT_WINDOW_MONTHS} modeling months.",
            "why_this_is_sound": "It is symmetric, reproducible, and avoids hand-picked dates.",
            "what_changes_if_it_fails": "Different windows change drift magnitude, especially for acquisition and maturity variables.",
        },
        {
            "step_number": 3,
            "topic": "numeric_metrics",
            "assumption": "Numeric drift uses both PSI and standardized mean difference.",
            "why_this_is_sound": "PSI captures shape change; SMD captures shift in central tendency. Using both is more robust than one metric alone.",
            "what_changes_if_it_fails": "A single metric can miss important shifts or overreact to small but noisy changes.",
        },
        {
            "step_number": 4,
            "topic": "categorical_metrics",
            "assumption": "Categorical drift uses category share difference and total variation distance.",
            "why_this_is_sound": "This shows both which categories moved and how large the full-distribution change is.",
            "what_changes_if_it_fails": "Only reading the top category can hide broader distributional change.",
        },
        {
            "step_number": 5,
            "topic": "relevance",
            "assumption": "Drift relevance is practical, not purely statistical.",
            "why_this_is_sound": "At this sample size, almost everything could be statistically significant. Practical thresholds are easier to justify operationally.",
            "what_changes_if_it_fails": "Pure significance testing would overstate the importance of tiny changes.",
        },
    ]
    df = pd.DataFrame(rows)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "audit_base_modelada_validation"],
        build_summary="Step-by-step assumptions that define how drift is measured and interpreted in this focused review.",
        rebuild_from_raw="No raw rebuild is needed for this table itself; it documents the assumptions used by etapa_12a_drift_explainable_v2.py after raw_para_base_modelada_v4.py has built the relevant-table layer.",
    )


def split_old_vs_recent(population: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    months = sorted(population["month"].dropna().unique().tolist())
    old_months = months[:RECENT_WINDOW_MONTHS]
    recent_months = months[-RECENT_WINDOW_MONTHS:]
    old = population[population["month"].isin(old_months)].copy()
    recent = population[population["month"].isin(recent_months)].copy()
    return old, recent, old_months, recent_months


def build_numeric_drift(
    population: pd.DataFrame,
    numeric_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    old, recent, old_months, recent_months = split_old_vs_recent(population)
    feature_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []

    for feature_name in numeric_features:
        old_series = pd.to_numeric(old[feature_name], errors="coerce")
        recent_series = pd.to_numeric(recent[feature_name], errors="coerce")
        if old_series.notna().sum() == 0 or recent_series.notna().sum() == 0:
            continue

        old_mean = float(old_series.mean())
        recent_mean = float(recent_series.mean())
        pooled_std = np.sqrt((np.nanvar(old_series, ddof=0) + np.nanvar(recent_series, ddof=0)) / 2.0)
        if (pd.isna(pooled_std) or pooled_std == 0) and np.isclose(old_mean, recent_mean, equal_nan=True):
            smd = 0.0
        elif pooled_std and not np.isnan(pooled_std):
            smd = (recent_mean - old_mean) / pooled_std
        else:
            smd = float("nan")
        psi_value = psi_numeric(old_series, recent_series)
        level = numeric_drift_level(psi_value, smd)

        feature_rows.append(
            {
                "feature_name": feature_name,
                "old_month_start": str(min(old_months)),
                "old_month_end": str(max(old_months)),
                "recent_month_start": str(min(recent_months)),
                "recent_month_end": str(max(recent_months)),
                "old_rows": int(old_series.notna().sum()),
                "recent_rows": int(recent_series.notna().sum()),
                "old_mean": old_mean,
                "recent_mean": recent_mean,
                "old_median": float(old_series.median()),
                "recent_median": float(recent_series.median()),
                "mean_delta": recent_mean - old_mean,
                "standardized_mean_diff": smd,
                "psi": psi_value,
                "drift_level": level,
                "drift_relevance": "relevant" if level in {"high_drift", "medium_drift"} else "limited",
            }
        )

    for target_name in ["target_churn_m1", "target_return_active_m1"]:
        old_rate = float(pd.to_numeric(old[target_name], errors="coerce").mean())
        recent_rate = float(pd.to_numeric(recent[target_name], errors="coerce").mean())
        outcome_rows.append(
            {
                "metric_name": target_name,
                "old_month_start": str(min(old_months)),
                "old_month_end": str(max(old_months)),
                "recent_month_start": str(min(recent_months)),
                "recent_month_end": str(max(recent_months)),
                "old_rate": old_rate,
                "recent_rate": recent_rate,
                "rate_diff_pp": (recent_rate - old_rate) * 100,
            }
        )

    numeric_df = pd.DataFrame(feature_rows)
    if not numeric_df.empty:
        numeric_df["_drift_rank"] = numeric_df["drift_level"].map(drift_rank)
        numeric_df = numeric_df.sort_values(["_drift_rank", "psi"], ascending=[True, False]).drop(columns="_drift_rank").reset_index(drop=True)
    numeric_df = attach_reference(
        numeric_df,
        source_tables=["mart_teacher_month_persona_ready", "audit_persona_feature_readiness", "dim_persona_range_candidates"],
        build_summary="Numeric drift between the oldest and most recent six modeling months, measured with mean/median change, standardized mean difference, and PSI.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, define old and recent windows, then recompute mean/median change, SMD, and PSI for each selected numeric feature.",
    )

    outcome_df = pd.DataFrame(outcome_rows)
    outcome_df = attach_reference(
        outcome_df,
        source_tables=["mart_teacher_month_persona_ready"],
        build_summary="Outcome-rate drift for churn and return between the oldest and most recent six modeling months.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the same modeling population from mart_teacher_month_persona_ready, then compare target rates across the old and recent windows.",
    )
    return numeric_df, outcome_df


def build_categorical_drift(
    population: pd.DataFrame,
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    old, recent, old_months, recent_months = split_old_vs_recent(population)
    rows: List[Dict[str, Any]] = []

    for feature_name in categorical_features:
        old_series = normalize_text(old[feature_name])
        recent_series = normalize_text(recent[feature_name])
        categories = sorted(set(old_series.unique()).union(set(recent_series.unique())))
        old_share = old_series.value_counts(normalize=True, dropna=False)
        recent_share = recent_series.value_counts(normalize=True, dropna=False)
        total_variation = 0.5 * float(
            sum(abs(float(old_share.get(cat, 0.0)) - float(recent_share.get(cat, 0.0))) for cat in categories)
        )
        max_diff_pp = (
            max(abs((float(recent_share.get(cat, 0.0)) - float(old_share.get(cat, 0.0))) * 100) for cat in categories)
            if categories
            else float("nan")
        )
        level = categorical_drift_level(total_variation, max_diff_pp)

        for cat in categories:
            rows.append(
                {
                    "feature_name": feature_name,
                    "category_value": cat,
                    "old_month_start": str(min(old_months)),
                    "old_month_end": str(max(old_months)),
                    "recent_month_start": str(min(recent_months)),
                    "recent_month_end": str(max(recent_months)),
                    "old_share": float(old_share.get(cat, 0.0)),
                    "recent_share": float(recent_share.get(cat, 0.0)),
                    "share_diff_pp": (float(recent_share.get(cat, 0.0)) - float(old_share.get(cat, 0.0))) * 100,
                    "feature_total_variation": total_variation,
                    "feature_max_share_diff_pp": max_diff_pp,
                    "drift_level": level,
                    "drift_relevance": "relevant" if level in {"high_drift", "medium_drift"} else "limited",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_drift_rank"] = df["drift_level"].map(drift_rank)
        df["_abs_diff"] = df["share_diff_pp"].abs()
        df = df.sort_values(["_drift_rank", "_abs_diff"], ascending=[True, False]).drop(columns=["_drift_rank", "_abs_diff"]).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher"],
        build_summary="Categorical drift between the oldest and most recent six modeling months, measured with category share difference and total variation distance.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the same modeling population from mart_teacher_month_persona_ready, then compare category shares for each selected dimension across the old and recent windows.",
    )


def build_key_findings(
    numeric_drift: pd.DataFrame,
    categorical_drift: pd.DataFrame,
    outcome_drift: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    numeric_view = strip_reference_cols(numeric_drift).head(5)
    for _, row in numeric_view.iterrows():
        rows.append(
            {
                "finding_group": "numeric_drift",
                "finding_label": str(row["feature_name"]),
                "evidence_value": float(row["psi"]),
                "evidence_unit": "psi",
                "why_it_matters": "This feature changed materially between the oldest and most recent periods.",
                "interpretation": f"{row['feature_name']} is classified as {row['drift_level']}, so models trained on older data may read this feature differently in recent data.",
            }
        )

    categorical_view = strip_reference_cols(categorical_drift).head(5)
    for _, row in categorical_view.iterrows():
        rows.append(
            {
                "finding_group": "categorical_drift",
                "finding_label": f"{row['feature_name']}::{row['category_value']}",
                "evidence_value": float(row["share_diff_pp"]),
                "evidence_unit": "pp",
                "why_it_matters": "The composition of this dimension changed materially over time.",
                "interpretation": f"{row['feature_name']} moved by {row['share_diff_pp']:.2f} pp for category {row['category_value']}, which can change who enters the modeled population.",
            }
        )

    for _, row in strip_reference_cols(outcome_drift).iterrows():
        rows.append(
            {
                "finding_group": "outcome_drift",
                "finding_label": str(row["metric_name"]),
                "evidence_value": float(row["rate_diff_pp"]),
                "evidence_unit": "pp",
                "why_it_matters": "The target itself moved over time, so performance must be checked on recent data.",
                "interpretation": f"{row['metric_name']} changed by {row['rate_diff_pp']:.2f} pp between the oldest and most recent windows.",
            }
        )

    df = pd.DataFrame(rows)
    return attach_reference(
        df,
        source_tables=["analytics_drift_numeric_explainable_v2", "analytics_drift_categorical_explainable_v2", "analytics_drift_outcome_explainable_v2"],
        build_summary="Plain-English drift findings table derived from the main numeric, categorical, and outcome drift outputs.",
        rebuild_from_raw="Rerun etapa_12a_drift_explainable_v2.py after raw_para_base_modelada_v4.py; this table is derived from the drift output tables generated in the same run.",
    )


def build_summary_payload(
    population_summary: pd.DataFrame,
    numeric_drift: pd.DataFrame,
    categorical_drift: pd.DataFrame,
    outcome_drift: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "population_summary": strip_reference_cols(population_summary).to_dict(orient="records"),
        "top_numeric_drift": strip_reference_cols(numeric_drift).head(8).to_dict(orient="records"),
        "top_categorical_drift": strip_reference_cols(categorical_drift).head(8).to_dict(orient="records"),
        "outcome_drift": strip_reference_cols(outcome_drift).to_dict(orient="records"),
    }


def write_summary_markdown(path: Path, cfg: Config, payload: Dict[str, Any]) -> None:
    population = payload["population_summary"][0] if payload["population_summary"] else {}
    lines = [
        "# Drift Review v2",
        "",
        "## Paths",
        "",
        f"- Source DuckDB: `{cfg.source_duckdb_path}`",
        f"- Output directory: `{cfg.output_dir}`",
        "",
        "## Step By Step",
        "",
        "1. Start from the relevant-table layer exported by `raw_para_base_modelada_v4.py`.",
        "2. Freeze the modeling population in `mart_teacher_month_persona_ready`.",
        f"3. Compare the first {RECENT_WINDOW_MONTHS} modeling months against the last {RECENT_WINDOW_MONTHS} modeling months.",
        "4. Measure numeric drift with PSI and standardized mean difference.",
        "5. Measure categorical drift with share difference and total variation.",
        "6. Check whether churn and return themselves drifted across time.",
        "",
        "## Population",
        "",
        f"- Rows: {population.get('rows', 'n/a')}",
        f"- Teachers: {population.get('teachers', 'n/a')}",
        f"- Period: {population.get('month_start', 'n/a')} to {population.get('month_end', 'n/a')}",
        "",
        "## Top Numeric Drift",
    ]
    for row in payload["top_numeric_drift"][:6]:
        lines.append(
            f"- `{row['feature_name']}` | `{row['drift_level']}` | old_mean={row['old_mean']:.4f} | recent_mean={row['recent_mean']:.4f} | psi={row['psi']:.4f}"
        )
    lines.extend(["", "## Top Categorical Drift"])
    for row in payload["top_categorical_drift"][:6]:
        lines.append(
            f"- `{row['feature_name']}::{row['category_value']}` | `{row['drift_level']}` | diff_pp={row['share_diff_pp']:.2f} | tv={row['feature_total_variation']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "1. Run `raw_para_base_modelada_v4.py`.",
            "2. Run `etapa_12a_drift_explainable_v2.py`.",
        ]
    )
    write_markdown(path, lines)


def main() -> None:
    args = parse_args()
    cfg = build_config(base_dir=args.base_dir, source_dir=args.source_dir, output_dir=args.output_dir)
    source_conn = connect_source(cfg)
    output_conn = connect_output(cfg)

    try:
        tables = load_public_tables(source_conn)
        direct_tables = [
            "audit_base_modelada_validation",
            "audit_persona_feature_readiness",
            "dim_persona_range_candidates",
            "dim_teacher",
            "mart_teacher_month_persona_ready",
        ]
        input_map = build_input_map(
            tables,
            direct_tables=direct_tables,
            flag_column="used_directly_for_drift",
            analysis_summary="Inventory of the declared relevant tables and whether each one is used directly in the drift analysis.",
        )
        population, population_summary = prepare_model_population(tables)
        assumptions = build_assumptions_table()
        feature_candidates, numeric_features, _, context_drift_features = build_feature_candidates(tables, population)
        numeric_drift, outcome_drift = build_numeric_drift(population, numeric_features)
        categorical_drift = build_categorical_drift(population, context_drift_features)
        key_findings = build_key_findings(numeric_drift, categorical_drift, outcome_drift)

        outputs = {
            "analytics_drift_input_map_v2": input_map,
            "analytics_drift_assumptions_v2": assumptions,
            "analytics_drift_population_summary_v2": population_summary,
            "analytics_drift_numeric_explainable_v2": numeric_drift,
            "analytics_drift_categorical_explainable_v2": categorical_drift,
            "analytics_drift_outcome_explainable_v2": outcome_drift,
            "analytics_drift_key_findings_v2": key_findings,
        }
        output_reference = build_output_reference(
            cfg,
            outputs.keys(),
            build_summary="Manifest of all drift outputs generated by etapa_12a_drift_explainable_v2.py.",
        )
        outputs["analytics_drift_output_reference_v2"] = output_reference

        for table_name, df in outputs.items():
            persist_table(output_conn, cfg, table_name, df)

        payload = build_summary_payload(population_summary, numeric_drift, categorical_drift, outcome_drift)
        write_json(cfg.output_dir / "json" / "drift_summary_v2.json", payload)
        write_summary_markdown(cfg.output_dir / "audit" / "drift_summary_v2.md", cfg, payload)
    finally:
        source_conn.close()
        output_conn.close()


if __name__ == "__main__":
    main()
