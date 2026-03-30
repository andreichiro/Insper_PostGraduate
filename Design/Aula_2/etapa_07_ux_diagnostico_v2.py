#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    V2Config,
    build_config,
    connect_duckdb,
    persist_df_to_duckdb,
    setup_logging,
    utc_now_iso,
    write_df_bundle,
    write_json,
    write_markdown,
)


COHORT_VARIANTS: Sequence[str] = ("same_month_only", "near_entry_0_1m")
PRODUCT_THRESHOLDS: Dict[str, float] = {
    "session_count_month": 2.0,
    "first7d_events": 12.0,
    "content_views_month": 6.0,
    "total_session_minutes_month": 16.684,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 07 v2: diagnostico UX/produto por jornada e abandono.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def persist_output(conn: Any, cfg: V2Config, name: str, df: pd.DataFrame) -> None:
    persist_df_to_duckdb(conn, name, df)
    write_df_bundle(cfg.output_dir, name, df)


def require_tables(conn: Any, table_names: Sequence[str]) -> None:
    existing = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    missing = [name for name in table_names if name not in existing]
    if missing:
        raise RuntimeError(
            "Tabelas obrigatorias ausentes para etapa_07_ux_diagnostico_v2.py: "
            + ", ".join(sorted(missing))
        )


def compress_top(series: pd.Series, topn: int = 10) -> pd.Series:
    normalized = series.fillna("missing").astype(str)
    normalized = normalized.replace({"<missing>": "missing", "None": "missing", "nan": "missing"})
    top_values = normalized.value_counts(dropna=False).head(topn).index
    return normalized.where(normalized.isin(top_values), "other")


def select_cohort_variant(df: pd.DataFrame, cohort_variant: str) -> pd.DataFrame:
    if cohort_variant == "same_month_only":
        return df[df["cohort_variant_same_month_only"] == 1].copy()
    if cohort_variant == "near_entry_0_1m":
        return df[df["cohort_variant_near_entry_0_1m"] == 1].copy()
    raise ValueError(f"cohort_variant desconhecido: {cohort_variant}")


def build_first_session_journey_mart(conn: Any) -> pd.DataFrame:
    query = """
    WITH onboarding AS (
      SELECT
        teacher_unique_id,
        first_month,
        data_entrada_month,
        months_after_entry,
        analysis_population,
        teacher_population_status,
        utm_group,
        returned_active_m1,
        returned_any_session_m1,
        returned_any_download_m1,
        session_count_month,
        total_session_minutes_month,
        active_days_month,
        activity_events_month,
        content_views_month,
        other_activity_non_download_events_month,
        strict_download_count_month,
        strict_value_flag,
        used_mobile_flag,
        used_desktop_flag,
        first7d_events,
        first7d_active_days,
        first7d_sessions,
        first7d_session_minutes,
        first_event_type AS onboarding_first_event_type,
        first_event_action AS onboarding_first_event_action,
        first_utm_source AS onboarding_first_utm_source,
        first_device AS onboarding_first_device,
        session_without_interaction_flag,
        cohort_variant_same_month_only,
        cohort_variant_near_entry_0_1m
      FROM mart_teacher_onboarding_first_month_v2
    ),
    first_session_ranked AS (
      SELECT
        o.teacher_unique_id,
        s.session_row_hash AS first_session_row_hash,
        s.session_start_ts AS first_session_start_ts,
        s.session_end_ts AS first_session_end_ts,
        s.duration_sec AS first_session_duration_sec,
        s.duration_min AS first_session_duration_min,
        ROW_NUMBER() OVER (
          PARTITION BY o.teacher_unique_id
          ORDER BY s.session_start_ts NULLS LAST, s.session_row_hash
        ) AS rn
      FROM onboarding o
      LEFT JOIN fct_session_clean s
        ON o.teacher_unique_id = s.teacher_unique_id
       AND CAST(o.first_month AS DATE) = s.session_month
    ),
    first_session AS (
      SELECT * EXCLUDE(rn)
      FROM first_session_ranked
      WHERE rn = 1
    ),
    session_interactions AS (
      SELECT
        fs.teacher_unique_id,
        fs.first_session_row_hash,
        i.interaction_row_hash,
        i.interaction_ts,
        i.event_type,
        i.event_family,
        i.event_action,
        i.utm_source,
        i.device_group,
        COALESCE(i.is_download_event, 0) AS is_download_event,
        COALESCE(i.is_visualization_event, 0) AS is_visualization_event,
        COALESCE(i.is_navigation_event, 0) AS is_navigation_event,
        COALESCE(i.is_activity_event, 0) AS is_activity_event,
        COALESCE(i.is_other_activity_non_download_event, 0) AS is_other_activity_non_download_event,
        CASE
          WHEN COALESCE(i.is_download_event, 0) = 1
            OR COALESCE(i.is_visualization_event, 0) = 1
            OR COALESCE(i.is_other_activity_non_download_event, 0) = 1 THEN 1
          ELSE 0
        END AS is_meaningful_event
      FROM first_session fs
      LEFT JOIN fct_interaction_clean i
        ON fs.teacher_unique_id = i.teacher_unique_id
       AND fs.first_session_start_ts IS NOT NULL
       AND i.interaction_ts >= fs.first_session_start_ts
       AND i.interaction_ts <= fs.first_session_end_ts
    ),
    ranked_interactions AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY teacher_unique_id
          ORDER BY interaction_ts NULLS LAST, interaction_row_hash
        ) AS rn_asc,
        ROW_NUMBER() OVER (
          PARTITION BY teacher_unique_id
          ORDER BY interaction_ts DESC NULLS LAST, interaction_row_hash DESC
        ) AS rn_desc
      FROM session_interactions
    ),
    meaningful_interactions AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY teacher_unique_id
          ORDER BY interaction_ts, interaction_row_hash
        ) AS rn_meaningful
      FROM session_interactions
      WHERE interaction_row_hash IS NOT NULL
        AND is_meaningful_event = 1
    ),
    interaction_summary AS (
      SELECT
        teacher_unique_id,
        COUNT(interaction_row_hash) AS first_session_interactions,
        SUM(is_download_event) AS first_session_downloads,
        SUM(is_visualization_event) AS first_session_views,
        SUM(is_other_activity_non_download_event) AS first_session_other_actions,
        SUM(is_navigation_event) AS first_session_navigation_events,
        SUM(is_meaningful_event) AS first_session_meaningful_events,
        MAX(CASE WHEN rn_asc = 1 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_1_token,
        MAX(CASE WHEN rn_asc = 2 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_2_token,
        MAX(CASE WHEN rn_asc = 3 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_3_token,
        MAX(CASE WHEN rn_asc = 4 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_4_token,
        MAX(CASE WHEN rn_asc = 5 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_5_token
      FROM ranked_interactions
      GROUP BY 1
    ),
    first_event AS (
      SELECT
        teacher_unique_id,
        interaction_ts AS first_session_first_event_ts,
        event_type AS first_session_first_event_type,
        event_family AS first_session_first_event_family,
        event_action AS first_session_first_event_action,
        utm_source AS first_session_first_event_utm_source,
        device_group AS first_session_first_event_device
      FROM ranked_interactions
      WHERE rn_asc = 1
    ),
    last_event AS (
      SELECT
        teacher_unique_id,
        interaction_ts AS first_session_last_event_ts,
        event_type AS first_session_last_event_type,
        event_family AS first_session_last_event_family,
        event_action AS first_session_last_event_action
      FROM ranked_interactions
      WHERE rn_desc = 1
    ),
    first_meaningful AS (
      SELECT
        teacher_unique_id,
        interaction_ts AS first_session_first_meaningful_ts,
        event_type AS first_session_first_meaningful_type,
        event_family AS first_session_first_meaningful_family,
        event_action AS first_session_first_meaningful_action
      FROM meaningful_interactions
      WHERE rn_meaningful = 1
    )
    SELECT
      o.*,
      fs.first_session_row_hash,
      fs.first_session_start_ts,
      fs.first_session_end_ts,
      fs.first_session_duration_sec,
      fs.first_session_duration_min,
      ia.first_session_interactions,
      ia.first_session_downloads,
      ia.first_session_views,
      ia.first_session_other_actions,
      ia.first_session_navigation_events,
      ia.first_session_meaningful_events,
      ia.step_1_token,
      ia.step_2_token,
      ia.step_3_token,
      ia.step_4_token,
      ia.step_5_token,
      fe.first_session_first_event_ts,
      fe.first_session_first_event_type,
      fe.first_session_first_event_family,
      fe.first_session_first_event_action,
      fe.first_session_first_event_utm_source,
      fe.first_session_first_event_device,
      fm.first_session_first_meaningful_ts,
      fm.first_session_first_meaningful_type,
      fm.first_session_first_meaningful_family,
      fm.first_session_first_meaningful_action,
      le.first_session_last_event_ts,
      le.first_session_last_event_type,
      le.first_session_last_event_family,
      le.first_session_last_event_action
    FROM onboarding o
    LEFT JOIN first_session fs
      ON o.teacher_unique_id = fs.teacher_unique_id
    LEFT JOIN interaction_summary ia
      ON o.teacher_unique_id = ia.teacher_unique_id
    LEFT JOIN first_event fe
      ON o.teacher_unique_id = fe.teacher_unique_id
    LEFT JOIN first_meaningful fm
      ON o.teacher_unique_id = fm.teacher_unique_id
    LEFT JOIN last_event le
      ON o.teacher_unique_id = le.teacher_unique_id
    ORDER BY o.teacher_unique_id
    """
    journey = conn.execute(query).fetchdf()
    if journey.empty:
        return journey

    for col in [
        "first_month",
        "data_entrada_month",
        "first_session_start_ts",
        "first_session_end_ts",
        "first_session_first_event_ts",
        "first_session_first_meaningful_ts",
        "first_session_last_event_ts",
    ]:
        if col in journey.columns:
            journey[col] = pd.to_datetime(journey[col], errors="coerce")

    numeric_cols = [
        "returned_active_m1",
        "returned_any_session_m1",
        "returned_any_download_m1",
        "session_count_month",
        "total_session_minutes_month",
        "active_days_month",
        "activity_events_month",
        "content_views_month",
        "other_activity_non_download_events_month",
        "strict_download_count_month",
        "strict_value_flag",
        "used_mobile_flag",
        "used_desktop_flag",
        "first7d_events",
        "first7d_active_days",
        "first7d_sessions",
        "first7d_session_minutes",
        "session_without_interaction_flag",
        "cohort_variant_same_month_only",
        "cohort_variant_near_entry_0_1m",
        "first_session_duration_sec",
        "first_session_duration_min",
        "first_session_interactions",
        "first_session_downloads",
        "first_session_views",
        "first_session_other_actions",
        "first_session_navigation_events",
        "first_session_meaningful_events",
    ]
    for col in numeric_cols:
        if col in journey.columns:
            journey[col] = pd.to_numeric(journey[col], errors="coerce")

    journey["first_session_missing_flag"] = journey["first_session_row_hash"].isna().astype(int)
    journey["first_session_has_interaction_flag"] = (journey["first_session_interactions"].fillna(0) > 0).astype(int)
    journey["first_session_has_meaningful_action_flag"] = (journey["first_session_meaningful_events"].fillna(0) > 0).astype(int)
    journey["second_session_same_month_flag"] = (journey["session_count_month"].fillna(0) >= PRODUCT_THRESHOLDS["session_count_month"]).astype(int)

    journey["hit_threshold_sessions_flag"] = (journey["session_count_month"].fillna(0) >= PRODUCT_THRESHOLDS["session_count_month"]).astype(int)
    journey["hit_threshold_first7d_events_flag"] = (journey["first7d_events"].fillna(0) >= PRODUCT_THRESHOLDS["first7d_events"]).astype(int)
    journey["hit_threshold_content_views_flag"] = (journey["content_views_month"].fillna(0) >= PRODUCT_THRESHOLDS["content_views_month"]).astype(int)
    journey["hit_threshold_minutes_flag"] = (journey["total_session_minutes_month"].fillna(0) >= PRODUCT_THRESHOLDS["total_session_minutes_month"]).astype(int)
    threshold_cols = [
        "hit_threshold_sessions_flag",
        "hit_threshold_first7d_events_flag",
        "hit_threshold_content_views_flag",
        "hit_threshold_minutes_flag",
    ]
    journey["activation_threshold_hits_count"] = journey[threshold_cols].sum(axis=1)
    journey["activation_core_3of4_flag"] = (journey["activation_threshold_hits_count"] >= 3).astype(int)

    def normalize_text(series: pd.Series, default: str = "missing") -> pd.Series:
        out = series.fillna(default).astype(str)
        return out.replace({"<missing>": default, "None": default, "nan": default}).str.strip().replace("", default)

    journey["first_session_entry_surface"] = normalize_text(
        journey["first_session_first_event_utm_source"].where(
            normalize_text(journey["first_session_first_event_utm_source"]) != "missing",
            journey["onboarding_first_utm_source"],
        )
    )
    journey["first_session_device_raw"] = normalize_text(
        journey["first_session_first_event_device"].where(
            normalize_text(journey["first_session_first_event_device"]) != "missing",
            journey["onboarding_first_device"],
        ),
        default="unknown",
    )

    def device_bucket(row: pd.Series) -> str:
        device = str(row["first_session_device_raw"]).strip().lower()
        if device in {"mobile", "desktop"}:
            return device
        if pd.to_numeric(row.get("used_mobile_flag", 0), errors="coerce") == 1 and pd.to_numeric(row.get("used_desktop_flag", 0), errors="coerce") == 1:
            return "mixed"
        if pd.to_numeric(row.get("used_mobile_flag", 0), errors="coerce") == 1:
            return "mobile"
        if pd.to_numeric(row.get("used_desktop_flag", 0), errors="coerce") == 1:
            return "desktop"
        return "unknown"

    journey["first_session_device_bucket"] = journey.apply(device_bucket, axis=1)

    journey["secs_to_first_interaction"] = (
        journey["first_session_first_event_ts"] - journey["first_session_start_ts"]
    ).dt.total_seconds()
    journey["secs_to_first_meaningful_action"] = (
        journey["first_session_first_meaningful_ts"] - journey["first_session_start_ts"]
    ).dt.total_seconds()

    def action_group(value: Any) -> str:
        text = str(value).strip().lower() if value is not None else "missing"
        if text in {"download", "view", "create", "share"}:
            return text
        if text in {"missing", "<missing>", "none", "nan", ""}:
            return "missing"
        return "other"

    journey["first_session_first_meaningful_action_group"] = journey["first_session_first_meaningful_action"].apply(action_group)
    journey["first_session_first_event_action_group"] = journey["first_session_first_event_action"].apply(action_group)

    def exit_state(row: pd.Series) -> str:
        interactions = float(pd.to_numeric(row["first_session_interactions"], errors="coerce") or 0)
        downloads = float(pd.to_numeric(row["first_session_downloads"], errors="coerce") or 0)
        views = float(pd.to_numeric(row["first_session_views"], errors="coerce") or 0)
        other_actions = float(pd.to_numeric(row["first_session_other_actions"], errors="coerce") or 0)
        navigation = float(pd.to_numeric(row["first_session_navigation_events"], errors="coerce") or 0)
        if interactions <= 0:
            return "session_end_without_interaction"
        if downloads > 0:
            return "ended_after_download"
        if views > 0 and other_actions <= 0:
            return "ended_after_view_only"
        if other_actions > 0 and downloads <= 0:
            return "ended_after_activity_no_download"
        if navigation >= interactions:
            return "navigation_only"
        return "other_exit_state"

    journey["first_session_exit_state"] = journey.apply(exit_state, axis=1)

    def journey_label(row: pd.Series) -> str:
        if int(row["first_session_missing_flag"]) == 1:
            return "missing_first_session"
        if int(row["first_session_has_interaction_flag"]) == 0:
            return "session_without_interaction"
        if int(row["second_session_same_month_flag"]) == 0 and pd.to_numeric(row["first_session_downloads"], errors="coerce") > 0:
            return "one_session_download_no_repeat"
        if int(row["second_session_same_month_flag"]) == 1 and pd.to_numeric(row["first_session_downloads"], errors="coerce") > 0:
            return "download_then_repeat"
        if int(row["second_session_same_month_flag"]) == 0 and pd.to_numeric(row["first_session_meaningful_events"], errors="coerce") > 0:
            return "one_session_activity_no_repeat"
        if int(row["second_session_same_month_flag"]) == 1 and pd.to_numeric(row["first_session_meaningful_events"], errors="coerce") > 0:
            return "activity_then_repeat"
        return "other_first_session_journey"

    journey["journey_pattern_label"] = journey.apply(journey_label, axis=1)
    journey["step_sequence_first5"] = (
        normalize_text(journey["step_1_token"])
        + ">"
        + normalize_text(journey["step_2_token"])
        + ">"
        + normalize_text(journey["step_3_token"])
        + ">"
        + normalize_text(journey["step_4_token"])
        + ">"
        + normalize_text(journey["step_5_token"])
    )
    journey["first_session_entry_surface_top"] = compress_top(journey["first_session_entry_surface"], topn=10)
    return journey


def load_active_teacher_months(conn: Any) -> pd.DataFrame:
    query = """
    SELECT
      tm.teacher_unique_id,
      tm.month,
      COALESCE(dt.utm_group, 'missing') AS utm_group,
      tm.session_count_month,
      tm.total_session_minutes_month,
      tm.active_days_month,
      tm.activity_events_month,
      tm.strict_download_count_month,
      tm.content_views_month,
      tm.returned_active_m1,
      tm.returned_any_download_m1,
      tm.next_month_observed_flag,
      tm.lifetime_active_months,
      tm.active_user_flag
    FROM fct_teacher_month tm
    INNER JOIN dim_teacher dt
      ON tm.teacher_unique_id = dt.teacher_unique_id
    WHERE COALESCE(tm.active_user_flag, 0) = 1
    ORDER BY tm.teacher_unique_id, tm.month
    """
    df = conn.execute(query).fetchdf()
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    for col in [
        "session_count_month",
        "total_session_minutes_month",
        "active_days_month",
        "activity_events_month",
        "strict_download_count_month",
        "content_views_month",
        "returned_active_m1",
        "returned_any_download_m1",
        "next_month_observed_flag",
        "lifetime_active_months",
        "active_user_flag",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_heavy_month_flag(active_months: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if active_months.empty:
        return active_months.copy(), float("nan")
    heavy = active_months.copy()
    feature_cols = [
        "activity_events_month",
        "active_days_month",
        "total_session_minutes_month",
        "strict_download_count_month",
    ]
    transformed: Dict[str, pd.Series] = {}
    for col in feature_cols:
        transformed[col] = np.log1p(pd.to_numeric(heavy[col], errors="coerce").fillna(0))
    score = 0.0
    for col in feature_cols:
        series = transformed[col]
        std = float(series.std(ddof=0))
        denom = std if std > 0 else 1.0
        score = score + (series - float(series.mean())) / denom
    heavy["heavy_intensity_score"] = pd.to_numeric(score, errors="coerce")
    threshold = float(heavy["heavy_intensity_score"].quantile(0.90)) if not heavy.empty else float("nan")
    heavy["heavy_month_flag"] = (heavy["heavy_intensity_score"] >= threshold).astype(int)
    heavy["abandoned_after_heavy_flag"] = (
        (heavy["heavy_month_flag"] == 1)
        & (heavy["next_month_observed_flag"].fillna(0) == 1)
        & (heavy["returned_active_m1"].fillna(0) == 0)
    ).astype(int)
    return heavy, threshold


def build_heavy_abandonment_outputs(active_months: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    heavy_df, _ = add_heavy_month_flag(active_months)
    heavy_events = heavy_df[heavy_df["heavy_month_flag"] == 1].copy()
    if heavy_events.empty:
        empty_summary = pd.DataFrame()
        empty_patterns = pd.DataFrame()
        return empty_summary, empty_patterns

    heavy_events = heavy_events.sort_values(["teacher_unique_id", "month"]).reset_index(drop=True)
    first_heavy = heavy_events.groupby("teacher_unique_id", as_index=False).head(1).copy()
    first_heavy = first_heavy.rename(
        columns={
            "month": "first_heavy_month",
            "lifetime_active_months": "lifetime_active_months_at_first_heavy",
            "heavy_intensity_score": "first_heavy_intensity_score",
        }
    )
    last_heavy = heavy_events.groupby("teacher_unique_id", as_index=False).tail(1).copy()
    last_heavy = last_heavy.rename(
        columns={
            "month": "last_heavy_month",
            "lifetime_active_months": "lifetime_active_months_at_last_heavy",
            "heavy_intensity_score": "last_heavy_intensity_score",
            "returned_active_m1": "returned_after_last_heavy_m1",
            "returned_any_download_m1": "returned_any_download_after_last_heavy_m1",
            "next_month_observed_flag": "last_heavy_next_month_observed_flag",
            "abandoned_after_heavy_flag": "abandoned_after_last_heavy_flag",
            "session_count_month": "last_heavy_session_count_month",
            "total_session_minutes_month": "last_heavy_total_session_minutes_month",
            "content_views_month": "last_heavy_content_views_month",
            "activity_events_month": "last_heavy_activity_events_month",
            "utm_group": "last_heavy_utm_group",
        }
    )
    agg = (
        heavy_events.groupby("teacher_unique_id", dropna=False)
        .agg(
            heavy_months_count=("month", "count"),
            avg_heavy_intensity_score=("heavy_intensity_score", "mean"),
            max_heavy_intensity_score=("heavy_intensity_score", "max"),
            heavy_months_returned_active_rate=("returned_active_m1", "mean"),
        )
        .reset_index()
    )
    summary = agg.merge(
        first_heavy[
            [
                "teacher_unique_id",
                "first_heavy_month",
                "lifetime_active_months_at_first_heavy",
                "first_heavy_intensity_score",
                "utm_group",
            ]
        ].rename(columns={"utm_group": "first_heavy_utm_group"}),
        on="teacher_unique_id",
        how="left",
    )
    summary = summary.merge(
        last_heavy[
            [
                "teacher_unique_id",
                "last_heavy_month",
                "lifetime_active_months_at_last_heavy",
                "last_heavy_intensity_score",
                "returned_after_last_heavy_m1",
                "returned_any_download_after_last_heavy_m1",
                "last_heavy_next_month_observed_flag",
                "abandoned_after_last_heavy_flag",
                "last_heavy_session_count_month",
                "last_heavy_total_session_minutes_month",
                "last_heavy_content_views_month",
                "last_heavy_activity_events_month",
                "last_heavy_utm_group",
            ]
        ],
        on="teacher_unique_id",
        how="left",
    )
    summary["heavy_user_status"] = np.select(
        [
            summary["abandoned_after_last_heavy_flag"].fillna(0) == 1,
            summary["last_heavy_next_month_observed_flag"].fillna(0) == 1,
        ],
        [
            "abandoned_after_last_heavy",
            "returned_after_last_heavy",
        ],
        default="last_heavy_censored",
    )

    eligible_heavy = heavy_events[heavy_events["next_month_observed_flag"].fillna(0) == 1].copy()
    eligible_heavy["month_str"] = eligible_heavy["month"].dt.strftime("%Y-%m")
    pattern_frames: List[pd.DataFrame] = []

    overall = pd.DataFrame(
        [
            {
                "slice_type": "overall",
                "slice_value": "__all__",
                "heavy_months": int(len(eligible_heavy)),
                "teachers": int(eligible_heavy["teacher_unique_id"].nunique()),
                "returned_active_rate": float(eligible_heavy["returned_active_m1"].mean()) if not eligible_heavy.empty else np.nan,
                "abandoned_after_heavy_rate": float(eligible_heavy["abandoned_after_heavy_flag"].mean()) if not eligible_heavy.empty else np.nan,
                "avg_heavy_intensity_score": float(eligible_heavy["heavy_intensity_score"].mean()) if not eligible_heavy.empty else np.nan,
            }
        ]
    )
    pattern_frames.append(overall)

    for slice_type, slice_col in [
        ("lifetime_active_months", "lifetime_active_months"),
        ("month", "month_str"),
        ("utm_group", "utm_group"),
    ]:
        grouped = (
            eligible_heavy.groupby(slice_col, dropna=False)
            .agg(
                heavy_months=("teacher_unique_id", "count"),
                teachers=("teacher_unique_id", "nunique"),
                returned_active_rate=("returned_active_m1", "mean"),
                abandoned_after_heavy_rate=("abandoned_after_heavy_flag", "mean"),
                avg_heavy_intensity_score=("heavy_intensity_score", "mean"),
            )
            .reset_index()
            .rename(columns={slice_col: "slice_value"})
        )
        grouped["slice_type"] = slice_type
        pattern_frames.append(grouped)

    patterns = pd.concat(pattern_frames, ignore_index=True)
    patterns["slice_value"] = patterns["slice_value"].astype(str)
    return summary, patterns.sort_values(["slice_type", "heavy_months"], ascending=[True, False]).reset_index(drop=True)


def build_journey_summary(journey: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for cohort_variant in COHORT_VARIANTS:
        subset = select_cohort_variant(journey, cohort_variant)
        if subset.empty:
            continue
        grouped = (
            subset.groupby("journey_pattern_label", dropna=False)
            .agg(
                teachers=("teacher_unique_id", "nunique"),
                returned_active_rate=("returned_active_m1", "mean"),
                returned_any_session_rate=("returned_any_session_m1", "mean"),
                avg_first_session_minutes=("first_session_duration_min", "mean"),
                avg_first7d_events=("first7d_events", "mean"),
                avg_activation_threshold_hits=("activation_threshold_hits_count", "mean"),
                share_mobile=("first_session_device_bucket", lambda s: (s.astype(str) == "mobile").mean()),
            )
            .reset_index()
        )
        grouped["cohort_variant"] = cohort_variant
        grouped["share_cohort"] = grouped["teachers"] / max(1, subset["teacher_unique_id"].nunique())
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["cohort_variant", "teachers"], ascending=[True, False]).reset_index(drop=True)


def build_step_dropoff(journey: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    step_defs = [
        ("first_session_present", lambda df: df["first_session_missing_flag"] == 0),
        ("interaction_in_first_session", lambda df: df["first_session_has_interaction_flag"] == 1),
        ("meaningful_action_in_first_session", lambda df: df["first_session_has_meaningful_action_flag"] == 1),
        ("second_session_same_month", lambda df: df["second_session_same_month_flag"] == 1),
        ("activation_core_3of4", lambda df: df["activation_core_3of4_flag"] == 1),
        ("returned_active_m1", lambda df: pd.to_numeric(df["returned_active_m1"], errors="coerce").fillna(0) == 1),
    ]

    base = journey.copy()
    base["first_session_entry_surface_top"] = compress_top(base["first_session_entry_surface"], topn=8)

    for cohort_variant in COHORT_VARIANTS:
        cohort = select_cohort_variant(base, cohort_variant)
        if cohort.empty:
            continue
        slice_specs = [("overall", None), ("entry_surface", "first_session_entry_surface_top"), ("device", "first_session_device_bucket")]
        for slice_type, slice_col in slice_specs:
            if slice_col is None:
                slice_groups = [("__all__", cohort)]
            else:
                slice_groups = list(cohort.groupby(slice_col, dropna=False))
            for slice_value, group in slice_groups:
                total = int(group["teacher_unique_id"].nunique())
                if total < 50 and slice_type != "overall":
                    continue
                cumulative_mask = pd.Series(True, index=group.index)
                prev_n = total
                for order, (step_name, cond_fn) in enumerate(step_defs, start=1):
                    step_mask = cumulative_mask & cond_fn(group)
                    n_step = int(group.loc[step_mask, "teacher_unique_id"].nunique())
                    rows.append(
                        {
                            "cohort_variant": cohort_variant,
                            "slice_type": slice_type,
                            "slice_value": str(slice_value),
                            "step_order": order,
                            "step_name": step_name,
                            "users_reaching_step": n_step,
                            "share_of_cohort": n_step / max(1, total),
                            "share_of_previous_step": n_step / max(1, prev_n),
                            "returned_active_rate_within_step": float(pd.to_numeric(group.loc[step_mask, "returned_active_m1"], errors="coerce").mean()) if n_step > 0 else np.nan,
                        }
                    )
                    cumulative_mask = step_mask
                    prev_n = n_step
    return pd.DataFrame(rows).sort_values(["cohort_variant", "slice_type", "slice_value", "step_order"]).reset_index(drop=True)


def build_risk_cohorts(journey: pd.DataFrame, heavy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    onboarding = select_cohort_variant(journey, "near_entry_0_1m").copy()
    onboarding["reference_month"] = onboarding["first_month"]

    cohort_specs = [
        (
            "onboarding_session_without_interaction",
            10,
            "onboarding",
            "tracking_risk",
            onboarding["first_session_has_interaction_flag"] == 0,
            "Primeira sessao observada sem interacao capturada.",
            "Instrumentar e simplificar a primeira acao observavel da sessao inicial.",
        ),
        (
            "onboarding_one_session_download_no_repeat",
            20,
            "onboarding",
            "supported_robust",
            (onboarding["first_session_downloads"].fillna(0) > 0) & (onboarding["second_session_same_month_flag"] == 0),
            "Baixou na primeira sessao, mas nao construiu repeticao no mesmo mes.",
            "Trocar foco de download pontual por loop de continuidade e segunda sessao.",
        ),
        (
            "onboarding_one_session_activity_no_repeat",
            30,
            "onboarding",
            "supported_robust",
            (onboarding["first_session_downloads"].fillna(0) <= 0)
            & (onboarding["first_session_has_meaningful_action_flag"] == 1)
            & (onboarding["second_session_same_month_flag"] == 0),
            "Teve acao util na primeira sessao, mas nao repetiu o uso no mesmo mes.",
            "Introduzir ganchos de retorno apos a primeira acao util sem depender de download.",
        ),
        (
            "onboarding_low_activation_no_repeat",
            40,
            "onboarding",
            "supported_robust",
            (onboarding["activation_threshold_hits_count"] <= 1) & (pd.to_numeric(onboarding["returned_active_m1"], errors="coerce").fillna(0) == 0),
            "Nao atingiu sinais minimos de ativacao e nao retornou no mes seguinte.",
            "Usar metas de ativacao comportamental no onboarding e nudges na primeira semana.",
        ),
    ]

    base_cols = [
        "teacher_unique_id",
        "utm_group",
        "first_session_entry_surface",
        "first_session_device_bucket",
        "journey_pattern_label",
        "session_count_month",
        "returned_active_m1",
        "returned_any_session_m1",
        "activation_threshold_hits_count",
        "reference_month",
    ]

    for label, priority_rank, user_stage, evidence_class, mask, reason, action in cohort_specs:
        subset = onboarding.loc[mask, base_cols].copy()
        if subset.empty:
            continue
        subset["risk_cohort_label"] = label
        subset["priority_rank"] = priority_rank
        subset["user_stage"] = user_stage
        subset["evidence_class"] = evidence_class
        subset["reason"] = reason
        subset["recommended_action"] = action
        rows.append(subset)

    if not heavy_summary.empty:
        heavy = heavy_summary[heavy_summary["abandoned_after_last_heavy_flag"].fillna(0) == 1].copy()
        if not heavy.empty:
            heavy_rows = heavy[
                [
                    "teacher_unique_id",
                    "last_heavy_utm_group",
                    "last_heavy_month",
                    "lifetime_active_months_at_last_heavy",
                    "heavy_months_count",
                    "returned_after_last_heavy_m1",
                ]
            ].copy()
            heavy_rows = heavy_rows.rename(
                columns={
                    "last_heavy_utm_group": "utm_group",
                    "last_heavy_month": "reference_month",
                    "returned_after_last_heavy_m1": "returned_active_m1",
                }
            )
            heavy_rows["first_session_entry_surface"] = "not_applicable"
            heavy_rows["first_session_device_bucket"] = "not_applicable"
            heavy_rows["journey_pattern_label"] = "heavy_user_abandoned_after_last_heavy"
            heavy_rows["session_count_month"] = np.nan
            heavy_rows["returned_any_session_m1"] = np.nan
            heavy_rows["activation_threshold_hits_count"] = np.nan
            heavy_rows["risk_cohort_label"] = "heavy_user_abandoned_after_last_heavy"
            heavy_rows["priority_rank"] = 50
            heavy_rows["user_stage"] = "post_activation"
            heavy_rows["evidence_class"] = "supported_correlational"
            heavy_rows["reason"] = "Usuario atingiu mes heavy e depois nao voltou ativo no mes seguinte."
            heavy_rows["recommended_action"] = "Acionar protecao de heavy users logo apos meses de pico, especialmente nos 2 primeiros meses ativos."
            rows.append(
                heavy_rows[
                    base_cols
                    + [
                        "risk_cohort_label",
                        "priority_rank",
                        "user_stage",
                        "evidence_class",
                        "reason",
                        "recommended_action",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["reference_month"] = pd.to_datetime(out["reference_month"], errors="coerce")
    out = out.sort_values(["priority_rank", "teacher_unique_id", "reference_month"]).reset_index(drop=True)
    out["is_primary_cohort_for_teacher"] = (
        out.groupby("teacher_unique_id").cumcount() == 0
    ).astype(int)
    return out


def build_decision_table(
    journey: pd.DataFrame,
    heavy_summary: pd.DataFrame,
    heavy_patterns: pd.DataFrame,
) -> pd.DataFrame:
    onboarding = select_cohort_variant(journey, "near_entry_0_1m").copy()
    rows: List[Dict[str, Any]] = []

    def top_values(series: pd.Series, topn: int = 3) -> str:
        counts = compress_top(series, topn=topn).value_counts(dropna=False).head(topn)
        return ", ".join(counts.index.astype(str).tolist()) if not counts.empty else "none"

    def add_issue(
        issue_id: str,
        evidence_class: str,
        decision_direction: str,
        subset: pd.DataFrame,
        comparator: pd.DataFrame,
        issue_stage: str,
        decision: str,
        note: str,
        target_slice: str,
        outcome_col: str = "returned_active_m1",
    ) -> None:
        affected_teachers = int(subset["teacher_unique_id"].nunique()) if not subset.empty else 0
        affected_share = affected_teachers / max(1, onboarding["teacher_unique_id"].nunique())
        subset_rate = float(pd.to_numeric(subset[outcome_col], errors="coerce").mean()) if not subset.empty else np.nan
        comparator_rate = float(pd.to_numeric(comparator[outcome_col], errors="coerce").mean()) if not comparator.empty else np.nan
        delta_pp = (comparator_rate - subset_rate) * 100 if pd.notna(subset_rate) and pd.notna(comparator_rate) else np.nan
        priority_score = affected_teachers * max(0.0, (comparator_rate - subset_rate) if pd.notna(delta_pp) else 0.0)
        rows.append(
            {
                "issue_id": issue_id,
                "issue_stage": issue_stage,
                "evidence_class": evidence_class,
                "decision_direction": decision_direction,
                "affected_teachers": affected_teachers,
                "affected_share_cohort": affected_share,
                "subset_return_active_rate": subset_rate,
                "comparator_return_active_rate": comparator_rate,
                "delta_return_active_pp": delta_pp,
                "priority_score": priority_score,
                "target_slice": target_slice,
                "decision": decision,
                "note": note,
            }
        )

    no_interaction = onboarding[onboarding["first_session_has_interaction_flag"] == 0].copy()
    any_interaction = onboarding[onboarding["first_session_has_interaction_flag"] == 1].copy()
    add_issue(
        "first_session_no_interaction",
        "tracking_risk",
        "do",
        no_interaction,
        any_interaction,
        "onboarding",
        "Instrumentar e redesenhar a primeira sessao para garantir uma primeira acao observavel e util.",
        "Problema misto de UX e tracking; atacar primeiro superficies de entrada com maior concentracao.",
        top_values(no_interaction["first_session_entry_surface"]),
    )

    one_session_download = onboarding[
        (onboarding["first_session_downloads"].fillna(0) > 0) & (onboarding["second_session_same_month_flag"] == 0)
    ].copy()
    repeat_after_download = onboarding[
        (onboarding["first_session_downloads"].fillna(0) > 0) & (onboarding["second_session_same_month_flag"] == 1)
    ].copy()
    add_issue(
        "one_session_download_no_repeat",
        "supported_robust",
        "do",
        one_session_download,
        repeat_after_download,
        "onboarding",
        "Parar de tratar download inicial como sucesso final; otimizar o caminho para segunda sessao.",
        "Usuarios que baixam e nao repetem no mesmo mes se comportam como uso pontual, nao como ativacao.",
        top_values(one_session_download["first_session_entry_surface"]),
    )

    one_session_activity = onboarding[
        (onboarding["first_session_downloads"].fillna(0) <= 0)
        & (onboarding["first_session_has_meaningful_action_flag"] == 1)
        & (onboarding["second_session_same_month_flag"] == 0)
    ].copy()
    repeat_after_activity = onboarding[
        (onboarding["first_session_downloads"].fillna(0) <= 0)
        & (onboarding["first_session_has_meaningful_action_flag"] == 1)
        & (onboarding["second_session_same_month_flag"] == 1)
    ].copy()
    add_issue(
        "one_session_activity_no_repeat",
        "supported_robust",
        "do",
        one_session_activity,
        repeat_after_activity,
        "onboarding",
        "Criar mecanismos de continuidade apos a primeira acao util sem depender de download.",
        "O problema nao e falta de valor inicial; e quebra no loop de repeticao.",
        top_values(one_session_activity["first_session_entry_surface"]),
    )

    low_activation = onboarding[onboarding["activation_threshold_hits_count"] <= 1].copy()
    healthy_activation = onboarding[onboarding["activation_core_3of4_flag"] == 1].copy()
    add_issue(
        "low_activation_first_week",
        "supported_robust",
        "do",
        low_activation,
        healthy_activation,
        "onboarding",
        "Usar metas de ativacao comportamental como KPI de onboarding: 2 sessoes, 12 eventos em 7 dias, 6 views e 16.684 min.",
        "A maior diferenca de retorno esta entre baixa intensidade inicial e ativacao profunda.",
        top_values(low_activation["first_session_entry_surface"]),
    )

    heavy_eligible = heavy_summary[heavy_summary["last_heavy_next_month_observed_flag"].fillna(0) == 1].copy()
    heavy_early = heavy_eligible[
        pd.to_numeric(heavy_eligible["lifetime_active_months_at_last_heavy"], errors="coerce").fillna(np.inf) <= 2
    ].copy()
    heavy_mature = heavy_eligible[
        pd.to_numeric(heavy_eligible["lifetime_active_months_at_last_heavy"], errors="coerce").fillna(-np.inf) >= 3
    ].copy()
    add_issue(
        "heavy_user_abandonment_after_peak",
        "supported_correlational",
        "do",
        heavy_early,
        heavy_mature,
        "post_activation",
        "Criar protecao de churn para heavy users, especialmente logo apos o primeiro mes heavy.",
        "Mesmo meses heavy ainda perdem usuarios; a perda e maior nos primeiros 1-2 meses ativos heavy.",
        top_values(heavy_early.get("last_heavy_utm_group", pd.Series(dtype=str))),
        outcome_col="returned_after_last_heavy_m1",
    )

    add_issue(
        "do_not_force_initial_download",
        "not_supported",
        "do_not_do",
        onboarding,
        onboarding,
        "onboarding",
        "Nao usar download inicial como meta principal de UX/onboarding.",
        "A evidencia controlada nao sustenta download inicial como driver melhor que atividade util sem download.",
        "global",
    )

    decisions = pd.DataFrame(rows)
    if decisions.empty:
        return decisions
    decisions = decisions.sort_values(["priority_score", "affected_teachers"], ascending=[False, False]).reset_index(drop=True)
    decisions["priority_rank"] = np.arange(1, len(decisions) + 1)

    if not heavy_patterns.empty:
        early_heavy = heavy_patterns[
            (heavy_patterns["slice_type"] == "lifetime_active_months")
            & (heavy_patterns["slice_value"].isin(["1.0", "2.0", "1", "2"]))
        ]
        if not early_heavy.empty:
            early_heavy_churn = float(early_heavy["abandoned_after_heavy_rate"].mean())
            decisions.loc[
                decisions["issue_id"] == "heavy_user_abandonment_after_peak",
                "note",
            ] = decisions.loc[
                decisions["issue_id"] == "heavy_user_abandonment_after_peak",
                "note",
            ] + f" churn_medio_nos_2_primeiros_meses_heavy={early_heavy_churn:.4f}"
    return decisions


def build_summary_payload(
    journey: pd.DataFrame,
    heavy_summary: pd.DataFrame,
    risk_cohorts: pd.DataFrame,
    decisions: pd.DataFrame,
) -> Dict[str, Any]:
    onboarding = select_cohort_variant(journey, "near_entry_0_1m")
    heavy_abandoned = heavy_summary[heavy_summary["abandoned_after_last_heavy_flag"].fillna(0) == 1].copy() if not heavy_summary.empty else pd.DataFrame()
    top_decisions = decisions.sort_values("priority_rank").head(5).copy() if not decisions.empty else pd.DataFrame()
    return {
        "generated_at_utc": utc_now_iso(),
        "onboarding_rows_near_entry": int(len(onboarding)),
        "journey_rows": int(len(journey)),
        "session_without_interaction_teachers": int(onboarding[onboarding["first_session_has_interaction_flag"] == 0]["teacher_unique_id"].nunique()) if not onboarding.empty else 0,
        "one_session_download_no_repeat_teachers": int(
            onboarding[
                (onboarding["first_session_downloads"].fillna(0) > 0)
                & (onboarding["second_session_same_month_flag"] == 0)
            ]["teacher_unique_id"].nunique()
        )
        if not onboarding.empty
        else 0,
        "activation_core_3of4_teachers": int(onboarding[onboarding["activation_core_3of4_flag"] == 1]["teacher_unique_id"].nunique()) if not onboarding.empty else 0,
        "heavy_users_total": int(len(heavy_summary)) if not heavy_summary.empty else 0,
        "heavy_users_abandoned_after_last_heavy": int(len(heavy_abandoned)) if not heavy_abandoned.empty else 0,
        "primary_risk_users": int(risk_cohorts[risk_cohorts["is_primary_cohort_for_teacher"] == 1]["teacher_unique_id"].nunique()) if not risk_cohorts.empty else 0,
        "top_decisions": top_decisions[
            ["priority_rank", "issue_id", "affected_teachers", "delta_return_active_pp", "decision_direction"]
        ].to_dict(orient="records")
        if not top_decisions.empty
        else [],
    }


def write_summary_markdown(path: Path, summary: Dict[str, Any], decisions: pd.DataFrame) -> None:
    lines = [
        "# UX diagnostico v2",
        "",
        f"- Gerado em UTC: {summary['generated_at_utc']}",
        f"- Linhas na mart de jornada: {summary['journey_rows']}",
        f"- Coorte near_entry_0_1m: {summary['onboarding_rows_near_entry']}",
        f"- Usuarios com primeira sessao sem interacao: {summary['session_without_interaction_teachers']}",
        f"- Usuarios com download na primeira sessao e sem repeticao no mes: {summary['one_session_download_no_repeat_teachers']}",
        f"- Usuarios com ativacao core 3/4: {summary['activation_core_3of4_teachers']}",
        f"- Heavy users totais: {summary['heavy_users_total']}",
        f"- Heavy users que abandonaram apos o ultimo mes heavy: {summary['heavy_users_abandoned_after_last_heavy']}",
        f"- Usuarios em coortes primarias de risco: {summary['primary_risk_users']}",
        "",
        "## Top Decisions",
    ]
    if decisions.empty:
        lines.append("- none")
    else:
        for _, row in decisions.sort_values("priority_rank").head(6).iterrows():
            lines.append(
                f"- `#{int(row['priority_rank'])}` `{row['issue_id']}` | `{row['decision_direction']}` | afetados={int(row['affected_teachers'])} | delta_pp={float(row['delta_return_active_pp']):.2f} | {row['decision']}"
            )
    write_markdown(path, lines)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    conn = connect_duckdb(cfg)
    try:
        require_tables(
            conn,
            [
                "mart_teacher_onboarding_first_month_v2",
                "fct_session_clean",
                "fct_interaction_clean",
                "fct_teacher_month",
                "dim_teacher",
            ],
        )

        journey = build_first_session_journey_mart(conn)
        active_months = load_active_teacher_months(conn)
        heavy_summary, heavy_patterns = build_heavy_abandonment_outputs(active_months)
        journey_summary = build_journey_summary(journey)
        step_dropoff = build_step_dropoff(journey)
        risk_cohorts = build_risk_cohorts(journey, heavy_summary)
        decision_table = build_decision_table(journey, heavy_summary, heavy_patterns)

        outputs: Dict[str, pd.DataFrame] = {
            "mart_teacher_first_session_journey_v2": journey,
            "mart_teacher_heavy_abandonment_v2": heavy_summary,
            "analytics_heavy_abandonment_patterns_v2": heavy_patterns,
            "analytics_ux_journey_summary_v2": journey_summary,
            "analytics_ux_step_dropoff_v2": step_dropoff,
            "analytics_ux_risk_cohorts_v2": risk_cohorts,
            "analytics_ux_decision_table_v2": decision_table,
        }
        for name, df in outputs.items():
            persist_output(conn, cfg, name, df)

        summary = build_summary_payload(journey, heavy_summary, risk_cohorts, decision_table)
        write_json(cfg.output_dir / "json" / "ux_diagnostic_summary_v2.json", summary)
        write_summary_markdown(cfg.output_dir / "audit" / "ux_diagnostic_summary_v2.md", summary, decision_table)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
