#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    STRICT_VALUE_EVENTS,
    V2Config,
    build_config,
    connect_duckdb,
    persist_df_to_duckdb,
    register_raw_views,
    setup_logging,
    sql_event_action_expr,
    utc_now_iso,
    write_df_bundle,
    write_json,
    write_markdown,
)


COHORT_VARIANTS: Sequence[str] = ("same_month_only", "near_entry_0_1m")
OUTCOME_VARIANTS: Sequence[str] = ("returned_active_m1", "returned_any_session_m1")
PRODUCT_THRESHOLDS: Dict[str, float | None] = {
    "session_count_month": 2.0,
    "first7d_events": 12.0,
    "content_views_month": 6.0,
    "total_session_minutes_month": 16.684,
    "active_days_month": None,
}
GAP_SHARE_ALERT_THRESHOLD = 0.10
GAP_TOP1_UTM_CONCENTRATION_THRESHOLD = 0.50
DEVICE_COMPLETENESS_MIN = 0.50
FIRST_MISSING_ALERT_JUMP = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 06 v2: onboarding validado com auditoria e hipoteses.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def persist_output(conn: Any, cfg: V2Config, name: str, df: pd.DataFrame) -> None:
    persist_df_to_duckdb(conn, name, df)
    write_df_bundle(cfg.output_dir, name, df)


def require_tables(conn: Any, table_names: Sequence[str]) -> None:
    existing = {
        row[0]
        for row in conn.execute("SHOW TABLES").fetchall()
    }
    missing = [name for name in table_names if name not in existing]
    if missing:
        raise RuntimeError(
            "Tabelas obrigatorias ausentes para etapa_06_onboarding_validado_v2.py: "
            + ", ".join(sorted(missing))
        )


def build_onboarding_mart(conn: Any) -> pd.DataFrame:
    query = """
    WITH first_touch AS (
      SELECT
        teacher_unique_id,
        MIN(month) AS first_month
      FROM fct_teacher_month
      WHERE COALESCE(session_count_month, 0) > 0
         OR COALESCE(interaction_rows_month, 0) > 0
      GROUP BY 1
    ),
    first_session_ranked AS (
      SELECT
        s.teacher_unique_id,
        s.session_start_ts,
        s.duration_min,
        s.duration_sec,
        ROW_NUMBER() OVER (
          PARTITION BY s.teacher_unique_id
          ORDER BY s.session_start_ts, s.session_row_hash
        ) AS rn,
        MIN(s.session_start_ts) OVER (PARTITION BY s.teacher_unique_id) AS first_session_ts
      FROM fct_session_clean s
      INNER JOIN first_touch ft
        ON s.teacher_unique_id = ft.teacher_unique_id
       AND s.session_month = ft.first_month
    ),
    first_session_anchor AS (
      SELECT
        teacher_unique_id,
        MIN(session_start_ts) AS first_session_ts
      FROM first_session_ranked
      GROUP BY 1
    ),
    first_event_ranked AS (
      SELECT
        i.teacher_unique_id,
        i.interaction_ts,
        i.event_type,
        i.event_family,
        i.event_action,
        i.device_group,
        i.utm_source,
        i.is_download_event,
        i.is_visualization_event,
        i.is_other_activity_non_download_event,
        ROW_NUMBER() OVER (
          PARTITION BY i.teacher_unique_id
          ORDER BY i.interaction_ts, i.interaction_row_hash
        ) AS rn,
        MIN(i.interaction_ts) OVER (PARTITION BY i.teacher_unique_id) AS first_interaction_ts
      FROM fct_interaction_clean i
      INNER JOIN first_touch ft
        ON i.teacher_unique_id = ft.teacher_unique_id
       AND i.interaction_month = ft.first_month
    ),
    first_interaction_anchor AS (
      SELECT
        teacher_unique_id,
        MIN(first_interaction_ts) AS first_interaction_ts
      FROM first_event_ranked
      GROUP BY 1
    ),
    onboarding_anchor AS (
      SELECT
        ft.teacher_unique_id,
        COALESCE(fsa.first_session_ts, fia.first_interaction_ts) AS onboarding_anchor_ts
      FROM first_touch ft
      LEFT JOIN first_session_anchor fsa
        ON ft.teacher_unique_id = fsa.teacher_unique_id
      LEFT JOIN first_interaction_anchor fia
        ON ft.teacher_unique_id = fia.teacher_unique_id
    ),
    first_event_summary AS (
      SELECT
        fer.teacher_unique_id,
        MAX(CASE WHEN fer.rn = 1 THEN fer.event_type END) AS first_event_type,
        MAX(CASE WHEN fer.rn = 1 THEN fer.event_family END) AS first_event_family,
        MAX(CASE WHEN fer.rn = 1 THEN fer.event_action END) AS first_event_action,
        MAX(CASE WHEN fer.rn = 1 THEN fer.device_group END) AS first_device,
        MAX(CASE WHEN fer.rn = 1 THEN fer.utm_source END) AS first_utm_source,
        SUM(CASE WHEN fer.rn <= 3 THEN fer.is_download_event ELSE 0 END) AS first3_downloads,
        SUM(CASE WHEN fer.rn <= 3 THEN fer.is_visualization_event ELSE 0 END) AS first3_views,
        SUM(CASE WHEN fer.rn <= 3 THEN fer.is_other_activity_non_download_event ELSE 0 END) AS first3_other_actions,
        COUNT(*) FILTER (WHERE fer.interaction_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY) AS first7d_events,
        COUNT(DISTINCT CAST(fer.interaction_ts AS DATE)) FILTER (
          WHERE fer.interaction_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY
        ) AS first7d_active_days
      FROM first_event_ranked fer
      LEFT JOIN onboarding_anchor oa
        ON fer.teacher_unique_id = oa.teacher_unique_id
      GROUP BY 1
    ),
    first_session_summary AS (
      SELECT
        fsr.teacher_unique_id,
        MAX(CASE WHEN fsr.rn = 1 THEN fsr.duration_min END) AS first_session_minutes,
        COUNT(*) FILTER (WHERE fsr.session_start_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY) AS first7d_sessions,
        SUM(fsr.duration_min) FILTER (WHERE fsr.session_start_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY) AS first7d_session_minutes
      FROM first_session_ranked fsr
      LEFT JOIN onboarding_anchor oa
        ON fsr.teacher_unique_id = oa.teacher_unique_id
      GROUP BY 1
    ),
    next_month_session_flag AS (
      SELECT
        teacher_unique_id,
        month AS next_month,
        CASE WHEN COALESCE(session_count_month, 0) > 0 THEN 1.0 ELSE 0.0 END AS returned_any_session_m1
      FROM fct_teacher_month
    )
    SELECT
      tm.teacher_unique_id,
      tm.month AS first_month,
      DATE_TRUNC('month', dt.data_entrada) AS data_entrada_month,
      CASE
        WHEN dt.data_entrada IS NULL THEN NULL
        ELSE DATE_DIFF('month', DATE_TRUNC('month', dt.data_entrada), tm.month)
      END AS months_after_entry,
      'core_teacher_onboarding_first_month_v2' AS analysis_population,
      dt.population_status AS teacher_population_status,
      COALESCE(dt.utm_group, 'missing') AS utm_group,
      tm.returned_active_m1,
      COALESCE(nm.returned_any_session_m1, 0.0) AS returned_any_session_m1,
      tm.returned_any_download_m1,
      tm.session_count_month,
      tm.total_session_minutes_month,
      tm.active_days_month,
      tm.activity_events_month,
      tm.content_views_month,
      tm.other_activity_non_download_events_month,
      tm.strict_download_count_month,
      tm.strict_value_flag,
      tm.used_mobile_flag,
      tm.used_desktop_flag,
      tm.no_download_flag,
      tm.no_download_view_only_flag,
      tm.no_download_view_plus_action_flag,
      tm.no_download_action_only_flag,
      tm.session_exposed_no_activity_no_download_flag,
      tm.session_exposed_activity_no_download_flag,
      fe.first_event_type,
      fe.first_event_family,
      fe.first_event_action,
      fe.first_device,
      fe.first_utm_source,
      fe.first3_downloads,
      fe.first3_views,
      fe.first3_other_actions,
      fe.first7d_events,
      fe.first7d_active_days,
      fs.first_session_minutes,
      fs.first7d_sessions,
      fs.first7d_session_minutes,
      CASE
        WHEN fe.first_event_type IS NULL OR fe.first_event_type IN ('<missing>', 'missing') THEN 1
        ELSE 0
      END AS first_event_missing_flag,
      CASE
        WHEN fe.first_event_action IS NULL OR fe.first_event_action IN ('<missing>', 'missing', 'None') THEN 1
        ELSE 0
      END AS first_event_action_missing_flag,
      CASE
        WHEN fe.first_utm_source IS NULL OR fe.first_utm_source IN ('<missing>', 'missing') THEN 1
        ELSE 0
      END AS first_utm_missing_flag,
      CASE
        WHEN fe.first_device IS NULL OR fe.first_device IN ('<missing>', 'missing', 'unknown') THEN 1
        ELSE 0
      END AS first_device_missing_flag,
      CASE
        WHEN COALESCE(tm.session_count_month, 0) > 0 AND COALESCE(tm.interaction_rows_month, 0) = 0 THEN 1
        ELSE 0
      END AS session_without_interaction_flag,
      CASE
        WHEN COALESCE(tm.used_mobile_flag, 0) = 1 AND COALESCE(tm.used_desktop_flag, 0) = 1 THEN 'mixed'
        WHEN COALESCE(tm.used_mobile_flag, 0) = 1 THEN 'mobile'
        WHEN COALESCE(tm.used_desktop_flag, 0) = 1 THEN 'desktop'
        ELSE 'no_observed_device'
      END AS device_observation_bucket,
      1 AS cohort_variant_all_first_observed,
      CASE
        WHEN dt.data_entrada IS NOT NULL
         AND DATE_DIFF('month', DATE_TRUNC('month', dt.data_entrada), tm.month) = 0 THEN 1
        ELSE 0
      END AS cohort_variant_same_month_only,
      CASE
        WHEN dt.data_entrada IS NOT NULL
         AND DATE_DIFF('month', DATE_TRUNC('month', dt.data_entrada), tm.month) BETWEEN 0 AND 1 THEN 1
        ELSE 0
      END AS cohort_variant_near_entry_0_1m
    FROM fct_teacher_month tm
    INNER JOIN first_touch ft
      ON tm.teacher_unique_id = ft.teacher_unique_id
     AND tm.month = ft.first_month
    INNER JOIN dim_teacher dt
      ON tm.teacher_unique_id = dt.teacher_unique_id
    LEFT JOIN next_month_session_flag nm
      ON tm.teacher_unique_id = nm.teacher_unique_id
     AND tm.next_month = nm.next_month
    LEFT JOIN first_event_summary fe
      ON tm.teacher_unique_id = fe.teacher_unique_id
    LEFT JOIN first_session_summary fs
      ON tm.teacher_unique_id = fs.teacher_unique_id
    WHERE tm.next_month_observed_flag = 1
    ORDER BY tm.teacher_unique_id
    """
    mart = conn.execute(query).fetchdf()
    if mart.empty:
        return mart

    mart["first_month"] = pd.to_datetime(mart["first_month"], errors="coerce")
    mart["data_entrada_month"] = pd.to_datetime(mart["data_entrada_month"], errors="coerce")
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
        "no_download_flag",
        "no_download_view_only_flag",
        "no_download_view_plus_action_flag",
        "no_download_action_only_flag",
        "session_exposed_no_activity_no_download_flag",
        "session_exposed_activity_no_download_flag",
        "first3_downloads",
        "first3_views",
        "first3_other_actions",
        "first7d_events",
        "first7d_active_days",
        "first_session_minutes",
        "first7d_sessions",
        "first7d_session_minutes",
        "first_event_missing_flag",
        "first_event_action_missing_flag",
        "first_utm_missing_flag",
        "first_device_missing_flag",
        "session_without_interaction_flag",
        "cohort_variant_all_first_observed",
        "cohort_variant_same_month_only",
        "cohort_variant_near_entry_0_1m",
    ]
    for col in numeric_cols:
        if col in mart.columns:
            mart[col] = pd.to_numeric(mart[col], errors="coerce")
    return mart


def build_raw_population_coverage(conn: Any) -> pd.DataFrame:
    query = """
    WITH raw_entries_enriched AS (
      SELECT
        'raw_entries' AS source_table,
        LOWER(COALESCE(e.user_type, 'missing')) AS user_type,
        CASE
          WHEN LOWER(COALESCE(e.user_type, '')) = 'registered' AND d.unique_id IS NOT NULL THEN 'core_teacher'
          WHEN LOWER(COALESCE(e.user_type, '')) = 'registered' AND d.unique_id IS NULL THEN 'shadow_registered'
          WHEN LOWER(COALESCE(e.user_type, '')) = 'anonymous' THEN 'shadow_anonymous'
          WHEN LOWER(COALESCE(e.user_type, '')) = 'seo' THEN 'shadow_seo'
          ELSE 'other'
        END AS population_bucket,
        e.unique_id AS source_unique_id,
        d.unique_id AS teacher_unique_id,
        e.data_inicio AS ts_start,
        e.data_fim AS ts_end,
        NULL::VARCHAR AS event_type,
        NULL::VARCHAR AS utm_source
      FROM raw_entries e
      LEFT JOIN raw_dim_teachers d USING(unique_id)
    ),
    raw_interactions_enriched AS (
      SELECT
        'raw_interactions' AS source_table,
        LOWER(COALESCE(i.user_type, 'missing')) AS user_type,
        CASE
          WHEN LOWER(COALESCE(i.user_type, '')) = 'registered' AND d.unique_id IS NOT NULL THEN 'core_teacher'
          WHEN LOWER(COALESCE(i.user_type, '')) = 'registered' AND d.unique_id IS NULL THEN 'shadow_registered'
          WHEN LOWER(COALESCE(i.user_type, '')) = 'anonymous' THEN 'shadow_anonymous'
          WHEN LOWER(COALESCE(i.user_type, '')) = 'seo' THEN 'shadow_seo'
          ELSE 'other'
        END AS population_bucket,
        i.unique_id AS source_unique_id,
        d.unique_id AS teacher_unique_id,
        i.data_inicio AS ts_start,
        NULL::TIMESTAMP AS ts_end,
        i.event_type,
        i.utm_source
      FROM raw_interactions i
      LEFT JOIN raw_dim_teachers d USING(unique_id)
    ),
    raw_session_enriched AS (
      SELECT
        'fct_session_raw' AS source_table,
        LOWER(COALESCE(user_type, 'missing')) AS user_type,
        population_bucket,
        source_unique_id,
        teacher_unique_id,
        session_start_ts AS ts_start,
        session_end_ts AS ts_end,
        NULL::VARCHAR AS event_type,
        NULL::VARCHAR AS utm_source
      FROM fct_session_raw
    ),
    unioned AS (
      SELECT * FROM raw_entries_enriched
      UNION ALL
      SELECT * FROM raw_interactions_enriched
      UNION ALL
      SELECT * FROM raw_session_enriched
    )
    SELECT
      source_table,
      user_type,
      population_bucket,
      COUNT(*) AS rows,
      COUNT(DISTINCT source_unique_id) AS distinct_ids,
      COUNT(DISTINCT teacher_unique_id) AS matched_teacher_ids,
      AVG(CASE WHEN teacher_unique_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS match_rate,
      AVG(
        CASE
          WHEN source_table = 'raw_entries' THEN CASE WHEN ts_start IS NULL OR ts_end IS NULL THEN 1.0 ELSE 0.0 END
          WHEN source_table = 'raw_interactions' THEN CASE WHEN ts_start IS NULL THEN 1.0 ELSE 0.0 END
          WHEN source_table = 'fct_session_raw' THEN CASE WHEN ts_start IS NULL OR ts_end IS NULL THEN 1.0 ELSE 0.0 END
          ELSE NULL
        END
      ) AS null_timestamp_rate,
      AVG(
        CASE
          WHEN source_table = 'raw_interactions' THEN CASE WHEN event_type IS NULL OR TRIM(event_type) = '' THEN 1.0 ELSE 0.0 END
          ELSE NULL
        END
      ) AS missing_event_type_rate,
      AVG(
        CASE
          WHEN source_table = 'raw_interactions' THEN CASE WHEN utm_source IS NULL OR TRIM(utm_source) = '' THEN 1.0 ELSE 0.0 END
          ELSE NULL
        END
      ) AS missing_utm_rate
    FROM unioned
    GROUP BY 1, 2, 3
    """
    detail = conn.execute(query).fetchdf()

    def weighted_metric(group: pd.DataFrame, col: str) -> float:
        valid = group[["rows", col]].copy()
        valid["rows"] = pd.to_numeric(valid["rows"], errors="coerce").fillna(0)
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
        valid = valid.dropna(subset=[col])
        if valid.empty or valid["rows"].sum() <= 0:
            return float("nan")
        return float(np.average(valid[col], weights=valid["rows"]))

    overall_rows: List[Dict[str, Any]] = []
    for (source_table, user_type), group in detail.groupby(["source_table", "user_type"], dropna=False, sort=False):
        overall_rows.append(
            {
                "source_table": source_table,
                "user_type": user_type,
                "population_bucket": "__all__",
                "rows": int(pd.to_numeric(group["rows"], errors="coerce").fillna(0).sum()),
                "distinct_ids": int(pd.to_numeric(group["distinct_ids"], errors="coerce").fillna(0).sum()),
                "matched_teacher_ids": int(pd.to_numeric(group["matched_teacher_ids"], errors="coerce").fillna(0).sum()),
                "match_rate": weighted_metric(group, "match_rate"),
                "null_timestamp_rate": weighted_metric(group, "null_timestamp_rate"),
                "missing_event_type_rate": weighted_metric(group, "missing_event_type_rate"),
                "missing_utm_rate": weighted_metric(group, "missing_utm_rate"),
            }
        )
    overall = pd.DataFrame(overall_rows, columns=detail.columns.tolist())
    out = pd.concat([detail, overall], ignore_index=True)
    return out.sort_values(["source_table", "user_type", "population_bucket"]).reset_index(drop=True)


def build_reconciliation_audit(conn: Any, mart: pd.DataFrame) -> pd.DataFrame:
    first_touch = conn.execute(
        """
        WITH first_touch AS (
          SELECT
            tm.teacher_unique_id,
            MIN(tm.month) AS first_month
          FROM fct_teacher_month tm
          WHERE COALESCE(tm.session_count_month, 0) > 0
             OR COALESCE(tm.interaction_rows_month, 0) > 0
          GROUP BY 1
        )
        SELECT
          ft.teacher_unique_id,
          ft.first_month,
          COALESCE(dt.utm_group, 'missing') AS utm_group
        FROM first_touch ft
        LEFT JOIN dim_teacher dt
          ON ft.teacher_unique_id = dt.teacher_unique_id
        """
    ).fetchdf()
    first_touch["first_month"] = pd.to_datetime(first_touch["first_month"], errors="coerce")

    all_rows = (
        first_touch.groupby(["first_month", "utm_group"], dropna=False)
        .agg(first_touch_rows=("teacher_unique_id", "nunique"))
        .reset_index()
    )
    all_total = (
        first_touch.groupby(["first_month"], dropna=False)
        .agg(first_touch_rows=("teacher_unique_id", "nunique"))
        .reset_index()
    )
    all_total["utm_group"] = "__all__"
    all_rows = pd.concat([all_rows, all_total], ignore_index=True)

    mart_work = mart.copy()
    mart_work["first_month"] = pd.to_datetime(mart_work["first_month"], errors="coerce")
    mart_work["utm_group"] = mart_work["utm_group"].fillna("missing")
    mart_rows = (
        mart_work.groupby(["first_month", "utm_group"], dropna=False)
        .agg(
            eligible_first_touch_rows=("teacher_unique_id", "nunique"),
            same_month_only_rows=("cohort_variant_same_month_only", "sum"),
            near_entry_0_1m_rows=("cohort_variant_near_entry_0_1m", "sum"),
            rows_missing_first_event=("first_event_missing_flag", "sum"),
            rows_missing_first_action=("first_event_action_missing_flag", "sum"),
            rows_missing_first_utm=("first_utm_missing_flag", "sum"),
            rows_missing_first_device=("first_device_missing_flag", "sum"),
            rows_session_without_interaction=("session_without_interaction_flag", "sum"),
        )
        .reset_index()
    )
    mart_total = (
        mart_work.groupby(["first_month"], dropna=False)
        .agg(
            eligible_first_touch_rows=("teacher_unique_id", "nunique"),
            same_month_only_rows=("cohort_variant_same_month_only", "sum"),
            near_entry_0_1m_rows=("cohort_variant_near_entry_0_1m", "sum"),
            rows_missing_first_event=("first_event_missing_flag", "sum"),
            rows_missing_first_action=("first_event_action_missing_flag", "sum"),
            rows_missing_first_utm=("first_utm_missing_flag", "sum"),
            rows_missing_first_device=("first_device_missing_flag", "sum"),
            rows_session_without_interaction=("session_without_interaction_flag", "sum"),
        )
        .reset_index()
    )
    mart_total["utm_group"] = "__all__"
    mart_rows = pd.concat([mart_rows, mart_total], ignore_index=True)

    out = all_rows.merge(mart_rows, on=["first_month", "utm_group"], how="left")
    fill_zero_cols = [
        "eligible_first_touch_rows",
        "same_month_only_rows",
        "near_entry_0_1m_rows",
        "rows_missing_first_event",
        "rows_missing_first_action",
        "rows_missing_first_utm",
        "rows_missing_first_device",
        "rows_session_without_interaction",
    ]
    for col in fill_zero_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["share_missing_first_event"] = out["rows_missing_first_event"] / out["eligible_first_touch_rows"].replace(0, np.nan)
    out["share_missing_first_action"] = out["rows_missing_first_action"] / out["eligible_first_touch_rows"].replace(0, np.nan)
    out["share_missing_first_utm"] = out["rows_missing_first_utm"] / out["eligible_first_touch_rows"].replace(0, np.nan)
    out["share_missing_first_device"] = out["rows_missing_first_device"] / out["eligible_first_touch_rows"].replace(0, np.nan)
    out["share_session_without_interaction"] = (
        out["rows_session_without_interaction"] / out["eligible_first_touch_rows"].replace(0, np.nan)
    )
    out = out.sort_values(["utm_group", "first_month"]).reset_index(drop=True)

    out["missing_first_event_6m_median"] = np.nan
    out["missing_first_event_jump_pp"] = np.nan
    out["missing_first_event_alert_flag"] = 0
    out["missing_first_action_6m_median"] = np.nan
    out["missing_first_action_jump_pp"] = np.nan
    out["missing_first_action_alert_flag"] = 0
    out["missing_first_utm_6m_median"] = np.nan
    out["missing_first_utm_jump_pp"] = np.nan
    out["missing_first_utm_alert_flag"] = 0
    out["missing_first_device_6m_median"] = np.nan
    out["missing_first_device_jump_pp"] = np.nan
    out["missing_first_device_alert_flag"] = 0

    total_mask = out["utm_group"] == "__all__"
    for share_col, median_col, jump_col, flag_col in [
        ("share_missing_first_event", "missing_first_event_6m_median", "missing_first_event_jump_pp", "missing_first_event_alert_flag"),
        ("share_missing_first_action", "missing_first_action_6m_median", "missing_first_action_jump_pp", "missing_first_action_alert_flag"),
        ("share_missing_first_utm", "missing_first_utm_6m_median", "missing_first_utm_jump_pp", "missing_first_utm_alert_flag"),
        ("share_missing_first_device", "missing_first_device_6m_median", "missing_first_device_jump_pp", "missing_first_device_alert_flag"),
    ]:
        series = out.loc[total_mask, share_col].astype(float)
        rolling = series.shift(1).rolling(6, min_periods=3).median()
        jump = series - rolling
        out.loc[total_mask, median_col] = rolling.values
        out.loc[total_mask, jump_col] = jump.values
        out.loc[total_mask & (jump > FIRST_MISSING_ALERT_JUMP), flag_col] = 1

    return out.sort_values(["first_month", "utm_group"]).reset_index(drop=True)


def build_tracking_gaps_audit(conn: Any, mart: pd.DataFrame) -> pd.DataFrame:
    gap_teachers = mart[mart["session_without_interaction_flag"] == 1].copy()
    if gap_teachers.empty:
        return pd.DataFrame(
            columns=[
                "slice_type",
                "slice_value",
                "gap_teachers",
                "all_teachers_in_slice",
                "gap_rate_within_slice",
                "share_gap_teachers_overall",
                "gap_sessions",
                "median_duration_min",
                "p90_duration_min",
                "avg_duration_min",
                "share_le_10s",
                "share_le_30s",
                "share_le_60s",
                "return_active_rate",
            ]
        )

    gap_teachers = gap_teachers[
        [
            "teacher_unique_id",
            "first_month",
            "utm_group",
            "device_observation_bucket",
            "teacher_population_status",
            "returned_active_m1",
        ]
    ].copy()
    conn.register("_gap_teachers_v2", gap_teachers)
    gap_sessions = conn.execute(
        """
        SELECT
          g.teacher_unique_id,
          g.first_month,
          g.utm_group,
          g.device_observation_bucket,
          g.teacher_population_status,
          g.returned_active_m1,
          s.duration_sec,
          s.duration_min
        FROM _gap_teachers_v2 g
        LEFT JOIN fct_session_clean s
          ON g.teacher_unique_id = s.teacher_unique_id
         AND g.first_month = s.session_month
        """
    ).fetchdf()

    def aggregate_slice(slice_type: str, teacher_col: str | None) -> pd.DataFrame:
        if teacher_col is None:
            teacher_group = gap_teachers.assign(slice_type=slice_type, slice_value="__all__")
            teacher_den = mart.assign(slice_type=slice_type, slice_value="__all__")
            session_group = gap_sessions.assign(slice_type=slice_type, slice_value="__all__")
        else:
            teacher_group = gap_teachers.assign(
                slice_type=slice_type,
                slice_value=gap_teachers[teacher_col].fillna("missing").astype(str),
            )
            teacher_den = mart.assign(
                slice_type=slice_type,
                slice_value=mart[teacher_col].fillna("missing").astype(str),
            )
            session_group = gap_sessions.assign(
                slice_type=slice_type,
                slice_value=gap_sessions[teacher_col].fillna("missing").astype(str),
            )

        teacher_stats = (
            teacher_group.groupby(["slice_type", "slice_value"], dropna=False)
            .agg(
                gap_teachers=("teacher_unique_id", "nunique"),
                return_active_rate=("returned_active_m1", "mean"),
            )
            .reset_index()
        )
        den_stats = (
            teacher_den.groupby(["slice_type", "slice_value"], dropna=False)
            .agg(all_teachers_in_slice=("teacher_unique_id", "nunique"))
            .reset_index()
        )
        session_stats = (
            session_group.groupby(["slice_type", "slice_value"], dropna=False)
            .agg(
                gap_sessions=("duration_min", "count"),
                median_duration_min=("duration_min", "median"),
                p90_duration_min=("duration_min", lambda s: s.quantile(0.9) if not s.dropna().empty else np.nan),
                avg_duration_min=("duration_min", "mean"),
                share_le_10s=("duration_sec", lambda s: (pd.to_numeric(s, errors="coerce").fillna(np.inf) <= 10).mean() if len(s) else np.nan),
                share_le_30s=("duration_sec", lambda s: (pd.to_numeric(s, errors="coerce").fillna(np.inf) <= 30).mean() if len(s) else np.nan),
                share_le_60s=("duration_sec", lambda s: (pd.to_numeric(s, errors="coerce").fillna(np.inf) <= 60).mean() if len(s) else np.nan),
            )
            .reset_index()
        )
        out = teacher_stats.merge(den_stats, on=["slice_type", "slice_value"], how="left")
        out = out.merge(session_stats, on=["slice_type", "slice_value"], how="left")
        out["gap_rate_within_slice"] = out["gap_teachers"] / out["all_teachers_in_slice"].replace(0, np.nan)
        out["share_gap_teachers_overall"] = out["gap_teachers"] / max(1, gap_teachers["teacher_unique_id"].nunique())
        return out

    frames = [
        aggregate_slice("overall", None),
        aggregate_slice("month", "first_month"),
        aggregate_slice("utm_group", "utm_group"),
        aggregate_slice("device_observation_bucket", "device_observation_bucket"),
        aggregate_slice("teacher_population_status", "teacher_population_status"),
    ]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["slice_type", "gap_teachers"], ascending=[True, False]).reset_index(drop=True)


def build_lineage_recheck(conn: Any, mart: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    sample_size = min(1000, len(mart))
    if sample_size == 0:
        return pd.DataFrame(columns=["field_name", "sampled_rows", "mismatch_rows", "mismatch_rate", "status", "note"])

    sample = mart[
        [
            "teacher_unique_id",
            "first_month",
            "strict_value_flag",
            "returned_active_m1",
            "returned_any_session_m1",
            "session_exposed_no_activity_no_download_flag",
            "session_exposed_activity_no_download_flag",
            "first_event_action",
        ]
    ].sample(sample_size, random_state=random_seed)
    conn.register("_onboarding_lineage_sample_v2", sample)

    query = f"""
    WITH sample AS (
      SELECT
        teacher_unique_id,
        first_month,
        first_month + INTERVAL 1 MONTH AS next_month
      FROM _onboarding_lineage_sample_v2
    ),
    first_month_interactions AS (
      SELECT
        s.teacher_unique_id,
        COALESCE(r.event_type, '<missing>') AS event_type,
        LOWER(COALESCE(r.event_type, '')) AS event_type_lower,
        {sql_event_action_expr('r.event_type')} AS event_action,
        r.data_inicio AS interaction_ts,
        ROW_NUMBER() OVER (
          PARTITION BY s.teacher_unique_id
          ORDER BY r.data_inicio, HASH(r.unique_id, r.data_inicio, r.event_type, r.content_type, r.id_aula, r.utm_source)
        ) AS rn
      FROM sample s
      LEFT JOIN raw_interactions r
        ON r.unique_id = s.teacher_unique_id
       AND LOWER(COALESCE(r.user_type, '')) = 'registered'
       AND DATE_TRUNC('month', r.data_inicio) = s.first_month
    ),
    first_month_interaction_agg AS (
      SELECT
        teacher_unique_id,
        MAX(CASE WHEN event_type IN ({", ".join(f"'{evt}'" for evt in STRICT_VALUE_EVENTS)}) THEN 1 ELSE 0 END) AS strict_value_flag_raw,
        MAX(CASE WHEN event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0 ELSE 1 END) AS active_user_flag_raw,
        MAX(CASE WHEN rn = 1 THEN event_action END) AS first_event_action_raw
      FROM first_month_interactions
      GROUP BY 1
    ),
    next_month_interactions AS (
      SELECT
        s.teacher_unique_id,
        COALESCE(r.event_type, '<missing>') AS event_type,
        LOWER(COALESCE(r.event_type, '')) AS event_type_lower
      FROM sample s
      LEFT JOIN raw_interactions r
        ON r.unique_id = s.teacher_unique_id
       AND LOWER(COALESCE(r.user_type, '')) = 'registered'
       AND DATE_TRUNC('month', r.data_inicio) = s.next_month
    ),
    next_month_interaction_agg AS (
      SELECT
        teacher_unique_id,
        MAX(CASE WHEN event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0 ELSE 1 END) AS returned_active_m1_raw
      FROM next_month_interactions
      GROUP BY 1
    ),
    first_month_session_agg AS (
      SELECT
        s.teacher_unique_id,
        COUNT(c.session_row_hash) AS session_count_first_month_raw
      FROM sample s
      LEFT JOIN fct_session_clean c
        ON s.teacher_unique_id = c.teacher_unique_id
       AND s.first_month = c.session_month
      GROUP BY 1
    ),
    next_month_session_agg AS (
      SELECT
        s.teacher_unique_id,
        CASE WHEN COUNT(c.session_row_hash) > 0 THEN 1 ELSE 0 END AS returned_any_session_m1_raw
      FROM sample s
      LEFT JOIN fct_session_clean c
        ON s.teacher_unique_id = c.teacher_unique_id
       AND s.next_month = c.session_month
      GROUP BY 1
    )
    SELECT
      sm.teacher_unique_id,
      sm.first_month,
      COALESCE(fia.strict_value_flag_raw, 0) AS strict_value_flag_raw,
      COALESCE(nia.returned_active_m1_raw, 0) AS returned_active_m1_raw,
      COALESCE(nsa.returned_any_session_m1_raw, 0) AS returned_any_session_m1_raw,
      CASE
        WHEN COALESCE(fsa.session_count_first_month_raw, 0) > 0
         AND COALESCE(fia.active_user_flag_raw, 0) <= 0
         AND COALESCE(fia.strict_value_flag_raw, 0) <= 0 THEN 1
        ELSE 0
      END AS session_exposed_no_activity_no_download_flag_raw,
      CASE
        WHEN COALESCE(fsa.session_count_first_month_raw, 0) > 0
         AND COALESCE(fia.active_user_flag_raw, 0) > 0
         AND COALESCE(fia.strict_value_flag_raw, 0) <= 0 THEN 1
        ELSE 0
      END AS session_exposed_activity_no_download_flag_raw,
      fia.first_event_action_raw
    FROM _onboarding_lineage_sample_v2 sm
    LEFT JOIN first_month_interaction_agg fia
      ON sm.teacher_unique_id = fia.teacher_unique_id
    LEFT JOIN next_month_interaction_agg nia
      ON sm.teacher_unique_id = nia.teacher_unique_id
    LEFT JOIN first_month_session_agg fsa
      ON sm.teacher_unique_id = fsa.teacher_unique_id
    LEFT JOIN next_month_session_agg nsa
      ON sm.teacher_unique_id = nsa.teacher_unique_id
    """
    recomputed = conn.execute(query).fetchdf()
    check = sample.merge(recomputed, on=["teacher_unique_id", "first_month"], how="left")

    def mismatch_rate(lhs: pd.Series, rhs: pd.Series) -> tuple[int, float]:
        lhs_norm = lhs.copy()
        rhs_norm = rhs.copy()
        if lhs_norm.dtype.kind in {"O", "U", "S"}:
            lhs_norm = lhs_norm.fillna("missing").astype(str).replace({"<missing>": "missing", "<na>": "missing"})
            rhs_norm = rhs_norm.fillna("missing").astype(str).replace({"<missing>": "missing", "<na>": "missing"})
        else:
            lhs_norm = pd.to_numeric(lhs_norm, errors="coerce").fillna(-999999)
            rhs_norm = pd.to_numeric(rhs_norm, errors="coerce").fillna(-999999)
        mismatches = int((lhs_norm != rhs_norm).sum())
        return mismatches, mismatches / max(1, len(lhs_norm))

    fields = [
        ("strict_value_flag", "strict_value_flag_raw"),
        ("returned_active_m1", "returned_active_m1_raw"),
        ("returned_any_session_m1", "returned_any_session_m1_raw"),
        ("session_exposed_no_activity_no_download_flag", "session_exposed_no_activity_no_download_flag_raw"),
        ("session_exposed_activity_no_download_flag", "session_exposed_activity_no_download_flag_raw"),
        ("first_event_action", "first_event_action_raw"),
    ]
    rows: List[Dict[str, Any]] = []
    for expected, observed in fields:
        mismatches, rate = mismatch_rate(check[expected], check[observed])
        rows.append(
            {
                "field_name": expected,
                "sampled_rows": len(check),
                "mismatch_rows": mismatches,
                "mismatch_rate": rate,
                "status": "pass" if mismatches == 0 else "fail",
                "note": f"Recomputed from raw/fct sample against mart column `{expected}`.",
            }
        )
    return pd.DataFrame(rows)


def compress_top(series: pd.Series, topn: int = 10) -> pd.Series:
    normalized = series.fillna("missing").astype(str)
    top_values = normalized.value_counts(dropna=False).head(topn).index
    return normalized.where(normalized.isin(top_values), "other")


def add_rank_band(df: pd.DataFrame, value_col: str, band_col: str, n_bins: int = 4) -> pd.DataFrame:
    out = df.copy()
    values = pd.to_numeric(out[value_col], errors="coerce").fillna(0)
    unique_values = int(values.nunique(dropna=True))
    bins = min(n_bins, max(2, unique_values))
    if unique_values < 2:
        out[band_col] = "Q1"
        return out
    try:
        out[band_col] = pd.qcut(
            values.rank(method="first"),
            q=bins,
            labels=[f"Q{i + 1}" for i in range(bins)],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        out[band_col] = "Q1"
    return out


def select_cohort_variant(mart: pd.DataFrame, cohort_variant: str) -> pd.DataFrame:
    if cohort_variant == "same_month_only":
        return mart[mart["cohort_variant_same_month_only"] == 1].copy()
    if cohort_variant == "near_entry_0_1m":
        return mart[mart["cohort_variant_near_entry_0_1m"] == 1].copy()
    if cohort_variant == "all_first_observed":
        return mart[mart["cohort_variant_all_first_observed"] == 1].copy()
    raise ValueError(f"cohort_variant desconhecido: {cohort_variant}")


def proportion_diff_ci(success_a: int, total_a: int, success_b: int, total_b: int) -> tuple[float, float, float]:
    if total_a <= 0 or total_b <= 0:
        return float("nan"), float("nan"), float("nan")
    rate_a = success_a / total_a
    rate_b = success_b / total_b
    effect = rate_a - rate_b
    se = np.sqrt(rate_a * (1 - rate_a) / total_a + rate_b * (1 - rate_b) / total_b)
    return float(effect), float(effect - 1.96 * se), float(effect + 1.96 * se)


def stratified_risk_difference(
    df: pd.DataFrame,
    exposed_col: str,
    outcome_col: str,
    strata_cols: Sequence[str],
) -> Dict[str, Any]:
    base = df[list(strata_cols) + [exposed_col, outcome_col]].copy()
    base[exposed_col] = pd.to_numeric(base[exposed_col], errors="coerce")
    base[outcome_col] = pd.to_numeric(base[outcome_col], errors="coerce")
    base = base.dropna(subset=[exposed_col, outcome_col])
    rows: List[Dict[str, Any]] = []
    for _, group in base.groupby(list(strata_cols), dropna=False, sort=False):
        n_exposed = int((group[exposed_col] == 1).sum())
        n_unexposed = int((group[exposed_col] == 0).sum())
        if n_exposed == 0 or n_unexposed == 0:
            continue
        p_exposed = float(group.loc[group[exposed_col] == 1, outcome_col].mean())
        p_unexposed = float(group.loc[group[exposed_col] == 0, outcome_col].mean())
        n_total = int(len(group))
        rows.append(
            {
                "n_total": n_total,
                "diff": p_exposed - p_unexposed,
                "var": (
                    p_exposed * (1 - p_exposed) / n_exposed
                    + p_unexposed * (1 - p_unexposed) / n_unexposed
                ),
            }
        )
    if not rows:
        return {
            "effect": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_obs": 0,
            "strata_used": 0,
        }
    frame = pd.DataFrame(rows)
    frame["weight"] = frame["n_total"] / frame["n_total"].sum()
    effect = float((frame["diff"] * frame["weight"]).sum())
    se = float(np.sqrt(((frame["weight"] ** 2) * frame["var"]).sum()))
    return {
        "effect": effect,
        "ci_low": effect - 1.96 * se,
        "ci_high": effect + 1.96 * se,
        "n_obs": int(frame["n_total"].sum()),
        "strata_used": int(len(frame)),
    }


def build_temporal_cliff_rows(
    mart: pd.DataFrame,
    benchmark: pd.DataFrame,
    cohort_variant: str,
    outcome_variant: str,
) -> List[Dict[str, Any]]:
    cohort = select_cohort_variant(mart, cohort_variant).copy()
    if cohort.empty or benchmark.empty:
        return []
    teacher_ids = set(cohort["teacher_unique_id"].astype(str).tolist())
    benchmark_work = benchmark[benchmark["teacher_unique_id"].astype(str).isin(teacher_ids)].copy()
    if benchmark_work.empty:
        return []
    cohort["month_str"] = pd.to_datetime(cohort["first_month"], errors="coerce").dt.strftime("%Y-%m")
    cohort["utm_group_top"] = compress_top(cohort["utm_group"], topn=10)
    benchmark_work["month_str"] = pd.to_datetime(benchmark_work["month"], errors="coerce").dt.strftime("%Y-%m")
    benchmark_work["utm_group_top"] = compress_top(benchmark_work["utm_group"], topn=10)
    combined = pd.concat(
        [
            cohort[["month_str", "utm_group_top", outcome_variant]].assign(experienced_month_flag=0),
            benchmark_work[["month_str", "utm_group_top", outcome_variant]].assign(experienced_month_flag=1),
        ],
        ignore_index=True,
    )
    combined[outcome_variant] = pd.to_numeric(combined[outcome_variant], errors="coerce")
    combined = combined.dropna(subset=[outcome_variant])
    if combined.empty:
        return []
    stats = stratified_risk_difference(
        combined,
        exposed_col="experienced_month_flag",
        outcome_col=outcome_variant,
        strata_cols=["month_str", "utm_group_top"],
    )
    cohort_rate = float(cohort[outcome_variant].mean())
    benchmark_rate = float(benchmark_work[outcome_variant].mean())
    row_status = "pass" if pd.notna(stats["ci_low"]) and stats["ci_low"] > 0 else "fail"
    return [
        {
            "hypothesis_id": "temporal_cliff_m1",
            "cohort_variant": cohort_variant,
            "outcome_variant": outcome_variant,
            "sensitivity_variant": "same_teacher_population",
            "effect_type": "later_active_vs_onboarding_risk_difference_same_population",
            "effect": stats["effect"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "n_obs": stats["n_obs"],
            "strata_used": stats["strata_used"],
            "status": row_status,
            "evidence_class": "",
            "note": f"onboarding_rate={cohort_rate:.4f}; later_active_rate={benchmark_rate:.4f}; teachers_in_overlap={len(teacher_ids)}",
        }
    ]


def build_experienced_benchmark(conn: Any) -> pd.DataFrame:
    query = """
    SELECT
      tm.teacher_unique_id,
      tm.month,
      COALESCE(dt.utm_group, 'missing') AS utm_group,
      tm.lifetime_active_months,
      tm.returned_active_m1,
      CASE WHEN COALESCE(nm.session_count_month, 0) > 0 THEN 1.0 ELSE 0.0 END AS returned_any_session_m1
    FROM fct_teacher_month tm
    INNER JOIN dim_teacher dt
      ON tm.teacher_unique_id = dt.teacher_unique_id
    LEFT JOIN fct_teacher_month nm
      ON tm.teacher_unique_id = nm.teacher_unique_id
     AND tm.next_month = nm.month
    WHERE COALESCE(tm.active_user_flag, 0) = 1
      AND COALESCE(tm.lifetime_active_months, 0) >= 2
      AND tm.next_month_observed_flag = 1
    ORDER BY tm.teacher_unique_id, tm.month
    """
    benchmark = conn.execute(query).fetchdf()
    if benchmark.empty:
        return benchmark
    benchmark["month"] = pd.to_datetime(benchmark["month"], errors="coerce")
    benchmark["lifetime_active_months"] = pd.to_numeric(benchmark["lifetime_active_months"], errors="coerce")
    benchmark["returned_active_m1"] = pd.to_numeric(benchmark["returned_active_m1"], errors="coerce")
    benchmark["returned_any_session_m1"] = pd.to_numeric(benchmark["returned_any_session_m1"], errors="coerce")
    benchmark["utm_group"] = benchmark["utm_group"].fillna("missing").astype(str)
    return benchmark


def build_intensity_rows(mart: pd.DataFrame, cohort_variant: str, outcome_variant: str) -> List[Dict[str, Any]]:
    subset = select_cohort_variant(mart, cohort_variant).copy()
    subset[outcome_variant] = pd.to_numeric(subset[outcome_variant], errors="coerce")
    subset = subset.dropna(subset=[outcome_variant])
    metrics = [
        "session_count_month",
        "total_session_minutes_month",
        "content_views_month",
        "first7d_events",
        "active_days_month",
    ]
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        work = subset[["teacher_unique_id", outcome_variant, metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce").fillna(0)
        if work.empty:
            continue
        work = add_rank_band(work, metric, "quartile_band", n_bins=4)
        rates = (
            work.groupby("quartile_band", dropna=False)[outcome_variant]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values("quartile_band")
        )
        band_order = rates["quartile_band"].tolist()
        mean_values = rates["mean"].tolist()
        monotonic = all(
            mean_values[i] <= mean_values[i + 1] + 1e-12
            for i in range(len(mean_values) - 1)
        )
        q1 = rates[rates["quartile_band"] == "Q1"].head(1)
        q4 = rates[rates["quartile_band"] == "Q4"].head(1)
        if q1.empty or q4.empty:
            effect = ci_low = ci_high = float("nan")
        else:
            effect, ci_low, ci_high = proportion_diff_ci(
                int(round(float(q4["mean"].iloc[0] * q4["count"].iloc[0]))),
                int(q4["count"].iloc[0]),
                int(round(float(q1["mean"].iloc[0] * q1["count"].iloc[0]))),
                int(q1["count"].iloc[0]),
            )
        rows.append(
            {
                "hypothesis_id": "early_intensity_monotonic",
                "cohort_variant": cohort_variant,
                "outcome_variant": outcome_variant,
                "sensitivity_variant": f"metric_{metric}",
                "effect_type": "q4_vs_q1_risk_difference",
                "effect": effect,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_obs": int(len(work)),
                "strata_used": int(len(rates)),
                "status": "pass" if monotonic and pd.notna(ci_low) and ci_low > 0 else "fail",
                "evidence_class": "",
                "note": " | ".join(
                    f"{band}={rate:.4f}"
                    for band, rate in zip(band_order, mean_values)
                ),
            }
        )
    return rows


def build_stratified_hypothesis_row(
    mart: pd.DataFrame,
    hypothesis_id: str,
    cohort_variant: str,
    outcome_variant: str,
    subset_mask: pd.Series,
    exposure_col: str,
    sensitivity_variant: str,
    extra_mask: pd.Series | None = None,
) -> Dict[str, Any]:
    subset = select_cohort_variant(mart, cohort_variant).copy()
    subset = subset[subset_mask.loc[subset.index]].copy()
    if extra_mask is not None:
        subset = subset[extra_mask.loc[subset.index]].copy()
    if subset.empty:
        return {
            "hypothesis_id": hypothesis_id,
            "cohort_variant": cohort_variant,
            "outcome_variant": outcome_variant,
            "sensitivity_variant": sensitivity_variant,
            "effect_type": "stratified_risk_difference",
            "effect": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_obs": 0,
            "strata_used": 0,
            "status": "review",
            "evidence_class": "",
            "note": "Sem observacoes apos filtros.",
        }
    subset["month_str"] = pd.to_datetime(subset["first_month"], errors="coerce").dt.strftime("%Y-%m")
    subset["utm_group_top"] = compress_top(subset["utm_group"], topn=10)
    subset = add_rank_band(subset, "session_count_month", "session_intensity_band", n_bins=4)
    subset[exposure_col] = pd.to_numeric(subset[exposure_col], errors="coerce")
    subset[outcome_variant] = pd.to_numeric(subset[outcome_variant], errors="coerce")
    subset = subset.dropna(subset=[exposure_col, outcome_variant])
    stats = stratified_risk_difference(
        subset,
        exposed_col=exposure_col,
        outcome_col=outcome_variant,
        strata_cols=["month_str", "utm_group_top", "session_intensity_band"],
    )
    expected_positive = {
        "activity_no_download_vs_no_activity_no_download",
        "mobile_association",
        "first_view_vs_first_download",
        "strict_value_vs_activity_no_download",
    }
    row_status = "pass" if pd.notna(stats["ci_low"]) and stats["ci_low"] > 0 and hypothesis_id in expected_positive else "fail"
    return {
        "hypothesis_id": hypothesis_id,
        "cohort_variant": cohort_variant,
        "outcome_variant": outcome_variant,
        "sensitivity_variant": sensitivity_variant,
        "effect_type": "stratified_risk_difference",
        "effect": stats["effect"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "n_obs": stats["n_obs"],
        "strata_used": stats["strata_used"],
        "status": row_status if stats["n_obs"] > 0 else "review",
        "evidence_class": "",
        "note": "Strata: month_str + utm_group_top + session_intensity_band.",
    }


def build_hypothesis_table(mart: pd.DataFrame, tracking_context: Dict[str, Any], conn: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    mart_work = mart.copy()
    benchmark = build_experienced_benchmark(conn)
    mart_work["returned_active_m1"] = pd.to_numeric(mart_work["returned_active_m1"], errors="coerce")
    mart_work["returned_any_session_m1"] = pd.to_numeric(mart_work["returned_any_session_m1"], errors="coerce")
    mart_work["first_event_action"] = mart_work["first_event_action"].fillna("missing").astype(str)
    mart_work["first_event_action_view_flag"] = (mart_work["first_event_action"] == "view").astype(int)
    mart_work["first_event_action_download_flag"] = (mart_work["first_event_action"] == "download").astype(int)

    low_device_months_by_variant: Dict[str, List[str]] = {}
    for cohort_variant in COHORT_VARIANTS:
        cohort = select_cohort_variant(mart_work, cohort_variant).copy()
        if cohort.empty:
            low_device_months_by_variant[cohort_variant] = []
            continue
        cohort["month_str"] = pd.to_datetime(cohort["first_month"], errors="coerce").dt.strftime("%Y-%m")
        device_completeness = (
            cohort.groupby("month_str", dropna=False)
            .agg(
                device_observed_rate=(
                    "device_observation_bucket",
                    lambda s: (s.fillna("missing").astype(str) != "no_observed_device").mean(),
                )
            )
            .reset_index()
        )
        low_device_months_by_variant[cohort_variant] = (
            device_completeness[device_completeness["device_observed_rate"] < DEVICE_COMPLETENESS_MIN]["month_str"]
            .astype(str)
            .tolist()
        )

    for cohort_variant in COHORT_VARIANTS:
        for outcome_variant in OUTCOME_VARIANTS:
            rows.extend(build_temporal_cliff_rows(mart_work, benchmark, cohort_variant, outcome_variant))
            rows.extend(build_intensity_rows(mart_work, cohort_variant, outcome_variant))

            cohort_subset = select_cohort_variant(mart_work, cohort_variant)
            if cohort_subset.empty:
                continue

            mask_activity_pair = (
                (mart_work["session_exposed_activity_no_download_flag"] == 1)
                | (mart_work["session_exposed_no_activity_no_download_flag"] == 1)
            )
            mart_work["activity_no_download_exposure"] = (
                mart_work["session_exposed_activity_no_download_flag"] == 1
            ).astype(int)
            rows.append(
                build_stratified_hypothesis_row(
                    mart_work,
                    "activity_no_download_vs_no_activity_no_download",
                    cohort_variant,
                    outcome_variant,
                    mask_activity_pair,
                    "activity_no_download_exposure",
                    "primary",
                )
            )

            mask_strict_pair = (
                (mart_work["strict_value_flag"] == 1)
                | (mart_work["session_exposed_activity_no_download_flag"] == 1)
            )
            mart_work["strict_value_exposure"] = (mart_work["strict_value_flag"] == 1).astype(int)
            rows.append(
                build_stratified_hypothesis_row(
                    mart_work,
                    "strict_value_vs_activity_no_download",
                    cohort_variant,
                    outcome_variant,
                    mask_strict_pair,
                    "strict_value_exposure",
                    "primary",
                )
            )

            mask_view_download = mart_work["first_event_action"].isin(["view", "download"])
            rows.append(
                build_stratified_hypothesis_row(
                    mart_work,
                    "first_view_vs_first_download",
                    cohort_variant,
                    outcome_variant,
                    mask_view_download,
                    "first_event_action_view_flag",
                    "primary",
                )
            )

            rows.append(
                build_stratified_hypothesis_row(
                    mart_work,
                    "mobile_association",
                    cohort_variant,
                    outcome_variant,
                    pd.Series(True, index=mart_work.index),
                    "used_mobile_flag",
                    "all_months",
                )
            )
            allowed_months = set(
                pd.to_datetime(select_cohort_variant(mart_work, cohort_variant)["first_month"], errors="coerce")
                .dt.strftime("%Y-%m")
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ) - set(low_device_months_by_variant[cohort_variant])
            extra_mask = pd.to_datetime(mart_work["first_month"], errors="coerce").dt.strftime("%Y-%m").isin(allowed_months)
            rows.append(
                build_stratified_hypothesis_row(
                    mart_work,
                    "mobile_association",
                    cohort_variant,
                    outcome_variant,
                    pd.Series(True, index=mart_work.index),
                    "used_mobile_flag",
                    "device_complete_months",
                    extra_mask=extra_mask,
                )
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    session_gap_material = bool(
        tracking_context.get("session_without_interaction_max_monthly_share", 0.0) > GAP_SHARE_ALERT_THRESHOLD
        or tracking_context.get("session_without_interaction_top1_utm_concentration", 0.0) > GAP_TOP1_UTM_CONCENTRATION_THRESHOLD
    )

    evidence_class: Dict[str, str] = {}
    for hypothesis_id in sorted(table["hypothesis_id"].dropna().unique().tolist()):
        current = table[table["hypothesis_id"] == hypothesis_id].copy()
        required = current[
            current["cohort_variant"].isin(COHORT_VARIANTS)
            & current["outcome_variant"].isin(OUTCOME_VARIANTS)
        ].copy()

        if hypothesis_id == "temporal_cliff_m1":
            evidence_class[hypothesis_id] = (
                "supported_robust" if (required["status"] == "pass").all() and not required.empty else "not_supported"
            )
        elif hypothesis_id == "early_intensity_monotonic":
            evidence_class[hypothesis_id] = (
                "supported_robust" if (required["status"] == "pass").all() and not required.empty else "not_supported"
            )
        elif hypothesis_id == "activity_no_download_vs_no_activity_no_download":
            if session_gap_material:
                evidence_class[hypothesis_id] = "tracking_risk"
            else:
                evidence_class[hypothesis_id] = (
                    "supported_robust" if (required["status"] == "pass").all() and not required.empty else "not_supported"
                )
        elif hypothesis_id == "strict_value_vs_activity_no_download":
            evidence_class[hypothesis_id] = (
                "supported_correlational" if (required["status"] == "pass").all() and not required.empty else "not_supported"
            )
        elif hypothesis_id == "first_view_vs_first_download":
            evidence_class[hypothesis_id] = (
                "supported_correlational" if (required["status"] == "pass").all() and not required.empty else "not_supported"
            )
        elif hypothesis_id == "mobile_association":
            primary = required[required["sensitivity_variant"] == "all_months"]
            device_complete = required[required["sensitivity_variant"] == "device_complete_months"]
            if not primary.empty and (primary["status"] == "pass").all() and not device_complete.empty and (device_complete["status"] == "pass").all():
                evidence_class[hypothesis_id] = "supported_correlational"
            elif not primary.empty and (primary["status"] == "pass").all():
                evidence_class[hypothesis_id] = "tracking_risk"
            else:
                evidence_class[hypothesis_id] = "not_supported"
        else:
            evidence_class[hypothesis_id] = "tracking_risk"

    table["evidence_class"] = table["hypothesis_id"].map(evidence_class)
    table["status"] = np.where(
        table["evidence_class"].isin(["supported_robust", "supported_correlational"]),
        np.where(table["status"] == "pass", "pass", "review"),
        np.where(table["evidence_class"] == "tracking_risk", "review", "fail"),
    )
    table["note"] = np.where(
        table["hypothesis_id"] == "mobile_association",
        table["note"] + " low_device_months=" + table["cohort_variant"].map(low_device_months_by_variant).apply(lambda values: ",".join(values) if values else "none"),
        table["note"],
    )
    table["note"] = np.where(
        (table["hypothesis_id"] == "activity_no_download_vs_no_activity_no_download") & (table["evidence_class"] == "tracking_risk"),
        table["note"] + " downgraded_by=session_without_interaction_tracking",
        table["note"],
    )
    tracking_context["low_device_months_by_variant"] = low_device_months_by_variant
    tracking_context["hypothesis_evidence_class"] = evidence_class
    return table.sort_values(["hypothesis_id", "cohort_variant", "outcome_variant", "sensitivity_variant"]).reset_index(drop=True)


def build_thresholds_table(mart: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "session_count_month",
        "total_session_minutes_month",
        "content_views_month",
        "first7d_events",
        "active_days_month",
    ]
    rows: List[Dict[str, Any]] = []
    for cohort_variant in ("all_first_observed",) + tuple(COHORT_VARIANTS):
        subset = select_cohort_variant(mart, cohort_variant)
        if subset.empty:
            continue
        for outcome_variant in OUTCOME_VARIANTS:
            work = subset.copy()
            work[outcome_variant] = pd.to_numeric(work[outcome_variant], errors="coerce")
            work = work.dropna(subset=[outcome_variant])
            if work.empty:
                continue
            for metric in metrics:
                series = pd.to_numeric(work[metric], errors="coerce").fillna(0)
                work_metric = work[["teacher_unique_id", outcome_variant]].copy()
                work_metric[metric] = series
                work_metric = add_rank_band(work_metric, metric, "band_label", n_bins=4)
                grouped = (
                    work_metric.groupby("band_label", dropna=False)
                    .agg(
                        n_obs=("teacher_unique_id", "nunique"),
                        outcome_rate=(outcome_variant, "mean"),
                        band_min=(metric, "min"),
                        band_max=(metric, "max"),
                    )
                    .reset_index()
                    .sort_values("band_label")
                )
                monotonic = grouped["outcome_rate"].is_monotonic_increasing
                q25 = float(series.quantile(0.25))
                q50 = float(series.quantile(0.50))
                q75 = float(series.quantile(0.75))
                q90 = float(series.quantile(0.90))
                for _, row in grouped.iterrows():
                    rows.append(
                        {
                            "metric_name": metric,
                            "cohort_variant": cohort_variant,
                            "outcome_variant": outcome_variant,
                            "band_label": row["band_label"],
                            "n_obs": int(row["n_obs"]),
                            "outcome_rate": float(row["outcome_rate"]),
                            "band_min": float(row["band_min"]),
                            "band_max": float(row["band_max"]),
                            "q25": q25,
                            "q50": q50,
                            "q75": q75,
                            "q90": q90,
                            "recommended_threshold_value": PRODUCT_THRESHOLDS.get(metric),
                            "recommended_threshold_kind": "frozen_default" if PRODUCT_THRESHOLDS.get(metric) is not None else "analytical_only",
                            "monotonic_non_decreasing_flag": int(monotonic),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["metric_name", "cohort_variant", "outcome_variant", "band_label"]
    ).reset_index(drop=True)


def run_secondary_model(mart: pd.DataFrame) -> Dict[str, Any]:
    subset = select_cohort_variant(mart, "near_entry_0_1m").copy()
    subset = subset.dropna(subset=["returned_active_m1"]).copy()
    if subset.empty:
        return {"status": "unavailable", "note": "Sem linhas na coorte near_entry_0_1m."}

    subset["month"] = pd.to_datetime(subset["first_month"], errors="coerce")
    months = sorted(subset["month"].dropna().unique().tolist())
    if len(months) < 4:
        return {"status": "weak", "note": "Poucos meses para split temporal consistente."}

    split_idx = max(1, int(len(months) * 0.70))
    train_months = set(months[:split_idx])
    test_months = set(months[split_idx:])
    train = subset[subset["month"].isin(train_months)].copy()
    test = subset[subset["month"].isin(test_months)].copy()
    if train.empty or test.empty or train["returned_active_m1"].nunique() < 2 or test["returned_active_m1"].nunique() < 2:
        return {"status": "weak", "note": "Split temporal sem variacao suficiente."}

    train["utm_group_top"] = compress_top(train["utm_group"], topn=10)
    allowed_utm = set(train["utm_group_top"].unique().tolist())
    test["utm_group_top"] = (
        test["utm_group"].fillna("missing").astype(str).where(test["utm_group"].fillna("missing").astype(str).isin(allowed_utm), "other")
    )
    train["first_event_action_model"] = train["first_event_action"].fillna("missing").astype(str)
    allowed_actions = set(train["first_event_action_model"].value_counts().head(8).index.tolist())
    train["first_event_action_model"] = train["first_event_action_model"].where(train["first_event_action_model"].isin(allowed_actions), "other")
    test["first_event_action_model"] = test["first_event_action"].fillna("missing").astype(str)
    test["first_event_action_model"] = test["first_event_action_model"].where(test["first_event_action_model"].isin(allowed_actions), "other")

    numeric_features = [
        "session_count_month",
        "total_session_minutes_month",
        "active_days_month",
        "activity_events_month",
        "content_views_month",
        "other_activity_non_download_events_month",
        "strict_download_count_month",
        "used_mobile_flag",
        "used_desktop_flag",
        "session_without_interaction_flag",
        "first7d_events",
        "first7d_active_days",
        "first_session_minutes",
        "first7d_sessions",
        "first7d_session_minutes",
    ]
    categorical_features = ["utm_group_top", "first_event_action_model"]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value=0)),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        [
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=500, C=0.5)),
        ]
    )
    model.fit(train[numeric_features + categorical_features], train["returned_active_m1"].astype(int))
    scores = model.predict_proba(test[numeric_features + categorical_features])[:, 1]
    auc = float(roc_auc_score(test["returned_active_m1"].astype(int), scores))
    return {
        "status": "ok" if auc > 0.65 else "weak",
        "auc": auc,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_month_start": str(min(train_months)),
        "train_month_end": str(max(train_months)),
        "test_month_start": str(min(test_months)),
        "test_month_end": str(max(test_months)),
    }


def build_tracking_context(
    coverage: pd.DataFrame,
    reconciliation: pd.DataFrame,
    tracking_gaps: pd.DataFrame,
) -> Dict[str, Any]:
    entries_registered = coverage[
        (coverage["source_table"] == "raw_entries")
        & (coverage["user_type"] == "registered")
        & (coverage["population_bucket"] == "__all__")
    ]
    interactions_registered = coverage[
        (coverage["source_table"] == "raw_interactions")
        & (coverage["user_type"] == "registered")
        & (coverage["population_bucket"] == "__all__")
    ]
    entries_registered_nulls = coverage[
        (coverage["source_table"] == "raw_entries")
        & (coverage["user_type"] == "registered")
        & (coverage["population_bucket"] == "__all__")
    ]
    interactions_registered_nulls = coverage[
        (coverage["source_table"] == "raw_interactions")
        & (coverage["user_type"] == "registered")
        & (coverage["population_bucket"] == "__all__")
    ]
    overall_months = reconciliation[reconciliation["utm_group"] == "__all__"].copy()
    month_gap = tracking_gaps[tracking_gaps["slice_type"] == "month"].copy()
    utm_gap = tracking_gaps[tracking_gaps["slice_type"] == "utm_group"].copy()
    return {
        "registered_entries_match_rate": float(entries_registered["match_rate"].iloc[0]) if not entries_registered.empty else np.nan,
        "registered_interactions_match_rate": float(interactions_registered["match_rate"].iloc[0]) if not interactions_registered.empty else np.nan,
        "raw_entries_null_timestamp_rate": float(entries_registered_nulls["null_timestamp_rate"].iloc[0]) if not entries_registered_nulls.empty else np.nan,
        "raw_interactions_null_timestamp_rate": float(interactions_registered_nulls["null_timestamp_rate"].iloc[0]) if not interactions_registered_nulls.empty else np.nan,
        "missing_first_event_alert_months": overall_months[overall_months["missing_first_event_alert_flag"] == 1]["first_month"]
        .dt.strftime("%Y-%m")
        .dropna()
        .tolist(),
        "missing_first_action_alert_months": overall_months[overall_months["missing_first_action_alert_flag"] == 1]["first_month"]
        .dt.strftime("%Y-%m")
        .dropna()
        .tolist(),
        "missing_first_utm_alert_months": overall_months[overall_months["missing_first_utm_alert_flag"] == 1]["first_month"]
        .dt.strftime("%Y-%m")
        .dropna()
        .tolist(),
        "missing_first_device_alert_months": overall_months[overall_months["missing_first_device_alert_flag"] == 1]["first_month"]
        .dt.strftime("%Y-%m")
        .dropna()
        .tolist(),
        "session_without_interaction_max_monthly_share": float(month_gap["gap_rate_within_slice"].max()) if not month_gap.empty else 0.0,
        "session_without_interaction_top1_utm_concentration": float(utm_gap["share_gap_teachers_overall"].max()) if not utm_gap.empty else 0.0,
    }


def build_summary_payload(
    mart: pd.DataFrame,
    coverage: pd.DataFrame,
    reconciliation: pd.DataFrame,
    tracking_gaps: pd.DataFrame,
    lineage_recheck: pd.DataFrame,
    hypothesis_table: pd.DataFrame,
    thresholds: pd.DataFrame,
    secondary_model: Dict[str, Any],
    tracking_context: Dict[str, Any],
) -> Dict[str, Any]:
    required_classes = tracking_context.get("hypothesis_evidence_class", {})
    coverage_ok = (
        tracking_context.get("registered_entries_match_rate", 0.0) >= 0.95
        and tracking_context.get("registered_interactions_match_rate", 0.0) >= 0.95
        and tracking_context.get("raw_entries_null_timestamp_rate", 1.0) == 0.0
        and tracking_context.get("raw_interactions_null_timestamp_rate", 1.0) == 0.0
    )
    default_thresholds = {
        metric: value
        for metric, value in PRODUCT_THRESHOLDS.items()
        if value is not None
    }
    return {
        "generated_at_utc": utc_now_iso(),
        "coverage_ok": coverage_ok,
        "mart_rows": int(len(mart)),
        "mart_same_month_rows": int(pd.to_numeric(mart["cohort_variant_same_month_only"], errors="coerce").fillna(0).sum()),
        "mart_near_entry_rows": int(pd.to_numeric(mart["cohort_variant_near_entry_0_1m"], errors="coerce").fillna(0).sum()),
        "registered_entries_match_rate": tracking_context.get("registered_entries_match_rate"),
        "registered_interactions_match_rate": tracking_context.get("registered_interactions_match_rate"),
        "raw_entries_null_timestamp_rate": tracking_context.get("raw_entries_null_timestamp_rate"),
        "raw_interactions_null_timestamp_rate": tracking_context.get("raw_interactions_null_timestamp_rate"),
        "missing_first_event_alert_months": tracking_context.get("missing_first_event_alert_months", []),
        "missing_first_action_alert_months": tracking_context.get("missing_first_action_alert_months", []),
        "missing_first_utm_alert_months": tracking_context.get("missing_first_utm_alert_months", []),
        "missing_first_device_alert_months": tracking_context.get("missing_first_device_alert_months", []),
        "session_without_interaction_max_monthly_share": tracking_context.get("session_without_interaction_max_monthly_share"),
        "session_without_interaction_top1_utm_concentration": tracking_context.get("session_without_interaction_top1_utm_concentration"),
        "low_device_months_by_variant": tracking_context.get("low_device_months_by_variant", {}),
        "hypothesis_evidence_class": required_classes,
        "lineage_recheck_failures": lineage_recheck[lineage_recheck["status"] != "pass"]["field_name"].tolist() if not lineage_recheck.empty else [],
        "secondary_model": secondary_model,
        "frozen_product_thresholds": default_thresholds,
        "artifacts": [
            "mart_teacher_onboarding_first_month_v2",
            "audit_raw_population_coverage_v2",
            "audit_onboarding_population_reconciliation_v2",
            "audit_onboarding_tracking_gaps_v2",
            "audit_onboarding_lineage_recheck_v2",
            "analytics_onboarding_hypothesis_validation_v2",
            "analytics_onboarding_thresholds_v2",
        ],
    }


def write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "nan"
        return f"{float(value):.{digits}f}"

    lines = [
        "# Onboarding validado v2",
        "",
        f"- Gerado em UTC: {summary['generated_at_utc']}",
        f"- coverage_ok: {summary['coverage_ok']}",
        f"- Linhas na mart: {summary['mart_rows']}",
        f"- Coorte same_month_only: {summary['mart_same_month_rows']}",
        f"- Coorte near_entry_0_1m: {summary['mart_near_entry_rows']}",
        f"- Match rate raw_entries registered: {fmt(summary['registered_entries_match_rate'], 4)}",
        f"- Match rate raw_interactions registered: {fmt(summary['registered_interactions_match_rate'], 4)}",
        f"- Null timestamp raw_entries: {fmt(summary['raw_entries_null_timestamp_rate'], 6)}",
        f"- Null timestamp raw_interactions: {fmt(summary['raw_interactions_null_timestamp_rate'], 6)}",
        f"- Missing first_event alerts: {', '.join(summary['missing_first_event_alert_months']) if summary['missing_first_event_alert_months'] else 'none'}",
        f"- Missing first_action alerts: {', '.join(summary['missing_first_action_alert_months']) if summary['missing_first_action_alert_months'] else 'none'}",
        f"- Missing first_utm alerts: {', '.join(summary['missing_first_utm_alert_months']) if summary['missing_first_utm_alert_months'] else 'none'}",
        f"- Missing first_device alerts: {', '.join(summary['missing_first_device_alert_months']) if summary['missing_first_device_alert_months'] else 'none'}",
        f"- Session without interaction max monthly share: {fmt(summary['session_without_interaction_max_monthly_share'], 4)}",
        f"- Session without interaction top1 utm concentration: {fmt(summary['session_without_interaction_top1_utm_concentration'], 4)}",
        f"- Secondary model status: {summary['secondary_model'].get('status')}",
        f"- Secondary model AUC: {summary['secondary_model'].get('auc', float('nan'))}",
        "",
        "## Hypothesis evidence class",
    ]
    for hypothesis_id, evidence_class in sorted(summary["hypothesis_evidence_class"].items()):
        lines.append(f"- `{hypothesis_id}`: `{evidence_class}`")
    lines.extend(
        [
            "",
            "## Frozen product thresholds",
        ]
    )
    for metric, value in sorted(summary["frozen_product_thresholds"].items()):
        lines.append(f"- `{metric}`: {value}")
    write_markdown(path, lines)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    conn = connect_duckdb(cfg)
    try:
        register_raw_views(conn, cfg.data_dir)
        require_tables(
            conn,
            [
                "raw_entries",
                "raw_interactions",
                "raw_dim_teachers",
                "fct_session_raw",
                "fct_session_clean",
                "fct_interaction_clean",
                "fct_teacher_month",
                "dim_teacher",
            ],
        )

        mart = build_onboarding_mart(conn)
        coverage = build_raw_population_coverage(conn)
        reconciliation = build_reconciliation_audit(conn, mart)
        tracking_gaps = build_tracking_gaps_audit(conn, mart)
        lineage_recheck = build_lineage_recheck(conn, mart, cfg.random_seed)
        tracking_context = build_tracking_context(coverage, reconciliation, tracking_gaps)
        hypothesis_table = build_hypothesis_table(mart, tracking_context, conn)
        thresholds = build_thresholds_table(mart)
        secondary_model = run_secondary_model(mart)

        outputs: Dict[str, pd.DataFrame] = {
            "mart_teacher_onboarding_first_month_v2": mart,
            "audit_raw_population_coverage_v2": coverage,
            "audit_onboarding_population_reconciliation_v2": reconciliation,
            "audit_onboarding_tracking_gaps_v2": tracking_gaps,
            "audit_onboarding_lineage_recheck_v2": lineage_recheck,
            "analytics_onboarding_hypothesis_validation_v2": hypothesis_table,
            "analytics_onboarding_thresholds_v2": thresholds,
        }
        for name, df in outputs.items():
            persist_output(conn, cfg, name, df)

        summary = build_summary_payload(
            mart=mart,
            coverage=coverage,
            reconciliation=reconciliation,
            tracking_gaps=tracking_gaps,
            lineage_recheck=lineage_recheck,
            hypothesis_table=hypothesis_table,
            thresholds=thresholds,
            secondary_model=secondary_model,
            tracking_context=tracking_context,
        )
        write_json(cfg.output_dir / "json" / "onboarding_validation_summary_v2.json", summary)
        write_summary_markdown(cfg.output_dir / "audit" / "onboarding_validation_summary_v2.md", summary)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
