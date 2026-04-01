"""Construção das bases analíticas usadas no pipeline modelled -> ml."""

from __future__ import annotations

import json
from typing import Any, List, Sequence

import duckdb
import pandas as pd

from . import analysis_setup as setup

def build_onboarding_mart(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
    WITH first_touch AS (
      SELECT
        teacher_unique_id,
        MIN(month) AS first_month
      FROM fct_teacher_month
      WHERE COALESCE(clean_entry_session_count_month, 0) > 0
         OR COALESCE(interaction_rows_month, 0) > 0
      GROUP BY 1
    ),
    first_session_anchor AS (
      SELECT
        s.teacher_unique_id,
        MIN(s.session_start_ts) AS first_session_ts
      FROM fct_session_clean s
      INNER JOIN first_touch ft
        ON s.teacher_unique_id = ft.teacher_unique_id
       AND s.session_month = CAST(ft.first_month AS DATE)
      GROUP BY 1
    ),
    first_interaction_anchor AS (
      SELECT
        i.teacher_unique_id,
        MIN(i.interaction_ts) AS first_interaction_ts
      FROM fct_interaction_clean i
      INNER JOIN first_touch ft
        ON i.teacher_unique_id = ft.teacher_unique_id
       AND i.interaction_month = CAST(ft.first_month AS DATE)
      GROUP BY 1
    ),
    onboarding_anchor AS (
      SELECT
        ft.teacher_unique_id,
        ft.first_month,
        fsa.first_session_ts,
        fia.first_interaction_ts,
        COALESCE(fsa.first_session_ts, fia.first_interaction_ts) AS onboarding_anchor_ts
      FROM first_touch ft
      LEFT JOIN first_session_anchor fsa ON ft.teacher_unique_id = fsa.teacher_unique_id
      LEFT JOIN first_interaction_anchor fia ON ft.teacher_unique_id = fia.teacher_unique_id
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
        ) AS rn
      FROM fct_interaction_clean i
      INNER JOIN first_touch ft
        ON i.teacher_unique_id = ft.teacher_unique_id
       AND i.interaction_month = CAST(ft.first_month AS DATE)
    ),
    first_event_summary AS (
      SELECT
        fer.teacher_unique_id,
        MAX(CASE WHEN fer.rn = 1 THEN fer.event_type END) AS first_event_type,
        MAX(CASE WHEN fer.rn = 1 THEN fer.event_family END) AS first_event_family,
        MAX(CASE WHEN fer.rn = 1 THEN fer.event_action END) AS first_event_action,
        MAX(CASE WHEN fer.rn = 1 THEN fer.device_group END) AS first_device,
        MAX(CASE WHEN fer.rn = 1 THEN fer.utm_source END) AS first_utm_source,
        SUM(CASE WHEN fer.rn <= 3 THEN COALESCE(fer.is_download_event, 0) ELSE 0 END) AS first3_interaction_downloads,
        SUM(CASE WHEN fer.rn <= 3 THEN COALESCE(fer.is_visualization_event, 0) ELSE 0 END) AS first3_interaction_views,
        SUM(CASE WHEN fer.rn <= 3 THEN COALESCE(fer.is_other_activity_non_download_event, 0) ELSE 0 END) AS first3_interaction_other_actions,
        COUNT(*) FILTER (WHERE fer.interaction_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY) AS first7d_events,
        COUNT(DISTINCT CAST(fer.interaction_ts AS DATE)) FILTER (
          WHERE fer.interaction_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY
        ) AS first7d_active_days
      FROM first_event_ranked fer
      INNER JOIN onboarding_anchor oa
        ON fer.teacher_unique_id = oa.teacher_unique_id
      GROUP BY 1
    ),
    first_session_ranked AS (
      SELECT
        s.teacher_unique_id,
        s.session_start_ts,
        s.duration_min,
        ROW_NUMBER() OVER (
          PARTITION BY s.teacher_unique_id
          ORDER BY s.session_start_ts, s.session_row_hash
        ) AS rn
      FROM fct_session_clean s
      INNER JOIN first_touch ft
        ON s.teacher_unique_id = ft.teacher_unique_id
       AND s.session_month = CAST(ft.first_month AS DATE)
    ),
    first_session_summary AS (
      SELECT
        fsr.teacher_unique_id,
        MAX(CASE WHEN fsr.rn = 1 THEN fsr.duration_min END) AS first_session_minutes,
        COUNT(*) FILTER (WHERE fsr.session_start_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY) AS first7d_sessions,
        SUM(fsr.duration_min) FILTER (WHERE fsr.session_start_ts < oa.onboarding_anchor_ts + INTERVAL 7 DAY) AS first7d_session_minutes
      FROM first_session_ranked fsr
      INNER JOIN onboarding_anchor oa
        ON fsr.teacher_unique_id = oa.teacher_unique_id
      GROUP BY 1
    )
    SELECT
      oa.teacher_unique_id,
      oa.first_month,
      oa.first_session_ts,
      oa.first_interaction_ts,
      oa.onboarding_anchor_ts,
      DATE_TRUNC('month', dt.teacher_data_entrada) AS data_entrada_month,
      CASE
        WHEN dt.teacher_data_entrada IS NULL THEN NULL
        ELSE DATE_DIFF('month', DATE_TRUNC('month', dt.teacher_data_entrada), oa.first_month)
      END AS months_after_entry,
      COALESCE(dt.teacher_population_status, 'missing') AS teacher_population_status,
      COALESCE(dt.teacher_utm_group, 'missing') AS utm_group,
      COALESCE(dt.teacher_estado, 'missing') AS teacher_estado,
      COALESCE(dt.teacher_currentstage, 'missing') AS teacher_currentstage,
      COALESCE(dt.teacher_currentsubject_group, 'missing') AS teacher_currentsubject_group,
      fe.first_event_type,
      fe.first_event_family,
      fe.first_event_action,
      fe.first_device,
      fe.first_utm_source,
      fe.first3_interaction_downloads,
      fe.first3_interaction_views,
      fe.first3_interaction_other_actions,
      fe.first7d_events,
      fe.first7d_active_days,
      fs.first_session_minutes,
      fs.first7d_sessions,
      fs.first7d_session_minutes,
      CASE WHEN fe.first_event_type IS NULL THEN 1 ELSE 0 END AS first_event_missing_flag,
      CASE WHEN fe.first_event_action IS NULL THEN 1 ELSE 0 END AS first_event_action_missing_flag,
      CASE WHEN fe.first_utm_source IS NULL THEN 1 ELSE 0 END AS first_utm_missing_flag,
      CASE WHEN fe.first_device IS NULL OR LOWER(fe.first_device) IN ('unknown', 'missing') THEN 1 ELSE 0 END AS first_device_missing_flag
    FROM onboarding_anchor oa
    LEFT JOIN dim_teacher dt ON oa.teacher_unique_id = dt.teacher_unique_id
    LEFT JOIN first_event_summary fe ON oa.teacher_unique_id = fe.teacher_unique_id
    LEFT JOIN first_session_summary fs ON oa.teacher_unique_id = fs.teacher_unique_id
    WHERE oa.onboarding_anchor_ts IS NOT NULL
    ORDER BY oa.teacher_unique_id
    """
    frame = conn.execute(query).fetchdf()
    for col in ["first_month", "first_session_ts", "first_interaction_ts", "onboarding_anchor_ts", "data_entrada_month"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame

def build_first_session_journey_mart(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
    WITH onboarding AS (
      SELECT *
      FROM mart_onboarding_population_v1
    ),
    first_session_ranked AS (
      SELECT
        o.teacher_unique_id,
        s.session_row_hash,
        s.session_start_ts,
        s.session_end_ts,
        s.duration_sec,
        s.duration_min,
        ROW_NUMBER() OVER (
          PARTITION BY o.teacher_unique_id
          ORDER BY s.session_start_ts NULLS LAST, s.session_row_hash
        ) AS rn
      FROM onboarding o
      LEFT JOIN fct_session_clean s
        ON o.teacher_unique_id = s.teacher_unique_id
       AND s.session_month = CAST(o.first_month AS DATE)
    ),
    first_session AS (
      SELECT * EXCLUDE(rn)
      FROM first_session_ranked
      WHERE rn = 1
    ),
    session_interactions AS (
      SELECT
        fs.teacher_unique_id,
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
       AND fs.session_start_ts IS NOT NULL
       AND i.interaction_ts >= fs.session_start_ts
       AND i.interaction_ts <= fs.session_end_ts
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
        SUM(is_meaningful_event) AS first_session_meaningful_events
      FROM ranked_interactions
      GROUP BY 1
    ),
    meaningful_step_summary AS (
      SELECT
        teacher_unique_id,
        MAX(CASE WHEN rn_meaningful = 1 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_1_token,
        MAX(CASE WHEN rn_meaningful = 2 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_2_token,
        MAX(CASE WHEN rn_meaningful = 3 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_3_token,
        MAX(CASE WHEN rn_meaningful = 4 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_4_token,
        MAX(CASE WHEN rn_meaningful = 5 THEN COALESCE(event_action, event_family, event_type, 'missing') END) AS step_5_token
      FROM meaningful_interactions
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
    first_meaningful AS (
      SELECT
        teacher_unique_id,
        interaction_ts AS first_session_first_meaningful_ts,
        event_type AS first_session_first_meaningful_type,
        event_family AS first_session_first_meaningful_family,
        event_action AS first_session_first_meaningful_action
      FROM meaningful_interactions
      WHERE rn_meaningful = 1
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
    )
    SELECT
      o.*,
      fs.session_row_hash AS first_session_row_hash,
      fs.session_start_ts AS first_session_start_ts,
      fs.session_end_ts AS first_session_end_ts,
      fs.duration_sec AS first_session_duration_sec,
      fs.duration_min AS first_session_duration_min,
      ia.first_session_interactions,
      ia.first_session_downloads,
      ia.first_session_views,
      ia.first_session_other_actions,
      ia.first_session_navigation_events,
      ia.first_session_meaningful_events,
      ms.step_1_token,
      ms.step_2_token,
      ms.step_3_token,
      ms.step_4_token,
      ms.step_5_token,
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
    LEFT JOIN first_session fs ON o.teacher_unique_id = fs.teacher_unique_id
    LEFT JOIN interaction_summary ia ON o.teacher_unique_id = ia.teacher_unique_id
    LEFT JOIN meaningful_step_summary ms ON o.teacher_unique_id = ms.teacher_unique_id
    LEFT JOIN first_event fe ON o.teacher_unique_id = fe.teacher_unique_id
    LEFT JOIN first_meaningful fm ON o.teacher_unique_id = fm.teacher_unique_id
    LEFT JOIN last_event le ON o.teacher_unique_id = le.teacher_unique_id
    ORDER BY o.teacher_unique_id
    """
    journey = conn.execute(query).fetchdf()
    for col in [
        "first_month",
        "first_session_start_ts",
        "first_session_end_ts",
        "first_session_first_event_ts",
        "first_session_first_meaningful_ts",
        "first_session_last_event_ts",
        "onboarding_anchor_ts",
    ]:
        journey[col] = pd.to_datetime(journey[col], errors="coerce")

    numeric_cols = [
        "months_after_entry",
        "first3_interaction_downloads",
        "first3_interaction_views",
        "first3_interaction_other_actions",
        "first7d_events",
        "first7d_active_days",
        "first_session_minutes",
        "first7d_sessions",
        "first7d_session_minutes",
        "first_event_missing_flag",
        "first_event_action_missing_flag",
        "first_utm_missing_flag",
        "first_device_missing_flag",
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
    journey["session_without_interaction_flag"] = (
        (journey["first_session_missing_flag"] == 0) & (journey["first_session_has_interaction_flag"] == 0)
    ).astype(int)

    journey["first_session_entry_surface"] = journey["first_session_first_event_utm_source"].map(setup.normalize_text)
    journey.loc[journey["first_session_entry_surface"] == "missing", "first_session_entry_surface"] = journey["first_utm_source"].map(setup.normalize_text)
    journey["first_session_device_raw"] = journey["first_session_first_event_device"].map(lambda x: setup.normalize_text(x, default="unknown"))

    def device_bucket(row: pd.Series) -> str:
        raw = setup.normalize_text(row.get("first_session_device_raw"), default="unknown")
        if raw in {"mobile", "desktop"}:
            return raw
        fallback = setup.normalize_text(row.get("first_device"), default="unknown")
        if fallback in {"mobile", "desktop"}:
            return fallback
        return "unknown"

    def action_group(value: Any) -> str:
        text = setup.normalize_text(value)
        if text in {"download", "view", "create", "share", "submit"}:
            return text
        if text == "missing":
            return "missing"
        return "other"

    def exit_state(row: pd.Series) -> str:
        if int(row["first_session_missing_flag"]) == 1:
            return "missing_first_session"
        if int(row["first_session_has_interaction_flag"]) == 0:
            return "session_without_interaction"
        if pd.to_numeric(row["first_session_downloads"], errors="coerce") > 0:
            return "ended_after_download"
        if pd.to_numeric(row["first_session_views"], errors="coerce") > 0 and pd.to_numeric(row["first_session_other_actions"], errors="coerce") <= 0:
            return "ended_after_view_only"
        if pd.to_numeric(row["first_session_other_actions"], errors="coerce") > 0:
            return "ended_after_activity"
        return "other_exit_state"

    journey["first_session_device_bucket"] = journey.apply(device_bucket, axis=1)
    journey["secs_to_first_interaction"] = (journey["first_session_first_event_ts"] - journey["first_session_start_ts"]).dt.total_seconds()
    journey["secs_to_first_meaningful_action"] = (journey["first_session_first_meaningful_ts"] - journey["first_session_start_ts"]).dt.total_seconds()
    journey["first_session_first_meaningful_action_group"] = journey["first_session_first_meaningful_action"].apply(action_group)
    journey["first_session_first_event_action_group"] = journey["first_session_first_event_action"].apply(action_group)
    journey["first_session_exit_state"] = journey.apply(exit_state, axis=1)
    for token_col in ["step_1_token", "step_2_token", "step_3_token", "step_4_token", "step_5_token"]:
        journey[token_col] = journey[token_col].map(setup.normalize_text)
    step_cols = ["step_1_token", "step_2_token", "step_3_token", "step_4_token", "step_5_token"]
    journey["observed_step_count_first5"] = (journey[step_cols].fillna("missing") != "missing").sum(axis=1)
    journey["step_sequence_first5"] = journey[step_cols].fillna("missing").agg(">".join, axis=1)
    journey["step_sequence_observed_first5"] = journey[step_cols].apply(
        lambda row: ">".join([token for token in row.tolist() if setup.normalize_text(token) != "missing"]) if any(setup.normalize_text(token) != "missing" for token in row.tolist()) else "missing",
        axis=1,
    )
    return journey

def build_future_metrics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    max_observed_ts = conn.execute(
        """
        SELECT MAX(ts) AS max_ts
        FROM (
          SELECT MAX(session_start_ts) AS ts FROM fct_session_clean
          UNION ALL
          SELECT MAX(interaction_ts) AS ts FROM fct_interaction_clean
          UNION ALL
          SELECT MAX(formation_ts) AS ts FROM fct_formation_clean
          UNION ALL
          SELECT MAX(mari_created_ts) AS ts FROM fct_mari_conversation_resolved
          UNION ALL
          SELECT MAX(help_ts) AS ts FROM fct_mari_help_resolved
        )
        """
    ).fetchone()[0]
    conn.execute("CREATE OR REPLACE TEMP VIEW official_anchor AS SELECT * FROM mart_first_session_journey_v1")
    query = f"""
    WITH base AS (
      SELECT
        teacher_unique_id,
        first_month,
        onboarding_anchor_ts AS anchor_ts,
        onboarding_anchor_ts + INTERVAL 7 DAY AS label_start_ts,
        onboarding_anchor_ts + INTERVAL {setup.LABEL_WINDOW_DAYS + 7} DAY AS label_end_ts,
        onboarding_anchor_ts + INTERVAL {setup.LABEL_WINDOW_DAYS + 7 + setup.POST_LABEL_BLOCK_DAYS} DAY AS validator_1_end_ts,
        onboarding_anchor_ts + INTERVAL {setup.LABEL_WINDOW_DAYS + 7 + setup.POST_LABEL_BLOCK_DAYS * 2} DAY AS validator_2_end_ts,
        onboarding_anchor_ts + INTERVAL {setup.LABEL_WINDOW_DAYS + 7 + setup.POST_LABEL_BLOCK_DAYS * 3} DAY AS validator_3_end_ts
      FROM official_anchor
      WHERE onboarding_anchor_ts IS NOT NULL
    ),
    session_label AS (
      SELECT
        b.teacher_unique_id,
        COUNT(*) AS future_sessions,
        SUM(COALESCE(s.duration_min, 0)) AS future_session_minutes
      FROM base b
      LEFT JOIN fct_session_clean s
        ON b.teacher_unique_id = s.teacher_unique_id
       AND s.session_start_ts >= b.label_start_ts
       AND s.session_start_ts < b.label_end_ts
      GROUP BY 1
    ),
    interaction_label AS (
      SELECT
        b.teacher_unique_id,
        COUNT(*) AS future_interactions,
        SUM(COALESCE(i.is_activity_event, 0)) AS future_activity_events,
        COUNT(DISTINCT CAST(i.interaction_ts AS DATE)) AS future_active_days,
        COUNT(DISTINCT COALESCE(NULLIF(TRIM(i.event_action), ''), NULLIF(TRIM(i.event_family), ''), NULLIF(TRIM(i.event_type), ''))) AS future_distinct_actions,
        SUM(COALESCE(i.is_download_event, 0)) AS future_downloads,
        SUM(COALESCE(i.is_content_view_event, 0)) AS future_content_views,
        SUM(COALESCE(i.lesson_mapped_flag, 0)) AS future_mapped_lessons
      FROM base b
      LEFT JOIN fct_interaction_clean i
        ON b.teacher_unique_id = i.teacher_unique_id
       AND i.interaction_ts >= b.label_start_ts
       AND i.interaction_ts < b.label_end_ts
      GROUP BY 1
    ),
    formation_label AS (
      SELECT
        b.teacher_unique_id,
        COUNT(*) AS future_formation_events
      FROM base b
      LEFT JOIN fct_formation_clean f
        ON b.teacher_unique_id = f.teacher_unique_id
       AND f.formation_ts >= b.label_start_ts
       AND f.formation_ts < b.label_end_ts
      GROUP BY 1
    ),
    mari_help_label AS (
      SELECT
        b.teacher_unique_id,
        COUNT(*) AS future_mari_help_events
      FROM base b
      LEFT JOIN fct_mari_help_resolved h
        ON b.teacher_unique_id = h.teacher_unique_id
       AND h.help_ts >= b.label_start_ts
       AND h.help_ts < b.label_end_ts
      GROUP BY 1
    ),
    mari_conversation_label AS (
      SELECT
        b.teacher_unique_id,
        COUNT(*) AS future_mari_conversation_events
      FROM base b
      LEFT JOIN fct_mari_conversation_resolved c
        ON b.teacher_unique_id = c.teacher_unique_id
       AND c.mari_created_ts >= b.label_start_ts
       AND c.mari_created_ts < b.label_end_ts
      GROUP BY 1
    ),
    weekly_sessions AS (
      SELECT
        b.teacher_unique_id,
        FLOOR(DATE_DIFF('day', b.label_start_ts, s.session_start_ts) / 7.0) AS week_idx,
        COUNT(*) AS session_count
      FROM base b
      INNER JOIN fct_session_clean s
        ON b.teacher_unique_id = s.teacher_unique_id
       AND s.session_start_ts >= b.label_start_ts
       AND s.session_start_ts < b.label_end_ts
      GROUP BY 1, 2
    ),
    weekly_activity AS (
      SELECT
        b.teacher_unique_id,
        FLOOR(DATE_DIFF('day', b.label_start_ts, i.interaction_ts) / 7.0) AS week_idx,
        SUM(COALESCE(i.is_activity_event, 0)) AS activity_event_count
      FROM base b
      INNER JOIN fct_interaction_clean i
        ON b.teacher_unique_id = i.teacher_unique_id
       AND i.interaction_ts >= b.label_start_ts
       AND i.interaction_ts < b.label_end_ts
      GROUP BY 1, 2
    ),
    weekly_business AS (
      SELECT
        COALESCE(ws.teacher_unique_id, wa.teacher_unique_id) AS teacher_unique_id,
        COALESCE(ws.week_idx, wa.week_idx) AS week_idx,
        COALESCE(ws.session_count, 0) AS session_count,
        COALESCE(wa.activity_event_count, 0) AS activity_event_count
      FROM weekly_sessions ws
      FULL OUTER JOIN weekly_activity wa
        ON ws.teacher_unique_id = wa.teacher_unique_id
       AND ws.week_idx = wa.week_idx
    ),
    weekly_business_count AS (
      SELECT
        teacher_unique_id,
        COUNT(*) FILTER (WHERE session_count > 0 AND activity_event_count > 0) AS future_business_active_weeks
      FROM weekly_business
      GROUP BY 1
    ),
    post_sessions AS (
      SELECT
        b.teacher_unique_id,
        FLOOR(DATE_DIFF('day', b.label_end_ts, s.session_start_ts) / {setup.POST_LABEL_BLOCK_DAYS}.0) AS block_idx,
        COUNT(*) AS session_count
      FROM base b
      INNER JOIN fct_session_clean s
        ON b.teacher_unique_id = s.teacher_unique_id
       AND s.session_start_ts >= b.label_end_ts
       AND s.session_start_ts < b.validator_3_end_ts
      GROUP BY 1, 2
    ),
    post_activity AS (
      SELECT
        b.teacher_unique_id,
        FLOOR(DATE_DIFF('day', b.label_end_ts, i.interaction_ts) / {setup.POST_LABEL_BLOCK_DAYS}.0) AS block_idx,
        SUM(COALESCE(i.is_activity_event, 0)) AS activity_event_count
      FROM base b
      INNER JOIN fct_interaction_clean i
        ON b.teacher_unique_id = i.teacher_unique_id
       AND i.interaction_ts >= b.label_end_ts
       AND i.interaction_ts < b.validator_3_end_ts
      GROUP BY 1, 2
    ),
    post_active AS (
      SELECT
        COALESCE(ps.teacher_unique_id, pa.teacher_unique_id) AS teacher_unique_id,
        COALESCE(ps.block_idx, pa.block_idx) AS block_idx,
        COALESCE(ps.session_count, 0) AS session_count,
        COALESCE(pa.activity_event_count, 0) AS activity_event_count
      FROM post_sessions ps
      FULL OUTER JOIN post_activity pa
        ON ps.teacher_unique_id = pa.teacher_unique_id
       AND ps.block_idx = pa.block_idx
    ),
    post_validator AS (
      SELECT
        teacher_unique_id,
        MAX(CASE WHEN block_idx = 0 AND session_count > 0 AND activity_event_count > 0 THEN 1 ELSE 0 END) AS returned_active_post_label_m1,
        MAX(CASE WHEN block_idx = 1 AND session_count > 0 AND activity_event_count > 0 THEN 1 ELSE 0 END) AS returned_active_post_label_m2,
        MAX(CASE WHEN block_idx = 2 AND session_count > 0 AND activity_event_count > 0 THEN 1 ELSE 0 END) AS returned_active_post_label_m3
      FROM post_active
      GROUP BY 1
    ),
    post_days AS (
      SELECT
        b.teacher_unique_id,
        COUNT(DISTINCT CAST(i.interaction_ts AS DATE)) AS active_days_post_label_3m
      FROM base b
      LEFT JOIN fct_interaction_clean i
        ON b.teacher_unique_id = i.teacher_unique_id
       AND i.interaction_ts >= b.label_end_ts
       AND i.interaction_ts < b.validator_3_end_ts
       AND COALESCE(i.is_activity_event, 0) = 1
      GROUP BY 1
    )
    SELECT
      b.teacher_unique_id,
      b.first_month,
      b.anchor_ts,
      b.label_start_ts,
      b.label_end_ts,
      b.validator_1_end_ts,
      b.validator_2_end_ts,
      b.validator_3_end_ts,
      CASE WHEN b.validator_3_end_ts <= TIMESTAMP '{pd.Timestamp(max_observed_ts).strftime("%Y-%m-%d %H:%M:%S")}' THEN 1 ELSE 0 END AS full_followup_observed_flag,
      COALESCE(sl.future_sessions, 0) AS future_sessions,
      COALESCE(sl.future_session_minutes, 0) AS future_session_minutes,
      COALESCE(il.future_interactions, 0) AS future_interactions,
      COALESCE(il.future_activity_events, 0) AS future_activity_events,
      COALESCE(il.future_active_days, 0) AS future_active_days,
      COALESCE(il.future_distinct_actions, 0) AS future_distinct_actions,
      COALESCE(il.future_downloads, 0) AS future_downloads,
      COALESCE(il.future_content_views, 0) AS future_content_views,
      COALESCE(il.future_mapped_lessons, 0) AS future_mapped_lessons,
      COALESCE(fl.future_formation_events, 0) AS future_formation_events,
      COALESCE(mhl.future_mari_help_events, 0) AS future_mari_help_events,
      COALESCE(mcl.future_mari_conversation_events, 0) AS future_mari_conversation_events,
      COALESCE(wbc.future_business_active_weeks, 0) AS future_business_active_weeks,
      COALESCE(pv.returned_active_post_label_m1, 0) AS returned_active_post_label_m1,
      COALESCE(pv.returned_active_post_label_m2, 0) AS returned_active_post_label_m2,
      COALESCE(pv.returned_active_post_label_m3, 0) AS returned_active_post_label_m3,
      COALESCE(pd.active_days_post_label_3m, 0) AS active_days_post_label_3m,
      CASE
        WHEN COALESCE(pv.returned_active_post_label_m1, 0)
           + COALESCE(pv.returned_active_post_label_m2, 0)
           + COALESCE(pv.returned_active_post_label_m3, 0) >= 2 THEN 1
        ELSE 0
      END AS sustained_active_2of3_post_label
    FROM base b
    LEFT JOIN session_label sl ON b.teacher_unique_id = sl.teacher_unique_id
    LEFT JOIN interaction_label il ON b.teacher_unique_id = il.teacher_unique_id
    LEFT JOIN formation_label fl ON b.teacher_unique_id = fl.teacher_unique_id
    LEFT JOIN mari_help_label mhl ON b.teacher_unique_id = mhl.teacher_unique_id
    LEFT JOIN mari_conversation_label mcl ON b.teacher_unique_id = mcl.teacher_unique_id
    LEFT JOIN weekly_business_count wbc ON b.teacher_unique_id = wbc.teacher_unique_id
    LEFT JOIN post_validator pv ON b.teacher_unique_id = pv.teacher_unique_id
    LEFT JOIN post_days pd ON b.teacher_unique_id = pd.teacher_unique_id
    ORDER BY b.teacher_unique_id
    """
    metrics = conn.execute(query).fetchdf()
    for col in ["first_month", "anchor_ts", "label_start_ts", "label_end_ts", "validator_1_end_ts", "validator_2_end_ts", "validator_3_end_ts"]:
        metrics[col] = pd.to_datetime(metrics[col], errors="coerce")
    numeric_cols = [col for col in metrics.columns if col not in {"teacher_unique_id", "first_month", "anchor_ts", "label_start_ts", "label_end_ts", "validator_1_end_ts", "validator_2_end_ts", "validator_3_end_ts"}]
    for col in numeric_cols:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    return metrics

def select_active_features(
    fit_train: pd.DataFrame,
    feature_names: Sequence[str],
    calibration_holdout: pd.DataFrame | None = None,
) -> list[str]:
    outer_active = [name for name in feature_names if name in fit_train.columns and not fit_train[name].isna().all()]
    if not outer_active:
        return []
    if calibration_holdout is None or calibration_holdout.empty:
        return outer_active
    return [
        name
        for name in outer_active
        if not calibration_holdout[name].isna().all()
    ]

def build_feature_eligibility_log(feature_registry: pd.DataFrame, track_registry: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    for feature in feature_registry.to_dict(orient="records"):
        for track in track_registry["track_name"]:
            rows.append(
                {
                    "feature_name": feature["feature_name"],
                    "track_name": track,
                    "eligible_flag": int(feature[f"allowed_in_{track}"]),
                    "feature_class": feature["feature_class"],
                    "pit_class": feature["pit_class"],
                    "behavior_family": feature["behavior_family"],
                }
            )
    return pd.DataFrame(rows)

def build_leakage_audit(feature_registry: pd.DataFrame, label_registry: pd.DataFrame, scoring_scenarios: pd.DataFrame) -> pd.DataFrame:
    feature_rows = feature_registry.to_dict(orient="records")
    label_rows = label_registry[label_registry["official_flag"] == 1].to_dict(orient="records")
    scenario_rows = scoring_scenarios.to_dict(orient="records")
    track_allowed_classes = {
        "S1": {"context", "s1"},
        "S7": {"context", "s7"},
        "S1_PLUS_S7": {"context", "s1", "s7"},
        "STRICT_CONTEXT": {"context"},
    }
    rows: List[dict[str, Any]] = []
    for scenario in scenario_rows:
        track_name = scenario["track_name"]
        definition_name = scenario["definition_name"]
        matching_labels = [row for row in label_rows if row["label_name"] == definition_name]
        if not matching_labels:
            continue
        label_row = matching_labels[0]
        label_sources = set(json.loads(label_row["source_columns_json"]))
        label_source_table = str(label_row["source_table"])
        for feature in feature_rows:
            if int(feature[f"allowed_in_{track_name}"]) != 1:
                continue
            feature_sources = set(json.loads(feature["source_columns_json"]))
            same_source_column_flag = int(bool(feature_sources & label_sources))
            temporal_window_ok_flag = int(label_row["window_start_day"] > scenario["score_window_end_day"])
            feature_source_table = str(feature["source_table"])
            source_table_matches_label_table_flag = int(feature_source_table == label_source_table)
            source_column_future_named_flag = int(
                any(
                    str(source_col).startswith("future_")
                    or "_post_label" in str(source_col)
                    or str(source_col).startswith("returned_active_post_label")
                    for source_col in feature_sources
                )
            )
            source_touches_future_window_flag = int(
                source_table_matches_label_table_flag == 1
                or source_column_future_named_flag == 1
                or same_source_column_flag == 1
            )
            available_at_score_time_flag = int(feature[f"allowed_in_{track_name}"] == 1)
            pit_safe_flag = int(feature["feature_class"] in track_allowed_classes.get(track_name, set()))
            high_risk_future_touch_flag = int(source_touches_future_window_flag == 1 or temporal_window_ok_flag == 0)
            leakage_flag = int(same_source_column_flag == 1 or temporal_window_ok_flag == 0 or source_touches_future_window_flag == 1)
            rows.append(
                {
                    "problem_key": scenario["problem_key"],
                    "definition_name": definition_name,
                    "track_name": track_name,
                    "feature_name": feature["feature_name"],
                    "feature_source_table": feature_source_table,
                    "feature_source_columns_json": feature["source_columns_json"],
                    "feature_class": feature["feature_class"],
                    "feature_pit_class": feature["pit_class"],
                    "feature_behavior_family": feature["behavior_family"],
                    "label_source_table": label_source_table,
                    "label_source_columns_json": label_row["source_columns_json"],
                    "label_window_start_day": int(label_row["window_start_day"]),
                    "label_window_end_day": int(label_row["window_end_day"]),
                    "score_window_end_day": int(scenario["score_window_end_day"]),
                    "available_at_score_time_flag": available_at_score_time_flag,
                    "pit_safe_flag": pit_safe_flag,
                    "same_source_column_flag": same_source_column_flag,
                    "source_table_matches_label_table_flag": source_table_matches_label_table_flag,
                    "source_column_future_named_flag": source_column_future_named_flag,
                    "source_touches_future_window_flag": source_touches_future_window_flag,
                    "temporal_window_ok_flag": temporal_window_ok_flag,
                    "high_risk_future_touch_flag": high_risk_future_touch_flag,
                    "definition_b_specific_flag": int(definition_name == "definition_b_label"),
                    "leakage_flag": leakage_flag,
                }
            )
    return pd.DataFrame(rows)


def summarize_leakage_audit(leakage_audit: pd.DataFrame) -> pd.DataFrame:
    if leakage_audit.empty:
        return pd.DataFrame(
            columns=[
                "problem_key",
                "definition_name",
                "track_name",
                "audited_features",
                "features_with_leakage_flag",
                "features_with_source_overlap",
                "features_with_future_named_source",
                "features_touching_label_source_table",
                "features_with_temporal_violation",
                "features_with_high_risk_future_touch",
                "all_features_available_at_score_time_flag",
                "all_features_pit_safe_flag",
                "any_leakage_flag",
            ]
        )
    grouped = (
        leakage_audit.groupby(["problem_key", "definition_name", "track_name"], as_index=False)
        .agg(
            audited_features=("feature_name", "nunique"),
            features_with_leakage_flag=("leakage_flag", "sum"),
            features_with_source_overlap=("same_source_column_flag", "sum"),
            features_with_future_named_source=("source_column_future_named_flag", "sum"),
            features_touching_label_source_table=("source_table_matches_label_table_flag", "sum"),
            features_with_temporal_violation=("temporal_window_ok_flag", lambda values: int((pd.Series(values) == 0).sum())),
            features_with_high_risk_future_touch=("high_risk_future_touch_flag", "sum"),
            all_features_available_at_score_time_flag=("available_at_score_time_flag", "min"),
            all_features_pit_safe_flag=("pit_safe_flag", "min"),
        )
    )
    grouped["any_leakage_flag"] = (grouped["features_with_leakage_flag"] > 0).astype(int)
    return grouped

def build_official_frame(journey: pd.DataFrame, future_metrics: pd.DataFrame, selection_df: pd.DataFrame) -> pd.DataFrame:
    frame = journey.merge(future_metrics, on=["teacher_unique_id", "first_month"], how="inner", suffixes=("", "_future"))
    frame = frame.loc[frame["full_followup_observed_flag"] == 1].copy()
    official_a = selection_df[(selection_df["definition_group"] == "definition_a") & selection_df["official_status"].str.startswith("official")]
    label_columns: dict[str, pd.Series] = {}
    for row in official_a.to_dict(orient="records"):
        rule = json.loads(row["rule_json"])
        name = setup.build_definition_a_label_name(rule)
        label_columns[name] = setup.apply_rule_to_frame(frame, rule).astype(int)
    definition_b_spec = setup.get_definition_b_spec()
    metric_name = definition_b_spec["metric_name"]
    operator = definition_b_spec["operator"]
    threshold = float(definition_b_spec["threshold"])
    label_columns["definition_b_label"] = setup.apply_operator(frame[metric_name], operator, threshold).astype(int)
    if label_columns:
        frame = pd.concat([frame, pd.DataFrame(label_columns, index=frame.index)], axis=1)
    return frame
