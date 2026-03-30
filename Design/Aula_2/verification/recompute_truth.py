#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import load_manifest_spec, utc_now_iso, write_json


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def cramers_v_from_table(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    mat = table.to_numpy()
    if mat.size == 0:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(mat)
    n = mat.sum()
    if n == 0:
        return float("nan")
    r, c = mat.shape
    denom = n * max(1, min(r - 1, c - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def build_activity_tier(interaction_count: pd.Series) -> pd.Series:
    values = interaction_count.fillna(0)
    result = pd.Series(index=values.index, data="inativo", dtype="object")
    non_zero_idx = values[values > 0].index
    if len(non_zero_idx) < 10:
        result.loc[non_zero_idx] = "ativo"
        return result
    q = values.loc[non_zero_idx]
    try:
        tiers = pd.qcut(q, q=3, labels=["interessado", "medio", "heavy"])
        result.loc[non_zero_idx] = tiers.astype(str)
    except ValueError:
        result.loc[non_zero_idx] = "ativo"
    return result


def build_views(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    conn.execute("PRAGMA threads=4")
    conn.execute(
        f"CREATE VIEW dim_teachers AS SELECT * FROM read_csv('{q(data_dir / 'dim_teachers.csv')}', delim=';', header=true, ignore_errors=true)"
    )
    for fname, view in [
        ("fct_teachers_entries.csv", "entries"),
        ("fct_teachers_contents_interactions.csv", "interactions"),
        ("stg_lessons.csv", "lessons"),
        ("stg_formation.csv", "formation"),
        ("stg_mari_ia_conversation.csv", "mari_conv"),
        ("stg_mari_ia_reports.csv", "mari_reports"),
        ("fct_mari_ia_eventos_isso_ajudou.csv", "mari_help"),
    ]:
        conn.execute(
            f"CREATE VIEW {view} AS SELECT * FROM read_csv_auto('{q(data_dir / fname)}', header=true)"
        )


def build_teacher_dataset(conn: duckdb.DuckDBPyConnection, churn_days: int, conversion_days: int) -> pd.DataFrame:
    conversion_hours = conversion_days * 24.0
    sql = f"""
    WITH snapshot AS (
        SELECT GREATEST(
            (SELECT max(data_fim) FROM entries),
            (SELECT max(data_inicio) FROM interactions),
            (SELECT max(updatedat) FROM mari_conv),
            (SELECT max(date) FROM mari_help)
        ) AS snapshot_ts
    ),
    entries_agg AS (
        SELECT
            unique_id,
            COUNT(*) AS session_count,
            MIN(data_inicio) AS first_entry_ts,
            MAX(data_fim) AS last_entry_ts,
            SUM(GREATEST(epoch(data_fim) - epoch(data_inicio), 0)) / 60.0 AS total_session_min,
            AVG(GREATEST(epoch(data_fim) - epoch(data_inicio), 0)) / 60.0 AS avg_session_min
        FROM entries
        GROUP BY unique_id
    ),
    interaction_base AS (
        SELECT
            unique_id,
            user_agent_device_type,
            data_inicio,
            event_type,
            id_aula,
            lower(coalesce(event_type, '')) AS event_lower
        FROM interactions
    ),
    interactions_agg AS (
        SELECT
            unique_id,
            COUNT(*) AS interaction_count,
            SUM(CASE WHEN event_lower LIKE '%aula%' THEN 1 ELSE 0 END) AS aula_event_count,
            SUM(CASE WHEN event_lower LIKE '%prova%' THEN 1 ELSE 0 END) AS prova_event_count,
            SUM(CASE WHEN event_lower LIKE '%plano%' THEN 1 ELSE 0 END) AS plano_event_count,
            SUM(CASE WHEN event_lower LIKE '%download%' THEN 1 ELSE 0 END) AS download_event_count,
            SUM(CASE WHEN event_lower LIKE '%visualizacao%' THEN 1 ELSE 0 END) AS visualizacao_event_count,
            SUM(CASE WHEN event_lower LIKE '%metodologia_ativa%' THEN 1 ELSE 0 END) AS metodologia_ativa_event_count,
            SUM(CASE WHEN event_lower LIKE '%_ia%' OR event_lower LIKE '% ia %' OR event_lower LIKE '%mari%' THEN 1 ELSE 0 END) AS ia_event_count,
            COUNT(DISTINCT id_aula) AS unique_lessons_count,
            MIN(data_inicio) AS first_interaction_ts,
            MAX(data_inicio) AS last_interaction_ts,
            MIN(
                CASE
                    WHEN event_lower LIKE '%aula%'
                      OR event_lower LIKE '%prova%'
                      OR event_lower LIKE '%download%'
                      OR event_lower LIKE '%visualizacao%'
                      OR event_lower LIKE '%plano%'
                    THEN data_inicio
                    ELSE NULL
                END
            ) AS first_value_ts
        FROM interaction_base
        GROUP BY unique_id
    ),
    device_counts AS (
        SELECT
            unique_id,
            SUM(CASE WHEN lower(coalesce(user_agent_device_type, '')) = 'desktop' THEN 1 ELSE 0 END) AS desktop_events,
            SUM(CASE WHEN lower(coalesce(user_agent_device_type, '')) = 'mobile' THEN 1 ELSE 0 END) AS mobile_events,
            SUM(CASE WHEN lower(coalesce(user_agent_device_type, '')) = 'tablet' THEN 1 ELSE 0 END) AS tablet_events,
            SUM(CASE WHEN user_agent_device_type IS NULL THEN 1 ELSE 0 END) AS unknown_device_events
        FROM interactions
        GROUP BY unique_id
    ),
    device_primary AS (
        SELECT
            unique_id,
            CASE
                WHEN desktop_events >= mobile_events AND desktop_events >= tablet_events AND desktop_events > 0 THEN 'desktop'
                WHEN mobile_events >= desktop_events AND mobile_events >= tablet_events AND mobile_events > 0 THEN 'mobile'
                WHEN tablet_events > 0 THEN 'tablet'
                ELSE 'unknown'
            END AS primary_device,
            desktop_events,
            mobile_events,
            tablet_events,
            unknown_device_events
        FROM device_counts
    ),
    discipline_pref AS (
        SELECT unique_id, disciplina AS top_discipline, lesson_events
        FROM (
            SELECT
                i.unique_id,
                l.disciplina,
                COUNT(*) AS lesson_events,
                ROW_NUMBER() OVER (PARTITION BY i.unique_id ORDER BY COUNT(*) DESC, l.disciplina) AS rn
            FROM interactions i
            JOIN lessons l
                ON CAST(i.id_aula AS VARCHAR) = CAST(l.id_aula AS VARCHAR)
            WHERE l.disciplina IS NOT NULL
            GROUP BY i.unique_id, l.disciplina
        ) x
        WHERE rn = 1
    ),
    formation_agg AS (
        SELECT
            unique_id_aprendizap AS unique_id,
            COUNT(*) AS formation_records,
            SUM(CASE WHEN completionstatus = 'complete' THEN 1 ELSE 0 END) AS formation_complete_records,
            AVG(progress) AS formation_avg_progress,
            MAX(updatedat) AS formation_last_update_ts
        FROM formation
        GROUP BY unique_id_aprendizap
    ),
    mari_agg AS (
        SELECT
            unique_id_aprendizap AS unique_id,
            COUNT(*) AS mari_conv_count,
            SUM(CASE WHEN userreaction IS NOT NULL THEN 1 ELSE 0 END) AS mari_reaction_count,
            MIN(createdat) AS first_mari_ts,
            MAX(updatedat) AS last_mari_ts
        FROM mari_conv
        GROUP BY unique_id_aprendizap
    )
    SELECT
        d.unique_id,
        d.utm_origin,
        d.tela_origem,
        d.estado,
        d.total_alunos,
        d.tipo_total_alunos,
        d.alunos_diretos,
        d.alunos_indiretos,
        d.login_google,
        d.currentstage,
        d.currentsubject,
        d.selectedstages,
        d.visualizou_metodologia_ativa,
        d.data_entrada,

        COALESCE(e.session_count, 0) AS session_count,
        e.first_entry_ts,
        e.last_entry_ts,
        COALESCE(e.total_session_min, 0.0) AS total_session_min,
        COALESCE(e.avg_session_min, 0.0) AS avg_session_min,

        COALESCE(i.interaction_count, 0) AS interaction_count,
        COALESCE(i.aula_event_count, 0) AS aula_event_count,
        COALESCE(i.prova_event_count, 0) AS prova_event_count,
        COALESCE(i.plano_event_count, 0) AS plano_event_count,
        COALESCE(i.download_event_count, 0) AS download_event_count,
        COALESCE(i.visualizacao_event_count, 0) AS visualizacao_event_count,
        COALESCE(i.metodologia_ativa_event_count, 0) AS metodologia_ativa_event_count,
        COALESCE(i.ia_event_count, 0) AS ia_event_count,
        COALESCE(i.unique_lessons_count, 0) AS unique_lessons_count,
        i.first_interaction_ts,
        i.last_interaction_ts,
        i.first_value_ts,

        COALESCE(dp.primary_device, 'unknown') AS primary_device,
        COALESCE(dp.desktop_events, 0) AS desktop_events,
        COALESCE(dp.mobile_events, 0) AS mobile_events,
        COALESCE(dp.tablet_events, 0) AS tablet_events,
        COALESCE(dp.unknown_device_events, 0) AS unknown_device_events,

        di.top_discipline,
        COALESCE(di.lesson_events, 0) AS top_discipline_events,

        COALESCE(f.formation_records, 0) AS formation_records,
        COALESCE(f.formation_complete_records, 0) AS formation_complete_records,
        f.formation_avg_progress,

        COALESCE(m.mari_conv_count, 0) AS mari_conv_count,
        COALESCE(m.mari_reaction_count, 0) AS mari_reaction_count,

        s.snapshot_ts,

        CASE
            WHEN i.first_value_ts IS NOT NULL
            THEN 1 ELSE 0
        END AS activated_flag,

        CASE
            WHEN d.data_entrada IS NOT NULL
            THEN (epoch(s.snapshot_ts) - epoch(d.data_entrada)) / 86400.0
            ELSE NULL
        END AS account_age_days,

        CASE
            WHEN i.first_value_ts IS NOT NULL
              AND d.data_entrada IS NOT NULL
              AND i.first_value_ts >= d.data_entrada
            THEN (epoch(i.first_value_ts) - epoch(d.data_entrada)) / 3600.0
            ELSE NULL
        END AS time_to_first_value_hours,

        CASE
            WHEN i.first_value_ts IS NOT NULL
              AND d.data_entrada IS NOT NULL
              AND i.first_value_ts >= d.data_entrada
              AND (epoch(i.first_value_ts) - epoch(d.data_entrada)) / 3600.0 <= {conversion_hours}
            THEN 1
            ELSE 0
        END AS converted_within_window,

        GREATEST(
            COALESCE(e.last_entry_ts, d.data_entrada),
            COALESCE(i.last_interaction_ts, d.data_entrada),
            COALESCE(m.last_mari_ts, d.data_entrada),
            d.data_entrada
        ) AS last_activity_ts,

        (epoch(s.snapshot_ts) - epoch(
            GREATEST(
                COALESCE(e.last_entry_ts, d.data_entrada),
                COALESCE(i.last_interaction_ts, d.data_entrada),
                COALESCE(m.last_mari_ts, d.data_entrada),
                d.data_entrada
            )
        )) / 86400.0 AS days_since_last_activity,

        CASE
            WHEN d.data_entrada IS NULL THEN NULL
            WHEN ((epoch(s.snapshot_ts) - epoch(d.data_entrada)) / 86400.0) < {churn_days} THEN NULL
            WHEN i.first_value_ts IS NULL THEN NULL
            ELSE 1
        END AS churn_eligible_flag,

        CASE
            WHEN d.data_entrada IS NULL THEN NULL
            WHEN ((epoch(s.snapshot_ts) - epoch(d.data_entrada)) / 86400.0) < {churn_days} THEN NULL
            WHEN i.first_value_ts IS NULL THEN NULL
            WHEN (
                (epoch(s.snapshot_ts) - epoch(
                    GREATEST(
                        COALESCE(e.last_entry_ts, d.data_entrada),
                        COALESCE(i.last_interaction_ts, d.data_entrada),
                        COALESCE(m.last_mari_ts, d.data_entrada),
                        d.data_entrada
                    )
                )) / 86400.0
            ) > {churn_days}
            THEN 1
            ELSE 0
        END AS churn_label
    FROM dim_teachers d
    CROSS JOIN snapshot s
    LEFT JOIN entries_agg e ON d.unique_id = e.unique_id
    LEFT JOIN interactions_agg i ON d.unique_id = i.unique_id
    LEFT JOIN device_primary dp ON d.unique_id = dp.unique_id
    LEFT JOIN discipline_pref di ON d.unique_id = di.unique_id
    LEFT JOIN formation_agg f ON d.unique_id = f.unique_id
    LEFT JOIN mari_agg m ON d.unique_id = m.unique_id
    """
    teacher_df = conn.execute(sql).fetchdf()

    numeric_candidates = [
        "total_alunos",
        "alunos_diretos",
        "alunos_indiretos",
        "login_google",
        "visualizou_metodologia_ativa",
        "session_count",
        "total_session_min",
        "avg_session_min",
        "interaction_count",
        "aula_event_count",
        "prova_event_count",
        "plano_event_count",
        "download_event_count",
        "visualizacao_event_count",
        "metodologia_ativa_event_count",
        "ia_event_count",
        "unique_lessons_count",
        "desktop_events",
        "mobile_events",
        "tablet_events",
        "unknown_device_events",
        "top_discipline_events",
        "formation_records",
        "formation_complete_records",
        "formation_avg_progress",
        "mari_conv_count",
        "mari_reaction_count",
        "activated_flag",
        "account_age_days",
        "time_to_first_value_hours",
        "converted_within_window",
        "days_since_last_activity",
        "churn_eligible_flag",
        "churn_label",
    ]
    for col in numeric_candidates:
        if col in teacher_df.columns:
            teacher_df[col] = pd.to_numeric(teacher_df[col], errors="coerce")

    teacher_df.loc[teacher_df["time_to_first_value_hours"] < 0, "time_to_first_value_hours"] = np.nan
    teacher_df["activity_tier"] = build_activity_tier(teacher_df["interaction_count"].fillna(0))
    teacher_df["ai_used_flag"] = (
        (teacher_df["mari_conv_count"].fillna(0) > 0)
        | (teacher_df["ia_event_count"].fillna(0) > 0)
    ).astype(int)
    return teacher_df


def normalize_utm(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "missing"
    s = str(x).strip().lower()
    if s in {"", "none", "<na>"}:
        return "missing"
    if "google ads" in s or "seo ads" in s:
        return "paid_search"
    if "seo org" in s:
        return "organic_search"
    if "landing" in s:
        return "landing"
    if "blog" in s:
        return "blog"
    if "mídias sociais" in s or "midias sociais" in s or "social" in s:
        return "social"
    if "convite_escola" in s:
        return "school_invite"
    if "push" in s or "notificacao" in s:
        return "push_or_notification"
    if "mari" in s:
        return "mari"
    return "other"


def compute_users_panel(conn: duckdb.DuckDBPyConnection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users_panel = conn.execute(
        """
        WITH entries_dim AS (
            SELECT e.* FROM entries e INNER JOIN dim_teachers d USING(unique_id)
        ),
        interactions_dim AS (
            SELECT i.* FROM interactions i INNER JOIN dim_teachers d USING(unique_id)
        ),
        new_users AS (
            SELECT date_trunc('month', data_entrada) AS month, COUNT(DISTINCT unique_id)::BIGINT AS new_users
            FROM dim_teachers
            WHERE data_entrada IS NOT NULL
            GROUP BY 1
        ),
        entries_reg AS (
            SELECT date_trunc('month', data_inicio) AS month, COUNT(DISTINCT unique_id)::BIGINT AS mau_registered_entries
            FROM entries_dim
            WHERE data_inicio IS NOT NULL AND lower(coalesce(user_type,''))='registered'
            GROUP BY 1
        ),
        inter_reg AS (
            SELECT date_trunc('month', data_inicio) AS month, COUNT(DISTINCT unique_id)::BIGINT AS mau_registered_interactions
            FROM interactions_dim
            WHERE data_inicio IS NOT NULL AND lower(coalesce(user_type,''))='registered'
            GROUP BY 1
        ),
        all_months AS (
            SELECT month FROM new_users
            UNION
            SELECT month FROM entries_reg
            UNION
            SELECT month FROM inter_reg
        )
        SELECT m.month, n.new_users, e.mau_registered_entries, i.mau_registered_interactions
        FROM all_months m
        LEFT JOIN new_users n USING(month)
        LEFT JOIN entries_reg e USING(month)
        LEFT JOIN inter_reg i USING(month)
        ORDER BY month
        """
    ).fetchdf()

    retention = conn.execute(
        """
        WITH entries_dim AS (
            SELECT e.* FROM entries e INNER JOIN dim_teachers d USING(unique_id)
        ),
        reg_month AS (
            SELECT DISTINCT unique_id, date_trunc('month', data_inicio) AS month
            FROM entries_dim
            WHERE data_inicio IS NOT NULL
              AND lower(coalesce(user_type,''))='registered'
        ),
        active AS (
            SELECT month, COUNT(*)::BIGINT AS active_users
            FROM reg_month
            GROUP BY 1
        ),
        retained AS (
            SELECT r1.month, COUNT(*)::BIGINT AS retained_next_month
            FROM reg_month r1
            INNER JOIN reg_month r2
              ON r1.unique_id = r2.unique_id
             AND date_trunc('month', r1.month + INTERVAL '1 month') = r2.month
            GROUP BY 1
        ),
        max_m AS (SELECT MAX(month) AS max_month FROM reg_month)
        SELECT
            a.month,
            a.active_users,
            COALESCE(r.retained_next_month,0) AS retained_next_month,
            COALESCE(r.retained_next_month,0)::DOUBLE / NULLIF(a.active_users,0) AS retention_rate,
            1 - COALESCE(r.retained_next_month,0)::DOUBLE / NULLIF(a.active_users,0) AS drop_rate
        FROM active a
        LEFT JOIN retained r USING(month)
        CROSS JOIN max_m
        WHERE a.month < max_m.max_month
        ORDER BY a.month
        """
    ).fetchdf()

    cutoff = conn.execute(
        """
        WITH entries_dim AS (
            SELECT e.* FROM entries e INNER JOIN dim_teachers d USING(unique_id)
        ),
        interactions_dim AS (
            SELECT i.* FROM interactions i INNER JOIN dim_teachers d USING(unique_id)
        )
        SELECT
            (SELECT MAX(data_inicio) FROM interactions_dim WHERE lower(coalesce(user_type,''))='registered') AS max_interactions_registered_ts
        """
    ).fetchdf()

    return users_panel, retention, cutoff


def compute_summary_non_survival(conn: duckdb.DuckDBPyConnection, teacher_df: pd.DataFrame) -> Dict[str, Any]:
    state_missing_pct = float(((teacher_df["estado"].isna()) | (teacher_df["estado"].astype(str).str.strip() == "")).mean())
    utm_missing_pct = float(((teacher_df["utm_origin"].isna()) | (teacher_df["utm_origin"].astype(str).str.strip() == "")).mean())

    sessions = conn.execute(
        """
        WITH entries_dim AS (
            SELECT e.* FROM entries e INNER JOIN dim_teachers d USING(unique_id)
        )
        SELECT
            COUNT(*) AS total_sessions,
            SUM(CASE WHEN (epoch(data_fim)-epoch(data_inicio)) <= 5 THEN 1 ELSE 0 END) AS le_5s
        FROM entries_dim
        """
    ).fetchdf().iloc[0]

    return_gap_overall = conn.execute(
        """
        WITH entries_dim AS (
            SELECT e.* FROM entries e INNER JOIN dim_teachers d USING(unique_id)
        ),
        ordered AS (
            SELECT unique_id, data_inicio, LAG(data_inicio) OVER (PARTITION BY unique_id ORDER BY data_inicio) AS prev_ts
            FROM entries_dim
            WHERE data_inicio IS NOT NULL
        ),
        gaps AS (
            SELECT (epoch(data_inicio)-epoch(prev_ts))/86400.0 AS gap_days
            FROM ordered
            WHERE prev_ts IS NOT NULL
        ),
        clean AS (
            SELECT * FROM gaps WHERE gap_days >= 0 AND gap_days <= 365
        )
        SELECT MEDIAN(gap_days) AS median_gap_days
        FROM clean
        """
    ).fetchdf().iloc[0]

    return_gap_heavy = conn.execute(
        """
        WITH entries_dim AS (
            SELECT e.* FROM entries e INNER JOIN dim_teachers d USING(unique_id)
        ),
        interactions_dim AS (
            SELECT i.* FROM interactions i INNER JOIN dim_teachers d USING(unique_id)
        ),
        inter_count AS (
            SELECT unique_id, COUNT(*)::DOUBLE AS interaction_count FROM interactions_dim GROUP BY 1
        ),
        thr AS (
            SELECT quantile(interaction_count, 0.9) AS p90 FROM inter_count WHERE interaction_count > 0
        ),
        flags AS (
            SELECT
                d.unique_id,
                CASE
                    WHEN COALESCE(i.interaction_count,0) > 0
                     AND COALESCE(i.interaction_count,0) >= COALESCE((SELECT p90 FROM thr), 1e18)
                    THEN 'heavy'
                    ELSE 'base_regular'
                END AS profile
            FROM dim_teachers d
            LEFT JOIN inter_count i USING(unique_id)
        ),
        ordered AS (
            SELECT unique_id, data_inicio, LAG(data_inicio) OVER (PARTITION BY unique_id ORDER BY data_inicio) AS prev_ts
            FROM entries_dim
            WHERE data_inicio IS NOT NULL
        ),
        gaps AS (
            SELECT unique_id, (epoch(data_inicio)-epoch(prev_ts))/86400.0 AS gap_days
            FROM ordered
            WHERE prev_ts IS NOT NULL
        ),
        clean AS (
            SELECT * FROM gaps WHERE gap_days >= 0 AND gap_days <= 365
        )
        SELECT f.profile, MEDIAN(c.gap_days) AS median_gap_days
        FROM clean c
        INNER JOIN flags f USING(unique_id)
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()

    heavy_median = np.nan
    base_median = np.nan
    if not return_gap_heavy.empty:
        x = return_gap_heavy[return_gap_heavy["profile"] == "heavy"]
        y = return_gap_heavy[return_gap_heavy["profile"] == "base_regular"]
        if not x.empty:
            heavy_median = float(x.iloc[0]["median_gap_days"])
        if not y.empty:
            base_median = float(y.iloc[0]["median_gap_days"])

    users_panel, retention, cutoff = compute_users_panel(conn)
    users_panel["month"] = pd.to_datetime(users_panel["month"], errors="coerce")
    users_panel = users_panel.sort_values("month")

    latest_new_month = None
    latest_new_count = None
    new_non_null = users_panel.dropna(subset=["new_users"]) if not users_panel.empty else pd.DataFrame()
    if not new_non_null.empty:
        lr = new_non_null.iloc[-1]
        latest_new_month = str(pd.to_datetime(lr["month"]).date())
        latest_new_count = int(lr["new_users"])

    recent_slope = np.nan
    max_inter_ts = pd.NaT
    if not cutoff.empty:
        max_inter_ts = pd.to_datetime(cutoff.iloc[0]["max_interactions_registered_ts"], errors="coerce")

    if pd.notna(max_inter_ts):
        last_complete = (max_inter_ts.to_period("M") - 1).to_timestamp()
        recent = users_panel[(users_panel["month"] <= last_complete) & users_panel["mau_registered_interactions"].notna()].tail(6)
        if len(recent) >= 2:
            x = np.arange(len(recent), dtype=float)
            y = recent["mau_registered_interactions"].astype(float).to_numpy()
            recent_slope = float(np.polyfit(x, y, 1)[0])

    retention_recent_avg = np.nan
    if not retention.empty:
        retention["month"] = pd.to_datetime(retention["month"], errors="coerce")
        retention_recent = retention.dropna(subset=["retention_rate"]).tail(6)
        if len(retention_recent) > 0:
            retention_recent_avg = float(retention_recent["retention_rate"].mean())

    return {
        "state_missing_pct": state_missing_pct,
        "utm_missing_pct": utm_missing_pct,
        "short_sessions_le_5s": int(sessions["le_5s"]),
        "short_sessions_rate_le_5s": float(sessions["le_5s"] / sessions["total_sessions"]),
        "return_gap_median_days": float(return_gap_overall["median_gap_days"]),
        "return_gap_heavy_median_days": heavy_median,
        "return_gap_base_median_days": base_median,
        "latest_new_users_month": latest_new_month,
        "latest_new_users_count": latest_new_count,
        "recent_6m_mau_interactions_slope_users_per_month": recent_slope,
        "retention_recent_avg_6m": retention_recent_avg,
    }


def compute_core_metrics(conn: duckdb.DuckDBPyConnection, teacher_df: pd.DataFrame) -> Dict[str, Any]:
    df = teacher_df.copy()
    df["utm_group"] = df["utm_origin"].apply(normalize_utm)
    df["estado_group"] = df["estado"].fillna("missing").replace("", "missing")

    active = df[df["interaction_count"] > 0]["interaction_count"]
    thr = float(active.quantile(0.90)) if len(active) else 0.0
    df["heavy_user_flag"] = ((df["interaction_count"] >= thr) & (df["interaction_count"] > 0)).astype(int)

    state_stats = (
        df.groupby("estado_group", dropna=False)
        .agg(
            teachers=("unique_id", "count"),
            conversion_rate=("converted_within_window", "mean"),
            churn_rate=("churn_label", "mean"),
            median_interactions=("interaction_count", "median"),
        )
        .reset_index()
    )

    utm_stats = (
        df.groupby("utm_group", dropna=False)
        .agg(
            teachers=("unique_id", "count"),
            conversion_rate=("converted_within_window", "mean"),
            churn_rate=("churn_label", "mean"),
            median_interactions=("interaction_count", "median"),
        )
        .reset_index()
    )

    geo_assoc_rows: List[Dict[str, Any]] = []
    for target in ["converted_within_window", "churn_label", "heavy_user_flag"]:
        tab = pd.crosstab(df["estado_group"], df[target])
        if tab.shape[0] > 1 and tab.shape[1] > 1:
            _, p, _, _ = chi2_contingency(tab)
            geo_assoc_rows.append(
                {
                    "association": f"estado_group vs {target}",
                    "method": "chi2 + cramers_v",
                    "effect_size": cramers_v_from_table(tab),
                    "p_value": float(p),
                }
            )

    for num in ["interaction_count", "session_count", "total_alunos"]:
        temp = df[["estado_group", num]].dropna()
        counts = temp["estado_group"].value_counts()
        valid = counts[counts >= 200].index
        temp = temp[temp["estado_group"].isin(valid)]
        groups = [g[num].to_numpy() for _, g in temp.groupby("estado_group")]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            n = len(temp)
            k = len(groups)
            eta2 = (stat - k + 1) / (n - k) if n > k else np.nan
            geo_assoc_rows.append(
                {
                    "association": f"estado_group vs {num}",
                    "method": "kruskal + eta2",
                    "effect_size": float(eta2) if pd.notna(eta2) else np.nan,
                    "p_value": float(p),
                }
            )

    geo_associations = pd.DataFrame(geo_assoc_rows)

    # Journey path
    seq = df[["unique_id", "first_interaction_ts", "first_value_ts", "aula_event_count", "prova_event_count"]].copy()
    seq["first_aula_ts"] = np.where(seq["aula_event_count"] > 0, seq["first_value_ts"], pd.NaT)
    seq["first_prova_ts"] = np.where(seq["prova_event_count"] > 0, seq["first_value_ts"], pd.NaT)

    # Better path from interactions.
    path_df = conn.execute(
        """
        WITH interactions_dim AS (
            SELECT i.* FROM interactions i INNER JOIN dim_teachers d USING(unique_id)
        ),
        agg AS (
            SELECT
                unique_id,
                MIN(CASE WHEN lower(coalesce(event_type,'')) LIKE '%aula%' THEN data_inicio ELSE NULL END) AS first_aula_ts,
                MIN(CASE WHEN lower(coalesce(event_type,'')) LIKE '%prova%' THEN data_inicio ELSE NULL END) AS first_prova_ts
            FROM interactions_dim
            GROUP BY unique_id
        )
        SELECT * FROM agg
        """
    ).fetchdf()

    if not path_df.empty:
        path_df["first_aula_ts"] = pd.to_datetime(path_df["first_aula_ts"], errors="coerce")
        path_df["first_prova_ts"] = pd.to_datetime(path_df["first_prova_ts"], errors="coerce")
        both = path_df[path_df["first_aula_ts"].notna() & path_df["first_prova_ts"].notna()].copy()
        both["lag_days_prova_minus_aula"] = (both["first_prova_ts"] - both["first_aula_ts"]).dt.total_seconds() / 86400.0
        both["path"] = np.select(
            [both["lag_days_prova_minus_aula"] > 0, both["lag_days_prova_minus_aula"] < 0],
            ["aula_then_prova", "prova_then_aula"],
            default="same_day",
        )
        journey_path = both["path"].value_counts(dropna=False).to_dict()
    else:
        journey_path = {}

    # Correlation blocks
    num_cols = [
        "session_count",
        "interaction_count",
        "aula_event_count",
        "prova_event_count",
        "ia_event_count",
        "total_session_min",
        "time_to_first_value_hours",
        "total_alunos",
        "converted_within_window",
        "churn_label",
        "heavy_user_flag",
    ]
    num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr(method="spearman")
    pairs: List[Dict[str, Any]] = []
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1 :]:
            val = corr.loc[c1, c2]
            if pd.notna(val):
                pairs.append({"var1": c1, "var2": c2, "spearman": float(val), "abs_spearman": float(abs(val))})
    top_corr_pairs = pd.DataFrame(pairs).sort_values("abs_spearman", ascending=False).head(20)

    cat_cols = ["estado_group", "utm_group", "primary_device", "currentstage"]
    cat_pairs: List[Dict[str, Any]] = []
    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i + 1 :]:
            tab = pd.crosstab(df[c1], df[c2])
            if tab.shape[0] > 1 and tab.shape[1] > 1:
                cat_pairs.append({"var1": c1, "var2": c2, "cramers_v": cramers_v_from_table(tab)})
    cat_corr_pairs = pd.DataFrame(cat_pairs).sort_values("cramers_v", ascending=False)

    max_geo_effect_size = float(geo_associations["effect_size"].max()) if not geo_associations.empty else np.nan

    return {
        "state_stats": state_stats,
        "utm_stats": utm_stats,
        "geo_associations": geo_associations,
        "top_corr_pairs": top_corr_pairs,
        "cat_corr_pairs": cat_corr_pairs,
        "journey_path_counts": journey_path,
        "max_geo_effect_size": max_geo_effect_size,
    }


def compute_hotjar_summary(data_dir: Path) -> Dict[str, Any]:
    files = [
        "hotjar_pesquisa_desktop.xlsx",
        "hotjar_pesquisa_mobile.xlsx",
        "hotjar_teste_interesse.xlsx",
    ]
    frames: List[pd.DataFrame] = []
    for fname in files:
        p = data_dir / fname
        if not p.exists():
            continue
        df = pd.read_excel(p)
        df["source_file"] = fname
        frames.append(df)

    if not frames:
        return {"rows": 0, "feedback_repeat_users": 0}

    hot = pd.concat(frames, ignore_index=True, sort=False)

    user_col = None
    for c in hot.columns:
        if "hotjar user id" in c.lower():
            user_col = c
            break

    repeat_users = 0
    if user_col is not None:
        vc = hot[user_col].value_counts(dropna=True)
        repeat_users = int((vc > 1).sum())

    return {
        "rows": int(len(hot)),
        "feedback_repeat_users": repeat_users,
    }


def h1_h15(teacher_df: pd.DataFrame, conn: duckdb.DuckDBPyConnection, monthly_df: pd.DataFrame, hotjar: Dict[str, Any], alpha: float, min_segment_n: int, random_seed: int, max_cluster_sample: int = 50000) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    def push(hid: str, status: str, evidence: str, p: Any = None, eff: Any = None, n: Any = None) -> None:
        results.append(
            {
                "hypothesis_id": hid,
                "status": status,
                "evidence": evidence,
                "p_value": None if p is None else float(p),
                "effect_size": None if eff is None else float(eff),
                "n_obs": None if n is None else int(n),
            }
        )

    # H1
    d = teacher_df[["aula_event_count", "prova_event_count"]].fillna(0)
    d = d[(d["aula_event_count"] + d["prova_event_count"]) > 0]
    if len(d) < 50:
        push("H1", "inconclusive", "Insufficient teachers with aula/prova activity.", n=len(d))
    else:
        diff = d["aula_event_count"] - d["prova_event_count"]
        try:
            _, p = wilcoxon(diff)
        except ValueError:
            p = 1.0
        med = float(np.median(diff))
        if p < alpha and med > 0:
            st = "validated"
        elif p < alpha and med < 0:
            st = "rejected"
        else:
            st = "inconclusive"
        push("H1", st, f"median_diff={med:.3f}; p={p:.3g}", p=p, eff=med, n=len(d))

    # H2
    d2 = monthly_df.copy()
    d2 = d2[(d2["prova_events"].fillna(0) > 0) & (d2["aula_events"].fillna(0) > 0)]
    if len(d2) < 6:
        push("H2", "inconclusive", "Not enough months with both aula/prova activity.", n=len(d2))
    else:
        cont = np.vstack([d2["aula_events"].to_numpy(), d2["prova_events"].to_numpy()])
        chi2, p, _, _ = chi2_contingency(cont)
        n = cont.sum()
        r, c = cont.shape
        v = np.sqrt(chi2 / (n * max(1, min(r - 1, c - 1)))) if n > 0 else np.nan
        push("H2", "validated" if p < alpha else "inconclusive", f"chi2={chi2:.2f}; p={p:.3g}; v={v:.3f}", p=p, eff=v, n=len(d2))

    # H3
    d3 = teacher_df[["time_to_first_value_hours", "churn_label"]].dropna()
    if len(d3) < 100:
        push("H3", "inconclusive", "Insufficient data.", n=len(d3))
    else:
        rho, p = spearmanr(d3["time_to_first_value_hours"], d3["churn_label"])
        if p < alpha and rho >= 0.05:
            st = "validated"
        elif p < alpha and rho <= -0.05:
            st = "rejected"
        else:
            st = "inconclusive"
        push("H3", st, f"rho={rho:.3f}; p={p:.3g}", p=p, eff=rho, n=len(d3))

    # H4
    seg_cols = ["estado", "currentstage", "primary_device", "utm_origin", "top_discipline"]
    sig = 0
    max_eff = 0.0
    min_p = None
    for col in seg_cols:
        if col not in teacher_df.columns:
            continue
        x = teacher_df[[col, "converted_within_window"]].copy()
        x[col] = x[col].fillna("<NULL>").astype(str)
        cnt = x[col].value_counts()
        valid = cnt[cnt >= min_segment_n].index
        x = x[x[col].isin(valid)]
        if x[col].nunique() < 2:
            continue
        tab = pd.crosstab(x[col], x["converted_within_window"]).to_numpy()
        if tab.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(tab)
        n = tab.sum()
        r, c = tab.shape
        v = np.sqrt(chi2 / (n * max(1, min(r - 1, c - 1)))) if n > 0 else 0.0
        rates = x.groupby(col, dropna=False)["converted_within_window"].mean().sort_values(ascending=False)
        spread = float(rates.max() - rates.min())
        if p < alpha and spread >= 0.03:
            sig += 1
        max_eff = max(max_eff, float(v))
        min_p = p if min_p is None else min(min_p, p)
    push("H4", "validated" if sig > 0 else "inconclusive", f"significant_segments={sig}", p=min_p, eff=max_eff, n=len(teacher_df))

    # H5
    x5 = teacher_df[["primary_device", "churn_label", "converted_within_window", "interaction_count"]].copy()
    x5 = x5[x5["primary_device"].isin(["mobile", "desktop"])]
    if len(x5) < 500:
        push("H5", "inconclusive", "Insufficient mobile/desktop teachers.", n=len(x5))
    else:
        churn_tab = pd.crosstab(x5["primary_device"], x5["churn_label"])
        p_churn = 1.0
        v_churn = np.nan
        if churn_tab.shape == (2, 2):
            _, p_churn, _, _ = chi2_contingency(churn_tab.to_numpy())
            v_churn = cramers_v_from_table(churn_tab)
        conv_tab = pd.crosstab(x5["primary_device"], x5["converted_within_window"])
        p_conv = 1.0
        v_conv = np.nan
        if conv_tab.shape == (2, 2):
            _, p_conv, _, _ = chi2_contingency(conv_tab.to_numpy())
            v_conv = cramers_v_from_table(conv_tab)
        m = x5[x5["primary_device"] == "mobile"]["interaction_count"].fillna(0)
        d = x5[x5["primary_device"] == "desktop"]["interaction_count"].fillna(0)
        _, p_inter = mannwhitneyu(m, d, alternative="two-sided")
        st = "validated" if (p_churn < alpha or p_conv < alpha or p_inter < alpha) else "inconclusive"
        push("H5", st, f"p_churn={p_churn:.3g}; p_conv={p_conv:.3g}; p_inter={p_inter:.3g}", p=min(p_churn, p_conv, p_inter), eff=max(np.nan_to_num(v_churn), np.nan_to_num(v_conv)), n=len(x5))

    # H6
    h6 = conn.execute(
        """
        SELECT
            SUM(CASE WHEN lower(coalesce(event_type,'')) LIKE '%search%' OR lower(coalesce(event_type,'')) LIKE '%busca%' THEN 1 ELSE 0 END) AS search_events,
            SUM(CASE WHEN lower(coalesce(event_type,'')) LIKE '%click%' THEN 1 ELSE 0 END) AS click_events
        FROM interactions
        """
    ).fetchdf().iloc[0]
    se = int(h6["search_events"] or 0)
    ce = int(h6["click_events"] or 0)
    if se == 0:
        push("H6", "not_testable", f"No search events found; click_events={ce}")
    else:
        push("H6", "inconclusive", f"search_events={se}; click_events={ce}")

    # H7
    h7 = conn.execute(
        """
        SELECT
            COUNT(DISTINCT h.user_id) AS help_users,
            COUNT(DISTINCT CASE WHEN d.unique_id IS NOT NULL THEN h.user_id END) AS help_users_in_dim
        FROM mari_help h
        LEFT JOIN dim_teachers d ON h.user_id = d.unique_id
        """
    ).fetchdf().iloc[0]
    hu = int(h7["help_users"] or 0)
    hm = int(h7["help_users_in_dim"] or 0)
    if hm == 0:
        push("H7", "not_testable", f"No overlap between mari_help and dim (help_users={hu})")
    else:
        push("H7", "inconclusive", f"matched={hm}", n=hm)

    # H8
    seg_cols_8 = ["estado", "currentstage", "primary_device", "utm_origin"]
    sig8 = 0
    best_eta = 0.0
    minp8 = None
    for col in seg_cols_8:
        if col not in teacher_df.columns:
            continue
        x = teacher_df[[col, "time_to_first_value_hours"]].dropna().copy()
        x[col] = x[col].astype(str)
        cnt = x[col].value_counts()
        valid = cnt[cnt >= min_segment_n].index
        x = x[x[col].isin(valid)]
        if x[col].nunique() < 2:
            continue
        groups = [g["time_to_first_value_hours"].to_numpy() for _, g in x.groupby(col)]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        n = len(x)
        k = len(groups)
        eta2 = float((stat - k + 1) / (n - k)) if n > k else 0.0
        medians = x.groupby(col)["time_to_first_value_hours"].median().sort_values(ascending=False)
        spread = float(medians.iloc[0] - medians.iloc[-1]) if len(medians) > 1 else 0.0
        if p < alpha and spread > 1.0:
            sig8 += 1
        best_eta = max(best_eta, max(0.0, eta2))
        minp8 = p if minp8 is None else min(minp8, p)
    push("H8", "validated" if sig8 > 0 else "inconclusive", f"significant_segments={sig8}", p=minp8, eff=best_eta, n=int(teacher_df["time_to_first_value_hours"].notna().sum()))

    # H9
    desc_tables = ["dim_teachers", "entries", "interactions"]
    found: List[str] = []
    for t in desc_tables:
        dsc = conn.execute(f"DESCRIBE {t}").fetchdf()
        for c in dsc["column_name"].astype(str):
            low = c.lower()
            if "onboard" in low or "experiment" in low or "variant" in low:
                found.append(f"{t}.{c}")
    if not found:
        push("H9", "not_testable", "No onboarding/experiment/variant columns found")
    else:
        push("H9", "inconclusive", f"found={found}")

    # H10
    fcols = [
        "session_count",
        "interaction_count",
        "aula_event_count",
        "prova_event_count",
        "plano_event_count",
        "download_event_count",
        "visualizacao_event_count",
        "time_to_first_value_hours",
        "formation_avg_progress",
        "mari_conv_count",
        "ia_event_count",
    ]
    fcols = [c for c in fcols if c in teacher_df.columns]
    if len(fcols) < 4:
        push("H10", "inconclusive", "Insufficient clustering features")
    else:
        x = teacher_df[fcols].copy().fillna(0.0)
        x = np.log1p(np.clip(x, a_min=0, a_max=None))
        if len(x) > max_cluster_sample:
            x_sample = x.sample(max_cluster_sample, random_state=random_seed)
        else:
            x_sample = x
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_sample)
        best_score = -1.0
        best_k = None
        for k in range(2, 7):
            model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
            labels = model.fit_predict(x_scaled)
            score = silhouette_score(x_scaled, labels)
            if score > best_score:
                best_score = float(score)
                best_k = k
        st = "validated" if best_score >= 0.2 else "inconclusive"
        push("H10", st, f"best_k={best_k}; silhouette={best_score:.3f}", eff=best_score, n=len(x_sample))

    # H11
    if "activity_tier" not in teacher_df.columns:
        push("H11", "inconclusive", "activity_tier missing")
    else:
        x = teacher_df[["activity_tier", "churn_label"]].dropna()
        tab = pd.crosstab(x["activity_tier"], x["churn_label"])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            push("H11", "inconclusive", "Insufficient variability")
        else:
            chi2, p, _, _ = chi2_contingency(tab.to_numpy())
            v = cramers_v_from_table(tab)
            st = "validated" if (p < alpha and v >= 0.05) else "inconclusive"
            push("H11", st, f"chi2={chi2:.2f}; p={p:.3g}; v={v:.3f}", p=p, eff=v, n=len(x))

    # H12
    req = {"ai_used_flag", "activity_tier", "churn_label"}
    if not req.issubset(teacher_df.columns):
        push("H12", "inconclusive", "Missing ai/profile/churn features")
    else:
        x = teacher_df[["ai_used_flag", "activity_tier", "churn_label"]].dropna()
        effects: List[Tuple[str, float, float]] = []
        for tier, grp in x.groupby("activity_tier"):
            tab = pd.crosstab(grp["ai_used_flag"], grp["churn_label"])
            if tab.shape != (2, 2) or len(grp) < min_segment_n:
                continue
            _, p, _, _ = chi2_contingency(tab.to_numpy())
            diff = float(grp.loc[grp["ai_used_flag"] == 1, "churn_label"].mean() - grp.loc[grp["ai_used_flag"] == 0, "churn_label"].mean())
            effects.append((str(tier), diff, float(p)))
        if not effects:
            push("H12", "inconclusive", "Insufficient per-profile sample")
        else:
            signs = set(np.sign([e[1] for e in effects if not np.isclose(e[1], 0.0)]).tolist())
            significant_any = any(p < alpha for _, _, p in effects)
            mixed_direction = (1.0 in signs) and (-1.0 in signs)
            max_abs = max(abs(d) for _, d, _ in effects)
            if significant_any and mixed_direction and max_abs >= 0.01:
                st = "validated"
            elif significant_any and not mixed_direction:
                st = "rejected"
            else:
                st = "inconclusive"
            push("H12", st, " | ".join([f"{t}:{d:.3f},p={p:.3g}" for t, d, p in effects]), p=min(p for _, _, p in effects), eff=max_abs, n=len(x))

    # H13
    req13 = {"visualizou_metodologia_ativa", "aula_event_count", "plano_event_count"}
    if not req13.issubset(teacher_df.columns):
        push("H13", "inconclusive", "Missing required columns")
    else:
        x = teacher_df[["visualizou_metodologia_ativa", "aula_event_count", "plano_event_count"]].copy()
        x["flag"] = (x["visualizou_metodologia_ativa"].fillna(0) > 0).astype(int)
        x["usage_metric"] = x["aula_event_count"].fillna(0) + x["plano_event_count"].fillna(0)
        g1 = x.loc[x["flag"] == 1, "usage_metric"]
        g0 = x.loc[x["flag"] == 0, "usage_metric"]
        if len(g1) < 30 or len(g0) < 30:
            push("H13", "inconclusive", "Insufficient group sizes", n=len(x))
        else:
            _, p = mannwhitneyu(g1, g0, alternative="two-sided")
            med = float(np.median(g1) - np.median(g0))
            if p < alpha and med > 0:
                st = "validated"
            elif p < alpha and med < 0:
                st = "rejected"
            else:
                st = "inconclusive"
            push("H13", st, f"median_diff={med:.3f}; p={p:.3g}", p=p, eff=med, n=len(x))

    # H14
    req14 = {"total_alunos", "interaction_count"}
    if not req14.issubset(teacher_df.columns):
        push("H14", "inconclusive", "Missing total_alunos or interaction_count")
    else:
        x = teacher_df[["total_alunos", "interaction_count"]].dropna()
        x = x[x["total_alunos"] >= 0]
        if len(x) < 100:
            push("H14", "inconclusive", "Insufficient observations", n=len(x))
        else:
            rho, p = spearmanr(x["total_alunos"], x["interaction_count"])
            st = "validated" if (p < alpha and abs(float(rho)) >= 0.1) else "inconclusive"
            push("H14", st, f"rho={rho:.3f}; p={p:.3g}", p=p, eff=rho, n=len(x))

    # H15
    rows = int(hotjar.get("rows", 0) or 0)
    rep = int(hotjar.get("feedback_repeat_users", 0) or 0)
    push("H15", "not_testable", f"Hotjar rows={rows}; repeat_hotjar_users={rep}; no identity bridge")

    return pd.DataFrame(results)


def compute_monthly_solution_usage(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH x AS (
            SELECT
                date_trunc('month', data_inicio) AS month,
                SUM(CASE WHEN lower(coalesce(event_type, '')) LIKE '%aula%' THEN 1 ELSE 0 END) AS aula_events,
                SUM(CASE WHEN lower(coalesce(event_type, '')) LIKE '%prova%' THEN 1 ELSE 0 END) AS prova_events
            FROM interactions
            GROUP BY 1
        )
        SELECT * FROM x WHERE month IS NOT NULL ORDER BY month
        """
    ).fetchdf()


def compare_truth_with_baseline(
    baseline_output_dir: Path,
    truth_summary: Dict[str, Any],
    truth_hyp: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    baseline_summary_path = baseline_output_dir / "reports" / "analise_inicial_dos_dados_summary.json"
    baseline_hyp_path = baseline_output_dir / "hypothesis_results.csv"

    if baseline_summary_path.exists():
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        for k, tv in truth_summary.items():
            if k not in baseline_summary:
                rows.append({"entity": "summary", "key": k, "baseline": None, "truth": tv, "status": "extra_truth_metric", "delta": None})
                continue
            bv = baseline_summary[k]
            if isinstance(tv, (int, float)) and isinstance(bv, (int, float)) and not (pd.isna(tv) and pd.isna(bv)):
                delta = abs(float(tv) - float(bv))
                status = "match" if delta <= 1e-9 else "mismatch"
            else:
                delta = None
                status = "match" if str(tv) == str(bv) else "mismatch"
            rows.append({"entity": "summary", "key": k, "baseline": bv, "truth": tv, "status": status, "delta": delta})

    if baseline_hyp_path.exists():
        bh = pd.read_csv(baseline_hyp_path)
        bh2 = bh[["hypothesis_id", "status"]].copy()
        th2 = truth_hyp[["hypothesis_id", "status"]].copy()
        m = bh2.merge(th2, on="hypothesis_id", how="outer", suffixes=("_baseline", "_truth"))
        for _, r in m.iterrows():
            b = r.get("status_baseline")
            t = r.get("status_truth")
            status = "match" if str(b) == str(t) else "mismatch"
            rows.append(
                {
                    "entity": "hypothesis",
                    "key": str(r.get("hypothesis_id")),
                    "baseline": b,
                    "truth": t,
                    "status": status,
                    "delta": None,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    default_base = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
    parser = argparse.ArgumentParser(description="Independent truth recomputation for non-survival analysis.")
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--baseline-output-dir", type=Path, default=None)
    parser.add_argument("--spec-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir or (base_dir / "base_aprendizap")).resolve()
    baseline_output_dir = (args.baseline_output_dir or (base_dir / "analysis_output")).resolve()
    spec_file = (args.spec_file or (base_dir / "verification" / "spec" / "non_survival_manifest.yaml")).resolve()
    _ = load_manifest_spec(spec_file)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    consolidated_path = baseline_output_dir / "consolidated_status.json"
    consolidated = json.loads(consolidated_path.read_text(encoding="utf-8")) if consolidated_path.exists() else {}
    cfg = consolidated.get("run_metadata", {}).get("config", {})

    alpha = float(cfg.get("alpha", 0.05))
    min_segment_n = int(cfg.get("min_segment_n", 200))
    random_seed = int(cfg.get("random_seed", 42))
    churn_days = int(cfg.get("churn_days", 30))
    conversion_days = int(cfg.get("conversion_days", 30))

    conn = duckdb.connect()
    build_views(conn, data_dir)

    teacher_df = build_teacher_dataset(conn, churn_days=churn_days, conversion_days=conversion_days)
    monthly_df = compute_monthly_solution_usage(conn)
    hotjar = compute_hotjar_summary(data_dir)

    summary_truth = compute_summary_non_survival(conn, teacher_df)
    write_json(out_dir / "truth_summary_non_survival.json", summary_truth)

    core = compute_core_metrics(conn, teacher_df)
    core_json = {
        "max_geo_effect_size": core["max_geo_effect_size"],
        "journey_path_counts": core["journey_path_counts"],
    }
    write_json(out_dir / "truth_core_metrics.json", core_json)

    core["state_stats"].to_csv(out_dir / "truth_state_stats.csv", index=False)
    core["utm_stats"].to_csv(out_dir / "truth_utm_stats.csv", index=False)
    core["geo_associations"].to_csv(out_dir / "truth_geo_associations.csv", index=False)
    core["top_corr_pairs"].to_csv(out_dir / "truth_top_corr_pairs.csv", index=False)
    core["cat_corr_pairs"].to_csv(out_dir / "truth_cat_corr_pairs.csv", index=False)

    hyp = h1_h15(
        teacher_df=teacher_df,
        conn=conn,
        monthly_df=monthly_df,
        hotjar=hotjar,
        alpha=alpha,
        min_segment_n=min_segment_n,
        random_seed=random_seed,
    )
    hyp.to_csv(out_dir / "truth_hypothesis_results.csv", index=False)

    diff = compare_truth_with_baseline(
        baseline_output_dir=baseline_output_dir,
        truth_summary=summary_truth,
        truth_hyp=hyp,
    )
    diff.to_csv(out_dir / "truth_vs_baseline_diff.csv", index=False)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "data_dir": str(data_dir),
        "baseline_output_dir": str(baseline_output_dir),
        "teacher_rows": int(len(teacher_df)),
        "hypotheses_total": int(len(hyp)),
        "summary_mismatches": int(((diff["entity"] == "summary") & (diff["status"] == "mismatch")).sum()) if not diff.empty else 0,
        "hypothesis_status_mismatches": int(((diff["entity"] == "hypothesis") & (diff["status"] != "match")).sum()) if not diff.empty else 0,
    }
    write_json(out_dir / "truth_recompute_summary.json", summary)
    print(str(out_dir / "truth_recompute_summary.json"))


if __name__ == "__main__":
    main()
