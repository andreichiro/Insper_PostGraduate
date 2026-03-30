#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import numpy as np
import pandas as pd

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    STRICT_VALUE_EVENTS,
    V2Config,
    build_config,
    classify_currentsubject_group,
    classify_discipline_group,
    connect_duckdb,
    ensure_output_dirs,
    month_diff,
    normalize_utm,
    parquet_only_mode,
    q,
    register_raw_views,
    setup_logging,
    sql_device_expr,
    sql_event_action_expr,
    sql_event_family_expr,
    sql_id_aula_semantic_expr,
    utc_now_iso,
    write_json,
    write_markdown,
)


FULL_CSV_TABLES = {
    "audit_base_modelada_validation",
    "base_modelada_v2",
    "dim_teacher",
    "dim_lesson",
    "dim_event",
    "dim_device",
    "dim_calendar",
    "bridge_mari_conversation_teacher",
    "fct_session_clean",
    "fct_interaction_clean",
    "fct_teacher_month",
    "fct_mari_help_resolved",
}
SAMPLE_ONLY_TABLES = {
    "bridge_teacher_identity_audit",
    "fct_session_raw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 02 v2: modelagem estrela e fatos limpos.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def persist_table(
    conn: duckdb.DuckDBPyConnection,
    cfg: V2Config,
    table_name: str,
    df: pd.DataFrame | None = None,
    sample_rows: int = 5_000,
) -> Dict[str, str]:
    csv_dir = cfg.output_dir / "csv"
    parquet_dir = cfg.output_dir / "parquet"
    csv_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    if df is not None:
        conn.register("_persist_df", df)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _persist_df")

    parquet_path = parquet_dir / f"{table_name}.parquet"
    conn.execute(f"COPY {table_name} TO '{q(parquet_path)}' (FORMAT PARQUET)")

    written = {"parquet": str(parquet_path)}
    if parquet_only_mode():
        return written
    if table_name in FULL_CSV_TABLES:
        csv_path = csv_dir / f"{table_name}.csv"
        conn.execute(f"COPY {table_name} TO '{q(csv_path)}' (HEADER, DELIMITER ',')")
        written["csv"] = str(csv_path)
    elif table_name in SAMPLE_ONLY_TABLES:
        sample_path = csv_dir / f"{table_name}_sample.csv"
        conn.execute(f"COPY (SELECT * FROM {table_name} LIMIT {int(sample_rows)}) TO '{q(sample_path)}' (HEADER, DELIMITER ',')")
        written["csv_sample"] = str(sample_path)
    return written


def create_bridge_mari_conversation_teacher(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH union_src AS (
          SELECT id_mari, unique_id_aprendizap, 'mari_reports' AS source_table
          FROM raw_mari_reports
          WHERE unique_id_aprendizap IS NOT NULL
          UNION ALL
          SELECT id_mari, unique_id_aprendizap, 'mari_conv' AS source_table
          FROM raw_mari_conv
          WHERE unique_id_aprendizap IS NOT NULL
        ),
        agg AS (
          SELECT
            id_mari,
            COUNT(DISTINCT unique_id_aprendizap) AS teacher_resolution_count,
            MIN(unique_id_aprendizap) AS resolved_teacher_candidate,
            string_agg(DISTINCT unique_id_aprendizap, ' | ' ORDER BY unique_id_aprendizap) AS teacher_candidates,
            COUNT(*) FILTER (WHERE source_table='mari_reports') AS report_rows,
            COUNT(*) FILTER (WHERE source_table='mari_conv') AS conv_rows
          FROM union_src
          GROUP BY 1
        )
        SELECT
          id_mari,
          CASE WHEN teacher_resolution_count=1 THEN resolved_teacher_candidate END AS teacher_unique_id,
          teacher_resolution_count,
          CASE WHEN teacher_resolution_count=1 THEN 1 ELSE 0 END AS is_unambiguous,
          CASE
            WHEN report_rows > 0 AND conv_rows > 0 THEN 'reports_and_conv'
            WHEN report_rows > 0 THEN 'reports_only'
            WHEN conv_rows > 0 THEN 'conv_only'
            ELSE 'none'
          END AS resolution_source,
          teacher_candidates,
          report_rows,
          conv_rows
        FROM agg
        ORDER BY id_mari
        """
    ).fetchdf()


def create_dim_teacher(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute(
        """
        WITH coverage AS (
          SELECT
            d.unique_id,
            CASE WHEN e.unique_id IS NOT NULL THEN 1 ELSE 0 END AS has_registered_entry,
            CASE WHEN i.unique_id IS NOT NULL THEN 1 ELSE 0 END AS has_registered_interaction,
            CASE WHEN f.unique_id_aprendizap IS NOT NULL THEN 1 ELSE 0 END AS has_formation,
            CASE WHEN mc.unique_id_aprendizap IS NOT NULL THEN 1 ELSE 0 END AS has_mari_conv,
            CASE WHEN mr.unique_id_aprendizap IS NOT NULL THEN 1 ELSE 0 END AS has_mari_reports
          FROM raw_dim_teachers d
          LEFT JOIN (SELECT DISTINCT unique_id FROM raw_entries WHERE lower(coalesce(user_type,''))='registered') e ON d.unique_id=e.unique_id
          LEFT JOIN (SELECT DISTINCT unique_id FROM raw_interactions WHERE lower(coalesce(user_type,''))='registered') i ON d.unique_id=i.unique_id
          LEFT JOIN (SELECT DISTINCT unique_id_aprendizap FROM raw_formation) f ON d.unique_id=f.unique_id_aprendizap
          LEFT JOIN (SELECT DISTINCT unique_id_aprendizap FROM raw_mari_conv) mc ON d.unique_id=mc.unique_id_aprendizap
          LEFT JOIN (SELECT DISTINCT unique_id_aprendizap FROM raw_mari_reports) mr ON d.unique_id=mr.unique_id_aprendizap
        )
        SELECT
          d.unique_id AS teacher_unique_id,
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
          d.selectedsubjectsem,
          d.selectedsubjectsfundii,
          d.visualizou_metodologia_ativa,
          d.data_entrada,
          c.has_registered_entry,
          c.has_registered_interaction,
          c.has_formation,
          c.has_mari_conv,
          c.has_mari_reports,
          CASE WHEN d.estado IS NULL OR trim(d.estado)='' THEN 1 ELSE 0 END AS is_estado_missing,
          CASE WHEN d.utm_origin IS NULL OR trim(d.utm_origin)='' THEN 1 ELSE 0 END AS is_utm_missing,
          CASE WHEN d.total_alunos IS NULL THEN 1 ELSE 0 END AS is_total_alunos_missing,
          CASE WHEN d.total_alunos < 0 THEN 1 ELSE 0 END AS is_total_alunos_negative,
          CASE WHEN d.alunos_diretos < 0 THEN 1 ELSE 0 END AS is_alunos_diretos_negative,
          CASE WHEN d.alunos_indiretos < 0 THEN 1 ELSE 0 END AS is_alunos_indiretos_negative,
          CASE WHEN d.login_google IS NOT NULL AND d.login_google NOT IN (0,1) THEN 1 ELSE 0 END AS is_login_google_invalid
        FROM raw_dim_teachers d
        LEFT JOIN coverage c ON d.unique_id=c.unique_id
        ORDER BY d.unique_id
        """
    ).fetchdf()
    df["utm_group"] = df["utm_origin"].apply(normalize_utm)
    df["currentsubject_group"] = [
        classify_currentsubject_group(stage, subject)
        for stage, subject in zip(df["currentstage"], df["currentsubject"])
    ]
    df["population_status"] = np.where(
        df["has_registered_interaction"].fillna(0).astype(int) == 1,
        "teacher_with_registered_activity",
        "teacher_without_registered_activity",
    )
    return df


def create_dim_lesson(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute("SELECT * FROM raw_lessons ORDER BY id_aula").fetchdf()
    df = df.rename(columns={"id_aula": "lesson_id"})
    df["discipline_group"] = df["disciplina"].apply(classify_discipline_group)
    df["lesson_id_semantic"] = "lesson_like_22char"
    df["is_active_methodology_missing"] = df["possui_metodologia_ativa"].isna().astype(int)
    return df


def create_dim_event(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        f"""
        WITH base AS (
          SELECT
            coalesce(event_type, '<missing>') AS event_type,
            lower(coalesce(event_type, '')) AS event_type_lower,
            {sql_event_family_expr('event_type')} AS event_family,
            {sql_event_action_expr('event_type')} AS event_action,
            COUNT(*) AS rows_total
          FROM raw_interactions
          GROUP BY 1, 2, 3, 4
        )
        SELECT
          event_type,
          event_family,
          event_action,
          rows_total,
          CASE WHEN event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END AS is_strict_value_event,
          CASE WHEN event_action='download' THEN 1 ELSE 0 END AS is_download_event,
          CASE WHEN event_action='view' THEN 1 ELSE 0 END AS is_visualization_event,
          CASE WHEN event_action='navigation' THEN 1 ELSE 0 END AS is_navigation_event,
          CASE WHEN event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0 ELSE 1 END AS is_activity_event
        FROM base
        ORDER BY rows_total DESC, event_type
        """
    ).fetchdf()


def create_dim_device() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"device_group": "desktop", "description": "Desktop reconhecido pelo raw."},
            {"device_group": "mobile", "description": "Mobile reconhecido pelo raw."},
            {"device_group": "tablet", "description": "Tablet reconhecido pelo raw."},
            {"device_group": "unknown", "description": "Device ausente ou não padronizado."},
        ]
    )


def create_dim_calendar(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute("SELECT * FROM raw_school_calendar ORDER BY month_start, uf, rede").fetchdf()
    df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")
    df["school_phase"] = np.select(
        [
            df["month"].isin([1, 7, 12]),
            df["month"].isin([2, 3, 4, 5, 6]),
            df["month"].isin([8, 9, 10, 11]),
        ],
        [
            "ferias_ou_transicao",
            "periodo_letivo_semestre_1",
            "periodo_letivo_semestre_2",
        ],
        default="outro",
    )
    return df


def create_bridge_teacher_identity_audit(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE bridge_teacher_identity_audit AS
        WITH entries_base AS (
          SELECT unique_id AS source_key, string_agg(DISTINCT lower(coalesce(user_type,'missing')), '|') AS source_user_types
          FROM raw_entries
          GROUP BY 1
        ),
        interactions_base AS (
          SELECT unique_id AS source_key, string_agg(DISTINCT lower(coalesce(user_type,'missing')), '|') AS source_user_types
          FROM raw_interactions
          GROUP BY 1
        ),
        formation_base AS (
          SELECT DISTINCT unique_id_aprendizap AS source_key FROM raw_formation
        ),
        mari_conv_base AS (
          SELECT DISTINCT unique_id_aprendizap AS source_key FROM raw_mari_conv WHERE unique_id_aprendizap IS NOT NULL
        ),
        mari_reports_base AS (
          SELECT DISTINCT unique_id_aprendizap AS source_key FROM raw_mari_reports WHERE unique_id_aprendizap IS NOT NULL
        ),
        mari_help_bridge AS (
          SELECT
            h.user_id AS source_key,
            COUNT(DISTINCT b.teacher_unique_id) FILTER (WHERE b.is_unambiguous=1 AND b.teacher_unique_id IS NOT NULL) AS resolved_teacher_count,
            MIN(b.teacher_unique_id) FILTER (WHERE b.is_unambiguous=1 AND b.teacher_unique_id IS NOT NULL) AS resolved_teacher_unique_id
          FROM raw_mari_help h
          LEFT JOIN bridge_mari_conversation_teacher b ON h.user_id=b.id_mari
          GROUP BY 1
        )
        SELECT
          'raw_entries' AS source_table,
          'unique_id' AS source_key_name,
          e.source_key,
          e.source_user_types,
          'exact_unique_id' AS resolution_path,
          CASE WHEN d.unique_id IS NOT NULL THEN d.unique_id END AS resolved_teacher_unique_id,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS resolved_teacher_count,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS is_unambiguous,
          'uuid36' AS source_key_domain
        FROM entries_base e
        LEFT JOIN raw_dim_teachers d ON e.source_key=d.unique_id
        UNION ALL
        SELECT
          'raw_interactions',
          'unique_id',
          i.source_key,
          i.source_user_types,
          'exact_unique_id',
          CASE WHEN d.unique_id IS NOT NULL THEN d.unique_id END AS resolved_teacher_unique_id,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS resolved_teacher_count,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS is_unambiguous,
          'uuid36' AS source_key_domain
        FROM interactions_base i
        LEFT JOIN raw_dim_teachers d ON i.source_key=d.unique_id
        UNION ALL
        SELECT
          'raw_formation',
          'unique_id_aprendizap',
          f.source_key,
          NULL,
          'exact_same_domain',
          CASE WHEN d.unique_id IS NOT NULL THEN d.unique_id END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          'uuid36'
        FROM formation_base f
        LEFT JOIN raw_dim_teachers d ON f.source_key=d.unique_id
        UNION ALL
        SELECT
          'raw_mari_conv',
          'unique_id_aprendizap',
          m.source_key,
          NULL,
          'exact_same_domain',
          CASE WHEN d.unique_id IS NOT NULL THEN d.unique_id END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          'uuid36'
        FROM mari_conv_base m
        LEFT JOIN raw_dim_teachers d ON m.source_key=d.unique_id
        UNION ALL
        SELECT
          'raw_mari_reports',
          'unique_id_aprendizap',
          m.source_key,
          NULL,
          'exact_same_domain',
          CASE WHEN d.unique_id IS NOT NULL THEN d.unique_id END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          'uuid36'
        FROM mari_reports_base m
        LEFT JOIN raw_dim_teachers d ON m.source_key=d.unique_id
        UNION ALL
        SELECT
          'raw_mari_help',
          'user_id',
          h.source_key,
          NULL,
          'semantic_mari_bridge',
          h.resolved_teacher_unique_id,
          h.resolved_teacher_count,
          CASE WHEN h.resolved_teacher_count=1 THEN 1 ELSE 0 END,
          'hex64_upper'
        FROM mari_help_bridge h
        """
    )


def create_fct_session_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_session_raw AS
        SELECT
          CAST(hash(e.unique_id, e.user_type, e.data_inicio, e.data_fim) AS UBIGINT) AS session_row_hash,
          e.unique_id AS source_unique_id,
          d.unique_id AS teacher_unique_id,
          lower(coalesce(e.user_type,'missing')) AS user_type,
          date_trunc('month', e.data_inicio) AS session_month,
          e.data_inicio AS session_start_ts,
          e.data_fim AS session_end_ts,
          CASE WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL THEN GREATEST(epoch(e.data_fim) - epoch(e.data_inicio), 0) END AS duration_sec,
          CASE WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL AND e.data_fim < e.data_inicio THEN 1 ELSE 0 END AS is_negative_duration,
          CASE WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL AND GREATEST(epoch(e.data_fim) - epoch(e.data_inicio), 0) <= 1 THEN 1 ELSE 0 END AS is_ping_session_le_1s,
          CASE WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL AND GREATEST(epoch(e.data_fim) - epoch(e.data_inicio), 0) <= 5 THEN 1 ELSE 0 END AS is_ping_session_le_5s,
          CASE WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL AND GREATEST(epoch(e.data_fim) - epoch(e.data_inicio), 0) <= 10 THEN 1 ELSE 0 END AS is_ping_session_le_10s,
          CASE
            WHEN lower(coalesce(e.user_type,''))='registered' AND d.unique_id IS NOT NULL THEN 'core_teacher'
            WHEN lower(coalesce(e.user_type,''))='registered' AND d.unique_id IS NULL THEN 'shadow_registered'
            WHEN lower(coalesce(e.user_type,''))='anonymous' THEN 'shadow_anonymous'
            WHEN lower(coalesce(e.user_type,''))='seo' THEN 'shadow_seo'
            ELSE 'other'
          END AS population_bucket,
          CASE WHEN lower(coalesce(e.user_type,''))='registered' AND d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS is_core_teacher_session
        FROM raw_entries e
        LEFT JOIN raw_dim_teachers d USING(unique_id)
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_session_clean AS
        SELECT
          *,
          duration_sec / 60.0 AS duration_min
        FROM fct_session_raw
        WHERE is_core_teacher_session=1
          AND is_negative_duration=0
          AND is_ping_session_le_5s=0
        """
    )


def create_fct_interaction_clean(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE fct_interaction_clean AS
        WITH base AS (
          SELECT
            CAST(hash(i.unique_id, i.data_inicio, i.event_type, i.content_type, i.id_aula, i.utm_source) AS UBIGINT) AS interaction_row_hash,
            i.unique_id AS source_unique_id,
            d.unique_id AS teacher_unique_id,
            lower(coalesce(i.user_type,'missing')) AS user_type,
            date_trunc('month', i.data_inicio) AS interaction_month,
            i.data_inicio AS interaction_ts,
            coalesce(i.event_type, '<missing>') AS event_type,
            lower(coalesce(i.event_type, '')) AS event_type_lower,
            coalesce(i.content_type, '<missing>') AS content_type,
            coalesce(i.utm_source, '<missing>') AS utm_source,
            i.id_aula,
            {sql_device_expr('i.user_agent_device_type')} AS device_group,
            {sql_event_family_expr('i.event_type')} AS event_family,
            {sql_event_action_expr('i.event_type')} AS event_action,
            {sql_id_aula_semantic_expr('i.id_aula')} AS id_aula_semantic
          FROM raw_interactions i
          LEFT JOIN raw_dim_teachers d USING(unique_id)
          WHERE lower(coalesce(i.user_type,''))='registered'
            AND d.unique_id IS NOT NULL
            AND i.data_inicio IS NOT NULL
        )
        SELECT
          b.*,
          CASE WHEN b.event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END AS is_strict_value_event,
          CASE WHEN b.event_action='download' THEN 1 ELSE 0 END AS is_download_event,
          CASE WHEN b.event_action='view' THEN 1 ELSE 0 END AS is_visualization_event,
          CASE WHEN b.event_action='navigation' THEN 1 ELSE 0 END AS is_navigation_event,
          CASE WHEN b.event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0 ELSE 1 END AS is_activity_event,
          CASE WHEN b.event_action='view' AND b.event_family IN ('aula', 'plano', 'prova') THEN 1 ELSE 0 END AS is_content_view_event,
          CASE
            WHEN b.event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0
            WHEN b.event_action='download' THEN 0
            WHEN b.event_action='view' AND b.event_family IN ('aula', 'plano', 'prova') THEN 0
            ELSE 1
          END AS is_other_activity_non_download_event,
          CASE WHEN b.id_aula_semantic='lesson_like_22char' THEN 1 ELSE 0 END AS lesson_join_allowed,
          CASE WHEN l.id_aula IS NOT NULL THEN 1 ELSE 0 END AS lesson_mapped_flag,
          l.id_aula AS lesson_id,
          l.disciplina,
          l.nivel,
          l.ano,
          l.ano_em,
          l.unidade,
          l.bncc,
          l.possui_metodologia_ativa,
          l.total_metodologias_ativa
        FROM base b
        LEFT JOIN raw_lessons l
          ON b.id_aula=l.id_aula
         AND b.id_aula_semantic='lesson_like_22char'
        """
    )


def create_fct_mari_help_resolved(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_help_resolved AS
        SELECT
          h.user_id AS id_mari,
          b.teacher_unique_id,
          b.teacher_resolution_count,
          b.is_unambiguous,
          b.resolution_source,
          h.date AS help_ts,
          date_trunc('month', h.date) AS help_month,
          h.turno,
          h.key,
          h.isso_ajudou,
          h.isso_ajudou_num
        FROM raw_mari_help h
        INNER JOIN bridge_mari_conversation_teacher b
          ON h.user_id=b.id_mari
        WHERE b.is_unambiguous=1
          AND b.teacher_unique_id IS NOT NULL
        """
    )


def build_teacher_month(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    interactions_month = conn.execute(
        """
        SELECT
          teacher_unique_id,
          interaction_month AS month,
          COUNT(*) AS interaction_rows_month,
          SUM(is_activity_event) AS activity_events_month,
          COUNT(DISTINCT CAST(interaction_ts AS DATE)) FILTER (WHERE is_activity_event=1) AS active_days_month,
          SUM(CASE WHEN event_family='aula' THEN 1 ELSE 0 END) AS aula_events_month,
          SUM(CASE WHEN event_family='plano' THEN 1 ELSE 0 END) AS plano_events_month,
          SUM(CASE WHEN event_family='prova' THEN 1 ELSE 0 END) AS prova_events_month,
          SUM(CASE WHEN event_family='ia' THEN 1 ELSE 0 END) AS ia_events_month,
          SUM(is_download_event) AS download_count_month,
          SUM(CASE WHEN event_type='download_aula' THEN 1 ELSE 0 END) AS download_aula_count_month,
          SUM(CASE WHEN event_type='download_plano_aula' THEN 1 ELSE 0 END) AS download_plano_count_month,
          SUM(CASE WHEN event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END) AS strict_download_count_month,
          SUM(is_content_view_event) AS content_views_month,
          SUM(is_other_activity_non_download_event) AS other_activity_non_download_events_month,
          COUNT(DISTINCT lesson_id) FILTER (WHERE lesson_mapped_flag=1) AS mapped_lessons_month,
          MAX(is_strict_value_event) AS strict_value_flag,
          MAX(CASE WHEN is_activity_event=1 THEN 1 ELSE 0 END) AS active_user_flag,
          MAX(CASE WHEN event_family='aula' AND is_content_view_event=1 THEN 1 ELSE 0 END) AS viewed_aula_flag,
          MAX(CASE WHEN event_family='plano' AND is_content_view_event=1 THEN 1 ELSE 0 END) AS viewed_plano_flag,
          MAX(CASE WHEN event_family='prova' AND is_content_view_event=1 THEN 1 ELSE 0 END) AS viewed_prova_flag,
          MAX(CASE WHEN event_family='ia' AND is_activity_event=1 THEN 1 ELSE 0 END) AS used_ia_flag,
          MAX(CASE WHEN device_group='desktop' THEN 1 ELSE 0 END) AS used_desktop_flag,
          MAX(CASE WHEN device_group='mobile' THEN 1 ELSE 0 END) AS used_mobile_flag,
          MAX(interaction_ts) AS last_interaction_ts_month
        FROM fct_interaction_clean
        GROUP BY 1, 2
        """
    ).fetchdf()

    sessions_month = conn.execute(
        """
        SELECT
          teacher_unique_id,
          session_month AS month,
          COUNT(*) AS session_count_month,
          SUM(duration_sec) / 60.0 AS total_session_minutes_month,
          AVG(duration_sec) / 60.0 AS avg_session_minutes_month,
          MAX(session_end_ts) AS last_session_ts_month
        FROM fct_session_clean
        GROUP BY 1, 2
        """
    ).fetchdf()

    if interactions_month.empty and sessions_month.empty:
        return pd.DataFrame()

    for df in [interactions_month, sessions_month]:
        if not df.empty:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")

    month_df = sessions_month.merge(interactions_month, on=["teacher_unique_id", "month"], how="outer")
    numeric_fill_zero = [
        "session_count_month",
        "total_session_minutes_month",
        "avg_session_minutes_month",
        "interaction_rows_month",
        "activity_events_month",
        "active_days_month",
        "aula_events_month",
        "plano_events_month",
        "prova_events_month",
        "ia_events_month",
        "download_count_month",
        "download_aula_count_month",
        "download_plano_count_month",
        "strict_download_count_month",
        "content_views_month",
        "other_activity_non_download_events_month",
        "mapped_lessons_month",
        "strict_value_flag",
        "active_user_flag",
        "viewed_aula_flag",
        "viewed_plano_flag",
        "viewed_prova_flag",
        "used_ia_flag",
        "used_desktop_flag",
        "used_mobile_flag",
    ]
    for col in numeric_fill_zero:
        if col in month_df.columns:
            month_df[col] = pd.to_numeric(month_df[col], errors="coerce").fillna(0)

    month_df["month"] = pd.to_datetime(month_df["month"], errors="coerce")
    month_df = month_df[month_df["month"].notna()].copy()
    month_df = month_df.sort_values(["teacher_unique_id", "month"]).reset_index(drop=True)
    month_df["no_download_flag"] = (month_df["strict_download_count_month"] <= 0).astype(int)
    month_df["no_download_view_only_flag"] = (
        (month_df["no_download_flag"] == 1)
        & (month_df["content_views_month"] > 0)
        & (month_df["other_activity_non_download_events_month"] <= 0)
    ).astype(int)
    month_df["no_download_view_plus_action_flag"] = (
        (month_df["no_download_flag"] == 1)
        & (month_df["content_views_month"] > 0)
        & (month_df["other_activity_non_download_events_month"] > 0)
    ).astype(int)
    month_df["no_download_action_only_flag"] = (
        (month_df["no_download_flag"] == 1)
        & (month_df["content_views_month"] <= 0)
        & (month_df["other_activity_non_download_events_month"] > 0)
    ).astype(int)
    month_df["session_exposed_no_download_flag"] = (
        (month_df["session_count_month"] > 0) & (month_df["strict_download_count_month"] <= 0)
    ).astype(int)
    month_df["session_exposed_no_activity_no_download_flag"] = (
        (month_df["session_count_month"] > 0)
        & (month_df["activity_events_month"] <= 0)
        & (month_df["strict_download_count_month"] <= 0)
    ).astype(int)
    month_df["session_exposed_activity_no_download_flag"] = (
        (month_df["session_count_month"] > 0)
        & (month_df["activity_events_month"] > 0)
        & (month_df["strict_download_count_month"] <= 0)
    ).astype(int)
    month_df["month_num"] = month_df["month"].dt.year * 12 + month_df["month"].dt.month
    max_month = month_df["month"].max()
    max_month_num = int(max_month.year * 12 + max_month.month)
    month_df["next_month"] = month_df["month"] + pd.offsets.MonthBegin(1)

    next_cols = month_df[
        [
            "teacher_unique_id",
            "month",
            "active_user_flag",
            "strict_value_flag",
            "strict_download_count_month",
        ]
    ].copy()
    next_cols = next_cols.rename(
        columns={
            "month": "next_month",
            "active_user_flag": "next_month_active_user_flag",
            "strict_value_flag": "next_month_strict_value_flag",
            "strict_download_count_month": "next_month_strict_download_count",
        }
    )
    month_df = month_df.merge(next_cols, on=["teacher_unique_id", "next_month"], how="left")
    month_df["next_month_observed_flag"] = (month_df["month_num"] < max_month_num).astype(int)

    for col in ["next_month_active_user_flag", "next_month_strict_value_flag", "next_month_strict_download_count"]:
        month_df[col] = pd.to_numeric(month_df[col], errors="coerce")

    month_df["returned_active_m1"] = np.where(
        month_df["next_month_observed_flag"] == 1,
        month_df["next_month_active_user_flag"].fillna(0),
        np.nan,
    )
    month_df["returned_strict_value_m1"] = np.where(
        month_df["next_month_observed_flag"] == 1,
        month_df["next_month_strict_value_flag"].fillna(0),
        np.nan,
    )
    month_df["returned_any_download_m1"] = np.where(
        month_df["next_month_observed_flag"] == 1,
        (month_df["next_month_strict_download_count"].fillna(0) > 0).astype(float),
        np.nan,
    )
    month_df["strict_user_flag"] = np.where(
        month_df["strict_value_flag"] == 1,
        np.where(month_df["next_month_observed_flag"] == 1, month_df["returned_active_m1"], np.nan),
        0,
    )
    month_df["strict_return_value_m1"] = np.where(
        month_df["strict_value_flag"] == 1,
        np.where(month_df["next_month_observed_flag"] == 1, month_df["returned_strict_value_m1"], np.nan),
        0,
    )
    month_df["lifetime_active_months"] = month_df.groupby("teacher_unique_id")["active_user_flag"].cumsum()
    month_df["lifetime_active_minutes_total"] = month_df.groupby("teacher_unique_id")["total_session_minutes_month"].cumsum()

    def add_streaks(group: pd.DataFrame, flag_col: str, current_col: str, max_col: str) -> pd.DataFrame:
        current_values: List[int] = []
        max_values: List[int] = []
        current = 0
        running_max = 0
        prev_month: pd.Timestamp | None = None
        prev_flag = 0
        for _, row in group.iterrows():
            flag = int(row[flag_col])
            month = pd.Timestamp(row["month"])
            if flag == 1:
                if prev_month is not None and month_diff(month, prev_month) == 1 and prev_flag == 1:
                    current += 1
                else:
                    current = 1
            else:
                current = 0
            running_max = max(running_max, current)
            current_values.append(current)
            max_values.append(running_max)
            prev_month = month
            prev_flag = flag
        out = group.copy()
        out[current_col] = current_values
        out[max_col] = max_values
        return out

    active_streak_frames = [
        add_streaks(frame.sort_values("month"), "active_user_flag", "active_streak_current_months", "active_streak_max_months")
        for _, frame in month_df.groupby("teacher_unique_id", sort=False)
    ]
    month_df = pd.concat(active_streak_frames, ignore_index=True) if active_streak_frames else month_df
    strict_streak_frames = [
        add_streaks(frame.sort_values("month"), "strict_value_flag", "strict_streak_current_months", "strict_streak_max_months")
        for _, frame in month_df.groupby("teacher_unique_id", sort=False)
    ]
    month_df = pd.concat(strict_streak_frames, ignore_index=True) if strict_streak_frames else month_df
    return month_df


def create_base_modelada(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE base_modelada_v2 AS
        SELECT
          tm.teacher_unique_id,
          tm.month,
          'teacher_month' AS base_grain,
          'core_teacher_month' AS analysis_population,
          dt.population_status AS teacher_population_status,
          dt.utm_origin AS teacher_utm_origin,
          dt.utm_group AS teacher_utm_group,
          dt.tela_origem AS teacher_tela_origem,
          dt.estado AS teacher_estado,
          dt.total_alunos AS teacher_total_alunos,
          dt.tipo_total_alunos AS teacher_tipo_total_alunos,
          dt.alunos_diretos AS teacher_alunos_diretos,
          dt.alunos_indiretos AS teacher_alunos_indiretos,
          dt.login_google AS teacher_login_google,
          dt.currentstage AS teacher_currentstage,
          dt.currentsubject AS teacher_currentsubject,
          dt.currentsubject_group AS teacher_currentsubject_group,
          dt.selectedstages AS teacher_selectedstages,
          dt.selectedsubjectsem AS teacher_selectedsubjectsem,
          dt.selectedsubjectsfundii AS teacher_selectedsubjectsfundii,
          dt.visualizou_metodologia_ativa AS teacher_visualizou_metodologia_ativa,
          dt.data_entrada AS teacher_data_entrada,
          dt.has_registered_entry,
          dt.has_registered_interaction,
          dt.has_formation,
          dt.has_mari_conv,
          dt.has_mari_reports,
          dt.is_estado_missing,
          dt.is_utm_missing,
          dt.is_total_alunos_missing,
          dt.is_total_alunos_negative,
          dt.is_alunos_diretos_negative,
          dt.is_alunos_indiretos_negative,
          dt.is_login_google_invalid,
          tm.session_count_month,
          tm.total_session_minutes_month,
          tm.avg_session_minutes_month,
          tm.interaction_rows_month,
          tm.activity_events_month,
          tm.active_days_month,
          tm.aula_events_month,
          tm.plano_events_month,
          tm.prova_events_month,
          tm.ia_events_month,
          tm.download_count_month,
          tm.download_aula_count_month,
          tm.download_plano_count_month,
          tm.strict_download_count_month,
          tm.content_views_month,
          tm.other_activity_non_download_events_month,
          tm.mapped_lessons_month,
          tm.strict_value_flag,
          tm.active_user_flag,
          tm.viewed_aula_flag,
          tm.viewed_plano_flag,
          tm.viewed_prova_flag,
          tm.used_ia_flag,
          tm.used_desktop_flag,
          tm.used_mobile_flag,
          tm.last_interaction_ts_month,
          tm.last_session_ts_month,
          tm.no_download_flag,
          tm.no_download_view_only_flag,
          tm.no_download_view_plus_action_flag,
          tm.no_download_action_only_flag,
          tm.session_exposed_no_download_flag,
          tm.session_exposed_no_activity_no_download_flag,
          tm.session_exposed_activity_no_download_flag,
          tm.month_num,
          tm.next_month,
          tm.next_month_active_user_flag,
          tm.next_month_strict_value_flag,
          tm.next_month_strict_download_count,
          tm.next_month_observed_flag,
          tm.returned_active_m1,
          tm.returned_strict_value_m1,
          tm.returned_any_download_m1,
          tm.strict_user_flag,
          tm.strict_return_value_m1,
          tm.lifetime_active_months,
          tm.lifetime_active_minutes_total,
          tm.active_streak_current_months,
          tm.active_streak_max_months,
          tm.strict_streak_current_months,
          tm.strict_streak_max_months
        FROM fct_teacher_month tm
        INNER JOIN dim_teacher dt USING(teacher_unique_id)
        WHERE tm.month IS NOT NULL
        ORDER BY tm.teacher_unique_id, tm.month
        """
    )


def build_base_modelada_validation(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add_check(check_name: str, metric_value: Any, status: str, note: str) -> None:
        rows.append(
            {
                "check_name": check_name,
                "metric_value": metric_value,
                "status": status,
                "note": note,
            }
        )

    base_rows = int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2").fetchone()[0] or 0)
    fact_rows = int(conn.execute("SELECT COUNT(*) FROM fct_teacher_month WHERE month IS NOT NULL").fetchone()[0] or 0)
    grain_duplicates = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT teacher_unique_id, month, COUNT(*) AS dup_count
              FROM base_modelada_v2
              GROUP BY 1, 2
              HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        or 0
    )
    missing_teacher = int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2 WHERE teacher_unique_id IS NULL").fetchone()[0] or 0)
    missing_month = int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2 WHERE month IS NULL").fetchone()[0] or 0)
    join_gap = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM fct_teacher_month tm
            LEFT JOIN dim_teacher dt USING(teacher_unique_id)
            WHERE tm.month IS NOT NULL
              AND dt.teacher_unique_id IS NULL
            """
        ).fetchone()[0]
        or 0
    )
    active_reconcile = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE CAST(active_user_flag AS INTEGER) <> CASE WHEN coalesce(activity_events_month, 0) > 0 THEN 1 ELSE 0 END
            """
        ).fetchone()[0]
        or 0
    )
    strict_reconcile = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE CAST(strict_value_flag AS INTEGER) <> CASE WHEN coalesce(strict_download_count_month, 0) > 0 THEN 1 ELSE 0 END
            """
        ).fetchone()[0]
        or 0
    )
    session_sum_diff = float(
        conn.execute(
            """
            SELECT ABS(
              coalesce((SELECT SUM(session_count_month) FROM base_modelada_v2), 0)
              - coalesce((SELECT SUM(session_count_month) FROM fct_teacher_month WHERE month IS NOT NULL), 0)
            )
            """
        ).fetchone()[0]
        or 0.0
    )
    strict_download_sum_diff = float(
        conn.execute(
            """
            SELECT ABS(
              coalesce((SELECT SUM(strict_download_count_month) FROM base_modelada_v2), 0)
              - coalesce((SELECT SUM(strict_download_count_month) FROM fct_teacher_month WHERE month IS NOT NULL), 0)
            )
            """
        ).fetchone()[0]
        or 0.0
    )
    null_ts_excluded = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM raw_interactions i
            INNER JOIN raw_dim_teachers d USING(unique_id)
            WHERE lower(coalesce(i.user_type, '')) = 'registered'
              AND i.data_inicio IS NULL
            """
        ).fetchone()[0]
        or 0
    )

    add_check(
        "row_count_matches_fct_teacher_month",
        base_rows - fact_rows,
        "pass" if base_rows == fact_rows else "fail",
        "A base modelada deve ter exatamente 1 linha por teacher-month válido do fato mensal.",
    )
    add_check(
        "grain_teacher_month_unique",
        grain_duplicates,
        "pass" if grain_duplicates == 0 else "fail",
        "A chave (teacher_unique_id, month) deve ser única na base modelada.",
    )
    add_check(
        "missing_teacher_unique_id",
        missing_teacher,
        "pass" if missing_teacher == 0 else "fail",
        "A base modelada não pode perder a chave do professor.",
    )
    add_check(
        "missing_month",
        missing_month,
        "pass" if missing_month == 0 else "fail",
        "A base modelada não pode conter linhas sem mês.",
    )
    add_check(
        "dim_teacher_join_gap",
        join_gap,
        "pass" if join_gap == 0 else "fail",
        "Todo teacher-month modelado precisa encontrar exatamente 1 teacher na dimensão.",
    )
    add_check(
        "active_user_flag_reconciles",
        active_reconcile,
        "pass" if active_reconcile == 0 else "fail",
        "active_user_flag precisa refletir activity_events_month > 0.",
    )
    add_check(
        "strict_value_flag_reconciles",
        strict_reconcile,
        "pass" if strict_reconcile == 0 else "fail",
        "strict_value_flag precisa refletir strict_download_count_month > 0.",
    )
    add_check(
        "session_count_sum_diff_vs_fact",
        session_sum_diff,
        "pass" if session_sum_diff == 0 else "fail",
        "A soma de sessões da base modelada deve bater exatamente com fct_teacher_month.",
    )
    add_check(
        "strict_download_sum_diff_vs_fact",
        strict_download_sum_diff,
        "pass" if strict_download_sum_diff == 0 else "fail",
        "A soma de strict downloads da base modelada deve bater exatamente com fct_teacher_month.",
    )
    add_check(
        "registered_matched_interactions_with_null_timestamp_excluded",
        null_ts_excluded,
        "pass",
        "Interações registered com match e data_inicio nula são excluídas da base modelada por não terem mês confiável.",
    )
    return pd.DataFrame(rows)


def persist_samples_and_tables(conn: duckdb.DuckDBPyConnection, cfg: V2Config, table_names: List[str]) -> Dict[str, Dict[str, str]]:
    written: Dict[str, Dict[str, str]] = {}
    for name in table_names:
        written[name] = persist_table(conn, cfg, name)
    return written


def run_stage_02_modelagem(cfg: V2Config) -> Dict[str, Any]:
    ensure_output_dirs(cfg.output_dir)
    conn = connect_duckdb(cfg)
    try:
        register_raw_views(conn, cfg.data_dir)

        bridge_mari = create_bridge_mari_conversation_teacher(conn)
        persist_table(conn, cfg, "bridge_mari_conversation_teacher", bridge_mari)

        create_bridge_teacher_identity_audit(conn)

        dim_teacher = create_dim_teacher(conn)
        dim_lesson = create_dim_lesson(conn)
        dim_event = create_dim_event(conn)
        dim_device = create_dim_device()
        dim_calendar = create_dim_calendar(conn)

        persist_table(conn, cfg, "dim_teacher", dim_teacher)
        persist_table(conn, cfg, "dim_lesson", dim_lesson)
        persist_table(conn, cfg, "dim_event", dim_event)
        persist_table(conn, cfg, "dim_device", dim_device)
        persist_table(conn, cfg, "dim_calendar", dim_calendar)

        create_fct_session_tables(conn)
        create_fct_interaction_clean(conn)
        create_fct_mari_help_resolved(conn)

        teacher_month = build_teacher_month(conn)
        persist_table(conn, cfg, "fct_teacher_month", teacher_month)
        create_base_modelada(conn)
        persist_table(conn, cfg, "base_modelada_v2")
        base_validation = build_base_modelada_validation(conn)
        persist_table(conn, cfg, "audit_base_modelada_validation", base_validation)

        table_names = [
            "bridge_teacher_identity_audit",
            "fct_session_raw",
            "fct_session_clean",
            "fct_interaction_clean",
            "fct_mari_help_resolved",
        ]
        persist_samples_and_tables(conn, cfg, table_names)

        summary = {
            "generated_at_utc": utc_now_iso(),
            "duckdb_path": str(cfg.duckdb_path),
            "tables_materialized": {
                name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] or 0)
                for name in [
                    "dim_teacher",
                    "dim_lesson",
                    "dim_event",
                    "dim_device",
                    "dim_calendar",
                    "bridge_teacher_identity_audit",
                    "bridge_mari_conversation_teacher",
                    "fct_session_raw",
                    "fct_session_clean",
                    "fct_interaction_clean",
                    "fct_teacher_month",
                    "fct_mari_help_resolved",
                    "base_modelada_v2",
                ]
            },
            "strict_value_events": STRICT_VALUE_EVENTS,
            "core_population_rule": "raw_interactions/raw_entries com user_type='registered' e unique_id com match exato em dim_teachers.",
            "ping_rule_clean": "Excluir sessões <=5s do fct_session_clean.",
            "base_modelada_definition": "base_modelada_v2 = 1 linha por teacher-month do core, com fct_teacher_month enriquecido por dim_teacher sem joins de cardinalidade ambígua.",
            "base_modelada_exports": {
                "csv": str(cfg.output_dir / "csv" / "base_modelada_v2.csv"),
                "parquet": str(cfg.output_dir / "parquet" / "base_modelada_v2.parquet"),
            },
            "export_notes": {
                "full_csv_tables": sorted(FULL_CSV_TABLES),
                "sample_only_tables": sorted(SAMPLE_ONLY_TABLES),
            },
            "base_modelada_validation_status": (
                "pass"
                if base_validation[base_validation["status"] == "fail"].empty
                else "fail"
            ),
        }
        write_json(cfg.output_dir / "json" / "star_schema_summary_v2.json", summary)
        write_json(
            cfg.output_dir / "json" / "base_modelada_summary_v2.json",
            {
                "generated_at_utc": summary["generated_at_utc"],
                "table_name": "base_modelada_v2",
                "grain": "teacher_unique_id x month",
                "row_count": int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2").fetchone()[0] or 0),
                "distinct_teachers": int(conn.execute("SELECT COUNT(DISTINCT teacher_unique_id) FROM base_modelada_v2").fetchone()[0] or 0),
                "min_month": str(conn.execute("SELECT MIN(month) FROM base_modelada_v2").fetchone()[0]),
                "max_month": str(conn.execute("SELECT MAX(month) FROM base_modelada_v2").fetchone()[0]),
                "definition": summary["base_modelada_definition"],
                "exports": summary["base_modelada_exports"],
                "validation_status": summary["base_modelada_validation_status"],
            },
        )

        md_lines = [
            "# Star schema v2",
            "",
            f"- Gerado em UTC: {summary['generated_at_utc']}",
            f"- DuckDB materializada em: `{summary['duckdb_path']}`",
            f"- Regra de população core: {summary['core_population_rule']}",
            f"- Regra de ping do clean: {summary['ping_rule_clean']}",
            f"- Base modelada: {summary['base_modelada_definition']}",
            f"- CSV base modelada: `{summary['base_modelada_exports']['csv']}`",
            f"- Parquet base modelada: `{summary['base_modelada_exports']['parquet']}`",
            f"- Validação base modelada: `{summary['base_modelada_validation_status']}`",
            "",
            "## Tabelas materializadas",
        ]
        for name, count in summary["tables_materialized"].items():
            md_lines.append(f"- `{name}`: {count:,} linhas")
        md_lines.extend(
            [
                "",
                "## Exportação",
                f"- CSV full: {', '.join(summary['export_notes']['full_csv_tables'])}",
                f"- CSV sample apenas: {', '.join(summary['export_notes']['sample_only_tables'])}",
            ]
        )
        write_markdown(cfg.output_dir / "audit" / "star_schema_summary_v2.md", md_lines)
        write_markdown(
            cfg.output_dir / "audit" / "base_modelada_validation_v2.md",
            [
                "# Validação da base modelada v2",
                "",
                *[
                    f"- {row['check_name']}: {row['status']} ({row['metric_value']}) - {row['note']}"
                    for row in base_validation.to_dict(orient="records")
                ],
            ],
        )
        return summary
    finally:
        conn.close()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    run_stage_02_modelagem(cfg)


if __name__ == "__main__":
    main()
