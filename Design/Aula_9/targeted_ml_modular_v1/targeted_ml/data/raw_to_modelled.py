from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.data.sources import resolve_dataset_root
from targeted_ml.orchestration.artifacts import ProjectPaths, write_json


STRICT_VALUE_EVENTS = ["download_aula", "download_plano_aula"]
VALID_LESSON_ID_RE = r"^[A-Za-z0-9]{22}$"
UUID36_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
HEX64_UPPER_RE = r"^[0-9A-F]{64}$"
REQUIRED_RAW_FILES = [
    "dim_teachers.csv",
    "fct_teachers_entries.csv",
    "fct_teachers_contents_interactions.csv",
    "stg_lessons.csv",
    "stg_formation.csv",
    "stg_mari_ia_conversation.csv",
    "stg_mari_ia_reports.csv",
    "fct_mari_ia_eventos_isso_ajudou.csv",
    "calendario_escolar_uf_rede.csv",
]
MODELED_TABLES = [
    "dim_teacher",
    "dim_lesson",
    "dim_event",
    "dim_device",
    "dim_calendar",
    "bridge_mari_conversation_teacher",
    "bridge_teacher_identity_audit",
    "fct_session_raw",
    "fct_session_clean",
    "fct_interaction_clean",
    "fct_formation_clean",
    "fct_mari_conversation_resolved",
    "fct_mari_reports_resolved",
    "fct_mari_help_resolved",
    "fct_teacher_month",
    "base_modelada_v2",
    "mart_teacher_month_panel",
    "mart_teacher_month_cluster_ready",
    "mart_teacher_month_persona_ready",
    "mart_teacher_cluster_ready",
    "mart_teacher_persona_ready",
    "audit_persona_feature_readiness",
    "dim_persona_range_candidates",
    "audit_base_modelada_validation",
]


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def normalize_utm(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    lower = text.lower()
    if "google ads" in lower:
        return "google_ads"
    if "seo ads" in lower:
        return "seo_ads"
    if "seo org" in lower or "seo orgânico" in lower or "seo organico" in lower:
        return "seo_organico"
    if "landing" in lower:
        return "landing_page"
    if "blog" in lower:
        return "blog"
    if "social" in lower or "mídia" in lower or "midia" in lower:
        return "social"
    if "bot" in lower:
        return "bot"
    if "push" in lower:
        return "push"
    return re.sub(r"[^a-z0-9_]+", "_", lower).strip("_") or "other"


def classify_discipline_group(discipline: Any) -> str:
    if discipline is None or (isinstance(discipline, float) and np.isnan(discipline)):
        return "missing"
    lower = str(discipline).strip().lower()
    if not lower:
        return "missing"
    if lower in {"história", "historia", "geografia", "filosofia", "sociologia"}:
        return "Ciencias Humanas"
    if lower in {"ciências", "ciencias", "biologia", "química", "quimica", "física", "fisica"}:
        return "Ciencias da Natureza"
    if lower in {"matemática", "matematica"}:
        return "Matematica"
    if lower in {"português", "portugues", "inglês", "ingles", "literatura", "redação", "redacao"}:
        return "Linguagens"
    if lower in {"arte", "artes", "educação física", "educacao fisica"}:
        return "Artes e Complementares"
    return "Outras"


def classify_currentsubject_group(stage: Any, subject: Any) -> str:
    if subject is None or (isinstance(subject, float) and np.isnan(subject)):
        return "missing"
    subject_text = str(subject).strip().lower()
    stage_text = str(stage).strip().lower() if stage is not None else ""
    if subject_text in {"linguagens", "linguagem"}:
        return "Linguagens"
    if subject_text in {"ciencias", "ciências"}:
        return "Ciencias da Natureza"
    if subject_text in {"matematica", "matemática"}:
        return "Matematica"
    if subject_text in {"humanas", "ciências humanas", "ciencias humanas"}:
        return "Ciencias Humanas"
    code_map = {
        "1": "Linguagens",
        "2": "Matematica",
        "3": "Ciencias da Natureza",
        "4": "Ciencias Humanas",
        "5": "Linguagens",
        "6": "Matematica",
        "7": "Ciencias da Natureza",
        "8": "Ciencias Humanas",
    }
    if subject_text in code_map:
        if stage_text in {"em", "ensino_medio"} and subject_text == "1":
            return "Linguagens"
        return code_map[subject_text]
    return "Outras"


def sql_event_family_expr(column_name: str) -> str:
    return f"""
    CASE
      WHEN {column_name} IS NULL OR trim({column_name})='' THEN 'missing'
      WHEN lower({column_name}) LIKE '%plano%' THEN 'plano'
      WHEN lower({column_name}) LIKE '%prova%' OR lower({column_name}) LIKE '%avaliacao%' THEN 'prova'
      WHEN lower({column_name}) LIKE '%aula%' THEN 'aula'
      WHEN lower({column_name}) LIKE '%ia%' OR lower({column_name}) LIKE '%mari%' THEN 'ia'
      WHEN lower({column_name}) LIKE '%metodologia%' THEN 'metodologia'
      WHEN lower({column_name}) LIKE '%relatorio%' THEN 'relatorio'
      WHEN lower({column_name}) LIKE '%conquista%' THEN 'conquista'
      ELSE 'other'
    END
    """


def sql_event_action_expr(column_name: str) -> str:
    return f"""
    CASE
      WHEN {column_name} IS NULL OR trim({column_name})='' THEN 'missing'
      WHEN lower({column_name}) LIKE '%download%' OR lower({column_name}) LIKE '%baixar%' THEN 'download'
      WHEN lower({column_name}) LIKE '%visualizacao%' OR lower({column_name}) LIKE '%view%' THEN 'view'
      WHEN lower({column_name}) LIKE '%criacao%' OR lower({column_name}) LIKE '%criar%' OR lower({column_name}) LIKE '%salva%' OR lower({column_name}) LIKE '%edicao%' THEN 'create'
      WHEN lower({column_name}) LIKE '%compart%' OR lower({column_name}) LIKE '%envio_email%' THEN 'share'
      WHEN lower({column_name}) LIKE 'acesso_%' OR lower({column_name}) LIKE 'fechar_%' OR lower({column_name}) LIKE 'botao_%' THEN 'navigation'
      ELSE 'other'
    END
    """


def sql_id_aula_semantic_expr(column_name: str) -> str:
    return f"""
    CASE
      WHEN {column_name} IS NULL OR trim({column_name})='' THEN 'missing'
      WHEN regexp_matches({column_name}, '{VALID_LESSON_ID_RE}') THEN 'lesson_like_22char'
      WHEN regexp_matches({column_name}, '^[0-9]+$') THEN 'numeric_only'
      WHEN {column_name} IN ('s', 'S') THEN 'placeholder_s'
      WHEN lower({column_name}) LIKE '%conquista%' THEN 'navigation_token'
      WHEN regexp_matches({column_name}, '^[A-Za-z_]+$') THEN 'alpha_token'
      ELSE 'other'
    END
    """


def sql_device_expr(column_name: str) -> str:
    return f"""
    CASE
      WHEN {column_name} IS NULL OR trim({column_name})='' THEN 'unknown'
      WHEN lower(trim({column_name})) IN ('desktop', 'mobile', 'tablet') THEN lower(trim({column_name}))
      ELSE 'unknown'
    END
    """


def _resolve_raw_source_dir(spec: AnalysisSpec) -> Path | None:
    root = Path(spec.data.dataset_root).resolve()
    configured = getattr(spec.data, "raw_relative_path", Path("raw/base_aprendizap"))
    candidates: list[Path] = []
    if Path(configured).is_absolute():
        candidates.append(Path(configured).resolve())
    else:
        candidates.extend(
            [
                (root / configured).resolve(),
                (root / "raw" / "base_aprendizap").resolve(),
                (root / "base_aprendizap").resolve(),
                root,
            ]
        )
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        if all((candidate / file_name).exists() for file_name in REQUIRED_RAW_FILES):
            return candidate
    return None


def resolve_raw_source_dir(spec: AnalysisSpec) -> Path | None:
    return _resolve_raw_source_dir(spec)


def _persist_table(conn: duckdb.DuckDBPyConnection, paths: ProjectPaths, table_name: str, df: pd.DataFrame | None = None) -> Path:
    if df is not None:
        conn.register("_persist_df", df)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _persist_df")
        conn.unregister("_persist_df")
    parquet_path = paths.modelled_parquet_dir / f"{table_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY {table_name} TO '{q(parquet_path)}' (FORMAT PARQUET)")
    return parquet_path


def _normalize_modelled_tables(conn: duckdb.DuckDBPyConnection, table_names: list[str]) -> None:
    for table_name in table_names:
        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
        projections: list[str] = []
        for column_name, column_type, *_ in schema:
            upper = str(column_type).upper()
            quoted = f'"{column_name}"'
            if "TIMESTAMP" in upper:
                projections.append(f"COALESCE({quoted}, TIMESTAMP '1900-01-01 00:00:00') AS {quoted}")
            elif upper == "DATE":
                projections.append(f"COALESCE({quoted}, DATE '1900-01-01') AS {quoted}")
            elif any(token in upper for token in ["VARCHAR", "CHAR", "TEXT", "STRING"]):
                projections.append(f"COALESCE({quoted}, 'missing') AS {quoted}")
            elif upper == "BOOLEAN":
                projections.append(f"COALESCE({quoted}, FALSE) AS {quoted}")
            elif any(token in upper for token in ["INT", "DOUBLE", "FLOAT", "REAL", "DECIMAL"]):
                projections.append(f"COALESCE({quoted}, -1) AS {quoted}")
            else:
                projections.append(quoted)
        projection_sql = ",\n          ".join(projections)
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT
              {projection_sql}
            FROM {table_name}
            """
        )


def _normalize_teacher_negative_sentinels(conn: duckdb.DuckDBPyConnection, table_names: list[str]) -> None:
    for table_name in table_names:
        columns = {row[0] for row in conn.execute(f"DESCRIBE {table_name}").fetchall()}
        assignments: list[str] = []
        if "teacher_total_alunos" in columns:
            assignments.append("teacher_total_alunos = CASE WHEN teacher_total_alunos < 0 THEN -1 ELSE teacher_total_alunos END")
        if "teacher_alunos_diretos" in columns:
            assignments.append("teacher_alunos_diretos = CASE WHEN teacher_alunos_diretos < 0 THEN -1 ELSE teacher_alunos_diretos END")
        if "teacher_alunos_indiretos" in columns:
            assignments.append("teacher_alunos_indiretos = CASE WHEN teacher_alunos_indiretos < 0 THEN -1 ELSE teacher_alunos_indiretos END")
        if assignments:
            conn.execute(f"UPDATE {table_name} SET {', '.join(assignments)}")


def _register_raw_views(conn: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    conn.execute(
        f"CREATE OR REPLACE TEMP VIEW raw_dim_teachers AS SELECT * FROM read_csv('{q(raw_dir / 'dim_teachers.csv')}', delim=';', header=true, ignore_errors=true)"
    )
    for fname, view_name in [
        ("fct_teachers_entries.csv", "raw_entries"),
        ("fct_teachers_contents_interactions.csv", "raw_interactions"),
        ("stg_lessons.csv", "raw_lessons"),
        ("stg_formation.csv", "raw_formation"),
        ("stg_mari_ia_conversation.csv", "raw_mari_conv"),
        ("stg_mari_ia_reports.csv", "raw_mari_reports"),
        ("fct_mari_ia_eventos_isso_ajudou.csv", "raw_mari_help"),
        ("calendario_escolar_uf_rede.csv", "raw_school_calendar"),
    ]:
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW {view_name} AS SELECT * FROM read_csv_auto('{q(raw_dir / fname)}', header=true)"
        )


def _create_bridge_mari_conversation_teacher(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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


def _create_bridge_teacher_identity_audit(conn: duckdb.DuckDBPyConnection) -> None:
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
          CASE
            WHEN regexp_matches(e.source_key, '""" + UUID36_RE + """') THEN 'uuid36'
            ELSE 'other'
          END AS source_key_domain
        FROM entries_base e
        LEFT JOIN raw_dim_teachers d ON e.source_key=d.unique_id
        UNION ALL
        SELECT
          'raw_interactions',
          'unique_id',
          i.source_key,
          i.source_user_types,
          'exact_unique_id',
          CASE WHEN d.unique_id IS NOT NULL THEN d.unique_id END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE
            WHEN regexp_matches(i.source_key, '""" + UUID36_RE + """') THEN 'uuid36'
            ELSE 'other'
          END
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
          CASE
            WHEN regexp_matches(h.source_key, '""" + HEX64_UPPER_RE + """') THEN 'hex64_upper'
            ELSE 'other'
          END
        FROM mari_help_bridge h
        """
    )


def _create_dim_lesson(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute(
        f"""
        SELECT *
        FROM raw_lessons
        WHERE regexp_matches(coalesce(id_aula, ''), '{VALID_LESSON_ID_RE}')
        ORDER BY id_aula
        """
    ).fetchdf()
    df = df.rename(columns={"id_aula": "lesson_id"})
    df["discipline_group"] = df["disciplina"].apply(classify_discipline_group)
    df["lesson_id_semantic"] = "lesson_like_22char"
    df["is_active_methodology_missing"] = df["possui_metodologia_ativa"].isna().astype(int)
    return df


def _create_dim_event(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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


def _create_dim_device() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"device_group": "desktop", "description": "Desktop reconhecido pelo raw."},
            {"device_group": "mobile", "description": "Mobile reconhecido pelo raw."},
            {"device_group": "tablet", "description": "Tablet reconhecido pelo raw."},
            {"device_group": "unknown", "description": "Device ausente ou nao padronizado."},
        ]
    )


def _create_dim_calendar(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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


def _create_fct_session_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_session_raw AS
        SELECT
          CAST(hash(e.unique_id, e.user_type, e.data_inicio, e.data_fim) AS UBIGINT) AS session_row_hash,
          e.unique_id AS source_unique_id,
          d.unique_id AS teacher_unique_id,
          lower(coalesce(e.user_type,'missing')) AS user_type,
          date_trunc('month', TRY_CAST(e.data_inicio AS TIMESTAMP)) AS session_month,
          TRY_CAST(e.data_inicio AS TIMESTAMP) AS session_start_ts,
          TRY_CAST(e.data_fim AS TIMESTAMP) AS session_end_ts,
          CASE
            WHEN TRY_CAST(e.data_inicio AS TIMESTAMP) IS NOT NULL AND TRY_CAST(e.data_fim AS TIMESTAMP) IS NOT NULL
            THEN GREATEST(epoch(TRY_CAST(e.data_fim AS TIMESTAMP)) - epoch(TRY_CAST(e.data_inicio AS TIMESTAMP)), 0)
          END AS duration_sec,
          CASE
            WHEN TRY_CAST(e.data_inicio AS TIMESTAMP) IS NOT NULL AND TRY_CAST(e.data_fim AS TIMESTAMP) IS NOT NULL AND TRY_CAST(e.data_fim AS TIMESTAMP) < TRY_CAST(e.data_inicio AS TIMESTAMP)
            THEN 1 ELSE 0
          END AS is_negative_duration,
          CASE
            WHEN TRY_CAST(e.data_inicio AS TIMESTAMP) IS NOT NULL AND TRY_CAST(e.data_fim AS TIMESTAMP) IS NOT NULL
             AND GREATEST(epoch(TRY_CAST(e.data_fim AS TIMESTAMP)) - epoch(TRY_CAST(e.data_inicio AS TIMESTAMP)), 0) <= 1
            THEN 1 ELSE 0
          END AS is_ping_session_le_1s,
          CASE
            WHEN TRY_CAST(e.data_inicio AS TIMESTAMP) IS NOT NULL AND TRY_CAST(e.data_fim AS TIMESTAMP) IS NOT NULL
             AND GREATEST(epoch(TRY_CAST(e.data_fim AS TIMESTAMP)) - epoch(TRY_CAST(e.data_inicio AS TIMESTAMP)), 0) <= 5
            THEN 1 ELSE 0
          END AS is_ping_session_le_5s,
          CASE
            WHEN TRY_CAST(e.data_inicio AS TIMESTAMP) IS NOT NULL AND TRY_CAST(e.data_fim AS TIMESTAMP) IS NOT NULL
             AND GREATEST(epoch(TRY_CAST(e.data_fim AS TIMESTAMP)) - epoch(TRY_CAST(e.data_inicio AS TIMESTAMP)), 0) <= 10
            THEN 1 ELSE 0
          END AS is_ping_session_le_10s,
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


def _create_fct_interaction_clean(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE fct_interaction_clean AS
        WITH base AS (
          SELECT
            CAST(hash(i.unique_id, i.data_inicio, i.event_type, i.content_type, i.id_aula, i.utm_source) AS UBIGINT) AS interaction_row_hash,
            i.unique_id AS source_unique_id,
            d.unique_id AS teacher_unique_id,
            lower(coalesce(i.user_type,'missing')) AS user_type,
            date_trunc('month', TRY_CAST(i.data_inicio AS TIMESTAMP)) AS interaction_month,
            TRY_CAST(i.data_inicio AS TIMESTAMP) AS interaction_ts,
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
            AND TRY_CAST(i.data_inicio AS TIMESTAMP) IS NOT NULL
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


def _create_fct_formation_clean(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_formation_clean AS
        SELECT
          CAST(hash(f.unique_id_aprendizap, f.itemid, f.createdat, f.updatedat, f.type, f.completionstatus, f.progress, f.questionstatus) AS UBIGINT) AS formation_row_hash,
          d.unique_id AS teacher_unique_id,
          COALESCE(TRY_CAST(f.createdat AS TIMESTAMP), TRY_CAST(f.updatedat AS TIMESTAMP)) AS formation_ts,
          CAST(date_trunc('month', COALESCE(TRY_CAST(f.createdat AS TIMESTAMP), TRY_CAST(f.updatedat AS TIMESTAMP))) AS DATE) AS formation_month,
          f.itemid AS item_id,
          f.type AS item_type,
          f.completionstatus AS completion_status,
          TRY_CAST(f.progress AS BIGINT) AS progress,
          coalesce(f.questionstatus, 'missing') AS question_status,
          COALESCE(TRY_CAST(f.coursemodulecount AS BIGINT), -1) AS course_module_count,
          COALESCE(TRY_CAST(f.moduleblockcount AS BIGINT), -1) AS module_block_count,
          COALESCE(TRY_CAST(f.quizquestioncount AS BIGINT), -1) AS quiz_question_count,
          CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS teacher_matched_flag,
          CASE WHEN COALESCE(TRY_CAST(f.createdat AS TIMESTAMP), TRY_CAST(f.updatedat AS TIMESTAMP)) IS NULL THEN 1 ELSE 0 END AS is_missing_timestamp,
          CASE WHEN lower(coalesce(f.completionstatus, ''))='complete' THEN 1 ELSE 0 END AS is_complete_status
        FROM raw_formation f
        LEFT JOIN raw_dim_teachers d ON f.unique_id_aprendizap = d.unique_id
        WHERE d.unique_id IS NOT NULL
        """
    )


def _create_fct_mari_conversation_resolved(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_conversation_resolved AS
        SELECT
          c.id_mari,
          b.teacher_unique_id,
          TRY_CAST(c.createdat AS TIMESTAMP) AS mari_created_ts,
          TRY_CAST(c.updatedat AS TIMESTAMP) AS mari_updated_ts,
          date_trunc('month', COALESCE(TRY_CAST(c.updatedat AS TIMESTAMP), TRY_CAST(c.createdat AS TIMESTAMP))) AS mari_month,
          coalesce(c.originsource, 'missing') AS origin_source,
          coalesce(c.userreaction, 'missing') AS user_reaction,
          CASE WHEN c.userlastmessage IS NULL OR trim(c.userlastmessage)='' THEN 0 ELSE 1 END AS has_user_message,
          CASE WHEN c.ailastmessage IS NULL OR trim(c.ailastmessage)='' THEN 0 ELSE 1 END AS has_ai_message,
          CASE WHEN TRY_CAST(c.createdat AS TIMESTAMP) IS NULL THEN 1 ELSE 0 END AS is_mari_created_ts_missing,
          CASE WHEN TRY_CAST(c.updatedat AS TIMESTAMP) IS NULL THEN 1 ELSE 0 END AS is_mari_updated_ts_missing
        FROM raw_mari_conv c
        INNER JOIN bridge_mari_conversation_teacher b
          ON c.id_mari = b.id_mari
        WHERE b.is_unambiguous=1
          AND b.teacher_unique_id IS NOT NULL
        """
    )


def _create_fct_mari_reports_resolved(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_reports_resolved AS
        SELECT
          r.id_mari,
          b.teacher_unique_id,
          TRY_CAST(r.updatedat AS TIMESTAMP) AS report_ts,
          CAST(date_trunc('month', TRY_CAST(r.updatedat AS TIMESTAMP)) AS DATE) AS report_month,
          r.key AS report_key,
          r.value AS report_value,
          coalesce(r.metadata, 'missing') AS report_metadata
        FROM raw_mari_reports r
        INNER JOIN bridge_mari_conversation_teacher b
          ON r.id_mari = b.id_mari
        WHERE b.is_unambiguous=1
          AND b.teacher_unique_id IS NOT NULL
        """
    )


def _create_fct_mari_help_resolved(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_help_resolved AS
        SELECT
          h.user_id AS id_mari,
          b.teacher_unique_id,
          b.teacher_resolution_count,
          b.is_unambiguous,
          b.resolution_source,
          TRY_CAST(h.date AS TIMESTAMP) AS help_ts,
          CAST(date_trunc('month', TRY_CAST(h.date AS TIMESTAMP)) AS DATE) AS help_month,
          h.turno,
          h.key,
          h.isso_ajudou,
          TRY_CAST(h.isso_ajudou_num AS INTEGER) AS isso_ajudou_num
        FROM raw_mari_help h
        INNER JOIN bridge_mari_conversation_teacher b
          ON h.user_id=b.id_mari
        WHERE b.is_unambiguous=1
          AND b.teacher_unique_id IS NOT NULL
        """
    )


def _build_teacher_month(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    interactions_month = conn.execute(
        """
        SELECT
          teacher_unique_id,
          CAST(interaction_month AS TIMESTAMP) AS month,
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
          MAX(CASE WHEN is_activity_event=1 THEN 1 ELSE 0 END) AS active_user_flag,
          MAX(CASE WHEN event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END) AS strict_value_flag,
          MAX(CASE WHEN event_family='aula' AND is_content_view_event=1 THEN 1 ELSE 0 END) AS viewed_aula_flag,
          MAX(CASE WHEN event_family='plano' AND is_content_view_event=1 THEN 1 ELSE 0 END) AS viewed_plano_flag,
          MAX(CASE WHEN event_family='prova' AND is_content_view_event=1 THEN 1 ELSE 0 END) AS viewed_prova_flag,
          MAX(CASE WHEN event_family='ia' AND is_activity_event=1 THEN 1 ELSE 0 END) AS used_ia_flag,
          MAX(CASE WHEN device_group='desktop' THEN 1 ELSE 0 END) AS used_desktop_flag,
          MAX(CASE WHEN device_group='mobile' THEN 1 ELSE 0 END) AS used_mobile_flag
        FROM fct_interaction_clean
        GROUP BY 1, 2
        """
    ).fetchdf()
    raw_sessions_month = conn.execute(
        """
        SELECT
          teacher_unique_id,
          CAST(session_month AS TIMESTAMP) AS month,
          COUNT(*) AS raw_entry_session_count_month,
          SUM(CASE WHEN is_ping_session_le_5s=1 THEN 1 ELSE 0 END) AS ping_entry_session_count_month
        FROM fct_session_raw
        WHERE is_core_teacher_session=1
        GROUP BY 1, 2
        """
    ).fetchdf()
    clean_sessions_month = conn.execute(
        """
        SELECT
          teacher_unique_id,
          CAST(session_month AS TIMESTAMP) AS month,
          COUNT(*) AS clean_entry_session_count_month,
          SUM(duration_sec) / 60.0 AS clean_entry_total_session_minutes_month,
          AVG(duration_sec) / 60.0 AS clean_entry_avg_session_minutes_month
        FROM fct_session_clean
        GROUP BY 1, 2
        """
    ).fetchdf()
    if interactions_month.empty and raw_sessions_month.empty and clean_sessions_month.empty:
        return pd.DataFrame()
    month_df = raw_sessions_month.merge(clean_sessions_month, on=["teacher_unique_id", "month"], how="outer")
    month_df = month_df.merge(interactions_month, on=["teacher_unique_id", "month"], how="outer")
    numeric_fill_zero = [
        "raw_entry_session_count_month",
        "ping_entry_session_count_month",
        "clean_entry_session_count_month",
        "clean_entry_total_session_minutes_month",
        "clean_entry_avg_session_minutes_month",
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
        "active_user_flag",
        "strict_value_flag",
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
    month_df["interaction_signal_flag"] = (month_df["interaction_rows_month"] > 0).astype(int)
    month_df["entry_signal_flag"] = (month_df["raw_entry_session_count_month"] > 0).astype(int)
    month_df["clean_entry_signal_flag"] = (month_df["clean_entry_session_count_month"] > 0).astype(int)
    month_df["only_ping_entry_flag"] = (
        (month_df["raw_entry_session_count_month"] > 0)
        & (month_df["clean_entry_session_count_month"] <= 0)
        & (month_df["ping_entry_session_count_month"] >= month_df["raw_entry_session_count_month"])
    ).astype(int)
    month_df["any_signal_flag"] = (
        (month_df["interaction_signal_flag"] == 1) | (month_df["entry_signal_flag"] == 1)
    ).astype(int)
    month_df["month_signal_class"] = np.select(
        [
            (month_df["interaction_signal_flag"] == 1) & (month_df["clean_entry_signal_flag"] == 1),
            (month_df["interaction_signal_flag"] == 1) & (month_df["only_ping_entry_flag"] == 1),
            (month_df["interaction_signal_flag"] == 1) & (month_df["entry_signal_flag"] == 0),
            (month_df["clean_entry_signal_flag"] == 1) & (month_df["interaction_signal_flag"] == 0),
            month_df["only_ping_entry_flag"] == 1,
            (month_df["entry_signal_flag"] == 1) & (month_df["clean_entry_signal_flag"] == 0),
        ],
        [
            "interaction_with_clean_entry",
            "interaction_with_ping_only_entry",
            "interaction_without_entry",
            "clean_entry_without_interaction",
            "ping_only_entry",
            "raw_entry_without_clean_entry",
        ],
        default="no_signal",
    )
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
    month_df["clean_entry_exposed_no_download_flag"] = (
        (month_df["clean_entry_session_count_month"] > 0) & (month_df["strict_download_count_month"] <= 0)
    ).astype(int)
    month_df["clean_entry_exposed_no_activity_no_download_flag"] = (
        (month_df["clean_entry_session_count_month"] > 0)
        & (month_df["activity_events_month"] <= 0)
        & (month_df["strict_download_count_month"] <= 0)
    ).astype(int)
    month_df["clean_entry_exposed_activity_no_download_flag"] = (
        (month_df["clean_entry_session_count_month"] > 0)
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
        month_df["next_month_active_user_flag"].fillna(0).astype(int),
        np.nan,
    )
    month_df["returned_strict_value_m1"] = np.where(
        month_df["next_month_observed_flag"] == 1,
        month_df["next_month_strict_value_flag"].fillna(0).astype(int),
        np.nan,
    )
    month_df["returned_any_download_m1"] = np.where(
        month_df["next_month_observed_flag"] == 1,
        (month_df["next_month_strict_download_count"].fillna(0) > 0).astype(int),
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
    month_df["lifetime_clean_entry_minutes_total"] = month_df.groupby("teacher_unique_id")["clean_entry_total_session_minutes_month"].cumsum()

    def add_streaks(group: pd.DataFrame, flag_col: str, current_col: str, max_col: str) -> pd.DataFrame:
        current_values: list[int] = []
        max_values: list[int] = []
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

    active_frames = [
        add_streaks(frame.sort_values("month"), "active_user_flag", "active_streak_current_months", "active_streak_max_months")
        for _, frame in month_df.groupby("teacher_unique_id", sort=False)
    ]
    month_df = pd.concat(active_frames, ignore_index=True) if active_frames else month_df
    strict_frames = [
        add_streaks(frame.sort_values("month"), "strict_value_flag", "strict_streak_current_months", "strict_streak_max_months")
        for _, frame in month_df.groupby("teacher_unique_id", sort=False)
    ]
    month_df = pd.concat(strict_frames, ignore_index=True) if strict_frames else month_df
    ordered_columns = [
        "teacher_unique_id",
        "month",
        "raw_entry_session_count_month",
        "ping_entry_session_count_month",
        "clean_entry_session_count_month",
        "clean_entry_total_session_minutes_month",
        "clean_entry_avg_session_minutes_month",
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
        "interaction_signal_flag",
        "entry_signal_flag",
        "clean_entry_signal_flag",
        "only_ping_entry_flag",
        "any_signal_flag",
        "month_signal_class",
        "strict_value_flag",
        "active_user_flag",
        "viewed_aula_flag",
        "viewed_plano_flag",
        "viewed_prova_flag",
        "used_ia_flag",
        "used_desktop_flag",
        "used_mobile_flag",
        "no_download_flag",
        "no_download_view_only_flag",
        "no_download_view_plus_action_flag",
        "no_download_action_only_flag",
        "clean_entry_exposed_no_download_flag",
        "clean_entry_exposed_no_activity_no_download_flag",
        "clean_entry_exposed_activity_no_download_flag",
        "month_num",
        "next_month",
        "next_month_observed_flag",
        "returned_active_m1",
        "returned_strict_value_m1",
        "returned_any_download_m1",
        "strict_user_flag",
        "strict_return_value_m1",
        "lifetime_active_months",
        "lifetime_clean_entry_minutes_total",
        "active_streak_current_months",
        "active_streak_max_months",
        "strict_streak_current_months",
        "strict_streak_max_months",
    ]
    return month_df[ordered_columns].copy()


def _create_dim_teacher(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute(
        """
        WITH coverage AS (
          SELECT
            d.unique_id,
            CASE WHEN e.unique_id IS NOT NULL THEN 1 ELSE 0 END AS has_registered_entry,
            CASE WHEN i.unique_id IS NOT NULL THEN 1 ELSE 0 END AS has_registered_interaction,
            CASE WHEN f.unique_id_aprendizap IS NOT NULL THEN 1 ELSE 0 END AS has_formation
          FROM raw_dim_teachers d
          LEFT JOIN (SELECT DISTINCT unique_id FROM raw_entries WHERE lower(coalesce(user_type,''))='registered') e ON d.unique_id=e.unique_id
          LEFT JOIN (SELECT DISTINCT unique_id FROM raw_interactions WHERE lower(coalesce(user_type,''))='registered') i ON d.unique_id=i.unique_id
          LEFT JOIN (SELECT DISTINCT unique_id_aprendizap FROM raw_formation) f ON d.unique_id=f.unique_id_aprendizap
        ),
        teacher_month_summary AS (
          SELECT
            teacher_unique_id,
            MIN(month) AS teacher_first_observed_month,
            MAX(month) AS teacher_last_observed_month,
            COUNT(*) AS teacher_observed_months_total,
            SUM(active_user_flag) AS teacher_active_months_total,
            SUM(strict_value_flag) AS teacher_strict_months_total,
            SUM(strict_download_count_month) AS teacher_total_strict_downloads,
            SUM(download_count_month) AS teacher_total_downloads,
            SUM(clean_entry_session_count_month) AS teacher_total_clean_entry_sessions,
            SUM(clean_entry_total_session_minutes_month) AS teacher_total_clean_entry_minutes,
            MAX(active_streak_max_months) AS teacher_active_streak_max_months,
            MAX(strict_streak_max_months) AS teacher_strict_streak_max_months
          FROM fct_teacher_month
          GROUP BY 1
        ),
        dataset_end AS (
          SELECT MAX(month_num) AS dataset_end_month_num
          FROM fct_teacher_month
        )
        SELECT
          d.unique_id AS teacher_unique_id,
          CASE
            WHEN coalesce(tms.teacher_active_months_total, 0) > 0
              OR coalesce(c.has_registered_interaction, 0) = 1
            THEN 'teacher_with_registered_activity'
            ELSE 'teacher_without_registered_activity'
          END AS teacher_population_status,
          d.utm_origin AS teacher_utm_origin,
          d.tela_origem AS teacher_tela_origem,
          d.estado AS teacher_estado,
          d.total_alunos AS teacher_total_alunos,
          d.tipo_total_alunos AS teacher_tipo_total_alunos,
          d.alunos_diretos AS teacher_alunos_diretos,
          d.alunos_indiretos AS teacher_alunos_indiretos,
          d.login_google AS teacher_login_google,
          d.currentstage AS teacher_currentstage,
          d.currentsubject AS teacher_currentsubject,
          d.selectedstages AS teacher_selectedstages,
          d.selectedsubjectsem AS teacher_selectedsubjectsem,
          d.selectedsubjectsfundii AS teacher_selectedsubjectsfundii,
          d.visualizou_metodologia_ativa AS teacher_visualizou_metodologia_ativa,
          TRY_CAST(d.data_entrada AS TIMESTAMP) AS teacher_data_entrada,
          tms.teacher_first_observed_month,
          tms.teacher_last_observed_month,
          CASE
            WHEN tms.teacher_last_observed_month IS NULL THEN NULL
            ELSE de.dataset_end_month_num - (EXTRACT(year FROM tms.teacher_last_observed_month) * 12 + EXTRACT(month FROM tms.teacher_last_observed_month))
          END AS teacher_months_since_last_observed_month_dataset_end,
          tms.teacher_observed_months_total,
          tms.teacher_active_months_total,
          tms.teacher_strict_months_total,
          tms.teacher_total_strict_downloads,
          tms.teacher_total_downloads,
          tms.teacher_total_clean_entry_sessions,
          tms.teacher_total_clean_entry_minutes,
          tms.teacher_active_streak_max_months,
          tms.teacher_strict_streak_max_months,
          c.has_registered_entry,
          c.has_registered_interaction,
          c.has_formation,
          CASE WHEN d.estado IS NULL OR trim(d.estado)='' THEN 1 ELSE 0 END AS is_estado_missing,
          CASE WHEN d.utm_origin IS NULL OR trim(d.utm_origin)='' THEN 1 ELSE 0 END AS is_utm_missing,
          CASE WHEN d.tela_origem IS NULL OR trim(d.tela_origem)='' THEN 1 ELSE 0 END AS is_tela_origem_missing,
          CASE WHEN d.total_alunos IS NULL THEN 1 ELSE 0 END AS is_total_alunos_missing,
          CASE WHEN d.tipo_total_alunos IS NULL OR trim(d.tipo_total_alunos)='' THEN 1 ELSE 0 END AS is_tipo_total_alunos_missing,
          CASE WHEN d.alunos_diretos IS NULL THEN 1 ELSE 0 END AS is_alunos_diretos_missing,
          CASE WHEN d.alunos_indiretos IS NULL THEN 1 ELSE 0 END AS is_alunos_indiretos_missing,
          CASE WHEN d.login_google IS NULL THEN 1 ELSE 0 END AS is_login_google_missing,
          CASE WHEN d.currentstage IS NULL OR trim(d.currentstage)='' THEN 1 ELSE 0 END AS is_currentstage_missing,
          CASE WHEN d.currentsubject IS NULL OR trim(d.currentsubject)='' THEN 1 ELSE 0 END AS is_currentsubject_missing,
          CASE WHEN d.selectedstages IS NULL OR trim(d.selectedstages)='' THEN 1 ELSE 0 END AS is_selectedstages_missing,
          CASE WHEN d.selectedsubjectsem IS NULL OR trim(d.selectedsubjectsem)='' THEN 1 ELSE 0 END AS is_selectedsubjectsem_missing,
          CASE WHEN d.selectedsubjectsfundii IS NULL OR trim(d.selectedsubjectsfundii)='' THEN 1 ELSE 0 END AS is_selectedsubjectsfundii_missing,
          CASE WHEN d.visualizou_metodologia_ativa IS NULL THEN 1 ELSE 0 END AS is_visualizou_metodologia_ativa_missing,
          CASE WHEN d.total_alunos < 0 THEN 1 ELSE 0 END AS is_total_alunos_negative,
          CASE WHEN d.alunos_diretos < 0 THEN 1 ELSE 0 END AS is_alunos_diretos_negative,
          CASE WHEN d.alunos_indiretos < 0 THEN 1 ELSE 0 END AS is_alunos_indiretos_negative,
          CASE WHEN d.login_google IS NOT NULL AND d.login_google NOT IN (0,1) THEN 1 ELSE 0 END AS is_login_google_invalid
        FROM raw_dim_teachers d
        LEFT JOIN coverage c ON d.unique_id=c.unique_id
        LEFT JOIN teacher_month_summary tms ON d.unique_id=tms.teacher_unique_id
        CROSS JOIN dataset_end de
        WHERE tms.teacher_unique_id IS NOT NULL
        ORDER BY d.unique_id
        """
    ).fetchdf()
    df["teacher_utm_group"] = df["teacher_utm_origin"].apply(normalize_utm)
    df["teacher_currentsubject_group"] = [
        classify_currentsubject_group(stage, subject)
        for stage, subject in zip(df["teacher_currentstage"], df["teacher_currentsubject"])
    ]
    return df


def _create_base_modelada(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE base_modelada_v2 AS
        SELECT
          tm.teacher_unique_id,
          tm.month,
          dt.teacher_population_status,
          dt.teacher_utm_origin,
          dt.teacher_utm_group,
          dt.teacher_tela_origem,
          dt.teacher_estado,
          dt.teacher_total_alunos,
          dt.teacher_tipo_total_alunos,
          dt.teacher_alunos_diretos,
          dt.teacher_alunos_indiretos,
          dt.teacher_login_google,
          dt.teacher_currentstage,
          dt.teacher_currentsubject,
          dt.teacher_currentsubject_group,
          dt.teacher_selectedstages,
          dt.teacher_selectedsubjectsem,
          dt.teacher_selectedsubjectsfundii,
          dt.teacher_visualizou_metodologia_ativa,
          dt.teacher_data_entrada,
          dt.has_registered_entry,
          dt.has_registered_interaction,
          dt.has_formation,
          dt.is_estado_missing,
          dt.is_utm_missing,
          dt.is_tela_origem_missing,
          dt.is_total_alunos_missing,
          dt.is_tipo_total_alunos_missing,
          dt.is_alunos_diretos_missing,
          dt.is_alunos_indiretos_missing,
          dt.is_login_google_missing,
          dt.is_currentstage_missing,
          dt.is_currentsubject_missing,
          dt.is_selectedstages_missing,
          dt.is_selectedsubjectsem_missing,
          dt.is_selectedsubjectsfundii_missing,
          dt.is_visualizou_metodologia_ativa_missing,
          dt.is_total_alunos_negative,
          dt.is_alunos_diretos_negative,
          dt.is_alunos_indiretos_negative,
          dt.is_login_google_invalid,
          tm.raw_entry_session_count_month,
          tm.ping_entry_session_count_month,
          tm.clean_entry_session_count_month,
          tm.clean_entry_total_session_minutes_month,
          tm.clean_entry_avg_session_minutes_month,
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
          tm.interaction_signal_flag,
          tm.entry_signal_flag,
          tm.clean_entry_signal_flag,
          tm.only_ping_entry_flag,
          tm.any_signal_flag,
          tm.month_signal_class,
          tm.strict_value_flag,
          tm.active_user_flag,
          tm.viewed_aula_flag,
          tm.viewed_plano_flag,
          tm.viewed_prova_flag,
          tm.used_ia_flag,
          tm.used_desktop_flag,
          tm.used_mobile_flag,
          tm.no_download_flag,
          tm.no_download_view_only_flag,
          tm.no_download_view_plus_action_flag,
          tm.no_download_action_only_flag,
          tm.clean_entry_exposed_no_download_flag,
          tm.clean_entry_exposed_no_activity_no_download_flag,
          tm.clean_entry_exposed_activity_no_download_flag,
          tm.month_num,
          tm.next_month,
          tm.next_month_observed_flag,
          tm.returned_active_m1,
          tm.returned_strict_value_m1,
          tm.returned_any_download_m1,
          tm.strict_user_flag,
          tm.strict_return_value_m1,
          tm.lifetime_active_months,
          tm.lifetime_clean_entry_minutes_total,
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


def _create_mart_teacher_month_panel(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_month_panel AS
        WITH teacher_bounds AS (
          SELECT
            teacher_unique_id,
            MIN(month) AS first_observed_month
          FROM fct_teacher_month
          GROUP BY 1
        ),
        dataset_end AS (
          SELECT MAX(month) AS last_month
          FROM fct_teacher_month
        ),
        panel AS (
          SELECT
            tb.teacher_unique_id,
            gs.month::TIMESTAMP AS month
          FROM teacher_bounds tb
          CROSS JOIN dataset_end de
          CROSS JOIN generate_series(tb.first_observed_month, de.last_month, INTERVAL 1 MONTH) AS gs(month)
        ),
        joined AS (
          SELECT
            p.teacher_unique_id,
            p.month,
            COALESCE(tm.month_num, EXTRACT(year FROM p.month) * 12 + EXTRACT(month FROM p.month)) AS month_num,
            CASE WHEN tm.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END AS observed_month_flag,
            CASE WHEN COALESCE(tm.any_signal_flag, 0) = 0 THEN 1 ELSE 0 END AS no_signal_month_flag,
            COALESCE(tm.any_signal_flag, 0) AS any_signal_flag,
            COALESCE(tm.interaction_signal_flag, 0) AS interaction_signal_flag,
            COALESCE(tm.entry_signal_flag, 0) AS entry_signal_flag,
            COALESCE(tm.clean_entry_signal_flag, 0) AS clean_entry_signal_flag,
            COALESCE(tm.only_ping_entry_flag, 0) AS only_ping_entry_flag,
            COALESCE(tm.month_signal_class, CASE WHEN tm.teacher_unique_id IS NULL THEN 'no_signal' ELSE 'missing' END) AS month_signal_class,
            COALESCE(tm.active_user_flag, 0) AS active_user_flag,
            COALESCE(tm.strict_value_flag, 0) AS strict_value_flag,
            COALESCE(tm.strict_download_count_month, 0) AS strict_download_count_month,
            COALESCE(tm.download_count_month, 0) AS download_count_month,
            COALESCE(tm.activity_events_month, 0) AS activity_events_month,
            COALESCE(tm.active_days_month, 0) AS active_days_month,
            COALESCE(tm.content_views_month, 0) AS content_views_month,
            COALESCE(tm.other_activity_non_download_events_month, 0) AS other_activity_non_download_events_month,
            COALESCE(tm.aula_events_month, 0) AS aula_events_month,
            COALESCE(tm.plano_events_month, 0) AS plano_events_month,
            COALESCE(tm.prova_events_month, 0) AS prova_events_month,
            COALESCE(tm.ia_events_month, 0) AS ia_events_month,
            COALESCE(tm.mapped_lessons_month, 0) AS mapped_lessons_month,
            COALESCE(tm.raw_entry_session_count_month, 0) AS raw_entry_session_count_month,
            COALESCE(tm.ping_entry_session_count_month, 0) AS ping_entry_session_count_month,
            COALESCE(tm.clean_entry_session_count_month, 0) AS clean_entry_session_count_month,
            COALESCE(tm.clean_entry_total_session_minutes_month, 0) AS clean_entry_total_session_minutes_month,
            COALESCE(tm.clean_entry_avg_session_minutes_month, 0) AS clean_entry_avg_session_minutes_month,
            dt.teacher_estado,
            dt.teacher_currentsubject_group,
            dt.teacher_currentstage,
            dt.teacher_utm_group
          FROM panel p
          LEFT JOIN fct_teacher_month tm
            ON p.teacher_unique_id = tm.teacher_unique_id
           AND p.month = tm.month
          LEFT JOIN dim_teacher dt
            ON p.teacher_unique_id = dt.teacher_unique_id
        ),
        streaks AS (
          SELECT
            *,
            MAX(CASE WHEN any_signal_flag = 1 THEN month_num END) OVER (
              PARTITION BY teacher_unique_id
              ORDER BY month
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS last_signal_month_num,
            MAX(CASE WHEN active_user_flag = 1 THEN month_num END) OVER (
              PARTITION BY teacher_unique_id
              ORDER BY month
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS last_active_month_num,
            MAX(CASE WHEN strict_value_flag = 1 THEN month_num END) OVER (
              PARTITION BY teacher_unique_id
              ORDER BY month
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS last_strict_month_num
          FROM joined
        )
        SELECT
          teacher_unique_id,
          month,
          month_num,
          observed_month_flag,
          no_signal_month_flag,
          any_signal_flag,
          interaction_signal_flag,
          entry_signal_flag,
          clean_entry_signal_flag,
          only_ping_entry_flag,
          month_signal_class,
          active_user_flag,
          strict_value_flag,
          strict_download_count_month,
          download_count_month,
          activity_events_month,
          active_days_month,
          content_views_month,
          other_activity_non_download_events_month,
          aula_events_month,
          plano_events_month,
          prova_events_month,
          ia_events_month,
          mapped_lessons_month,
          raw_entry_session_count_month,
          ping_entry_session_count_month,
          clean_entry_session_count_month,
          clean_entry_total_session_minutes_month,
          clean_entry_avg_session_minutes_month,
          CASE WHEN last_signal_month_num IS NULL THEN NULL ELSE month_num - last_signal_month_num END AS months_since_last_signal,
          CASE WHEN last_active_month_num IS NULL THEN NULL ELSE month_num - last_active_month_num END AS months_since_last_active,
          CASE WHEN last_strict_month_num IS NULL THEN NULL ELSE month_num - last_strict_month_num END AS months_since_last_strict_value,
          teacher_estado,
          teacher_currentsubject_group,
          teacher_currentstage,
          teacher_utm_group
        FROM streaks
        ORDER BY teacher_unique_id, month
        """
    )


def _create_mart_teacher_month_cluster_ready(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_month_cluster_ready AS
        SELECT
          b.teacher_unique_id,
          b.month,
          b.month_num,
          CASE WHEN b.interaction_signal_flag = 1 THEN 1 ELSE 0 END AS cluster_analysis_eligible_flag,
          b.activity_events_month,
          b.active_days_month,
          b.strict_download_count_month,
          b.download_count_month,
          b.content_views_month,
          b.other_activity_non_download_events_month,
          b.aula_events_month,
          b.plano_events_month,
          b.prova_events_month,
          b.ia_events_month,
          b.mapped_lessons_month,
          b.clean_entry_session_count_month,
          b.clean_entry_total_session_minutes_month,
          b.clean_entry_avg_session_minutes_month,
          b.clean_entry_signal_flag,
          b.only_ping_entry_flag,
          b.interaction_signal_flag,
          b.month_signal_class,
          b.teacher_estado,
          b.teacher_currentsubject_group,
          b.teacher_currentstage,
          b.teacher_utm_group,
          b.used_desktop_flag,
          b.used_mobile_flag
        FROM base_modelada_v2 b
        ORDER BY b.teacher_unique_id, b.month
        """
    )


def _create_mart_teacher_month_persona_ready(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_month_persona_ready AS
        SELECT
          b.teacher_unique_id,
          b.month,
          b.month_num,
          1 AS observed_month_flag,
          CASE WHEN coalesce(b.active_user_flag, 0) > 0 THEN 1 ELSE 0 END AS persona_analysis_eligible_flag,
          b.active_user_flag,
          b.strict_value_flag,
          b.strict_user_flag,
          b.returned_active_m1,
          b.returned_any_download_m1,
          b.returned_strict_value_m1,
          b.strict_return_value_m1,
          b.next_month_observed_flag,
          b.interaction_rows_month,
          b.activity_events_month,
          b.active_days_month,
          b.raw_entry_session_count_month,
          b.ping_entry_session_count_month,
          b.clean_entry_session_count_month,
          b.clean_entry_total_session_minutes_month,
          b.clean_entry_avg_session_minutes_month,
          b.strict_download_count_month,
          b.download_count_month,
          b.download_aula_count_month,
          b.download_plano_count_month,
          b.content_views_month,
          b.other_activity_non_download_events_month,
          b.aula_events_month,
          b.plano_events_month,
          b.prova_events_month,
          b.ia_events_month,
          b.mapped_lessons_month,
          b.interaction_signal_flag,
          b.entry_signal_flag,
          b.clean_entry_signal_flag,
          b.only_ping_entry_flag,
          b.any_signal_flag,
          b.month_signal_class,
          b.used_desktop_flag,
          b.used_mobile_flag,
          b.used_ia_flag,
          b.no_download_flag,
          b.no_download_view_only_flag,
          b.no_download_view_plus_action_flag,
          b.no_download_action_only_flag,
          b.clean_entry_exposed_no_download_flag,
          b.clean_entry_exposed_no_activity_no_download_flag,
          b.clean_entry_exposed_activity_no_download_flag,
          b.lifetime_active_months,
          b.lifetime_clean_entry_minutes_total,
          b.active_streak_current_months,
          b.active_streak_max_months,
          b.strict_streak_current_months,
          b.strict_streak_max_months,
          b.teacher_population_status,
          b.teacher_estado,
          b.teacher_currentsubject_group,
          b.teacher_currentstage,
          b.teacher_utm_group,
          b.teacher_total_alunos,
          b.teacher_tipo_total_alunos,
          b.is_estado_missing,
          b.is_currentsubject_missing,
          b.is_utm_missing,
          b.is_total_alunos_missing,
          b.is_tipo_total_alunos_missing
        FROM base_modelada_v2 b
        ORDER BY b.teacher_unique_id, b.month
        """
    )


def _create_mart_teacher_cluster_ready(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_cluster_ready AS
        WITH observed AS (
          SELECT
            p.*,
            COALESCE(tm.used_mobile_flag, 0) AS used_mobile_flag,
            COALESCE(tm.used_desktop_flag, 0) AS used_desktop_flag
          FROM mart_teacher_month_panel
          p
          LEFT JOIN fct_teacher_month tm
            ON p.teacher_unique_id = tm.teacher_unique_id
           AND p.month = tm.month
          WHERE p.observed_month_flag = 1
        ),
        eligible AS (
          SELECT *
          FROM mart_teacher_month_cluster_ready
          WHERE cluster_analysis_eligible_flag = 1
        ),
        summary AS (
          SELECT
            o.teacher_unique_id,
            MAX(CASE WHEN e.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END) AS cluster_analysis_eligible_flag,
            SUM(o.active_user_flag) AS teacher_active_months_total,
            SUM(o.strict_value_flag) AS teacher_strict_months_total,
            AVG(o.active_user_flag) AS teacher_active_month_share,
            AVG(o.strict_value_flag) AS teacher_strict_month_share,
            AVG(e.activity_events_month) AS avg_activity_events_active_month,
            STDDEV_POP(e.activity_events_month) AS std_activity_events_active_month,
            AVG(e.active_days_month) AS avg_active_days_active_month,
            STDDEV_POP(e.active_days_month) AS std_active_days_active_month,
            AVG(e.strict_download_count_month) AS avg_strict_downloads_active_month,
            STDDEV_POP(e.strict_download_count_month) AS std_strict_downloads_active_month,
            AVG(e.download_count_month) AS avg_downloads_active_month,
            AVG(e.content_views_month) AS avg_content_views_active_month,
            AVG(e.other_activity_non_download_events_month) AS avg_other_actions_active_month,
            AVG(e.aula_events_month) AS avg_aula_events_active_month,
            AVG(e.plano_events_month) AS avg_plano_events_active_month,
            AVG(e.prova_events_month) AS avg_prova_events_active_month,
            AVG(e.ia_events_month) AS avg_ia_events_active_month,
            AVG(e.mapped_lessons_month) AS avg_mapped_lessons_active_month,
            AVG(e.clean_entry_session_count_month) AS avg_clean_entry_sessions_active_month,
            AVG(e.clean_entry_total_session_minutes_month) AS avg_clean_entry_minutes_active_month,
            AVG(o.clean_entry_signal_flag) AS teacher_clean_entry_coverage_share,
            AVG(o.only_ping_entry_flag) AS teacher_only_ping_month_share,
            AVG(CASE WHEN o.interaction_signal_flag = 1 AND o.entry_signal_flag = 0 THEN 1 ELSE 0 END) AS teacher_interaction_without_entry_share,
            AVG(o.used_mobile_flag) AS teacher_mobile_month_share,
            AVG(o.used_desktop_flag) AS teacher_desktop_month_share
          FROM observed o
          LEFT JOIN eligible e
            ON o.teacher_unique_id = e.teacher_unique_id
           AND o.month = e.month
          GROUP BY 1
        )
        SELECT
          s.*,
          dt.teacher_estado,
          dt.teacher_currentsubject_group,
          dt.teacher_currentstage,
          dt.teacher_utm_group
        FROM summary s
        LEFT JOIN dim_teacher dt
          ON s.teacher_unique_id = dt.teacher_unique_id
        ORDER BY s.teacher_unique_id
        """
    )


def _create_mart_teacher_persona_ready(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_persona_ready AS
        WITH observed AS (
          SELECT *
          FROM mart_teacher_month_persona_ready
          WHERE observed_month_flag = 1
        ),
        eligible AS (
          SELECT *
          FROM mart_teacher_month_persona_ready
          WHERE persona_analysis_eligible_flag = 1
        ),
        rates AS (
          SELECT
            teacher_unique_id,
            AVG(returned_active_m1) FILTER (WHERE next_month_observed_flag = 1) AS teacher_returned_active_rate_observed,
            AVG(returned_any_download_m1) FILTER (WHERE next_month_observed_flag = 1) AS teacher_returned_download_rate_observed,
            AVG(strict_user_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_strict_user_rate_observed
          FROM mart_teacher_month_persona_ready
          GROUP BY 1
        ),
        summary AS (
          SELECT
            o.teacher_unique_id,
            MAX(o.teacher_population_status) AS teacher_population_status,
            MAX(o.teacher_estado) AS teacher_estado,
            MAX(o.teacher_currentsubject_group) AS teacher_currentsubject_group,
            MAX(o.teacher_currentstage) AS teacher_currentstage,
            MAX(o.teacher_utm_group) AS teacher_utm_group,
            MAX(o.teacher_total_alunos) AS teacher_total_alunos,
            MAX(o.teacher_tipo_total_alunos) AS teacher_tipo_total_alunos,
            MAX(dt.teacher_months_since_last_observed_month_dataset_end) AS teacher_months_since_last_observed_month_dataset_end,
            MAX(o.is_estado_missing) AS is_estado_missing,
            MAX(o.is_currentsubject_missing) AS is_currentsubject_missing,
            MAX(o.is_utm_missing) AS is_utm_missing,
            MAX(o.is_total_alunos_missing) AS is_total_alunos_missing,
            MAX(o.is_tipo_total_alunos_missing) AS is_tipo_total_alunos_missing,
            MIN(o.month) AS teacher_first_observed_month,
            MAX(o.month) AS teacher_last_observed_month,
            COUNT(*) AS teacher_observed_months_total,
            SUM(o.active_user_flag) AS teacher_active_months_total,
            SUM(o.strict_value_flag) AS teacher_strict_months_total,
            SUM(o.persona_analysis_eligible_flag) AS teacher_persona_eligible_months_total,
            AVG(o.active_user_flag) AS teacher_active_month_share,
            AVG(o.strict_value_flag) AS teacher_strict_month_share,
            MAX(CASE WHEN e.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END) AS teacher_persona_analysis_eligible_flag,
            AVG(e.activity_events_month) AS avg_activity_events_active_month,
            STDDEV_POP(e.activity_events_month) AS std_activity_events_active_month,
            AVG(e.active_days_month) AS avg_active_days_active_month,
            STDDEV_POP(e.active_days_month) AS std_active_days_active_month,
            AVG(e.strict_download_count_month) AS avg_strict_downloads_active_month,
            STDDEV_POP(e.strict_download_count_month) AS std_strict_downloads_active_month,
            AVG(e.download_count_month) AS avg_downloads_active_month,
            AVG(e.content_views_month) AS avg_content_views_active_month,
            AVG(e.other_activity_non_download_events_month) AS avg_other_actions_active_month,
            AVG(e.aula_events_month) AS avg_aula_events_active_month,
            AVG(e.plano_events_month) AS avg_plano_events_active_month,
            AVG(e.prova_events_month) AS avg_prova_events_active_month,
            AVG(e.ia_events_month) AS avg_ia_events_active_month,
            AVG(e.mapped_lessons_month) AS avg_mapped_lessons_active_month,
            AVG(e.clean_entry_session_count_month) AS avg_clean_entry_sessions_active_month,
            AVG(e.clean_entry_total_session_minutes_month) AS avg_clean_entry_minutes_active_month,
            AVG(o.clean_entry_signal_flag) AS teacher_clean_entry_coverage_share,
            AVG(o.only_ping_entry_flag) AS teacher_only_ping_month_share,
            AVG(CASE WHEN o.interaction_signal_flag = 1 AND o.entry_signal_flag = 0 THEN 1 ELSE 0 END) AS teacher_interaction_without_entry_share,
            AVG(o.used_mobile_flag) AS teacher_mobile_month_share,
            AVG(o.used_desktop_flag) AS teacher_desktop_month_share,
            AVG(o.no_download_flag) AS teacher_no_download_month_share,
            AVG(o.no_download_view_only_flag) AS teacher_view_only_no_download_month_share,
            AVG(o.no_download_view_plus_action_flag) AS teacher_view_plus_action_no_download_month_share,
            AVG(o.no_download_action_only_flag) AS teacher_action_only_no_download_month_share,
            MAX(o.active_streak_max_months) AS teacher_active_streak_max_months,
            MAX(o.strict_streak_max_months) AS teacher_strict_streak_max_months
          FROM observed o
          LEFT JOIN eligible e
            ON o.teacher_unique_id = e.teacher_unique_id
           AND o.month = e.month
          LEFT JOIN dim_teacher dt
            ON o.teacher_unique_id = dt.teacher_unique_id
          GROUP BY 1
        )
        SELECT
          s.*,
          r.teacher_returned_active_rate_observed,
          r.teacher_returned_download_rate_observed,
          r.teacher_strict_user_rate_observed
        FROM summary s
        LEFT JOIN rates r
          ON s.teacher_unique_id = r.teacher_unique_id
        ORDER BY s.teacher_unique_id
        """
    )


def _build_audit_persona_feature_readiness(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute("SELECT * FROM mart_teacher_month_persona_ready").fetchdf()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "feature_level",
                "feature_role",
                "definition",
                "missing_rate",
                "zero_share",
                "std",
                "recommended_for_persona_analysis",
                "recommended_for_persona_ranges",
                "recommended_for_behavior_clustering",
                "caveat",
            ]
        )
    candidate_features = {
        "activity_events_month": ("teacher_month", "behavior_core", "quantidade de acoes validas no mes", "Sinal comportamental central para uso."),
        "active_days_month": ("teacher_month", "behavior_core", "dias distintos com atividade no mes", "Regularidade temporal."),
        "strict_download_count_month": ("teacher_month", "behavior_core", "downloads strict no mes", "Sinal principal de valor pedagogico."),
        "download_count_month": ("teacher_month", "behavior_support", "downloads totais no mes", "Versao mais ampla de downloads."),
        "content_views_month": ("teacher_month", "behavior_support", "views de conteudo no mes", "Ajuda a separar consumo passivo."),
        "other_activity_non_download_events_month": ("teacher_month", "behavior_support", "outras acoes sem download no mes", "Acoes nao resumidas por download."),
        "clean_entry_total_session_minutes_month": ("teacher_month", "intensity", "minutos de sessao limpa no mes", "Intensidade temporal de uso."),
        "clean_entry_session_count_month": ("teacher_month", "intensity", "sessoes limpas no mes", "Frequencia de retorno as sessoes."),
    }
    rows: list[dict[str, Any]] = []
    for feature_name, (feature_level, feature_role, definition, caveat) in candidate_features.items():
        values = pd.to_numeric(df.get(feature_name), errors="coerce")
        missing_rate = float(values.isna().mean()) if len(values) else float("nan")
        zero_share = float((values.fillna(0) == 0).mean()) if len(values) else float("nan")
        std = float(values.std(ddof=0)) if len(values) else float("nan")
        recommended = int(np.isfinite(std) and std > 0 and missing_rate <= 0.4)
        rows.append(
            {
                "feature_name": feature_name,
                "feature_level": feature_level,
                "feature_role": feature_role,
                "definition": definition,
                "missing_rate": missing_rate,
                "zero_share": zero_share,
                "std": std,
                "recommended_for_persona_analysis": recommended,
                "recommended_for_persona_ranges": recommended,
                "recommended_for_behavior_clustering": recommended,
                "caveat": caveat,
            }
        )
    return pd.DataFrame(rows)


def _build_dim_persona_range_candidates(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    audit = conn.execute("SELECT * FROM audit_persona_feature_readiness").fetchdf()
    data = conn.execute(
        """
        SELECT *
        FROM mart_teacher_month_persona_ready
        WHERE persona_analysis_eligible_flag = 1
        """
    ).fetchdf()
    rows: list[dict[str, Any]] = []
    for feature_name in audit.loc[audit["recommended_for_persona_ranges"] == 1, "feature_name"].tolist():
        values = pd.to_numeric(data.get(feature_name), errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "feature_name": feature_name,
                "feature_level": "teacher_month",
                "population_used": "meses elegiveis para persona",
                "n_rows": int(len(values)),
                "min_value": float(values.min()),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "p50": float(values.quantile(0.50)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "max_value": float(values.max()),
                "zero_share": float((values == 0).mean()),
                "note": "Usar faixas data-driven; nao congelar cortes arbitrarios sem validar estabilidade.",
            }
        )
    return pd.DataFrame(rows)


def _build_base_modelada_validation(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_check(check_name: str, metric_value: Any, status: str, note: str) -> None:
        rows.append(
            {
                "check_name": check_name,
                "metric_value": metric_value,
                "status": status,
                "note": note,
            }
        )

    def scalar(query: str) -> Any:
        value = conn.execute(query).fetchone()[0]
        return value

    def null_cell_count(table_name: str) -> int:
        columns = [row[0] for row in conn.execute(f"DESCRIBE {table_name}").fetchall()]
        if not columns:
            return 0
        expr = " + ".join([f"SUM(CASE WHEN \"{column}\" IS NULL THEN 1 ELSE 0 END)" for column in columns])
        return int(scalar(f"SELECT {expr} FROM {table_name}") or 0)

    base_rows = int(scalar("SELECT COUNT(*) FROM base_modelada_v2") or 0)
    fact_rows = int(scalar("SELECT COUNT(*) FROM fct_teacher_month WHERE month IS NOT NULL") or 0)
    distinct_teachers = int(scalar("SELECT COUNT(*) FROM dim_teacher") or 0)
    grain_duplicates = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM (
              SELECT teacher_unique_id, month, COUNT(*) AS dup_count
              FROM base_modelada_v2
              GROUP BY 1, 2
              HAVING COUNT(*) > 1
            )
            """
        )
        or 0
    )
    missing_teacher = int(scalar("SELECT COUNT(*) FROM base_modelada_v2 WHERE teacher_unique_id IS NULL") or 0)
    missing_month = int(scalar("SELECT COUNT(*) FROM base_modelada_v2 WHERE month IS NULL") or 0)
    join_gap = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM fct_teacher_month tm
            LEFT JOIN dim_teacher dt USING(teacher_unique_id)
            WHERE tm.month IS NOT NULL
              AND dt.teacher_unique_id IS NULL
            """
        )
        or 0
    )
    active_reconcile = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE CAST(active_user_flag AS INTEGER) <> CASE WHEN coalesce(activity_events_month, 0) > 0 THEN 1 ELSE 0 END
            """
        )
        or 0
    )
    strict_reconcile = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE CAST(strict_value_flag AS INTEGER) <> CASE WHEN coalesce(strict_download_count_month, 0) > 0 THEN 1 ELSE 0 END
            """
        )
        or 0
    )
    session_sum_diff = float(
        scalar(
            """
            SELECT ABS(
              coalesce((SELECT SUM(clean_entry_session_count_month) FROM base_modelada_v2), 0)
              - coalesce((SELECT SUM(clean_entry_session_count_month) FROM fct_teacher_month WHERE month IS NOT NULL), 0)
            )
            """
        )
        or 0.0
    )
    strict_download_sum_diff = float(
        scalar(
            """
            SELECT ABS(
              coalesce((SELECT SUM(strict_download_count_month) FROM base_modelada_v2), 0)
              - coalesce((SELECT SUM(strict_download_count_month) FROM fct_teacher_month WHERE month IS NOT NULL), 0)
            )
            """
        )
        or 0.0
    )
    null_ts_excluded = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM raw_interactions i
            INNER JOIN raw_dim_teachers d USING(unique_id)
            WHERE lower(coalesce(i.user_type, '')) = 'registered'
              AND TRY_CAST(i.data_inicio AS TIMESTAMP) IS NULL
            """
        )
        or 0
    )
    base_no_nulls = null_cell_count("base_modelada_v2")
    clean_interaction_null_month = int(scalar("SELECT COUNT(*) FROM fct_interaction_clean WHERE interaction_month IS NULL") or 0)
    clean_session_null_core = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM fct_session_clean
            WHERE session_month IS NULL
               OR session_start_ts IS NULL
               OR session_end_ts IS NULL
               OR duration_min IS NULL
            """
        )
        or 0
    )
    base_any_signal_gap = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE CAST(any_signal_flag AS INTEGER) <> 1
            """
        )
        or 0
    )
    strict_requires_interaction_gap = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE coalesce(strict_download_count_month, 0) > 0
              AND CAST(interaction_signal_flag AS INTEGER) = 0
            """
        )
        or 0
    )
    invalid_negative_counts = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE (CAST(is_total_alunos_negative AS INTEGER) = 1 AND teacher_total_alunos <> -1)
               OR (CAST(is_alunos_diretos_negative AS INTEGER) = 1 AND teacher_alunos_diretos <> -1)
               OR (CAST(is_alunos_indiretos_negative AS INTEGER) = 1 AND teacher_alunos_indiretos <> -1)
            """
        )
        or 0
    )
    dim_teacher_no_nulls = null_cell_count("dim_teacher")
    other_relevant_no_nulls = sum(
        null_cell_count(table_name)
        for table_name in [
            "fct_session_clean",
            "fct_interaction_clean",
            "fct_formation_clean",
            "fct_mari_conversation_resolved",
            "fct_mari_reports_resolved",
            "fct_mari_help_resolved",
            "dim_lesson",
        ]
    )
    auxiliary_no_nulls = sum(
        null_cell_count(table_name)
        for table_name in [
            "dim_event",
            "dim_device",
            "dim_calendar",
            "bridge_mari_conversation_teacher",
            "bridge_teacher_identity_audit",
            "audit_persona_feature_readiness",
            "dim_persona_range_candidates",
        ]
    )
    fact_no_nulls = null_cell_count("fct_teacher_month")
    panel_cover_gap = int(
        scalar(
            """
            SELECT ABS(
              coalesce((SELECT COUNT(*) FROM mart_teacher_month_panel WHERE observed_month_flag = 1), 0)
              - coalesce((SELECT COUNT(*) FROM base_modelada_v2), 0)
            )
            """
        )
        or 0
    )
    panel_no_signal_behavior_gap = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM mart_teacher_month_panel
            WHERE no_signal_month_flag = 1
              AND (
                coalesce(any_signal_flag, 0) > 0
                OR coalesce(activity_events_month, 0) > 0
                OR coalesce(download_count_month, 0) > 0
                OR coalesce(content_views_month, 0) > 0
                OR coalesce(other_activity_non_download_events_month, 0) > 0
                OR coalesce(raw_entry_session_count_month, 0) > 0
                OR coalesce(ping_entry_session_count_month, 0) > 0
                OR coalesce(clean_entry_session_count_month, 0) > 0
                OR coalesce(active_user_flag, 0) > 0
                OR coalesce(strict_value_flag, 0) > 0
              )
            """
        )
        or 0
    )
    persona_teacher_month_gap = int(
        scalar("SELECT ABS((SELECT COUNT(*) FROM mart_teacher_month_persona_ready) - (SELECT COUNT(*) FROM base_modelada_v2))")
        or 0
    )
    persona_teacher_gap = int(
        scalar("SELECT ABS((SELECT COUNT(*) FROM mart_teacher_persona_ready) - (SELECT COUNT(*) FROM dim_teacher))") or 0
    )
    cluster_teacher_month_gap = int(
        scalar("SELECT ABS((SELECT COUNT(*) FROM mart_teacher_month_cluster_ready) - (SELECT COUNT(*) FROM base_modelada_v2))")
        or 0
    )
    cluster_teacher_gap = int(
        scalar("SELECT ABS((SELECT COUNT(*) FROM mart_teacher_cluster_ready) - (SELECT COUNT(*) FROM dim_teacher))") or 0
    )
    persona_eligible_gap = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM mart_teacher_month_persona_ready
            WHERE CAST(persona_analysis_eligible_flag AS INTEGER)
                <> CASE WHEN coalesce(active_user_flag, 0) > 0 THEN 1 ELSE 0 END
            """
        )
        or 0
    )
    teacher_persona_eligible_gap = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM mart_teacher_persona_ready
            WHERE CAST(teacher_persona_analysis_eligible_flag AS INTEGER)
                <> CASE WHEN coalesce(teacher_active_months_total, 0) > 0 THEN 1 ELSE 0 END
            """
        )
        or 0
    )
    dim_lesson_invalid_gap = int(
        scalar(
            """
            SELECT COUNT(*)
            FROM dim_lesson
            WHERE NOT regexp_matches(coalesce(lesson_id, ''), '^[A-Za-z0-9]{22}$')
            """
        )
        or 0
    )
    lesson_like_match_rate = float(
        scalar(
            """
            SELECT 100.0 * AVG(CASE WHEN lesson_mapped_flag = 1 THEN 1.0 ELSE 0.0 END)
            FROM fct_interaction_clean
            WHERE lesson_join_allowed = 1
            """
        )
        or 0.0
    )
    raw_lessons_nonstandard_rows = int(
        scalar(
            f"""
            SELECT COUNT(*)
            FROM raw_lessons
            WHERE NOT regexp_matches(coalesce(id_aula, ''), '{VALID_LESSON_ID_RE}')
            """
        )
        or 0
    )
    add_check(
        "distinct_teachers_positive",
        distinct_teachers,
        "pass" if distinct_teachers > 0 else "fail",
        "A base modelada deve conter professores distintos.",
    )
    add_check(
        "row_count_matches_fct_teacher_month",
        base_rows - fact_rows,
        "pass" if base_rows == fact_rows else "fail",
        "A base modelada deve ter exatamente 1 linha por teacher-month valido do fato mensal.",
    )
    add_check(
        "grain_teacher_month_unique",
        grain_duplicates,
        "pass" if grain_duplicates == 0 else "fail",
        "A chave (teacher_unique_id, month) deve ser unica na base modelada.",
    )
    add_check(
        "missing_teacher_unique_id",
        missing_teacher,
        "pass" if missing_teacher == 0 else "fail",
        "A base modelada nao pode perder a chave do professor.",
    )
    add_check(
        "missing_month",
        missing_month,
        "pass" if missing_month == 0 else "fail",
        "A base modelada nao pode conter linhas sem mes.",
    )
    add_check(
        "dim_teacher_join_gap",
        join_gap,
        "pass" if join_gap == 0 else "fail",
        "Todo teacher-month modelado precisa encontrar exatamente 1 teacher na dimensao.",
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
        "clean_entry_count_sum_diff_vs_fact",
        session_sum_diff,
        "pass" if session_sum_diff == 0 else "fail",
        "A soma de clean entries da base modelada deve bater exatamente com fct_teacher_month.",
    )
    add_check(
        "strict_download_sum_diff_vs_fact",
        strict_download_sum_diff,
        "pass" if strict_download_sum_diff == 0 else "fail",
        "A soma de strict downloads da base modelada deve bater exatamente com fct_teacher_month.",
    )
    add_check(
        "base_modelada_has_no_nulls",
        base_no_nulls,
        "pass" if base_no_nulls == 0 else "fail",
        "A base modelada final deve exportar sem nulls; missings ficam explicitados por flags ou sentinelas.",
    )
    add_check(
        "clean_interactions_have_no_null_month",
        clean_interaction_null_month,
        "pass" if clean_interaction_null_month == 0 else "fail",
        "fct_interaction_clean nao pode ter interaction_month nulo.",
    )
    add_check(
        "clean_sessions_have_no_null_core_fields",
        clean_session_null_core,
        "pass" if clean_session_null_core == 0 else "fail",
        "fct_session_clean nao pode manter session_month, timestamps ou duracao nulos.",
    )
    add_check(
        "base_any_signal_flag_reconciles",
        base_any_signal_gap,
        "pass" if base_any_signal_gap == 0 else "fail",
        "Na base observada, toda linha precisa representar algum sinal observado no mês.",
    )
    add_check(
        "strict_download_requires_interaction_signal",
        strict_requires_interaction_gap,
        "pass" if strict_requires_interaction_gap == 0 else "fail",
        "Download strict só pode existir em mês com sinal de interaction.",
    )
    add_check(
        "base_invalid_negative_counts_normalized",
        invalid_negative_counts,
        "pass" if invalid_negative_counts == 0 else "fail",
        "Campos numericos de cadastro invalidos precisam ser normalizados para a sentinela -1 na base final.",
    )
    add_check(
        "dim_teacher_final_has_no_nulls",
        dim_teacher_no_nulls,
        "pass" if dim_teacher_no_nulls == 0 else "fail",
        "A dimensao final relevante deve sair sem nulls.",
    )
    add_check(
        "other_relevant_tables_have_no_nulls",
        other_relevant_no_nulls,
        "pass" if other_relevant_no_nulls == 0 else "fail",
        "As demais tabelas relevantes exportadas tambem devem sair sem nulls SQL.",
    )
    add_check(
        "auxiliary_tables_have_no_nulls",
        auxiliary_no_nulls,
        "pass" if auxiliary_no_nulls == 0 else "fail",
        "As tabelas auxiliares exportadas tambem devem sair sem nulls SQL; ausencias ficam explicitadas por sentinelas e flags.",
    )
    add_check(
        "fct_teacher_month_final_has_no_nulls",
        fact_no_nulls,
        "pass" if fact_no_nulls == 0 else "fail",
        "O fato mensal final relevante deve sair sem nulls.",
    )
    add_check(
        "panel_rows_cover_observed_base",
        panel_cover_gap,
        "pass" if panel_cover_gap == 0 else "fail",
        "O painel densificado deve conter exatamente todas as linhas observadas da base no subconjunto observed_month_flag=1.",
    )
    add_check(
        "panel_no_signal_has_no_behavior",
        panel_no_signal_behavior_gap,
        "pass" if panel_no_signal_behavior_gap == 0 else "fail",
        "Meses no_signal do painel nao podem carregar comportamento, download ou sinais observados.",
    )
    add_check(
        "persona_teacher_month_rows_match_base",
        persona_teacher_month_gap,
        "pass" if persona_teacher_month_gap == 0 else "fail",
        "A mart mensal de personas deve cobrir exatamente as mesmas linhas da base_modelada_v2.",
    )
    add_check(
        "persona_teacher_rows_match_dim_teacher",
        persona_teacher_gap,
        "pass" if persona_teacher_gap == 0 else "fail",
        "A mart de personas por professor deve cobrir os mesmos professores da dim_teacher.",
    )
    add_check(
        "cluster_teacher_month_rows_match_base",
        cluster_teacher_month_gap,
        "pass" if cluster_teacher_month_gap == 0 else "fail",
        "A mart mensal cluster_ready deve cobrir exatamente as mesmas linhas observadas da base_modelada_v2.",
    )
    add_check(
        "cluster_teacher_rows_match_dim_teacher",
        cluster_teacher_gap,
        "pass" if cluster_teacher_gap == 0 else "fail",
        "A mart cluster_ready por professor deve cobrir os mesmos professores da dim_teacher.",
    )
    add_check(
        "persona_eligible_flag_reconciles",
        persona_eligible_gap,
        "pass" if persona_eligible_gap == 0 else "fail",
        "persona_analysis_eligible_flag deve refletir active_user_flag no grao teacher-month.",
    )
    add_check(
        "teacher_persona_eligible_flag_reconciles",
        teacher_persona_eligible_gap,
        "pass" if teacher_persona_eligible_gap == 0 else "fail",
        "teacher_persona_analysis_eligible_flag deve refletir existencia de ao menos um mes ativo.",
    )
    add_check(
        "dim_lesson_contains_only_valid_observed_ids",
        dim_lesson_invalid_gap,
        "pass" if dim_lesson_invalid_gap == 0 else "fail",
        "A dim_lesson final deve conter apenas ids semanticamente validos.",
    )
    add_check(
        "lesson_like_match_rate_info",
        lesson_like_match_rate,
        "pass",
        "Cobertura de join de lessons considerando apenas ids de aula semanticamente validos.",
    )
    add_check(
        "raw_lessons_nonstandard_rows_info",
        raw_lessons_nonstandard_rows,
        "pass",
        "Quantidade de ids nao padronizados existente no stg_lessons bruto; esses ids nao entram na dim_lesson final.",
    )
    add_check(
        "registered_matched_interactions_with_null_timestamp_excluded",
        null_ts_excluded,
        "pass",
        "Interacoes registered com match e data_inicio nula sao excluidas da base modelada por nao terem mes confiavel.",
    )
    return pd.DataFrame(rows)


def rebuild_modelled_from_raw(spec: AnalysisSpec, paths: ProjectPaths) -> dict[str, Any]:
    raw_dir = _resolve_raw_source_dir(spec)
    if raw_dir is None:
        raise FileNotFoundError(
            f"Could not find raw source files under dataset_root={Path(spec.data.dataset_root).resolve()}"
        )
    dataset_root = resolve_dataset_root(Path(spec.data.dataset_root))
    paths.modelled_parquet_dir.mkdir(parents=True, exist_ok=True)
    paths.modelled_duckdb.parent.mkdir(parents=True, exist_ok=True)
    if paths.modelled_parquet_dir.exists():
        shutil.rmtree(paths.modelled_parquet_dir)
    paths.modelled_parquet_dir.mkdir(parents=True, exist_ok=True)
    if paths.modelled_duckdb.exists():
        paths.modelled_duckdb.unlink()
    conn = duckdb.connect(str(paths.modelled_duckdb))
    conn.execute("PRAGMA threads=4")
    try:
        _register_raw_views(conn, raw_dir)
        bridge_mari = _create_bridge_mari_conversation_teacher(conn)
        _persist_table(conn, paths, "bridge_mari_conversation_teacher", bridge_mari)
        _create_bridge_teacher_identity_audit(conn)
        _persist_table(conn, paths, "bridge_teacher_identity_audit")
        _persist_table(conn, paths, "dim_lesson", _create_dim_lesson(conn))
        _persist_table(conn, paths, "dim_event", _create_dim_event(conn))
        _persist_table(conn, paths, "dim_device", _create_dim_device())
        _persist_table(conn, paths, "dim_calendar", _create_dim_calendar(conn))
        _create_fct_session_tables(conn)
        _persist_table(conn, paths, "fct_session_raw")
        _persist_table(conn, paths, "fct_session_clean")
        _create_fct_interaction_clean(conn)
        _persist_table(conn, paths, "fct_interaction_clean")
        _create_fct_formation_clean(conn)
        _persist_table(conn, paths, "fct_formation_clean")
        _create_fct_mari_conversation_resolved(conn)
        _persist_table(conn, paths, "fct_mari_conversation_resolved")
        _create_fct_mari_reports_resolved(conn)
        _persist_table(conn, paths, "fct_mari_reports_resolved")
        _create_fct_mari_help_resolved(conn)
        _persist_table(conn, paths, "fct_mari_help_resolved")
        _persist_table(conn, paths, "fct_teacher_month", _build_teacher_month(conn))
        _persist_table(conn, paths, "dim_teacher", _create_dim_teacher(conn))
        _create_base_modelada(conn)
        _persist_table(conn, paths, "base_modelada_v2")
        _create_mart_teacher_month_panel(conn)
        _persist_table(conn, paths, "mart_teacher_month_panel")
        _create_mart_teacher_month_cluster_ready(conn)
        _persist_table(conn, paths, "mart_teacher_month_cluster_ready")
        _create_mart_teacher_month_persona_ready(conn)
        _persist_table(conn, paths, "mart_teacher_month_persona_ready")
        _create_mart_teacher_cluster_ready(conn)
        _persist_table(conn, paths, "mart_teacher_cluster_ready")
        _create_mart_teacher_persona_ready(conn)
        _persist_table(conn, paths, "mart_teacher_persona_ready")
        pre_audit_tables = [
            "bridge_mari_conversation_teacher",
            "bridge_teacher_identity_audit",
            "dim_lesson",
            "dim_event",
            "dim_device",
            "dim_calendar",
            "fct_session_raw",
            "fct_session_clean",
            "fct_interaction_clean",
            "fct_formation_clean",
            "fct_mari_conversation_resolved",
            "fct_mari_reports_resolved",
            "fct_mari_help_resolved",
            "fct_teacher_month",
            "dim_teacher",
            "base_modelada_v2",
            "mart_teacher_month_panel",
            "mart_teacher_month_cluster_ready",
            "mart_teacher_month_persona_ready",
            "mart_teacher_cluster_ready",
            "mart_teacher_persona_ready",
        ]
        _normalize_modelled_tables(conn, pre_audit_tables)
        _normalize_teacher_negative_sentinels(
            conn,
            [
                "dim_teacher",
                "base_modelada_v2",
                "mart_teacher_month_persona_ready",
                "mart_teacher_persona_ready",
            ],
        )
        for table_name in pre_audit_tables:
            _persist_table(conn, paths, table_name)
        _persist_table(conn, paths, "audit_persona_feature_readiness", _build_audit_persona_feature_readiness(conn))
        _persist_table(conn, paths, "dim_persona_range_candidates", _build_dim_persona_range_candidates(conn))
        _normalize_modelled_tables(conn, ["audit_persona_feature_readiness", "dim_persona_range_candidates"])
        _persist_table(conn, paths, "audit_persona_feature_readiness")
        _persist_table(conn, paths, "dim_persona_range_candidates")
        _persist_table(conn, paths, "audit_base_modelada_validation", _build_base_modelada_validation(conn))
        table_counts = {
            table_name: int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] or 0)
            for table_name in MODELED_TABLES
        }
    finally:
        conn.close()
    summary = {
        "dataset_root": str(dataset_root),
        "modeled_source": "raw",
        "raw_source_dir": str(raw_dir),
        "raw_files_used": REQUIRED_RAW_FILES,
        "modelled_tables_materialized": table_counts,
        "modelled_duckdb": str(paths.modelled_duckdb),
        "modelled_parquet_dir": str(paths.modelled_parquet_dir),
        "build_mode": "raw_to_modelled_rebuild",
    }
    write_json(paths.modelled_dir / "modelled_build_summary_v1.json", summary)
    return summary
