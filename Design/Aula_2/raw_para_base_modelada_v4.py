from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("raw_para_base_modelada_v2")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
VALID_LESSON_ID_RE = r"^[A-Za-z0-9]{22}$"
STRICT_VALUE_EVENTS = ("download_aula", "download_plano_aula")


@dataclass(frozen=True)
class Config:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    duckdb_path: Path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def qi(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arquivo unico: raw -> base_modelada_v2.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir or base_dir / "base_aprendizap").resolve()
    output_dir = (args.output_dir or base_dir / "analysis_output_v2").resolve()
    duckdb_path = output_dir / "duckdb" / "base_modelada_v2.duckdb"
    return Config(
        base_dir=base_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        duckdb_path=duckdb_path,
    )


def ensure_output_dirs(output_dir: Path) -> None:
    for subdir in [
        "audit",
        "parquet",
        "json",
        "duckdb",
        "tabelas_relevantes",
        "tabelas_relevantes/parquet",
        "tabelas_auxiliares",
        "tabelas_auxiliares/parquet",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def connect_duckdb(cfg: Config) -> duckdb.DuckDBPyConnection:
    ensure_output_dirs(cfg.output_dir)
    conn = duckdb.connect(str(cfg.duckdb_path))
    conn.execute("PRAGMA threads=4")
    return conn


def register_raw_views(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    required = [
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
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes: {', '.join(sorted(missing))}")

    conn.execute(
        f"CREATE OR REPLACE VIEW raw_dim_teachers AS SELECT * FROM read_csv('{q(data_dir / 'dim_teachers.csv')}', delim=';', header=true, ignore_errors=true)"
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
            f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_csv_auto('{q(data_dir / fname)}', header=true)"
        )


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
    if "seo org" in lower or "seo organico" in lower or "seo orgânico" in lower:
        return "seo_organico"
    if "landing" in lower:
        return "landing_page"
    if "blog" in lower:
        return "blog"
    if "social" in lower or "midia" in lower or "mídia" in lower:
        return "social"
    if "push" in lower:
        return "push"
    return re.sub(r"[^a-z0-9_]+", "_", lower).strip("_") or "other"


def classify_currentsubject_group(stage: Any, subject: Any) -> str:
    if subject is None or (isinstance(subject, float) and np.isnan(subject)):
        return "missing"
    subject_text = str(subject).strip().lower()
    stage_text = str(stage).strip().lower() if stage is not None else ""
    if subject_text in {"linguagens", "linguagem"}:
        return "Linguagens"
    if subject_text in {"ciencias", "ciências"}:
        return "Ciências da Natureza"
    if subject_text in {"matematica", "matemática"}:
        return "Matemática"
    if subject_text in {"humanas", "ciências humanas", "ciencias humanas"}:
        return "Ciências Humanas"
    code_map = {
        "1": "Linguagens",
        "2": "Matemática",
        "3": "Ciências da Natureza",
        "4": "Ciências Humanas",
        "5": "Linguagens",
        "6": "Matemática",
        "7": "Ciências da Natureza",
        "8": "Ciências Humanas",
    }
    if subject_text in code_map:
        if stage_text in {"em", "ensino_medio"} and subject_text == "1":
            return "Linguagens"
        return code_map[subject_text]
    return "Outras"


def classify_discipline_group(discipline: Any) -> str:
    if discipline is None or (isinstance(discipline, float) and np.isnan(discipline)):
        return "missing"
    lower = str(discipline).strip().lower()
    if not lower:
        return "missing"
    if lower in {"história", "historia", "geografia", "filosofia", "sociologia"}:
        return "Ciências Humanas"
    if lower in {"ciências", "ciencias", "biologia", "química", "quimica", "física", "fisica"}:
        return "Ciências da Natureza"
    if lower in {"matemática", "matematica"}:
        return "Matemática"
    if lower in {"português", "portugues", "inglês", "ingles", "literatura", "redação", "redacao"}:
        return "Linguagens"
    if lower in {"arte", "artes", "educação física", "educacao fisica"}:
        return "Artes e Complementares"
    return "Outras"


def sql_device_expr(column_name: str) -> str:
    return f"""
    CASE
      WHEN {column_name} IS NULL OR trim({column_name})='' THEN 'unknown'
      WHEN lower(trim({column_name})) IN ('desktop', 'mobile', 'tablet') THEN lower(trim({column_name}))
      ELSE 'unknown'
    END
    """


def sql_event_family_expr(column_name: str) -> str:
    return f"""
    CASE
      WHEN {column_name} IS NULL OR trim({column_name})='' THEN 'missing'
      WHEN lower({column_name}) LIKE '%plano%' THEN 'plano'
      WHEN lower({column_name}) LIKE '%prova%' OR lower({column_name}) LIKE '%avaliacao%' THEN 'prova'
      WHEN lower({column_name}) LIKE '%aula%' THEN 'aula'
      WHEN lower({column_name}) LIKE '%ia%' OR lower({column_name}) LIKE '%mari%' THEN 'ia'
      WHEN lower({column_name}) LIKE '%conquista%' THEN 'conquista'
      WHEN lower({column_name}) LIKE '%relatorio%' THEN 'relatorio'
      ELSE 'other'
    END
    """


def sql_event_action_expr(column_name: str) -> str:
    return f"""
    CASE
      WHEN {column_name} IS NULL OR trim({column_name})='' THEN 'missing'
      WHEN lower({column_name}) LIKE '%download%' OR lower({column_name}) LIKE '%baixar%' THEN 'download'
      WHEN lower({column_name}) LIKE '%visualizacao%' THEN 'view'
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


def month_diff(current_month: pd.Timestamp, previous_month: pd.Timestamp) -> int:
    return (current_month.year - previous_month.year) * 12 + (current_month.month - previous_month.month)


def persist_table(conn: duckdb.DuckDBPyConnection, output_dir: Path, table_name: str) -> Dict[str, str]:
    csv_path = output_dir / "csv" / f"{table_name}.csv"
    parquet_path = output_dir / "parquet" / f"{table_name}.parquet"
    return persist_table_to_paths(conn, table_name, csv_path, parquet_path)


def persist_table_to_paths(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    csv_path: Path,
    parquet_path: Path,
    *,
    export_csv: bool = True,
) -> Dict[str, str]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY {table_name} TO '{q(parquet_path)}' (FORMAT PARQUET)")
    written = {"parquet": str(parquet_path)}
    if export_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        conn.execute(f"COPY {table_name} TO '{q(csv_path)}' (HEADER, DELIMITER ',')")
        written["csv"] = str(csv_path)
    return written


def count_table_nulls(conn: duckdb.DuckDBPyConnection, table_name: str) -> int:
    columns = conn.execute(f"DESCRIBE {table_name}").fetchdf()["column_name"].tolist()
    if not columns:
        return 0
    expr = " + ".join([f"COALESCE(SUM(CASE WHEN {qi(col)} IS NULL THEN 1 ELSE 0 END), 0)" for col in columns])
    return int(conn.execute(f"SELECT {expr} AS null_count_total FROM {table_name}").fetchone()[0] or 0)


def reset_generated_export_dirs(output_dir: Path) -> None:
    for parquet_dir in [
        output_dir / "tabelas_relevantes" / "parquet",
        output_dir / "tabelas_auxiliares" / "parquet",
    ]:
        parquet_dir.mkdir(parents=True, exist_ok=True)
        for parquet_file in parquet_dir.glob("*.parquet"):
            parquet_file.unlink()

    for legacy_dir in [
        output_dir / "tabelas_relevantes" / "csv",
        output_dir / "tabelas_auxiliares" / "csv",
    ]:
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)

    for stale_file in [
        output_dir / "tabelas_relevantes" / "tabelas_relevantes_v2.json",
        output_dir / "tabelas_relevantes" / "manifesto_tabelas_relevantes_v2.json",
        output_dir / "tabelas_auxiliares" / "manifesto_tabelas_auxiliares_v2.json",
        output_dir / "parquet" / "base_modelada_v2.parquet",
        output_dir / "parquet" / "audit_base_modelada_validation.parquet",
        output_dir / "csv" / "base_modelada_v2.csv",
        output_dir / "csv" / "audit_base_modelada_validation.csv",
    ]:
        if stale_file.exists():
            stale_file.unlink()


def numeric_feature_stats(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    missing_rate = float(numeric.isna().mean())
    filled = numeric.fillna(0)
    zero_share = float((filled == 0).mean())
    std = float(filled.std(ddof=0))
    return {
        "missing_rate": missing_rate,
        "zero_share": zero_share,
        "std": std,
    }


def create_persona_ready_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_month_persona_ready_final_v2 AS
        SELECT
          teacher_unique_id,
          month,
          month_num,
          1 AS observed_month_flag,
          active_user_flag AS persona_analysis_eligible_flag,
          active_user_flag,
          strict_value_flag,
          strict_user_flag,
          returned_active_m1,
          returned_any_download_m1,
          returned_strict_value_m1,
          strict_return_value_m1,
          next_month_observed_flag,
          interaction_rows_month,
          activity_events_month,
          active_days_month,
          raw_entry_session_count_month,
          ping_entry_session_count_month,
          clean_entry_session_count_month,
          clean_entry_total_session_minutes_month,
          clean_entry_avg_session_minutes_month,
          strict_download_count_month,
          download_count_month,
          download_aula_count_month,
          download_plano_count_month,
          content_views_month,
          other_activity_non_download_events_month,
          aula_events_month,
          plano_events_month,
          prova_events_month,
          ia_events_month,
          mapped_lessons_month,
          interaction_signal_flag,
          entry_signal_flag,
          clean_entry_signal_flag,
          only_ping_entry_flag,
          any_signal_flag,
          month_signal_class,
          used_desktop_flag,
          used_mobile_flag,
          used_ia_flag,
          no_download_flag,
          no_download_view_only_flag,
          no_download_view_plus_action_flag,
          no_download_action_only_flag,
          clean_entry_exposed_no_download_flag,
          clean_entry_exposed_no_activity_no_download_flag,
          clean_entry_exposed_activity_no_download_flag,
          lifetime_active_months,
          lifetime_clean_entry_minutes_total,
          active_streak_current_months,
          active_streak_max_months,
          strict_streak_current_months,
          strict_streak_max_months,
          teacher_population_status,
          teacher_estado,
          teacher_currentsubject_group,
          teacher_currentstage,
          teacher_utm_group,
          teacher_total_alunos,
          teacher_tipo_total_alunos,
          is_estado_missing,
          is_currentsubject_missing,
          is_utm_missing,
          is_total_alunos_missing,
          is_tipo_total_alunos_missing
        FROM base_modelada_v2
        ORDER BY teacher_unique_id, month
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_persona_ready_final_v2 AS
        WITH month_base AS (
          SELECT *
          FROM mart_teacher_month_persona_ready_final_v2
        ),
        agg AS (
          SELECT
            teacher_unique_id,
            MIN(month) AS teacher_first_observed_month,
            MAX(month) AS teacher_last_observed_month,
            COUNT(*) AS teacher_observed_months_total,
            SUM(active_user_flag) AS teacher_active_months_total,
            SUM(strict_value_flag) AS teacher_strict_months_total,
            SUM(persona_analysis_eligible_flag) AS teacher_persona_eligible_months_total,
            AVG(activity_events_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_activity_events_active_month,
            STDDEV_POP(activity_events_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS std_activity_events_active_month,
            AVG(active_days_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_active_days_active_month,
            STDDEV_POP(active_days_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS std_active_days_active_month,
            AVG(strict_download_count_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_strict_downloads_active_month,
            STDDEV_POP(strict_download_count_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS std_strict_downloads_active_month,
            AVG(download_count_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_downloads_active_month,
            AVG(content_views_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_content_views_active_month,
            AVG(other_activity_non_download_events_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_other_actions_active_month,
            AVG(aula_events_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_aula_events_active_month,
            AVG(plano_events_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_plano_events_active_month,
            AVG(prova_events_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_prova_events_active_month,
            AVG(ia_events_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_ia_events_active_month,
            AVG(mapped_lessons_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_mapped_lessons_active_month,
            AVG(clean_entry_session_count_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_clean_entry_sessions_active_month,
            AVG(clean_entry_total_session_minutes_month) FILTER (WHERE persona_analysis_eligible_flag = 1) AS avg_clean_entry_minutes_active_month,
            AVG(clean_entry_signal_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_clean_entry_coverage_share,
            AVG(only_ping_entry_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_only_ping_month_share,
            AVG(CASE WHEN interaction_signal_flag = 1 AND entry_signal_flag = 0 THEN 1 ELSE 0 END) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_interaction_without_entry_share,
            AVG(used_mobile_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_mobile_month_share,
            AVG(used_desktop_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_desktop_month_share,
            AVG(no_download_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_no_download_month_share,
            AVG(no_download_view_only_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_view_only_no_download_month_share,
            AVG(no_download_view_plus_action_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_view_plus_action_no_download_month_share,
            AVG(no_download_action_only_flag) FILTER (WHERE persona_analysis_eligible_flag = 1) AS teacher_action_only_no_download_month_share,
            AVG(returned_active_m1) FILTER (WHERE next_month_observed_flag = 1 AND persona_analysis_eligible_flag = 1 AND returned_active_m1 >= 0) AS teacher_returned_active_rate_observed,
            AVG(returned_any_download_m1) FILTER (WHERE next_month_observed_flag = 1 AND persona_analysis_eligible_flag = 1 AND returned_any_download_m1 >= 0) AS teacher_returned_download_rate_observed,
            AVG(strict_user_flag) FILTER (WHERE next_month_observed_flag = 1 AND strict_value_flag = 1 AND strict_user_flag >= 0) AS teacher_strict_user_rate_observed,
            MAX(active_streak_max_months) AS teacher_active_streak_max_months,
            MAX(strict_streak_max_months) AS teacher_strict_streak_max_months
          FROM month_base
          GROUP BY 1
        )
        SELECT
          dt.teacher_unique_id,
          dt.teacher_population_status,
          dt.teacher_estado,
          dt.teacher_currentsubject_group,
          dt.teacher_currentstage,
          dt.teacher_utm_group,
          dt.teacher_total_alunos,
          dt.teacher_tipo_total_alunos,
          dt.teacher_months_since_last_observed_month_dataset_end,
          dt.is_estado_missing,
          dt.is_currentsubject_missing,
          dt.is_utm_missing,
          dt.is_total_alunos_missing,
          dt.is_tipo_total_alunos_missing,
          agg.teacher_first_observed_month,
          agg.teacher_last_observed_month,
          agg.teacher_observed_months_total,
          agg.teacher_active_months_total,
          agg.teacher_strict_months_total,
          agg.teacher_persona_eligible_months_total,
          CASE WHEN agg.teacher_observed_months_total > 0 THEN agg.teacher_active_months_total * 1.0 / agg.teacher_observed_months_total ELSE 0 END AS teacher_active_month_share,
          CASE WHEN agg.teacher_observed_months_total > 0 THEN agg.teacher_strict_months_total * 1.0 / agg.teacher_observed_months_total ELSE 0 END AS teacher_strict_month_share,
          CASE WHEN agg.teacher_active_months_total > 0 THEN 1 ELSE 0 END AS teacher_persona_analysis_eligible_flag,
          COALESCE(agg.avg_activity_events_active_month, 0) AS avg_activity_events_active_month,
          COALESCE(agg.std_activity_events_active_month, 0) AS std_activity_events_active_month,
          COALESCE(agg.avg_active_days_active_month, 0) AS avg_active_days_active_month,
          COALESCE(agg.std_active_days_active_month, 0) AS std_active_days_active_month,
          COALESCE(agg.avg_strict_downloads_active_month, 0) AS avg_strict_downloads_active_month,
          COALESCE(agg.std_strict_downloads_active_month, 0) AS std_strict_downloads_active_month,
          COALESCE(agg.avg_downloads_active_month, 0) AS avg_downloads_active_month,
          COALESCE(agg.avg_content_views_active_month, 0) AS avg_content_views_active_month,
          COALESCE(agg.avg_other_actions_active_month, 0) AS avg_other_actions_active_month,
          COALESCE(agg.avg_aula_events_active_month, 0) AS avg_aula_events_active_month,
          COALESCE(agg.avg_plano_events_active_month, 0) AS avg_plano_events_active_month,
          COALESCE(agg.avg_prova_events_active_month, 0) AS avg_prova_events_active_month,
          COALESCE(agg.avg_ia_events_active_month, 0) AS avg_ia_events_active_month,
          COALESCE(agg.avg_mapped_lessons_active_month, 0) AS avg_mapped_lessons_active_month,
          COALESCE(agg.avg_clean_entry_sessions_active_month, 0) AS avg_clean_entry_sessions_active_month,
          COALESCE(agg.avg_clean_entry_minutes_active_month, 0) AS avg_clean_entry_minutes_active_month,
          COALESCE(agg.teacher_clean_entry_coverage_share, 0) AS teacher_clean_entry_coverage_share,
          COALESCE(agg.teacher_only_ping_month_share, 0) AS teacher_only_ping_month_share,
          COALESCE(agg.teacher_interaction_without_entry_share, 0) AS teacher_interaction_without_entry_share,
          COALESCE(agg.teacher_mobile_month_share, 0) AS teacher_mobile_month_share,
          COALESCE(agg.teacher_desktop_month_share, 0) AS teacher_desktop_month_share,
          COALESCE(agg.teacher_no_download_month_share, 0) AS teacher_no_download_month_share,
          COALESCE(agg.teacher_view_only_no_download_month_share, 0) AS teacher_view_only_no_download_month_share,
          COALESCE(agg.teacher_view_plus_action_no_download_month_share, 0) AS teacher_view_plus_action_no_download_month_share,
          COALESCE(agg.teacher_action_only_no_download_month_share, 0) AS teacher_action_only_no_download_month_share,
          COALESCE(agg.teacher_returned_active_rate_observed, 0) AS teacher_returned_active_rate_observed,
          COALESCE(agg.teacher_returned_download_rate_observed, 0) AS teacher_returned_download_rate_observed,
          COALESCE(agg.teacher_strict_user_rate_observed, 0) AS teacher_strict_user_rate_observed,
          COALESCE(agg.teacher_active_streak_max_months, 0) AS teacher_active_streak_max_months,
          COALESCE(agg.teacher_strict_streak_max_months, 0) AS teacher_strict_streak_max_months
        FROM dim_teacher_final_v2 dt
        INNER JOIN agg USING(teacher_unique_id)
        ORDER BY dt.teacher_unique_id
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_month_panel_final_v2 AS
        WITH dataset_max AS (
          SELECT MAX(month) AS dataset_last_month
          FROM base_modelada_v2
        ),
        teacher_bounds AS (
          SELECT
            teacher_unique_id,
            teacher_first_observed_month,
            dataset_last_month
          FROM dim_teacher_final_v2
          CROSS JOIN dataset_max
          WHERE teacher_first_observed_month IS NOT NULL
        ),
        panel_months AS (
          SELECT
            tb.teacher_unique_id,
            gs.month_value::TIMESTAMP AS month
          FROM teacher_bounds tb
          CROSS JOIN LATERAL generate_series(
            tb.teacher_first_observed_month,
            tb.dataset_last_month,
            INTERVAL '1 month'
          ) AS gs(month_value)
        ),
        joined AS (
          SELECT
            p.teacher_unique_id,
            p.month,
            COALESCE(b.month_num, EXTRACT(year FROM p.month) * 12 + EXTRACT(month FROM p.month))::INTEGER AS month_num,
            CASE WHEN b.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END AS observed_month_flag,
            COALESCE(b.any_signal_flag, 0) AS any_signal_flag,
            COALESCE(b.interaction_signal_flag, 0) AS interaction_signal_flag,
            COALESCE(b.entry_signal_flag, 0) AS entry_signal_flag,
            COALESCE(b.clean_entry_signal_flag, 0) AS clean_entry_signal_flag,
            COALESCE(b.only_ping_entry_flag, 0) AS only_ping_entry_flag,
            COALESCE(b.month_signal_class, 'no_signal') AS month_signal_class,
            COALESCE(b.active_user_flag, 0) AS active_user_flag,
            COALESCE(b.strict_value_flag, 0) AS strict_value_flag,
            COALESCE(b.strict_download_count_month, 0) AS strict_download_count_month,
            COALESCE(b.download_count_month, 0) AS download_count_month,
            COALESCE(b.activity_events_month, 0) AS activity_events_month,
            COALESCE(b.active_days_month, 0) AS active_days_month,
            COALESCE(b.content_views_month, 0) AS content_views_month,
            COALESCE(b.other_activity_non_download_events_month, 0) AS other_activity_non_download_events_month,
            COALESCE(b.aula_events_month, 0) AS aula_events_month,
            COALESCE(b.plano_events_month, 0) AS plano_events_month,
            COALESCE(b.prova_events_month, 0) AS prova_events_month,
            COALESCE(b.ia_events_month, 0) AS ia_events_month,
            COALESCE(b.mapped_lessons_month, 0) AS mapped_lessons_month,
            COALESCE(b.raw_entry_session_count_month, 0) AS raw_entry_session_count_month,
            COALESCE(b.ping_entry_session_count_month, 0) AS ping_entry_session_count_month,
            COALESCE(b.clean_entry_session_count_month, 0) AS clean_entry_session_count_month,
            COALESCE(b.clean_entry_total_session_minutes_month, 0) AS clean_entry_total_session_minutes_month,
            COALESCE(b.clean_entry_avg_session_minutes_month, 0) AS clean_entry_avg_session_minutes_month,
            COALESCE(b.teacher_estado, 'missing') AS teacher_estado,
            COALESCE(b.teacher_currentsubject_group, 'missing') AS teacher_currentsubject_group,
            COALESCE(b.teacher_currentstage, 'missing') AS teacher_currentstage,
            COALESCE(b.teacher_utm_group, 'missing') AS teacher_utm_group
          FROM panel_months p
          LEFT JOIN base_modelada_v2 b
            ON p.teacher_unique_id = b.teacher_unique_id
           AND p.month = b.month
        ),
        final AS (
          SELECT
            *,
            CASE WHEN observed_month_flag = 0 THEN 1 ELSE 0 END AS no_signal_month_flag,
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
          CASE WHEN last_signal_month_num IS NULL THEN -1 ELSE month_num - last_signal_month_num END AS months_since_last_signal,
          CASE WHEN last_active_month_num IS NULL THEN -1 ELSE month_num - last_active_month_num END AS months_since_last_active,
          CASE WHEN last_strict_month_num IS NULL THEN -1 ELSE month_num - last_strict_month_num END AS months_since_last_strict_value,
          teacher_estado,
          teacher_currentsubject_group,
          teacher_currentstage,
          teacher_utm_group
        FROM final
        ORDER BY teacher_unique_id, month
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_month_cluster_ready_final_v2 AS
        SELECT
          teacher_unique_id,
          month,
          month_num,
          persona_analysis_eligible_flag AS cluster_analysis_eligible_flag,
          activity_events_month,
          active_days_month,
          strict_download_count_month,
          download_count_month,
          content_views_month,
          other_activity_non_download_events_month,
          aula_events_month,
          plano_events_month,
          prova_events_month,
          ia_events_month,
          mapped_lessons_month,
          clean_entry_session_count_month,
          clean_entry_total_session_minutes_month,
          clean_entry_avg_session_minutes_month,
          clean_entry_signal_flag,
          only_ping_entry_flag,
          interaction_signal_flag,
          month_signal_class,
          teacher_estado,
          teacher_currentsubject_group,
          teacher_currentstage,
          teacher_utm_group,
          used_desktop_flag,
          used_mobile_flag
        FROM mart_teacher_month_persona_ready_final_v2
        ORDER BY teacher_unique_id, month
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE mart_teacher_cluster_ready_final_v2 AS
        SELECT
          teacher_unique_id,
          teacher_persona_analysis_eligible_flag AS cluster_analysis_eligible_flag,
          teacher_active_months_total,
          teacher_strict_months_total,
          teacher_active_month_share,
          teacher_strict_month_share,
          avg_activity_events_active_month,
          std_activity_events_active_month,
          avg_active_days_active_month,
          std_active_days_active_month,
          avg_strict_downloads_active_month,
          std_strict_downloads_active_month,
          avg_downloads_active_month,
          avg_content_views_active_month,
          avg_other_actions_active_month,
          avg_aula_events_active_month,
          avg_plano_events_active_month,
          avg_prova_events_active_month,
          avg_ia_events_active_month,
          avg_mapped_lessons_active_month,
          avg_clean_entry_sessions_active_month,
          avg_clean_entry_minutes_active_month,
          teacher_clean_entry_coverage_share,
          teacher_only_ping_month_share,
          teacher_interaction_without_entry_share,
          teacher_mobile_month_share,
          teacher_desktop_month_share,
          teacher_estado,
          teacher_currentsubject_group,
          teacher_currentstage,
          teacher_utm_group
        FROM mart_teacher_persona_ready_final_v2
        ORDER BY teacher_unique_id
        """
    )

    month_df = conn.execute("SELECT * FROM mart_teacher_month_persona_ready_final_v2").fetchdf()
    teacher_df = conn.execute("SELECT * FROM mart_teacher_persona_ready_final_v2").fetchdf()

    feature_specs = [
        ("activity_events_month", "teacher_month", "behavior_core", "quantidade de ações válidas no mês", "Sinal comportamental central para uso.", 1, 1),
        ("active_days_month", "teacher_month", "behavior_core", "dias distintos com atividade no mês", "Regularidade temporal.", 1, 1),
        ("strict_download_count_month", "teacher_month", "behavior_core", "downloads strict no mês", "Sinal principal de valor pedagógico.", 1, 1),
        ("download_count_month", "teacher_month", "behavior_core", "todos os downloads no mês", "Apoio para intensidade de download além do strict.", 1, 1),
        ("content_views_month", "teacher_month", "behavior_core", "visualizações de conteúdo no mês", "Exploração de conteúdo.", 1, 1),
        ("other_activity_non_download_events_month", "teacher_month", "behavior_core", "outras ações sem download no mês", "Ação sem download.", 1, 1),
        ("aula_events_month", "teacher_month", "behavior_core", "eventos de aula no mês", "Composição do comportamento.", 0, 1),
        ("plano_events_month", "teacher_month", "behavior_core", "eventos de plano no mês", "Composição do comportamento.", 0, 1),
        ("prova_events_month", "teacher_month", "behavior_core", "eventos de prova no mês", "Composição do comportamento.", 0, 1),
        ("ia_events_month", "teacher_month", "behavior_core", "eventos de IA no mês", "Uso de ferramenta.", 0, 1),
        ("clean_entry_session_count_month", "teacher_month", "session_telemetry_support", "contagem de sessões limpas de entry no mês", "Telemetria auxiliar; não usar como definição primária de acesso.", 0, 0),
        ("clean_entry_total_session_minutes_month", "teacher_month", "session_telemetry_support", "tempo total limpo de entry no mês", "Telemetria auxiliar de duração.", 0, 0),
        ("clean_entry_avg_session_minutes_month", "teacher_month", "session_telemetry_support", "duração média das sessões limpas de entry", "Telemetria auxiliar; cobertura imperfeita de entry.", 0, 0),
        ("only_ping_entry_flag", "teacher_month", "session_telemetry_support", "mês com apenas entry ping", "Importante para diagnosticar inconsistência entre session telemetry e comportamento.", 0, 0),
        ("interaction_signal_flag", "teacher_month", "signal_reconciliation", "há ao menos uma interaction no mês", "Usar para reconciliação entre fontes.", 0, 0),
        ("entry_signal_flag", "teacher_month", "signal_reconciliation", "há ao menos um entry raw no mês", "Usar para reconciliação entre fontes.", 0, 0),
        ("month_signal_class", "teacher_month", "signal_reconciliation", "classe de reconciliação do mês observado", "Explica se o mês foi observado por interaction, entry limpo, ping ou mistura.", 0, 0),
        ("mapped_lessons_month", "teacher_month", "behavior_support", "lessons mapeadas no mês", "Útil só com caveat de lesson mapping.", 0, 0),
        ("used_mobile_flag", "teacher_month", "context_interpretation", "uso mobile no mês", "Contexto de acesso.", 0, 0),
        ("used_desktop_flag", "teacher_month", "context_interpretation", "uso desktop no mês", "Contexto de acesso.", 0, 0),
        ("teacher_estado", "teacher_month", "context_interpretation", "UF do professor", "Interpretar sempre com flag de missing.", 0, 0),
        ("teacher_currentsubject_group", "teacher_month", "context_interpretation", "grupo de disciplina do cadastro", "Pode ajudar a descrever personas.", 0, 0),
        ("teacher_utm_group", "teacher_month", "context_interpretation", "origem padronizada do cadastro", "Canal de entrada, não usar como eixo central.", 0, 0),
        ("avg_activity_events_active_month", "teacher", "behavior_core", "média de ações válidas nos meses ativos", "Intensidade média de comportamento.", 1, 1),
        ("avg_active_days_active_month", "teacher", "behavior_core", "média de dias ativos nos meses ativos", "Regularidade média.", 1, 1),
        ("avg_strict_downloads_active_month", "teacher", "behavior_core", "downloads strict médios nos meses ativos", "Valor pedagógico médio.", 1, 1),
        ("avg_downloads_active_month", "teacher", "behavior_core", "downloads médios nos meses ativos", "Intensidade média de download.", 1, 1),
        ("avg_content_views_active_month", "teacher", "behavior_core", "visualizações médias nos meses ativos", "Exploração média de conteúdo.", 1, 1),
        ("avg_other_actions_active_month", "teacher", "behavior_core", "outras ações médias nos meses ativos", "Ação média sem download.", 1, 1),
        ("avg_clean_entry_sessions_active_month", "teacher", "session_telemetry_support", "média de sessões limpas de entry nos meses ativos", "Telemetria auxiliar; não usar como frequência principal se houver conflito de cobertura.", 0, 0),
        ("avg_clean_entry_minutes_active_month", "teacher", "session_telemetry_support", "média de minutos limpos de entry nos meses ativos", "Telemetria auxiliar de duração.", 0, 0),
        ("teacher_clean_entry_coverage_share", "teacher", "session_telemetry_support", "share de meses ativos com entry limpo observado", "Mede cobertura de telemetria de sessão.", 0, 0),
        ("teacher_only_ping_month_share", "teacher", "session_telemetry_support", "share de meses ativos com apenas entry ping", "Diagnóstico de meses comportamentais sem sessão limpa.", 0, 0),
        ("teacher_interaction_without_entry_share", "teacher", "signal_reconciliation", "share de meses ativos com interaction sem entry", "Cobertura relativa entre fontes.", 0, 0),
        ("teacher_active_month_share", "teacher", "behavior_core", "share de meses ativos", "Persistência de uso.", 1, 1),
        ("teacher_strict_month_share", "teacher", "behavior_core", "share de meses com strict_value", "Persistência de valor.", 1, 1),
        ("teacher_returned_active_rate_observed", "teacher", "outcome_evaluation", "taxa de retorno ativo observada", "Usar para avaliar personas, não para formá-las.", 0, 0),
        ("teacher_returned_download_rate_observed", "teacher", "outcome_evaluation", "taxa de retorno com download observada", "Usar para avaliar personas, não para formá-las.", 0, 0),
        ("teacher_estado", "teacher", "context_interpretation", "UF do professor", "Interpretar sempre com flag de missing.", 0, 0),
        ("teacher_currentsubject_group", "teacher", "context_interpretation", "grupo de disciplina do professor", "Interpretar com caveat do cadastro.", 0, 0),
        ("teacher_utm_group", "teacher", "context_interpretation", "origem padronizada do cadastro", "Contexto de aquisição.", 0, 0),
    ]

    readiness_rows: List[Dict[str, Any]] = []
    for feature_name, feature_level, feature_role, definition, caveat, range_ready, clustering_ready in feature_specs:
        frame = month_df if feature_level == "teacher_month" else teacher_df
        stats = numeric_feature_stats(frame[feature_name]) if feature_name in frame.columns and pd.api.types.is_numeric_dtype(frame[feature_name]) else {
            "missing_rate": float(frame[feature_name].astype("string").isna().mean()) if feature_name in frame.columns else 1.0,
            "zero_share": 0.0,
            "std": 0.0,
        }
        readiness_rows.append(
            {
                "feature_name": feature_name,
                "feature_level": feature_level,
                "feature_role": feature_role,
                "definition": definition,
                "missing_rate": stats["missing_rate"],
                "zero_share": stats["zero_share"],
                "std": stats["std"],
                "recommended_for_persona_analysis": int(feature_role in {"behavior_core", "behavior_support", "session_telemetry_support", "outcome_evaluation"}),
                "recommended_for_persona_ranges": int(range_ready),
                "recommended_for_behavior_clustering": int(clustering_ready),
                "caveat": caveat,
            }
        )

    readiness_df = pd.DataFrame(readiness_rows)
    conn.register("_persona_feature_readiness_df", readiness_df)
    conn.execute("CREATE OR REPLACE TABLE audit_persona_feature_readiness_final_v2 AS SELECT * FROM _persona_feature_readiness_df")

    range_specs = [
        ("activity_events_month", "teacher_month", "meses elegíveis para persona", month_df.loc[month_df["persona_analysis_eligible_flag"] == 1, "activity_events_month"]),
        ("active_days_month", "teacher_month", "meses elegíveis para persona", month_df.loc[month_df["persona_analysis_eligible_flag"] == 1, "active_days_month"]),
        ("strict_download_count_month", "teacher_month", "meses elegíveis para persona", month_df.loc[month_df["persona_analysis_eligible_flag"] == 1, "strict_download_count_month"]),
        ("content_views_month", "teacher_month", "meses elegíveis para persona", month_df.loc[month_df["persona_analysis_eligible_flag"] == 1, "content_views_month"]),
        ("other_activity_non_download_events_month", "teacher_month", "meses elegíveis para persona", month_df.loc[month_df["persona_analysis_eligible_flag"] == 1, "other_activity_non_download_events_month"]),
        ("clean_entry_total_session_minutes_month", "teacher_month", "meses elegíveis para persona", month_df.loc[month_df["persona_analysis_eligible_flag"] == 1, "clean_entry_total_session_minutes_month"]),
        ("avg_activity_events_active_month", "teacher", "professores elegíveis para persona", teacher_df.loc[teacher_df["teacher_persona_analysis_eligible_flag"] == 1, "avg_activity_events_active_month"]),
        ("avg_active_days_active_month", "teacher", "professores elegíveis para persona", teacher_df.loc[teacher_df["teacher_persona_analysis_eligible_flag"] == 1, "avg_active_days_active_month"]),
        ("avg_strict_downloads_active_month", "teacher", "professores elegíveis para persona", teacher_df.loc[teacher_df["teacher_persona_analysis_eligible_flag"] == 1, "avg_strict_downloads_active_month"]),
        ("avg_content_views_active_month", "teacher", "professores elegíveis para persona", teacher_df.loc[teacher_df["teacher_persona_analysis_eligible_flag"] == 1, "avg_content_views_active_month"]),
        ("avg_other_actions_active_month", "teacher", "professores elegíveis para persona", teacher_df.loc[teacher_df["teacher_persona_analysis_eligible_flag"] == 1, "avg_other_actions_active_month"]),
    ]
    range_rows: List[Dict[str, Any]] = []
    for feature_name, feature_level, population_label, series in range_specs:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            stats_row = {key: 0.0 for key in ["min_value", "p10", "p25", "p50", "p75", "p90", "p95", "max_value", "zero_share"]}
            n_rows = 0
        else:
            stats_row = {
                "min_value": float(numeric.min()),
                "p10": float(numeric.quantile(0.10)),
                "p25": float(numeric.quantile(0.25)),
                "p50": float(numeric.quantile(0.50)),
                "p75": float(numeric.quantile(0.75)),
                "p90": float(numeric.quantile(0.90)),
                "p95": float(numeric.quantile(0.95)),
                "max_value": float(numeric.max()),
                "zero_share": float((numeric == 0).mean()),
            }
            n_rows = int(numeric.shape[0])
        range_rows.append(
            {
                "feature_name": feature_name,
                "feature_level": feature_level,
                "population_used": population_label,
                "n_rows": n_rows,
                **stats_row,
                "note": "Usar faixas data-driven; não congelar cortes arbitrários sem validar estabilidade.",
            }
        )
    range_df = pd.DataFrame(range_rows)
    conn.register("_persona_range_candidates_df", range_df)
    conn.execute("CREATE OR REPLACE TABLE dim_persona_range_candidates_final_v2 AS SELECT * FROM _persona_range_candidates_df")


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
          LEFT JOIN (
            SELECT DISTINCT unique_id
            FROM raw_entries
            WHERE lower(coalesce(user_type, '')) = 'registered'
          ) e ON d.unique_id = e.unique_id
          LEFT JOIN (
            SELECT DISTINCT unique_id
            FROM raw_interactions
            WHERE lower(coalesce(user_type, '')) = 'registered'
          ) i ON d.unique_id = i.unique_id
          LEFT JOIN (SELECT DISTINCT unique_id_aprendizap FROM raw_formation) f ON d.unique_id = f.unique_id_aprendizap
          LEFT JOIN (SELECT DISTINCT unique_id_aprendizap FROM raw_mari_conv WHERE unique_id_aprendizap IS NOT NULL) mc ON d.unique_id = mc.unique_id_aprendizap
          LEFT JOIN (SELECT DISTINCT unique_id_aprendizap FROM raw_mari_reports WHERE unique_id_aprendizap IS NOT NULL) mr ON d.unique_id = mr.unique_id_aprendizap
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
          CASE WHEN d.estado IS NULL OR trim(d.estado) = '' THEN 1 ELSE 0 END AS is_estado_missing,
          CASE WHEN d.utm_origin IS NULL OR trim(d.utm_origin) = '' THEN 1 ELSE 0 END AS is_utm_missing,
          CASE WHEN d.tela_origem IS NULL OR trim(d.tela_origem) = '' THEN 1 ELSE 0 END AS is_tela_origem_missing,
          CASE WHEN d.total_alunos IS NULL THEN 1 ELSE 0 END AS is_total_alunos_missing,
          CASE WHEN d.tipo_total_alunos IS NULL OR trim(d.tipo_total_alunos) = '' THEN 1 ELSE 0 END AS is_tipo_total_alunos_missing,
          CASE WHEN d.alunos_diretos IS NULL THEN 1 ELSE 0 END AS is_alunos_diretos_missing,
          CASE WHEN d.alunos_indiretos IS NULL THEN 1 ELSE 0 END AS is_alunos_indiretos_missing,
          CASE WHEN d.login_google IS NULL THEN 1 ELSE 0 END AS is_login_google_missing,
          CASE WHEN d.currentstage IS NULL OR trim(d.currentstage) = '' THEN 1 ELSE 0 END AS is_currentstage_missing,
          CASE WHEN d.currentsubject IS NULL OR trim(d.currentsubject) = '' THEN 1 ELSE 0 END AS is_currentsubject_missing,
          CASE WHEN d.selectedstages IS NULL OR trim(d.selectedstages) = '' THEN 1 ELSE 0 END AS is_selectedstages_missing,
          CASE WHEN d.selectedsubjectsem IS NULL OR trim(d.selectedsubjectsem) = '' THEN 1 ELSE 0 END AS is_selectedsubjectsem_missing,
          CASE WHEN d.selectedsubjectsfundii IS NULL OR trim(d.selectedsubjectsfundii) = '' THEN 1 ELSE 0 END AS is_selectedsubjectsfundii_missing,
          CASE WHEN d.visualizou_metodologia_ativa IS NULL THEN 1 ELSE 0 END AS is_visualizou_metodologia_ativa_missing,
          CASE WHEN d.total_alunos < 0 THEN 1 ELSE 0 END AS is_total_alunos_negative,
          CASE WHEN d.alunos_diretos < 0 THEN 1 ELSE 0 END AS is_alunos_diretos_negative,
          CASE WHEN d.alunos_indiretos < 0 THEN 1 ELSE 0 END AS is_alunos_indiretos_negative,
          CASE WHEN d.login_google IS NOT NULL AND d.login_google NOT IN (0, 1) THEN 1 ELSE 0 END AS is_login_google_invalid
        FROM raw_dim_teachers d
        LEFT JOIN coverage c ON d.unique_id = c.unique_id
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
    conn.register("_dim_teacher_df", df)
    conn.execute("CREATE OR REPLACE TABLE dim_teacher AS SELECT * FROM _dim_teacher_df")
    return df


def create_bridge_mari_conversation_teacher(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE bridge_mari_conversation_teacher AS
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
            COUNT(*) FILTER (WHERE source_table = 'mari_reports') AS report_rows,
            COUNT(*) FILTER (WHERE source_table = 'mari_conv') AS conv_rows
          FROM union_src
          GROUP BY 1
        )
        SELECT
          id_mari,
          CASE WHEN teacher_resolution_count = 1 THEN resolved_teacher_candidate END AS teacher_unique_id,
          teacher_resolution_count,
          CASE WHEN teacher_resolution_count = 1 THEN 1 ELSE 0 END AS is_unambiguous,
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
    )


def create_bridge_teacher_identity_audit(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE bridge_teacher_identity_audit AS
        WITH entries_base AS (
          SELECT unique_id AS source_key, string_agg(DISTINCT lower(coalesce(user_type, 'missing')), '|') AS source_user_types
          FROM raw_entries
          GROUP BY 1
        ),
        interactions_base AS (
          SELECT unique_id AS source_key, string_agg(DISTINCT lower(coalesce(user_type, 'missing')), '|') AS source_user_types
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
            COUNT(DISTINCT b.teacher_unique_id) FILTER (WHERE b.is_unambiguous = 1 AND b.teacher_unique_id IS NOT NULL) AS resolved_teacher_count,
            MIN(b.teacher_unique_id) FILTER (WHERE b.is_unambiguous = 1 AND b.teacher_unique_id IS NOT NULL) AS resolved_teacher_unique_id
          FROM raw_mari_help h
          LEFT JOIN bridge_mari_conversation_teacher b ON h.user_id = b.id_mari
          GROUP BY 1
        )
        SELECT
          'raw_entries' AS source_table,
          'unique_id' AS source_key_name,
          e.source_key,
          e.source_user_types,
          'exact_unique_id' AS resolution_path,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN d.teacher_unique_id END AS resolved_teacher_unique_id,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END AS resolved_teacher_count,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END AS is_unambiguous,
          'uuid36' AS source_key_domain
        FROM entries_base e
        LEFT JOIN dim_teacher d ON e.source_key = d.teacher_unique_id
        UNION ALL
        SELECT
          'raw_interactions',
          'unique_id',
          i.source_key,
          i.source_user_types,
          'exact_unique_id',
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN d.teacher_unique_id END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          'uuid36'
        FROM interactions_base i
        LEFT JOIN dim_teacher d ON i.source_key = d.teacher_unique_id
        UNION ALL
        SELECT
          'raw_formation',
          'unique_id_aprendizap',
          f.source_key,
          NULL,
          'exact_same_domain',
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN d.teacher_unique_id END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          'uuid36'
        FROM formation_base f
        LEFT JOIN dim_teacher d ON f.source_key = d.teacher_unique_id
        UNION ALL
        SELECT
          'raw_mari_conv',
          'unique_id_aprendizap',
          m.source_key,
          NULL,
          'exact_same_domain',
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN d.teacher_unique_id END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          'uuid36'
        FROM mari_conv_base m
        LEFT JOIN dim_teacher d ON m.source_key = d.teacher_unique_id
        UNION ALL
        SELECT
          'raw_mari_reports',
          'unique_id_aprendizap',
          m.source_key,
          NULL,
          'exact_same_domain',
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN d.teacher_unique_id END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END,
          'uuid36'
        FROM mari_reports_base m
        LEFT JOIN dim_teacher d ON m.source_key = d.teacher_unique_id
        UNION ALL
        SELECT
          'raw_mari_help',
          'user_id',
          h.source_key,
          NULL,
          'semantic_mari_bridge',
          h.resolved_teacher_unique_id,
          h.resolved_teacher_count,
          CASE WHEN h.resolved_teacher_count = 1 THEN 1 ELSE 0 END,
          'hex64_upper'
        FROM mari_help_bridge h
        """
    )


def create_dim_event(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE dim_event AS
        WITH raw_base AS (
          SELECT
            coalesce(event_type, '<missing>') AS event_type,
            lower(coalesce(event_type, '')) AS event_type_lower,
            {sql_event_family_expr('event_type')} AS event_family,
            {sql_event_action_expr('event_type')} AS event_action,
            COUNT(*) AS raw_rows_total
          FROM raw_interactions
          GROUP BY 1, 2, 3, 4
        ),
        core_base AS (
          SELECT
            coalesce(event_type, '<missing>') AS event_type,
            COUNT(*) AS core_rows_total,
            SUM(is_activity_event) AS core_activity_rows_total,
            SUM(is_download_event) AS core_download_rows_total
          FROM fct_interaction_clean
          GROUP BY 1
        )
        SELECT
          r.event_type,
          r.event_family,
          r.event_action,
          r.raw_rows_total,
          coalesce(c.core_rows_total, 0) AS core_rows_total,
          coalesce(c.core_activity_rows_total, 0) AS core_activity_rows_total,
          coalesce(c.core_download_rows_total, 0) AS core_download_rows_total,
          CASE WHEN r.event_type IN {STRICT_VALUE_EVENTS} THEN 1 ELSE 0 END AS is_strict_value_event,
          CASE WHEN event_action = 'download' THEN 1 ELSE 0 END AS is_download_event,
          CASE WHEN event_action = 'view' THEN 1 ELSE 0 END AS is_visualization_event,
          CASE WHEN event_action = 'navigation' THEN 1 ELSE 0 END AS is_navigation_event,
          CASE WHEN r.event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0 ELSE 1 END AS is_activity_event
        FROM raw_base r
        LEFT JOIN core_base c USING(event_type)
        ORDER BY core_rows_total DESC, raw_rows_total DESC, event_type
        """
    )


def create_dim_lesson(conn: duckdb.DuckDBPyConnection) -> None:
    df = conn.execute(
        f"""
        WITH observed_lessons AS (
          SELECT
            id_aula AS lesson_id,
            COUNT(*) AS interaction_rows_total,
            COUNT(DISTINCT teacher_unique_id) AS distinct_teachers_total,
            MIN(interaction_ts) AS first_observed_ts,
            MAX(interaction_ts) AS last_observed_ts,
            SUM(is_download_event) AS download_events_total,
            SUM(is_strict_value_event) AS strict_download_events_total,
            SUM(is_content_view_event) AS content_view_events_total
          FROM fct_interaction_clean
          WHERE id_aula_semantic = 'lesson_like_22char'
          GROUP BY 1
        ),
        lesson_meta AS (
          SELECT
            id_aula AS lesson_id,
            titulo,
            nivel,
            ano,
            ano_em,
            disciplina,
            unidade,
            bncc,
            possui_metodologia_ativa,
            total_metodologias_ativa,
            CASE WHEN regexp_matches(id_aula, '{VALID_LESSON_ID_RE}') THEN 1 ELSE 0 END AS raw_lesson_id_valid_flag
          FROM raw_lessons
        )
        SELECT
          o.lesson_id,
          o.interaction_rows_total,
          o.distinct_teachers_total,
          o.first_observed_ts,
          o.last_observed_ts,
          o.download_events_total,
          o.strict_download_events_total,
          o.content_view_events_total,
          CASE WHEN m.lesson_id IS NOT NULL THEN 1 ELSE 0 END AS lesson_metadata_matched_flag,
          m.titulo,
          m.nivel,
          m.ano,
          m.ano_em,
          m.disciplina,
          m.unidade,
          m.bncc,
          m.possui_metodologia_ativa,
          m.total_metodologias_ativa,
          coalesce(m.raw_lesson_id_valid_flag, 0) AS raw_lesson_id_valid_flag
        FROM observed_lessons o
        LEFT JOIN lesson_meta m USING(lesson_id)
        ORDER BY o.lesson_id
        """
    ).fetchdf()
    df["discipline_group"] = df["disciplina"].apply(classify_discipline_group)
    df["lesson_id_semantic"] = "lesson_like_22char"
    df["is_active_methodology_missing"] = df["possui_metodologia_ativa"].isna().astype(int)
    df["is_metadata_missing"] = (df["lesson_metadata_matched_flag"].fillna(0).astype(int) == 0).astype(int)
    conn.register("_dim_lesson_df", df)
    conn.execute("CREATE OR REPLACE TABLE dim_lesson AS SELECT * FROM _dim_lesson_df")


def create_dim_device(conn: duckdb.DuckDBPyConnection) -> None:
    conn.register(
        "_dim_device_df",
        pd.DataFrame(
            [
                {"device_group": "desktop", "description": "Desktop reconhecido pelo raw."},
                {"device_group": "mobile", "description": "Mobile reconhecido pelo raw."},
                {"device_group": "tablet", "description": "Tablet reconhecido pelo raw."},
                {"device_group": "unknown", "description": "Device ausente ou não padronizado."},
            ]
        ),
    )
    conn.execute("CREATE OR REPLACE TABLE dim_device AS SELECT * FROM _dim_device_df")


def create_dim_calendar(conn: duckdb.DuckDBPyConnection) -> None:
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
    conn.register("_dim_calendar_df", df)
    conn.execute("CREATE OR REPLACE TABLE dim_calendar AS SELECT * FROM _dim_calendar_df")


def create_session_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_session_raw AS
        SELECT
          CAST(hash(e.unique_id, e.user_type, e.data_inicio, e.data_fim) AS UBIGINT) AS session_row_hash,
          e.unique_id AS source_unique_id,
          d.unique_id AS teacher_unique_id,
          lower(coalesce(e.user_type, 'missing')) AS user_type,
          date_trunc('month', e.data_inicio) AS session_month,
          e.data_inicio AS session_start_ts,
          e.data_fim AS session_end_ts,
          CASE
            WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL THEN GREATEST(epoch(e.data_fim) - epoch(e.data_inicio), 0)
          END AS duration_sec,
          CASE
            WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL AND e.data_fim < e.data_inicio THEN 1
            ELSE 0
          END AS is_negative_duration,
          CASE
            WHEN e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL AND GREATEST(epoch(e.data_fim) - epoch(e.data_inicio), 0) <= 5 THEN 1
            ELSE 0
          END AS is_ping_session_le_5s,
          CASE
            WHEN lower(coalesce(e.user_type, '')) = 'registered' AND d.unique_id IS NOT NULL THEN 1
            ELSE 0
          END AS is_core_teacher_session
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
        WHERE is_core_teacher_session = 1
          AND is_negative_duration = 0
          AND is_ping_session_le_5s = 0
        """
    )


def create_interaction_clean(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE fct_interaction_clean AS
        WITH base AS (
          SELECT
            CAST(hash(i.unique_id, i.data_inicio, i.event_type, i.content_type, i.id_aula, i.utm_source) AS UBIGINT) AS interaction_row_hash,
            i.unique_id AS source_unique_id,
            d.unique_id AS teacher_unique_id,
            lower(coalesce(i.user_type, 'missing')) AS user_type,
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
          WHERE lower(coalesce(i.user_type, '')) = 'registered'
            AND d.unique_id IS NOT NULL
            AND i.data_inicio IS NOT NULL
        )
        SELECT
          b.*,
          CASE WHEN b.event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END AS is_strict_value_event,
          CASE WHEN b.event_action = 'download' THEN 1 ELSE 0 END AS is_download_event,
          CASE WHEN b.event_action = 'view' THEN 1 ELSE 0 END AS is_visualization_event,
          CASE WHEN b.event_action = 'navigation' THEN 1 ELSE 0 END AS is_navigation_event,
          CASE WHEN b.event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0 ELSE 1 END AS is_activity_event,
          CASE WHEN b.event_action = 'view' AND b.event_family IN ('aula', 'plano', 'prova') THEN 1 ELSE 0 END AS is_content_view_event,
          CASE
            WHEN b.event_type_lower IN ('', 'acesso_aba_conquistas', 'fechar_conquista_obtida') THEN 0
            WHEN b.event_action = 'download' THEN 0
            WHEN b.event_action = 'view' AND b.event_family IN ('aula', 'plano', 'prova') THEN 0
            ELSE 1
          END AS is_other_activity_non_download_event,
          CASE WHEN b.id_aula_semantic = 'lesson_like_22char' THEN 1 ELSE 0 END AS lesson_join_allowed,
          CASE WHEN l.id_aula IS NOT NULL THEN 1 ELSE 0 END AS lesson_mapped_flag,
          l.id_aula AS lesson_id
        FROM base b
        LEFT JOIN raw_lessons l
          ON b.id_aula = l.id_aula
         AND b.id_aula_semantic = 'lesson_like_22char'
        """
    )


def create_formation_clean(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_formation_clean AS
        SELECT
          CAST(hash(f.unique_id_aprendizap, f.itemid, f.createdat, f.updatedat, f.type, f.completionstatus) AS UBIGINT) AS formation_row_hash,
          f.unique_id_aprendizap AS teacher_unique_id,
          coalesce(f.updatedat, f.createdat) AS formation_ts,
          date_trunc('month', coalesce(f.updatedat, f.createdat)) AS formation_month,
          coalesce(f.itemid, 'missing') AS item_id,
          coalesce(f.type, 'missing') AS item_type,
          coalesce(f.completionstatus, 'missing') AS completion_status,
          coalesce(f.progress, 0) AS progress,
          coalesce(f.questionstatus, 'missing') AS question_status,
          coalesce(f.coursemodulecount, -1) AS course_module_count,
          coalesce(f.moduleblockcount, -1) AS module_block_count,
          coalesce(f.quizquestioncount, -1) AS quiz_question_count,
          CASE WHEN d.teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END AS teacher_matched_flag,
          CASE WHEN coalesce(f.updatedat, f.createdat) IS NULL THEN 1 ELSE 0 END AS is_missing_timestamp,
          CASE WHEN lower(coalesce(f.completionstatus, '')) = 'complete' THEN 1 ELSE 0 END AS is_complete_status
        FROM raw_formation f
        LEFT JOIN dim_teacher d ON f.unique_id_aprendizap = d.teacher_unique_id
        WHERE d.teacher_unique_id IS NOT NULL
          AND coalesce(f.updatedat, f.createdat) IS NOT NULL
        """
    )


def create_mari_conversation_resolved(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_conversation_resolved AS
        SELECT
          c.id_mari,
          d.teacher_unique_id,
          c.createdat AS mari_created_ts,
          c.updatedat AS mari_updated_ts,
          date_trunc('month', coalesce(c.updatedat, c.createdat)) AS mari_month,
          coalesce(c.originsource, 'missing') AS origin_source,
          coalesce(c.userreaction, 'missing') AS user_reaction,
          CASE WHEN c.userlastmessage IS NULL OR trim(c.userlastmessage) = '' THEN 0 ELSE 1 END AS has_user_message,
          CASE WHEN c.ailastmessage IS NULL OR trim(c.ailastmessage) = '' THEN 0 ELSE 1 END AS has_ai_message
        FROM raw_mari_conv c
        INNER JOIN dim_teacher d ON c.unique_id_aprendizap = d.teacher_unique_id
        WHERE coalesce(c.updatedat, c.createdat) IS NOT NULL
        """
    )


def create_mari_reports_resolved(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_reports_resolved AS
        SELECT
          r.id_mari,
          d.teacher_unique_id,
          r.updatedat AS report_ts,
          date_trunc('month', r.updatedat) AS report_month,
          coalesce(r.key, 'missing') AS report_key,
          coalesce(r.value, 'missing') AS report_value,
          coalesce(r.metadata, 'missing') AS report_metadata
        FROM raw_mari_reports r
        INNER JOIN dim_teacher d ON r.unique_id_aprendizap = d.teacher_unique_id
        WHERE r.updatedat IS NOT NULL
        """
    )


def create_mari_help_resolved(conn: duckdb.DuckDBPyConnection) -> None:
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
          coalesce(h.turno, 'missing') AS turno,
          coalesce(h.key, 'missing') AS help_key,
          coalesce(h.isso_ajudou, 'missing') AS isso_ajudou,
          coalesce(h.isso_ajudou_num, -1) AS isso_ajudou_num
        FROM raw_mari_help h
        INNER JOIN bridge_mari_conversation_teacher b
          ON h.user_id = b.id_mari
        WHERE b.is_unambiguous = 1
          AND b.teacher_unique_id IS NOT NULL
          AND h.date IS NOT NULL
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
          COUNT(DISTINCT CAST(interaction_ts AS DATE)) FILTER (WHERE is_activity_event = 1) AS active_days_month,
          SUM(CASE WHEN event_family = 'aula' THEN 1 ELSE 0 END) AS aula_events_month,
          SUM(CASE WHEN event_family = 'plano' THEN 1 ELSE 0 END) AS plano_events_month,
          SUM(CASE WHEN event_family = 'prova' THEN 1 ELSE 0 END) AS prova_events_month,
          SUM(CASE WHEN event_family = 'ia' THEN 1 ELSE 0 END) AS ia_events_month,
          SUM(is_download_event) AS download_count_month,
          SUM(CASE WHEN event_type = 'download_aula' THEN 1 ELSE 0 END) AS download_aula_count_month,
          SUM(CASE WHEN event_type = 'download_plano_aula' THEN 1 ELSE 0 END) AS download_plano_count_month,
          SUM(CASE WHEN event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END) AS strict_download_count_month,
          SUM(is_content_view_event) AS content_views_month,
          SUM(is_other_activity_non_download_event) AS other_activity_non_download_events_month,
          COUNT(DISTINCT lesson_id) FILTER (WHERE lesson_mapped_flag = 1) AS mapped_lessons_month,
          MAX(is_strict_value_event) AS strict_value_flag,
          MAX(CASE WHEN is_activity_event = 1 THEN 1 ELSE 0 END) AS active_user_flag,
          MAX(CASE WHEN event_family = 'aula' AND is_content_view_event = 1 THEN 1 ELSE 0 END) AS viewed_aula_flag,
          MAX(CASE WHEN event_family = 'plano' AND is_content_view_event = 1 THEN 1 ELSE 0 END) AS viewed_plano_flag,
          MAX(CASE WHEN event_family = 'prova' AND is_content_view_event = 1 THEN 1 ELSE 0 END) AS viewed_prova_flag,
          MAX(CASE WHEN event_family = 'ia' AND is_activity_event = 1 THEN 1 ELSE 0 END) AS used_ia_flag,
          MAX(CASE WHEN device_group = 'desktop' THEN 1 ELSE 0 END) AS used_desktop_flag,
          MAX(CASE WHEN device_group = 'mobile' THEN 1 ELSE 0 END) AS used_mobile_flag,
          MAX(interaction_ts) AS last_interaction_ts_month
        FROM fct_interaction_clean
        GROUP BY 1, 2
        """
    ).fetchdf()

    entry_sessions_month = conn.execute(
        """
        SELECT
          teacher_unique_id,
          session_month AS month,
          COUNT(*) AS raw_entry_session_count_month,
          SUM(CASE WHEN is_ping_session_le_5s = 1 THEN 1 ELSE 0 END) AS ping_entry_session_count_month,
          SUM(CASE WHEN is_ping_session_le_5s = 0 AND is_negative_duration = 0 THEN 1 ELSE 0 END) AS clean_entry_session_count_month,
          SUM(CASE WHEN is_ping_session_le_5s = 0 AND is_negative_duration = 0 THEN duration_sec ELSE 0 END) / 60.0 AS clean_entry_total_session_minutes_month,
          AVG(CASE WHEN is_ping_session_le_5s = 0 AND is_negative_duration = 0 THEN duration_sec END) / 60.0 AS clean_entry_avg_session_minutes_month,
          MAX(session_end_ts) AS last_raw_entry_session_ts_month,
          MAX(CASE WHEN is_ping_session_le_5s = 0 AND is_negative_duration = 0 THEN session_end_ts END) AS last_clean_entry_session_ts_month
        FROM fct_session_raw
        WHERE is_core_teacher_session = 1
        GROUP BY 1, 2
        """
    ).fetchdf()

    for frame in [interactions_month, entry_sessions_month]:
        if not frame.empty:
            frame["month"] = pd.to_datetime(frame["month"], errors="coerce")

    month_df = entry_sessions_month.merge(interactions_month, on=["teacher_unique_id", "month"], how="outer")
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
    month_df["interaction_signal_flag"] = (month_df["interaction_rows_month"] > 0).astype(int)
    month_df["entry_signal_flag"] = (month_df["raw_entry_session_count_month"] > 0).astype(int)
    month_df["clean_entry_signal_flag"] = (month_df["clean_entry_session_count_month"] > 0).astype(int)
    month_df["only_ping_entry_flag"] = (
        (month_df["entry_signal_flag"] == 1)
        & (month_df["clean_entry_session_count_month"] <= 0)
        & (month_df["ping_entry_session_count_month"] == month_df["raw_entry_session_count_month"])
    ).astype(int)
    month_df["any_signal_flag"] = (
        (month_df["interaction_signal_flag"] == 1) | (month_df["entry_signal_flag"] == 1)
    ).astype(int)
    month_df["month_signal_class"] = np.select(
        [
            (month_df["interaction_signal_flag"] == 1) & (month_df["clean_entry_signal_flag"] == 1),
            (month_df["interaction_signal_flag"] == 1) & (month_df["only_ping_entry_flag"] == 1),
            (month_df["interaction_signal_flag"] == 1) & (month_df["entry_signal_flag"] == 0),
            (month_df["interaction_signal_flag"] == 0) & (month_df["clean_entry_signal_flag"] == 1),
            (month_df["interaction_signal_flag"] == 0) & (month_df["only_ping_entry_flag"] == 1),
            (month_df["interaction_signal_flag"] == 0) & (month_df["entry_signal_flag"] == 1),
        ],
        [
            "interaction_with_clean_entry",
            "interaction_with_only_ping_entry",
            "interaction_without_entry_signal",
            "clean_entry_without_interaction",
            "ping_entry_without_interaction",
            "entry_signal_without_interaction",
        ],
        default="other_signal_mix",
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
        ["teacher_unique_id", "month", "active_user_flag", "strict_value_flag", "strict_download_count_month"]
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
    month_df["lifetime_clean_entry_minutes_total"] = month_df.groupby("teacher_unique_id")[
        "clean_entry_total_session_minutes_month"
    ].cumsum()

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

    conn.register("_teacher_month_df", month_df)
    conn.execute("CREATE OR REPLACE TABLE fct_teacher_month AS SELECT * FROM _teacher_month_df")
    return month_df


def create_base_modelada(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE base_modelada_v2 AS
        SELECT
          tm.teacher_unique_id,
          tm.month,
          dt.population_status AS teacher_population_status,
          coalesce(dt.utm_origin, 'missing') AS teacher_utm_origin,
          coalesce(dt.utm_group, 'missing') AS teacher_utm_group,
          coalesce(dt.tela_origem, 'missing') AS teacher_tela_origem,
          coalesce(dt.estado, 'missing') AS teacher_estado,
          CASE WHEN dt.total_alunos IS NULL OR dt.total_alunos < 0 THEN -1 ELSE dt.total_alunos END AS teacher_total_alunos,
          coalesce(dt.tipo_total_alunos, 'missing') AS teacher_tipo_total_alunos,
          CASE WHEN dt.alunos_diretos IS NULL OR dt.alunos_diretos < 0 THEN -1 ELSE dt.alunos_diretos END AS teacher_alunos_diretos,
          CASE WHEN dt.alunos_indiretos IS NULL OR dt.alunos_indiretos < 0 THEN -1 ELSE dt.alunos_indiretos END AS teacher_alunos_indiretos,
          CASE WHEN dt.login_google IN (0, 1) THEN dt.login_google ELSE -1 END AS teacher_login_google,
          coalesce(dt.currentstage, 'missing') AS teacher_currentstage,
          coalesce(dt.currentsubject, 'missing') AS teacher_currentsubject,
          coalesce(dt.currentsubject_group, 'missing') AS teacher_currentsubject_group,
          coalesce(dt.selectedstages, 'missing') AS teacher_selectedstages,
          coalesce(dt.selectedsubjectsem, 'missing') AS teacher_selectedsubjectsem,
          coalesce(dt.selectedsubjectsfundii, 'missing') AS teacher_selectedsubjectsfundii,
          CASE WHEN dt.visualizou_metodologia_ativa IN (0, 1) THEN dt.visualizou_metodologia_ativa ELSE -1 END AS teacher_visualizou_metodologia_ativa,
          dt.data_entrada AS teacher_data_entrada,
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
          coalesce(CAST(tm.returned_active_m1 AS INTEGER), -1) AS returned_active_m1,
          coalesce(CAST(tm.returned_strict_value_m1 AS INTEGER), -1) AS returned_strict_value_m1,
          coalesce(CAST(tm.returned_any_download_m1 AS INTEGER), -1) AS returned_any_download_m1,
          coalesce(CAST(tm.strict_user_flag AS INTEGER), -1) AS strict_user_flag,
          coalesce(CAST(tm.strict_return_value_m1 AS INTEGER), -1) AS strict_return_value_m1,
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


def create_relevant_final_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_lesson_final_v2 AS
        SELECT
          coalesce(lesson_id, 'missing') AS lesson_id,
          coalesce(interaction_rows_total, 0) AS interaction_rows_total,
          coalesce(distinct_teachers_total, 0) AS distinct_teachers_total,
          first_observed_ts,
          last_observed_ts,
          coalesce(download_events_total, 0) AS download_events_total,
          coalesce(strict_download_events_total, 0) AS strict_download_events_total,
          coalesce(content_view_events_total, 0) AS content_view_events_total,
          coalesce(lesson_metadata_matched_flag, 0) AS lesson_metadata_matched_flag,
          coalesce(titulo, 'missing') AS lesson_title,
          coalesce(nivel, 'missing') AS lesson_level,
          coalesce(ano, -1) AS lesson_year,
          coalesce(ano_em, -1) AS lesson_year_em,
          coalesce(disciplina, 'missing') AS lesson_discipline,
          coalesce(discipline_group, 'missing') AS lesson_discipline_group,
          coalesce(unidade, 'missing') AS lesson_unit,
          coalesce(bncc, 'missing') AS lesson_bncc,
          CASE WHEN possui_metodologia_ativa IN (0, 1) THEN possui_metodologia_ativa ELSE -1 END AS lesson_has_active_methodology,
          coalesce(total_metodologias_ativa, -1) AS lesson_total_active_methodologies,
          coalesce(lesson_id_semantic, 'missing') AS lesson_id_semantic,
          coalesce(raw_lesson_id_valid_flag, 0) AS raw_lesson_id_valid_flag,
          coalesce(is_active_methodology_missing, 0) AS is_active_methodology_missing,
          coalesce(is_metadata_missing, 0) AS is_metadata_missing
        FROM dim_lesson
        ORDER BY lesson_id
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_event_final_v2 AS
        SELECT
          coalesce(event_type, '<missing>') AS event_type,
          coalesce(event_family, 'missing') AS event_family,
          coalesce(event_action, 'missing') AS event_action,
          coalesce(raw_rows_total, 0) AS raw_rows_total,
          coalesce(core_rows_total, 0) AS core_rows_total,
          coalesce(core_activity_rows_total, 0) AS core_activity_rows_total,
          coalesce(core_download_rows_total, 0) AS core_download_rows_total,
          coalesce(is_strict_value_event, 0) AS is_strict_value_event,
          coalesce(is_download_event, 0) AS is_download_event,
          coalesce(is_visualization_event, 0) AS is_visualization_event,
          coalesce(is_navigation_event, 0) AS is_navigation_event,
          coalesce(is_activity_event, 0) AS is_activity_event
        FROM dim_event
        ORDER BY core_rows_total DESC, raw_rows_total DESC, event_type
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_device_final_v2 AS
        SELECT
          coalesce(device_group, 'unknown') AS device_group,
          coalesce(description, 'missing') AS description
        FROM dim_device
        ORDER BY device_group
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_calendar_final_v2 AS
        SELECT
          coalesce(year, -1) AS calendar_year,
          coalesce(month, -1) AS calendar_month,
          month_start,
          coalesce(uf, 'missing') AS uf,
          coalesce(rede, 'missing') AS rede,
          coalesce(business_days, -1) AS business_days,
          coalesce(official_holiday_weekdays, -1) AS official_holiday_weekdays,
          coalesce(school_days_estimate, -1) AS school_days_estimate,
          coalesce(calendar_source, 'missing') AS calendar_source,
          coalesce(school_phase, 'missing') AS school_phase
        FROM dim_calendar
        ORDER BY month_start, uf, rede
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_teacher_final_v2 AS
        WITH dataset_max AS (
          SELECT MAX(month) AS dataset_last_month
          FROM base_modelada_v2
        ),
        teacher_usage AS (
          SELECT
            b.teacher_unique_id,
            MIN(b.month) AS teacher_first_observed_month,
            MAX(b.month) AS teacher_last_observed_month,
            COUNT(*) AS teacher_observed_months_total,
            SUM(b.active_user_flag) AS teacher_active_months_total,
            SUM(b.strict_value_flag) AS teacher_strict_months_total,
            SUM(b.strict_download_count_month) AS teacher_total_strict_downloads,
            SUM(b.download_count_month) AS teacher_total_downloads,
            SUM(b.clean_entry_session_count_month) AS teacher_total_clean_entry_sessions,
            SUM(b.clean_entry_total_session_minutes_month) AS teacher_total_clean_entry_minutes,
            MAX(b.active_streak_max_months) AS teacher_active_streak_max_months,
            MAX(b.strict_streak_max_months) AS teacher_strict_streak_max_months,
            (
              (EXTRACT(year FROM m.dataset_last_month) * 12 + EXTRACT(month FROM m.dataset_last_month))
              - (EXTRACT(year FROM MAX(b.month)) * 12 + EXTRACT(month FROM MAX(b.month)))
            )::BIGINT AS teacher_months_since_last_observed_month_dataset_end
          FROM base_modelada_v2 b
          CROSS JOIN dataset_max m
          GROUP BY 1, m.dataset_last_month
        )
        SELECT DISTINCT
          b.teacher_unique_id,
          b.teacher_population_status,
          b.teacher_utm_origin,
          b.teacher_utm_group,
          b.teacher_tela_origem,
          b.teacher_estado,
          b.teacher_total_alunos,
          b.teacher_tipo_total_alunos,
          b.teacher_alunos_diretos,
          b.teacher_alunos_indiretos,
          b.teacher_login_google,
          b.teacher_currentstage,
          b.teacher_currentsubject,
          b.teacher_currentsubject_group,
          b.teacher_selectedstages,
          b.teacher_selectedsubjectsem,
          b.teacher_selectedsubjectsfundii,
          b.teacher_visualizou_metodologia_ativa,
          b.teacher_data_entrada,
          u.teacher_first_observed_month,
          u.teacher_last_observed_month,
          u.teacher_months_since_last_observed_month_dataset_end,
          u.teacher_observed_months_total,
          u.teacher_active_months_total,
          u.teacher_strict_months_total,
          u.teacher_total_strict_downloads,
          u.teacher_total_downloads,
          u.teacher_total_clean_entry_sessions,
          u.teacher_total_clean_entry_minutes,
          u.teacher_active_streak_max_months,
          u.teacher_strict_streak_max_months,
          b.has_registered_entry,
          b.has_registered_interaction,
          b.has_formation,
          b.is_estado_missing,
          b.is_utm_missing,
          b.is_tela_origem_missing,
          b.is_total_alunos_missing,
          b.is_tipo_total_alunos_missing,
          b.is_alunos_diretos_missing,
          b.is_alunos_indiretos_missing,
          b.is_login_google_missing,
          b.is_currentstage_missing,
          b.is_currentsubject_missing,
          b.is_selectedstages_missing,
          b.is_selectedsubjectsem_missing,
          b.is_selectedsubjectsfundii_missing,
          b.is_visualizou_metodologia_ativa_missing,
          b.is_total_alunos_negative,
          b.is_alunos_diretos_negative,
          b.is_alunos_indiretos_negative,
          b.is_login_google_invalid
        FROM base_modelada_v2 b
        INNER JOIN teacher_usage u USING(teacher_unique_id)
        ORDER BY teacher_unique_id
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_session_clean_final_v2 AS
        SELECT
          session_row_hash,
          teacher_unique_id,
          coalesce(user_type, 'missing') AS user_type,
          session_month,
          session_start_ts,
          session_end_ts,
          coalesce(duration_sec, 0) AS duration_sec,
          coalesce(duration_min, 0) AS duration_min
        FROM fct_session_clean
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_interaction_clean_final_v2 AS
        SELECT
          interaction_row_hash,
          teacher_unique_id,
          interaction_month,
          interaction_ts,
          coalesce(event_type, '<missing>') AS event_type,
          coalesce(content_type, '<missing>') AS content_type,
          coalesce(utm_source, '<missing>') AS utm_source,
          coalesce(device_group, 'unknown') AS device_group,
          coalesce(event_family, 'missing') AS event_family,
          coalesce(event_action, 'missing') AS event_action,
          coalesce(id_aula, 'missing') AS raw_lesson_id,
          coalesce(id_aula_semantic, 'missing') AS id_aula_semantic,
          coalesce(lesson_join_allowed, 0) AS lesson_join_allowed,
          coalesce(lesson_mapped_flag, 0) AS lesson_mapped_flag,
          coalesce(lesson_id, 'missing') AS lesson_id,
          coalesce(is_strict_value_event, 0) AS is_strict_value_event,
          coalesce(is_download_event, 0) AS is_download_event,
          coalesce(is_visualization_event, 0) AS is_visualization_event,
          coalesce(is_navigation_event, 0) AS is_navigation_event,
          coalesce(is_activity_event, 0) AS is_activity_event,
          coalesce(is_content_view_event, 0) AS is_content_view_event,
          coalesce(is_other_activity_non_download_event, 0) AS is_other_activity_non_download_event
        FROM fct_interaction_clean
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_formation_clean_final_v2 AS
        SELECT
          formation_row_hash,
          teacher_unique_id,
          formation_ts,
          formation_month,
          coalesce(item_id, 'missing') AS item_id,
          coalesce(item_type, 'missing') AS item_type,
          coalesce(completion_status, 'missing') AS completion_status,
          coalesce(progress, 0) AS progress,
          coalesce(question_status, 'missing') AS question_status,
          coalesce(course_module_count, -1) AS course_module_count,
          coalesce(module_block_count, -1) AS module_block_count,
          coalesce(quiz_question_count, -1) AS quiz_question_count,
          coalesce(teacher_matched_flag, 0) AS teacher_matched_flag,
          coalesce(is_missing_timestamp, 0) AS is_missing_timestamp,
          coalesce(is_complete_status, 0) AS is_complete_status
        FROM fct_formation_clean
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_teacher_month_final_v2 AS
        SELECT
          teacher_unique_id,
          month,
          raw_entry_session_count_month,
          ping_entry_session_count_month,
          clean_entry_session_count_month,
          clean_entry_total_session_minutes_month,
          clean_entry_avg_session_minutes_month,
          interaction_rows_month,
          activity_events_month,
          active_days_month,
          aula_events_month,
          plano_events_month,
          prova_events_month,
          ia_events_month,
          download_count_month,
          download_aula_count_month,
          download_plano_count_month,
          strict_download_count_month,
          content_views_month,
          other_activity_non_download_events_month,
          mapped_lessons_month,
          interaction_signal_flag,
          entry_signal_flag,
          clean_entry_signal_flag,
          only_ping_entry_flag,
          any_signal_flag,
          month_signal_class,
          strict_value_flag,
          active_user_flag,
          viewed_aula_flag,
          viewed_plano_flag,
          viewed_prova_flag,
          used_ia_flag,
          used_desktop_flag,
          used_mobile_flag,
          no_download_flag,
          no_download_view_only_flag,
          no_download_view_plus_action_flag,
          no_download_action_only_flag,
          clean_entry_exposed_no_download_flag,
          clean_entry_exposed_no_activity_no_download_flag,
          clean_entry_exposed_activity_no_download_flag,
          month_num,
          next_month,
          next_month_observed_flag,
          returned_active_m1,
          returned_strict_value_m1,
          returned_any_download_m1,
          strict_user_flag,
          strict_return_value_m1,
          lifetime_active_months,
          lifetime_clean_entry_minutes_total,
          active_streak_current_months,
          active_streak_max_months,
          strict_streak_current_months,
          strict_streak_max_months
        FROM base_modelada_v2
        ORDER BY teacher_unique_id, month
        """
    )


def create_auxiliary_final_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TABLE bridge_teacher_identity_audit_final_v2 AS
        SELECT
          coalesce(source_table, 'missing') AS source_table,
          coalesce(source_key_name, 'missing') AS source_key_name,
          coalesce(source_key, 'missing') AS source_key,
          coalesce(source_user_types, 'missing') AS source_user_types,
          coalesce(resolution_path, 'missing') AS resolution_path,
          coalesce(resolved_teacher_unique_id, 'unresolved') AS resolved_teacher_unique_id,
          coalesce(resolved_teacher_count, 0) AS resolved_teacher_count,
          coalesce(is_unambiguous, 0) AS is_unambiguous,
          coalesce(source_key_domain, 'missing') AS source_key_domain,
          CASE WHEN resolved_teacher_unique_id IS NOT NULL THEN 1 ELSE 0 END AS is_resolved_flag
        FROM bridge_teacher_identity_audit
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE bridge_mari_conversation_teacher_final_v2 AS
        SELECT
          coalesce(id_mari, 'missing') AS id_mari,
          coalesce(teacher_unique_id, 'unresolved_or_ambiguous') AS teacher_unique_id,
          coalesce(teacher_resolution_count, 0) AS teacher_resolution_count,
          coalesce(is_unambiguous, 0) AS is_unambiguous,
          coalesce(resolution_source, 'missing') AS resolution_source,
          coalesce(teacher_candidates, 'missing') AS teacher_candidates,
          coalesce(report_rows, 0) AS report_rows,
          coalesce(conv_rows, 0) AS conv_rows
        FROM bridge_mari_conversation_teacher
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_conversation_resolved_final_v2 AS
        SELECT
          coalesce(id_mari, 'missing') AS id_mari,
          coalesce(teacher_unique_id, 'missing') AS teacher_unique_id,
          coalesce(mari_created_ts, mari_updated_ts, TIMESTAMP '1900-01-01 00:00:00') AS mari_created_ts,
          coalesce(mari_updated_ts, mari_created_ts, TIMESTAMP '1900-01-01 00:00:00') AS mari_updated_ts,
          coalesce(mari_month, date_trunc('month', coalesce(mari_updated_ts, mari_created_ts)), TIMESTAMP '1900-01-01 00:00:00') AS mari_month,
          coalesce(origin_source, 'missing') AS origin_source,
          coalesce(user_reaction, 'missing') AS user_reaction,
          coalesce(has_user_message, 0) AS has_user_message,
          coalesce(has_ai_message, 0) AS has_ai_message,
          CASE WHEN mari_created_ts IS NULL THEN 1 ELSE 0 END AS is_mari_created_ts_missing,
          CASE WHEN mari_updated_ts IS NULL THEN 1 ELSE 0 END AS is_mari_updated_ts_missing
        FROM fct_mari_conversation_resolved
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_reports_resolved_final_v2 AS
        SELECT
          coalesce(id_mari, 'missing') AS id_mari,
          coalesce(teacher_unique_id, 'missing') AS teacher_unique_id,
          report_ts,
          report_month,
          coalesce(report_key, 'missing') AS report_key,
          coalesce(report_value, 'missing') AS report_value,
          coalesce(report_metadata, 'missing') AS report_metadata
        FROM fct_mari_reports_resolved
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_mari_help_resolved_final_v2 AS
        SELECT
          coalesce(id_mari, 'missing') AS id_mari,
          coalesce(teacher_unique_id, 'missing') AS teacher_unique_id,
          coalesce(teacher_resolution_count, 0) AS teacher_resolution_count,
          coalesce(is_unambiguous, 0) AS is_unambiguous,
          coalesce(resolution_source, 'missing') AS resolution_source,
          help_ts,
          help_month,
          coalesce(turno, 'missing') AS turno,
          coalesce(help_key, 'missing') AS help_key,
          coalesce(isso_ajudou, 'missing') AS isso_ajudou,
          coalesce(isso_ajudou_num, -1) AS isso_ajudou_num
        FROM fct_mari_help_resolved
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE fct_session_raw_final_v2 AS
        SELECT
          session_row_hash,
          coalesce(source_unique_id, 'missing') AS source_unique_id,
          coalesce(teacher_unique_id, 'unmatched_teacher') AS teacher_unique_id,
          coalesce(user_type, 'missing') AS user_type,
          coalesce(session_month, TIMESTAMP '1900-01-01 00:00:00') AS session_month,
          coalesce(session_start_ts, TIMESTAMP '1900-01-01 00:00:00') AS session_start_ts,
          coalesce(session_end_ts, TIMESTAMP '1900-01-01 00:00:00') AS session_end_ts,
          coalesce(duration_sec, -1) AS duration_sec,
          coalesce(is_negative_duration, 0) AS is_negative_duration,
          coalesce(is_ping_session_le_5s, 0) AS is_ping_session_le_5s,
          coalesce(is_core_teacher_session, 0) AS is_core_teacher_session,
          CASE WHEN session_month IS NULL THEN 1 ELSE 0 END AS is_session_month_missing,
          CASE WHEN session_start_ts IS NULL THEN 1 ELSE 0 END AS is_session_start_missing,
          CASE WHEN session_end_ts IS NULL THEN 1 ELSE 0 END AS is_session_end_missing
        FROM fct_session_raw
        """
    )


def build_validation_table(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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
    base_teachers = int(conn.execute("SELECT COUNT(DISTINCT teacher_unique_id) FROM base_modelada_v2").fetchone()[0] or 0)
    fact_rows = int(conn.execute("SELECT COUNT(*) FROM fct_teacher_month").fetchone()[0] or 0)
    duplicate_grain = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT teacher_unique_id, month, COUNT(*) AS c
              FROM base_modelada_v2
              GROUP BY 1, 2
              HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        or 0
    )
    missing_month = int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2 WHERE month IS NULL").fetchone()[0] or 0)
    missing_teacher = int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2 WHERE teacher_unique_id IS NULL").fetchone()[0] or 0)
    clean_null_month = int(conn.execute("SELECT COUNT(*) FROM fct_interaction_clean WHERE interaction_month IS NULL").fetchone()[0] or 0)
    raw_registered_matched_null_ts = int(
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
    null_count_total = count_table_nulls(conn, "base_modelada_v2")
    dim_teacher_final_nulls = count_table_nulls(conn, "dim_teacher_final_v2")
    dim_event_final_nulls = count_table_nulls(conn, "dim_event_final_v2")
    dim_device_final_nulls = count_table_nulls(conn, "dim_device_final_v2")
    dim_calendar_final_nulls = count_table_nulls(conn, "dim_calendar_final_v2")
    session_clean_final_nulls = count_table_nulls(conn, "fct_session_clean_final_v2")
    interaction_clean_final_nulls = count_table_nulls(conn, "fct_interaction_clean_final_v2")
    formation_clean_final_nulls = count_table_nulls(conn, "fct_formation_clean_final_v2")
    teacher_month_final_nulls = count_table_nulls(conn, "fct_teacher_month_final_v2")
    panel_month_final_nulls = count_table_nulls(conn, "mart_teacher_month_panel_final_v2")
    persona_month_final_nulls = count_table_nulls(conn, "mart_teacher_month_persona_ready_final_v2")
    persona_teacher_final_nulls = count_table_nulls(conn, "mart_teacher_persona_ready_final_v2")
    cluster_month_final_nulls = count_table_nulls(conn, "mart_teacher_month_cluster_ready_final_v2")
    cluster_teacher_final_nulls = count_table_nulls(conn, "mart_teacher_cluster_ready_final_v2")
    persona_feature_readiness_final_nulls = count_table_nulls(conn, "audit_persona_feature_readiness_final_v2")
    persona_range_candidates_final_nulls = count_table_nulls(conn, "dim_persona_range_candidates_final_v2")
    dim_lesson_final_nulls = count_table_nulls(conn, "dim_lesson_final_v2")
    bridge_teacher_identity_audit_final_nulls = count_table_nulls(conn, "bridge_teacher_identity_audit_final_v2")
    bridge_mari_conversation_teacher_final_nulls = count_table_nulls(conn, "bridge_mari_conversation_teacher_final_v2")
    mari_conv_final_nulls = count_table_nulls(conn, "fct_mari_conversation_resolved_final_v2")
    mari_reports_final_nulls = count_table_nulls(conn, "fct_mari_reports_resolved_final_v2")
    mari_help_final_nulls = count_table_nulls(conn, "fct_mari_help_resolved_final_v2")
    session_raw_final_nulls = count_table_nulls(conn, "fct_session_raw_final_v2")
    lesson_join_stats = conn.execute(
        """
        SELECT
          COUNT(*) FILTER (WHERE id_aula_semantic = 'lesson_like_22char') AS lesson_like_rows,
          COUNT(*) FILTER (WHERE id_aula_semantic = 'lesson_like_22char' AND lesson_mapped_flag = 1) AS lesson_like_matched_rows
        FROM fct_interaction_clean
        """
    ).fetchone()
    lesson_like_rows = int(lesson_join_stats[0] or 0)
    lesson_like_matched_rows = int(lesson_join_stats[1] or 0)
    lesson_like_match_rate = (100.0 * lesson_like_matched_rows / lesson_like_rows) if lesson_like_rows else 0.0
    raw_lesson_nonstandard_rows = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM raw_lessons
            WHERE NOT regexp_matches(id_aula, '{VALID_LESSON_ID_RE}')
            """
        ).fetchone()[0]
        or 0
    )
    dim_lesson_nonstandard_rows = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM dim_lesson_final_v2
            WHERE NOT regexp_matches(lesson_id, '{VALID_LESSON_ID_RE}')
            """
        ).fetchone()[0]
        or 0
    )
    invalid_negative_values = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE teacher_total_alunos < -1
               OR teacher_alunos_diretos < -1
               OR teacher_alunos_indiretos < -1
            """
        ).fetchone()[0]
        or 0
    )
    persona_month_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_month_persona_ready_final_v2").fetchone()[0] or 0)
    persona_teacher_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_persona_ready_final_v2").fetchone()[0] or 0)
    cluster_month_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_month_cluster_ready_final_v2").fetchone()[0] or 0)
    cluster_teacher_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_cluster_ready_final_v2").fetchone()[0] or 0)
    panel_month_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_month_panel_final_v2").fetchone()[0] or 0)
    panel_observed_rows = int(
        conn.execute("SELECT COUNT(*) FROM mart_teacher_month_panel_final_v2 WHERE observed_month_flag = 1").fetchone()[0] or 0
    )
    dim_teacher_rows = int(conn.execute("SELECT COUNT(*) FROM dim_teacher_final_v2").fetchone()[0] or 0)
    persona_eligible_reconcile = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM mart_teacher_month_persona_ready_final_v2
            WHERE persona_analysis_eligible_flag <> active_user_flag
            """
        ).fetchone()[0]
        or 0
    )
    teacher_persona_eligible_reconcile = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM mart_teacher_persona_ready_final_v2
            WHERE teacher_persona_analysis_eligible_flag <> CASE WHEN teacher_active_months_total > 0 THEN 1 ELSE 0 END
            """
        ).fetchone()[0]
        or 0
    )
    base_any_signal_reconcile = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE any_signal_flag <> 1
            """
        ).fetchone()[0]
        or 0
    )
    strict_download_without_interaction_signal = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM base_modelada_v2
            WHERE strict_download_count_month > 0
              AND interaction_signal_flag <> 1
            """
        ).fetchone()[0]
        or 0
    )
    panel_no_signal_with_behavior = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM mart_teacher_month_panel_final_v2
            WHERE no_signal_month_flag = 1
              AND (
                strict_download_count_month > 0
                OR active_user_flag > 0
                OR activity_events_month > 0
                OR interaction_signal_flag > 0
                OR entry_signal_flag > 0
              )
            """
        ).fetchone()[0]
        or 0
    )
    clean_session_nulls = int(
        conn.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN session_month IS NULL THEN 1 ELSE 0 END), 0)
              + COALESCE(SUM(CASE WHEN session_start_ts IS NULL THEN 1 ELSE 0 END), 0)
              + COALESCE(SUM(CASE WHEN session_end_ts IS NULL THEN 1 ELSE 0 END), 0)
              + COALESCE(SUM(CASE WHEN duration_sec IS NULL THEN 1 ELSE 0 END), 0)
            FROM fct_session_clean
            """
        ).fetchone()[0]
        or 0
    )
    add_check(
        "row_count_matches_fct_teacher_month",
        float(base_rows - fact_rows),
        "pass" if base_rows == fact_rows else "fail",
        "A base modelada deve ter exatamente 1 linha por teacher-month valido do fato mensal.",
    )
    add_check(
        "distinct_teachers_positive",
        float(base_teachers),
        "pass" if base_teachers > 0 else "fail",
        "A base modelada deve conter professores distintos.",
    )
    add_check(
        "grain_teacher_month_unique",
        float(duplicate_grain),
        "pass" if duplicate_grain == 0 else "fail",
        "A chave (teacher_unique_id, month) deve ser unica na base modelada.",
    )
    add_check(
        "missing_teacher_unique_id",
        float(missing_teacher),
        "pass" if missing_teacher == 0 else "fail",
        "A base modelada nao pode perder a chave do professor.",
    )
    add_check(
        "missing_month",
        float(missing_month),
        "pass" if missing_month == 0 else "fail",
        "A base modelada nao pode conter month nulo.",
    )
    add_check(
        "base_modelada_has_no_nulls",
        float(null_count_total),
        "pass" if null_count_total == 0 else "fail",
        "A base modelada final deve exportar sem nulls; missings ficam explicitados por flags ou sentinelas.",
    )
    add_check(
        "clean_interactions_have_no_null_month",
        float(clean_null_month),
        "pass" if clean_null_month == 0 else "fail",
        "fct_interaction_clean nao pode ter interaction_month nulo.",
    )
    add_check(
        "clean_sessions_have_no_null_core_fields",
        float(clean_session_nulls),
        "pass" if clean_session_nulls == 0 else "fail",
        "fct_session_clean nao pode manter session_month, timestamps ou duracao nulos.",
    )
    add_check(
        "active_user_flag_reconciles",
        float(active_reconcile),
        "pass" if active_reconcile == 0 else "fail",
        "active_user_flag precisa refletir activity_events_month > 0.",
    )
    add_check(
        "strict_value_flag_reconciles",
        float(strict_reconcile),
        "pass" if strict_reconcile == 0 else "fail",
        "strict_value_flag precisa refletir strict_download_count_month > 0.",
    )
    add_check(
        "base_any_signal_flag_reconciles",
        float(base_any_signal_reconcile),
        "pass" if base_any_signal_reconcile == 0 else "fail",
        "Na base observada, toda linha precisa representar algum sinal observado no mês.",
    )
    add_check(
        "strict_download_requires_interaction_signal",
        float(strict_download_without_interaction_signal),
        "pass" if strict_download_without_interaction_signal == 0 else "fail",
        "Download strict só pode existir em mês com sinal de interaction.",
    )
    add_check(
        "registered_matched_interactions_with_null_timestamp_excluded",
        float(raw_registered_matched_null_ts),
        "pass",
        "Interacoes registered com match e data_inicio nula sao excluidas por nao terem mes confiavel.",
    )
    add_check(
        "base_invalid_negative_counts_normalized",
        float(invalid_negative_values),
        "pass" if invalid_negative_values == 0 else "fail",
        "Campos numericos de cadastro invalidos precisam ser normalizados para a sentinela -1 na base final.",
    )
    add_check(
        "dim_teacher_final_has_no_nulls",
        float(dim_teacher_final_nulls),
        "pass" if dim_teacher_final_nulls == 0 else "fail",
        "A dimensao final relevante deve sair sem nulls.",
    )
    add_check(
        "other_relevant_tables_have_no_nulls",
        float(
            dim_event_final_nulls
            + dim_device_final_nulls
            + dim_calendar_final_nulls
            + session_clean_final_nulls
            + interaction_clean_final_nulls
            + formation_clean_final_nulls
            + panel_month_final_nulls
            + persona_month_final_nulls
            + persona_teacher_final_nulls
            + cluster_month_final_nulls
            + cluster_teacher_final_nulls
            + persona_feature_readiness_final_nulls
            + persona_range_candidates_final_nulls
        ),
        "pass"
        if (
            dim_event_final_nulls
            + dim_device_final_nulls
            + dim_calendar_final_nulls
            + session_clean_final_nulls
            + interaction_clean_final_nulls
            + formation_clean_final_nulls
            + panel_month_final_nulls
            + persona_month_final_nulls
            + persona_teacher_final_nulls
            + cluster_month_final_nulls
            + cluster_teacher_final_nulls
            + persona_feature_readiness_final_nulls
            + persona_range_candidates_final_nulls
        )
        == 0
        else "fail",
        "As demais tabelas relevantes exportadas tambem devem sair sem nulls SQL.",
    )
    add_check(
        "auxiliary_tables_have_no_nulls",
        float(
            dim_lesson_final_nulls
            + bridge_teacher_identity_audit_final_nulls
            + bridge_mari_conversation_teacher_final_nulls
            + mari_conv_final_nulls
            + mari_reports_final_nulls
            + mari_help_final_nulls
            + session_raw_final_nulls
        ),
        "pass"
        if (
            dim_lesson_final_nulls
            + bridge_teacher_identity_audit_final_nulls
            + bridge_mari_conversation_teacher_final_nulls
            + mari_conv_final_nulls
            + mari_reports_final_nulls
            + mari_help_final_nulls
            + session_raw_final_nulls
        )
        == 0
        else "fail",
        "As tabelas auxiliares exportadas tambem devem sair sem nulls SQL; ausencias ficam explicitadas por sentinelas e flags.",
    )
    add_check(
        "fct_teacher_month_final_has_no_nulls",
        float(teacher_month_final_nulls),
        "pass" if teacher_month_final_nulls == 0 else "fail",
        "O fato mensal final relevante deve sair sem nulls.",
    )
    add_check(
        "panel_rows_cover_observed_base",
        float(panel_observed_rows - base_rows),
        "pass" if panel_observed_rows == base_rows else "fail",
        "O painel densificado deve conter exatamente todas as linhas observadas da base no subconjunto observed_month_flag=1.",
    )
    add_check(
        "panel_no_signal_has_no_behavior",
        float(panel_no_signal_with_behavior),
        "pass" if panel_no_signal_with_behavior == 0 else "fail",
        "Meses no_signal do painel não podem carregar comportamento, download ou sinais observados.",
    )
    add_check(
        "persona_teacher_month_rows_match_base",
        float(persona_month_rows - base_rows),
        "pass" if persona_month_rows == base_rows else "fail",
        "A mart mensal de personas deve cobrir exatamente as mesmas linhas da base_modelada_v2.",
    )
    add_check(
        "persona_teacher_rows_match_dim_teacher",
        float(persona_teacher_rows - dim_teacher_rows),
        "pass" if persona_teacher_rows == dim_teacher_rows else "fail",
        "A mart de personas por professor deve cobrir os mesmos professores da dim_teacher_final_v2.",
    )
    add_check(
        "cluster_teacher_month_rows_match_base",
        float(cluster_month_rows - base_rows),
        "pass" if cluster_month_rows == base_rows else "fail",
        "A mart mensal cluster_ready deve cobrir exatamente as mesmas linhas observadas da base_modelada_v2.",
    )
    add_check(
        "cluster_teacher_rows_match_dim_teacher",
        float(cluster_teacher_rows - dim_teacher_rows),
        "pass" if cluster_teacher_rows == dim_teacher_rows else "fail",
        "A mart cluster_ready por professor deve cobrir os mesmos professores da dim_teacher_final_v2.",
    )
    add_check(
        "persona_eligible_flag_reconciles",
        float(persona_eligible_reconcile),
        "pass" if persona_eligible_reconcile == 0 else "fail",
        "persona_analysis_eligible_flag deve refletir active_user_flag no grão teacher-month.",
    )
    add_check(
        "teacher_persona_eligible_flag_reconciles",
        float(teacher_persona_eligible_reconcile),
        "pass" if teacher_persona_eligible_reconcile == 0 else "fail",
        "teacher_persona_analysis_eligible_flag deve refletir existência de ao menos um mês ativo.",
    )
    add_check(
        "dim_lesson_contains_only_valid_observed_ids",
        float(dim_lesson_nonstandard_rows),
        "pass" if dim_lesson_nonstandard_rows == 0 else "fail",
        "A dim_lesson final deve conter apenas ids observados nas interacoes e semanticamente validos.",
    )
    add_check(
        "lesson_like_match_rate_info",
        float(round(lesson_like_match_rate, 4)),
        "pass",
        "Cobertura de join de lessons considerando apenas ids de aula semanticamente validos; usar dim_lesson apenas nesse subconjunto.",
    )
    add_check(
        "raw_lessons_nonstandard_rows_info",
        float(raw_lesson_nonstandard_rows),
        "pass",
        "Quantidade de ids nao padronizados existente no stg_lessons bruto; esses ids nao entram na dim_lesson final.",
    )
    validation_df = pd.DataFrame(rows)
    conn.register("_validation_df", validation_df)
    conn.execute("CREATE OR REPLACE TABLE audit_base_modelada_validation AS SELECT * FROM _validation_df")
    return validation_df


def run(cfg: Config) -> Dict[str, Any]:
    ensure_output_dirs(cfg.output_dir)
    reset_generated_export_dirs(cfg.output_dir)
    conn = connect_duckdb(cfg)
    try:
        LOGGER.info("Registrando views raw")
        register_raw_views(conn, cfg.data_dir)
        LOGGER.info("Criando dimensao principal de professores")
        create_dim_teacher(conn)
        LOGGER.info("Criando bridges de identidade e dimensoes secundarias")
        create_bridge_mari_conversation_teacher(conn)
        create_bridge_teacher_identity_audit(conn)
        create_dim_device(conn)
        create_dim_calendar(conn)
        LOGGER.info("Criando fatos limpos de sessoes, interacoes, formacao e Mari")
        create_session_tables(conn)
        create_interaction_clean(conn)
        create_dim_event(conn)
        create_dim_lesson(conn)
        create_formation_clean(conn)
        create_mari_conversation_resolved(conn)
        create_mari_reports_resolved(conn)
        create_mari_help_resolved(conn)
        LOGGER.info("Construindo fato mensal de professores e base modelada")
        LOGGER.info("Iniciando build_teacher_month")
        build_teacher_month(conn)
        LOGGER.info("Finalizado build_teacher_month")
        LOGGER.info("Iniciando create_base_modelada")
        create_base_modelada(conn)
        LOGGER.info("Finalizado create_base_modelada")
        LOGGER.info("Iniciando create_relevant_final_tables")
        create_relevant_final_tables(conn)
        LOGGER.info("Finalizado create_relevant_final_tables")
        LOGGER.info("Iniciando create_persona_ready_tables")
        create_persona_ready_tables(conn)
        LOGGER.info("Finalizado create_persona_ready_tables")
        LOGGER.info("Iniciando create_auxiliary_final_tables")
        create_auxiliary_final_tables(conn)
        LOGGER.info("Finalizado create_auxiliary_final_tables")
        LOGGER.info("Executando validacoes")
        LOGGER.info("Iniciando build_validation_table")
        validation_df = build_validation_table(conn)
        LOGGER.info("Finalizado build_validation_table")

        LOGGER.info("Exportando tabelas relevantes e auxiliares em parquet")
        relevant_exports = {
            "base_modelada_v2": persist_table_to_paths(
                conn,
                "base_modelada_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "base_modelada_v2.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "base_modelada_v2.parquet",
                export_csv=False,
            ),
            "dim_teacher": persist_table_to_paths(
                conn,
                "dim_teacher_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "dim_teacher.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "dim_teacher.parquet",
                export_csv=False,
            ),
            "dim_event": persist_table_to_paths(
                conn,
                "dim_event_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "dim_event.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "dim_event.parquet",
                export_csv=False,
            ),
            "dim_device": persist_table_to_paths(
                conn,
                "dim_device_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "dim_device.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "dim_device.parquet",
                export_csv=False,
            ),
            "dim_calendar": persist_table_to_paths(
                conn,
                "dim_calendar_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "dim_calendar.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "dim_calendar.parquet",
                export_csv=False,
            ),
            "fct_session_clean": persist_table_to_paths(
                conn,
                "fct_session_clean_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "fct_session_clean.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "fct_session_clean.parquet",
                export_csv=False,
            ),
            "fct_interaction_clean": persist_table_to_paths(
                conn,
                "fct_interaction_clean_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "fct_interaction_clean.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "fct_interaction_clean.parquet",
                export_csv=False,
            ),
            "fct_formation_clean": persist_table_to_paths(
                conn,
                "fct_formation_clean_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "fct_formation_clean.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "fct_formation_clean.parquet",
                export_csv=False,
            ),
            "fct_teacher_month": persist_table_to_paths(
                conn,
                "fct_teacher_month_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "fct_teacher_month.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "fct_teacher_month.parquet",
                export_csv=False,
            ),
            "mart_teacher_month_panel": persist_table_to_paths(
                conn,
                "mart_teacher_month_panel_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "mart_teacher_month_panel.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "mart_teacher_month_panel.parquet",
                export_csv=False,
            ),
            "mart_teacher_month_persona_ready": persist_table_to_paths(
                conn,
                "mart_teacher_month_persona_ready_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "mart_teacher_month_persona_ready.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "mart_teacher_month_persona_ready.parquet",
                export_csv=False,
            ),
            "mart_teacher_persona_ready": persist_table_to_paths(
                conn,
                "mart_teacher_persona_ready_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "mart_teacher_persona_ready.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "mart_teacher_persona_ready.parquet",
                export_csv=False,
            ),
            "mart_teacher_month_cluster_ready": persist_table_to_paths(
                conn,
                "mart_teacher_month_cluster_ready_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "mart_teacher_month_cluster_ready.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "mart_teacher_month_cluster_ready.parquet",
                export_csv=False,
            ),
            "mart_teacher_cluster_ready": persist_table_to_paths(
                conn,
                "mart_teacher_cluster_ready_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "mart_teacher_cluster_ready.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "mart_teacher_cluster_ready.parquet",
                export_csv=False,
            ),
            "audit_persona_feature_readiness": persist_table_to_paths(
                conn,
                "audit_persona_feature_readiness_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "audit_persona_feature_readiness.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "audit_persona_feature_readiness.parquet",
                export_csv=False,
            ),
            "dim_persona_range_candidates": persist_table_to_paths(
                conn,
                "dim_persona_range_candidates_final_v2",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "dim_persona_range_candidates.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "dim_persona_range_candidates.parquet",
                export_csv=False,
            ),
            "audit_base_modelada_validation": persist_table_to_paths(
                conn,
                "audit_base_modelada_validation",
                cfg.output_dir / "tabelas_relevantes" / "csv" / "audit_base_modelada_validation.csv",
                cfg.output_dir / "tabelas_relevantes" / "parquet" / "audit_base_modelada_validation.parquet",
                export_csv=False,
            ),
        }
        auxiliary_exports = {
            "bridge_teacher_identity_audit": persist_table_to_paths(
                conn,
                "bridge_teacher_identity_audit_final_v2",
                cfg.output_dir / "tabelas_auxiliares" / "csv" / "bridge_teacher_identity_audit.csv",
                cfg.output_dir / "tabelas_auxiliares" / "parquet" / "bridge_teacher_identity_audit.parquet",
                export_csv=False,
            ),
            "dim_lesson": persist_table_to_paths(
                conn,
                "dim_lesson_final_v2",
                cfg.output_dir / "tabelas_auxiliares" / "csv" / "dim_lesson.csv",
                cfg.output_dir / "tabelas_auxiliares" / "parquet" / "dim_lesson.parquet",
                export_csv=False,
            ),
            "bridge_mari_conversation_teacher": persist_table_to_paths(
                conn,
                "bridge_mari_conversation_teacher_final_v2",
                cfg.output_dir / "tabelas_auxiliares" / "csv" / "bridge_mari_conversation_teacher.csv",
                cfg.output_dir / "tabelas_auxiliares" / "parquet" / "bridge_mari_conversation_teacher.parquet",
                export_csv=False,
            ),
            "fct_mari_conversation_resolved": persist_table_to_paths(
                conn,
                "fct_mari_conversation_resolved_final_v2",
                cfg.output_dir / "tabelas_auxiliares" / "csv" / "fct_mari_conversation_resolved.csv",
                cfg.output_dir / "tabelas_auxiliares" / "parquet" / "fct_mari_conversation_resolved.parquet",
                export_csv=False,
            ),
            "fct_mari_reports_resolved": persist_table_to_paths(
                conn,
                "fct_mari_reports_resolved_final_v2",
                cfg.output_dir / "tabelas_auxiliares" / "csv" / "fct_mari_reports_resolved.csv",
                cfg.output_dir / "tabelas_auxiliares" / "parquet" / "fct_mari_reports_resolved.parquet",
                export_csv=False,
            ),
            "fct_mari_help_resolved": persist_table_to_paths(
                conn,
                "fct_mari_help_resolved_final_v2",
                cfg.output_dir / "tabelas_auxiliares" / "csv" / "fct_mari_help_resolved.csv",
                cfg.output_dir / "tabelas_auxiliares" / "parquet" / "fct_mari_help_resolved.parquet",
                export_csv=False,
            ),
            "fct_session_raw": persist_table_to_paths(
                conn,
                "fct_session_raw_final_v2",
                cfg.output_dir / "tabelas_auxiliares" / "csv" / "fct_session_raw.csv",
                cfg.output_dir / "tabelas_auxiliares" / "parquet" / "fct_session_raw.parquet",
                export_csv=False,
            ),
        }
        LOGGER.info("Montando sumarios e manifestos")
        failed_checks = int((validation_df["status"] == "fail").sum())
        estado_missing_rows = int(
            conn.execute("SELECT COUNT(*) FROM base_modelada_v2 WHERE teacher_estado = 'missing'").fetchone()[0] or 0
        )
        base_rows = int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2").fetchone()[0] or 0)
        raw_session_rows = int(conn.execute("SELECT COUNT(*) FROM fct_session_raw").fetchone()[0] or 0)
        ping_rows = int(conn.execute("SELECT COUNT(*) FROM fct_session_raw WHERE is_ping_session_le_5s = 1").fetchone()[0] or 0)
        lesson_quality = conn.execute(
            """
            SELECT
              COUNT(*) AS rows_total,
              COUNT(*) FILTER (WHERE lesson_metadata_matched_flag = 1) AS metadata_matched_rows,
              COUNT(*) FILTER (WHERE lesson_bncc = 'missing') AS bncc_missing_rows,
              COUNT(*) FILTER (WHERE lesson_has_active_methodology = -1) AS methodology_missing_rows,
              SUM(is_metadata_missing) AS metadata_missing_rows
            FROM dim_lesson_final_v2
            """
        ).fetchone()
        lesson_rows_total = int(lesson_quality[0] or 0)
        interaction_lesson_quality = conn.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE id_aula_semantic = 'lesson_like_22char') AS lesson_like_rows,
              COUNT(*) FILTER (WHERE id_aula_semantic = 'lesson_like_22char' AND lesson_mapped_flag = 1) AS matched_rows
            FROM fct_interaction_clean
            """
        ).fetchone()
        interaction_lesson_like_rows = int(interaction_lesson_quality[0] or 0)
        interaction_lesson_matched_rows = int(interaction_lesson_quality[1] or 0)
        table_groups = {
            "tabelas_relevantes": {
                "description": "Tabelas principais para analytics, EDA e modelagem futura, ja limpas, joinadas e com colunas semanticamente relevantes.",
                "tables": {
                    "base_modelada_v2": {
                        **relevant_exports["base_modelada_v2"],
                        "grain": "teacher_unique_id x month",
                        "description": "Tabela central pronta para analytics mensal por professor.",
                    },
                    "dim_teacher": {
                        **relevant_exports["dim_teacher"],
                        "grain": "teacher_unique_id",
                        "description": "Dimensao principal do professor com atributos de cadastro e agregados de uso.",
                    },
                    "dim_event": {
                        **relevant_exports["dim_event"],
                        "grain": "event_type",
                        "description": "Taxonomia de eventos com contagens raw e core para apoiar leitura semantica.",
                    },
                    "dim_device": {
                        **relevant_exports["dim_device"],
                        "grain": "device_group",
                        "description": "Padronizacao dos devices observados nas interacoes.",
                    },
                    "dim_calendar": {
                        **relevant_exports["dim_calendar"],
                        "grain": "month_start x uf x rede",
                        "description": "Calendario escolar para enriquecimento temporal e regional.",
                    },
                    "fct_session_clean": {
                        **relevant_exports["fct_session_clean"],
                        "grain": "session_row_hash",
                        "description": "Telemetria limpa de entry/sessão core, sem ping e sem duração inválida; não usar isoladamente como definição de acesso.",
                    },
                    "fct_interaction_clean": {
                        **relevant_exports["fct_interaction_clean"],
                        "grain": "interaction_row_hash",
                        "description": "Fato limpo de interacoes core, com taxonomia de evento e flags de lesson join.",
                    },
                    "fct_formation_clean": {
                        **relevant_exports["fct_formation_clean"],
                        "grain": "formation_row_hash",
                        "description": "Fato limpo de formacao, ligado por unique_id do professor.",
                    },
                    "fct_teacher_month": {
                        **relevant_exports["fct_teacher_month"],
                        "grain": "teacher_unique_id x month",
                        "description": "Fato mensal observado com comportamento via interactions e telemetria de entry explicitamente separada.",
                    },
                    "mart_teacher_month_panel": {
                        **relevant_exports["mart_teacher_month_panel"],
                        "grain": "teacher_unique_id x month",
                        "description": "Painel mensal densificado para no_signal real, gaps, retorno e abandono sem confundir ausência com limpeza.",
                    },
                    "mart_teacher_month_persona_ready": {
                        **relevant_exports["mart_teacher_month_persona_ready"],
                        "grain": "teacher_unique_id x month",
                        "description": "Camada mensal curada para construir personas por faixas e comportamento, com reconciliação de sinais e telemetria de entry separada.",
                    },
                    "mart_teacher_persona_ready": {
                        **relevant_exports["mart_teacher_persona_ready"],
                        "grain": "teacher_unique_id",
                        "description": "Rollup por professor com sinais de comportamento agregados, pronto para personas no nível teacher.",
                    },
                    "mart_teacher_month_cluster_ready": {
                        **relevant_exports["mart_teacher_month_cluster_ready"],
                        "grain": "teacher_unique_id x month",
                        "description": "Camada mensal enxuta para clustering comportamental, priorizando features centrais e mantendo telemetria de entry como suporte.",
                    },
                    "mart_teacher_cluster_ready": {
                        **relevant_exports["mart_teacher_cluster_ready"],
                        "grain": "teacher_unique_id",
                        "description": "Rollup por professor com features contínuas e shares pronto para clustering e comparação de perfis.",
                    },
                    "audit_persona_feature_readiness": {
                        **relevant_exports["audit_persona_feature_readiness"],
                        "grain": "feature_name x feature_level",
                        "description": "Dicionário auditado das features indicadas para personas, ranges e clustering comportamental.",
                    },
                    "dim_persona_range_candidates": {
                        **relevant_exports["dim_persona_range_candidates"],
                        "grain": "feature_name x feature_level",
                        "description": "Quantis e candidatos de faixas data-driven para construir personas sem cortes arbitrários.",
                    },
                    "audit_base_modelada_validation": {
                        **relevant_exports["audit_base_modelada_validation"],
                        "grain": "check_name",
                        "description": "Resultado das validacoes de integridade, nulidade e reconciliacao.",
                    },
                },
            },
            "tabelas_auxiliares": {
                "description": "Tabelas auxiliares, auditaveis ou secundarias; lessons e Mari ficam aqui por exigirem cautela semantica adicional.",
                "tables": {
                    "bridge_teacher_identity_audit": {
                        **auxiliary_exports["bridge_teacher_identity_audit"],
                        "grain": "source_table x source_key",
                        "description": "Auditoria de resolucao de identidade entre chaves de origem e professor.",
                    },
                    "dim_lesson": {
                        **auxiliary_exports["dim_lesson"],
                        "grain": "lesson_id",
                        "description": "Universo observado de lessons validos nas interacoes, com enriquecimento opcional do stg_lessons.",
                    },
                    "bridge_mari_conversation_teacher": {
                        **auxiliary_exports["bridge_mari_conversation_teacher"],
                        "grain": "id_mari",
                        "description": "Ponte de resolucao Mari -> professor com flags de ambiguidade.",
                    },
                    "fct_mari_conversation_resolved": {
                        **auxiliary_exports["fct_mari_conversation_resolved"],
                        "grain": "id_mari x mari_updated_ts",
                        "description": "Conversas Mari resolvidas para professor quando a chave e confiavel.",
                    },
                    "fct_mari_reports_resolved": {
                        **auxiliary_exports["fct_mari_reports_resolved"],
                        "grain": "id_mari x report_ts x report_key",
                        "description": "Reports Mari resolvidos para professor, mantidos fora do core principal.",
                    },
                    "fct_mari_help_resolved": {
                        **auxiliary_exports["fct_mari_help_resolved"],
                        "grain": "id_mari x help_ts x help_key",
                        "description": "Eventos de feedback da Mari apenas quando a ponte resolve univocamente.",
                    },
                    "fct_session_raw": {
                        **auxiliary_exports["fct_session_raw"],
                        "grain": "session_row_hash",
                        "description": "Sessao raw auditavel com sentinelas e flags de missing para diagnostico.",
                    },
                },
            },
        }
        summary = {
            "generated_at_utc": utc_now_iso(),
            "duckdb_path": str(cfg.duckdb_path),
            "table_name": "base_modelada_v2",
            "grain": "teacher_unique_id x month",
            "row_count": int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2").fetchone()[0] or 0),
            "distinct_teachers": int(conn.execute("SELECT COUNT(DISTINCT teacher_unique_id) FROM base_modelada_v2").fetchone()[0] or 0),
            "min_month": str(conn.execute("SELECT MIN(month) FROM base_modelada_v2").fetchone()[0]),
            "max_month": str(conn.execute("SELECT MAX(month) FROM base_modelada_v2").fetchone()[0]),
            "exports": relevant_exports["base_modelada_v2"],
            "validation_parquet": relevant_exports["audit_base_modelada_validation"]["parquet"],
            "validation_status": "pass" if failed_checks == 0 else "fail",
            "table_groups": table_groups,
            "definitions": {
                "active_user": "fez algo na plataforma no mes",
                "strict_value": "download_aula ou download_plano_aula",
                "strict_user": "fez strict_value em t e retornou ativo em t+1",
                "strict_download_count_month": "contagem mensal de downloads strict, separada da flag strict_value",
                "persona_analysis_eligible_flag": "mês elegível para personas = active_user_flag",
                "month_signal_class": "reconciliação entre behavior via interactions e telemetria de entry no mês observado",
                "no_signal_month_flag": "mês do painel densificado sem interaction e sem entry observados",
            },
            "quality_highlights": {
                "teacher_estado": {
                    "column": "teacher_estado",
                    "missing_flag": "is_estado_missing",
                    "missing_rows": estado_missing_rows,
                    "missing_pct": round((100.0 * estado_missing_rows / base_rows), 2) if base_rows else 0.0,
                    "note": "Usar sempre junto com a flag de missing; a cobertura pode limitar segmentacoes por estado.",
                },
                "session_duration": {
                    "columns": ["clean_entry_total_session_minutes_month", "clean_entry_avg_session_minutes_month"],
                    "source": "fct_session_raw -> telemetria limpa de entry",
                    "ping_rows_removed": ping_rows,
                    "ping_pct_of_raw_sessions": round((100.0 * ping_rows / raw_session_rows), 2) if raw_session_rows else 0.0,
                    "note": "Duração é telemetria auxiliar de entry: usa apenas sessões com início/fim válidos, sem duração negativa e sem ping <= 5s; não define acesso por si só.",
                },
                "stg_lessons": {
                    "note": "Dimensao util, mas secundaria em confianca frente a dim_teacher + fct_interaction_clean. A dim_lesson final nasce do universo observado nas interacoes validas e usa stg_lessons apenas como enriquecimento opcional.",
                    "rows_total_dim_lesson_observed": lesson_rows_total,
                    "metadata_matched_rows": int(lesson_quality[1] or 0),
                    "bncc_missing_rows": int(lesson_quality[2] or 0),
                    "methodology_missing_rows": int(lesson_quality[3] or 0),
                    "metadata_missing_rows": int(lesson_quality[4] or 0),
                    "raw_lessons_nonstandard_rows_excluded": int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM raw_lessons WHERE NOT regexp_matches(id_aula, '{VALID_LESSON_ID_RE}')"
                        ).fetchone()[0]
                        or 0
                    ),
                    "interaction_lesson_like_match_rate_pct": round(
                        100.0 * interaction_lesson_matched_rows / interaction_lesson_like_rows, 2
                    )
                    if interaction_lesson_like_rows
                    else 0.0,
                },
                "mari": {
                    "note": "Mari foi materializada em tabelas auxiliares separadas. Nao entra no core principal nem na base_modelada_v2 por cautela semantica.",
                },
                "personas": {
                    "note": "O pacote relevante agora inclui painel densificado, marts de persona e marts cluster_ready; comportamento vem de interactions e entry fica como telemetria auxiliar explícita.",
                    "teacher_month_panel": "mart_teacher_month_panel",
                    "teacher_month_mart": "mart_teacher_month_persona_ready",
                    "teacher_mart": "mart_teacher_persona_ready",
                    "teacher_month_cluster_mart": "mart_teacher_month_cluster_ready",
                    "teacher_cluster_mart": "mart_teacher_cluster_ready",
                    "feature_readiness": "audit_persona_feature_readiness",
                    "range_candidates": "dim_persona_range_candidates",
                },
            },
            "raw_registered_matched_null_timestamp_excluded": int(
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
            ),
        }
        write_json(
            cfg.output_dir / "tabelas_relevantes" / "manifesto_tabelas_relevantes_v2.json",
            table_groups["tabelas_relevantes"],
        )
        write_json(
            cfg.output_dir / "tabelas_auxiliares" / "manifesto_tabelas_auxiliares_v2.json",
            table_groups["tabelas_auxiliares"],
        )
        write_json(cfg.output_dir / "json" / "base_modelada_single_file_summary_v2.json", summary)
        write_markdown(
            cfg.output_dir / "audit" / "base_modelada_single_file_validation_v2.md",
            [
                "# Base modelada v2 - validacao do arquivo unico",
                "",
                f"- DuckDB: `{summary['duckdb_path']}`",
                f"- Parquet: `{summary['exports']['parquet']}`",
                f"- Validation status: `{summary['validation_status']}`",
                f"- Tabelas relevantes: `{cfg.output_dir / 'tabelas_relevantes' / 'manifesto_tabelas_relevantes_v2.json'}`",
                f"- Tabelas auxiliares: `{cfg.output_dir / 'tabelas_auxiliares' / 'manifesto_tabelas_auxiliares_v2.json'}`",
                "",
                *[
                    f"- {row['check_name']}: {row['status']} ({row['metric_value']}) - {row['note']}"
                    for row in validation_df.to_dict(orient="records")
                ],
            ],
        )
        write_markdown(
            cfg.output_dir / "tabelas_relevantes" / "guia_uso_tabelas_relevantes_v2.md",
            [
                "# Guia rapido das tabelas modeladas v2",
                "",
                "## Tabelas centrais",
                "",
                "- `base_modelada_v2`: tabela principal no grain `teacher_unique_id x month`, pronta para cohorts, retorno, abandono, strict_user, EDA e modelagem.",
                "- `dim_teacher`: atributos do professor e agregados de uso no horizonte completo observado.",
                "- `fct_teacher_month`: fato mensal observado com comportamento e telemetria de entry separados.",
                "- `mart_teacher_month_panel`: painel densificado para meses com e sem sinal, base correta para churn, gaps e abandono.",
                "- `mart_teacher_month_persona_ready`: camada mensal curada para construir personas por faixas e comportamento.",
                "- `mart_teacher_persona_ready`: rollup por professor pronto para personas no nível teacher.",
                "- `mart_teacher_month_cluster_ready`: camada mensal enxuta para clustering comportamental.",
                "- `mart_teacher_cluster_ready`: rollup por professor pronto para clustering e comparação de perfis.",
                "- `audit_persona_feature_readiness`: tabela que diz quais features são adequadas para personas, ranges e clustering comportamental.",
                "- `dim_persona_range_candidates`: quantis e candidatos de faixas data-driven para as principais variáveis comportamentais.",
                "- `fct_interaction_clean`: interacoes limpas para drilldown, taxonomia de eventos, views e downloads.",
                "- `fct_session_clean`: telemetria limpa de entry para duração e cobertura de sessão; não tratar como verdade única de acesso.",
                "- `fct_formation_clean`: fato de formacao resolvido por professor, mantido como tabela relevante secundaria.",
                "- `dim_event`, `dim_device`, `dim_calendar`: dimensoes de apoio para enriquecimento semantico e temporal.",
                "",
                "## Tabelas auxiliares",
                "",
                "- `dim_lesson`: auxiliar e secundaria; parte do universo observado de `id_aula` validos nas interacoes e so usa `stg_lessons` como enriquecimento quando ha match.",
                "- `bridge_teacher_identity_audit`: auditoria de cobertura e resolucao de chaves.",
                "- `bridge_mari_conversation_teacher` e fatos `Mari`: ficam fora do core por cautela semantica e ambiguidade de chave.",
                "- `fct_session_raw`: tabela auditavel para diagnostico de ping, timestamp ausente e match de professor.",
                "",
                "## Definicoes fixas",
                "",
                "- `active_user`: fez algo na plataforma no mes.",
                "- `strict_value`: `download_aula` ou `download_plano_aula`.",
                "- `strict_user`: fez `strict_value` em `t` e retornou ativo em `t+1`.",
                "- `strict_download_count_month`: contagem mensal de downloads strict, separada da flag.",
                "- `persona_analysis_eligible_flag`: mês elegível para personas, equivalente a `active_user_flag`.",
                "- `month_signal_class`: reconciliação entre behavior via interactions e telemetria de entry no mês observado.",
                "- `no_signal_month_flag`: só existe no painel densificado e representa ausência real de sinal no mês.",
                "",
                "## Caveats",
                "",
                "- `teacher_estado` deve ser usado junto com `is_estado_missing`.",
                "- duração de sessão é telemetria auxiliar de `entry`; usar sempre com `month_signal_class`, `entry_signal_flag` e `clean_entry_signal_flag`.",
                "- `dim_lesson` nao deve ser tratada como chave primária de confianca da análise; o core e `dim_teacher` + fatos limpos.",
            ],
        )
        return summary
    finally:
        conn.close()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config(args)
    summary = run(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
