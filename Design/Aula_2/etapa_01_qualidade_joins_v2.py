#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    V2Config,
    build_config,
    connect_duckdb,
    ensure_output_dirs,
    fmt_pct,
    q,
    register_raw_views,
    setup_logging,
    sql_id_aula_semantic_expr,
    utc_now_iso,
    write_df_bundle,
    write_json,
    write_markdown,
)


TABLE_SPECS = {
    "raw_dim_teachers": {"key_column": "unique_id", "ts_columns": ["data_entrada"]},
    "raw_entries": {"key_column": "unique_id", "ts_columns": ["data_inicio", "data_fim"]},
    "raw_interactions": {"key_column": "unique_id", "ts_columns": ["data_inicio"]},
    "raw_lessons": {"key_column": "id_aula", "ts_columns": []},
    "raw_formation": {"key_column": "unique_id_aprendizap", "ts_columns": ["createdat", "updatedat"]},
    "raw_mari_conv": {"key_column": "id_mari", "ts_columns": ["createdat", "updatedat"]},
    "raw_mari_reports": {"key_column": "id_mari", "ts_columns": ["updatedat"]},
    "raw_mari_help": {"key_column": "user_id", "ts_columns": ["date"]},
    "raw_school_calendar": {"key_column": "month_start", "ts_columns": ["month_start"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 01 v2: auditoria raw e validação de joins.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def table_inventory(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for table_name, spec in TABLE_SPECS.items():
        key_column = spec["key_column"]
        ts_columns = spec["ts_columns"]
        min_ts = None
        max_ts = None
        if ts_columns:
            min_expr = "LEAST(" + ", ".join([f"MIN({col})" for col in ts_columns]) + ")"
            max_expr = "GREATEST(" + ", ".join([f"MAX({col})" for col in ts_columns]) + ")"
            try:
                ts_row = conn.execute(f"SELECT {min_expr} AS min_ts, {max_expr} AS max_ts FROM {table_name}").fetchone()
                min_ts, max_ts = ts_row
            except duckdb.Error:
                min_ts = None
                max_ts = None
        row_count, distinct_keys = conn.execute(
            f"SELECT COUNT(*) AS row_count, COUNT(DISTINCT {key_column}) AS distinct_keys FROM {table_name}"
        ).fetchone()
        rows.append(
            {
                "table_name": table_name,
                "key_column": key_column,
                "row_count": int(row_count),
                "distinct_keys": int(distinct_keys),
                "min_ts": min_ts,
                "max_ts": max_ts,
            }
        )
    return pd.DataFrame(rows)


def nulls_missing_profile(conn: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    columns_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
    row_count = int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] or 0)
    rows: List[Dict[str, Any]] = []
    for _, col_row in columns_df.iterrows():
        col = str(col_row["column_name"])
        col_type = str(col_row["column_type"]).upper()
        blank_expr = f"SUM(CASE WHEN trim(CAST({col} AS VARCHAR))='' THEN 1 ELSE 0 END)"
        if "CHAR" not in col_type and "VARCHAR" not in col_type:
            blank_expr = "0"
        query = f"""
        SELECT
          COUNT(*) AS row_count,
          SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
          {blank_expr} AS blank_count,
          COUNT(DISTINCT {col}) AS distinct_count
        FROM {table_name}
        """
        out = conn.execute(query).fetchone()
        null_count = int(out[1] or 0)
        blank_count = int(out[2] or 0)
        rows.append(
            {
                "table_name": table_name,
                "column_name": col,
                "column_type": col_type,
                "row_count": row_count,
                "null_count": null_count,
                "blank_count": blank_count,
                "missing_count": null_count + blank_count,
                "missing_rate": float((null_count + blank_count) / row_count) if row_count else None,
                "distinct_count": int(out[3] or 0),
            }
        )
    return pd.DataFrame(rows)


def business_rules(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add_rule(rule_name: str, sql: str, expected: str, severity: str, table_name: str) -> None:
        value = conn.execute(sql).fetchone()[0]
        rows.append(
            {
                "table_name": table_name,
                "rule_name": rule_name,
                "metric_value": float(value or 0.0),
                "expected": expected,
                "severity": severity,
            }
        )

    add_rule(
        "entries_negative_duration_count",
        "SELECT COUNT(*) FROM raw_entries WHERE data_inicio IS NOT NULL AND data_fim IS NOT NULL AND data_fim < data_inicio",
        "esperado_zero",
        "critical",
        "raw_entries",
    )
    add_rule(
        "entries_missing_timestamp_rate",
        "SELECT AVG(CASE WHEN data_inicio IS NULL OR data_fim IS NULL THEN 1.0 ELSE 0.0 END) FROM raw_entries",
        "esperado_baixo",
        "major",
        "raw_entries",
    )
    add_rule(
        "interactions_missing_timestamp_rate",
        "SELECT AVG(CASE WHEN data_inicio IS NULL THEN 1.0 ELSE 0.0 END) FROM raw_interactions",
        "esperado_zero",
        "critical",
        "raw_interactions",
    )
    add_rule(
        "interactions_missing_event_type_rate",
        "SELECT AVG(CASE WHEN event_type IS NULL OR trim(event_type)='' THEN 1.0 ELSE 0.0 END) FROM raw_interactions",
        "esperado_baixo",
        "major",
        "raw_interactions",
    )
    add_rule(
        "dim_negative_total_alunos_rate",
        "SELECT AVG(CASE WHEN total_alunos < 0 THEN 1.0 ELSE 0.0 END) FROM raw_dim_teachers",
        "esperado_zero",
        "critical",
        "raw_dim_teachers",
    )
    add_rule(
        "dim_negative_alunos_indiretos_rate",
        "SELECT AVG(CASE WHEN alunos_indiretos < 0 THEN 1.0 ELSE 0.0 END) FROM raw_dim_teachers",
        "esperado_zero",
        "critical",
        "raw_dim_teachers",
    )
    add_rule(
        "dim_invalid_login_google_rate",
        "SELECT AVG(CASE WHEN login_google IS NOT NULL AND login_google NOT IN (0,1) THEN 1.0 ELSE 0.0 END) FROM raw_dim_teachers",
        "esperado_zero",
        "major",
        "raw_dim_teachers",
    )
    add_rule(
        "dim_total_diff_direct_plus_indirect_rate",
        "SELECT AVG(CASE WHEN total_alunos IS NOT NULL AND alunos_diretos IS NOT NULL AND alunos_indiretos IS NOT NULL AND total_alunos <> alunos_diretos + alunos_indiretos THEN 1.0 ELSE 0.0 END) FROM raw_dim_teachers",
        "esperado_baixo",
        "major",
        "raw_dim_teachers",
    )

    return pd.DataFrame(rows)


def ping_sensitivity(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH base AS (
          SELECT
            lower(coalesce(e.user_type,'missing')) AS user_type,
            CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS matched_teacher,
            GREATEST(epoch(e.data_fim) - epoch(e.data_inicio), 0) AS duration_sec
          FROM raw_entries e
          LEFT JOIN raw_dim_teachers d USING(unique_id)
          WHERE e.data_inicio IS NOT NULL AND e.data_fim IS NOT NULL
        ),
        thresholds AS (
          SELECT * FROM (VALUES (1), (5), (10)) AS t(threshold_sec)
        )
        SELECT
          threshold_sec,
          user_type,
          matched_teacher,
          COUNT(*) AS rows_total,
          SUM(CASE WHEN duration_sec <= threshold_sec THEN 1 ELSE 0 END) AS ping_rows,
          AVG(CASE WHEN duration_sec <= threshold_sec THEN 1.0 ELSE 0.0 END) AS ping_rate,
          quantile_cont(duration_sec, 0.5) AS median_duration_sec,
          quantile_cont(duration_sec, 0.9) AS p90_duration_sec
        FROM base
        CROSS JOIN thresholds
        GROUP BY 1, 2, 3
        ORDER BY threshold_sec, rows_total DESC, user_type
        """
    ).fetchdf()


def join_contracts(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add_row(
        contract_name: str,
        source_ids: int,
        matched_ids: int,
        ambiguous_ids: int,
        contract_semantics: str,
        resolution_path: str,
        note: str,
    ) -> None:
        rows.append(
            {
                "contract_name": contract_name,
                "source_ids": int(source_ids),
                "matched_ids": int(matched_ids),
                "coverage_rate": float(matched_ids / source_ids) if source_ids else None,
                "ambiguous_ids": int(ambiguous_ids),
                "contract_semantics": contract_semantics,
                "resolution_path": resolution_path,
                "note": note,
            }
        )

    source_ids, matched_ids = conn.execute(
        "SELECT COUNT(DISTINCT unique_id), COUNT(DISTINCT e.unique_id) FILTER (WHERE d.unique_id IS NOT NULL) FROM raw_entries e LEFT JOIN raw_dim_teachers d USING(unique_id)"
    ).fetchone()
    add_row("entries.unique_id -> dim_teachers.unique_id", source_ids, matched_ids, 0, "exact", "unique_id", "Join exato por chave UUID36.")

    source_ids, matched_ids = conn.execute(
        "SELECT COUNT(DISTINCT unique_id), COUNT(DISTINCT i.unique_id) FILTER (WHERE d.unique_id IS NOT NULL) FROM raw_interactions i LEFT JOIN raw_dim_teachers d USING(unique_id)"
    ).fetchone()
    add_row("interactions.unique_id -> dim_teachers.unique_id", source_ids, matched_ids, 0, "exact", "unique_id", "Join exato por chave UUID36.")

    source_ids, matched_ids = conn.execute(
        "SELECT COUNT(DISTINCT unique_id_aprendizap), COUNT(DISTINCT f.unique_id_aprendizap) FILTER (WHERE d.unique_id IS NOT NULL) FROM raw_formation f LEFT JOIN raw_dim_teachers d ON f.unique_id_aprendizap=d.unique_id"
    ).fetchone()
    add_row("formation.unique_id_aprendizap -> dim_teachers.unique_id", source_ids, matched_ids, 0, "exact_same_domain", "unique_id_aprendizap = unique_id", "Domínio de UUID36 semanticamente consistente.")

    source_ids, matched_ids = conn.execute(
        "SELECT COUNT(DISTINCT unique_id_aprendizap), COUNT(DISTINCT m.unique_id_aprendizap) FILTER (WHERE d.unique_id IS NOT NULL) FROM raw_mari_conv m LEFT JOIN raw_dim_teachers d ON m.unique_id_aprendizap=d.unique_id"
    ).fetchone()
    add_row("mari_conv.unique_id_aprendizap -> dim_teachers.unique_id", source_ids, matched_ids, 0, "exact_same_domain", "unique_id_aprendizap = unique_id", "Domínio de UUID36 semanticamente consistente.")

    source_ids, matched_ids = conn.execute(
        "SELECT COUNT(DISTINCT unique_id_aprendizap), COUNT(DISTINCT m.unique_id_aprendizap) FILTER (WHERE d.unique_id IS NOT NULL) FROM raw_mari_reports m LEFT JOIN raw_dim_teachers d ON m.unique_id_aprendizap=d.unique_id"
    ).fetchone()
    add_row("mari_reports.unique_id_aprendizap -> dim_teachers.unique_id", source_ids, matched_ids, 0, "exact_same_domain", "unique_id_aprendizap = unique_id", "Cobertura maior que mari_conv, mas com conflitos possíveis por id_mari.")

    source_ids, matched_ids = conn.execute(
        "SELECT COUNT(DISTINCT user_id), COUNT(DISTINCT h.user_id) FILTER (WHERE d.unique_id IS NOT NULL) FROM raw_mari_help h LEFT JOIN raw_dim_teachers d ON h.user_id=d.unique_id"
    ).fetchone()
    add_row("mari_help.user_id -> dim_teachers.unique_id", source_ids, matched_ids, 0, "invalid_direct", "não permitido", "Domínio real é hash de conversa, não UUID de professor.")

    source_ids, matched_ids, ambiguous_ids = conn.execute(
        """
        WITH bridge AS (
          SELECT
            h.user_id,
            COUNT(DISTINCT r.unique_id_aprendizap) FILTER (WHERE r.unique_id_aprendizap IS NOT NULL) AS teacher_count
          FROM raw_mari_help h
          LEFT JOIN raw_mari_reports r ON h.user_id=r.id_mari
          GROUP BY 1
        )
        SELECT
          COUNT(*) AS source_ids,
          SUM(CASE WHEN teacher_count=1 THEN 1 ELSE 0 END) AS matched_ids,
          SUM(CASE WHEN teacher_count>1 THEN 1 ELSE 0 END) AS ambiguous_ids
        FROM bridge
        """
    ).fetchone()
    add_row(
        "mari_help.user_id -> mari_reports.id_mari -> teacher",
        source_ids,
        matched_ids,
        ambiguous_ids,
        "semantic_bridge_unambiguous_only",
        "user_id -> id_mari -> unique_id_aprendizap -> unique_id",
        "Aceitar só resoluções unívocas; ambiguidades ficam fora do core.",
    )

    source_ids, matched_ids = conn.execute(
        f"""
        WITH base AS (
          SELECT
            {sql_id_aula_semantic_expr('i.id_aula')} AS semantic_class,
            i.id_aula,
            l.id_aula AS lesson_id
          FROM raw_interactions i
          LEFT JOIN raw_lessons l ON i.id_aula=l.id_aula
        )
        SELECT
          COUNT(DISTINCT id_aula) FILTER (WHERE semantic_class='lesson_like_22char') AS source_ids,
          COUNT(DISTINCT id_aula) FILTER (WHERE semantic_class='lesson_like_22char' AND lesson_id IS NOT NULL) AS matched_ids
        FROM base
        """
    ).fetchone()
    add_row(
        "interactions.id_aula(valid_semantic) -> lessons.id_aula",
        source_ids,
        matched_ids,
        0,
        "semantic_only",
        "id_aula regex 22 chars",
        "Placeholders e tokens não entram como aula.",
    )

    return pd.DataFrame(rows)


def join_match_by_month(conn: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    return conn.execute(
        f"""
        WITH x AS (
          SELECT
            date_trunc('month', t.data_inicio) AS month,
            t.unique_id,
            lower(coalesce(t.user_type,'missing')) AS user_type,
            CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END AS matched
          FROM {table_name} t
          LEFT JOIN raw_dim_teachers d USING(unique_id)
          WHERE t.data_inicio IS NOT NULL
        )
        SELECT
          '{table_name}' AS source_table,
          month,
          user_type,
          COUNT(*) AS rows_total,
          COUNT(DISTINCT unique_id) AS source_ids,
          COUNT(DISTINCT unique_id) FILTER (WHERE matched=1) AS matched_ids,
          AVG(CASE WHEN matched=1 THEN 1.0 ELSE 0.0 END) AS matched_row_rate,
          COUNT(DISTINCT unique_id) FILTER (WHERE matched=1) * 1.0 / NULLIF(COUNT(DISTINCT unique_id),0) AS matched_id_rate
        FROM x
        GROUP BY 1, 2, 3
        ORDER BY month, user_type
        """
    ).fetchdf()


def mari_bridge_quality(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH bridged AS (
          SELECT
            h.user_id,
            COUNT(DISTINCT r.unique_id_aprendizap) FILTER (WHERE r.unique_id_aprendizap IS NOT NULL) AS teacher_resolution_count,
            COUNT(*) AS joined_rows
          FROM raw_mari_help h
          LEFT JOIN raw_mari_reports r ON h.user_id=r.id_mari
          GROUP BY 1
        )
        SELECT
          teacher_resolution_count,
          COUNT(*) AS ids,
          AVG(joined_rows) AS avg_joined_rows,
          MAX(joined_rows) AS max_joined_rows
        FROM bridged
        GROUP BY 1
        ORDER BY teacher_resolution_count
        """
    ).fetchdf()


def id_aula_semantics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        f"""
        WITH base AS (
          SELECT
            {sql_id_aula_semantic_expr('i.id_aula')} AS id_aula_semantic,
            coalesce(i.event_type, '<missing>') AS event_type,
            i.id_aula,
            l.id_aula AS lesson_id
          FROM raw_interactions i
          LEFT JOIN raw_lessons l ON i.id_aula=l.id_aula
        )
        SELECT
          id_aula_semantic,
          event_type,
          COUNT(*) AS rows_total,
          COUNT(DISTINCT id_aula) FILTER (WHERE id_aula IS NOT NULL) AS distinct_ids,
          AVG(CASE WHEN lesson_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS lesson_mapping_rate_rows
        FROM base
        GROUP BY 1, 2
        ORDER BY rows_total DESC, id_aula_semantic, event_type
        """
    ).fetchdf()


def content_type_anomalies(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
          coalesce(content_type, '<missing>') AS content_type,
          COUNT(*) AS rows_total,
          COUNT(DISTINCT event_type) AS distinct_event_types,
          COUNT(DISTINCT id_aula) FILTER (WHERE id_aula IS NOT NULL) AS distinct_id_aula
        FROM raw_interactions
        GROUP BY 1
        HAVING content_type LIKE '%utm%' OR content_type LIKE '%202%22%' OR content_type LIKE '%aprendizap.com.br%' OR content_type LIKE '% %' OR content_type IN ('pagina_metodologia_ativa', 'pdf.pdf')
        ORDER BY rows_total DESC
        """
    ).fetchdf()


def teacher_side_coverage(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH base AS (
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
          COUNT(*) AS teachers_total,
          SUM(has_registered_entry) AS teachers_with_registered_entry,
          SUM(has_registered_interaction) AS teachers_with_registered_interaction,
          SUM(has_registered_entry=1 AND has_registered_interaction=1) AS teachers_with_both_entry_and_interaction,
          SUM(has_registered_entry=0 AND has_registered_interaction=1) AS teachers_with_interaction_only,
          SUM(has_registered_entry=1 AND has_registered_interaction=0) AS teachers_with_entry_only,
          SUM(has_registered_entry=0 AND has_registered_interaction=0) AS teachers_with_neither,
          SUM(has_formation) AS teachers_with_formation,
          SUM(has_mari_conv) AS teachers_with_mari_conv,
          SUM(has_mari_reports) AS teachers_with_mari_reports
        FROM base
        """
    ).fetchdf()


def unmatched_examples(conn: duckdb.DuckDBPyConnection, table_name: str, limit_rows: int = 20) -> pd.DataFrame:
    return conn.execute(
        f"""
        SELECT
          t.unique_id,
          lower(coalesce(t.user_type,'missing')) AS user_type,
          COUNT(*) AS rows_total,
          MIN(t.data_inicio) AS first_ts,
          MAX(t.data_inicio) AS last_ts
        FROM {table_name} t
        LEFT JOIN raw_dim_teachers d USING(unique_id)
        WHERE lower(coalesce(t.user_type,''))='registered'
          AND d.unique_id IS NULL
        GROUP BY 1, 2
        ORDER BY rows_total DESC
        LIMIT {int(limit_rows)}
        """
    ).fetchdf()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    paths = ensure_output_dirs(cfg.output_dir)
    conn = connect_duckdb(cfg)
    try:
        register_raw_views(conn, cfg.data_dir)

        tables = {
            "audit_table_inventory": table_inventory(conn),
            "audit_nulls_missing": pd.concat([nulls_missing_profile(conn, table) for table in TABLE_SPECS], ignore_index=True),
            "audit_business_rules": business_rules(conn),
            "audit_ping_sensitivity": ping_sensitivity(conn),
            "audit_join_contracts": join_contracts(conn),
            "audit_join_match_by_month": pd.concat(
                [
                    join_match_by_month(conn, "raw_entries"),
                    join_match_by_month(conn, "raw_interactions"),
                ],
                ignore_index=True,
            ),
            "audit_mari_bridge_quality": mari_bridge_quality(conn),
            "audit_id_aula_semantics": id_aula_semantics(conn),
            "audit_content_type_anomalies": content_type_anomalies(conn),
            "audit_teacher_side_coverage": teacher_side_coverage(conn),
            "audit_unmatched_registered_interactions_examples": unmatched_examples(conn, "raw_interactions"),
            "audit_unmatched_registered_entries_examples": unmatched_examples(conn, "raw_entries"),
        }

        for name, df in tables.items():
            write_df_bundle(cfg.output_dir, name, df)

        dim_max = tables["audit_table_inventory"].loc[
            tables["audit_table_inventory"]["table_name"] == "raw_dim_teachers", "max_ts"
        ].iloc[0]
        reg_inter_match = tables["audit_join_match_by_month"]
        reg_inter_match = reg_inter_match[
            (reg_inter_match["source_table"] == "raw_interactions") & (reg_inter_match["user_type"] == "registered")
        ].copy()
        reg_inter_match["month"] = pd.to_datetime(reg_inter_match["month"], errors="coerce")
        reg_inter_match = reg_inter_match.sort_values("month")
        drop = reg_inter_match[reg_inter_match["matched_id_rate"] < 0.80].head(1)
        first_drop_month = str(drop["month"].dt.strftime("%Y-%m").iloc[0]) if not drop.empty else None

        ping_5s = tables["audit_ping_sensitivity"]
        ping_5s = ping_5s[
            (ping_5s["threshold_sec"] == 5)
            & (ping_5s["user_type"] == "registered")
            & (ping_5s["matched_teacher"] == 1)
        ]
        ping_5s_rate = float(ping_5s["ping_rate"].iloc[0]) if not ping_5s.empty else None

        mari_bridge = tables["audit_join_contracts"]
        mari_bridge = mari_bridge[mari_bridge["contract_name"] == "mari_help.user_id -> mari_reports.id_mari -> teacher"]
        mari_bridge_coverage = float(mari_bridge["coverage_rate"].iloc[0]) if not mari_bridge.empty else None
        mari_bridge_ambiguity = int(mari_bridge["ambiguous_ids"].iloc[0]) if not mari_bridge.empty else None

        summary = {
            "generated_at_utc": utc_now_iso(),
            "dim_teachers_max_data_entrada": str(dim_max),
            "registered_match_drop_below_80pct_month": first_drop_month,
            "registered_core_ping_rate_le_5s": ping_5s_rate,
            "mari_help_semantic_bridge_coverage": mari_bridge_coverage,
            "mari_help_semantic_bridge_ambiguous_ids": mari_bridge_ambiguity,
            "null_profile_rows": int(len(tables["audit_nulls_missing"])),
            "join_contracts_rows": int(len(tables["audit_join_contracts"])),
        }
        write_json(paths["json"] / "audit_quality_summary_v2.json", summary)

        md_lines = [
            "# Auditoria raw e joins v2",
            "",
            f"- Gerado em UTC: {summary['generated_at_utc']}",
            f"- Última `data_entrada` no cadastro: {summary['dim_teachers_max_data_entrada']}",
            f"- Primeiro mês com `match_rate_ids < 80%` para `registered` em interações: {summary['registered_match_drop_below_80pct_month']}",
            f"- Taxa de sessões ping `<=5s` no core `registered+matched`: {fmt_pct(summary['registered_core_ping_rate_le_5s'], 2)}",
            f"- Cobertura da ponte semântica `mari_help -> teacher`: {fmt_pct(summary['mari_help_semantic_bridge_coverage'], 2)}",
            f"- IDs ambíguos nessa ponte: {summary['mari_help_semantic_bridge_ambiguous_ids']}",
            "",
            "## Artefatos principais",
            "- `audit_table_inventory`",
            "- `audit_nulls_missing`",
            "- `audit_business_rules`",
            "- `audit_ping_sensitivity`",
            "- `audit_join_contracts`",
            "- `audit_join_match_by_month`",
            "- `audit_mari_bridge_quality`",
            "- `audit_id_aula_semantics`",
            "- `audit_content_type_anomalies`",
            "- `audit_teacher_side_coverage`",
        ]
        write_markdown(paths["audit"] / "audit_quality_summary_v2.md", md_lines)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
