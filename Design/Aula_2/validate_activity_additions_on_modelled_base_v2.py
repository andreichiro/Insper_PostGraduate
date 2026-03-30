#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import duckdb


DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
DEFAULT_MODELLED_DUCKDB = DEFAULT_BASE_DIR / "analysis_output_v2" / "duckdb" / "base_modelada_v2.duckdb"
CANONICAL_MODELLED_WRAPPER = "gerar_base_modelada_v2.py"
CANONICAL_MODELLED_IMPLEMENTATION = "etapa_02_star_schema_v2.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida em leitura se as extensoes de atividade, navegacao, formacao e Mari IA "
            "podem ser derivadas da base_modelada_v2 atual sem mudar a semantica existente."
        )
    )
    parser.add_argument("--modelled-duckdb", type=Path, default=DEFAULT_MODELLED_DUCKDB)
    return parser.parse_args()


def validate(conn: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    specs = {
        "dim_teacher": ["teacher_unique_id", "estado", "currentstage", "currentsubject_group", "utm_group", "has_formation"],
        "fct_teacher_month": ["teacher_unique_id", "month", "clean_entry_session_count_month", "returned_active_m1"],
        "fct_session_clean": ["teacher_unique_id", "session_start_ts", "session_end_ts", "duration_min"],
        "fct_interaction_clean": ["teacher_unique_id", "interaction_ts", "event_type", "event_family", "event_action", "content_type"],
        "fct_formation_clean": ["teacher_unique_id", "formation_ts", "item_type", "progress"],
        "fct_mari_conversation_resolved_final_v2": ["teacher_unique_id", "mari_created_ts", "origin_source"],
        "fct_mari_help_resolved_final_v2": ["teacher_unique_id", "help_ts", "help_key", "isso_ajudou_num"],
        "fct_mari_reports_resolved_final_v2": ["teacher_unique_id", "report_ts", "report_key"],
    }
    rows: List[Dict[str, Any]] = []
    existing = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    for table_name, required_cols in specs.items():
        exists = table_name in existing
        actual_cols = set()
        row_count = 0
        if exists:
            desc = conn.execute(f"DESCRIBE {table_name}").fetchdf()
            actual_cols = set(desc["column_name"].astype(str))
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] or 0)
        missing_cols = [col for col in required_cols if col not in actual_cols]
        rows.append(
            {
                "object_name": table_name,
                "exists_flag": int(exists),
                "row_count": row_count,
                "missing_required_columns": missing_cols,
                "status": "ready" if exists and not missing_cols else "missing_columns_or_table",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not args.modelled_duckdb.exists():
        raise FileNotFoundError(f"Modelled DuckDB not found: {args.modelled_duckdb}")
    conn = duckdb.connect(str(args.modelled_duckdb), read_only=True)
    rows = validate(conn)
    payload = {
        "modelled_duckdb": str(args.modelled_duckdb.resolve()),
        "canonical_raw_to_modelled_wrapper": CANONICAL_MODELLED_WRAPPER,
        "canonical_raw_to_modelled_implementation": CANONICAL_MODELLED_IMPLEMENTATION,
        "validation_status": "ready" if all(row["status"] == "ready" for row in rows) else "blocked",
        "policy": {
            "change_mode": "additive_only",
            "backward_compatible": True,
            "notes": (
                "Preferir novas tabelas, views ou colunas aditivas. "
                "Nao remover, renomear ou alterar a semantica da base_modelada_v2 existente."
            ),
        },
        "checks": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
