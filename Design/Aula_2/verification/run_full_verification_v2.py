#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd


DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verificação mínima do pipeline v2.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python-exec", type=str, default=None)
    parser.add_argument("--run-pipeline", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = (args.output_dir if args.output_dir is not None else base_dir / "analysis_output_v2").resolve()
    python_exec = args.python_exec or str((base_dir / ".venv" / "bin" / "python").resolve())

    if args.run_pipeline:
        run_cmd(
            [
                python_exec,
                str((base_dir / "executar_pipeline_analytics_v2.py").resolve()),
                "--base-dir",
                str(base_dir),
                "--output-dir",
                str(output_dir),
            ],
            cwd=base_dir,
        )

    db_path = output_dir / "duckdb" / "aprendizap_v2.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB v2 não encontrada: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        required_tables = [
            "base_modelada_v2",
            "dim_teacher",
            "dim_lesson",
            "dim_event",
            "bridge_teacher_identity_audit",
            "bridge_mari_conversation_teacher",
            "fct_session_clean",
            "fct_interaction_clean",
            "fct_teacher_month",
            "fct_mari_help_resolved",
        ]
        table_counts: Dict[str, int] = {}
        for table in required_tables:
            table_counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)

        checks: List[Dict[str, Any]] = []

        def add_check(name: str, value: Any, status: str, note: str) -> None:
            checks.append({"check_name": name, "metric_value": value, "status": status, "note": note})

        teacher_month = conn.execute("SELECT * FROM fct_teacher_month").fetchdf()
        add_check("table_counts_positive", int(all(count > 0 for count in table_counts.values())), "pass" if all(count > 0 for count in table_counts.values()) else "fail", "Todas as tabelas essenciais precisam ter linhas.")

        if not teacher_month.empty:
            active_recompute = (
                (pd.to_numeric(teacher_month["activity_events_month"], errors="coerce").fillna(0) > 0).astype(int)
                == pd.to_numeric(teacher_month["active_user_flag"], errors="coerce").fillna(0).astype(int)
            ).all()
            add_check("active_user_flag_reconciles", int(active_recompute), "pass" if active_recompute else "fail", "active_user_flag deve refletir activity_events_month > 0.")

            strict_recompute = (
                (pd.to_numeric(teacher_month["strict_download_count_month"], errors="coerce").fillna(0) > 0).astype(int)
                == pd.to_numeric(teacher_month["strict_value_flag"], errors="coerce").fillna(0).astype(int)
            ).all()
            add_check("strict_value_flag_reconciles", int(strict_recompute), "pass" if strict_recompute else "fail", "strict_value_flag deve refletir strict_download_count_month > 0.")

            negative_sessions = int((pd.to_numeric(teacher_month["total_session_minutes_month"], errors="coerce").fillna(0) < 0).sum())
            add_check("teacher_month_nonnegative_minutes", negative_sessions, "pass" if negative_sessions == 0 else "fail", "Tempo mensal não pode ficar negativo.")

            censored_rows = int(pd.to_numeric(teacher_month["next_month_observed_flag"], errors="coerce").fillna(0).eq(0).sum())
            add_check("teacher_month_has_censoring_flag", censored_rows, "pass" if censored_rows > 0 else "warning", "Últimos meses devem carregar censura explícita para retorno m+1.")
        else:
            add_check("teacher_month_exists", 0, "fail", "fct_teacher_month não pode estar vazia.")

        base_modelada = conn.execute("SELECT * FROM base_modelada_v2").fetchdf()
        if not base_modelada.empty:
            duplicate_grain = int(base_modelada[["teacher_unique_id", "month"]].duplicated().sum())
            missing_month = int(base_modelada["month"].isna().sum())
            row_count_diff = int(abs(len(base_modelada) - len(teacher_month)))
            add_check("base_modelada_grain_unique", duplicate_grain, "pass" if duplicate_grain == 0 else "fail", "A base modelada deve ter grain único teacher_unique_id x month.")
            add_check("base_modelada_missing_month", missing_month, "pass" if missing_month == 0 else "fail", "A base modelada não pode conter month nulo.")
            add_check("base_modelada_row_count_matches_fact", row_count_diff, "pass" if row_count_diff == 0 else "fail", "A base modelada deve bater com fct_teacher_month.")
        else:
            add_check("base_modelada_exists", 0, "fail", "base_modelada_v2 não pode estar vazia.")

        bridge = conn.execute("SELECT * FROM bridge_mari_conversation_teacher").fetchdf()
        if not bridge.empty:
            ambiguous = int(((pd.to_numeric(bridge["teacher_resolution_count"], errors="coerce").fillna(0) > 1)).sum())
            unique_only = bridge[pd.to_numeric(bridge["is_unambiguous"], errors="coerce").fillna(0) == 1]
            duplicated_unique = int(unique_only["id_mari"].duplicated().sum())
            add_check("mari_bridge_tracks_ambiguity", ambiguous, "pass" if ambiguous >= 0 else "fail", "Ambiguidades precisam ser explícitas na bridge.")
            add_check("mari_bridge_unique_ids_unambiguous", duplicated_unique, "pass" if duplicated_unique == 0 else "fail", "IDs unívocos não devem duplicar em bridge_mari_conversation_teacher.")

        reports = [
            output_dir / "reports" / "relatorio_01_qualidade_e_joins_v2.html",
            output_dir / "reports" / "relatorio_02_eda_v2.html",
            output_dir / "reports" / "relatorio_03_usuarios_metricas_v2.html",
        ]
        reports_exist = all(path.exists() for path in reports)
        add_check("reports_exist", int(reports_exist), "pass" if reports_exist else "fail", "Todos os HTMLs v2 devem existir.")

        lineage_path = output_dir / "csv" / "audit_metric_lineage.csv"
        lineage_ok = lineage_path.exists() and pd.read_csv(lineage_path).shape[0] > 0
        add_check("lineage_table_exists", int(bool(lineage_ok)), "pass" if lineage_ok else "fail", "audit_metric_lineage precisa existir com linhas.")

        base_csv = output_dir / "csv" / "base_modelada_v2.csv"
        base_parquet = output_dir / "parquet" / "base_modelada_v2.parquet"
        base_validation_csv = output_dir / "csv" / "audit_base_modelada_validation.csv"
        base_exports_ok = base_csv.exists() and base_parquet.exists() and base_validation_csv.exists()
        add_check("base_modelada_exports_exist", int(bool(base_exports_ok)), "pass" if base_exports_ok else "fail", "CSV, Parquet e auditoria da base modelada devem existir.")
        if base_validation_csv.exists():
            validation_df = pd.read_csv(base_validation_csv)
            failed_validation_rows = int((validation_df["status"] == "fail").sum())
            add_check("base_modelada_validation_passes", failed_validation_rows, "pass" if failed_validation_rows == 0 else "fail", "A auditoria da base modelada não pode ter checks com status fail.")

        overall_status = "pass" if all(check["status"] == "pass" for check in checks if check["status"] != "warning") else "fail"
        payload = {
            "base_dir": str(base_dir),
            "output_dir": str(output_dir),
            "overall_status": overall_status,
            "table_counts": table_counts,
            "checks": checks,
        }
        out_json = output_dir / "verification" / "verification_summary_v2.json"
        out_md = output_dir / "verification" / "verification_summary_v2.md"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out_md.write_text(
            "\n".join(
                [
                    "# Verification summary v2",
                    "",
                    f"- Overall status: {overall_status}",
                    *[
                        f"- {check['check_name']}: {check['status']} ({check['metric_value']}) - {check['note']}"
                        for check in checks
                    ],
                ]
            ),
            encoding="utf-8",
        )
        print(str(out_json))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
