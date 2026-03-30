#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    V2Config,
    build_config,
    ensure_output_dirs,
    setup_logging,
    utc_now_iso,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 00 v2: inventário do legado analítico.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_chart_lineage(script_path: Path) -> pd.DataFrame:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CHART_LINEAGE":
                    payload = ast.literal_eval(node.value)
                    rows: List[Dict[str, Any]] = []
                    for chart_id, meta in payload.items():
                        rows.append(
                            {
                                "chart_id": chart_id,
                                "como_foi_gerado": meta.get("como_foi_gerado"),
                                "tabelas_usadas": meta.get("tabelas_usadas"),
                                "colunas_chave": meta.get("colunas_chave"),
                                "transformacoes_joins": meta.get("transformacoes_joins"),
                            }
                        )
                    return pd.DataFrame(rows)
    return pd.DataFrame(columns=["chart_id", "como_foi_gerado", "tabelas_usadas", "colunas_chave", "transformacoes_joins"])


def extract_html_headings(path: Path) -> List[str]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", content, flags=re.IGNORECASE | re.DOTALL)
    cleaned = [re.sub(r"<[^>]+>", "", h).strip() for h in headings]
    return [h for h in cleaned if h]


def extract_titles(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    title_match = re.search(r"<title>(.*?)</title>", content, flags=re.IGNORECASE | re.DOTALL)
    return {
        "file_name": path.name,
        "title": title_match.group(1).strip() if title_match else "Sem título",
        "headings": extract_html_headings(path),
    }


def extract_pipeline_steps(path: Path) -> pd.DataFrame:
    content = path.read_text(encoding="utf-8")
    matches = re.findall(r"run_step\(\s*config,\s*'([^']+)'", content)
    rows = [{"order": i + 1, "script_name": name} for i, name in enumerate(matches)]
    return pd.DataFrame(rows)


def script_inventory(base_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for script_name in [
        "executar_pipeline_analytics.py",
        "etapa_01_base.py",
        "etapa_02_deep_dive.py",
        "etapa_03_relatorio.py",
        "etapa_04_metricas_mensais.py",
    ]:
        path = base_dir / script_name
        rows.append(
            {
                "script_name": script_name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    paths = ensure_output_dirs(cfg.output_dir)

    legacy_reports_dir = cfg.base_dir / "analysis_output" / "reports"
    report_files = [
        legacy_reports_dir / "analise_inicial_dos_dados_interativa.html",
        legacy_reports_dir / "analise_inicial_da_metrica_interativa.html",
    ]
    report_inventory = pd.DataFrame([extract_titles(path) for path in report_files if path.exists()])
    if not report_inventory.empty:
        report_inventory["headings_count"] = report_inventory["headings"].apply(len)
        report_inventory["headings"] = report_inventory["headings"].apply(lambda values: " | ".join(values))

    chart_lineage = load_chart_lineage(cfg.base_dir / "etapa_03_relatorio.py")
    pipeline_steps = extract_pipeline_steps(cfg.base_dir / "executar_pipeline_analytics.py")
    scripts_df = script_inventory(cfg.base_dir)

    report_inventory.to_csv(paths["audit"] / "legacy_report_inventory.csv", index=False)
    chart_lineage.to_csv(paths["audit"] / "legacy_chart_lineage.csv", index=False)
    pipeline_steps.to_csv(paths["audit"] / "legacy_pipeline_steps.csv", index=False)
    scripts_df.to_csv(paths["audit"] / "legacy_script_inventory.csv", index=False)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "legacy_reports_found": int(len(report_inventory)),
        "legacy_chart_lineage_rows": int(len(chart_lineage)),
        "legacy_pipeline_steps": pipeline_steps["script_name"].tolist(),
        "preserved_visual_reference": [
            "HTML em português",
            "cartões-resumo em grid",
            "seções por narrativa analítica",
            "blocos com explicação metodológica e linhagem",
        ],
    }
    write_json(paths["json"] / "legacy_inventory_summary.json", summary)

    md_lines = [
        "# Inventário do legado analítico (v2)",
        "",
        f"- Gerado em UTC: {summary['generated_at_utc']}",
        f"- Relatórios legacy encontrados: {summary['legacy_reports_found']}",
        f"- Blocos de lineage recuperados: {summary['legacy_chart_lineage_rows']}",
        "",
        "## Pipeline legado",
    ]
    for _, row in pipeline_steps.iterrows():
        md_lines.append(f"- Etapa {int(row['order'])}: `{row['script_name']}`")
    md_lines.extend(
        [
            "",
            "## Preservar apenas como referência",
        ]
    )
    for item in summary["preserved_visual_reference"]:
        md_lines.append(f"- {item}")
    if not report_inventory.empty:
        md_lines.extend(
            [
                "",
                "## Relatórios legacy encontrados",
            ]
        )
        for _, row in report_inventory.iterrows():
            md_lines.append(f"- `{row['file_name']}`: {row['title']}")
    write_markdown(paths["audit"] / "legacy_inventory.md", md_lines)


if __name__ == "__main__":
    main()
