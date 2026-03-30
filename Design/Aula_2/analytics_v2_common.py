from __future__ import annotations

import html
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import duckdb
import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs


LOGGER = logging.getLogger("analytics_v2")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")

STRICT_VALUE_EVENTS: List[str] = ["download_aula", "download_plano_aula"]
NON_ACTIVITY_EVENTS: set[str] = {
    "",
    "acesso_aba_conquistas",
    "fechar_conquista_obtida",
}
VALID_LESSON_ID_RE = r"^[A-Za-z0-9]{22}$"
UUID36_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
HEX64_UPPER_RE = r"^[0-9A-F]{64}$"
PALETTE = ["#0B132B", "#1C2541", "#3A506B", "#5BC0BE", "#CDEFF0", "#E76F51", "#B23A48"]


@dataclass(frozen=True)
class V2Config:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    duckdb_path: Path
    random_seed: int = 42


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def build_config(base_dir: Path | None = None, data_dir: Path | None = None, output_dir: Path | None = None) -> V2Config:
    base = (base_dir or DEFAULT_BASE_DIR).resolve()
    data = (data_dir or base / "base_aprendizap").resolve()
    out = (output_dir or base / "analysis_output_v2").resolve()
    duckdb_path = out / "duckdb" / "aprendizap_v2.duckdb"
    return V2Config(
        base_dir=base,
        data_dir=data,
        output_dir=out,
        duckdb_path=duckdb_path,
    )


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "audit": output_dir / "audit",
        "csv": output_dir / "csv",
        "parquet": output_dir / "parquet",
        "reports": output_dir / "reports",
        "excel": output_dir / "excel",
        "json": output_dir / "json",
        "duckdb": output_dir / "duckdb",
        "verification": output_dir / "verification",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def connect_duckdb(cfg: V2Config, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    ensure_output_dirs(cfg.output_dir)
    conn = duckdb.connect(database=str(cfg.duckdb_path), read_only=read_only)
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
        raise FileNotFoundError(f"Arquivos ausentes em {data_dir}: {', '.join(sorted(missing))}")

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


def normalize_device(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    lower = str(value).strip().lower()
    if lower in {"desktop", "mobile", "tablet"}:
        return lower
    return "unknown"


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


def classify_event_family(event_type: Any) -> str:
    if event_type is None or (isinstance(event_type, float) and np.isnan(event_type)):
        return "missing"
    lower = str(event_type).strip().lower()
    if not lower:
        return "missing"
    if "plano" in lower:
        return "plano"
    if "prova" in lower or "avaliacao" in lower:
        return "prova"
    if "aula" in lower:
        return "aula"
    if "ia" in lower or "mari" in lower:
        return "ia"
    if "metodologia" in lower:
        return "metodologia"
    if "relatorio" in lower:
        return "relatorio"
    if "conquista" in lower:
        return "conquista"
    return "other"


def classify_event_action(event_type: Any) -> str:
    if event_type is None or (isinstance(event_type, float) and np.isnan(event_type)):
        return "missing"
    lower = str(event_type).strip().lower()
    if not lower:
        return "missing"
    if "download" in lower or "baixar" in lower:
        return "download"
    if "visualizacao" in lower or "view" in lower:
        return "view"
    if "criacao" in lower or "criar" in lower or "salva" in lower or "edicao" in lower:
        return "create"
    if "compart" in lower or "envio_email" in lower:
        return "share"
    if "acesso_" in lower or "fechar_" in lower or "botao_" in lower:
        return "navigation"
    return "other"


def id_domain_type(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    if re.fullmatch(UUID36_RE, text):
        return "uuid36"
    if re.fullmatch(HEX64_UPPER_RE, text):
        return "hex64_upper"
    if re.fullmatch(VALID_LESSON_ID_RE, text):
        return "lesson_like_22char"
    if re.fullmatch(r"^[0-9]+$", text):
        return "numeric_only"
    if re.fullmatch(r"^[A-Za-z]+$", text):
        return "alpha_token"
    return "other"


def classify_id_aula_semantic(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    if re.fullmatch(VALID_LESSON_ID_RE, text):
        return "lesson_like_22char"
    if re.fullmatch(r"^[0-9]+$", text):
        return "numeric_only"
    if text in {"s", "S"}:
        return "placeholder_s"
    if "conquista" in text.lower():
        return "navigation_token"
    if re.fullmatch(r"^[A-Za-z_]+$", text):
        return "alpha_token"
    return "other"


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
      WHEN lower({column_name}) LIKE '%visualizacao%' THEN 'view'
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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_parquet_duckdb(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(database=":memory:")
    try:
        conn.register("tmp_df", df)
        conn.execute(f"COPY tmp_df TO '{q(path)}' (FORMAT PARQUET)")
    finally:
        conn.close()


def parquet_only_mode() -> bool:
    return os.environ.get("ANALYTICS_V2_PARQUET_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def write_df_bundle(output_dir: Path, name: str, df: pd.DataFrame, subdir: str | None = None) -> Dict[str, str]:
    csv_path = output_dir / "csv" / f"{name}.csv"
    parquet_path = output_dir / "parquet" / f"{name}.parquet"
    if subdir:
        csv_path = output_dir / subdir / "csv" / f"{name}.csv"
        parquet_path = output_dir / subdir / "parquet" / f"{name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_duckdb(df, parquet_path)
    written = {"parquet": str(parquet_path)}
    if not parquet_only_mode():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        written["csv"] = str(csv_path)
    return written


def persist_df_to_duckdb(conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> None:
    conn.register("_persist_df_bundle", df)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _persist_df_bundle")


def load_table(output_dir: Path, name: str, prefer: str = "parquet") -> pd.DataFrame:
    if prefer == "parquet":
        p = output_dir / "parquet" / f"{name}.parquet"
        if p.exists():
            conn = duckdb.connect(database=":memory:")
            try:
                return conn.execute(f"SELECT * FROM read_parquet('{q(p)}')").fetchdf()
            finally:
                conn.close()
    c = output_dir / "csv" / f"{name}.csv"
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Tabela não encontrada para {name} em {output_dir}")


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def figure_to_html(fig: Any) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False, "responsive": True})


def html_escape(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def build_lineage_html(lineage: Dict[str, Any]) -> str:
    rows = [
        ("Tabelas usadas", lineage.get("tables_used") or lineage.get("raw_tables")),
        ("Origem parquet", lineage.get("parquet_sources")),
        ("População", lineage.get("population")),
        ("Grão", lineage.get("grain")),
        ("Joins", lineage.get("joins")),
        ("Filtros", lineage.get("filters")),
        ("Lógica principal", lineage.get("logic")),
        ("Como reproduzir", lineage.get("rebuild")),
        ("Caveats", lineage.get("caveats")),
    ]
    if not any(value is not None and str(value).strip() != "" for _, value in rows):
        return ""
    parts = ["<div class='lineage'>"]
    for label, value in rows:
        if value is None:
            continue
        parts.append(f"<p><b>{html_escape(label)}:</b> {html_escape(value)}</p>")
    parts.append("</div>")
    return "".join(parts)


def build_card_html(title: str, value: str, subtitle: str | None = None) -> str:
    subtitle_html = f"<p class='small'>{html_escape(subtitle)}</p>" if subtitle else ""
    return (
        "<div class='card'>"
        f"<h4>{html_escape(title)}</h4>"
        f"<div class='value'>{html_escape(value)}</div>"
        f"{subtitle_html}"
        "</div>"
    )


def build_table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p class='small'>Sem dados disponíveis.</p>"
    view = df.head(max_rows).copy()
    return view.to_html(index=False, classes="table", border=0, escape=True)


def render_report_html(
    title: str,
    subtitle: str,
    summary_cards_html: str,
    sections: Sequence[Dict[str, Any]],
) -> str:
    section_parts: List[str] = []
    for section in sections:
        blocks_html = []
        for block in section.get("blocks", []):
            lineage_html = build_lineage_html(block.get("lineage", {}))
            blocks_html.append(
                "<div class='chart-card'>"
                f"<h3>{html_escape(block.get('title', 'Bloco'))}</h3>"
                f"<p class='subtitle'>{html_escape(block.get('subtitle', ''))}</p>"
                f"{block.get('body_html', '')}"
                f"{lineage_html}"
                "</div>"
            )
        section_description = section.get("description", "")
        section_description_html = (
            f"<p class='section-text'>{html_escape(section_description)}</p>" if section_description else ""
        )
        section_parts.append(
            f"<section><h2>{html_escape(section.get('title', 'Seção'))}</h2>"
            f"{section_description_html}"
            f"{''.join(blocks_html)}</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html_escape(title)}</title>
  <script>{get_plotlyjs()}</script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #F7FAFC; color: #1A202C; }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 30px; color: #102A43; }}
    h2 {{ margin-top: 30px; margin-bottom: 8px; font-size: 22px; color: #102A43; }}
    h3 {{ margin: 0 0 8px 0; font-size: 18px; color: #102A43; }}
    h4 {{ margin: 0 0 6px 0; font-size: 13px; color: #486581; }}
    .small {{ color: #4A5568; font-size: 13px; }}
    .section-text {{ color: #627D98; font-size: 14px; margin-top: 0; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; }}
    .card .value {{ font-size: 24px; font-weight: 700; color: #102A43; }}
    .chart-card {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; margin: 14px 0 22px 0; }}
    .subtitle {{ margin: 0 0 10px 0; color: #627D98; font-size: 13px; }}
    .lineage {{ background: #F0F4F8; border: 1px solid #D9E2EC; border-radius: 8px; padding: 10px 12px; margin-top: 12px; }}
    .lineage p {{ margin: 6px 0; font-size: 12px; line-height: 1.45; color: #334E68; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; background: white; }}
    .table th, .table td {{ border: 1px solid #E2E8F0; padding: 8px 10px; text-align: left; }}
    .table th {{ background: #F0F4F8; color: #243B53; }}
    .note {{ background: #E6FFFA; border: 1px solid #81E6D9; border-radius: 8px; padding: 10px 12px; margin: 12px 0; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{html_escape(title)}</h1>
    <p class="small">{html_escape(subtitle)}</p>
    <div class="summary-grid">{summary_cards_html}</div>
    {''.join(section_parts)}
  </div>
</body>
</html>
"""


def build_metric_lineage_rows(items: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for item in items:
        lineage = item.get("lineage", {})
        records.append(
            {
                "artifact_name": item.get("artifact_name"),
                "report_name": item.get("report_name"),
                "artifact_type": item.get("artifact_type"),
                "raw_tables": lineage.get("raw_tables"),
                "population": lineage.get("population"),
                "grain": lineage.get("grain"),
                "joins": lineage.get("joins"),
                "filters": lineage.get("filters"),
                "logic": lineage.get("logic"),
                "caveats": lineage.get("caveats"),
            }
        )
    return pd.DataFrame(records)


def month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: pd.Series, y_score: pd.Series) -> float:
    from sklearn.metrics import average_precision_score

    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def top_decile_lift(y_true: pd.Series, y_score: pd.Series) -> float:
    frame = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
    if frame.empty or frame["y_true"].nunique() < 2:
        return float("nan")
    cutoff = frame["y_score"].quantile(0.9)
    top = frame[frame["y_score"] >= cutoff]
    base_rate = frame["y_true"].mean()
    if base_rate <= 0:
        return float("nan")
    return float(top["y_true"].mean() / base_rate)


def make_quantile_band_labels(values: pd.Series, n_bins: int = 3) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce")
    if cleaned.nunique(dropna=True) < n_bins:
        return pd.Series(["sem_faixa_confiavel"] * len(values), index=values.index, dtype="object")
    try:
        return pd.qcut(cleaned, q=n_bins, labels=[f"faixa_{i+1}" for i in range(n_bins)]).astype("object")
    except ValueError:
        return pd.Series(["sem_faixa_confiavel"] * len(values), index=values.index, dtype="object")
