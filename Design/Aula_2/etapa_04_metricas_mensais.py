from __future__ import annotations

"""
Etapa 04 - Metricas de valor e retencao (versao reconstruida).

Premissas operacionais desta versao:
- strict value e DEFINIDO por regra de negocio: download_aula OU download_plano_aula.
- eventos de visualizacao/click nao contam como strict value.
- analise de uplift por event_type permanece auxiliar (nao altera strict value do KPI).
- segmentacao Aula vs Prova e publicada explicitamente.
"""

import argparse
import base64
import hashlib
import json
import logging
import mimetypes
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LinearRegression

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except Exception:
    plt = None
    mdates = None
    LinearSegmentedColormap = None


LOGGER = logging.getLogger("etapa_04_metricas_mensais")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")

STRICT_DOWNLOAD_EVENTS = ["download_aula", "download_plano_aula"]

METRIC_DISPLAY_BY_CODE: Dict[str, str] = {
    "svs_t": "Value Conversion Rate",
    "sur_t": "Post-Value Retention_t",
    "sur_m2": "Post-Value Retention_m2",
    "sur_m4": "Post-Value Retention_m4",
    "sur_m6": "Post-Value Retention_m6",
    "rsva_m1": "Value-Qualified Retention_m1",
    "rsva_m2": "Value-Qualified Retention_m2",
    "rsva_m4": "Value-Qualified Retention_m4",
    "rsva_m6": "Value-Qualified Retention_m6",
    "retention_m1": "Retention_m1",
    "retention_m2": "Retention_m2",
    "retention_m4": "Retention_m4",
    "retention_m6": "Retention_m6",
}

METRIC_NAME_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"\bRSVA_mh\b", "Value-Qualified Retention_h"),
    (r"\bRSVA_m1\b", "Value-Qualified Retention_m1"),
    (r"\bRSVA_m2\b", "Value-Qualified Retention_m2"),
    (r"\bRSVA_m4\b", "Value-Qualified Retention_m4"),
    (r"\bRSVA_m6\b", "Value-Qualified Retention_m6"),
    (r"\bSUR_h\b", "Post-Value Retention_h"),
    (r"\bSUR_t\b", "Post-Value Retention_t"),
    (r"\bSUR_m2\b", "Post-Value Retention_m2"),
    (r"\bSUR_m4\b", "Post-Value Retention_m4"),
    (r"\bSUR_m6\b", "Post-Value Retention_m6"),
    (r"\bSVS_t\b", "Value Conversion Rate"),
    (r"\bRSVA\b", "Value-Qualified Retention"),
    (r"\bSUR\b", "Post-Value Retention"),
    (r"\bSVS\b", "Value Conversion Rate"),
]

METRIC_DEFINITION_ROWS: List[Dict[str, str]] = [
    {
        "code": "SVS_t",
        "display_name": "Value Conversion Rate",
        "calculation": "SVS_t = strict_users / active_users",
    },
    {
        "code": "SUR_t",
        "display_name": "Post-Value Retention_t",
        "calculation": "SUR_t = strict_retained_users / strict_users",
    },
    {
        "code": "SUR_h",
        "display_name": "Post-Value Retention_h",
        "calculation": "SUR_h = strict_retained_mh_users / strict_users (for h=2,4,6)",
    },
    {
        "code": "RSVA_m1",
        "display_name": "Value-Qualified Retention_m1",
        "calculation": "RSVA_m1 = strict_retained_users / active_users",
    },
    {
        "code": "RSVA_mh",
        "display_name": "Value-Qualified Retention_h",
        "calculation": "RSVA_mh = strict_retained_mh_users / active_users",
    },
]


@dataclass(frozen=True)
class Stage4Config:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    population_primary: str = "matched_registered"
    value_taxonomy_mode: str = "downloads_only"
    exclude_incomplete_month: bool = True
    confidence_level: float = 0.95
    reliability_method: str = "signal_to_noise"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_bool(value: str) -> bool:
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "sim", "s"}:
        return True
    if s in {"0", "false", "f", "no", "n", "nao", "não"}:
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano invalido: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 04: metricas de valor e retencao.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--population-primary",
        type=str,
        default="matched_registered",
        choices=["matched_registered", "matched_all", "all_traffic"],
    )
    parser.add_argument(
        "--value-taxonomy-mode",
        type=str,
        default="downloads_only",
        choices=["downloads_only"],
        help="Definicao operacional fixa de strict value: download_aula ou download_plano_aula.",
    )
    parser.add_argument("--exclude-incomplete-month", type=parse_bool, default=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--reliability-method",
        type=str,
        default="signal_to_noise",
        choices=["signal_to_noise"],
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage4Config:
    if not (0.0 < float(args.confidence_level) < 1.0):
        raise ValueError("confidence-level deve estar em (0,1).")
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir if args.data_dir is not None else base_dir / "base_aprendizap").resolve()
    output_dir = (args.output_dir if args.output_dir is not None else base_dir / "analysis_output").resolve()
    return Stage4Config(
        base_dir=base_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        population_primary=str(args.population_primary),
        value_taxonomy_mode="downloads_only",
        exclude_incomplete_month=bool(args.exclude_incomplete_month),
        confidence_level=float(args.confidence_level),
        reliability_method=str(args.reliability_method),
    )


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def metric_display_label(metric_code: str) -> str:
    return METRIC_DISPLAY_BY_CODE.get(str(metric_code or "").strip().lower(), str(metric_code or "n/d"))


def apply_metric_terminology_text(text: str) -> str:
    out = str(text or "")
    for pattern, repl in METRIC_NAME_REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    return out


def apply_metric_terminology_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {k: apply_metric_terminology_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [apply_metric_terminology_payload(v) for v in payload]
    if isinstance(payload, str):
        return apply_metric_terminology_text(payload)
    return payload


def build_metric_definitions_html() -> str:
    lines = ["<ul>"]
    for row in METRIC_DEFINITION_ROWS:
        lines.append(
            "<li>"
            f"<b>{row['display_name']}</b> (<code>{row['code']}</code>): "
            f"<code>{row['calculation']}</code>"
            "</li>"
        )
    lines.append("</ul>")
    return "".join(lines)


def build_embedded_image_src(path_value: Any) -> str:
    path_str = str(path_value or "").strip()
    if not path_str:
        return ""
    if path_str.startswith(("data:", "cid:", "http://", "https://")):
        return path_str
    image_path = Path(path_str)
    if not image_path.exists():
        return path_str
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        suffix = image_path.suffix.lower()
        if suffix == ".png":
            mime_type = "image/png"
        elif suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif suffix == ".svg":
            mime_type = "image/svg+xml"
        else:
            mime_type = "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_metric_definitions_markdown() -> List[str]:
    lines: List[str] = []
    for row in METRIC_DEFINITION_ROWS:
        lines.append(f"- {row['display_name']} (`{row['code']}`): `{row['calculation']}`")
    return lines


def brazil_school_calendar_phase(month_ts: pd.Timestamp) -> str:
    if pd.isna(month_ts):
        return "indefinido"
    m = int(pd.Timestamp(month_ts).month)
    if m == 1:
        return "ferias_verao"
    if m in {2, 3, 4, 5, 6}:
        return "periodo_letivo_semestre_1"
    if m == 7:
        return "recesso_meio_ano"
    if m in {8, 9, 10, 11}:
        return "periodo_letivo_semestre_2"
    return "encerramento_ano"


def ensure_required_paths(cfg: Stage4Config) -> None:
    required = [
        cfg.data_dir / "dim_teachers.csv",
        cfg.data_dir / "fct_teachers_contents_interactions.csv",
        cfg.data_dir / "fct_teachers_entries.csv",
        cfg.data_dir / "stg_lessons.csv",
        cfg.data_dir / "calendario_escolar_uf_rede.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes para etapa 04: {', '.join(missing)}")


def create_views(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    conn.execute("PRAGMA threads=4")
    conn.execute(
        f"CREATE VIEW dim_teachers AS SELECT * FROM read_csv('{q(data_dir / 'dim_teachers.csv')}', delim=';', header=true, ignore_errors=true)"
    )
    conn.execute(
        f"CREATE VIEW interactions AS SELECT * FROM read_csv_auto('{q(data_dir / 'fct_teachers_contents_interactions.csv')}', header=true)"
    )
    conn.execute(
        f"CREATE VIEW entries AS SELECT * FROM read_csv_auto('{q(data_dir / 'fct_teachers_entries.csv')}', header=true)"
    )
    conn.execute(
        f"CREATE VIEW lessons AS SELECT * FROM read_csv_auto('{q(data_dir / 'stg_lessons.csv')}', header=true)"
    )
    conn.execute(
        f"CREATE VIEW school_calendar_uf_rede AS SELECT * FROM read_csv_auto('{q(data_dir / 'calendario_escolar_uf_rede.csv')}', header=true)"
    )


def create_population_view(conn: duckdb.DuckDBPyConnection, population_primary: str) -> None:
    if population_primary == "matched_registered":
        sql = """
        CREATE OR REPLACE TEMP VIEW pop_primary_interactions AS
        SELECT
            i.unique_id,
            i.data_inicio,
            lower(coalesce(i.event_type,'')) AS event_type,
            i.id_aula,
            lower(coalesce(i.user_agent_device_type,'')) AS device_type,
            lower(coalesce(i.utm_source,'')) AS utm_source,
            d.utm_origin,
            d.currentstage,
            d.data_entrada,
            upper(trim(coalesce(d.estado,''))) AS uf,
            lower(coalesce(i.user_type,'')) AS user_type
        FROM interactions i
        INNER JOIN dim_teachers d USING(unique_id)
        WHERE i.data_inicio IS NOT NULL
          AND lower(coalesce(i.user_type,'')) = 'registered'
        """
    elif population_primary == "matched_all":
        sql = """
        CREATE OR REPLACE TEMP VIEW pop_primary_interactions AS
        SELECT
            i.unique_id,
            i.data_inicio,
            lower(coalesce(i.event_type,'')) AS event_type,
            i.id_aula,
            lower(coalesce(i.user_agent_device_type,'')) AS device_type,
            lower(coalesce(i.utm_source,'')) AS utm_source,
            d.utm_origin,
            d.currentstage,
            d.data_entrada,
            upper(trim(coalesce(d.estado,''))) AS uf,
            lower(coalesce(i.user_type,'')) AS user_type
        FROM interactions i
        INNER JOIN dim_teachers d USING(unique_id)
        WHERE i.data_inicio IS NOT NULL
        """
    else:
        sql = """
        CREATE OR REPLACE TEMP VIEW pop_primary_interactions AS
        SELECT
            i.unique_id,
            i.data_inicio,
            lower(coalesce(i.event_type,'')) AS event_type,
            i.id_aula,
            lower(coalesce(i.user_agent_device_type,'')) AS device_type,
            lower(coalesce(i.utm_source,'')) AS utm_source,
            d.utm_origin,
            d.currentstage,
            d.data_entrada,
            upper(trim(coalesce(d.estado,''))) AS uf,
            lower(coalesce(i.user_type,'')) AS user_type
        FROM interactions i
        LEFT JOIN dim_teachers d USING(unique_id)
        WHERE i.data_inicio IS NOT NULL
        """
    conn.execute(sql)


def load_snapshot_ts(conn: duckdb.DuckDBPyConnection, output_dir: Path) -> pd.Timestamp:
    cpath = output_dir / "consolidated_status.json"
    if cpath.exists():
        try:
            obj = json.loads(cpath.read_text(encoding="utf-8"))
            if "snapshot_ts" in obj and obj["snapshot_ts"]:
                ts = pd.to_datetime(obj["snapshot_ts"], errors="coerce")
                if pd.notna(ts):
                    return pd.Timestamp(ts)
        except Exception:
            pass
    ts = conn.execute("SELECT max(data_inicio) FROM pop_primary_interactions").fetchone()[0]
    return pd.to_datetime(ts, errors="coerce")


def load_school_calendar_month_uf(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    try:
        cal = conn.execute(
            """
            SELECT
              date_trunc('month', CAST(month_start AS DATE)) AS month,
              upper(trim(coalesce(uf,''))) AS uf,
              lower(trim(coalesce(rede,'todas'))) AS rede,
              AVG(CAST(school_days_estimate AS DOUBLE)) AS school_days_estimate,
              AVG(CAST(business_days AS DOUBLE)) AS business_days,
              AVG(CAST(official_holiday_weekdays AS DOUBLE)) AS official_holiday_weekdays,
              MAX(coalesce(calendar_source,'')) AS calendar_source
            FROM school_calendar_uf_rede
            WHERE lower(trim(coalesce(rede,'todas'))) = 'todas'
            GROUP BY 1,2,3
            """
        ).fetchdf()
    except Exception:
        return pd.DataFrame()
    if cal.empty:
        return cal
    cal["month"] = pd.to_datetime(cal["month"], errors="coerce")
    cal["uf"] = cal["uf"].astype(str).str.upper().str.strip()
    for c in ["school_days_estimate", "business_days", "official_holiday_weekdays"]:
        cal[c] = pd.to_numeric(cal[c], errors="coerce")
    cal["school_days_ratio"] = cal["school_days_estimate"] / cal["business_days"].replace(0, np.nan)
    return cal


def build_monthly_calendar_context(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    cal = load_school_calendar_month_uf(conn)
    if cal.empty:
        return pd.DataFrame()
    try:
        um = conn.execute(
            """
            SELECT DISTINCT
              unique_id,
              date_trunc('month', data_inicio) AS month,
              upper(trim(coalesce(uf,''))) AS uf
            FROM pop_primary_interactions
            WHERE data_inicio IS NOT NULL
            """
        ).fetchdf()
    except Exception:
        return pd.DataFrame()
    if um.empty:
        return pd.DataFrame()
    um["month"] = pd.to_datetime(um["month"], errors="coerce")
    um["uf"] = um["uf"].astype(str).str.upper().str.strip()
    merged = um.merge(cal, on=["month", "uf"], how="left")
    out = (
        merged.groupby("month", as_index=False)
        .agg(
            avg_school_days_estimate_active=("school_days_estimate", "mean"),
            avg_business_days_active=("business_days", "mean"),
            avg_official_holiday_weekdays_active=("official_holiday_weekdays", "mean"),
            avg_school_days_ratio_active=("school_days_ratio", "mean"),
            school_calendar_match_rate=("school_days_estimate", lambda s: float(pd.notna(s).mean()) if len(s) else np.nan),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out["brazil_school_calendar_phase"] = out["month"].apply(brazil_school_calendar_phase)
    return out


def wilson_interval(k: float, n: float, confidence_level: float) -> Tuple[float, float]:
    if n <= 0 or pd.isna(n):
        return (float("nan"), float("nan"))
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    p = float(k) / float(n)
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / den
    half = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def build_monthly_decomposition(
    conn: duckdb.DuckDBPyConnection,
    snapshot_ts: pd.Timestamp,
    exclude_incomplete_month: bool,
    confidence_level: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sql = """
    WITH events AS (
        SELECT
            unique_id,
            date_trunc('month', data_inicio) AS month,
            event_type
        FROM pop_primary_interactions
    ),
    user_month AS (
        SELECT unique_id, month, COUNT(*)::BIGINT AS total_events
        FROM events
        GROUP BY 1,2
    ),
    flags AS (
        SELECT
            e.unique_id,
            e.month,
            MAX(CASE WHEN e.event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END) AS strict_flag,
            MAX(CASE WHEN e.event_type LIKE '%aula%' OR e.event_type LIKE '%prova%' OR e.event_type LIKE '%download%' OR e.event_type LIKE '%visualizacao%' OR e.event_type LIKE '%plano%' THEN 1 ELSE 0 END) AS broad_flag
        FROM events e
        GROUP BY 1,2
    ),
    linked AS (
        SELECT
            u.unique_id,
            u.month,
            COALESCE(f.strict_flag, 0) AS strict_flag,
            COALESCE(f.broad_flag, 0) AS broad_flag,
            CASE WHEN u2.unique_id IS NULL THEN 0 ELSE 1 END AS next_active,
            CASE WHEN u3.unique_id IS NULL THEN 0 ELSE 1 END AS next2_active,
            CASE WHEN u4.unique_id IS NULL THEN 0 ELSE 1 END AS next4_active,
            CASE WHEN u6.unique_id IS NULL THEN 0 ELSE 1 END AS next6_active
        FROM user_month u
        LEFT JOIN flags f
          ON u.unique_id = f.unique_id
         AND u.month = f.month
        LEFT JOIN user_month u2
          ON u.unique_id = u2.unique_id
         AND date_trunc('month', u.month + INTERVAL '1 month') = u2.month
        LEFT JOIN user_month u3
          ON u.unique_id = u3.unique_id
         AND date_trunc('month', u.month + INTERVAL '2 month') = u3.month
        LEFT JOIN user_month u4
          ON u.unique_id = u4.unique_id
         AND date_trunc('month', u.month + INTERVAL '4 month') = u4.month
        LEFT JOIN user_month u6
          ON u.unique_id = u6.unique_id
         AND date_trunc('month', u.month + INTERVAL '6 month') = u6.month
    )
    SELECT
        month,
        COUNT(*)::BIGINT AS active_users,
        SUM(strict_flag)::BIGINT AS strict_users,
        SUM(broad_flag)::BIGINT AS broad_users,
        SUM(next_active)::BIGINT AS retained_users,
        SUM(next2_active)::BIGINT AS retained_m2_users,
        SUM(next4_active)::BIGINT AS retained_m4_users,
        SUM(next6_active)::BIGINT AS retained_m6_users,
        SUM(CASE WHEN strict_flag=1 AND next_active=1 THEN 1 ELSE 0 END)::BIGINT AS strict_retained_users,
        SUM(CASE WHEN strict_flag=1 AND next2_active=1 THEN 1 ELSE 0 END)::BIGINT AS strict_retained_m2_users,
        SUM(CASE WHEN strict_flag=1 AND next4_active=1 THEN 1 ELSE 0 END)::BIGINT AS strict_retained_m4_users,
        SUM(CASE WHEN strict_flag=1 AND next6_active=1 THEN 1 ELSE 0 END)::BIGINT AS strict_retained_m6_users,
        SUM(CASE WHEN broad_flag=1 AND next_active=1 THEN 1 ELSE 0 END)::BIGINT AS broad_retained_users
    FROM linked
    GROUP BY 1
    ORDER BY 1
    """
    monthly = conn.execute(sql).fetchdf()
    if monthly.empty:
        return monthly, pd.DataFrame()

    monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce")
    monthly["month_num"] = monthly["month"].dt.month
    monthly["brazil_school_calendar_phase"] = monthly["month"].apply(brazil_school_calendar_phase)

    cal_ctx = build_monthly_calendar_context(conn)
    if not cal_ctx.empty:
        monthly = monthly.merge(cal_ctx, on=["month", "brazil_school_calendar_phase"], how="left")

    snapshot_month = pd.Timestamp(snapshot_ts).to_period("M").to_timestamp()
    monthly["is_complete_month"] = monthly["month"] < snapshot_month if exclude_incomplete_month else True
    monthly["is_decision_month"] = (
        (monthly["month"] + pd.offsets.MonthBegin(1)) < snapshot_month if exclude_incomplete_month else True
    )
    monthly["is_decision_month_m2"] = (
        (monthly["month"] + pd.offsets.MonthBegin(2)) < snapshot_month if exclude_incomplete_month else True
    )
    monthly["is_decision_month_m4"] = (
        (monthly["month"] + pd.offsets.MonthBegin(4)) < snapshot_month if exclude_incomplete_month else True
    )
    monthly["is_decision_month_m6"] = (
        (monthly["month"] + pd.offsets.MonthBegin(6)) < snapshot_month if exclude_incomplete_month else True
    )

    monthly["retention_m1"] = monthly["retained_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["retention_m2"] = monthly["retained_m2_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["retention_m4"] = monthly["retained_m4_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["retention_m6"] = monthly["retained_m6_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["svs_t"] = monthly["strict_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["sur_t"] = monthly["strict_retained_users"] / monthly["strict_users"].replace(0, np.nan)
    monthly["sur_m2"] = monthly["strict_retained_m2_users"] / monthly["strict_users"].replace(0, np.nan)
    monthly["sur_m4"] = monthly["strict_retained_m4_users"] / monthly["strict_users"].replace(0, np.nan)
    monthly["sur_m6"] = monthly["strict_retained_m6_users"] / monthly["strict_users"].replace(0, np.nan)
    monthly["rsva_m1"] = monthly["strict_retained_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["rsva_m2"] = monthly["strict_retained_m2_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["rsva_m4"] = monthly["strict_retained_m4_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["rsva_m6"] = monthly["strict_retained_m6_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["broad_value_user_share"] = monthly["broad_users"] / monthly["active_users"].replace(0, np.nan)
    monthly["broad_retained_share"] = monthly["broad_retained_users"] / monthly["active_users"].replace(0, np.nan)

    m1_cols = ["retention_m1", "sur_t", "rsva_m1", "broad_retained_share"]
    m2_cols = ["retention_m2", "sur_m2", "rsva_m2"]
    m4_cols = ["retention_m4", "sur_m4", "rsva_m4"]
    m6_cols = ["retention_m6", "sur_m6", "rsva_m6"]
    monthly.loc[monthly["is_decision_month"] != True, m1_cols] = np.nan
    monthly.loc[monthly["is_decision_month_m2"] != True, m2_cols] = np.nan
    monthly.loc[monthly["is_decision_month_m4"] != True, m4_cols] = np.nan
    monthly.loc[monthly["is_decision_month_m6"] != True, m6_cols] = np.nan

    ci_specs = [
        ("retention_m1", "retained_users", "active_users", "is_decision_month"),
        ("retention_m2", "retained_m2_users", "active_users", "is_decision_month_m2"),
        ("retention_m4", "retained_m4_users", "active_users", "is_decision_month_m4"),
        ("retention_m6", "retained_m6_users", "active_users", "is_decision_month_m6"),
        ("svs_t", "strict_users", "active_users", "is_decision_month"),
        ("sur_t", "strict_retained_users", "strict_users", "is_decision_month"),
        ("sur_m2", "strict_retained_m2_users", "strict_users", "is_decision_month_m2"),
        ("sur_m4", "strict_retained_m4_users", "strict_users", "is_decision_month_m4"),
        ("sur_m6", "strict_retained_m6_users", "strict_users", "is_decision_month_m6"),
        ("rsva_m1", "strict_retained_users", "active_users", "is_decision_month"),
        ("rsva_m2", "strict_retained_m2_users", "active_users", "is_decision_month_m2"),
        ("rsva_m4", "strict_retained_m4_users", "active_users", "is_decision_month_m4"),
        ("rsva_m6", "strict_retained_m6_users", "active_users", "is_decision_month_m6"),
        ("broad_value_user_share", "broad_users", "active_users", "is_decision_month"),
        ("broad_retained_share", "broad_retained_users", "active_users", "is_decision_month"),
    ]
    for metric, num_col, den_col, dec_col in ci_specs:
        lows: List[float] = []
        highs: List[float] = []
        for _, row in monthly.iterrows():
            if not bool(row.get(dec_col, True)):
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            lo, hi = wilson_interval(float(row[num_col] or 0.0), float(row[den_col] or 0.0), confidence_level)
            lows.append(lo)
            highs.append(hi)
        monthly[f"{metric}_ci_low"] = lows
        monthly[f"{metric}_ci_high"] = highs
        monthly[f"{metric}_ci_half_width"] = (
            pd.to_numeric(monthly[f"{metric}_ci_high"], errors="coerce")
            - pd.to_numeric(monthly[f"{metric}_ci_low"], errors="coerce")
        ) / 2.0

    long_rows: List[Dict[str, Any]] = []
    metric_specs = [
        ("rsva_m1", "strict_retained_users", "active_users"),
        ("rsva_m2", "strict_retained_m2_users", "active_users"),
        ("rsva_m4", "strict_retained_m4_users", "active_users"),
        ("rsva_m6", "strict_retained_m6_users", "active_users"),
        ("svs_t", "strict_users", "active_users"),
        ("sur_t", "strict_retained_users", "strict_users"),
        ("sur_m2", "strict_retained_m2_users", "strict_users"),
        ("sur_m4", "strict_retained_m4_users", "strict_users"),
        ("sur_m6", "strict_retained_m6_users", "strict_users"),
        ("retention_m1", "retained_users", "active_users"),
        ("retention_m2", "retained_m2_users", "active_users"),
        ("retention_m4", "retained_m4_users", "active_users"),
        ("retention_m6", "retained_m6_users", "active_users"),
        ("broad_value_user_share", "broad_users", "active_users"),
        ("broad_retained_share", "broad_retained_users", "active_users"),
    ]
    for _, row in monthly.iterrows():
        for m, n, d in metric_specs:
            long_rows.append(
                {
                    "month": row["month"],
                    "metric": m,
                    "value": row.get(m),
                    "numerator": row.get(n),
                    "denominator": row.get(d),
                    "ci_low": row.get(f"{m}_ci_low"),
                    "ci_high": row.get(f"{m}_ci_high"),
                    "ci_half_width": row.get(f"{m}_ci_half_width"),
                    "is_decision_month": bool(row.get("is_decision_month", True)),
                }
            )
    monthly_long = pd.DataFrame(long_rows)
    return monthly, monthly_long


def build_metric_uncertainty_panel(
    monthly_long: pd.DataFrame,
    focus_metrics: List[str] | None = None,
) -> pd.DataFrame:
    cols = {"month", "metric", "value", "numerator", "denominator", "ci_low", "ci_high", "ci_half_width", "is_decision_month"}
    if monthly_long.empty or not cols.issubset(set(monthly_long.columns)):
        return pd.DataFrame(columns=["month", "metric", "value", "numerator", "denominator", "ci_low", "ci_high", "ci_half_width", "ci_width", "ci_half_width_pct_value", "is_decision_month"])
    out = monthly_long.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    for c in ["value", "numerator", "denominator", "ci_low", "ci_high", "ci_half_width"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if focus_metrics:
        focus = set([str(x) for x in focus_metrics])
        out = out[out["metric"].astype(str).isin(focus)].copy()
    out["ci_width"] = out["ci_high"] - out["ci_low"]
    out["ci_half_width_pct_value"] = out["ci_half_width"] / out["value"].abs().replace(0, np.nan)
    out = out.sort_values(["metric", "month"]).reset_index(drop=True)
    return out


def build_strict_cohort_survival_hazard(
    conn: duckdb.DuckDBPyConnection,
    snapshot_ts: pd.Timestamp,
    exclude_incomplete_month: bool,
    confidence_level: float = 0.95,
    max_horizon_months: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    sql_base = """
    WITH um AS (
      SELECT
        unique_id,
        date_trunc('month', data_inicio) AS month,
        MAX(CASE WHEN event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END) AS strict_flag
      FROM pop_primary_interactions
      WHERE data_inicio IS NOT NULL
      GROUP BY 1,2
    ),
    first_strict AS (
      SELECT
        unique_id,
        MIN(month) AS cohort_month
      FROM um
      WHERE strict_flag = 1
      GROUP BY 1
    ),
    next_return AS (
      SELECT
        f.unique_id,
        f.cohort_month,
        MIN(u.month) AS first_return_month
      FROM first_strict f
      LEFT JOIN um u
        ON u.unique_id = f.unique_id
       AND u.month > f.cohort_month
      GROUP BY 1,2
    )
    SELECT
      unique_id,
      cohort_month,
      first_return_month,
      CASE
        WHEN first_return_month IS NULL THEN NULL
        ELSE date_diff('month', cohort_month, first_return_month)
      END AS first_return_h
    FROM next_return
    """
    base = conn.execute(sql_base).fetchdf()
    if base.empty:
        return (
            pd.DataFrame(columns=["horizon_m", "n_eligible", "n_at_risk", "n_events_first_return", "hazard", "hazard_ci_low", "hazard_ci_high", "survival", "cumulative_return", "cumulative_return_ci_low", "cumulative_return_ci_high", "n_cohorts_eligible"]),
            pd.DataFrame(columns=["cohort_month", "horizon_m", "cohort_size", "n_at_risk", "n_events_first_return", "hazard", "survival", "cumulative_return"]),
            {
                "available": False,
                "reason": "empty_strict_cohort",
            },
        )

    base["cohort_month"] = pd.to_datetime(base["cohort_month"], errors="coerce")
    base["first_return_month"] = pd.to_datetime(base["first_return_month"], errors="coerce")
    base["first_return_h"] = pd.to_numeric(base["first_return_h"], errors="coerce")
    base = base.dropna(subset=["cohort_month"]).reset_index(drop=True)
    if base.empty:
        return (
            pd.DataFrame(columns=["horizon_m", "n_eligible", "n_at_risk", "n_events_first_return", "hazard", "hazard_ci_low", "hazard_ci_high", "survival", "cumulative_return", "cumulative_return_ci_low", "cumulative_return_ci_high", "n_cohorts_eligible"]),
            pd.DataFrame(columns=["cohort_month", "horizon_m", "cohort_size", "n_at_risk", "n_events_first_return", "hazard", "survival", "cumulative_return"]),
            {
                "available": False,
                "reason": "invalid_strict_cohort",
            },
        )

    max_month_sql = """
    SELECT
      MAX(date_trunc('month', data_inicio)) AS max_month
    FROM pop_primary_interactions
    WHERE data_inicio IS NOT NULL
    """
    max_month_val = conn.execute(max_month_sql).fetchdf().iloc[0]["max_month"]
    max_observed_month = pd.to_datetime(max_month_val, errors="coerce")
    snapshot_month = pd.Timestamp(snapshot_ts).to_period("M").to_timestamp()
    if exclude_incomplete_month:
        max_follow_month = snapshot_month - pd.offsets.MonthBegin(1)
        if pd.notna(max_observed_month):
            max_follow_month = min(max_follow_month, max_observed_month)
    else:
        max_follow_month = max_observed_month
    max_h = max(int(max_horizon_months), 1)

    pooled_rows: List[Dict[str, Any]] = []
    cohort_rows: List[Dict[str, Any]] = []
    survival_running = 1.0

    cohort_sizes = (
        base.groupby("cohort_month", as_index=False)
        .agg(cohort_size=("unique_id", "nunique"))
        .sort_values("cohort_month")
        .reset_index(drop=True)
    )

    for h in range(1, max_h + 1):
        threshold_month = max_follow_month - pd.offsets.MonthBegin(h)
        eligible_mask = base["cohort_month"] <= threshold_month
        n_eligible = int(eligible_mask.sum())
        n_cohorts_eligible = int(base.loc[eligible_mask, "cohort_month"].nunique())
        at_risk_mask = eligible_mask & (base["first_return_h"].isna() | (base["first_return_h"] >= h))
        event_mask = eligible_mask & (base["first_return_h"] == h)
        returned_by_h_mask = eligible_mask & (base["first_return_h"].notna()) & (base["first_return_h"] <= h)

        n_at_risk = int(at_risk_mask.sum())
        n_events = int(event_mask.sum())
        n_returned_by_h = int(returned_by_h_mask.sum())

        if n_at_risk > 0:
            hazard = float(n_events / n_at_risk)
            hz_lo, hz_hi = wilson_interval(float(n_events), float(n_at_risk), confidence_level)
            survival_running = survival_running * (1.0 - hazard)
        else:
            hazard = np.nan
            hz_lo, hz_hi = (np.nan, np.nan)

        if n_eligible > 0:
            cumulative_return = float(n_returned_by_h / n_eligible)
            cum_lo, cum_hi = wilson_interval(float(n_returned_by_h), float(n_eligible), confidence_level)
            survival_est = float(1.0 - cumulative_return)
        else:
            cumulative_return = np.nan
            cum_lo, cum_hi = (np.nan, np.nan)
            survival_est = np.nan

        pooled_rows.append(
            {
                "horizon_m": h,
                "n_eligible": n_eligible,
                "n_at_risk": n_at_risk,
                "n_events_first_return": n_events,
                "hazard": hazard,
                "hazard_ci_low": hz_lo,
                "hazard_ci_high": hz_hi,
                "survival": survival_est if pd.notna(survival_est) else survival_running,
                "cumulative_return": cumulative_return,
                "cumulative_return_ci_low": cum_lo,
                "cumulative_return_ci_high": cum_hi,
                "n_cohorts_eligible": n_cohorts_eligible,
            }
        )

    for _, c in cohort_sizes.iterrows():
        cohort_month = pd.to_datetime(c["cohort_month"], errors="coerce")
        cohort_size = int(pd.to_numeric(c["cohort_size"], errors="coerce"))
        if cohort_size <= 0 or pd.isna(cohort_month):
            continue
        cohort_df = base[base["cohort_month"] == cohort_month].copy()
        if pd.notna(max_follow_month):
            max_follow_h = int((int(max_follow_month.year) - int(cohort_month.year)) * 12 + (int(max_follow_month.month) - int(cohort_month.month)))
        else:
            max_follow_h = 0
        max_eval_h = max(0, min(max_h, max_follow_h))
        for h in range(1, max_eval_h + 1):
            at_risk_mask = cohort_df["first_return_h"].isna() | (cohort_df["first_return_h"] >= h)
            event_mask = cohort_df["first_return_h"] == h
            n_at_risk = int(at_risk_mask.sum())
            n_events = int(event_mask.sum())
            returned_by_h = int(((cohort_df["first_return_h"].notna()) & (cohort_df["first_return_h"] <= h)).sum())
            hazard = float(n_events / n_at_risk) if n_at_risk > 0 else np.nan
            cumulative_return = float(returned_by_h / cohort_size) if cohort_size > 0 else np.nan
            survival = float(1.0 - cumulative_return) if pd.notna(cumulative_return) else np.nan
            cohort_rows.append(
                {
                    "cohort_month": cohort_month,
                    "horizon_m": h,
                    "cohort_size": cohort_size,
                    "n_at_risk": n_at_risk,
                    "n_events_first_return": n_events,
                    "hazard": hazard,
                    "survival": survival,
                    "cumulative_return": cumulative_return,
                }
            )

    pooled_df = pd.DataFrame(pooled_rows).sort_values("horizon_m").reset_index(drop=True)
    cohort_df = pd.DataFrame(cohort_rows).sort_values(["cohort_month", "horizon_m"]).reset_index(drop=True)
    summary = {
        "available": True,
        "cohort_definition": "cohort_month = primeiro mês com strict_value (download_aula/download_plano_aula) por usuário; evento = primeiro mês de retorno ativo após cohort_month.",
        "max_follow_month": str(max_follow_month.date()) if pd.notna(max_follow_month) else None,
        "max_horizon_evaluated": int(max_h),
        "users_in_first_strict_cohort": int(len(base)),
        "cohort_month_start": str(base["cohort_month"].min().date()) if not base.empty else None,
        "cohort_month_end": str(base["cohort_month"].max().date()) if not base.empty else None,
        "cohort_months": int(base["cohort_month"].nunique()),
    }
    return pooled_df, cohort_df, summary


def _direction_from_delta_and_ci(delta: float, ci_curr: float, ci_prev: float) -> str:
    if pd.isna(delta):
        return "sem_dado"
    if pd.notna(ci_curr) and pd.notna(ci_prev):
        noise = abs(float(ci_curr)) + abs(float(ci_prev))
        if abs(float(delta)) <= noise:
            return "estavel"
    if float(delta) > 0:
        return "sobe"
    if float(delta) < 0:
        return "cai"
    return "estavel"


def build_rsva_drop_diagnostics(decomposition_df: pd.DataFrame) -> Dict[str, Any]:
    needed = {
        "month",
        "is_decision_month",
        "rsva_m1",
        "svs_t",
        "sur_t",
        "rsva_m1_ci_half_width",
        "svs_t_ci_half_width",
        "sur_t_ci_half_width",
    }
    if decomposition_df.empty or not needed.issubset(set(decomposition_df.columns)):
        return {"available": False, "reason": "missing_required_columns"}

    x = decomposition_df[decomposition_df["is_decision_month"] == True].copy().sort_values("month")
    if len(x) < 2:
        return {"available": False, "reason": "insufficient_rows"}

    x["d_rsva"] = x["rsva_m1"].diff()
    x["d_svs"] = x["svs_t"].diff()
    x["d_sur"] = x["sur_t"].diff()
    x["dir_rsva"] = [
        _direction_from_delta_and_ci(d, c, p)
        for d, c, p in zip(x["d_rsva"], x["rsva_m1_ci_half_width"], x["rsva_m1_ci_half_width"].shift(1))
    ]
    x["dir_svs"] = [
        _direction_from_delta_and_ci(d, c, p)
        for d, c, p in zip(x["d_svs"], x["svs_t_ci_half_width"], x["svs_t_ci_half_width"].shift(1))
    ]
    x["dir_sur"] = [
        _direction_from_delta_and_ci(d, c, p)
        for d, c, p in zip(x["d_sur"], x["sur_t_ci_half_width"], x["sur_t_ci_half_width"].shift(1))
    ]

    def classify(row: pd.Series) -> str:
        if row["dir_rsva"] != "cai":
            return "sem_queda_de_rsva"
        if row["dir_svs"] == "cai" and row["dir_sur"] == "estavel":
            return "queda_por_entrada_em_valor"
        if row["dir_svs"] == "estavel" and row["dir_sur"] == "cai":
            return "queda_por_continuidade_pos_valor"
        if row["dir_svs"] == "cai" and row["dir_sur"] == "cai":
            return "queda_mista_adocao_e_retencao"
        return "queda_combinacao_outros_sinais"

    x["diagnostico"] = x.apply(classify, axis=1)

    drops = x[x["dir_rsva"] == "cai"].copy()
    summary = (
        drops.groupby("diagnostico", dropna=False)["month"]
        .count()
        .reset_index(name="meses")
        .sort_values("meses", ascending=False)
    )
    if not summary.empty:
        summary["participacao"] = summary["meses"] / summary["meses"].sum()

    table = x[
        [
            "month",
            "rsva_m1",
            "svs_t",
            "sur_t",
            "d_rsva",
            "d_svs",
            "d_sur",
            "dir_rsva",
            "dir_svs",
            "dir_sur",
            "diagnostico",
        ]
    ].copy()
    return {
        "available": True,
        "drop_summary": summary.to_dict(orient="records"),
        "diagnostics_table": table.to_dict(orient="records"),
        "latest_12_diagnostics": table.sort_values("month").tail(12).to_dict(orient="records"),
        "dominant_drop_pattern": str(summary.iloc[0]["diagnostico"]) if not summary.empty else "sem_queda_observada",
    }


def build_event_family_segmentation(
    conn: duckdb.DuckDBPyConnection,
    snapshot_ts: pd.Timestamp,
    exclude_incomplete_month: bool,
) -> pd.DataFrame:
    sql = """
    WITH um AS (
      SELECT
        unique_id,
        date_trunc('month', data_inicio) AS month,
        MAX(CASE WHEN event_type LIKE '%aula%' THEN 1 ELSE 0 END) AS aula_flag,
        MAX(CASE WHEN event_type LIKE '%prova%' THEN 1 ELSE 0 END) AS prova_flag,
        MAX(CASE WHEN event_type LIKE '%plano%' THEN 1 ELSE 0 END) AS plano_flag,
        MAX(CASE WHEN event_type = 'download_aula' THEN 1 ELSE 0 END) AS download_aula_flag,
        MAX(CASE WHEN event_type = 'download_plano_aula' THEN 1 ELSE 0 END) AS download_plano_flag,
        MAX(CASE WHEN event_type = 'visualizacao_aula' THEN 1 ELSE 0 END) AS visualizacao_aula_flag,
        MAX(CASE WHEN event_type IN ('visualizacao_prova', 'visualizacao_prova_aprendizap') THEN 1 ELSE 0 END) AS visualizacao_prova_flag,
        MAX(CASE WHEN event_type IN ('prova_salva', 'prova_criada_edicao', 'download_avaliacao', 'envio_email_ou_baixou_prova') THEN 1 ELSE 0 END) AS prova_acao_flag
      FROM pop_primary_interactions
      GROUP BY 1,2
    )
    SELECT
      month,
      COUNT(*)::BIGINT AS active_users,
      SUM(aula_flag)::BIGINT AS aula_users,
      SUM(prova_flag)::BIGINT AS prova_users,
      SUM(plano_flag)::BIGINT AS plano_users,
      SUM(download_aula_flag)::BIGINT AS download_aula_users,
      SUM(download_plano_flag)::BIGINT AS download_plano_users,
      SUM(visualizacao_aula_flag)::BIGINT AS visualizacao_aula_users,
      SUM(CASE WHEN visualizacao_aula_flag=1 AND download_aula_flag=0 THEN 1 ELSE 0 END)::BIGINT AS visualizacao_aula_sem_download_users,
      SUM(visualizacao_prova_flag)::BIGINT AS visualizacao_prova_users,
      SUM(CASE WHEN visualizacao_prova_flag=1 AND prova_acao_flag=0 THEN 1 ELSE 0 END)::BIGINT AS visualizacao_prova_sem_acao_users
    FROM um
    GROUP BY 1
    ORDER BY 1
    """
    out = conn.execute(sql).fetchdf()
    if out.empty:
        return out
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out["month_num"] = out["month"].dt.month
    out["brazil_school_calendar_phase"] = out["month"].apply(brazil_school_calendar_phase)
    snapshot_month = pd.Timestamp(snapshot_ts).to_period("M").to_timestamp()
    out["is_complete_month"] = out["month"] < snapshot_month if exclude_incomplete_month else True
    out["is_decision_month"] = (
        (out["month"] + pd.offsets.MonthBegin(1)) < snapshot_month if exclude_incomplete_month else True
    )
    out["month_num"] = out["month"].dt.month
    out["brazil_school_calendar_phase"] = out["month"].apply(brazil_school_calendar_phase)

    out["aula_user_share"] = out["aula_users"] / out["active_users"].replace(0, np.nan)
    out["prova_user_share"] = out["prova_users"] / out["active_users"].replace(0, np.nan)
    out["plano_user_share"] = out["plano_users"] / out["active_users"].replace(0, np.nan)
    out["download_aula_user_share"] = out["download_aula_users"] / out["active_users"].replace(0, np.nan)
    out["download_plano_user_share"] = out["download_plano_users"] / out["active_users"].replace(0, np.nan)
    out["vis_aula_sem_download_share_active"] = out["visualizacao_aula_sem_download_users"] / out["active_users"].replace(0, np.nan)
    out["vis_aula_sem_download_share_viewers"] = out["visualizacao_aula_sem_download_users"] / out["visualizacao_aula_users"].replace(0, np.nan)
    out["vis_prova_sem_acao_share_viewers"] = out["visualizacao_prova_sem_acao_users"] / out["visualizacao_prova_users"].replace(0, np.nan)
    out["vis_aula_com_download_share_viewers"] = 1.0 - pd.to_numeric(out["vis_aula_sem_download_share_viewers"], errors="coerce")
    out["vis_prova_com_acao_share_viewers"] = 1.0 - pd.to_numeric(out["vis_prova_sem_acao_share_viewers"], errors="coerce")
    for c in ["vis_aula_com_download_share_viewers", "vis_prova_com_acao_share_viewers"]:
        out.loc[(pd.to_numeric(out[c], errors="coerce") < 0) | (pd.to_numeric(out[c], errors="coerce") > 1), c] = np.nan
    return out.sort_values("month").reset_index(drop=True)


def build_subject_download_analysis(
    conn: duckdb.DuckDBPyConnection,
    snapshot_ts: pd.Timestamp,
    exclude_incomplete_month: bool,
    top_n: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty_quality = pd.DataFrame(
        columns=[
            "month",
            "download_events",
            "events_with_id_aula",
            "events_with_disciplina",
            "is_complete_month",
            "is_decision_month",
            "pct_with_id_aula",
            "pct_with_disciplina",
        ]
    )
    empty_top = pd.DataFrame(columns=["disciplina", "download_events", "share_mapped_downloads"])
    empty_top_monthly = pd.DataFrame(
        columns=["month", "disciplina", "download_events", "mapped_download_events_month", "share_of_mapped_downloads"]
    )

    sql = """
    WITH downloads AS (
      SELECT
        unique_id,
        date_trunc('month', data_inicio) AS month,
        event_type,
        id_aula
      FROM pop_primary_interactions
      WHERE event_type IN ('download_aula', 'download_plano_aula')
    ),
    mapped AS (
      SELECT
        d.unique_id,
        d.month,
        d.event_type,
        d.id_aula,
        CASE
          WHEN l.disciplina IS NULL OR trim(CAST(l.disciplina AS VARCHAR)) = '' THEN '(sem_disciplina)'
          ELSE CAST(l.disciplina AS VARCHAR)
        END AS disciplina
      FROM downloads d
      LEFT JOIN lessons l
        ON d.id_aula = l.id_aula
    )
    SELECT
      month,
      disciplina,
      COUNT(*)::BIGINT AS download_events,
      SUM(CASE WHEN id_aula IS NOT NULL AND trim(CAST(id_aula AS VARCHAR)) <> '' THEN 1 ELSE 0 END)::BIGINT AS events_with_id_aula
    FROM mapped
    GROUP BY 1,2
    ORDER BY 1,2
    """
    long_df = conn.execute(sql).fetchdf()
    if long_df.empty:
        return empty_quality, empty_top, empty_top_monthly

    long_df["month"] = pd.to_datetime(long_df["month"], errors="coerce")
    long_df["download_events"] = pd.to_numeric(long_df["download_events"], errors="coerce").fillna(0.0)
    long_df["events_with_id_aula"] = pd.to_numeric(long_df["events_with_id_aula"], errors="coerce").fillna(0.0)

    quality = (
        long_df.groupby("month", as_index=False)
        .agg(
            download_events=("download_events", "sum"),
            events_with_id_aula=("events_with_id_aula", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    mapped_disc = (
        long_df[long_df["disciplina"] != "(sem_disciplina)"]
        .groupby("month", as_index=False)["download_events"]
        .sum()
        .rename(columns={"download_events": "events_with_disciplina"})
    )
    quality = quality.merge(mapped_disc, on="month", how="left")
    quality["events_with_disciplina"] = pd.to_numeric(quality["events_with_disciplina"], errors="coerce").fillna(0.0)

    snapshot_month = pd.Timestamp(snapshot_ts).to_period("M").to_timestamp()
    quality["is_complete_month"] = quality["month"] < snapshot_month if exclude_incomplete_month else True
    quality["is_decision_month"] = (
        (quality["month"] + pd.offsets.MonthBegin(1)) < snapshot_month if exclude_incomplete_month else True
    )
    quality["pct_with_id_aula"] = quality["events_with_id_aula"] / quality["download_events"].replace(0, np.nan)
    quality["pct_with_disciplina"] = quality["events_with_disciplina"] / quality["download_events"].replace(0, np.nan)

    long_df = long_df.merge(quality[["month", "is_decision_month"]], on="month", how="left")
    dec = long_df[long_df["is_decision_month"] == True].copy()
    if dec.empty:
        dec = long_df.copy()

    mapped_dec = dec[dec["disciplina"] != "(sem_disciplina)"].copy()
    top_overall = (
        mapped_dec.groupby("disciplina", as_index=False)["download_events"]
        .sum()
        .sort_values("download_events", ascending=False)
        .head(max(int(top_n), 1))
        .reset_index(drop=True)
    )
    total_mapped_dec = float(mapped_dec["download_events"].sum())
    top_overall["share_mapped_downloads"] = top_overall["download_events"] / total_mapped_dec if total_mapped_dec > 0 else np.nan

    if top_overall.empty:
        return quality, top_overall, empty_top_monthly

    top_names = top_overall["disciplina"].astype(str).tolist()
    top_monthly = mapped_dec[mapped_dec["disciplina"].isin(top_names)].copy()
    top_monthly = (
        top_monthly.groupby(["month", "disciplina"], as_index=False)["download_events"]
        .sum()
        .sort_values(["month", "download_events"], ascending=[True, False])
        .reset_index(drop=True)
    )
    mapped_month_totals = (
        mapped_dec.groupby("month", as_index=False)["download_events"]
        .sum()
        .rename(columns={"download_events": "mapped_download_events_month"})
    )
    top_monthly = top_monthly.merge(mapped_month_totals, on="month", how="left")
    top_monthly["share_of_mapped_downloads"] = (
        pd.to_numeric(top_monthly["download_events"], errors="coerce")
        / pd.to_numeric(top_monthly["mapped_download_events_month"], errors="coerce").replace(0, np.nan)
    )
    return quality, top_overall, top_monthly


def build_user_month_segment_base(conn: duckdb.DuckDBPyConnection, teacher_df: pd.DataFrame) -> pd.DataFrame:
    heavy_map = pd.DataFrame(columns=["unique_id", "heavy_user_flag"])
    if teacher_df is not None and not teacher_df.empty and "unique_id" in teacher_df.columns and "heavy_user_flag" in teacher_df.columns:
        heavy_map = teacher_df[["unique_id", "heavy_user_flag"]].copy()
        heavy_map["unique_id"] = heavy_map["unique_id"].astype(str).str.strip()
        heavy_map["heavy_user_flag"] = pd.to_numeric(heavy_map["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
        heavy_map = heavy_map.drop_duplicates(subset=["unique_id"], keep="last")
    conn.register("heavy_map_df", heavy_map)

    sql = """
    WITH inter AS (
      SELECT
        unique_id,
        date_trunc('month', data_inicio) AS month,
        upper(trim(coalesce(uf,''))) AS uf,
        event_type
      FROM pop_primary_interactions
      WHERE data_inicio IS NOT NULL
    ),
    um AS (
      SELECT
        unique_id,
        month,
        MAX(upper(trim(coalesce(uf,'')))) AS uf,
        COUNT(*)::BIGINT AS month_events,
        MAX(CASE WHEN event_type IN ('download_aula', 'download_plano_aula') THEN 1 ELSE 0 END) AS strict_flag
      FROM inter
      GROUP BY 1,2
    ),
    teacher_heavy AS (
      SELECT
        CAST(unique_id AS VARCHAR) AS unique_id,
        COALESCE(CAST(heavy_user_flag AS INTEGER), 0) AS heavy_user_flag
      FROM heavy_map_df
    ),
    heavy_base AS (
      SELECT
        i.unique_id,
        COUNT(*)::DOUBLE AS interaction_count_global
      FROM interactions i
      INNER JOIN dim_teachers d USING(unique_id)
      GROUP BY 1
    ),
    segmented AS (
      SELECT
        t.unique_id,
        t.month,
        t.uf,
        t.strict_flag,
        h.interaction_count_global,
        th.heavy_user_flag,
        CASE
          WHEN COALESCE(th.heavy_user_flag, 0) = 1
          THEN 'heavy_users'
          ELSE 'base_regular'
        END AS segment
      FROM um t
      LEFT JOIN heavy_base h
        ON t.unique_id = h.unique_id
      LEFT JOIN teacher_heavy th
        ON CAST(t.unique_id AS VARCHAR) = th.unique_id
    ),
    linked AS (
      SELECT
        u.unique_id,
        u.month,
        u.uf,
        u.strict_flag,
        u.interaction_count_global,
        u.heavy_user_flag,
        u.segment,
        CASE WHEN u2.unique_id IS NULL THEN 0 ELSE 1 END AS next_active
      FROM segmented u
      LEFT JOIN um u2
        ON u.unique_id = u2.unique_id
       AND date_trunc('month', u.month + INTERVAL '1 month') = u2.month
    )
    SELECT
      l.unique_id,
      l.month,
      l.uf,
      l.strict_flag,
      l.next_active,
      COALESCE(l.segment, 'base_regular') AS segment
    FROM linked l
    """
    df = conn.execute(sql).fetchdf()
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if "uf" in df.columns:
        df["uf"] = df["uf"].astype(str).str.upper().str.strip()
    df["strict_flag"] = pd.to_numeric(df["strict_flag"], errors="coerce").fillna(0).astype(int)
    df["next_active"] = pd.to_numeric(df["next_active"], errors="coerce").fillna(0).astype(int)
    df["rsva_outcome"] = (df["strict_flag"] * df["next_active"]).astype(int)
    df["segment"] = df["segment"].astype(str)
    return df


def build_segment_monthly_rsva_metrics(
    user_month_base: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    exclude_incomplete_month: bool,
) -> pd.DataFrame:
    if user_month_base.empty:
        return pd.DataFrame()

    base = user_month_base.copy()
    grouped = (
        base.groupby(["month", "segment"], as_index=False)
        .agg(
            active_users=("unique_id", "count"),
            strict_users=("strict_flag", "sum"),
            strict_retained_users=("rsva_outcome", "sum"),
        )
        .sort_values(["month", "segment"])
        .reset_index(drop=True)
    )
    all_users = (
        base.groupby("month", as_index=False)
        .agg(
            active_users=("unique_id", "count"),
            strict_users=("strict_flag", "sum"),
            strict_retained_users=("rsva_outcome", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    all_users["segment"] = "all_users"
    out = pd.concat([grouped, all_users], ignore_index=True, sort=False)

    out["svs_t"] = out["strict_users"] / out["active_users"].replace(0, np.nan)
    out["sur_t"] = out["strict_retained_users"] / out["strict_users"].replace(0, np.nan)
    out["rsva_m1"] = out["strict_retained_users"] / out["active_users"].replace(0, np.nan)

    snapshot_month = pd.Timestamp(snapshot_ts).to_period("M").to_timestamp()
    out["is_complete_month"] = out["month"] < snapshot_month if exclude_incomplete_month else True
    out["is_decision_month"] = (
        (out["month"] + pd.offsets.MonthBegin(1)) < snapshot_month if exclude_incomplete_month else True
    )

    ci_specs = [
        ("svs_t", "strict_users", "active_users"),
        ("sur_t", "strict_retained_users", "strict_users"),
        ("rsva_m1", "strict_retained_users", "active_users"),
    ]
    for metric, num_col, den_col in ci_specs:
        lows: List[float] = []
        highs: List[float] = []
        for _, row in out.iterrows():
            if row.get("is_decision_month") != True:
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            lo, hi = wilson_interval(float(row.get(num_col, 0.0) or 0.0), float(row.get(den_col, 0.0) or 0.0), 0.95)
            lows.append(lo)
            highs.append(hi)
        out[f"{metric}_ci_low"] = lows
        out[f"{metric}_ci_high"] = highs
        out[f"{metric}_ci_half_width"] = (
            pd.to_numeric(out[f"{metric}_ci_high"], errors="coerce")
            - pd.to_numeric(out[f"{metric}_ci_low"], errors="coerce")
        ) / 2.0
    return out.sort_values(["month", "segment"]).reset_index(drop=True)


def build_segment_drop_diagnostics(segment_monthly: pd.DataFrame) -> Dict[str, Any]:
    needed = {"month", "segment", "is_decision_month", "rsva_m1", "svs_t", "sur_t", "rsva_m1_ci_half_width", "svs_t_ci_half_width", "sur_t_ci_half_width"}
    if segment_monthly.empty or not needed.issubset(set(segment_monthly.columns)):
        return {"available": False, "reason": "missing_required_columns"}

    by_segment: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    latest_rows: List[Dict[str, Any]] = []
    ordered_segments = ["all_users", "heavy_users", "base_regular"]
    segments = ordered_segments + [s for s in sorted(segment_monthly["segment"].dropna().astype(str).unique().tolist()) if s not in ordered_segments]

    for seg in segments:
        seg_df = segment_monthly[segment_monthly["segment"] == seg].copy()
        if seg_df.empty:
            continue
        diag = build_rsva_drop_diagnostics(seg_df)
        if not diag.get("available"):
            continue
        by_segment[seg] = diag

        ds = pd.DataFrame(diag.get("drop_summary", []))
        if not ds.empty:
            ds = ds.copy()
            ds["segment"] = seg
            summary_rows.extend(ds[["segment", "diagnostico", "meses", "participacao"]].to_dict(orient="records"))

        lt = pd.DataFrame(diag.get("latest_12_diagnostics", []))
        if not lt.empty:
            lt = lt.copy()
            lt["segment"] = seg
            latest_rows.extend(
                lt[
                    [
                        "segment",
                        "month",
                        "rsva_m1",
                        "svs_t",
                        "sur_t",
                        "d_rsva",
                        "d_svs",
                        "d_sur",
                        "dir_rsva",
                        "dir_svs",
                        "dir_sur",
                        "diagnostico",
                    ]
                ].to_dict(orient="records")
            )

    return {
        "available": len(by_segment) > 0,
        "by_segment": by_segment,
        "drop_summary_by_segment": summary_rows,
        "latest_12_diagnostics_by_segment": latest_rows,
    }


def fit_rsva_linear_models(
    segment_monthly: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = {"month", "segment", "is_decision_month", "svs_t", "sur_t", "rsva_m1"}
    if segment_monthly.empty or not cols.issubset(set(segment_monthly.columns)):
        return pd.DataFrame(), pd.DataFrame()

    model_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []

    work = segment_monthly.copy()
    work = work[work["is_decision_month"] == True].copy()
    work = work.sort_values(["segment", "month"]).reset_index(drop=True)
    segments = sorted(work["segment"].dropna().astype(str).unique().tolist())
    for seg in segments:
        s = work[work["segment"] == seg].copy()
        s = s.dropna(subset=["svs_t", "sur_t", "rsva_m1"])
        if len(s) < 4:
            continue

        X_add = s[["svs_t", "sur_t"]].astype(float).to_numpy()
        y = s["rsva_m1"].astype(float).to_numpy()
        m_add = LinearRegression()
        m_add.fit(X_add, y)
        pred_add = m_add.predict(X_add)
        r2_add = float(m_add.score(X_add, y))

        X_int = s[["svs_t", "sur_t"]].astype(float).copy()
        X_int["svs_x_sur"] = X_int["svs_t"] * X_int["sur_t"]
        m_int = LinearRegression()
        m_int.fit(X_int.to_numpy(), y)
        pred_int = m_int.predict(X_int.to_numpy())
        r2_int = float(m_int.score(X_int.to_numpy(), y))
        n_months = int(len(s))
        min_additive = 24
        min_interaction = 36
        sample_ok_additive = bool(n_months >= min_additive)
        sample_ok_interaction = bool(n_months >= min_interaction)

        model_rows.append(
            {
                "segment": seg,
                "n_months": n_months,
                "additive_r2": r2_add,
                "additive_intercept": float(m_add.intercept_),
                "additive_coef_svs_t": float(m_add.coef_[0]),
                "additive_coef_sur_t": float(m_add.coef_[1]),
                "interaction_r2": r2_int,
                "interaction_intercept": float(m_int.intercept_),
                "interaction_coef_svs_t": float(m_int.coef_[0]),
                "interaction_coef_sur_t": float(m_int.coef_[1]),
                "interaction_coef_svs_x_sur": float(m_int.coef_[2]),
                "mean_abs_residual_additive": float(np.mean(np.abs(y - pred_add))),
                "min_months_additive_recommended": min_additive,
                "min_months_interaction_recommended": min_interaction,
                "sample_ok_additive": sample_ok_additive,
                "sample_ok_interaction": sample_ok_interaction,
            }
        )

        fit_rows.extend(
            [
                {
                    "month": row["month"],
                    "segment": seg,
                    "rsva_m1_observed": float(obs),
                    "rsva_m1_pred_additive": float(pa),
                    "rsva_m1_pred_interaction": float(pi),
                }
                for row, obs, pa, pi in zip(s.to_dict(orient="records"), y, pred_add, pred_int)
            ]
        )

    return pd.DataFrame(model_rows), pd.DataFrame(fit_rows)


def _diff_proportion_ci(
    exposed_target: pd.Series,
    unexposed_target: pd.Series,
    confidence_level: float,
    min_exposed: int = 30,
    min_unexposed: int = 30,
) -> Dict[str, Any]:
    exp = pd.to_numeric(exposed_target, errors="coerce").dropna().astype(float)
    unexp = pd.to_numeric(unexposed_target, errors="coerce").dropna().astype(float)
    n1 = int(len(exp))
    n0 = int(len(unexp))
    if n1 <= 0 or n0 <= 0:
        return {
            "n_exposed": n1,
            "n_unexposed": n0,
            "p_exposed": np.nan,
            "p_unexposed": np.nan,
            "effect": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "sufficient_sample": False,
            "n_strata_used": np.nan,
        }
    k1 = float(exp.sum())
    k0 = float(unexp.sum())
    p1 = k1 / n1
    p0 = k0 / n0
    effect = p1 - p0
    # Newcombe-Wilson interval for difference in proportions is more stable than Wald at small n.
    p1_low, p1_high = wilson_interval(k1, float(n1), confidence_level)
    p0_low, p0_high = wilson_interval(k0, float(n0), confidence_level)
    ci_low = p1_low - p0_high
    ci_high = p1_high - p0_low

    se = float(np.sqrt(max((p1 * (1.0 - p1)) / n1 + (p0 * (1.0 - p0)) / n0, 0.0)))
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    z_stat = effect / se if se > 0 else np.nan
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat)))) if pd.notna(z_stat) else np.nan
    sufficient_sample = bool(n1 >= int(min_exposed) and n0 >= int(min_unexposed))
    return {
        "n_exposed": n1,
        "n_unexposed": n0,
        "p_exposed": p1,
        "p_unexposed": p0,
        "effect": effect,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "sufficient_sample": sufficient_sample,
        "n_strata_used": np.nan,
    }


def _diff_proportion_ci_month_adjusted(
    df: pd.DataFrame,
    outcome_col: str,
    confidence_level: float,
    strata_cols: List[str] | None = None,
    min_exposed_total: int = 30,
    min_unexposed_total: int = 30,
    min_strata: int = 3,
) -> Dict[str, Any]:
    strata = [c for c in (strata_cols or ["month"]) if c in df.columns]
    if not strata:
        strata = ["month"]
    cols_needed = {"exposed", outcome_col, *strata}
    if df.empty or not cols_needed.issubset(set(df.columns)):
        return {
            "n_exposed": 0,
            "n_unexposed": 0,
            "p_exposed": np.nan,
            "p_unexposed": np.nan,
            "effect": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "sufficient_sample": False,
            "n_strata_used": 0,
        }

    keep_cols = list(dict.fromkeys(strata + ["exposed", outcome_col]))
    w = df[keep_cols].copy()
    if "month" in w.columns:
        w["month"] = pd.to_datetime(w["month"], errors="coerce")
    w["exposed"] = pd.to_numeric(w["exposed"], errors="coerce").fillna(0).astype(int)
    w[outcome_col] = pd.to_numeric(w[outcome_col], errors="coerce")
    drop_cols = [c for c in ["month", outcome_col] if c in w.columns]
    w = w.dropna(subset=drop_cols)
    if w.empty:
        return {
            "n_exposed": 0,
            "n_unexposed": 0,
            "p_exposed": np.nan,
            "p_unexposed": np.nan,
            "effect": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "sufficient_sample": False,
            "n_strata_used": 0,
        }

    n_exposed_total = int((w["exposed"] == 1).sum())
    n_unexposed_total = int((w["exposed"] == 0).sum())
    p_exposed = float(w.loc[w["exposed"] == 1, outcome_col].mean()) if n_exposed_total > 0 else np.nan
    p_unexposed = float(w.loc[w["exposed"] == 0, outcome_col].mean()) if n_unexposed_total > 0 else np.nan

    strata_rows: List[Tuple[float, float, float]] = []
    for _, g in w.groupby(strata):
        g1 = g[g["exposed"] == 1][outcome_col].dropna()
        g0 = g[g["exposed"] == 0][outcome_col].dropna()
        n1 = int(len(g1))
        n0 = int(len(g0))
        if n1 <= 0 or n0 <= 0:
            continue
        p1 = float(g1.mean())
        p0 = float(g0.mean())
        diff = p1 - p0
        var = max((p1 * (1.0 - p1)) / n1 + (p0 * (1.0 - p0)) / n0, 0.0)
        weight = float(n1 + n0)
        strata_rows.append((diff, var, weight))

    n_strata_used = int(len(strata_rows))
    sufficient_sample = bool(
        n_exposed_total >= int(min_exposed_total)
        and n_unexposed_total >= int(min_unexposed_total)
        and n_strata_used >= int(min_strata)
    )
    if n_strata_used <= 0:
        return {
            "n_exposed": n_exposed_total,
            "n_unexposed": n_unexposed_total,
            "p_exposed": p_exposed,
            "p_unexposed": p_unexposed,
            "effect": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "sufficient_sample": False,
            "n_strata_used": n_strata_used,
        }

    diffs = np.array([r[0] for r in strata_rows], dtype=float)
    vars_ = np.array([r[1] for r in strata_rows], dtype=float)
    weights = np.array([r[2] for r in strata_rows], dtype=float)
    wsum = float(np.sum(weights))
    wnorm = weights / wsum if wsum > 0 else np.zeros_like(weights)
    effect = float(np.sum(wnorm * diffs))
    se = float(np.sqrt(np.sum((wnorm**2) * vars_)))
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    ci_low = effect - z * se if pd.notna(se) else np.nan
    ci_high = effect + z * se if pd.notna(se) else np.nan
    z_stat = effect / se if se > 0 else np.nan
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat)))) if pd.notna(z_stat) else np.nan
    return {
        "n_exposed": n_exposed_total,
        "n_unexposed": n_unexposed_total,
        "p_exposed": p_exposed,
        "p_unexposed": p_unexposed,
        "effect": effect,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "sufficient_sample": sufficient_sample,
        "n_strata_used": n_strata_used,
    }


def _add_fdr_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    p_col: str = "p_value",
    q_col: str = "q_value",
    alpha: float = 0.10,
) -> pd.DataFrame:
    if df.empty or p_col not in df.columns:
        out = df.copy()
        if q_col not in out.columns:
            out[q_col] = np.nan
        out["fdr_significant"] = False
        return out

    out = df.copy()
    out[q_col] = np.nan
    out["fdr_significant"] = False
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        sub = out.loc[idx].copy()
        p = pd.to_numeric(sub[p_col], errors="coerce")
        valid = p.notna()
        if valid.sum() <= 0:
            continue
        p_valid = p[valid].to_numpy(dtype=float)
        m = len(p_valid)
        order = np.argsort(p_valid)
        ranked = p_valid[order]
        q_ranked = ranked * m / (np.arange(1, m + 1))
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_ranked = np.clip(q_ranked, 0.0, 1.0)
        q = np.full(m, np.nan, dtype=float)
        q[order] = q_ranked
        q_series = pd.Series(q, index=p[valid].index)
        out.loc[q_series.index, q_col] = q_series
        out.loc[q_series.index, "fdr_significant"] = q_series <= float(alpha)
    return out


def build_event_impacts_on_metrics(
    conn: duckdb.DuckDBPyConnection,
    user_month_base: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    exclude_incomplete_month: bool,
    confidence_level: float,
) -> pd.DataFrame:
    if user_month_base.empty:
        return pd.DataFrame()

    ev = conn.execute(
        """
        SELECT DISTINCT
          unique_id,
          date_trunc('month', data_inicio) AS month,
          event_type
        FROM pop_primary_interactions
        WHERE data_inicio IS NOT NULL
          AND event_type IS NOT NULL
          AND trim(event_type) <> ''
        """
    ).fetchdf()
    if ev.empty:
        return pd.DataFrame()
    ev["month"] = pd.to_datetime(ev["month"], errors="coerce")
    ev["event_type"] = ev["event_type"].astype(str)

    base = user_month_base.copy()
    snapshot_month = pd.Timestamp(snapshot_ts).to_period("M").to_timestamp()
    base["is_decision_month"] = (
        (base["month"] + pd.offsets.MonthBegin(1)) < snapshot_month if exclude_incomplete_month else True
    )
    base = base[base["is_decision_month"] == True].copy()
    if base.empty:
        return pd.DataFrame()

    cal = load_school_calendar_month_uf(conn)
    if not cal.empty:
        keep_cal = [c for c in ["month", "uf", "school_days_ratio", "school_days_estimate"] if c in cal.columns]
        cal = cal[keep_cal].copy()
        cal["month"] = pd.to_datetime(cal["month"], errors="coerce")
        if "uf" in cal.columns:
            cal["uf"] = cal["uf"].astype(str).str.upper().str.strip()

    all_events = sorted(ev["event_type"].dropna().astype(str).unique().tolist())
    rows: List[Dict[str, Any]] = []
    segment_specs = ["all_users", "base_regular", "heavy_users"]

    for seg in segment_specs:
        seg_base = base.copy() if seg == "all_users" else base[base["segment"] == seg].copy()
        if seg_base.empty:
            continue
        key_cols = ["unique_id", "month", "strict_flag", "next_active", "rsva_outcome"]
        if "uf" in seg_base.columns:
            key_cols.append("uf")
        keys = seg_base[key_cols].copy()
        if "uf" in keys.columns:
            keys["uf"] = keys["uf"].astype(str).str.upper().str.strip()
            keys.loc[keys["uf"] == "", "uf"] = "NA"
        else:
            keys["uf"] = "NA"
        if not cal.empty and {"month", "uf"}.issubset(set(keys.columns)):
            keys = keys.merge(cal, on=["month", "uf"], how="left")
            if "school_days_ratio" in keys.columns:
                sdr = pd.to_numeric(keys["school_days_ratio"], errors="coerce")
                keys["school_days_ratio_bin"] = np.select(
                    [
                        sdr.isna(),
                        sdr < 0.85,
                        (sdr >= 0.85) & (sdr < 0.93),
                        sdr >= 0.93,
                    ],
                    [
                        "nao_mapeado",
                        "jornada_baixa",
                        "jornada_media",
                        "jornada_alta",
                    ],
                    default="nao_mapeado",
                )

        strata_cols = ["month"]
        if "uf" in keys.columns and keys["uf"].nunique(dropna=True) > 1:
            strata_cols.append("uf")
        if "school_days_ratio_bin" in keys.columns and keys["school_days_ratio_bin"].nunique(dropna=True) > 1:
            strata_cols.append("school_days_ratio_bin")

        for event in all_events:
            present = ev[ev["event_type"] == event][["unique_id", "month"]].copy()
            present["exposed"] = 1
            merged = keys.merge(present, on=["unique_id", "month"], how="left")
            merged["exposed"] = merged["exposed"].fillna(0).astype(int)

            exp = merged[merged["exposed"] == 1]
            unexp = merged[merged["exposed"] == 0]

            rsva = _diff_proportion_ci_month_adjusted(
                merged,
                outcome_col="rsva_outcome",
                confidence_level=confidence_level,
                strata_cols=strata_cols,
                min_exposed_total=30,
                min_unexposed_total=30,
                min_strata=3,
            )
            rows.append(
                {
                    "segment": seg,
                    "event_type": event,
                    "metric": "rsva_m1",
                    **rsva,
                }
            )

            svs = _diff_proportion_ci_month_adjusted(
                merged,
                outcome_col="strict_flag",
                confidence_level=confidence_level,
                strata_cols=strata_cols,
                min_exposed_total=30,
                min_unexposed_total=30,
                min_strata=3,
            )
            rows.append(
                {
                    "segment": seg,
                    "event_type": event,
                    "metric": "svs_t",
                    **svs,
                }
            )

            exp_strict = exp[exp["strict_flag"] == 1]
            unexp_strict = unexp[unexp["strict_flag"] == 1]
            strict_merged = pd.concat(
                [
                    exp_strict.assign(exposed=1),
                    unexp_strict.assign(exposed=0),
                ],
                ignore_index=True,
                sort=False,
            )
            sur = _diff_proportion_ci_month_adjusted(
                strict_merged,
                outcome_col="next_active",
                confidence_level=confidence_level,
                strata_cols=strata_cols,
                min_exposed_total=30,
                min_unexposed_total=30,
                min_strata=3,
            )
            rows.append(
                {
                    "segment": seg,
                    "event_type": event,
                    "metric": "sur_t",
                    "sur_selection_conditioned_on_strict": True,
                    **sur,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _add_fdr_by_group(out, group_cols=["segment", "metric"], p_col="p_value", q_col="q_value", alpha=0.10)
    out["negative_confirmed"] = (
        (pd.to_numeric(out["ci_high"], errors="coerce") < 0)
        & out["fdr_significant"].fillna(False)
        & out["sufficient_sample"].fillna(False)
    )
    out["positive_confirmed"] = (
        (pd.to_numeric(out["ci_low"], errors="coerce") > 0)
        & out["fdr_significant"].fillna(False)
        & out["sufficient_sample"].fillna(False)
    )
    return out.sort_values(["segment", "metric", "effect"], ascending=[True, True, True]).reset_index(drop=True)


def classify_event_for_pattern(event_type: str) -> str:
    ev = str(event_type or "").strip().lower()
    if not ev:
        return "outros"
    if ev in STRICT_DOWNLOAD_EVENTS:
        return "strict_download_aula_plano"
    if ev == "entrou_nova_escola":
        return "sinal_entrou_nova_escola"
    if "missao" in ev:
        return "missao_click"
    if "conteudo_ia" in ev:
        return "conteudo_ia"
    if "comunidade" in ev or "conquista" in ev:
        return "comunidade_conquistas"
    if "prova" in ev or "avaliacao" in ev:
        if ev.startswith("visualizacao_"):
            return "prova_visualizacao"
        return "prova_acao"
    if "plano" in ev:
        if "visualizacao" in ev:
            return "plano_visualizacao"
        return "plano_acao"
    if "aula" in ev:
        if "visualizacao" in ev:
            return "aula_visualizacao"
        return "aula_outros_sem_download"
    return "outros"


def describe_drop_diagnostic(code: str) -> str:
    mapping = {
        "queda_por_entrada_em_valor": "Value-Qualified Retention caiu porque Value Conversion Rate caiu e Post-Value Retention ficou estável: menos usuários ativos chegaram ao valor estrito no mês.",
        "queda_por_continuidade_pos_valor": "Value-Qualified Retention caiu porque Post-Value Retention caiu e Value Conversion Rate ficou estável: os usuários chegam ao valor, mas voltam menos no mês seguinte.",
        "queda_mista_adocao_e_retencao": "Value-Qualified Retention caiu com queda simultânea de Value Conversion Rate e Post-Value Retention: piora combinada de entrada em valor e continuidade.",
        "queda_combinacao_outros_sinais": "Value-Qualified Retention caiu com sinais mistos, sem padrão dominante único em Value Conversion Rate ou Post-Value Retention.",
        "sem_queda_de_rsva": "Não houve queda relevante de Value-Qualified Retention no mês.",
    }
    return mapping.get(str(code or ""), "Diagnóstico não mapeado.")


def describe_event_type(event_type: str) -> str:
    ev = str(event_type or "").strip().lower()
    mapping = {
        "prova_salva": "Salvou prova (ação de prova).",
        "prova_criada_edicao": "Criou/ediou prova (ação de prova).",
        "visualizacao_prova": "Visualizou prova sem necessariamente agir.",
        "visualizacao_prova_aprendizap": "Visualizou prova no Aprendizap sem necessariamente agir.",
        "download_aula": "Baixou aula (valor estrito).",
        "download_plano_aula": "Baixou plano de aula (valor estrito).",
        "click_subaba_concluidas": "Navegou em aba de conquistas concluídas.",
        "botao_baixar_conquista_completada": "Baixou comprovante/artefato de conquista concluída.",
        "fechar_conquista_obtida": "Fechou modal/tela de conquista obtida.",
    }
    if ev in mapping:
        return mapping[ev]
    if "visualizacao" in ev:
        return "Evento de visualização."
    if "download" in ev:
        return "Evento de download."
    if "prova" in ev:
        return "Evento da família Prova."
    if "aula" in ev:
        return "Evento da família Aula."
    return "Evento de navegação/engajamento."


def build_event_class_impacts_on_metrics(
    conn: duckdb.DuckDBPyConnection,
    user_month_base: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    exclude_incomplete_month: bool,
    confidence_level: float,
) -> pd.DataFrame:
    if user_month_base.empty:
        return pd.DataFrame()

    ev = conn.execute(
        """
        SELECT DISTINCT
          unique_id,
          date_trunc('month', data_inicio) AS month,
          event_type
        FROM pop_primary_interactions
        WHERE data_inicio IS NOT NULL
          AND event_type IS NOT NULL
          AND trim(event_type) <> ''
        """
    ).fetchdf()
    if ev.empty:
        return pd.DataFrame()
    ev["month"] = pd.to_datetime(ev["month"], errors="coerce")
    ev["event_type"] = ev["event_type"].astype(str)
    ev["event_class"] = ev["event_type"].apply(classify_event_for_pattern)
    ev = ev[["unique_id", "month", "event_class"]].drop_duplicates()

    base = user_month_base.copy()
    snapshot_month = pd.Timestamp(snapshot_ts).to_period("M").to_timestamp()
    base["is_decision_month"] = (
        (base["month"] + pd.offsets.MonthBegin(1)) < snapshot_month if exclude_incomplete_month else True
    )
    base = base[base["is_decision_month"] == True].copy()
    if base.empty:
        return pd.DataFrame()

    cal = load_school_calendar_month_uf(conn)
    if not cal.empty:
        keep_cal = [c for c in ["month", "uf", "school_days_ratio", "school_days_estimate"] if c in cal.columns]
        cal = cal[keep_cal].copy()
        cal["month"] = pd.to_datetime(cal["month"], errors="coerce")
        if "uf" in cal.columns:
            cal["uf"] = cal["uf"].astype(str).str.upper().str.strip()

    # Remove a classe strict para evitar tautologia (SVS/RSVA ficam mecanicamente inflados).
    all_classes = [c for c in sorted(ev["event_class"].dropna().astype(str).unique().tolist()) if c != "strict_download_aula_plano"]
    rows: List[Dict[str, Any]] = []
    segment_specs = ["all_users", "base_regular", "heavy_users"]

    for seg in segment_specs:
        seg_base = base.copy() if seg == "all_users" else base[base["segment"] == seg].copy()
        if seg_base.empty:
            continue
        key_cols = ["unique_id", "month", "strict_flag", "next_active", "rsva_outcome"]
        if "uf" in seg_base.columns:
            key_cols.append("uf")
        keys = seg_base[key_cols].copy()
        if "uf" in keys.columns:
            keys["uf"] = keys["uf"].astype(str).str.upper().str.strip()
            keys.loc[keys["uf"] == "", "uf"] = "NA"
        else:
            keys["uf"] = "NA"
        if not cal.empty and {"month", "uf"}.issubset(set(keys.columns)):
            keys = keys.merge(cal, on=["month", "uf"], how="left")
            if "school_days_ratio" in keys.columns:
                sdr = pd.to_numeric(keys["school_days_ratio"], errors="coerce")
                keys["school_days_ratio_bin"] = np.select(
                    [
                        sdr.isna(),
                        sdr < 0.85,
                        (sdr >= 0.85) & (sdr < 0.93),
                        sdr >= 0.93,
                    ],
                    [
                        "nao_mapeado",
                        "jornada_baixa",
                        "jornada_media",
                        "jornada_alta",
                    ],
                    default="nao_mapeado",
                )

        strata_cols = ["month"]
        if "uf" in keys.columns and keys["uf"].nunique(dropna=True) > 1:
            strata_cols.append("uf")
        if "school_days_ratio_bin" in keys.columns and keys["school_days_ratio_bin"].nunique(dropna=True) > 1:
            strata_cols.append("school_days_ratio_bin")

        for ev_class in all_classes:
            present = ev[ev["event_class"] == ev_class][["unique_id", "month"]].copy()
            present["exposed"] = 1
            merged = keys.merge(present, on=["unique_id", "month"], how="left")
            merged["exposed"] = merged["exposed"].fillna(0).astype(int)

            exp = merged[merged["exposed"] == 1]
            unexp = merged[merged["exposed"] == 0]

            rsva = _diff_proportion_ci_month_adjusted(
                merged,
                outcome_col="rsva_outcome",
                confidence_level=confidence_level,
                strata_cols=strata_cols,
                min_exposed_total=30,
                min_unexposed_total=30,
                min_strata=3,
            )
            rows.append({"segment": seg, "event_class": ev_class, "metric": "rsva_m1", **rsva})

            svs = _diff_proportion_ci_month_adjusted(
                merged,
                outcome_col="strict_flag",
                confidence_level=confidence_level,
                strata_cols=strata_cols,
                min_exposed_total=30,
                min_unexposed_total=30,
                min_strata=3,
            )
            rows.append({"segment": seg, "event_class": ev_class, "metric": "svs_t", **svs})

            exp_strict = exp[exp["strict_flag"] == 1]
            unexp_strict = unexp[unexp["strict_flag"] == 1]
            strict_merged = pd.concat(
                [
                    exp_strict.assign(exposed=1),
                    unexp_strict.assign(exposed=0),
                ],
                ignore_index=True,
                sort=False,
            )
            sur = _diff_proportion_ci_month_adjusted(
                strict_merged,
                outcome_col="next_active",
                confidence_level=confidence_level,
                strata_cols=strata_cols,
                min_exposed_total=30,
                min_unexposed_total=30,
                min_strata=3,
            )
            rows.append({"segment": seg, "event_class": ev_class, "metric": "sur_t", "sur_selection_conditioned_on_strict": True, **sur})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _add_fdr_by_group(out, group_cols=["segment", "metric"], p_col="p_value", q_col="q_value", alpha=0.10)
    out["negative_confirmed"] = (
        (pd.to_numeric(out["ci_high"], errors="coerce") < 0)
        & out["fdr_significant"].fillna(False)
        & out["sufficient_sample"].fillna(False)
    )
    out["positive_confirmed"] = (
        (pd.to_numeric(out["ci_low"], errors="coerce") > 0)
        & out["fdr_significant"].fillna(False)
        & out["sufficient_sample"].fillna(False)
    )
    return out.sort_values(["segment", "metric", "effect"], ascending=[True, True, True]).reset_index(drop=True)


def build_taxonomy_auxiliary(
    conn: duckdb.DuckDBPyConnection,
    snapshot_ts: pd.Timestamp,
    confidence_level: float,
    population_primary: str,
) -> pd.DataFrame:
    base = conn.execute(
        """
        WITH um AS (
          SELECT unique_id, date_trunc('month', data_inicio) AS month
          FROM pop_primary_interactions
          GROUP BY 1,2
        )
        SELECT
          u.unique_id,
          u.month,
          CASE WHEN u2.unique_id IS NULL THEN 0 ELSE 1 END AS next_active
        FROM um u
        LEFT JOIN um u2
          ON u.unique_id=u2.unique_id
         AND date_trunc('month', u.month + INTERVAL '1 month') = u2.month
        """
    ).fetchdf()
    if base.empty:
        return pd.DataFrame()
    base["month"] = pd.to_datetime(base["month"], errors="coerce")

    ev = conn.execute(
        """
        SELECT DISTINCT unique_id, date_trunc('month', data_inicio) AS month, event_type
        FROM pop_primary_interactions
        WHERE event_type IS NOT NULL AND trim(event_type) <> ''
        """
    ).fetchdf()
    if ev.empty:
        return pd.DataFrame()
    ev["month"] = pd.to_datetime(ev["month"], errors="coerce")

    all_events = sorted(ev["event_type"].dropna().astype(str).unique().tolist())
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    rows: List[Dict[str, Any]] = []

    keys = base[["unique_id", "month"]].copy()
    for event in all_events:
        present = ev[ev["event_type"] == event][["unique_id", "month"]].copy()
        present["exposed"] = 1
        merged = keys.merge(present, on=["unique_id", "month"], how="left")
        merged["exposed"] = merged["exposed"].fillna(0).astype(int)
        merged = merged.merge(base, on=["unique_id", "month"], how="left")
        merged["next_active"] = merged["next_active"].fillna(0).astype(int)

        exp = merged[merged["exposed"] == 1]
        unexp = merged[merged["exposed"] == 0]
        n1 = len(exp)
        n0 = len(unexp)
        if n1 < 3 or n0 < 3:
            rows.append(
                {
                    "event_type": event,
                    "adjusted_uplift_next_active": np.nan,
                    "adjusted_uplift_se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "exposed_user_months": float(n1),
                    "comparable_user_months": float(n1 + n0),
                    "aux_uplift_positive_flag": False,
                }
            )
            continue

        p1 = float(exp["next_active"].mean())
        p0 = float(unexp["next_active"].mean())
        uplift = p1 - p0
        se = float(np.sqrt(max(p1 * (1.0 - p1) / n1 + p0 * (1.0 - p0) / n0, 0.0)))
        ci_low = uplift - z * se
        ci_high = uplift + z * se
        rows.append(
            {
                "event_type": event,
                "adjusted_uplift_next_active": uplift,
                "adjusted_uplift_se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "exposed_user_months": float(n1),
                "comparable_user_months": float(n1 + n0),
                "aux_uplift_positive_flag": bool(ci_low > 0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["aux_uplift_positive_flag", "ci_low", "adjusted_uplift_next_active", "exposed_user_months", "event_type"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    snap = pd.Timestamp(snapshot_ts)
    quarter = int((snap.month - 1) // 3 + 1)
    version = f"aux_uplift_{snap.year}Q{quarter}_conf{int(round(confidence_level * 100))}"
    model_payload = {
        "population_primary": population_primary,
        "method": "univariate_uplift_difference",
        "confidence_level": confidence_level,
        "snapshot_ts": str(snapshot_ts),
    }
    model_hash = hashlib.sha256(json.dumps(model_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    out["event_taxonomy_version"] = version
    out["effective_date"] = pd.Timestamp(year=snap.year, month=((quarter - 1) * 3) + 1, day=1).date().isoformat()
    out["recompute_cadence"] = "quarterly"
    out["model_hash"] = model_hash
    out["confidence_level"] = confidence_level
    out["taxonomy_method"] = "aux_uplift_event_level"
    return out


def build_metric_selection(monthly_df: pd.DataFrame, include_extended_horizons: bool = True) -> Dict[str, Any]:
    if monthly_df.empty:
        return {"available": False, "reason": "empty_monthly"}
    x = monthly_df.copy().sort_values("month")
    horizons = [1, 2, 4, 6] if include_extended_horizons else [1, 2]
    for h in horizons:
        x[f"next_h{h}_active_users"] = x["active_users"].shift(-h)

    candidate_horizon = {
        "rsva_m1": 1,
        "retention_m1": 1,
        "svs_t": 1,
        "sur_t": 1,
        "broad_retained_share": 1,
        "rsva_m2": 2,
        "retention_m2": 2,
        "sur_m2": 2,
    }
    if include_extended_horizons:
        candidate_horizon.update(
            {
                "rsva_m4": 4,
                "retention_m4": 4,
                "sur_m4": 4,
                "rsva_m6": 6,
                "retention_m6": 6,
                "sur_m6": 6,
            }
        )

    candidates = [m for m in candidate_horizon.keys() if m in x.columns]
    rows: List[Dict[str, Any]] = []
    for m in candidates:
        h = int(candidate_horizon.get(m, 1))
        target_col = f"next_h{h}_active_users"
        if target_col not in x.columns:
            continue
        s = pd.to_numeric(x[m], errors="coerce")
        d_h = x[[m, target_col]].dropna()
        rho_h = float(abs(spearmanr(d_h[m], d_h[target_col]).correlation)) if len(d_h) >= 3 else np.nan
        ci_hw_col = f"{m}_ci_half_width"
        ci_hw = pd.to_numeric(x[ci_hw_col], errors="coerce") if ci_hw_col in x.columns else pd.Series(dtype=float)
        signal = float(np.nanmedian(np.abs(s.diff()))) if len(s) > 1 else np.nan
        noise = float(np.nanmedian(ci_hw)) if len(ci_hw) else np.nan
        rel = signal / noise if pd.notna(signal) and pd.notna(noise) and noise > 0 else np.nan
        rows.append(
            {
                "metric": m,
                "predictive_horizon_m": h,
                "decision_eligible": not m.startswith("broad_"),
                "abs_spearman_target_mau": rho_h,
                "abs_spearman_next_mau": rho_h,  # compatibilidade com outputs legados
                "abs_spearman_next2_mau": np.nan,  # compatibilidade; sem uso no ranking novo
                "median_reliability_ratio": rel,
                "interquartile_range": float(np.nanpercentile(s.dropna(), 75) - np.nanpercentile(s.dropna(), 25)) if s.notna().sum() >= 4 else np.nan,
                "median_ci_half_width": float(np.nanmedian(ci_hw)) if len(ci_hw) else np.nan,
            }
        )
    rank = pd.DataFrame(rows)
    if rank.empty:
        return {"available": False, "reason": "no_candidates"}

    for c, asc in [("abs_spearman_target_mau", False), ("median_reliability_ratio", False), ("median_ci_half_width", True)]:
        rank[f"rank_{c}"] = rank[c].rank(ascending=asc, method="average")
    rank["composite_rank"] = rank[[c for c in rank.columns if c.startswith("rank_")]].mean(axis=1)
    rank = rank.sort_values(["decision_eligible", "composite_rank"], ascending=[False, True]).reset_index(drop=True)

    eligible = rank[rank["decision_eligible"] == True].copy()
    best = eligible.iloc[0].to_dict() if not eligible.empty else rank.iloc[0].to_dict()
    return {
        "available": True,
        "best_metric": best.get("metric"),
        "ranking_table": rank.to_dict(orient="records"),
        "selection_criterion": "rank over horizon-target predictive correlation + reliability",
        "decision_eligibility_rule": "broad_* metrics are diagnostic_only",
        "selection_mode": "extended_horizons" if include_extended_horizons else "short_window_m1_m2",
    }


def build_rsva_retention_m2_comparison(monthly_panel: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    if monthly_panel.empty:
        return {"available": False, "reason": "empty_monthly_panel"}
    hz = build_horizon_comparison(monthly_panel, horizons=[2], seed=seed)
    if hz.empty:
        return {"available": False, "reason": "missing_required_columns_or_no_overlap"}
    row = hz.iloc[0]
    return {
        "available": True,
        "total_months": int(pd.to_numeric(row.get("months_used"), errors="coerce")),
        "mean_diff_rsva_minus_retention": float(pd.to_numeric(row.get("mean_diff_rsva_minus_retention"), errors="coerce")),
        "mean_abs_diff": float(pd.to_numeric(row.get("mean_abs_diff"), errors="coerce")),
        "max_abs_diff": float(pd.to_numeric(row.get("max_abs_diff"), errors="coerce")),
        "bootstrap_mean_diff_ci_low": float(pd.to_numeric(row.get("bootstrap_mean_diff_ci_low"), errors="coerce")),
        "bootstrap_mean_diff_ci_high": float(pd.to_numeric(row.get("bootstrap_mean_diff_ci_high"), errors="coerce")),
        "bootstrap_mean_diff_abs_limit": float(pd.to_numeric(row.get("bootstrap_mean_diff_abs_limit"), errors="coerce")),
        "numeric_conclusion": str(row.get("numeric_conclusion") or "distantes_no_criterio"),
        "recommended_usage": "use_both_prioritize_rsva_for_product_value_and_retention_as_baseline",
    }


def build_horizon_comparison(monthly_panel: pd.DataFrame, horizons: List[int] | None = None, seed: int = 42) -> pd.DataFrame:
    if monthly_panel.empty:
        return pd.DataFrame()
    hs = sorted({int(h) for h in (horizons or [1, 2, 4, 6]) if int(h) > 0})
    rows: List[Dict[str, Any]] = []

    for h in hs:
        rsva_col = f"rsva_m{h}"
        retention_col = f"retention_m{h}"
        sur_col = "sur_t" if h == 1 else f"sur_m{h}"
        decision_col = "is_decision_month" if h == 1 else f"is_decision_month_m{h}"
        required = {rsva_col, retention_col, "svs_t", sur_col}
        if not required.issubset(set(monthly_panel.columns)):
            continue
        dec = monthly_panel.copy()
        if decision_col in dec.columns:
            dec = dec[dec[decision_col] == True].copy()
        dec = dec.sort_values("month")
        if dec.empty:
            continue

        rsva = pd.to_numeric(dec[rsva_col], errors="coerce")
        retention = pd.to_numeric(dec[retention_col], errors="coerce")
        svs = pd.to_numeric(dec["svs_t"], errors="coerce")
        sur = pd.to_numeric(dec[sur_col], errors="coerce")
        diff = (rsva - retention).dropna()

        lo = np.nan
        hi = np.nan
        lim = np.nan
        mean_diff = float(diff.mean()) if len(diff) else np.nan
        mean_abs_diff = float(diff.abs().mean()) if len(diff) else np.nan
        max_abs_diff = float(diff.abs().max()) if len(diff) else np.nan
        if len(diff):
            # Seed por horizonte evita dependência da ordem de processamento.
            rng = np.random.default_rng(int(seed) + int(h))
            arr = diff.to_numpy(dtype=float)
            bs = []
            for _ in range(2000):
                sample = rng.choice(arr, size=len(arr), replace=True)
                bs.append(float(np.mean(sample)))
            lo = float(np.nanpercentile(bs, 2.5))
            hi = float(np.nanpercentile(bs, 97.5))
            lim = max(abs(lo), abs(hi))

        avg_svs = float(svs.mean()) if svs.notna().any() else np.nan
        avg_sur = float(sur.mean()) if sur.notna().any() else np.nan
        avg_rsva = float(rsva.mean()) if rsva.notna().any() else np.nan
        avg_retention = float(retention.mean()) if retention.notna().any() else np.nan
        reconstructed = avg_svs * avg_sur if pd.notna(avg_svs) and pd.notna(avg_sur) else np.nan
        recon_error = avg_rsva - reconstructed if pd.notna(avg_rsva) and pd.notna(reconstructed) else np.nan

        rows.append(
            {
                "horizon_m": h,
                "months_used": int(len(diff)),
                "mean_diff_rsva_minus_retention": mean_diff,
                "mean_abs_diff": mean_abs_diff,
                "max_abs_diff": max_abs_diff,
                "bootstrap_mean_diff_ci_low": lo,
                "bootstrap_mean_diff_ci_high": hi,
                "bootstrap_mean_diff_abs_limit": lim,
                "numeric_conclusion": "proximas_no_criterio" if pd.notna(lim) and lim <= 0.005 else "distantes_no_criterio",
                "avg_rsva": avg_rsva,
                "avg_retention": avg_retention,
                "avg_svs_t": avg_svs,
                "avg_sur_h": avg_sur,
                "avg_rsva_reconstructed_svs_x_sur": reconstructed,
                "avg_reconstruction_error": recon_error,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("horizon_m").reset_index(drop=True)


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
    if "mari" in s:
        return "mari"
    return "other"


def build_data_usage_audit(conn: duckdb.DuckDBPyConnection, cfg: Stage4Config) -> Dict[str, Any]:
    all_inter = float(conn.execute("SELECT COUNT(*)::DOUBLE FROM interactions WHERE data_inicio IS NOT NULL").fetchone()[0] or 0.0)
    reg_inter = float(
        conn.execute(
            "SELECT COUNT(*)::DOUBLE FROM interactions WHERE data_inicio IS NOT NULL AND lower(coalesce(user_type,''))='registered'"
        ).fetchone()[0]
        or 0.0
    )
    primary = conn.execute(
        """
        SELECT
          COUNT(*)::DOUBLE AS n,
          COUNT(DISTINCT unique_id)::DOUBLE AS ids,
          AVG(CASE WHEN event_type IS NULL OR trim(event_type)='' THEN 1.0 ELSE 0.0 END) AS missing_event_type_rate
        FROM pop_primary_interactions
        """
    ).fetchone()
    primary_n = float(primary[0] or 0.0)
    primary_ids = float(primary[1] or 0.0)
    missing_event_type_rate = float(primary[2]) if primary and primary[2] is not None else np.nan

    utm = conn.execute(
        """
        SELECT coalesce(nullif(trim(utm_origin),''), utm_source) AS utm_raw, COUNT(*)::DOUBLE AS n
        FROM pop_primary_interactions
        GROUP BY 1
        """
    ).fetchdf()
    if utm.empty:
        seo_like_share = np.nan
    else:
        utm["utm_group"] = utm["utm_raw"].apply(normalize_utm)
        total = float(utm["n"].sum())
        seo = float(utm[utm["utm_group"].isin(["paid_search", "organic_search"])]["n"].sum())
        seo_like_share = seo / total if total > 0 else np.nan

    return {
        "population_primary": cfg.population_primary,
        "all_interactions_with_ts": all_inter,
        "registered_interactions_with_ts": reg_inter,
        "registered_share_all_interactions": reg_inter / all_inter if all_inter > 0 else np.nan,
        "primary_interactions": primary_n,
        "primary_unique_ids": primary_ids,
        "primary_interaction_share_of_all": primary_n / all_inter if all_inter > 0 else np.nan,
        "primary_missing_event_type_rate": missing_event_type_rate,
        "seo_like_share_in_primary": seo_like_share,
        "measure_alignment": [
            {
                "measure": "Filtro interactions sem data_inicio",
                "applied_in_stage4": True,
                "status": "applied",
                "details": "Aplicado na view pop_primary_interactions.",
            },
            {
                "measure": "Filtro user_type=registered",
                "applied_in_stage4": bool(cfg.population_primary == "matched_registered"),
                "status": "applied" if cfg.population_primary == "matched_registered" else "not_applied",
                "details": "Ativo somente em matched_registered.",
            },
            {
                "measure": "Exclusão de tráfego SEO",
                "applied_in_stage4": False,
                "status": "not_applied",
                "details": "SEO permanece na população e é analisado por segmentação.",
            },
        ],
    }


def load_teacher_dataset_from_output(output_dir: Path) -> pd.DataFrame:
    p_parquet = output_dir / "parquet" / "teacher_dataset.parquet"
    p_csv_full = output_dir / "teacher_dataset.csv"
    p_csv_sample = output_dir / "teacher_analytical_dataset_sample.csv"
    if p_parquet.exists():
        conn = duckdb.connect(database=":memory:")
        try:
            return conn.execute(f"SELECT * FROM read_parquet('{q(p_parquet)}')").fetchdf()
        finally:
            conn.close()
    if p_csv_full.exists():
        return pd.read_csv(p_csv_full)
    if p_csv_sample.exists():
        return pd.read_csv(p_csv_sample)
    return pd.DataFrame()


def _normalize_category(v: Any) -> str:
    if v is None:
        return "missing"
    if isinstance(v, float) and np.isnan(v):
        return "missing"
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>", "missing"}:
        return "missing"
    return s


def _two_prop_p_value(
    group_heavy: int,
    group_total: int,
    total_heavy: int,
    total_users: int,
) -> float:
    group_heavy = int(group_heavy)
    group_total = int(group_total)
    total_heavy = int(total_heavy)
    total_users = int(total_users)
    out_heavy = total_heavy - group_heavy
    out_total = total_users - group_total
    if group_total <= 0 or out_total <= 0:
        return np.nan
    pooled = total_heavy / total_users if total_users > 0 else np.nan
    if pd.isna(pooled) or pooled <= 0 or pooled >= 1:
        return np.nan
    se = np.sqrt(pooled * (1.0 - pooled) * ((1.0 / group_total) + (1.0 / out_total)))
    if se <= 0:
        return np.nan
    p_group = group_heavy / group_total
    p_out = out_heavy / out_total
    z = (p_group - p_out) / se
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def _build_heavy_enrichment_table(
    source_df: pd.DataFrame,
    category_col: str,
    dimension_label: str,
    min_users: int,
    min_heavy: int,
    base_population_label: str,
) -> pd.DataFrame:
    req = {"heavy_user_flag", category_col}
    if source_df.empty or not req.issubset(set(source_df.columns)):
        return pd.DataFrame()

    work = source_df[[category_col, "heavy_user_flag"]].copy()
    work[category_col] = work[category_col].apply(_normalize_category)
    work["heavy_user_flag"] = pd.to_numeric(work["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    grouped = (
        work.groupby(category_col, dropna=False)
        .agg(
            users_total=("heavy_user_flag", "size"),
            heavy_users=("heavy_user_flag", "sum"),
        )
        .reset_index()
        .rename(columns={category_col: "category"})
    )
    if grouped.empty:
        return grouped

    total_users = int(len(work))
    total_heavy = int(work["heavy_user_flag"].sum())
    base_heavy_rate = (total_heavy / total_users) if total_users > 0 else np.nan
    grouped["heavy_rate_in_category"] = grouped["heavy_users"] / grouped["users_total"].replace(0, np.nan)
    grouped["heavy_rate_overall"] = float(base_heavy_rate) if pd.notna(base_heavy_rate) else np.nan
    grouped["lift_vs_overall"] = grouped["heavy_rate_in_category"] / grouped["heavy_rate_overall"].replace(0, np.nan)
    grouped["delta_pp_vs_overall"] = grouped["heavy_rate_in_category"] - grouped["heavy_rate_overall"]
    grouped["p_value_vs_rest"] = grouped.apply(
        lambda r: _two_prop_p_value(
            int(r["heavy_users"]),
            int(r["users_total"]),
            total_heavy,
            total_users,
        ),
        axis=1,
    )
    grouped["reliable_heavy_type"] = (
        (grouped["users_total"] >= int(min_users))
        & (grouped["heavy_users"] >= int(min_heavy))
        & (pd.to_numeric(grouped["p_value_vs_rest"], errors="coerce") < 0.05)
        & (pd.to_numeric(grouped["lift_vs_overall"], errors="coerce") >= 1.10)
        & (grouped["category"] != "missing")
    )
    grouped["dimension"] = str(dimension_label)
    grouped["population_base"] = str(base_population_label)
    grouped["population_users_in_dimension"] = int(total_users)
    grouped["dimension_reliability_rule"] = (
        f"users_total>={int(min_users)}; heavy_users>={int(min_heavy)}; p_value<0.05; lift>=1.10; categoria!=missing"
    )
    grouped = grouped[
        [
            "dimension",
            "category",
            "users_total",
            "heavy_users",
            "heavy_rate_in_category",
            "heavy_rate_overall",
            "lift_vs_overall",
            "delta_pp_vs_overall",
            "p_value_vs_rest",
            "reliable_heavy_type",
            "population_base",
            "population_users_in_dimension",
            "dimension_reliability_rule",
        ]
    ].copy()
    grouped = grouped.sort_values(["reliable_heavy_type", "lift_vs_overall", "users_total"], ascending=[False, False, False])
    grouped["rank_within_dimension"] = (
        grouped.groupby("dimension")["lift_vs_overall"].rank(method="first", ascending=False).astype(int)
    )
    return grouped.reset_index(drop=True)


def build_user_time_preferences(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    WITH base AS (
      SELECT
        unique_id,
        CAST(strftime(data_inicio, '%w') AS INTEGER) AS dow,
        CAST(strftime(data_inicio, '%H') AS INTEGER) AS hour_of_day
      FROM pop_primary_interactions
      WHERE data_inicio IS NOT NULL
    ),
    totals AS (
      SELECT unique_id, COUNT(*)::BIGINT AS ts_events
      FROM base
      GROUP BY 1
    ),
    dow_pref AS (
      SELECT
        unique_id,
        dow,
        COUNT(*)::BIGINT AS dow_events,
        ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY COUNT(*) DESC, dow ASC) AS rn
      FROM base
      GROUP BY 1,2
    ),
    hour_prep AS (
      SELECT
        unique_id,
        CASE
          WHEN hour_of_day BETWEEN 0 AND 5 THEN 'madrugada'
          WHEN hour_of_day BETWEEN 6 AND 11 THEN 'manha'
          WHEN hour_of_day BETWEEN 12 AND 17 THEN 'tarde'
          ELSE 'noite'
        END AS hour_bin,
        CASE
          WHEN hour_of_day BETWEEN 0 AND 5 THEN 0
          WHEN hour_of_day BETWEEN 6 AND 11 THEN 1
          WHEN hour_of_day BETWEEN 12 AND 17 THEN 2
          ELSE 3
        END AS hour_ord
      FROM base
    ),
    hour_pref AS (
      SELECT
        unique_id,
        hour_bin,
        COUNT(*)::BIGINT AS hour_events,
        ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY COUNT(*) DESC, MIN(hour_ord) ASC) AS rn
      FROM hour_prep
      GROUP BY 1,2
    )
    SELECT
      t.unique_id,
      t.ts_events,
      CASE
        WHEN d.dow = 0 THEN 'domingo'
        WHEN d.dow = 1 THEN 'segunda'
        WHEN d.dow = 2 THEN 'terca'
        WHEN d.dow = 3 THEN 'quarta'
        WHEN d.dow = 4 THEN 'quinta'
        WHEN d.dow = 5 THEN 'sexta'
        WHEN d.dow = 6 THEN 'sabado'
        ELSE 'missing'
      END AS dominant_weekday,
      COALESCE(h.hour_bin, 'missing') AS dominant_hour_bin
    FROM totals t
    LEFT JOIN dow_pref d
      ON t.unique_id = d.unique_id
     AND d.rn = 1
    LEFT JOIN hour_pref h
      ON t.unique_id = h.unique_id
     AND h.rn = 1
    """
    out = conn.execute(sql).fetchdf()
    if out.empty:
        return out
    out["ts_events"] = pd.to_numeric(out["ts_events"], errors="coerce")
    out["dominant_weekday"] = out["dominant_weekday"].apply(_normalize_category)
    out["dominant_hour_bin"] = out["dominant_hour_bin"].apply(_normalize_category)
    return out


def build_heavy_user_type_profiles(
    conn: duckdb.DuckDBPyConnection,
    output_dir: Path,
    teacher_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    teacher_df = teacher_df.copy() if teacher_df is not None else load_teacher_dataset_from_output(output_dir)
    if teacher_df.empty:
        return pd.DataFrame(), {
            "status": "no_teacher_dataset",
            "notes": "teacher_dataset não encontrado no output_dir; tipologias heavy não calculadas.",
        }

    df = teacher_df.copy()
    if "unique_id" not in df.columns or "interaction_count" not in df.columns:
        return pd.DataFrame(), {
            "status": "missing_required_columns",
            "notes": "teacher_dataset sem colunas obrigatórias (unique_id, interaction_count).",
        }
    if "active_user_heavy_window_flag" in df.columns:
        df["active_user_heavy_window_flag"] = pd.to_numeric(
            df["active_user_heavy_window_flag"], errors="coerce"
        ).fillna(0).astype(int)
        active = df[df["active_user_heavy_window_flag"] == 1].copy()
        base_population_label = "active_users_heavy_window_flag_1"
    else:
        df["interaction_count"] = pd.to_numeric(df["interaction_count"], errors="coerce").fillna(0.0)
        active = df[df["interaction_count"] > 0].copy()
        base_population_label = "active_users_interaction_count_gt_0"
    if active.empty:
        return pd.DataFrame(), {
            "status": "no_active_users",
            "notes": "Não há usuários ativos para calcular tipologias heavy.",
        }

    if "heavy_user_flag" not in active.columns:
        return pd.DataFrame(), {
            "status": "missing_heavy_user_flag",
            "notes": "teacher_dataset sem heavy_user_flag da etapa 01; tipologias heavy não calculadas para evitar definição paralela.",
        }
    active["heavy_user_flag"] = pd.to_numeric(active["heavy_user_flag"], errors="coerce").fillna(0).astype(int)

    overall_heavy_rate = float(active["heavy_user_flag"].mean()) if len(active) else np.nan
    heavy_users = int(active["heavy_user_flag"].sum())

    state_col = "estado_group" if "estado_group" in active.columns else ("estado" if "estado" in active.columns else None)
    subject_col = "top_discipline" if "top_discipline" in active.columns else None
    device_col = "primary_device" if "primary_device" in active.columns else None

    tables: List[pd.DataFrame] = []
    if device_col:
        tables.append(
            _build_heavy_enrichment_table(
                active,
                category_col=device_col,
                dimension_label="device",
                min_users=300,
                min_heavy=30,
                base_population_label=base_population_label,
            )
        )
    if state_col:
        tables.append(
            _build_heavy_enrichment_table(
                active,
                category_col=state_col,
                dimension_label="location_estado",
                min_users=500,
                min_heavy=35,
                base_population_label=base_population_label,
            )
        )
    if subject_col:
        tables.append(
            _build_heavy_enrichment_table(
                active,
                category_col=subject_col,
                dimension_label="subject_top_discipline",
                min_users=300,
                min_heavy=25,
                base_population_label=base_population_label,
            )
        )

    time_pref = build_user_time_preferences(conn)
    if not time_pref.empty and {"unique_id", "ts_events", "dominant_weekday", "dominant_hour_bin"}.issubset(set(time_pref.columns)):
        time_df = active[["unique_id", "heavy_user_flag"]].merge(time_pref, on="unique_id", how="left")
        eligible = time_df[pd.to_numeric(time_df["ts_events"], errors="coerce") >= 20].copy()
        if not eligible.empty:
            tables.append(
                _build_heavy_enrichment_table(
                    eligible,
                    category_col="dominant_weekday",
                    dimension_label="usage_dominant_weekday",
                    min_users=500,
                    min_heavy=40,
                    base_population_label=f"{base_population_label}_with_ts_events_ge_20",
                )
            )
            tables.append(
                _build_heavy_enrichment_table(
                    eligible,
                    category_col="dominant_hour_bin",
                    dimension_label="usage_dominant_hour_bin",
                    min_users=500,
                    min_heavy=40,
                    base_population_label=f"{base_population_label}_with_ts_events_ge_20",
                )
            )

    tables = [t for t in tables if t is not None and not t.empty]
    if not tables:
        return pd.DataFrame(), {
            "status": "insufficient_signal",
            "notes": "Sem dimensões com cobertura mínima para tipologias heavy.",
            "active_users": int(len(active)),
            "heavy_users": heavy_users,
            "overall_heavy_rate": overall_heavy_rate,
        }

    out = pd.concat(tables, ignore_index=True, sort=False)
    out = out.sort_values(["dimension", "reliable_heavy_type", "lift_vs_overall", "users_total"], ascending=[True, False, False, False]).reset_index(drop=True)

    reliable = out[out["reliable_heavy_type"] == True].copy()
    top_reliable = reliable.groupby("dimension", as_index=False, group_keys=False).head(5) if not reliable.empty else pd.DataFrame()
    summary = {
        "status": "ok",
        "active_users": int(len(active)),
        "heavy_users": heavy_users,
        "overall_heavy_rate": overall_heavy_rate,
        "heavy_definition": "heavy_user_flag herdado da etapa 01 (heavy_score_fast_v1: PCA-1 em intensidade/consistência + threshold otimizado em holdout).",
        "active_population_rule": base_population_label,
        "time_preference_rule": "dominância de dia/horário calculada apenas para usuários com >=20 eventos com timestamp.",
        "lift_reference_rule": "lift_vs_overall e delta_pp_vs_overall usam a taxa heavy da própria base da dimensão (population_base).",
        "reliability_rule_global": "categoria confiável se users_total mínimo, heavy_users mínimo, p_value<0.05 e lift>=1.10.",
        "reliable_heavy_types_found": int(len(reliable)),
        "top_reliable_by_dimension": top_reliable[
            ["dimension", "category", "users_total", "heavy_users", "heavy_rate_in_category", "lift_vs_overall", "delta_pp_vs_overall", "p_value_vs_rest"]
        ].to_dict(orient="records")
        if not top_reliable.empty
        else [],
    }
    return out, summary


def build_pipeline_consistency_audit(
    cfg: Stage4Config,
    decomposition_df: pd.DataFrame,
    user_month_segment_base: pd.DataFrame,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add_check(check_id: str, status: str, detail: str, metrics: Dict[str, Any] | None = None) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": status,
                "detail": detail,
                "metrics": metrics or {},
            }
        )

    pipeline_script = cfg.base_dir / "executar_pipeline_analytics.py"
    stage4_in_pipeline = False
    if pipeline_script.exists():
        try:
            txt = pipeline_script.read_text(encoding="utf-8")
            stage4_in_pipeline = "etapa_04_metricas_mensais.py" in txt
        except Exception:
            stage4_in_pipeline = False
    add_check(
        "pipeline_runs_stage4",
        "pass" if stage4_in_pipeline else "fail",
        "Pipeline principal inclui execução automática da etapa 04.",
        {"pipeline_script": str(pipeline_script), "stage4_detected": bool(stage4_in_pipeline)},
    )

    if decomposition_df is not None and not decomposition_df.empty and {"rsva_m1", "svs_t", "sur_t"}.issubset(set(decomposition_df.columns)):
        dec = decomposition_df.copy()
        if "is_decision_month" in dec.columns:
            dec = dec[dec["is_decision_month"] == True].copy()
        if dec.empty:
            dec = decomposition_df.copy()
        err = (
            pd.to_numeric(dec["rsva_m1"], errors="coerce")
            - (
                pd.to_numeric(dec["svs_t"], errors="coerce")
                * pd.to_numeric(dec["sur_t"], errors="coerce")
            )
        ).abs()
        mae = float(err.mean()) if err.notna().any() else np.nan
        mx = float(err.max()) if err.notna().any() else np.nan
        add_check(
            "rsva_identity_consistency",
            "pass" if (pd.notna(mae) and mae <= 1e-10) else ("warning" if (pd.notna(mae) and mae <= 1e-6) else "fail"),
            "Identidade algébrica da decomposição: Value-Qualified Retention_m1 = Value Conversion Rate x Post-Value Retention_t.",
            {"mean_abs_error": mae, "max_abs_error": mx},
        )
    else:
        add_check(
            "rsva_identity_consistency",
            "warning",
            "Dados insuficientes para validar identidade Value-Qualified Retention = Value Conversion Rate x Post-Value Retention.",
            {},
        )

    teacher_df = load_teacher_dataset_from_output(cfg.output_dir)
    if (
        not teacher_df.empty
        and not user_month_segment_base.empty
        and {"unique_id", "interaction_count", "heavy_user_flag"}.issubset(set(teacher_df.columns))
        and {"unique_id", "segment"}.issubset(set(user_month_segment_base.columns))
    ):
        td = teacher_df.copy()
        td["interaction_count"] = pd.to_numeric(td["interaction_count"], errors="coerce").fillna(0.0)
        td["heavy_user_flag"] = pd.to_numeric(td["heavy_user_flag"], errors="coerce").fillna(0).astype(int)

        seg = user_month_segment_base[["unique_id", "segment"]].drop_duplicates("unique_id").copy()
        seg["stage4_heavy_flag"] = (seg["segment"].astype(str) == "heavy_users").astype(int)
        merged = td[["unique_id", "heavy_user_flag"]].merge(seg[["unique_id", "stage4_heavy_flag"]], on="unique_id", how="inner")
        if not merged.empty:
            mismatch_rate = float((merged["heavy_user_flag"] != merged["stage4_heavy_flag"]).mean())
            mismatch_n = int((merged["heavy_user_flag"] != merged["stage4_heavy_flag"]).sum())
            add_check(
                "heavy_assignment_alignment",
                "pass" if mismatch_rate == 0.0 else ("warning" if mismatch_rate <= 0.005 else "fail"),
                "Atribuição de heavy por usuário é consistente entre etapa 01 e etapa 04.",
                {"users_compared": int(len(merged)), "mismatch_users": mismatch_n, "mismatch_rate": mismatch_rate},
            )
        else:
            add_check(
                "heavy_assignment_alignment",
                "warning",
                "Sem interseção de usuários para comparar heavy entre etapa 01 e etapa 04.",
                {},
            )
        heavy_cols = {"heavy_score_pca1", "heavy_threshold_quantile", "heavy_threshold_value", "active_user_heavy_window_flag"}
        cluster_signal = bool(len(heavy_cols.intersection(set(td.columns))) >= 3)
        add_check(
            "heavy_definition_method_alignment",
            "pass" if cluster_signal else "warning",
            "Etapa 04 utiliza heavy_user_flag herdado da etapa 01 (heavy_score_fast_v1 com threshold congelado).",
            {"heavy_definition_columns_present": sorted(list(heavy_cols.intersection(set(td.columns))))},
        )
    else:
        add_check(
            "heavy_definition_alignment",
            "warning",
            "Arquivos/colunas insuficientes para checar alinhamento da definição de heavy.",
            {},
        )

    cluster_stage1 = cfg.output_dir / "cluster_profiles.csv"
    cluster_stage2 = cfg.output_dir / "deep_dive_cluster_profiles_detailed.csv"
    if cluster_stage1.exists() and cluster_stage2.exists():
        c1 = pd.read_csv(cluster_stage1)
        c2 = pd.read_csv(cluster_stage2)
        if {"cluster", "teachers"}.issubset(set(c1.columns)) and {"cluster", "teachers"}.issubset(set(c2.columns)):
            cnt1 = sorted(pd.to_numeric(c1["teachers"], errors="coerce").dropna().astype(int).tolist())
            cnt2 = sorted(pd.to_numeric(c2["teachers"], errors="coerce").dropna().astype(int).tolist())
            same_counts = cnt1 == cnt2
            same_k = len(cnt1) == len(cnt2)
            add_check(
                "cluster_output_alignment",
                "pass" if (same_k and same_counts) else "warning",
                "Perfis de cluster da etapa 01 e etapa 02 mantêm a mesma partição agregada.",
                {
                    "stage1_clusters": int(len(cnt1)),
                    "stage2_clusters": int(len(cnt2)),
                    "teacher_count_vector_equal": bool(same_counts),
                    "stage1_teacher_counts_sorted": cnt1,
                    "stage2_teacher_counts_sorted": cnt2,
                },
            )
        else:
            add_check(
                "cluster_output_alignment",
                "warning",
                "Arquivos de cluster sem colunas mínimas para comparação.",
                {},
            )
    else:
        add_check(
            "cluster_output_alignment",
            "warning",
            "Arquivos de cluster não encontrados para comparação etapa 01 vs etapa 02.",
            {"cluster_stage1_exists": cluster_stage1.exists(), "cluster_stage2_exists": cluster_stage2.exists()},
        )

    cstatus_path = cfg.output_dir / "consolidated_status.json"
    deep_summary_path = cfg.output_dir / "deep_dive_summary.json"
    if cstatus_path.exists() and deep_summary_path.exists():
        cs = read_json(cstatus_path)
        ds = read_json(deep_summary_path)
        f1 = (
            cs.get("clustering", {})
            .get("artifacts_summary", {})
            .get("cluster_feature_cols", [])
        )
        f2 = (
            ds.get("cluster_definition_consistency", {})
            .get("feature_columns", [])
        )
        same_features = list(f1) == list(f2) and len(f1) > 0
        add_check(
            "cluster_feature_definition_alignment",
            "pass" if same_features else "warning",
            "Conjunto de features de cluster está alinhado entre etapa 01 e etapa 02.",
            {"stage1_features": f1, "stage2_features": f2},
        )
    else:
        add_check(
            "cluster_feature_definition_alignment",
            "warning",
            "Metadados insuficientes para comparar features de cluster entre etapas.",
            {"consolidated_status_exists": cstatus_path.exists(), "deep_dive_summary_exists": deep_summary_path.exists()},
        )

    fail_n = sum(1 for c in checks if c.get("status") == "fail")
    warn_n = sum(1 for c in checks if c.get("status") == "warning")
    overall = "fail" if fail_n > 0 else ("warning" if warn_n > 0 else "pass")
    return {
        "generated_at_utc": utc_now_iso(),
        "overall_status": overall,
        "checks": checks,
        "counts": {
            "pass": int(sum(1 for c in checks if c.get("status") == "pass")),
            "warning": int(warn_n),
            "fail": int(fail_n),
        },
    }


def write_pipeline_consistency_report(output_dir: Path, audit_payload: Dict[str, Any]) -> str:
    report_path = output_dir / "reports" / "pipeline_consistency_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = audit_payload.get("checks", []) if isinstance(audit_payload, dict) else []
    lines = [
        "# Auditoria de consistência do pipeline (etapas 01-04)",
        "",
        f"- overall_status: {audit_payload.get('overall_status') if isinstance(audit_payload, dict) else 'n/d'}",
        f"- generated_at_utc: {audit_payload.get('generated_at_utc') if isinstance(audit_payload, dict) else 'n/d'}",
        "",
        "## Checks",
    ]
    if checks:
        for c in checks:
            cid = str(c.get("check_id", "check"))
            status = str(c.get("status", "n/d"))
            detail = str(c.get("detail", ""))
            lines.append(f"- {cid}: status={status} | {detail}")
    else:
        lines.append("- Sem checks disponíveis.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def build_metric_dictionary() -> Dict[str, Any]:
    return {
        "strict_value_definition": {
            "rule": "strict_value = download_aula OR download_plano_aula",
            "strict_events": STRICT_DOWNLOAD_EVENTS,
            "note": "visualização e click não contam como valor estrito",
        },
        "display_terminology": {
            "SVS_t": "Value Conversion Rate",
            "SUR_t": "Post-Value Retention_t",
            "SUR_h": "Post-Value Retention_h",
            "RSVA_m1": "Value-Qualified Retention_m1",
            "RSVA_mh": "Value-Qualified Retention_h",
        },
        "calculation_definitions": METRIC_DEFINITION_ROWS,
        "rsva_m1": {
            "name": "value_qualified_retention_m1",
            "display_name": "Value-Qualified Retention_m1",
            "formula": "RSVA_m1 = strict_retained_users / active_users",
            "probability_form": "P(active em m+1 e strict_value em m | active em m)",
            "numerator": "strict_retained_users",
            "denominator": "active_users",
        },
        "rsva_m2": {
            "name": "value_qualified_retention_m2",
            "display_name": "Value-Qualified Retention_m2",
            "formula": "RSVA_m2 = strict_retained_m2_users / active_users",
            "probability_form": "P(active em m+2 e strict_value em m | active em m)",
            "numerator": "strict_retained_m2_users",
            "denominator": "active_users",
        },
        "rsva_m4": {
            "name": "value_qualified_retention_m4",
            "display_name": "Value-Qualified Retention_m4",
            "formula": "RSVA_m4 = strict_retained_m4_users / active_users",
            "probability_form": "P(active em m+4 e strict_value em m | active em m)",
            "numerator": "strict_retained_m4_users",
            "denominator": "active_users",
        },
        "rsva_m6": {
            "name": "value_qualified_retention_m6",
            "display_name": "Value-Qualified Retention_m6",
            "formula": "RSVA_m6 = strict_retained_m6_users / active_users",
            "probability_form": "P(active em m+6 e strict_value em m | active em m)",
            "numerator": "strict_retained_m6_users",
            "denominator": "active_users",
        },
        "rsva_mh": {
            "display_name": "Value-Qualified Retention_h",
            "formula": "RSVA_mh = strict_retained_mh_users / active_users",
            "applies_to_horizons": [2, 4, 6],
            "expanded_metrics": ["rsva_m2", "rsva_m4", "rsva_m6"],
        },
        "svs_t": {
            "name": "value_conversion_rate",
            "display_name": "Value Conversion Rate",
            "formula": "SVS_t = strict_users / active_users",
            "probability_form": "P(strict_value em m | active em m)",
            "numerator": "strict_users",
            "denominator": "active_users",
        },
        "sur_t": {
            "name": "post_value_retention_t",
            "display_name": "Post-Value Retention_t",
            "formula": "SUR_t = strict_retained_users / strict_users",
            "probability_form": "P(active em m+1 | strict_value em m)",
            "numerator": "strict_retained_users",
            "denominator": "strict_users",
        },
        "sur_m2": {
            "name": "post_value_retention_m2",
            "display_name": "Post-Value Retention_m2",
            "formula": "SUR_m2 = strict_retained_m2_users / strict_users",
            "probability_form": "P(active em m+2 | strict_value em m)",
            "numerator": "strict_retained_m2_users",
            "denominator": "strict_users",
        },
        "sur_m4": {
            "name": "post_value_retention_m4",
            "display_name": "Post-Value Retention_m4",
            "formula": "SUR_m4 = strict_retained_m4_users / strict_users",
            "probability_form": "P(active em m+4 | strict_value em m)",
            "numerator": "strict_retained_m4_users",
            "denominator": "strict_users",
        },
        "sur_m6": {
            "name": "post_value_retention_m6",
            "display_name": "Post-Value Retention_m6",
            "formula": "SUR_m6 = strict_retained_m6_users / strict_users",
            "probability_form": "P(active em m+6 | strict_value em m)",
            "numerator": "strict_retained_m6_users",
            "denominator": "strict_users",
        },
        "sur_h": {
            "display_name": "Post-Value Retention_h",
            "formula": "SUR_h = strict_retained_mh_users / strict_users",
            "applies_to_horizons": [2, 4, 6],
            "expanded_metrics": ["sur_m2", "sur_m4", "sur_m6"],
        },
        "retention_m1": {
            "name": "retention_all_active_m1",
            "formula": "P(active em m+1 | active em m)",
            "numerator": "retained_users",
            "denominator": "active_users",
        },
        "retention_m2": {
            "name": "retention_all_active_m2",
            "formula": "P(active em m+2 | active em m)",
            "numerator": "retained_m2_users",
            "denominator": "active_users",
        },
        "retention_m4": {
            "name": "retention_all_active_m4",
            "formula": "P(active em m+4 | active em m)",
            "numerator": "retained_m4_users",
            "denominator": "active_users",
        },
        "retention_m6": {
            "name": "retention_all_active_m6",
            "formula": "P(active em m+6 | active em m)",
            "numerator": "retained_m6_users",
            "denominator": "active_users",
        },
        "identity": {
            "constraint": "RSVA_m1 = SVS_t * SUR_t",
            "constraint_display": "Value-Qualified Retention_m1 = Value Conversion Rate * Post-Value Retention_t",
            "description": "para cada horizonte h: RSVA_mh = SVS_t * SUR_h",
        },
    }


def _format_month_axis(ax: Any, interval: int = 1) -> None:
    if mdates is None:
        return
    iv = max(int(interval), 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=iv))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def generate_charts(
    output_dir: Path,
    monthly_panel: pd.DataFrame,
    metric_uncertainty_df: pd.DataFrame,
    strict_cohort_hazard_df: pd.DataFrame,
    diag: Dict[str, Any],
    segment_monthly_df: pd.DataFrame,
    segment_drop_diag: Dict[str, Any],
    taxonomy_df: pd.DataFrame,
    event_family_df: pd.DataFrame,
    subject_quality_df: pd.DataFrame,
    subject_top_overall_df: pd.DataFrame,
    subject_top_monthly_df: pd.DataFrame,
    rsva_linear_models_df: pd.DataFrame,
    rsva_linear_fit_df: pd.DataFrame,
    event_impacts_df: pd.DataFrame,
    event_class_impacts_df: pd.DataFrame,
    horizon_comparison_df: pd.DataFrame,
    heavy_user_types_df: pd.DataFrame,
) -> Dict[str, Any]:
    charts_dir = output_dir / "reports" / "metric_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    # Remove charts antigos para evitar confusao com artefatos legados.
    for old in charts_dir.glob("chart_*.png"):
        try:
            old.unlink()
        except Exception:
            pass

    if plt is None:
        return {"available": False, "reason": "matplotlib_unavailable", "charts": {}, "chart_details": {}}

    chart_paths: Dict[str, str] = {}
    chart_details: Dict[str, Dict[str, str]] = {}
    src = lambda *files: " + ".join([f"base_aprendizap/{f}" for f in files])

    panel = monthly_panel.copy().sort_values("month")
    dec = panel[panel["is_decision_month"] == True].copy() if "is_decision_month" in panel.columns else panel.copy()

    if not dec.empty and {"month", "rsva_m1", "retention_m1", "svs_t", "sur_t"}.issubset(set(dec.columns)):
        fig, ax = plt.subplots(figsize=(12.2, 5.4))
        ax.plot(dec["month"], dec["rsva_m1"], color="#0b7285", linewidth=2.3, label="RSVA_m1")
        ax.plot(dec["month"], dec["retention_m1"], color="#1d4ed8", linewidth=1.9, label="Retention_m1")
        ax.plot(dec["month"], dec["svs_t"], color="#f08c00", linewidth=1.7, linestyle="--", label="SVS")
        ax.plot(dec["month"], dec["sur_t"], color="#e03131", linewidth=1.7, linestyle="--", label="SUR")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("North-star e decomposicao")
        ax.set_ylabel("Taxa")
        ax.grid(alpha=0.25)
        _format_month_axis(ax, interval=2)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
        fig.subplots_adjust(right=0.80)
        p1 = charts_dir / "chart_01_north_star_trend.png"
        fig.savefig(p1, dpi=160, bbox_inches="tight")
        plt.close(fig)
        chart_paths["north_star_trend"] = str(p1)
        chart_details["north_star_trend"] = {
            "titulo": "North-star e decomposicao",
            "mede": "Série mensal de RSVA_m1, Retention_m1, SVS_t e SUR_t (meses de decisão).",
            "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
            "transformacoes": "Agregação professor-mês e cálculo das quatro taxas por mês.",
            "como_ler": "Compare nível e direção das quatro linhas ao longo do tempo; diferenças entre SVS_t e SUR_t ajudam a contextualizar variações de RSVA_m1.",
            "path": str(p1),
        }

    if not horizon_comparison_df.empty and {
        "horizon_m",
        "mean_abs_diff",
        "bootstrap_mean_diff_abs_limit",
        "avg_rsva",
        "avg_retention",
    }.issubset(set(horizon_comparison_df.columns)):
        hc = horizon_comparison_df.copy()
        for c in [
            "horizon_m",
            "mean_abs_diff",
            "bootstrap_mean_diff_abs_limit",
            "avg_rsva",
            "avg_retention",
            "avg_svs_t",
            "avg_sur_h",
            "avg_rsva_reconstructed_svs_x_sur",
            "avg_reconstruction_error",
        ]:
            if c in hc.columns:
                hc[c] = pd.to_numeric(hc[c], errors="coerce")
        hc = hc.dropna(subset=["horizon_m"]).sort_values("horizon_m")
        if not hc.empty:
            x = np.arange(len(hc))
            labels = [f"m+{int(v)}" for v in hc["horizon_m"].tolist()]

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(12.8, 8.6),
                sharex=True,
                gridspec_kw={"height_ratios": [1.6, 1.2]},
            )
            ax1.plot(x, hc["avg_rsva"], color="#0b7285", linewidth=2.2, marker="o", markersize=4, label="Média RSVA")
            ax1.plot(x, hc["avg_retention"], color="#1d4ed8", linewidth=2.0, marker="o", markersize=4, label="Média Retention")
            ax1.set_ylim(0.0, 1.0)
            ax1.set_ylabel("Taxa média")
            ax1.set_title("RSVA vs Retention por horizonte")
            ax1.grid(alpha=0.25)
            ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

            w = 0.34
            ax2.bar(
                x - (w / 2),
                hc["mean_abs_diff"],
                width=w,
                color="#c92a2a",
                alpha=0.86,
                label="Diferença média absoluta",
            )
            ax2.bar(
                x + (w / 2),
                hc["bootstrap_mean_diff_abs_limit"],
                width=w,
                color="#495057",
                alpha=0.80,
                label="Limite bootstrap (abs)",
            )
            ax2.set_ylabel("Diferença")
            ax2.set_xlabel("Horizonte")
            ax2.set_title("Magnitude da diferença RSVA - Retention por horizonte")
            ax2.grid(axis="y", alpha=0.25)
            ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=9)

            fig.subplots_adjust(right=0.79, hspace=0.34)
            p8 = charts_dir / "chart_08_horizon_rsva_retention.png"
            fig.savefig(p8, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["horizon_rsva_retention"] = str(p8)
            chart_details["horizon_rsva_retention"] = {
                "titulo": "RSVA vs Retention por horizonte (m+1, m+2, m+4, m+6)",
                "mede": "Compara nível médio de RSVA e Retention e a distância entre elas em cada horizonte.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Cálculo mensal de RSVA/Retention por horizonte; agregação da média e bootstrap da diferença média.",
                "como_ler": "Quanto menor a distância entre as duas linhas e menor a diferença absoluta, mais próximas as métricas ficam naquele horizonte.",
                "path": str(p8),
            }

        if not hc.empty and {
            "avg_svs_t",
            "avg_sur_h",
            "avg_rsva",
            "avg_rsva_reconstructed_svs_x_sur",
        }.issubset(set(hc.columns)):
            x = np.arange(len(hc))
            labels = [f"m+{int(v)}" for v in hc["horizon_m"].tolist()]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8, 8.8), sharex=True)
            w = 0.36
            ax1.bar(x - (w / 2), hc["avg_svs_t"], width=w, color="#1d4ed8", alpha=0.88, label="Média SVS_t")
            ax1.bar(x + (w / 2), hc["avg_sur_h"], width=w, color="#f03e3e", alpha=0.88, label="Média SUR_h")
            ax1.set_ylim(0.0, 1.0)
            ax1.set_ylabel("Taxa média")
            ax1.set_title("Decomposição média por horizonte")
            ax1.grid(axis="y", alpha=0.25)
            ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

            ax2.bar(x - (w / 2), hc["avg_rsva"], width=w, color="#0b7285", alpha=0.88, label="Média RSVA")
            ax2.bar(
                x + (w / 2),
                hc["avg_rsva_reconstructed_svs_x_sur"],
                width=w,
                color="#2b8a3e",
                alpha=0.88,
                label="Média (SVS_t x SUR_h)",
            )
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel("Taxa média")
            ax2.set_xlabel("Horizonte")
            ax2.set_title("RSVA observado vs reconstruído por SVS_t x SUR_h")
            ax2.grid(axis="y", alpha=0.25)
            ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=9)

            fig.subplots_adjust(right=0.79, hspace=0.38)
            p9 = charts_dir / "chart_09_horizon_svs_sur.png"
            fig.savefig(p9, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["horizon_svs_sur"] = str(p9)
            chart_details["horizon_svs_sur"] = {
                "titulo": "SVS e SUR por horizonte (m+1, m+2, m+4, m+6)",
                "mede": "Mostra a média de SVS_t e SUR_h e compara RSVA médio observado com RSVA reconstruído.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Para cada horizonte, calcula médias de SVS_t, SUR_h, RSVA e do produto SVS_t x SUR_h nos meses de decisão.",
                "como_ler": "No painel superior, compare SVS_t e SUR_h por horizonte; no inferior, compare RSVA observado vs SVS_t x SUR_h e o alinhamento entre as barras.",
                "path": str(p9),
            }

    diag_summary_df = pd.DataFrame((diag or {}).get("drop_summary", []))
    if not diag_summary_df.empty and {"diagnostico", "participacao", "meses"}.issubset(set(diag_summary_df.columns)):
        s = diag_summary_df.copy()
        s["participacao"] = pd.to_numeric(s["participacao"], errors="coerce")
        s["meses"] = pd.to_numeric(s["meses"], errors="coerce")
        s = s.dropna(subset=["participacao"]).sort_values("participacao", ascending=True)
        if not s.empty:
            cmap = {
                "queda_por_entrada_em_valor": "#e67700",
                "queda_por_continuidade_pos_valor": "#c92a2a",
                "queda_mista_adocao_e_retencao": "#5f3dc4",
                "queda_combinacao_outros_sinais": "#495057",
                "sem_queda_de_rsva": "#2b8a3e",
            }
            colors = s["diagnostico"].map(cmap).fillna("#868e96")
            fig, ax = plt.subplots(figsize=(11.0, 4.8))
            ax.barh(s["diagnostico"], s["participacao"], color=colors, alpha=0.9)
            for y, (_, r) in enumerate(s.iterrows()):
                ax.text(
                    float(r["participacao"]) + 0.01,
                    y,
                    f"{float(r['participacao']):.1%} ({int(r['meses'])} meses)",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#111827",
                )
            ax.set_xlim(0.0, max(1.0, float(s["participacao"].max()) + 0.12))
            ax.set_xlabel("Meses com queda de retenção")
            ax.set_title("Resumo histórico das quedas de retenção")
            ax.grid(axis="x", alpha=0.25)
            p10 = charts_dir / "chart_10_rsva_drop_summary.png"
            fig.savefig(p10, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["rsva_drop_summary"] = str(p10)
            chart_details["rsva_drop_summary"] = {
                "titulo": "Resumo histórico das quedas de retenção",
                "mede": "Distribuição dos diagnósticos apenas nos meses em que A retenção caiu.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Classificação mensal da queda de RSVA e participação de cada diagnóstico no histórico.",
                "como_ler": "A maior barra é o padrão dominante de queda no histórico.",
                "path": str(p10),
            }

    diag_df = pd.DataFrame((diag or {}).get("latest_12_diagnostics", []))
    # Chart 11 foi removido da apresentação; mantemos apenas os dados tabulares/diagnósticos.
    if False and not diag_df.empty and {"month", "rsva_m1", "svs_t", "sur_t", "d_rsva", "d_svs", "d_sur", "diagnostico"}.issubset(set(diag_df.columns)):
        diag_df = diag_df.copy()
        diag_df["month"] = pd.to_datetime(diag_df["month"], errors="coerce")
        diag_df = diag_df.sort_values("month")
        for c in ["rsva_m1", "svs_t", "sur_t", "d_rsva", "d_svs", "d_sur"]:
            diag_df[c] = pd.to_numeric(diag_df[c], errors="coerce")
        diag_df["sur_prev"] = diag_df["sur_t"] - diag_df["d_sur"]
        diag_df["contrib_svs"] = diag_df["d_svs"] * diag_df["sur_prev"]
        diag_df["contrib_sur"] = diag_df["svs_t"] * diag_df["d_sur"]
        diag_df["delta_recon"] = diag_df["contrib_svs"] + diag_df["contrib_sur"]
        diag_df["month_label"] = diag_df["month"].dt.strftime("%Y-%m")
        x = np.arange(len(diag_df))

        cmap = {
            "queda_por_entrada_em_valor": "#e67700",
            "queda_por_continuidade_pos_valor": "#c92a2a",
            "queda_mista_adocao_e_retencao": "#5f3dc4",
            "queda_combinacao_outros_sinais": "#495057",
            "sem_queda_de_rsva": "#2b8a3e",
        }
        diag_df["diag_color"] = diag_df["diagnostico"].map(cmap).fillna("#495057")

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(13.5, 9.8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.2, 1.4]},
        )
        ax1.plot(x, diag_df["rsva_m1"], color="#0b7285", linewidth=2.2, marker="o", markersize=3, label="RSVA")
        ax1.plot(x, diag_df["svs_t"], color="#1d4ed8", linewidth=1.8, marker="o", markersize=3, label="SVS")
        ax1.plot(x, diag_df["sur_t"], color="#f03e3e", linewidth=1.8, marker="o", markersize=3, label="SUR")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_ylabel("Taxa")
        ax1.set_title("Aplicação mês a mês do diagnóstico RSVA/SVS/SUR")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

        ax2.bar(x, diag_df["d_rsva"], color=diag_df["diag_color"], alpha=0.88)
        ax2.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax2.set_ylabel("Delta RSVA")
        ax2.set_title("Variação mensal do RSVA (cor = diagnóstico)")
        ax2.grid(axis="y", alpha=0.25)

        w = 0.36
        ax3.bar(x - w / 2, diag_df["contrib_svs"], width=w, color="#1d4ed8", alpha=0.85, label="Contribuição SVS")
        ax3.bar(x + w / 2, diag_df["contrib_sur"], width=w, color="#f03e3e", alpha=0.85, label="Contribuição SUR")
        ax3.plot(x, diag_df["d_rsva"], color="#111827", marker="o", linewidth=1.8, label="Delta RSVA observado")
        ax3.plot(x, diag_df["delta_recon"], color="#0f766e", marker="x", linewidth=1.4, linestyle="--", label="Delta reconstruído")
        ax3.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax3.set_ylabel("Contribuição")
        ax3.set_xlabel("Mês")
        ax3.grid(axis="y", alpha=0.25)
        ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
        tick_idx = x if len(x) <= 10 else x[::2]
        ax3.set_xticks(tick_idx)
        ax3.set_xticklabels(diag_df.iloc[tick_idx]["month_label"], rotation=35, ha="right", fontsize=8)
        fig.subplots_adjust(right=0.78, hspace=0.30)

        p11 = charts_dir / "chart_11_rsva_diagnostics_monthly.png"
        fig.savefig(p11, dpi=160, bbox_inches="tight")
        plt.close(fig)
        chart_paths["rsva_diagnostics_monthly"] = str(p11)
        chart_details["rsva_diagnostics_monthly"] = {
            "titulo": "Aplicação mês a mês (últimos 12 meses de decisão)",
            "mede": "Direção da variação mensal de RSVA e decomposição entre SVS e SUR.",
            "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
            "transformacoes": "Diferença mês a mês + contribuição algébrica de SVS e SUR.",
            "como_ler": "Leia os três painéis em conjunto: nível de RSVA/SVS/SUR, ΔRSVA por mês e contribuições de SVS/SUR para a variação observada.",
            "path": str(p11),
        }

    if not segment_monthly_df.empty and {"month", "segment", "rsva_m1", "svs_t", "sur_t"}.issubset(set(segment_monthly_df.columns)):
        seg = segment_monthly_df.copy()
        seg = seg[seg["segment"].isin(["all_users", "base_regular", "heavy_users"])].copy()
        if "is_decision_month" in seg.columns:
            seg = seg[seg["is_decision_month"] == True].copy()
        seg["month"] = pd.to_datetime(seg["month"], errors="coerce")
        seg = seg.sort_values(["month", "segment"])
        if not seg.empty:
            seg_colors = {"all_users": "#0b7285", "base_regular": "#1d4ed8", "heavy_users": "#c92a2a"}
            seg_labels = {"all_users": "todos", "base_regular": "regulares", "heavy_users": "heavy"}
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13.8, 10.8), sharex=True)
            for metric, ax, title in [
                ("rsva_m1", ax1, "RSVA_m1 por segmento"),
                ("sur_t", ax2, "SUR por segmento"),
                ("svs_t", ax3, "SVS por segmento"),
            ]:
                for seg_id in ["all_users", "base_regular", "heavy_users"]:
                    s = seg[seg["segment"] == seg_id].copy()
                    if s.empty:
                        continue
                    ax.plot(
                        s["month"],
                        pd.to_numeric(s[metric], errors="coerce"),
                        linewidth=2.0,
                        marker="o",
                        markersize=3,
                        color=seg_colors.get(seg_id, "#495057"),
                        label=seg_labels.get(seg_id, seg_id),
                    )
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("Taxa")
                ax.set_title(title)
                ax.grid(alpha=0.25)
            ax3.set_xlabel("Mês")
            _format_month_axis(ax3, interval=4)
            ax3.tick_params(axis="x", rotation=35, labelsize=8)
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
            fig.subplots_adjust(top=0.91, hspace=0.44)
            p12 = charts_dir / "chart_12_segment_rsva_svs_sur.png"
            fig.savefig(p12, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["segment_rsva_svs_sur"] = str(p12)
            chart_details["segment_rsva_svs_sur"] = {
                "titulo": "RSVA/SVS/SUR por segmento (todos vs regulares vs heavy)",
                "mede": "Diferença de nível e tendência de RSVA, SUR e SVS por segmento.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Segmentação em all_users/base_regular/heavy_users + cálculo mensal de RSVA, SVS e SUR.",
                "como_ler": "Compare nível, inclinação e distância entre as curvas por segmento em cada painel (RSVA, SUR e SVS).",
                "path": str(p12),
            }

    seg_drop_summary = pd.DataFrame((segment_drop_diag or {}).get("drop_summary_by_segment", []))
    if not seg_drop_summary.empty and {"segment", "diagnostico", "participacao"}.issubset(set(seg_drop_summary.columns)):
        plot_df = seg_drop_summary.copy()
        plot_df["participacao"] = pd.to_numeric(plot_df["participacao"], errors="coerce")
        plot_df["segment"] = plot_df["segment"].astype(str)
        plot_df = plot_df.dropna(subset=["participacao"])
        if not plot_df.empty:
            order = ["all_users", "base_regular", "heavy_users"]
            diag_order = [
                "queda_por_continuidade_pos_valor",
                "queda_por_entrada_em_valor",
                "queda_mista_adocao_e_retencao",
                "queda_combinacao_outros_sinais",
            ]
            pivot = (
                plot_df.pivot_table(index="segment", columns="diagnostico", values="participacao", aggfunc="sum")
                .reindex(order)
                .fillna(0.0)
            )
            cols = [c for c in diag_order if c in pivot.columns] + [c for c in pivot.columns if c not in diag_order]
            pivot = pivot[cols]
            colors = {
                "queda_por_continuidade_pos_valor": "#c92a2a",
                "queda_por_entrada_em_valor": "#e67700",
                "queda_mista_adocao_e_retencao": "#5f3dc4",
                "queda_combinacao_outros_sinais": "#495057",
            }
            fig, ax = plt.subplots(figsize=(11.8, 5.4))
            left = np.zeros(len(pivot))
            y = np.arange(len(pivot))
            for col in pivot.columns:
                vals = pd.to_numeric(pivot[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                ax.barh(y, vals, left=left, color=colors.get(col, "#94a3b8"), alpha=0.9, label=col)
                left = left + vals
            ax.set_xlim(0.0, 1.0)
            ax.set_yticks(y)
            ax.set_yticklabels([str(x) for x in pivot.index])
            ax.set_xlabel("Participacao das quedas de RSVA")
            ax.set_title("Diagnostico de queda por segmento")
            ax.grid(axis="x", alpha=0.25)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
            fig.subplots_adjust(right=0.74)
            p13 = charts_dir / "chart_13_segment_drop_patterns.png"
            fig.savefig(p13, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["segment_drop_patterns"] = str(p13)
            chart_details["segment_drop_patterns"] = {
                "titulo": "Distribuição das quedas de RSVA por segmento",
                "mede": "Qual tipo de queda domina em cada segmento.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Classificação das quedas mensais por diagnóstico e participação em cada segmento.",
                "como_ler": "Cada barra soma 100% das quedas do segmento; a maior faixa indica o padrão dominante.",
                "path": str(p13),
            }

    seg_latest = pd.DataFrame((segment_drop_diag or {}).get("latest_12_diagnostics_by_segment", []))
    if not seg_latest.empty and {"segment", "month", "d_rsva", "diagnostico"}.issubset(set(seg_latest.columns)):
        seg_latest = seg_latest.copy()
        seg_latest["month"] = pd.to_datetime(seg_latest["month"], errors="coerce")
        seg_latest["d_rsva"] = pd.to_numeric(seg_latest["d_rsva"], errors="coerce")
        seg_latest = seg_latest.dropna(subset=["month"]).sort_values(["segment", "month"])
        if not seg_latest.empty:
            seg_order = ["all_users", "base_regular", "heavy_users"]
            seg_labels = {"all_users": "Todos", "base_regular": "Regulares", "heavy_users": "Heavy"}
            cmap = {
                "queda_por_entrada_em_valor": "#e67700",
                "queda_por_continuidade_pos_valor": "#c92a2a",
                "queda_mista_adocao_e_retencao": "#5f3dc4",
                "queda_combinacao_outros_sinais": "#495057",
                "sem_queda_de_rsva": "#2b8a3e",
            }
            present_segments = [s for s in seg_order if s in set(seg_latest["segment"].astype(str))]
            if present_segments:
                fig, axes = plt.subplots(len(present_segments), 1, figsize=(13.4, 9.8), sharex=False)
                if len(present_segments) == 1:
                    axes = [axes]
                for ax, seg in zip(axes, present_segments):
                    s = seg_latest[seg_latest["segment"] == seg].copy().tail(12).reset_index(drop=True)
                    x = np.arange(len(s))
                    colors = s["diagnostico"].map(cmap).fillna("#868e96")
                    ax.bar(x, s["d_rsva"], color=colors, alpha=0.88)
                    ax.plot(x, s["d_rsva"], color="#111827", linewidth=1.2, marker="o", markersize=3)
                    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
                    ax.set_ylabel("ΔRSVA")
                    ax.set_title(f"{seg_labels.get(seg, seg)} | variação mensal de RSVA")
                    ax.grid(axis="y", alpha=0.25)
                    step = 1 if len(s) <= 8 else 2
                    tick_idx = x[::step]
                    ax.set_xticks(tick_idx)
                    ax.set_xticklabels(s.iloc[tick_idx]["month"].dt.strftime("%Y-%m"), rotation=35, ha="right", fontsize=8)
                axes[-1].set_xlabel("Mês")
                fig.subplots_adjust(hspace=0.56)
                p14 = charts_dir / "chart_14_segment_monthly_diagnostics.png"
                fig.savefig(p14, dpi=160, bbox_inches="tight")
                plt.close(fig)
                chart_paths["segment_monthly_diagnostics"] = str(p14)
                chart_details["segment_monthly_diagnostics"] = {
                "titulo": "Aplicação mês a mês do diagnóstico por segmento",
                "mede": "Variação mensal de RSVA nos últimos 12 meses por segmento, com cor do diagnóstico.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Cálculo mensal de ΔRSVA e classificação por diagnóstico para all_users, regulares e heavy.",
                "como_ler": "Barras abaixo de zero são quedas; a cor indica se a queda veio de SVS, SUR ou ambos.",
                    "path": str(p14),
                }

    if not event_family_df.empty and {
        "month",
        "aula_user_share",
        "prova_user_share",
        "download_aula_user_share",
        "download_plano_user_share",
        "vis_aula_sem_download_share_viewers",
        "vis_prova_sem_acao_share_viewers",
        "vis_aula_com_download_share_viewers",
        "vis_prova_com_acao_share_viewers",
    }.issubset(set(event_family_df.columns)):
        ef = event_family_df.copy()
        ef["month"] = pd.to_datetime(ef["month"], errors="coerce")
        if "is_decision_month" in ef.columns:
            ef = ef[ef["is_decision_month"] == True].copy()
        ef = ef.sort_values("month")
        if not ef.empty:
            fig, ax = plt.subplots(figsize=(13.2, 6.4))
            ax.plot(ef["month"], ef["aula_user_share"], color="#1d4ed8", linewidth=2.2, label="Share usuários Aula")
            ax.plot(ef["month"], ef["prova_user_share"], color="#d6336c", linewidth=2.2, label="Share usuários Prova")
            ax.plot(ef["month"], ef["download_aula_user_share"], color="#0b7285", linewidth=2.0, linestyle="--", label="Share download_aula")
            ax.plot(ef["month"], ef["download_plano_user_share"], color="#2b8a3e", linewidth=2.0, linestyle="--", label="Share download_plano_aula")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Share sobre usuários ativos")
            ax.set_xlabel("Mês")
            ax.set_title("Aula vs Prova e conversão em download")
            ax.grid(alpha=0.25)
            _format_month_axis(ax, interval=2)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
            fig.subplots_adjust(right=0.79)
            p15 = charts_dir / "chart_15_event_family_shares.png"
            fig.savefig(p15, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["event_family_shares"] = str(p15)
            chart_details["event_family_shares"] = {
                "titulo": "Shares mensais de Aula/Prova e downloads",
                "mede": "Participação mensal de Aula e Prova entre ativos e conversão em download.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Flags por professor-mês para família de evento e shares mensais.",
                "como_ler": "Compare as linhas de uso (Aula/Prova) com as linhas de download para ver a distância entre atividade e download no mesmo mês.",
                "path": str(p15),
            }

            fig, ax = plt.subplots(figsize=(13.2, 5.4))
            ax.plot(
                ef["month"],
                ef["vis_aula_sem_download_share_viewers"],
                color="#e67700",
                linewidth=2.2,
                marker="o",
                markersize=3,
                label="Aula: visualizou sem baixar",
            )
            ax.plot(
                ef["month"],
                ef["vis_prova_sem_acao_share_viewers"],
                color="#495057",
                linewidth=2.2,
                marker="o",
                markersize=3,
                label="Prova: visualizou sem ação",
            )
            ax.plot(
                ef["month"],
                ef["vis_aula_com_download_share_viewers"],
                color="#2b8a3e",
                linewidth=2.0,
                marker="o",
                markersize=3,
                linestyle="--",
                label="Aula: visualizou e baixou",
            )
            ax.plot(
                ef["month"],
                ef["vis_prova_com_acao_share_viewers"],
                color="#1d4ed8",
                linewidth=2.0,
                marker="o",
                markersize=3,
                linestyle="--",
                label="Prova: visualizou e fez ação",
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Taxa entre visualizadores")
            ax.set_xlabel("Mês")
            ax.set_title("Fricção após visualização (Aula vs Prova)")
            ax.grid(alpha=0.25)
            _format_month_axis(ax, interval=2)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=8)
            fig.subplots_adjust(top=0.86)
            p16 = charts_dir / "chart_16_event_family_friction.png"
            fig.savefig(p16, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["event_family_segmentation"] = str(p16)
            chart_details["event_family_segmentation"] = {
                "titulo": "Fricção de visualização sem ação",
                "mede": "Entre visualizadores, comparação entre não conversão e conversão para Aula e Prova.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Cálculo mensal das taxas visualização->não ação e visualização->ação (complemento exato) por família.",
                "como_ler": "Sem ação alto e com ação baixo indicam atrito; inverso indica boa conversão após visualização.",
                "path": str(p16),
            }

    if not subject_quality_df.empty:
        sq = subject_quality_df.copy()
        sq["month"] = pd.to_datetime(sq["month"], errors="coerce")
        if "is_decision_month" in sq.columns:
            sq = sq[sq["is_decision_month"] == True].copy()
        sq = sq.sort_values("month")
        if not sq.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 8.3), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.7]})
            ax1.plot(
                sq["month"],
                pd.to_numeric(sq["pct_with_id_aula"], errors="coerce"),
                color="#0b7285",
                linewidth=2.0,
                marker="o",
                markersize=3,
                label="downloads com id_aula",
            )
            ax1.plot(
                sq["month"],
                pd.to_numeric(sq["pct_with_disciplina"], errors="coerce"),
                color="#c92a2a",
                linewidth=2.0,
                marker="o",
                markersize=3,
                label="downloads mapeados para disciplina",
            )
            ax1.set_ylim(0.0, 1.0)
            ax1.set_ylabel("Cobertura")
            ax1.set_title("Cobertura do mapeamento de disciplina (downloads de Aula/Plano)")
            ax1.grid(alpha=0.25)
            ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

            tm = subject_top_monthly_df.copy() if not subject_top_monthly_df.empty else pd.DataFrame()
            if not tm.empty and {"month", "disciplina", "share_of_mapped_downloads"}.issubset(set(tm.columns)):
                tm["month"] = pd.to_datetime(tm["month"], errors="coerce")
                if "is_decision_month" in tm.columns:
                    tm = tm[tm["is_decision_month"] == True].copy()
                tm = tm.sort_values("month")
                if not tm.empty:
                    pivot = tm.pivot_table(
                        index="month",
                        columns="disciplina",
                        values="share_of_mapped_downloads",
                        aggfunc="sum",
                    ).sort_index()
                    pivot = pivot.reindex(sorted(sq["month"].dropna().unique())).fillna(0.0)
                    col_order = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
                    pivot = pivot[col_order]
                    palette = ["#1d4ed8", "#7c3aed", "#f59e0b", "#16a34a", "#dc2626", "#0f766e", "#334155"]
                    for i, col in enumerate(pivot.columns):
                        ax2.plot(
                            pivot.index,
                            pd.to_numeric(pivot[col], errors="coerce"),
                            linewidth=2.0,
                            marker="o",
                            markersize=3,
                            color=palette[i % len(palette)],
                            label=str(col),
                        )
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel("Share entre downloads com disciplina")
            ax2.set_xlabel("Mês")
            ax2.set_title("Disciplinas com maior participação nos downloads (top)")
            ax2.grid(alpha=0.25)
            _format_month_axis(ax2, interval=3)
            ax2.tick_params(axis="x", rotation=35, labelsize=8)
            ax2.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=False,
                fontsize=8,
                title="Disciplinas (ordem por média)",
                title_fontsize=8,
            )
            fig.subplots_adjust(right=0.76, hspace=0.40)
            p17 = charts_dir / "chart_17_subject_downloads.png"
            fig.savefig(p17, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["subject_downloads"] = str(p17)
            chart_details["subject_downloads"] = {
                "titulo": "Downloads por disciplina (Aula/Plano)",
                "mede": "Cobertura de mapeamento e evolução das disciplinas mais baixadas.",
                "tabelas_usadas": src("fct_teachers_contents_interactions.csv", "stg_lessons.csv", "dim_teachers.csv"),
                "transformacoes": "Join por id_aula + agregação mensal do share de disciplinas nos downloads.",
                "como_ler": "Painel superior: cobertura de mapeamento; painel inferior: participação mensal das disciplinas entre os downloads mapeados.",
                "path": str(p17),
            }

    if not rsva_linear_fit_df.empty and {"segment", "rsva_m1_observed", "rsva_m1_pred_additive"}.issubset(set(rsva_linear_fit_df.columns)):
        fit = rsva_linear_fit_df.copy()
        fit["rsva_m1_observed"] = pd.to_numeric(fit["rsva_m1_observed"], errors="coerce")
        fit["rsva_m1_pred_additive"] = pd.to_numeric(fit["rsva_m1_pred_additive"], errors="coerce")
        fit = fit.dropna(subset=["rsva_m1_observed", "rsva_m1_pred_additive"])
        if not fit.empty:
            seg_r2 = {}
            if not rsva_linear_models_df.empty:
                for _, r in rsva_linear_models_df.iterrows():
                    seg_r2[str(r.get("segment"))] = r.get("additive_r2")

            fig, ax = plt.subplots(figsize=(8.0, 6.2))
            seg_colors = {"all_users": "#0b7285", "base_regular": "#1d4ed8", "heavy_users": "#c92a2a"}
            for seg in sorted(fit["segment"].dropna().astype(str).unique().tolist()):
                s = fit[fit["segment"] == seg]
                label = seg
                r2v = seg_r2.get(seg)
                if pd.notna(r2v):
                    label = f"{seg} (R2={float(r2v):.3f})"
                ax.scatter(
                    s["rsva_m1_pred_additive"],
                    s["rsva_m1_observed"],
                    s=36,
                    alpha=0.85,
                    label=label,
                    color=seg_colors.get(seg, "#495057"),
                )
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("RSVA previsto (regressão linear aditiva)")
            ax.set_ylabel("RSVA observado")
            ax.set_title("RSVA/SVS/SUR: ajuste linear por segmento (regular vs heavy)")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
            fig.subplots_adjust(right=0.76)
            p18 = charts_dir / "chart_18_rsva_linear_segments.png"
            fig.savefig(p18, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["rsva_linear_segments"] = str(p18)
            chart_details["rsva_linear_segments"] = {
                "titulo": "Regressão linear RSVA ~ SVS + SUR (segmentos)",
                "mede": "Quao bem SVS e SUR explicam RSVA para todos, regulares e heavy.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Agregação mensal por segmento + regressão linear aditiva.",
                "como_ler": "Quanto mais próximos da diagonal (previsto = observado), melhor o ajuste descritivo no segmento; o gráfico não representa causalidade.",
                "path": str(p18),
            }

    if not event_impacts_df.empty and {"segment", "metric", "event_type", "effect", "ci_low", "ci_high"}.issubset(set(event_impacts_df.columns)):
        impacts = event_impacts_df.copy()
        impacts = impacts[(impacts["segment"] == "all_users") & (impacts["metric"] == "rsva_m1")].copy()
        impacts["effect"] = pd.to_numeric(impacts["effect"], errors="coerce")
        impacts["ci_low"] = pd.to_numeric(impacts["ci_low"], errors="coerce")
        impacts["ci_high"] = pd.to_numeric(impacts["ci_high"], errors="coerce")
        impacts = impacts.dropna(subset=["effect", "ci_low", "ci_high"])
        if not impacts.empty:
            plot_df = impacts.sort_values("effect", ascending=True).head(12).copy()
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(11.6, 6.4))
                has_neg = bool((pd.to_numeric(plot_df["ci_high"], errors="coerce") < 0).any())
                colors = np.where(
                    pd.to_numeric(plot_df["ci_high"], errors="coerce") < 0,
                    "#c92a2a",
                    np.where(pd.to_numeric(plot_df["ci_low"], errors="coerce") > 0, "#2b8a3e", "#868e96"),
                )
                center = plot_df["effect"].to_numpy(dtype=float)
                lo = plot_df["ci_low"].to_numpy(dtype=float)
                hi = plot_df["ci_high"].to_numpy(dtype=float)
                ax.barh(plot_df["event_type"], center, color=colors, alpha=0.86)
                ax.errorbar(center, plot_df["event_type"], xerr=[center - lo, hi - center], fmt="none", ecolor="black", elinewidth=1, capsize=2)
                ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
                if has_neg:
                    ax.set_title("Eventos com maior impacto negativo em RSVA (all_users)")
                else:
                    ax.set_title("Menores efeitos em RSVA (não há efeito negativo confirmado)")
                ax.set_xlabel("Efeito no RSVA (expostos - não expostos)")
                ax.grid(axis="x", alpha=0.25)
                p19 = charts_dir / "chart_19_negative_event_impacts_rsva.png"
                fig.savefig(p19, dpi=160, bbox_inches="tight")
                plt.close(fig)
                chart_paths["negative_event_impacts_rsva"] = str(p19)
                chart_details["negative_event_impacts_rsva"] = {
                    "titulo": "Eventos associados a variação de RSVA",
                    "mede": "Diferença de RSVA entre expostos e não expostos por event_type.",
                    "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv", "calendario_escolar_uf_rede.csv"),
                    "transformacoes": "Comparação exposto vs não exposto com estratificação por mês, UF e calendário (jornada escolar), com FDR.",
                    "como_ler": (
                        "Vermelho indica piora de RSVA (efeito < 0); verde indica melhora (efeito > 0). "
                        + ("Neste recorte há sinais negativos." if has_neg else "Neste recorte não houve efeito negativo confirmado em RSVA para all_users.")
                    ),
                    "path": str(p19),
                }

    if not event_impacts_df.empty and {"segment", "metric", "event_type", "effect", "negative_confirmed", "positive_confirmed"}.issubset(set(event_impacts_df.columns)):
        focus_names = [
            "prova_salva",
            "visualizacao_prova",
            "visualizacao_prova_aprendizap",
            "botao_baixar_conquista_completada",
            "fechar_conquista_obtida",
            "click_subaba_concluidas",
        ]
        f = event_impacts_df.copy()
        f = f[
            (f["segment"] == "all_users")
            & (f["metric"].isin(["rsva_m1", "svs_t", "sur_t"]))
            & (f["event_type"].astype(str).isin(focus_names))
        ].copy()
        f["effect"] = pd.to_numeric(f["effect"], errors="coerce")
        f = f.dropna(subset=["effect"])
        if not f.empty:
            metric_order = ["rsva_m1", "svs_t", "sur_t"]
            present_events = [e for e in focus_names if e in set(f["event_type"].astype(str))]
            pivot = (
                f.pivot_table(index="event_type", columns="metric", values="effect", aggfunc="mean")
                .reindex(index=present_events, columns=metric_order)
                .fillna(0.0)
            )
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(12.4, 4.9))
                mat = pivot.to_numpy(dtype=float)
                vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
                vmax = max(vmax, 0.03)
                if LinearSegmentedColormap is not None:
                    cmap = LinearSegmentedColormap.from_list("piora_melhora", ["#c92a2a", "#f8f9fa", "#2b8a3e"])
                else:
                    cmap = "RdYlGn"
                im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels(["RSVA", "SVS", "SUR"], fontsize=9)
                ax.set_title("Eventos foco: efeito em RSVA/SVS/SUR (all_users)")
                for i, ev_name in enumerate(pivot.index):
                    for j, metric in enumerate(pivot.columns):
                        val = float(pivot.iloc[i, j])
                        cell = f[(f["event_type"] == ev_name) & (f["metric"] == metric)]
                        marker = ""
                        if not cell.empty:
                            neg_ok = bool(cell["negative_confirmed"].fillna(False).iloc[0])
                            pos_ok = bool(cell["positive_confirmed"].fillna(False).iloc[0])
                            marker = "*" if (neg_ok or pos_ok) else ""
                        ax.text(j, i, f"{val:+.3f}{marker}", ha="center", va="center", fontsize=8, color="#111827")
                cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                cbar.set_label("Efeito (expostos - não expostos)")
                p23 = charts_dir / "chart_23_focus_event_impacts.png"
                fig.savefig(p23, dpi=160, bbox_inches="tight")
                plt.close(fig)
                chart_paths["focus_event_impacts"] = str(p23)
                chart_details["focus_event_impacts"] = {
                    "titulo": "Eventos foco e direção do efeito",
                    "mede": "Sinal de efeito dos eventos foco em RSVA, SVS e SUR para all_users.",
                    "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv", "calendario_escolar_uf_rede.csv"),
                    "transformacoes": "Comparação exposto vs não exposto com estratificação por mês, UF e calendário (jornada escolar).",
                    "como_ler": "Verde = melhora da métrica; vermelho = piora. '*' marca efeito confirmado (IC + FDR + amostra).",
                    "path": str(p23),
                }

    if not event_class_impacts_df.empty and {"segment", "metric", "event_class", "effect", "ci_low", "ci_high"}.issubset(set(event_class_impacts_df.columns)):
        cls = event_class_impacts_df.copy()
        cls = cls[(cls["segment"] == "all_users") & (cls["metric"].isin(["rsva_m1", "svs_t", "sur_t"]))].copy()
        cls["effect"] = pd.to_numeric(cls["effect"], errors="coerce")
        cls["ci_low"] = pd.to_numeric(cls["ci_low"], errors="coerce")
        cls["ci_high"] = pd.to_numeric(cls["ci_high"], errors="coerce")
        cls = cls.dropna(subset=["effect", "ci_low", "ci_high"])
        if not cls.empty:
            metric_order = [("rsva_m1", "RSVA"), ("svs_t", "SVS"), ("sur_t", "SUR")]
            fig, axes = plt.subplots(3, 1, figsize=(12.4, 10.5), sharex=False)
            for ax, (metric_key, metric_label) in zip(axes, metric_order):
                m = cls[cls["metric"] == metric_key].copy().sort_values("effect")
                if m.empty:
                    ax.axis("off")
                    continue
                neg = m.head(5)
                pos = m.tail(3)
                plot_df = pd.concat([neg, pos], ignore_index=True).drop_duplicates("event_class")
                plot_df = plot_df.sort_values("effect")
                colors = np.where(
                    plot_df["ci_high"] < 0,
                    "#c92a2a",
                    np.where(plot_df["ci_low"] > 0, "#2b8a3e", "#868e96"),
                )
                center = plot_df["effect"].to_numpy(dtype=float)
                lo = plot_df["ci_low"].to_numpy(dtype=float)
                hi = plot_df["ci_high"].to_numpy(dtype=float)
                ax.barh(plot_df["event_class"], center, color=colors, alpha=0.88)
                ax.errorbar(center, plot_df["event_class"], xerr=[center - lo, hi - center], fmt="none", ecolor="black", elinewidth=1, capsize=2)
                ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
                ax.set_title(f"Classes de evento com efeito em {metric_label}")
                ax.set_xlabel("Efeito (expostos - não expostos)")
                ax.grid(axis="x", alpha=0.25)
            fig.subplots_adjust(hspace=0.35)
            p20 = charts_dir / "chart_20_event_class_impacts.png"
            fig.savefig(p20, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["event_class_impacts"] = str(p20)
            chart_details["event_class_impacts"] = {
                "titulo": "Classes de eventos com efeito em RSVA/SVS/SUR",
                "mede": "Efeito estimado por classe de evento em RSVA, SVS e SUR (all_users).",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv", "calendario_escolar_uf_rede.csv"),
                "transformacoes": "Agrupamento de eventos em classes + comparação exposto vs não exposto com estratificação por mês, UF e calendário.",
                "como_ler": "Verde indica melhora da métrica; vermelho indica piora. Cinza indica efeito inconclusivo.",
                "path": str(p20),
            }

    if not event_class_impacts_df.empty and {"segment", "metric", "event_class", "effect"}.issubset(set(event_class_impacts_df.columns)):
        cls_all = event_class_impacts_df.copy()
        cls_all = cls_all[cls_all["metric"].isin(["rsva_m1", "svs_t", "sur_t"])].copy()
        cls_all["effect"] = pd.to_numeric(cls_all["effect"], errors="coerce")
        cls_all["event_class"] = cls_all["event_class"].astype(str)
        cls_all["segment"] = cls_all["segment"].astype(str)
        cls_all["metric"] = cls_all["metric"].astype(str)
        cls_all = cls_all.dropna(subset=["effect"])
        if not cls_all.empty:
            cls_all["seg_metric"] = cls_all["segment"] + " | " + cls_all["metric"]
            pivot = cls_all.pivot_table(index="event_class", columns="seg_metric", values="effect", aggfunc="mean")
            pivot = pivot.sort_index()
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(13.2, 6.8))
                mat = pivot.to_numpy(dtype=float)
                vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
                vmax = max(vmax, 0.05)
                if LinearSegmentedColormap is not None:
                    cmap = LinearSegmentedColormap.from_list("piora_melhora", ["#c92a2a", "#f8f9fa", "#2b8a3e"])
                else:
                    cmap = "RdYlGn"
                im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels([str(c) for c in pivot.columns], rotation=35, ha="right", fontsize=8)
                ax.set_title("Mapa de efeito por classe de evento (segmento x métrica)")
                cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                cbar.set_label("Efeito (expostos - não expostos)")
                p22 = charts_dir / "chart_22_event_class_effect_heatmap.png"
                fig.savefig(p22, dpi=160, bbox_inches="tight")
                plt.close(fig)
                chart_paths["event_class_effect_heatmap"] = str(p22)
                chart_details["event_class_effect_heatmap"] = {
                    "titulo": "Efeito por classe de evento (segmento x métrica)",
                    "mede": "Sinal de efeito para cada classe de evento em RSVA, SVS e SUR por segmento.",
                    "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv", "calendario_escolar_uf_rede.csv"),
                    "transformacoes": "Agrupamento por classe, segmentação mensal e estimativa de efeito com estratificação por mês, UF e calendário.",
                    "como_ler": "Verde indica efeito positivo (melhora da métrica) e vermelho indica efeito negativo (piora).",
                    "path": str(p22),
                }

    if not metric_uncertainty_df.empty and {"month", "metric", "value", "ci_low", "ci_high"}.issubset(set(metric_uncertainty_df.columns)):
        ud = metric_uncertainty_df.copy()
        ud["month"] = pd.to_datetime(ud["month"], errors="coerce")
        ud["value"] = pd.to_numeric(ud["value"], errors="coerce")
        ud["ci_low"] = pd.to_numeric(ud["ci_low"], errors="coerce")
        ud["ci_high"] = pd.to_numeric(ud["ci_high"], errors="coerce")
        if "is_decision_month" in ud.columns:
            ud = ud[ud["is_decision_month"] == True].copy()
        ud = ud.dropna(subset=["month", "metric", "value", "ci_low", "ci_high"])
        metric_order = [("rsva_m1", "RSVA_m1"), ("svs_t", "SVS_t"), ("sur_t", "SUR_t")]
        present = [(k, lbl) for k, lbl in metric_order if k in set(ud["metric"].astype(str))]
        if present:
            fig, axes = plt.subplots(len(present), 1, figsize=(13.2, 3.2 * len(present)), sharex=True)
            if len(present) == 1:
                axes = [axes]
            colors = {"rsva_m1": "#0b7285", "svs_t": "#1d4ed8", "sur_t": "#c92a2a"}
            for ax, (m, lbl) in zip(axes, present):
                s = ud[ud["metric"] == m].sort_values("month").copy()
                if s.empty:
                    ax.axis("off")
                    continue
                ax.plot(s["month"], s["value"], color=colors.get(m, "#334155"), linewidth=2.1, marker="o", markersize=3, label=lbl)
                ax.fill_between(
                    s["month"],
                    s["ci_low"],
                    s["ci_high"],
                    color=colors.get(m, "#334155"),
                    alpha=0.18,
                    label="IC95% (Wilson)",
                )
                ax.set_ylim(0.0, 1.0)
                ax.grid(alpha=0.25)
                ax.set_ylabel("Taxa")
                ax.set_title(f"{lbl} com banda de incerteza (IC95%)")
                ax.legend(loc="upper left", frameon=False, fontsize=8)
            axes[-1].set_xlabel("Mês")
            _format_month_axis(axes[-1], interval=3)
            axes[-1].tick_params(axis="x", rotation=35, labelsize=8)
            fig.subplots_adjust(hspace=0.42)
            p25 = charts_dir / "chart_25_metric_uncertainty_bands.png"
            fig.savefig(p25, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["metric_uncertainty_bands"] = str(p25)
            chart_details["metric_uncertainty_bands"] = {
                "titulo": "Bandas de incerteza das métricas (IC95%)",
                "mede": "Evolução mensal das métricas-chave com intervalos de confiança Wilson.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Agregação mês a mês dos numeradores/denominadores e cálculo de IC95% por proporção.",
                "como_ler": "Faixas mais largas indicam maior incerteza estatística (geralmente em meses com menor denominador).",
                "path": str(p25),
            }

    if not strict_cohort_hazard_df.empty and {"horizon_m", "hazard", "survival", "cumulative_return", "n_at_risk"}.issubset(set(strict_cohort_hazard_df.columns)):
        ch = strict_cohort_hazard_df.copy()
        ch["horizon_m"] = pd.to_numeric(ch["horizon_m"], errors="coerce")
        ch["hazard"] = pd.to_numeric(ch["hazard"], errors="coerce")
        ch["survival"] = pd.to_numeric(ch["survival"], errors="coerce")
        ch["cumulative_return"] = pd.to_numeric(ch["cumulative_return"], errors="coerce")
        ch["n_at_risk"] = pd.to_numeric(ch["n_at_risk"], errors="coerce")
        for c in ["hazard_ci_low", "hazard_ci_high", "cumulative_return_ci_low", "cumulative_return_ci_high"]:
            if c in ch.columns:
                ch[c] = pd.to_numeric(ch[c], errors="coerce")
        ch = ch.dropna(subset=["horizon_m"]).sort_values("horizon_m")
        if not ch.empty:
            x = ch["horizon_m"].astype(int).to_numpy()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0]})
            ax1.plot(x, ch["survival"], color="#1d4ed8", linewidth=2.2, marker="o", markersize=4, label="Survival (sem retorno até h)")
            ax1.plot(x, ch["cumulative_return"], color="#0b7285", linewidth=2.2, marker="o", markersize=4, label="Retorno acumulado até h")
            if {"cumulative_return_ci_low", "cumulative_return_ci_high"}.issubset(set(ch.columns)):
                ax1.fill_between(x, ch["cumulative_return_ci_low"], ch["cumulative_return_ci_high"], color="#0b7285", alpha=0.16)
            ax1.set_ylim(0.0, 1.0)
            ax1.set_ylabel("Probabilidade")
            ax1.set_title("Curva de survival e retorno acumulado (cohort strict)")
            ax1.grid(alpha=0.25)
            ax1.legend(loc="upper right", frameon=False, fontsize=8)

            ax2.bar(x, ch["hazard"], color="#c92a2a", alpha=0.86, label="Hazard de primeiro retorno em h")
            if {"hazard_ci_low", "hazard_ci_high"}.issubset(set(ch.columns)):
                center = ch["hazard"].to_numpy(dtype=float)
                lo = ch["hazard_ci_low"].to_numpy(dtype=float)
                hi = ch["hazard_ci_high"].to_numpy(dtype=float)
                ax2.errorbar(x, center, yerr=[center - lo, hi - center], fmt="none", ecolor="#111827", elinewidth=1, capsize=2)
            ax2.plot(x, (ch["n_at_risk"] / ch["n_at_risk"].max()).fillna(0.0), color="#495057", linewidth=1.8, marker="s", markersize=3, label="At-risk (normalizado)")
            ax2.set_ylim(0.0, 1.0)
            ax2.set_xlabel("Horizonte (meses desde o strict value)")
            ax2.set_ylabel("Hazard")
            ax2.set_title("Hazard de primeiro retorno por horizonte")
            ax2.grid(axis="y", alpha=0.25)
            ax2.legend(loc="upper right", frameon=False, fontsize=8)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"h={i}" for i in x], fontsize=9)
            fig.subplots_adjust(hspace=0.34)
            p26 = charts_dir / "chart_26_strict_cohort_survival_hazard.png"
            fig.savefig(p26, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["strict_cohort_survival_hazard"] = str(p26)
            chart_details["strict_cohort_survival_hazard"] = {
                "titulo": "Cohort survival/hazard após strict value",
                "mede": "Probabilidade de ainda não ter retornado, retorno acumulado e hazard de primeiro retorno por horizonte.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Cohort por primeiro mês de strict value por usuário + cálculo de first-return-time em meses.",
                "como_ler": "Hazard alto em h indica concentração de primeiro retorno nesse horizonte; survival cai conforme os retornos se acumulam.",
                "path": str(p26),
            }

    if not taxonomy_df.empty and {"event_type", "adjusted_uplift_next_active", "ci_low", "ci_high", "aux_uplift_positive_flag"}.issubset(set(taxonomy_df.columns)):
        top_pos = taxonomy_df.sort_values("adjusted_uplift_next_active", ascending=False).head(8)
        top_neg = taxonomy_df.sort_values("adjusted_uplift_next_active", ascending=True).head(4)
        plot_df = pd.concat([top_pos, top_neg], ignore_index=True).drop_duplicates("event_type")
        plot_df = plot_df.sort_values("adjusted_uplift_next_active", ascending=True)
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(11.2, 6.0))
            colors = np.where(plot_df["adjusted_uplift_next_active"] >= 0, "#2b8a3e", "#c92a2a")
            center = pd.to_numeric(plot_df["adjusted_uplift_next_active"], errors="coerce").to_numpy(dtype=float)
            lo = pd.to_numeric(plot_df["ci_low"], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(plot_df["ci_high"], errors="coerce").to_numpy(dtype=float)
            ax.barh(plot_df["event_type"], center, color=colors, alpha=0.86)
            ax.errorbar(center, plot_df["event_type"], xerr=[center - lo, hi - center], fmt="none", ecolor="black", elinewidth=1, capsize=2)
            ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
            ax.set_title("Eventos com maior/menor sinal de uplift (análise auxiliar)")
            ax.set_xlabel("Uplift em next_active")
            ax.grid(axis="x", alpha=0.25)
            p5 = charts_dir / "chart_05_taxonomy_uplift.png"
            fig.savefig(p5, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["taxonomy_uplift"] = str(p5)
            chart_details["taxonomy_uplift"] = {
                "titulo": "Eventos com maior/menor sinal de uplift (análise auxiliar)",
                "mede": "Uplift estimado em next_active para o recorte de eventos com maiores e menores valores no snapshot.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv"),
                "transformacoes": "Compara taxa de retorno entre expostos e não expostos por event_type.",
                "como_ler": "Barras positivas indicam associação com maior retorno e barras negativas com menor retorno; o gráfico é auxiliar e não define strict value.",
                "path": str(p5),
            }

    if (
        not heavy_user_types_df.empty
        and {
            "dimension",
            "category",
            "lift_vs_overall",
            "users_total",
            "reliable_heavy_type",
        }.issubset(set(heavy_user_types_df.columns))
    ):
        ht = heavy_user_types_df.copy()
        ht = ht[ht["reliable_heavy_type"] == True].copy()
        ht["dimension"] = ht["dimension"].astype(str)
        ht["category"] = ht["category"].astype(str)
        ht["lift_vs_overall"] = pd.to_numeric(ht["lift_vs_overall"], errors="coerce")
        ht["users_total"] = pd.to_numeric(ht["users_total"], errors="coerce")
        ht = ht.dropna(subset=["dimension", "category", "lift_vs_overall", "users_total"])
        if not ht.empty:
            dim_order_preferred = [
                "device",
                "location_estado",
                "subject_top_discipline",
                "usage_dominant_weekday",
                "usage_dominant_hour_bin",
            ]
            dims_present = list(ht["dimension"].drop_duplicates())
            dims = [d for d in dim_order_preferred if d in dims_present] + [d for d in dims_present if d not in dim_order_preferred]
            n_dims = len(dims)
            n_cols = 1 if n_dims <= 2 else 2
            n_rows = int(np.ceil(n_dims / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.8, 4.5 * n_rows))
            axes_arr = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])
            dim_label_map = {
                "device": "Device",
                "location_estado": "Localização (UF)",
                "subject_top_discipline": "Matéria (disciplina dominante)",
                "usage_dominant_weekday": "Dia dominante de uso",
                "usage_dominant_hour_bin": "Faixa horária dominante",
            }
            for i, dim in enumerate(dims):
                ax = axes_arr[i]
                d = ht[ht["dimension"] == dim].copy()
                d = d.sort_values(["lift_vs_overall", "users_total"], ascending=[False, False]).head(8)
                d = d.sort_values("lift_vs_overall", ascending=True)
                ax.barh(d["category"], d["lift_vs_overall"], color="#0b7285", alpha=0.88)
                ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
                for y, (_, row) in enumerate(d.iterrows()):
                    ax.text(
                        float(row["lift_vs_overall"]) + 0.02,
                        y,
                        f"n={int(row['users_total'])}",
                        va="center",
                        ha="left",
                        fontsize=8,
                        color="#111827",
                    )
                max_lift = float(pd.to_numeric(d["lift_vs_overall"], errors="coerce").max()) if not d.empty else 1.0
                ax.set_xlim(0.0, max(1.35, max_lift + 0.35))
                ax.set_title(dim_label_map.get(dim, dim))
                ax.set_xlabel("Lift vs taxa heavy da base da dimensão")
                ax.grid(axis="x", alpha=0.25)
                ax.tick_params(axis="y", labelsize=8)
            for j in range(n_dims, len(axes_arr)):
                axes_arr[j].axis("off")
            fig.suptitle("Tipologias de heavy users por dimensão (categorias confiáveis)", y=0.995)
            fig.subplots_adjust(hspace=0.42, wspace=0.36, top=0.94)
            p24 = charts_dir / "chart_24_heavy_user_types_lift.png"
            fig.savefig(p24, dpi=160, bbox_inches="tight")
            plt.close(fig)
            chart_paths["heavy_user_types_lift"] = str(p24)
            chart_details["heavy_user_types_lift"] = {
                "titulo": "Tipologias de heavy users por dimensão",
                "mede": "Categorias confiáveis com maior concentração relativa de heavy users (lift) por dimensão.",
                "tabelas_usadas": src("dim_teachers.csv", "fct_teachers_contents_interactions.csv", "stg_lessons.csv"),
                "transformacoes": "Heavy herdado da etapa 01 (heavy_score_fast_v1); teste de proporções vs restante da base por dimensão; filtros mínimos de amostra.",
                "como_ler": "Barras acima de 1.0 indicam enriquecimento de heavy na categoria; linha tracejada marca lift=1.",
                "path": str(p24),
            }

    chart_details = apply_metric_terminology_payload(chart_details)
    return {"available": True, "charts": chart_paths, "chart_details": chart_details}


def cleanup_legacy_artifacts(output_dir: Path) -> None:
    legacy_files = [
        "kpi_metric_selection_legacy_details.json",
        "kpi_teacher_risk_frame.csv",
        "kpi_teacher_risk_backtest.csv",
        "kpi_teacher_risk_oof_predictions.csv",
        "kpi_teacher_risk_scores_latest.csv",
        "kpi_teacher_risk_coefficients.csv",
        "kpi_teacher_risk_summary.json",
        "kpi_teacher_risk_review.json",
        "kpi_teacher_risk_scenario_comparison.csv",
        "kpi_teacher_risk_scenario_comparison.json",
        "kpi_heavy_lecture_segment_panel.csv",
        "kpi_heavy_lecture_segment_summary.json",
        "kpi_identity_gap_assessment.csv",
        "kpi_population_validity_panel.csv",
        "kpi_reliability_panel.csv",
        "kpi_horizon_panel.csv",
        "kpi_depth_panel.csv",
        "kpi_metric_candidates.csv",
    ]
    for fname in legacy_files:
        p = output_dir / fname
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def write_metric_review_report(
    cfg: Stage4Config,
    monthly_panel: pd.DataFrame,
    best_metric: Dict[str, Any],
    best_metric_short_window: Dict[str, Any],
    m2_comparison: Dict[str, Any],
    horizon_comparison_df: pd.DataFrame,
    data_usage_audit: Dict[str, Any],
    consistency_audit: Dict[str, Any] | None = None,
) -> str:
    report_path = cfg.output_dir / "reports" / "metric_review_stage4.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    best = best_metric.get("best_metric", "rsva_m1")
    best_short_window = best_metric_short_window.get("best_metric", "rsva_m1")
    c = m2_comparison
    horizon_lines: List[str] = []
    if not horizon_comparison_df.empty:
        hcmp = horizon_comparison_df.copy().sort_values("horizon_m")
        conc_map = {
            "near_equivalent": "proximas_no_criterio",
            "different": "distantes_no_criterio",
            "proximas_no_criterio": "proximas_no_criterio",
            "distantes_no_criterio": "distantes_no_criterio",
        }
        for _, r in hcmp.iterrows():
            h = int(pd.to_numeric(r.get("horizon_m"), errors="coerce"))
            abs_diff = pd.to_numeric(r.get("mean_abs_diff"), errors="coerce")
            lim = pd.to_numeric(r.get("bootstrap_mean_diff_abs_limit"), errors="coerce")
            conc = str(r.get("numeric_conclusion", "n/d"))
            conc_txt = conc_map.get(conc, conc)
            horizon_lines.append(
                f"- m+{h}: mean_abs_diff={abs_diff:.5f} | bootstrap_abs_limit={lim:.5f} | status={conc_txt}"
            )
    lines = [
        "# Stage 04 - Metric Review",
        "",
        "## Definição de valor estrito",
        "- strict_value = download_aula OR download_plano_aula",
        "- visualização/click NÃO entram em strict value",
        "",
        "## Nomenclatura e cálculo das métricas",
        "- SVS_t = Value Conversion Rate",
        "- SUR_h = Post-Value Retention_h",
        "- RSVA_mh = Value-Qualified Retention_h",
        *build_metric_definitions_markdown(),
        "",
        "## Best metric",
        f"- best_metric (horizontes estendidos): {best} ({metric_display_label(best)})",
        f"- best_metric (comparação curta m+1/m+2): {best_short_window} ({metric_display_label(best_short_window)})",
        "",
        "## Comparação Value-Qualified Retention vs Retention por horizonte",
        f"- m+2 (referência): mean_abs_diff={c.get('mean_abs_diff')} | bootstrap_abs_limit={c.get('bootstrap_mean_diff_abs_limit')}",
        *horizon_lines,
        "- recommended_usage: use Value-Qualified Retention como decisão de valor e Retention como baseline complementar por horizonte",
        "",
        "## Data usage",
        f"- population_primary: {data_usage_audit.get('population_primary')}",
        f"- seo_like_share_in_primary: {data_usage_audit.get('seo_like_share_in_primary')}",
    ]
    if consistency_audit:
        checks = consistency_audit.get("checks", [])
        key_checks = [c for c in checks if c.get("status") != "pass"][:8]
        lines.extend(
            [
                "",
                "## Consistência entre etapas 01-04",
                f"- overall_status: {consistency_audit.get('overall_status')}",
                f"- checks_pass: {consistency_audit.get('counts', {}).get('pass')}",
                f"- checks_warning: {consistency_audit.get('counts', {}).get('warning')}",
                f"- checks_fail: {consistency_audit.get('counts', {}).get('fail')}",
            ]
        )
        if key_checks:
            lines.append("- principais_alertas:")
            for c in key_checks:
                lines.append(f"- alerta {c.get('check_id')}: status={c.get('status')} | {c.get('detail')}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def write_metric_initial_analysis_package(
    cfg: Stage4Config,
    summary_payload: Dict[str, Any],
    chart_bundle: Dict[str, Any],
    best_metric: Dict[str, Any],
) -> Dict[str, str]:
    reports_dir = cfg.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_html = reports_dir / "analise_inicial_da_metrica_interativa.html"
    out_json = reports_dir / "analise_inicial_da_metrica_summary.json"
    out_md = reports_dir / "analise_inicial_da_metrica.md"

    horizon_cmp_df = pd.DataFrame(summary_payload.get("horizon_comparison", []))
    metric_selection_compare = summary_payload.get("metric_selection_compare", {}) or {}
    chart_details = (
        apply_metric_terminology_payload(chart_bundle.get("chart_details", {})) if isinstance(chart_bundle, dict) else {}
    )
    top_event = summary_payload.get("taxonomy_top_event", {}) or {}
    segment_definition_consistency = str(summary_payload.get("segment_definition_consistency") or "")
    metric_definitions_html = build_metric_definitions_html()

    def chart_block(key: str) -> str:
        meta = chart_details.get(key, {})
        if not meta:
            return ""
        img_src = build_embedded_image_src(meta.get("path"))
        img_html = (
            f"<img src='{img_src}' alt='{key}' style='width:100%; border-radius:8px; border:1px solid #E2E8F0;'/>"
            if img_src
            else "<p class='small'>Imagem indisponível para este gráfico.</p>"
        )
        return (
            "<div class='chart-card'>"
            f"<h3>{meta.get('titulo', key)}</h3>"
            f"<p><b>O que mostra:</b> {meta.get('mede', 'n/d')}</p>"
            f"<p><b>Tabelas usadas:</b> {meta.get('tabelas_usadas', 'n/d')}</p>"
            f"<p><b>Transformação principal:</b> {meta.get('transformacoes', 'n/d')}</p>"
            f"{img_html}"
            f"<p class='small'><b>Como ler:</b> {meta.get('como_ler', 'n/d')}</p>"
            "</div>"
        )

    diag_payload = summary_payload.get("rsva_drop_diagnostics", {}) or {}
    diag_summary_df = pd.DataFrame(diag_payload.get("drop_summary", []))
    if not diag_summary_df.empty:
        diag_summary_df = diag_summary_df.rename(
            columns={"diagnostico": "diagnóstico de queda do RSVA", "meses": "meses", "participacao": "participação"}
        )
        if "participação" in diag_summary_df.columns:
            diag_summary_df["participação"] = pd.to_numeric(diag_summary_df["participação"], errors="coerce").round(3)
        diag_summary_html = diag_summary_df.to_html(index=False, classes="table", border=0)
    else:
        diag_summary_html = "<p>Sem quedas suficientes para classificar.</p>"

    diag_recent_df = pd.DataFrame(diag_payload.get("latest_12_diagnostics", []))
    if not diag_recent_df.empty:
        keep = ["month", "d_rsva", "d_svs", "d_sur", "dir_rsva", "dir_svs", "dir_sur", "diagnostico"]
        show = diag_recent_df[[c for c in keep if c in diag_recent_df.columns]].copy()
        if "month" in show.columns:
            show["month"] = pd.to_datetime(show["month"], errors="coerce").dt.strftime("%Y-%m")
        for c in ["d_rsva", "d_svs", "d_sur"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(4)
        show = show.rename(
            columns={
                "month": "mes",
                "d_rsva": "variação RSVA",
                "d_svs": "variação SVS",
                "d_sur": "variação SUR",
                "dir_rsva": "direção RSVA",
                "dir_svs": "direção SVS",
                "dir_sur": "direção SUR",
            }
        )
        diag_recent_html = show.to_html(index=False, classes="table", border=0)
    else:
        diag_recent_html = "<p>Sem série suficiente.</p>"

    diag_legend_rows = [
        ("queda_por_entrada_em_valor", describe_drop_diagnostic("queda_por_entrada_em_valor")),
        ("queda_por_continuidade_pos_valor", describe_drop_diagnostic("queda_por_continuidade_pos_valor")),
        ("queda_mista_adocao_e_retencao", describe_drop_diagnostic("queda_mista_adocao_e_retencao")),
        ("queda_combinacao_outros_sinais", describe_drop_diagnostic("queda_combinacao_outros_sinais")),
    ]
    diag_legend_html = "<ul>" + "".join([f"<li><b>{c}</b>: {t}</li>" for c, t in diag_legend_rows]) + "</ul>"

    segment_diag_payload = summary_payload.get("segment_drop_diagnostics", {}) or {}
    seg_drop_summary_df = pd.DataFrame(segment_diag_payload.get("drop_summary_by_segment", []))
    if not seg_drop_summary_df.empty:
        sds = seg_drop_summary_df.copy()
        keep = ["segment", "diagnostico", "meses", "participacao"]
        sds = sds[[c for c in keep if c in sds.columns]].copy()
        if "participacao" in sds.columns:
            sds["participacao"] = pd.to_numeric(sds["participacao"], errors="coerce").round(3)
        sds = sds.rename(
            columns={
                "segment": "segmento",
                "diagnostico": "diagnóstico",
                "meses": "meses",
                "participacao": "participação",
            }
        )
        segment_drop_summary_html = sds.to_html(index=False, classes="table", border=0)
    else:
        segment_drop_summary_html = "<p>Sem série suficiente para diagnóstico por segmento.</p>"

    seg_latest_df = pd.DataFrame(segment_diag_payload.get("latest_12_diagnostics_by_segment", []))
    if not seg_latest_df.empty:
        sl = seg_latest_df.copy()
        keep = ["segment", "month", "d_rsva", "d_svs", "d_sur", "dir_rsva", "dir_svs", "dir_sur", "diagnostico"]
        sl = sl[[c for c in keep if c in sl.columns]].copy()
        if "month" in sl.columns:
            sl["month"] = pd.to_datetime(sl["month"], errors="coerce").dt.strftime("%Y-%m")
        for c in ["d_rsva", "d_svs", "d_sur"]:
            if c in sl.columns:
                sl[c] = pd.to_numeric(sl[c], errors="coerce").round(4)
        sl = sl.rename(
            columns={
                "segment": "segmento",
                "month": "mes",
                "d_rsva": "variação RSVA",
                "d_svs": "variação SVS",
                "d_sur": "variação SUR",
                "dir_rsva": "direção RSVA",
                "dir_svs": "direção SVS",
                "dir_sur": "direção SUR",
            }
        )
        segment_latest_html = sl.to_html(index=False, classes="table", border=0)
    else:
        segment_latest_html = "<p>Sem recorte recente por segmento.</p>"

    segment_interp_html = "<p>Sem interpretação segmentada disponível.</p>"
    if not seg_drop_summary_df.empty and {"segment", "diagnostico", "participacao"}.issubset(set(seg_drop_summary_df.columns)):
        s = seg_drop_summary_df.copy()
        s["participacao"] = pd.to_numeric(s["participacao"], errors="coerce")
        s = s.dropna(subset=["participacao"])
        if not s.empty:
            dom = (
                s.sort_values(["segment", "participacao"], ascending=[True, False])
                .groupby("segment", as_index=False)
                .head(1)
                .reset_index(drop=True)
            )
            lines = []
            dom_map: Dict[str, str] = {}
            dom_part: Dict[str, float] = {}
            for _, r in dom.iterrows():
                seg = str(r.get("segment"))
                diag_code = str(r.get("diagnostico"))
                part = float(r.get("participacao"))
                seg_name = {"all_users": "Todos", "base_regular": "Base regular", "heavy_users": "Heavy users"}.get(seg, seg)
                lines.append(f"<li><b>{seg_name}</b>: {diag_code} ({part:.1%}). {describe_drop_diagnostic(diag_code)}</li>")
                dom_map[seg] = diag_code
                dom_part[seg] = part
            if {"base_regular", "heavy_users"}.issubset(set(dom_map.keys())):
                if dom_map["base_regular"] == dom_map["heavy_users"]:
                    gap = abs(float(dom_part.get("heavy_users", np.nan)) - float(dom_part.get("base_regular", np.nan)))
                    if pd.notna(gap) and gap >= 0.15:
                        lines.append(
                            "<li><b>Leitura de relevância:</b> base regular e heavy têm o mesmo diagnóstico dominante, mas com intensidade diferente; em heavy a concentração de quedas por continuidade é maior.</li>"
                        )
                    else:
                        lines.append(
                            "<li><b>Leitura de relevância:</b> base regular e heavy têm o mesmo padrão dominante e intensidade parecida; a segmentação agrega pouco sinal adicional neste recorte.</li>"
                        )
                else:
                    lines.append(
                        "<li><b>Leitura de relevância:</b> base regular e heavy têm diagnósticos dominantes diferentes; a segmentação é útil para priorizar ações distintas.</li>"
                    )
            segment_interp_html = "<ul>" + "".join(lines) + "</ul>"

    event_family_df = pd.DataFrame(summary_payload.get("event_family_recent", []))
    if not event_family_df.empty:
        ef = event_family_df.copy()
        if "month" in ef.columns:
            ef["month"] = pd.to_datetime(ef["month"], errors="coerce").dt.strftime("%Y-%m")
        show_cols = [
            "month",
            "aula_user_share",
            "prova_user_share",
            "download_aula_user_share",
            "download_plano_user_share",
            "vis_aula_sem_download_share_viewers",
            "vis_prova_sem_acao_share_viewers",
            "vis_aula_com_download_share_viewers",
            "vis_prova_com_acao_share_viewers",
        ]
        ef = ef[[c for c in show_cols if c in ef.columns]].copy()
        for c in ef.columns:
            if c != "month":
                ef[c] = pd.to_numeric(ef[c], errors="coerce").round(4)
        ef = ef.rename(
            columns={
                "month": "mes",
                "aula_user_share": "share usuários Aula",
                "prova_user_share": "share usuários Prova",
                "download_aula_user_share": "share download_aula",
                "download_plano_user_share": "share download_plano_aula",
                "vis_aula_sem_download_share_viewers": "visualizou aula e não baixou (entre visualizadores)",
                "vis_prova_sem_acao_share_viewers": "visualizou prova e sem ação (entre visualizadores)",
                "vis_aula_com_download_share_viewers": "visualizou aula e baixou (entre visualizadores)",
                "vis_prova_com_acao_share_viewers": "visualizou prova e fez ação (entre visualizadores)",
            }
        )
        event_family_html = ef.to_html(index=False, classes="table", border=0)
    else:
        event_family_html = "<p>Sem dados de segmentação Aula vs Prova.</p>"

    subject_quality_df = pd.DataFrame(summary_payload.get("subject_quality_recent", []))
    if not subject_quality_df.empty:
        sq = subject_quality_df.copy()
        if "month" in sq.columns:
            sq["month"] = pd.to_datetime(sq["month"], errors="coerce").dt.strftime("%Y-%m")
        keep = ["month", "download_events", "pct_with_id_aula", "pct_with_disciplina"]
        sq = sq[[c for c in keep if c in sq.columns]].copy()
        for c in ["download_events", "pct_with_id_aula", "pct_with_disciplina"]:
            if c in sq.columns:
                sq[c] = pd.to_numeric(sq[c], errors="coerce").round(4)
        sq = sq.rename(
            columns={
                "month": "mes",
                "download_events": "downloads (Aula/Plano)",
                "pct_with_id_aula": "share com id_aula",
                "pct_with_disciplina": "share com disciplina mapeada",
            }
        )
        subject_quality_html = sq.to_html(index=False, classes="table", border=0)
    else:
        subject_quality_html = "<p>Sem dados de cobertura por disciplina.</p>"

    subject_top_df = pd.DataFrame(summary_payload.get("subject_top_overall", []))
    if not subject_top_df.empty:
        st = subject_top_df.copy()
        keep = ["disciplina", "download_events", "share_mapped_downloads"]
        st = st[[c for c in keep if c in st.columns]].copy()
        for c in ["download_events", "share_mapped_downloads"]:
            if c in st.columns:
                st[c] = pd.to_numeric(st[c], errors="coerce").round(4)
        st = st.rename(
            columns={
                "disciplina": "disciplina",
                "download_events": "downloads",
                "share_mapped_downloads": "share entre downloads com disciplina",
            }
        )
        subject_top_html = st.to_html(index=False, classes="table", border=0)
    else:
        subject_top_html = "<p>Sem ranking de disciplinas para o período.</p>"

    linear_models_df = pd.DataFrame(summary_payload.get("rsva_linear_models", []))
    if not linear_models_df.empty:
        lm = linear_models_df.copy()
        keep = [
            "segment",
            "n_months",
            "min_months_additive_recommended",
            "min_months_interaction_recommended",
            "sample_ok_additive",
            "sample_ok_interaction",
            "additive_r2",
            "additive_coef_svs_t",
            "additive_coef_sur_t",
            "interaction_r2",
            "interaction_coef_svs_x_sur",
            "mean_abs_residual_additive",
        ]
        lm = lm[[c for c in keep if c in lm.columns]].copy()
        for c in [x for x in lm.columns if x not in {"segment"}]:
            lm[c] = pd.to_numeric(lm[c], errors="coerce").round(4)
        lm = lm.rename(
            columns={
                "segment": "segmento",
                "n_months": "meses usados",
                "min_months_additive_recommended": "mínimo recomendado (aditivo)",
                "min_months_interaction_recommended": "mínimo recomendado (interação)",
                "sample_ok_additive": "amostra suficiente aditivo",
                "sample_ok_interaction": "amostra suficiente interação",
                "additive_r2": "R2 modelo aditivo",
                "additive_coef_svs_t": "coef SVS (aditivo)",
                "additive_coef_sur_t": "coef SUR (aditivo)",
                "interaction_r2": "R2 modelo com interação",
                "interaction_coef_svs_x_sur": "coef SVSxSUR (interação)",
                "mean_abs_residual_additive": "erro médio abs (aditivo)",
            }
        )
        linear_models_html = lm.to_html(index=False, classes="table", border=0)
    else:
        linear_models_html = "<p>Sem meses suficientes para regressão linear por segmento.</p>"

    impacts_top_df = pd.DataFrame(summary_payload.get("event_impacts_top_negative", []))
    if not impacts_top_df.empty:
        it = impacts_top_df.copy()
        keep = ["segment", "metric", "event_type", "effect", "ci_low", "ci_high", "q_value", "n_exposed", "n_unexposed", "sufficient_sample", "negative_confirmed"]
        it = it[[c for c in keep if c in it.columns]].copy()
        for c in ["effect", "ci_low", "ci_high", "q_value"]:
            if c in it.columns:
                it[c] = pd.to_numeric(it[c], errors="coerce").round(4)
        for c in ["n_exposed", "n_unexposed"]:
            if c in it.columns:
                it[c] = pd.to_numeric(it[c], errors="coerce").fillna(0).astype(int)
        it = it.rename(
            columns={
                "segment": "segmento",
                "metric": "métrica",
                "event_type": "evento",
                "effect": "efeito (expostos - não expostos)",
                "ci_low": "ic_low",
                "ci_high": "ic_high",
                "q_value": "q_value_fdr",
                "n_exposed": "n_expostos",
                "n_unexposed": "n_não_expostos",
                "sufficient_sample": "amostra_suficiente",
                "negative_confirmed": "negativo_confirmado",
            }
        )
        impacts_top_html = it.to_html(index=False, classes="table", border=0)
    else:
        impacts_top_html = "<p>Sem eventos com evidência negativa suficiente no recorte atual.</p>"

    class_impacts_top_df = pd.DataFrame(summary_payload.get("event_class_impacts_top", []))
    if not class_impacts_top_df.empty:
        ct = class_impacts_top_df.copy()
        keep = ["segment", "metric", "event_class", "effect", "ci_low", "ci_high", "q_value", "n_exposed", "n_unexposed", "sufficient_sample", "negative_confirmed", "positive_confirmed"]
        ct = ct[[c for c in keep if c in ct.columns]].copy()
        for c in ["effect", "ci_low", "ci_high", "q_value"]:
            if c in ct.columns:
                ct[c] = pd.to_numeric(ct[c], errors="coerce").round(4)
        for c in ["n_exposed", "n_unexposed"]:
            if c in ct.columns:
                ct[c] = pd.to_numeric(ct[c], errors="coerce").fillna(0).astype(int)
        ct = ct.rename(
            columns={
                "segment": "segmento",
                "metric": "métrica",
                "event_class": "classe_evento",
                "effect": "efeito (expostos - não expostos)",
                "ci_low": "ic_low",
                "ci_high": "ic_high",
                "q_value": "q_value_fdr",
                "n_exposed": "n_expostos",
                "n_unexposed": "n_não_expostos",
                "sufficient_sample": "amostra_suficiente",
                "negative_confirmed": "negativo_confirmado",
                "positive_confirmed": "positivo_confirmado",
            }
        )
        class_impacts_top_html = ct.to_html(index=False, classes="table", border=0)
    else:
        class_impacts_top_html = "<p>Sem classes com evidência estatística suficiente no recorte atual.</p>"

    focus_events_df = pd.DataFrame(summary_payload.get("focus_event_impacts", []))
    if not focus_events_df.empty:
        fe = focus_events_df.copy()
        keep = [
            "event_type",
            "event_explanation",
            "metric",
            "effect",
            "ci_low",
            "ci_high",
            "q_value",
            "n_exposed",
            "n_unexposed",
            "sufficient_sample",
            "negative_confirmed",
            "positive_confirmed",
        ]
        fe = fe[[c for c in keep if c in fe.columns]].copy()
        for c in ["effect", "ci_low", "ci_high", "q_value"]:
            if c in fe.columns:
                fe[c] = pd.to_numeric(fe[c], errors="coerce").round(4)
        for c in ["n_exposed", "n_unexposed"]:
            if c in fe.columns:
                fe[c] = pd.to_numeric(fe[c], errors="coerce").fillna(0).astype(int)
        fe["leitura_objetivo"] = np.where(
            pd.to_numeric(fe["effect"], errors="coerce") > 0,
            "associado à melhora da métrica",
            np.where(
                pd.to_numeric(fe["effect"], errors="coerce") < 0,
                "associado à piora da métrica",
                "efeito próximo de zero",
            ),
        )
        fe = fe.rename(
            columns={
                "event_type": "evento",
                "event_explanation": "interpretação do evento",
                "metric": "métrica",
                "effect": "efeito",
                "ci_low": "ic_low",
                "ci_high": "ic_high",
                "q_value": "q_value_fdr",
                "n_exposed": "n_expostos",
                "n_unexposed": "n_não_expostos",
                "sufficient_sample": "amostra_suficiente",
                "negative_confirmed": "negativo_confirmado",
                "positive_confirmed": "positivo_confirmado",
                "leitura_objetivo": "leitura no objetivo",
            }
        )
        focus_events_html = fe.to_html(index=False, classes="table", border=0)
    else:
        focus_events_html = "<p>Sem evidências suficientes para os eventos foco no recorte atual.</p>"

    horizon_cmp_html = "<p>Sem dados suficientes para comparação por horizonte.</p>"
    if not horizon_cmp_df.empty:
        hc = horizon_cmp_df.copy()
        keep = [
            "horizon_m",
            "months_used",
            "avg_rsva",
            "avg_retention",
            "mean_abs_diff",
            "bootstrap_mean_diff_abs_limit",
            "avg_svs_t",
            "avg_sur_h",
            "numeric_conclusion",
        ]
        hc = hc[[c for c in keep if c in hc.columns]].copy()
        for c in [
            "horizon_m",
            "months_used",
            "avg_rsva",
            "avg_retention",
            "mean_abs_diff",
            "bootstrap_mean_diff_abs_limit",
            "avg_svs_t",
            "avg_sur_h",
        ]:
            if c in hc.columns:
                hc[c] = pd.to_numeric(hc[c], errors="coerce")
        if "horizon_m" in hc.columns:
            hc = hc.sort_values("horizon_m")
            hc["horizon_m"] = hc["horizon_m"].apply(lambda v: f"m+{int(v)}" if pd.notna(v) else "n/d")
        if "numeric_conclusion" in hc.columns:
            hc["numeric_conclusion"] = (
                hc["numeric_conclusion"]
                .astype(str)
                .replace({"near_equivalent": "proximas_no_criterio", "different": "distantes_no_criterio"})
            )
        for c in ["avg_rsva", "avg_retention", "mean_abs_diff", "bootstrap_mean_diff_abs_limit", "avg_svs_t", "avg_sur_h"]:
            if c in hc.columns:
                hc[c] = pd.to_numeric(hc[c], errors="coerce").round(4)
        hc = hc.rename(
            columns={
                "horizon_m": "horizonte",
                "months_used": "meses usados",
                "avg_rsva": "RSVA médio",
                "avg_retention": "Retention médio",
                "mean_abs_diff": "diferença média abs",
                "bootstrap_mean_diff_abs_limit": "limite bootstrap abs",
                "avg_svs_t": "SVS médio",
                "avg_sur_h": "SUR médio (horizonte)",
                "numeric_conclusion": "conclusão numérica",
            }
        )
        horizon_cmp_html = hc.to_html(index=False, classes="table", border=0)

    uncertainty_recent_df = pd.DataFrame(summary_payload.get("metric_uncertainty_recent", []))
    uncertainty_html = "<p>Sem dados de bandas de incerteza para o recorte atual.</p>"
    if not uncertainty_recent_df.empty:
        ur = uncertainty_recent_df.copy()
        keep = ["month", "metric", "value", "ci_low", "ci_high", "ci_half_width", "numerator", "denominator"]
        ur = ur[[c for c in keep if c in ur.columns]].copy()
        if "month" in ur.columns:
            ur["month"] = pd.to_datetime(ur["month"], errors="coerce").dt.strftime("%Y-%m")
        for c in ["value", "ci_low", "ci_high", "ci_half_width"]:
            if c in ur.columns:
                ur[c] = pd.to_numeric(ur[c], errors="coerce").round(4)
        for c in ["numerator", "denominator"]:
            if c in ur.columns:
                ur[c] = pd.to_numeric(ur[c], errors="coerce").fillna(0).astype(int)
        ur = ur.rename(
            columns={
                "month": "mes",
                "metric": "métrica",
                "value": "valor",
                "ci_low": "ic95_low",
                "ci_high": "ic95_high",
                "ci_half_width": "semi_largura_ic95",
                "numerator": "numerador",
                "denominator": "denominador",
            }
        )
        uncertainty_html = ur.to_html(index=False, classes="table", border=0)

    cohort_hazard_df = pd.DataFrame(summary_payload.get("strict_cohort_hazard_curve", []))
    cohort_summary_payload = summary_payload.get("strict_cohort_summary", {}) or {}
    cohort_hazard_html = "<p>Sem dados suficientes para curva de survival/hazard no recorte atual.</p>"
    if not cohort_hazard_df.empty:
        ch = cohort_hazard_df.copy()
        keep = [
            "horizon_m",
            "n_eligible",
            "n_at_risk",
            "n_events_first_return",
            "hazard",
            "hazard_ci_low",
            "hazard_ci_high",
            "survival",
            "cumulative_return",
            "cumulative_return_ci_low",
            "cumulative_return_ci_high",
        ]
        ch = ch[[c for c in keep if c in ch.columns]].copy()
        for c in [x for x in ch.columns if x not in {"horizon_m"}]:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        if "horizon_m" in ch.columns:
            ch["horizon_m"] = pd.to_numeric(ch["horizon_m"], errors="coerce")
            ch = ch.sort_values("horizon_m")
            ch["horizon_m"] = ch["horizon_m"].apply(lambda v: f"h={int(v)}" if pd.notna(v) else "n/d")
        for c in ["hazard", "hazard_ci_low", "hazard_ci_high", "survival", "cumulative_return", "cumulative_return_ci_low", "cumulative_return_ci_high"]:
            if c in ch.columns:
                ch[c] = pd.to_numeric(ch[c], errors="coerce").round(4)
        for c in ["n_eligible", "n_at_risk", "n_events_first_return"]:
            if c in ch.columns:
                ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0).astype(int)
        ch = ch.rename(
            columns={
                "horizon_m": "horizonte",
                "n_eligible": "usuários elegíveis",
                "n_at_risk": "usuários em risco",
                "n_events_first_return": "eventos de 1º retorno",
                "hazard": "hazard",
                "hazard_ci_low": "hazard_ic95_low",
                "hazard_ci_high": "hazard_ic95_high",
                "survival": "survival",
                "cumulative_return": "retorno acumulado",
                "cumulative_return_ci_low": "retorno_acumulado_ic95_low",
                "cumulative_return_ci_high": "retorno_acumulado_ic95_high",
            }
        )
        cohort_hazard_html = ch.to_html(index=False, classes="table", border=0)
    cohort_definition = str(cohort_summary_payload.get("cohort_definition", ""))
    cohort_max_follow = str(cohort_summary_payload.get("max_follow_month", ""))
    cohort_count = cohort_summary_payload.get("users_in_first_strict_cohort")

    heavy_types_df = pd.DataFrame(summary_payload.get("heavy_user_types_reliable", []))
    heavy_types_summary_payload = summary_payload.get("heavy_user_types_summary", {}) or {}
    heavy_types_method = str(heavy_types_summary_payload.get("reliability_rule_global", ""))
    heavy_types_method = heavy_types_method.replace("<", "&lt;").replace(">", "&gt;")
    heavy_types_html = "<p>Sem categorias com evidência robusta de tipologia heavy no recorte atual.</p>"
    top_reliable = pd.DataFrame(heavy_types_summary_payload.get("top_reliable_by_dimension", []))
    if not top_reliable.empty and {"dimension", "category", "lift_vs_overall", "users_total"}.issubset(set(top_reliable.columns)):
        top_reliable["lift_vs_overall"] = pd.to_numeric(top_reliable["lift_vs_overall"], errors="coerce")
        top_reliable["users_total"] = pd.to_numeric(top_reliable["users_total"], errors="coerce")
        top_reliable = top_reliable.dropna(subset=["lift_vs_overall", "users_total"])
        if not top_reliable.empty:
            dim_label_map = {
                "device": "device",
                "location_estado": "localização",
                "subject_top_discipline": "matéria",
                "usage_dominant_weekday": "dia dominante",
                "usage_dominant_hour_bin": "faixa horária dominante",
            }
            lines: List[str] = ["<ul>"]
            for dim, grp in top_reliable.groupby("dimension", sort=False):
                g = grp.sort_values(["lift_vs_overall", "users_total"], ascending=[False, False]).head(3)
                cats = ", ".join([f"{str(r['category'])} (lift {float(r['lift_vs_overall']):.2f})" for _, r in g.iterrows()])
                lines.append(f"<li><b>{dim_label_map.get(str(dim), str(dim))}:</b> {cats}</li>")
            lines.append("</ul>")
            heavy_types_html = "".join(lines)

    best_metric_id = str(best_metric.get("best_metric") or "rsva_m1")
    product_metric_id = str(summary_payload.get("metric_recommendation_product") or "rsva_m1")
    segment_definition_html = (
        f"<p><b>Segmentação heavy/base_regular:</b> {segment_definition_consistency}</p>"
        if segment_definition_consistency
        else ""
    )
    chart_keys = set(chart_details.keys())
    if "rsva_drop_summary" in chart_keys:
        diag_summary_html = ""
    if "rsva_diagnostics_monthly" in chart_keys:
        diag_recent_html = ""
    if "segment_drop_patterns" in chart_keys:
        segment_drop_summary_html = ""
    if "segment_monthly_diagnostics" in chart_keys:
        segment_latest_html = ""
    if ("event_family_shares" in chart_keys) or ("event_family_segmentation" in chart_keys):
        event_family_html = ""
    if "subject_downloads" in chart_keys:
        subject_quality_html = ""
        subject_top_html = ""
    if "rsva_linear_segments" in chart_keys:
        linear_models_html = ""
    if ("event_class_impacts" in chart_keys) or ("event_class_effect_heatmap" in chart_keys):
        class_impacts_top_html = ""
    if "focus_event_impacts" in chart_keys:
        impacts_top_html = ""
        focus_events_html = ""
    if ("horizon_rsva_retention" in chart_keys) or ("horizon_svs_sur" in chart_keys):
        horizon_cmp_html = ""
    if "metric_uncertainty_bands" in chart_keys:
        uncertainty_html = ""
    if "strict_cohort_survival_hazard" in chart_keys:
        cohort_hazard_html = ""

    def fmt_pct(v: Any) -> str:
        try:
            fv = float(v)
            if pd.isna(fv):
                return "n/d"
            return f"{fv:.1%}"
        except Exception:
            return "n/d"

    def fmt_num(v: Any, ndigits: int = 4) -> str:
        try:
            fv = float(v)
            if pd.isna(fv):
                return "n/d"
            return f"{fv:.{ndigits}f}"
        except Exception:
            return "n/d"

    def fmt_delta(v: Any) -> str:
        try:
            fv = float(v)
            if pd.isna(fv):
                return "n/d"
            return f"{fv:+.4f}"
        except Exception:
            return "n/d"

    summary_cards: List[Dict[str, str]] = []
    priority_action = "Recuperar conversão de visualização para download em Aula e reforçar continuidade pós-valor (SUR), com foco adicional em heavy users."

    decomp_recent_df = pd.DataFrame(summary_payload.get("decomposition_recent", []))
    if not decomp_recent_df.empty and {"month", "rsva_m1", "svs_t", "sur_t"}.issubset(set(decomp_recent_df.columns)):
        drec = decomp_recent_df.copy()
        drec["month"] = pd.to_datetime(drec["month"], errors="coerce")
        drec = drec.sort_values("month").dropna(subset=["month"])
        if len(drec) >= 2:
            start = drec.iloc[0]
            end = drec.iloc[-1]
            summary_cards.append(
                {
                    "title": "Últimos 12 meses (decisão)",
                    "metric": (
                        f"RSVA {fmt_num(start['rsva_m1'])} -> {fmt_num(end['rsva_m1'])} ({fmt_delta(float(end['rsva_m1']) - float(start['rsva_m1']))}); "
                        f"SVS {fmt_num(start['svs_t'])} -> {fmt_num(end['svs_t'])}; SUR {fmt_num(start['sur_t'])} -> {fmt_num(end['sur_t'])}."
                    ),
                    "action": "Acompanhar semanalmente conversão em download_aula e retorno no mês seguinte.",
                }
            )

    if not diag_recent_df.empty and {"dir_rsva", "diagnostico", "month"}.issubset(set(diag_recent_df.columns)):
        dr = diag_recent_df.copy()
        dr["month"] = pd.to_datetime(dr["month"], errors="coerce")
        drops = dr[dr["dir_rsva"] == "cai"].copy()
        if not drops.empty:
            cnt = drops["diagnostico"].value_counts()
            top_share = float(cnt.iloc[0] / len(drops))
            top_codes = [str(x) for x in cnt[cnt == cnt.iloc[0]].index.tolist()]
            if len(top_codes) > 1:
                top_desc = f"empate entre {', '.join(top_codes)} ({fmt_pct(top_share)} cada)"
            else:
                top_desc = f"{top_codes[0]} ({fmt_pct(top_share)})"
            summary_cards.append(
                {
                    "title": "Quedas de RSVA (12 meses)",
                    "metric": f"{int(len(drops))} meses com queda; padrão dominante: {top_desc}.",
                    "action": "Nos meses de queda, separar sempre queda de entrada (SVS) vs continuidade (SUR) para intervenção direcionada.",
                }
            )
        else:
            summary_cards.append(
                {
                    "title": "Quedas de RSVA (12 meses)",
                    "metric": "Sem quedas relevantes de RSVA no recorte recente.",
                    "action": "Manter monitoramento mensal para detectar reversão de tendência.",
                }
            )

    seg_recent_df = pd.DataFrame(summary_payload.get("segment_recent_metrics", []))
    if not seg_recent_df.empty and {"segment", "month", "rsva_m1"}.issubset(set(seg_recent_df.columns)):
        sr = seg_recent_df.copy()
        sr["month"] = pd.to_datetime(sr["month"], errors="coerce")
        sr = sr.sort_values(["segment", "month"]).dropna(subset=["month"])
        deltas: Dict[str, float] = {}
        for seg_id in ["base_regular", "heavy_users"]:
            s = sr[sr["segment"] == seg_id].copy()
            if len(s) >= 2:
                deltas[seg_id] = float(pd.to_numeric(s.iloc[-1]["rsva_m1"], errors="coerce") - pd.to_numeric(s.iloc[0]["rsva_m1"], errors="coerce"))
        if deltas:
            summary_cards.append(
                {
                    "title": "Segmentos (regular vs heavy)",
                    "metric": f"Delta RSVA: regular {fmt_delta(deltas.get('base_regular'))} | heavy {fmt_delta(deltas.get('heavy_users'))}.",
                    "action": "Priorizar plano de retenção pós-valor em heavy quando o delta de heavy piorar mais que regular.",
                }
            )

    event_family_recent_raw = pd.DataFrame(summary_payload.get("event_family_recent", []))
    if not event_family_recent_raw.empty and {
        "aula_user_share",
        "download_aula_user_share",
        "vis_aula_sem_download_share_viewers",
    }.issubset(set(event_family_recent_raw.columns)):
        er = event_family_recent_raw.copy()
        er["month"] = pd.to_datetime(er["month"], errors="coerce")
        er = er.sort_values("month").dropna(subset=["month"])
        if len(er) >= 2:
            st = er.iloc[0]
            en = er.iloc[-1]
            summary_cards.append(
                {
                    "title": "Funnel Aula/Prova",
                    "metric": (
                        f"share usuários Aula {fmt_num(st['aula_user_share'])} -> {fmt_num(en['aula_user_share'])}; "
                        f"share download_aula {fmt_num(st['download_aula_user_share'])} -> {fmt_num(en['download_aula_user_share'])}; "
                        f"fricção Aula (visualizou sem baixar) {fmt_num(st['vis_aula_sem_download_share_viewers'])} -> {fmt_num(en['vis_aula_sem_download_share_viewers'])}."
                    ),
                    "action": "Atacar diretamente o passo visualização->download em Aula (conteúdo, CTA e fluxo).",
                }
            )

    if not horizon_cmp_df.empty and {"horizon_m", "mean_abs_diff", "numeric_conclusion"}.issubset(set(horizon_cmp_df.columns)):
        hsum = horizon_cmp_df.copy()
        hsum["horizon_m"] = pd.to_numeric(hsum["horizon_m"], errors="coerce")
        hsum["mean_abs_diff"] = pd.to_numeric(hsum["mean_abs_diff"], errors="coerce")
        hsum = hsum.dropna(subset=["horizon_m", "mean_abs_diff"]).sort_values("horizon_m")
        if not hsum.empty:
            best_row = hsum.sort_values("mean_abs_diff", ascending=True).iloc[0]
            worst_row = hsum.sort_values("mean_abs_diff", ascending=False).iloc[0]
            summary_cards.append(
                {
                    "title": "Horizontes (m+1, m+2, m+4, m+6)",
                    "metric": (
                        f"Menor diferença RSVA vs Retention: m+{int(best_row['horizon_m'])} ({fmt_num(best_row['mean_abs_diff'], 5)}); "
                        f"maior diferença: m+{int(worst_row['horizon_m'])} ({fmt_num(worst_row['mean_abs_diff'], 5)})."
                    ),
                    "action": "Comparar curto e médio prazo por horizonte e usar SUR_h para explicar mudanças de RSVA em cada janela.",
                }
            )

    if best_metric_id:
        changed = bool(metric_selection_compare.get("changed"))
        change_txt = "sim" if changed else "não"
        summary_cards.append(
            {
                "title": "Métrica de decisão",
                "metric": (
                    f"Métrica principal de decisão: {metric_display_label(product_metric_id)} ({product_metric_id}). "
                    f"Métricas de diagnóstico: SVS (entrada em valor), SUR (continuidade pós-valor) e Retention (retenção geral). "
                    f"Comparação técnica curta (m+1/m+2) mudou: {change_txt}."
                ),
                "action": "Para decisão de valor do produto, manter Value-Qualified Retention como métrica principal e Value Conversion Rate/Post-Value Retention como diagnóstico.",
            }
        )

    summary_boxes_html = ""
    if summary_cards:
        boxes = []
        for c in summary_cards:
            boxes.append(
                "<div class='summary-box'>"
                f"<h4>{apply_metric_terminology_text(c.get('title','Resumo'))}</h4>"
                f"<p>{apply_metric_terminology_text(c.get('metric','n/d'))}</p>"
                f"<p class='small'><b>Ação:</b> {apply_metric_terminology_text(c.get('action',''))}</p>"
                "</div>"
            )
        summary_boxes_html = "<div class='summary-grid'>" + "".join(boxes) + "</div>"

    priority_action_html = (
        "<div class='card'>"
        f"<p><b>Ação prioritária:</b> {apply_metric_terminology_text(priority_action)}</p>"
        "</div>"
    )

    diag_summary_label = "<p><b>Resumo histórico das quedas de Value-Qualified Retention:</b></p>" if (("rsva_drop_summary" in chart_keys) or str(diag_summary_html).strip()) else ""
    diag_recent_label = ""
    segment_drop_label = "<p><b>Resumo histórico das quedas por segmento:</b></p>" if str(segment_drop_summary_html).strip() else ""
    segment_recent_label = "<p><b>Aplicação mês a mês por segmento (últimos 12 meses):</b></p>" if str(segment_latest_html).strip() else ""
    event_family_label = "<p><b>Tabela complementar de segmentação Aula vs Prova:</b></p>" if str(event_family_html).strip() else ""
    subject_quality_label = "<p><b>Cobertura mensal do mapeamento:</b></p>" if str(subject_quality_html).strip() else ""
    subject_top_label = "<p><b>Disciplinas com mais downloads no período de decisão:</b></p>" if str(subject_top_html).strip() else ""
    impacts_label = "<p><b>Eventos foco (interpretação direta):</b></p>" if str(focus_events_html).strip() else ""
    horizon_label = "<p><b>Resumo por horizonte:</b></p>" if str(horizon_cmp_html).strip() else ""
    uncertainty_label = "<p><b>Tabela complementar de incerteza:</b></p>" if str(uncertainty_html).strip() else ""
    cohort_label = "<p><b>Tabela complementar de cohort survival/hazard:</b></p>" if str(cohort_hazard_html).strip() else ""

    diag_summary_html = apply_metric_terminology_text(diag_summary_html)
    diag_recent_html = apply_metric_terminology_text(diag_recent_html)
    segment_drop_summary_html = apply_metric_terminology_text(segment_drop_summary_html)
    segment_latest_html = apply_metric_terminology_text(segment_latest_html)
    segment_interp_html = apply_metric_terminology_text(segment_interp_html)
    event_family_html = apply_metric_terminology_text(event_family_html)
    subject_quality_html = apply_metric_terminology_text(subject_quality_html)
    subject_top_html = apply_metric_terminology_text(subject_top_html)
    linear_models_html = apply_metric_terminology_text(linear_models_html)
    impacts_top_html = apply_metric_terminology_text(impacts_top_html)
    class_impacts_top_html = apply_metric_terminology_text(class_impacts_top_html)
    focus_events_html = apply_metric_terminology_text(focus_events_html)
    horizon_cmp_html = apply_metric_terminology_text(horizon_cmp_html)
    uncertainty_html = apply_metric_terminology_text(uncertainty_html)
    cohort_hazard_html = apply_metric_terminology_text(cohort_hazard_html)
    heavy_types_html = apply_metric_terminology_text(heavy_types_html)
    diag_legend_html = apply_metric_terminology_text(diag_legend_html)
    segment_definition_html = apply_metric_terminology_text(segment_definition_html)
    event_family_label = apply_metric_terminology_text(event_family_label)
    subject_quality_label = apply_metric_terminology_text(subject_quality_label)
    subject_top_label = apply_metric_terminology_text(subject_top_label)
    impacts_label = apply_metric_terminology_text(impacts_label)
    horizon_label = apply_metric_terminology_text(horizon_label)
    uncertainty_label = apply_metric_terminology_text(uncertainty_label)
    cohort_label = apply_metric_terminology_text(cohort_label)

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #F7FAFC; color: #1A202C; }
    .container { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 8px 0; font-size: 30px; color: #102A43; }
    h2 { margin-top: 30px; margin-bottom: 8px; font-size: 22px; color: #102A43; }
    h3 { margin: 0 0 6px 0; font-size: 18px; color: #102A43; }
    p { margin: 8px 0; line-height: 1.45; }
    .small { color: #4A5568; font-size: 13px; }
    .card { background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; margin: 10px 0; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin: 10px 0 18px 0; }
    .summary-box { background: white; border: 1px solid #CBD5E1; border-left: 5px solid #0b7285; border-radius: 10px; padding: 10px 12px; }
    .summary-box h4 { margin: 0 0 6px 0; font-size: 15px; color: #102A43; }
    .summary-box p { margin: 4px 0; }
    .chart-card { background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; margin: 20px 0 28px 0; }
    .table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; background: white; }
    .table th, .table td { border: 1px solid #E2E8F0; padding: 8px; vertical-align: top; text-align: left; }
    .table th { background: #F0F4F8; color: #243B53; }
    """

    html = f"""
<!DOCTYPE html>
<html lang='pt-BR'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Análise inicial da métrica</title>
  <style>{css}</style>
</head>
<body>
  <div class='container'>
    <h1>Análise inicial da métrica</h1>
    <p class='small'>Versão com strict value fixo por regra de negócio.</p>

    <h2>0) Definições e escopo</h2>
    <div class='card'>
      <p><b>Recomendação de decisão de produto:</b> {metric_display_label(product_metric_id)} (<code>{product_metric_id}</code>), com Value Conversion Rate e Post-Value Retention_h para diagnóstico.</p>
      <p><b>Definição de strict value (regra final):</b> strict_value = <code>download_aula OR download_plano_aula</code></p>
      <p><b>O que NÃO entra em strict value:</b> visualização, click e demais eventos sem download.</p>
      <p><b>Eventos strict usados no KPI:</b> {', '.join(STRICT_DOWNLOAD_EVENTS)}</p>
      {segment_definition_html}
      <p><b>Nomenclatura usada nesta apresentação:</b> <code>SVS_t → Value Conversion Rate</code>; <code>SUR_h → Post-Value Retention_h</code>; <code>RSVA_mh → Value-Qualified Retention_h</code>.</p>
      <p><b>Como cada métrica foi calculada:</b></p>
      {metric_definitions_html}
      <p><b>Sazonalidade:</b> análises de impacto por evento/classe são ajustadas por mês, UF e calendário escolar brasileiro (rede = <code>todas</code>, por indisponibilidade de rede no cadastro de usuários).</p>
    </div>

    <h2>Resumo executivo</h2>
    {priority_action_html}
    {summary_boxes_html}

    <h2>1) Diagnóstico de queda (Value-Qualified Retention/Value Conversion Rate/Post-Value Retention)</h2>
    <p><b>Definições dos diagnósticos:</b></p>
    {diag_legend_html}
    {diag_summary_label}
    {chart_block('rsva_drop_summary')}
    {diag_summary_html}
    {diag_recent_label}

    <h2>2) Diferença por segmento (regular vs heavy)</h2>
    <p>Mesma decomposição de Value-Qualified Retention, Value Conversion Rate e Post-Value Retention, agora comparando todos, regulares e heavy.</p>
    {chart_block('segment_rsva_svs_sur')}
    {chart_block('segment_drop_patterns')}
    {chart_block('segment_monthly_diagnostics')}
    {segment_drop_label}
    {segment_drop_summary_html}
    <p><b>Interpretação prática por segmento:</b></p>
    {segment_interp_html}
    {segment_recent_label}
    {segment_latest_html}

    <h2>3) Segmentação de eventos (Aula vs Prova)</h2>
    <p>Aqui separamos Aula de Prova e medimos fricção de visualização sem ação de valor.</p>
    {chart_block('event_family_shares')}
    {chart_block('event_family_segmentation')}
    {event_family_label}
    {event_family_html}

    <h2>4) Disciplinas dos downloads (Aula/Plano)</h2>
    <p>A disciplina é lida de <code>stg_lessons.disciplina</code> via join por <code>id_aula</code> nos eventos de download.</p>
    {chart_block('subject_downloads')}
    {subject_quality_label}
    {subject_quality_html}
    {subject_top_label}
    {subject_top_html}

    <h2>5) Modelo linear Value-Qualified Retention/Value Conversion Rate/Post-Value Retention (regular vs heavy)</h2>
    <p>Regressão linear mensal por segmento para avaliar quanto Value Conversion Rate e Post-Value Retention explicam Value-Qualified Retention. A tabela informa se a amostra de meses é suficiente para interpretação estável.</p>
    <p class='small'><b>Dados e transformação:</b> usamos apenas meses de decisão; cada linha do modelo é um mês-segmento com Value-Qualified Retention, Value Conversion Rate e Post-Value Retention agregados. Modelo aditivo: Value-Qualified Retention = β0 + β1·Value Conversion Rate + β2·Post-Value Retention. Modelo com interação: Value-Qualified Retention = β0 + β1·Value Conversion Rate + β2·Post-Value Retention + β3·(Value Conversion Rate×Post-Value Retention).</p>
    <p class='small'><b>Leitura:</b> o modelo com interação tende a R²≈1 por identidade algébrica (Value-Qualified Retention = Value Conversion Rate×Post-Value Retention), então o sinal útil para diagnóstico vem do modelo aditivo e dos coeficientes por segmento. Isso não implica causalidade.</p>
    {chart_block('rsva_linear_segments')}
    {linear_models_html}

    <h2>6) Classes de eventos para padrões Value-Qualified Retention/Value Conversion Rate/Post-Value Retention</h2>
    <p>Classificações de eventos para identificar padrões reutilizáveis sem mexer na regra strict.</p>
    {chart_block('event_class_impacts')}
    {chart_block('event_class_effect_heatmap')}
    {class_impacts_top_html}

    <h2>7) Eventos com impacto nas métricas</h2>
    <p>Comparação expostos vs não expostos por event_type em Value-Qualified Retention, Value Conversion Rate e Post-Value Retention (com estratificação por mês, UF e calendário; FDR; filtro mínimo de amostra).</p>
    <p class='small'><b>Regra de cor:</b> verde = melhora da métrica; vermelho = piora da métrica.</p>
    <p class='small'><b>Nota técnica:</b> análises de Post-Value Retention são condicionadas a usuários strict e devem ser interpretadas como associação monitorada, não inferência causal.</p>
    {chart_block('negative_event_impacts_rsva')}
    {chart_block('focus_event_impacts')}
    {impacts_top_html}
    {impacts_label}
    {focus_events_html}

    <h2>8) Comparação de horizontes (m+1, m+2, m+4, m+6)</h2>
    <p>Esta seção compara Value-Qualified Retention e Retention em diferentes janelas e mostra como Value Conversion Rate/Post-Value Retention se comportam em cada horizonte.</p>
    {chart_block('horizon_rsva_retention')}
    {chart_block('horizon_svs_sur')}
    {horizon_label}
    {horizon_cmp_html}

    <h2>9) Bandas de incerteza das métricas</h2>
    <p>Intervalos de confiança (Wilson, 95%) para as métricas principais, usando numerador/denominador de cada mês.</p>
    {chart_block('metric_uncertainty_bands')}
    {uncertainty_label}
    {uncertainty_html}

    <h2>10) Cohort survival/hazard pós-valor</h2>
    <p>Coorte definida pelo primeiro mês com strict value por usuário; avaliamos tempo até o primeiro retorno ativo.</p>
    <p class='small'><b>Definição operacional:</b> {cohort_definition if cohort_definition else 'n/d'} | <b>Último mês de follow-up:</b> {cohort_max_follow if cohort_max_follow else 'n/d'} | <b>Usuários na coorte:</b> {cohort_count if cohort_count is not None else 'n/d'}.</p>
    {chart_block('strict_cohort_survival_hazard')}
    {cohort_label}
    {cohort_hazard_html}

    <h2>11) Análise auxiliar de eventos (uplift)</h2>
    <p><b>Importante:</b> uplift não define strict value do KPI; é apenas sinal auxiliar.</p>
    <p><b>Evento de maior uplift no snapshot:</b> {top_event.get('event_type', 'n/d')} | uplift={top_event.get('adjusted_uplift_next_active', np.nan)} | CI=[{top_event.get('ci_low', np.nan)}, {top_event.get('ci_high', np.nan)}]</p>
    {chart_block('taxonomy_uplift')}

    <h2>12) Evidências visuais complementares</h2>
    {chart_block('north_star_trend')}

    <h2>13) Tipologias de heavy users (localização, device, matéria, dias/horários)</h2>
    {chart_block('heavy_user_types_lift')}
  </div>
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")
    out_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "# Análise inicial da métrica",
                "",
                f"- Relatório interativo: `{out_html}`",
                f"- Sumário técnico: `{out_json}`",
                "",
                "- Definição strict value: download_aula OR download_plano_aula",
                "- Nomenclatura da apresentação: SVS_t -> Value Conversion Rate; SUR_h -> Post-Value Retention_h; RSVA_mh -> Value-Qualified Retention_h",
                *build_metric_definitions_markdown(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "interactive_html": str(out_html),
        "summary_json": str(out_json),
        "readme_md": str(out_md),
    }


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def extend_consolidated_status(
    cfg: Stage4Config,
    snapshot_ts: pd.Timestamp,
    strict_mode_effective: str,
    strict_events: List[str],
    taxonomy_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    best_metric: Dict[str, Any],
    chart_bundle: Dict[str, Any],
    metric_review_path: str,
    metric_initial_package_paths: Dict[str, str],
) -> None:
    cpath = cfg.output_dir / "consolidated_status.json"
    consolidated = read_json(cpath)
    if not consolidated:
        consolidated = {}

    latest = decomposition_df[decomposition_df["is_decision_month"] == True].tail(1)
    if latest.empty:
        latest = decomposition_df.tail(1)
    latest_row = latest.iloc[0].to_dict() if not latest.empty else {}

    taxonomy_version = taxonomy_df["event_taxonomy_version"].iloc[0] if not taxonomy_df.empty else None
    model_hash = taxonomy_df["model_hash"].iloc[0] if not taxonomy_df.empty else None
    effective_date = taxonomy_df["effective_date"].iloc[0] if not taxonomy_df.empty else None

    consolidated["snapshot_ts"] = str(snapshot_ts)
    consolidated["monthly_metric_contract"] = {
        "population_primary": cfg.population_primary,
        "strict_value_definition": "download_aula OR download_plano_aula",
        "segment_definition": "heavy_users = heavy_user_flag da etapa 01 (heavy_score_fast_v1: PCA-1 + threshold holdout), aplicado por unique_id na base mensal",
        "exclude_incomplete_month": cfg.exclude_incomplete_month,
        "north_star_metric": str(best_metric.get("best_metric") or "rsva_m1"),
        "north_star_metric_display": metric_display_label(str(best_metric.get("best_metric") or "rsva_m1")),
        "decomposition_identity": "para cada horizonte h: RSVA_mh = SVS_t * SUR_h",
        "decomposition_identity_display": "para cada horizonte h: Value-Qualified Retention_h = Value Conversion Rate * Post-Value Retention_h",
        "confidence_level": cfg.confidence_level,
    }
    consolidated["north_star_monthly"] = {
        "metric": str(best_metric.get("best_metric") or "rsva_m1"),
        "latest_month": latest_row.get("month"),
        "latest_value": latest_row.get(best_metric.get("best_metric") or "rsva_m1"),
        "retention_m1": latest_row.get("retention_m1"),
        "svs_t": latest_row.get("svs_t"),
        "sur_t": latest_row.get("sur_t"),
    }
    consolidated["taxonomy_versioning"] = {
        "taxonomy_mode_requested": cfg.value_taxonomy_mode,
        "taxonomy_mode_effective": strict_mode_effective,
        "event_taxonomy_version": taxonomy_version,
        "effective_date": effective_date,
        "model_hash": model_hash,
        "strict_event_count": int(len(strict_events)),
        "strict_events": strict_events,
        "auxiliary_only": True,
    }
    consolidated["metric_charts"] = chart_bundle
    consolidated["monthly_kpi_paths"] = {
        "kpi_monthly_panel_csv": str(cfg.output_dir / "kpi_monthly_panel.csv"),
        "kpi_monthly_long_csv": str(cfg.output_dir / "kpi_monthly_long.csv"),
        "kpi_decomposition_panel_csv": str(cfg.output_dir / "kpi_decomposition_panel.csv"),
        "kpi_event_family_panel_csv": str(cfg.output_dir / "kpi_event_family_panel.csv"),
        "kpi_subject_download_quality_csv": str(cfg.output_dir / "kpi_subject_download_quality.csv"),
        "kpi_subject_download_top_overall_csv": str(cfg.output_dir / "kpi_subject_download_top_overall.csv"),
        "kpi_subject_download_top_monthly_csv": str(cfg.output_dir / "kpi_subject_download_top_monthly.csv"),
        "kpi_rsva_segment_monthly_csv": str(cfg.output_dir / "kpi_rsva_segment_monthly.csv"),
        "kpi_rsva_linear_models_csv": str(cfg.output_dir / "kpi_rsva_linear_models.csv"),
        "kpi_rsva_linear_fit_csv": str(cfg.output_dir / "kpi_rsva_linear_fit.csv"),
        "kpi_event_impacts_on_metrics_csv": str(cfg.output_dir / "kpi_event_impacts_on_metrics.csv"),
        "kpi_event_class_impacts_on_metrics_csv": str(cfg.output_dir / "kpi_event_class_impacts_on_metrics.csv"),
        "kpi_rsva_diagnostics_table_csv": str(cfg.output_dir / "kpi_rsva_diagnostics_table.csv"),
        "event_taxonomy_learned_csv": str(cfg.output_dir / "event_taxonomy_learned.csv"),
        "kpi_segment_drop_summary_csv": str(cfg.output_dir / "kpi_segment_drop_summary.csv"),
        "kpi_segment_drop_latest_12_csv": str(cfg.output_dir / "kpi_segment_drop_latest_12.csv"),
        "kpi_metric_dictionary_json": str(cfg.output_dir / "kpi_metric_dictionary.json"),
        "kpi_metric_selection_details_json": str(cfg.output_dir / "kpi_metric_selection_details.json"),
        "kpi_data_usage_audit_json": str(cfg.output_dir / "kpi_data_usage_audit.json"),
        "kpi_rsva_retention_m2_comparison_json": str(cfg.output_dir / "kpi_rsva_retention_m2_comparison.json"),
        "kpi_horizon_comparison_csv": str(cfg.output_dir / "kpi_horizon_comparison.csv"),
        "kpi_horizon_comparison_json": str(cfg.output_dir / "kpi_horizon_comparison.json"),
        "kpi_event_impacts_summary_json": str(cfg.output_dir / "kpi_event_impacts_summary.json"),
        "kpi_heavy_user_types_csv": str(cfg.output_dir / "kpi_heavy_user_types.csv"),
        "kpi_heavy_user_types_summary_json": str(cfg.output_dir / "kpi_heavy_user_types_summary.json"),
        "kpi_metric_uncertainty_bands_csv": str(cfg.output_dir / "kpi_metric_uncertainty_bands.csv"),
        "kpi_strict_cohort_survival_hazard_csv": str(cfg.output_dir / "kpi_strict_cohort_survival_hazard.csv"),
        "kpi_strict_cohort_curve_by_cohort_csv": str(cfg.output_dir / "kpi_strict_cohort_curve_by_cohort.csv"),
        "kpi_strict_cohort_summary_json": str(cfg.output_dir / "kpi_strict_cohort_summary.json"),
        "pipeline_consistency_audit_json": str(cfg.output_dir / "pipeline_consistency_audit.json"),
        "pipeline_consistency_audit_md": str(cfg.output_dir / "reports" / "pipeline_consistency_audit.md"),
        "metric_review_report_md": metric_review_path,
        "analise_inicial_da_metrica_interativa_html": metric_initial_package_paths.get("interactive_html"),
        "analise_inicial_da_metrica_summary_json": metric_initial_package_paths.get("summary_json"),
    }
    consolidated["stage_04_metadata"] = {
        "generated_at_utc": utc_now_iso(),
        "config": asdict(cfg),
    }
    write_json(cpath, consolidated)


def main() -> None:
    setup_logging()
    cfg = build_config(parse_args())
    ensure_required_paths(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting etapa_04_metricas_mensais | output_dir=%s", cfg.output_dir)
    conn = duckdb.connect(database=":memory:")
    create_views(conn, cfg.data_dir)
    create_population_view(conn, cfg.population_primary)
    snapshot_ts = load_snapshot_ts(conn, cfg.output_dir)
    teacher_df_stage1 = load_teacher_dataset_from_output(cfg.output_dir)

    LOGGER.info("Computing decomposition with strict value fixed to downloads_only")
    decomposition_df, monthly_long = build_monthly_decomposition(
        conn=conn,
        snapshot_ts=snapshot_ts,
        exclude_incomplete_month=cfg.exclude_incomplete_month,
        confidence_level=cfg.confidence_level,
    )

    if decomposition_df.empty:
        raise RuntimeError("Sem dados apos aplicacao da populacao primaria.")

    best_metric_short_window = build_metric_selection(decomposition_df, include_extended_horizons=False)
    best_metric = build_metric_selection(decomposition_df, include_extended_horizons=True)
    m2_comparison = build_rsva_retention_m2_comparison(decomposition_df)
    horizon_comparison_df = build_horizon_comparison(decomposition_df, horizons=[1, 2, 4, 6])
    metric_uncertainty_df = build_metric_uncertainty_panel(
        monthly_long=monthly_long,
        focus_metrics=["rsva_m1", "svs_t", "sur_t", "sur_m2", "sur_m4", "sur_m6", "retention_m1", "retention_m2", "retention_m4", "retention_m6"],
    )
    strict_cohort_hazard_df, strict_cohort_curve_df, strict_cohort_summary = build_strict_cohort_survival_hazard(
        conn=conn,
        snapshot_ts=snapshot_ts,
        exclude_incomplete_month=cfg.exclude_incomplete_month,
        confidence_level=cfg.confidence_level,
        max_horizon_months=12,
    )
    rsva_diag = build_rsva_drop_diagnostics(decomposition_df)
    event_family = build_event_family_segmentation(
        conn=conn,
        snapshot_ts=snapshot_ts,
        exclude_incomplete_month=cfg.exclude_incomplete_month,
    )
    subject_quality, subject_top_overall, subject_top_monthly = build_subject_download_analysis(
        conn=conn,
        snapshot_ts=snapshot_ts,
        exclude_incomplete_month=cfg.exclude_incomplete_month,
        top_n=6,
    )
    user_month_segment_base = build_user_month_segment_base(conn, teacher_df_stage1)
    segment_monthly_rsva = build_segment_monthly_rsva_metrics(
        user_month_base=user_month_segment_base,
        snapshot_ts=snapshot_ts,
        exclude_incomplete_month=cfg.exclude_incomplete_month,
    )
    segment_drop_diag = build_segment_drop_diagnostics(segment_monthly_rsva)
    rsva_linear_models_df, rsva_linear_fit_df = fit_rsva_linear_models(segment_monthly_rsva)
    event_impacts_df = build_event_impacts_on_metrics(
        conn=conn,
        user_month_base=user_month_segment_base,
        snapshot_ts=snapshot_ts,
        exclude_incomplete_month=cfg.exclude_incomplete_month,
        confidence_level=cfg.confidence_level,
    )
    event_class_impacts_df = build_event_class_impacts_on_metrics(
        conn=conn,
        user_month_base=user_month_segment_base,
        snapshot_ts=snapshot_ts,
        exclude_incomplete_month=cfg.exclude_incomplete_month,
        confidence_level=cfg.confidence_level,
    )

    LOGGER.info("Computing auxiliary uplift event analysis")
    taxonomy_df = build_taxonomy_auxiliary(
        conn=conn,
        snapshot_ts=snapshot_ts,
        confidence_level=cfg.confidence_level,
        population_primary=cfg.population_primary,
    )

    data_usage_audit = build_data_usage_audit(conn=conn, cfg=cfg)
    heavy_user_types_df, heavy_user_types_summary = build_heavy_user_type_profiles(
        conn=conn,
        output_dir=cfg.output_dir,
        teacher_df=teacher_df_stage1,
    )
    consistency_audit = build_pipeline_consistency_audit(
        cfg=cfg,
        decomposition_df=decomposition_df,
        user_month_segment_base=user_month_segment_base,
    )
    consistency_report_path = write_pipeline_consistency_report(cfg.output_dir, consistency_audit)

    LOGGER.info("Rendering charts and reports")
    cleanup_legacy_artifacts(cfg.output_dir)
    chart_bundle = generate_charts(
        output_dir=cfg.output_dir,
        monthly_panel=decomposition_df,
        metric_uncertainty_df=metric_uncertainty_df,
        strict_cohort_hazard_df=strict_cohort_hazard_df,
        diag=rsva_diag,
        segment_monthly_df=segment_monthly_rsva,
        segment_drop_diag=segment_drop_diag,
        taxonomy_df=taxonomy_df,
        event_family_df=event_family,
        subject_quality_df=subject_quality,
        subject_top_overall_df=subject_top_overall,
        subject_top_monthly_df=subject_top_monthly,
        rsva_linear_models_df=rsva_linear_models_df,
        rsva_linear_fit_df=rsva_linear_fit_df,
        event_impacts_df=event_impacts_df,
        event_class_impacts_df=event_class_impacts_df,
        horizon_comparison_df=horizon_comparison_df,
        heavy_user_types_df=heavy_user_types_df,
    )

    metric_review_path = write_metric_review_report(
        cfg=cfg,
        monthly_panel=decomposition_df,
        best_metric=best_metric,
        best_metric_short_window=best_metric_short_window,
        m2_comparison=m2_comparison,
        horizon_comparison_df=horizon_comparison_df,
        data_usage_audit=data_usage_audit,
        consistency_audit=consistency_audit,
    )

    top_event = {}
    entrou_event = {}
    if not taxonomy_df.empty:
        tx_sorted = taxonomy_df.sort_values("adjusted_uplift_next_active", ascending=False).reset_index(drop=True)
        tx = tx_sorted.head(1)
        if not tx.empty:
            top_event = tx.iloc[0][["event_type", "adjusted_uplift_next_active", "ci_low", "ci_high"]].to_dict()
        hit = tx_sorted[tx_sorted["event_type"].astype(str) == "entrou_nova_escola"]
        if not hit.empty:
            idx = int(hit.index[0]) + 1
            row = hit.iloc[0]
            entrou_event = {
                "rank_uplift": idx,
                "event_type": str(row.get("event_type")),
                "adjusted_uplift_next_active": row.get("adjusted_uplift_next_active"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
            }

    event_family_recent = event_family.copy()
    if not event_family_recent.empty:
        if "is_decision_month" in event_family_recent.columns:
            event_family_recent = event_family_recent[event_family_recent["is_decision_month"] == True].copy()
        event_family_recent = event_family_recent.sort_values("month").tail(12)

    subject_quality_recent = subject_quality.copy()
    if not subject_quality_recent.empty:
        if "is_decision_month" in subject_quality_recent.columns:
            subject_quality_recent = subject_quality_recent[subject_quality_recent["is_decision_month"] == True].copy()
        subject_quality_recent = subject_quality_recent.sort_values("month").tail(12)

    decomposition_recent = decomposition_df.copy()
    if not decomposition_recent.empty:
        if "is_decision_month" in decomposition_recent.columns:
            decomposition_recent = decomposition_recent[decomposition_recent["is_decision_month"] == True].copy()
        keep_cols = [c for c in ["month", "rsva_m1", "svs_t", "sur_t"] if c in decomposition_recent.columns]
        decomposition_recent = decomposition_recent[keep_cols].sort_values("month").tail(12)

    segment_recent_metrics = segment_monthly_rsva.copy()
    if not segment_recent_metrics.empty:
        if "is_decision_month" in segment_recent_metrics.columns:
            segment_recent_metrics = segment_recent_metrics[segment_recent_metrics["is_decision_month"] == True].copy()
        keep_cols = [c for c in ["month", "segment", "rsva_m1", "svs_t", "sur_t"] if c in segment_recent_metrics.columns]
        segment_recent_metrics = segment_recent_metrics[keep_cols].sort_values(["segment", "month"]).groupby("segment", as_index=False, group_keys=False).tail(12)

    impacts_top_negative = pd.DataFrame()
    if not event_impacts_df.empty:
        impacts_top_negative = event_impacts_df[
            (event_impacts_df["negative_confirmed"] == True)
            & (event_impacts_df["segment"] == "all_users")
            & (event_impacts_df["metric"].isin(["rsva_m1", "svs_t", "sur_t"]))
        ].copy()
        if not impacts_top_negative.empty:
            impacts_top_negative = (
                impacts_top_negative.sort_values(["metric", "effect"], ascending=[True, True]).groupby("metric").head(8)
            )

    class_impacts_top = pd.DataFrame()
    if not event_class_impacts_df.empty:
        cls = event_class_impacts_df[
            (event_class_impacts_df["segment"] == "all_users")
            & (event_class_impacts_df["metric"].isin(["rsva_m1", "svs_t", "sur_t"]))
        ].copy()
        if not cls.empty:
            neg = cls[cls["negative_confirmed"] == True].sort_values(["metric", "effect"], ascending=[True, True]).groupby("metric").head(5)
            pos = cls[cls["positive_confirmed"] == True].sort_values(["metric", "effect"], ascending=[True, False]).groupby("metric").head(3)
            class_impacts_top = pd.concat([neg, pos], ignore_index=True).sort_values(["metric", "effect"], ascending=[True, True])

    focus_event_names = [
        "prova_salva",
        "visualizacao_prova",
        "visualizacao_prova_aprendizap",
        "botao_baixar_conquista_completada",
        "fechar_conquista_obtida",
        "click_subaba_concluidas",
    ]
    focus_event_impacts = pd.DataFrame()
    if not event_impacts_df.empty:
        focus_event_impacts = event_impacts_df[
            (event_impacts_df["segment"] == "all_users")
            & (event_impacts_df["metric"].isin(["rsva_m1", "svs_t", "sur_t"]))
            & (event_impacts_df["event_type"].astype(str).isin(focus_event_names))
        ].copy()
        if not focus_event_impacts.empty:
            focus_event_impacts["event_explanation"] = focus_event_impacts["event_type"].astype(str).apply(describe_event_type)
            focus_event_impacts = focus_event_impacts.sort_values(["event_type", "metric"])

    summary_payload = {
        "interactive_html": str(cfg.output_dir / "reports" / "analise_inicial_da_metrica_interativa.html"),
        "reports_dir": str(cfg.output_dir / "reports"),
        "snapshot_ts": str(snapshot_ts),
        "population_primary": cfg.population_primary,
        "strict_value_definition": "strict_value = (download_aula OR download_plano_aula)",
        "strict_value_events": STRICT_DOWNLOAD_EVENTS,
        "metric_display_terminology": {
            "SVS_t": "Value Conversion Rate",
            "SUR_h": "Post-Value Retention_h",
            "RSVA_mh": "Value-Qualified Retention_h",
        },
        "metric_calculation_definitions": METRIC_DEFINITION_ROWS,
        "taxonomy_auxiliary_note": "taxonomia por uplift e análise auxiliar de eventos; não define strict value dos KPIs.",
        "calendar_adjustment_scope": "calendario_escolar_uf_rede.csv usado em ajustes por mês, UF e jornada escolar; rede fixada em 'todas' por indisponibilidade de rede no cadastro de usuários.",
        "taxonomy_top_event": top_event,
        "taxonomy_entrou_nova_escola": entrou_event,
        "best_metric": best_metric.get("best_metric"),
        "best_metric_short_window": best_metric_short_window.get("best_metric"),
        "metric_recommendation_product": "rsva_m1",
        "metric_recommendation_changed_with_horizons": False,
        "metric_recommendation_note": "Com horizontes estendidos, o melhor indicador técnico pode mudar; para decisão de produto o recomendado permanece Value-Qualified Retention (RSVA_m1), com Value Conversion Rate (SVS_t) e Post-Value Retention (SUR_h/SUR_t) como diagnóstico.",
        "segment_definition_consistency": "heavy_users = heavy_user_flag herdado da etapa 01 (heavy_score_fast_v1: PCA-1 + threshold holdout), mesma definição reutilizada nas etapas 02/03/04.",
        "selection_criterion": best_metric.get("selection_criterion"),
        "decision_eligibility_rule": best_metric.get("decision_eligibility_rule"),
        "metric_selection_compare": {
            "best_metric": best_metric.get("best_metric"),
            "short_window_best_metric": best_metric_short_window.get("best_metric"),
            "changed": bool(str(best_metric.get("best_metric")) != str(best_metric_short_window.get("best_metric"))),
            "selection_mode": best_metric.get("selection_mode"),
            "short_window_selection_mode": best_metric_short_window.get("selection_mode"),
        },
        "rsva_retention_m2": m2_comparison,
        "horizon_comparison": horizon_comparison_df.to_dict(orient="records") if not horizon_comparison_df.empty else [],
        "rsva_drop_diagnostics": rsva_diag,
        "segment_drop_diagnostics": segment_drop_diag,
        "decomposition_recent": decomposition_recent.to_dict(orient="records") if not decomposition_recent.empty else [],
        "segment_recent_metrics": segment_recent_metrics.to_dict(orient="records") if not segment_recent_metrics.empty else [],
        "metric_uncertainty_recent": (
            metric_uncertainty_df[
                metric_uncertainty_df["metric"].astype(str).isin(["rsva_m1", "svs_t", "sur_t", "sur_m2", "sur_m4", "sur_m6"])
            ]
            .sort_values(["metric", "month"])
            .groupby("metric", as_index=False, group_keys=False)
            .tail(12)
            .to_dict(orient="records")
            if not metric_uncertainty_df.empty
            else []
        ),
        "strict_cohort_hazard_curve": strict_cohort_hazard_df.to_dict(orient="records") if not strict_cohort_hazard_df.empty else [],
        "strict_cohort_curve_by_cohort": (
            strict_cohort_curve_df.sort_values(["cohort_month", "horizon_m"]).to_dict(orient="records")
            if not strict_cohort_curve_df.empty
            else []
        ),
        "strict_cohort_summary": strict_cohort_summary,
        "event_family_recent": event_family_recent.to_dict(orient="records") if not event_family_recent.empty else [],
        "subject_quality_recent": subject_quality_recent.to_dict(orient="records") if not subject_quality_recent.empty else [],
        "subject_top_overall": subject_top_overall.to_dict(orient="records") if not subject_top_overall.empty else [],
        "rsva_linear_models": rsva_linear_models_df.to_dict(orient="records") if not rsva_linear_models_df.empty else [],
        "heavy_user_types_reliable": (
            heavy_user_types_df[heavy_user_types_df["reliable_heavy_type"] == True].to_dict(orient="records")
            if (not heavy_user_types_df.empty and "reliable_heavy_type" in heavy_user_types_df.columns)
            else []
        ),
        "heavy_user_types_summary": heavy_user_types_summary,
        "pipeline_consistency_audit": consistency_audit,
        "pipeline_consistency_report_md": consistency_report_path,
        "event_impacts_top_negative": impacts_top_negative.to_dict(orient="records") if not impacts_top_negative.empty else [],
        "event_class_impacts_top": class_impacts_top.to_dict(orient="records") if not class_impacts_top.empty else [],
        "focus_event_impacts": focus_event_impacts.to_dict(orient="records") if not focus_event_impacts.empty else [],
    }

    metric_initial_package_paths = write_metric_initial_analysis_package(
        cfg=cfg,
        summary_payload=summary_payload,
        chart_bundle=chart_bundle,
        best_metric=best_metric,
    )

    save_csv(cfg.output_dir / "kpi_monthly_panel.csv", decomposition_df.sort_values("month"))
    save_csv(cfg.output_dir / "kpi_monthly_long.csv", monthly_long.sort_values(["metric", "month"]))
    save_csv(cfg.output_dir / "kpi_decomposition_panel.csv", decomposition_df.sort_values("month"))
    save_csv(cfg.output_dir / "kpi_event_family_panel.csv", event_family.sort_values("month"))
    save_csv(cfg.output_dir / "kpi_subject_download_quality.csv", subject_quality.sort_values("month"))
    save_csv(cfg.output_dir / "kpi_subject_download_top_overall.csv", subject_top_overall.sort_values("download_events", ascending=False))
    save_csv(cfg.output_dir / "kpi_subject_download_top_monthly.csv", subject_top_monthly.sort_values(["month", "download_events"], ascending=[True, False]))
    save_csv(cfg.output_dir / "kpi_rsva_diagnostics_table.csv", pd.DataFrame(rsva_diag.get("diagnostics_table", [])))
    save_csv(cfg.output_dir / "kpi_rsva_segment_monthly.csv", segment_monthly_rsva.sort_values(["month", "segment"]))
    save_csv(cfg.output_dir / "kpi_segment_drop_summary.csv", pd.DataFrame(segment_drop_diag.get("drop_summary_by_segment", [])))
    save_csv(cfg.output_dir / "kpi_segment_drop_latest_12.csv", pd.DataFrame(segment_drop_diag.get("latest_12_diagnostics_by_segment", [])))
    save_csv(cfg.output_dir / "kpi_rsva_linear_models.csv", rsva_linear_models_df)
    save_csv(cfg.output_dir / "kpi_rsva_linear_fit.csv", rsva_linear_fit_df.sort_values(["month", "segment"]))
    save_csv(cfg.output_dir / "kpi_event_impacts_on_metrics.csv", event_impacts_df.sort_values(["segment", "metric", "effect"]))
    save_csv(cfg.output_dir / "kpi_event_class_impacts_on_metrics.csv", event_class_impacts_df.sort_values(["segment", "metric", "effect"]))
    save_csv(cfg.output_dir / "event_taxonomy_learned.csv", taxonomy_df)
    save_csv(cfg.output_dir / "kpi_horizon_comparison.csv", horizon_comparison_df)
    save_csv(cfg.output_dir / "kpi_heavy_user_types.csv", heavy_user_types_df)
    save_csv(cfg.output_dir / "kpi_metric_uncertainty_bands.csv", metric_uncertainty_df.sort_values(["metric", "month"]))
    save_csv(cfg.output_dir / "kpi_strict_cohort_survival_hazard.csv", strict_cohort_hazard_df.sort_values("horizon_m"))
    save_csv(cfg.output_dir / "kpi_strict_cohort_curve_by_cohort.csv", strict_cohort_curve_df.sort_values(["cohort_month", "horizon_m"]))

    write_json(cfg.output_dir / "kpi_metric_dictionary.json", build_metric_dictionary())
    write_json(cfg.output_dir / "kpi_metric_selection_details.json", best_metric)
    write_json(cfg.output_dir / "kpi_data_usage_audit.json", data_usage_audit)
    write_json(cfg.output_dir / "kpi_heavy_user_types_summary.json", heavy_user_types_summary)
    write_json(cfg.output_dir / "kpi_strict_cohort_summary.json", strict_cohort_summary if isinstance(strict_cohort_summary, dict) else {})
    write_json(cfg.output_dir / "pipeline_consistency_audit.json", consistency_audit)
    write_json(cfg.output_dir / "kpi_rsva_retention_m2_comparison.json", m2_comparison)
    write_json(
        cfg.output_dir / "kpi_horizon_comparison.json",
        {
            "rows": horizon_comparison_df.to_dict(orient="records") if not horizon_comparison_df.empty else [],
            "metric_selection_compare": {
                "best_metric": best_metric.get("best_metric"),
                "short_window_best_metric": best_metric_short_window.get("best_metric"),
                "changed": bool(str(best_metric.get("best_metric")) != str(best_metric_short_window.get("best_metric"))),
            },
        },
    )
    write_json(
        cfg.output_dir / "kpi_event_impacts_summary.json",
        {
            "negative_confirmed_rows": int((event_impacts_df["negative_confirmed"] == True).sum()) if not event_impacts_df.empty else 0,
            "all_users_negative_by_metric": (
                event_impacts_df[(event_impacts_df["segment"] == "all_users") & (event_impacts_df["negative_confirmed"] == True)]
                .groupby("metric", as_index=False)["event_type"]
                .count()
                .rename(columns={"event_type": "negative_events"})
                .to_dict(orient="records")
                if not event_impacts_df.empty
                else []
            ),
        },
    )

    extend_consolidated_status(
        cfg=cfg,
        snapshot_ts=snapshot_ts,
        strict_mode_effective="downloads_only",
        strict_events=STRICT_DOWNLOAD_EVENTS,
        taxonomy_df=taxonomy_df,
        decomposition_df=decomposition_df,
        best_metric=best_metric,
        chart_bundle=chart_bundle,
        metric_review_path=metric_review_path,
        metric_initial_package_paths=metric_initial_package_paths,
    )

    LOGGER.info("Etapa 04 finished successfully.")


if __name__ == "__main__":
    main()
