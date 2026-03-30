from __future__ import annotations

"""
Etapa 01 - Base analítica rigorosa (sem alvo de inatividade).

Objetivo:
- construir uma base por professor com transformações auditáveis e reprodutíveis;
- gerar métricas iniciais, hipóteses de adoção/uso e artefatos para relatório;
- salvar um consolidado técnico em JSON/Markdown.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("etapa_01_base")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")

CLUSTER_FEATURE_CANDIDATES: List[str] = [
    "aula_event_share",
    "prova_event_share",
    "plano_event_share",
    "download_event_share",
    "visualizacao_event_share",
    "ia_event_share",
    "desktop_event_share",
    "mobile_event_share",
    "tablet_event_share",
    "avg_session_min",
    "time_to_first_value_hours",
    "unique_lessons_count",
]

CLUSTER_INTENSITY_CANDIDATES: List[str] = [
    "interaction_count",
    "session_count",
    "total_session_min",
    "aula_event_count",
    "download_event_count",
    "visualizacao_event_count",
    "unique_lessons_count",
]

FAST_HEAVY_FEATURES: List[str] = [
    "interaction_count",
    "session_count",
    "total_session_min",
    "active_days",
    "recency_days",
]

FAST_HEAVY_THRESHOLD_QUANTILES: List[float] = [0.85, 0.88, 0.90, 0.92, 0.95]


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    random_seed: int = 42
    conversion_days: int = 30
    alpha: float = 0.05
    min_segment_n: int = 200
    max_cluster_sample: int = 50_000
    teacher_dataset_sample_rows: int = 10_000


@dataclass
class HypothesisResult:
    hypothesis_id: str
    statement: str
    status: str
    evidence: str
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    n_obs: Optional[int] = None
    assumptions: str = ""
    caveats: str = ""
    what_was_tested: str = ""
    statistical_test: str = ""
    null_hypothesis: str = ""
    decision_rule: str = ""
    alpha: float = 0.05
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": out_dir,
        "parquet": out_dir / "parquet",
        "csv": out_dir / "csv",
        "reports": out_dir / "reports",
        "excel": out_dir / "excel",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 01: base analítica sem alvo de inatividade.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--conversion-days", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-segment-n", type=int, default=200)
    parser.add_argument("--max-cluster-sample", type=int, default=50_000)
    parser.add_argument("--teacher-dataset-sample-rows", type=int, default=10_000)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir if args.data_dir is not None else base_dir / "base_aprendizap").resolve()
    output_dir = (args.output_dir if args.output_dir is not None else base_dir / "analysis_output").resolve()
    return PipelineConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        random_seed=int(args.random_seed),
        conversion_days=int(args.conversion_days),
        alpha=float(args.alpha),
        min_segment_n=int(args.min_segment_n),
        max_cluster_sample=int(args.max_cluster_sample),
        teacher_dataset_sample_rows=int(args.teacher_dataset_sample_rows),
    )


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
    required = [
        "dim_teachers.csv",
        "fct_teachers_entries.csv",
        "fct_teachers_contents_interactions.csv",
        "stg_lessons.csv",
        "stg_formation.csv",
        "stg_mari_ia_conversation.csv",
        "stg_mari_ia_reports.csv",
        "fct_mari_ia_eventos_isso_ajudou.csv",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes em {data_dir}: {', '.join(sorted(missing))}")

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


def build_teacher_dataset(conn: duckdb.DuckDBPyConnection, conversion_days: int) -> pd.DataFrame:
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
            COUNT(DISTINCT CAST(data_inicio AS DATE)) AS active_days,
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
            SUM(CASE WHEN user_agent_device_type IS NULL OR trim(coalesce(user_agent_device_type, '')) = '' THEN 1 ELSE 0 END) AS unknown_device_events
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
        COALESCE(i.active_days, 0) AS active_days,
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
            WHEN i.first_value_ts IS NOT NULL THEN 1 ELSE 0
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
        )) / 86400.0 AS days_since_last_activity
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
        "active_days",
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
        "recency_days",
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
    teacher_df["utm_group"] = teacher_df["utm_origin"].apply(normalize_utm)
    teacher_df["estado_group"] = teacher_df["estado"].fillna("missing").replace("", "missing")
    teacher_df["recency_days"] = pd.to_numeric(teacher_df["days_since_last_activity"], errors="coerce")
    teacher_df.loc[teacher_df["recency_days"] < 0, "recency_days"] = np.nan
    teacher_df["heavy_user_flag"] = 0
    teacher_df["heavy_cluster_flag"] = 0
    teacher_df["behavior_cluster_id"] = -1
    teacher_df["engagement_intensity_score"] = np.nan
    teacher_df["cluster_fit_population"] = "inactive_or_unassigned"
    teacher_df["active_user_heavy_window_flag"] = 0
    teacher_df["heavy_score_pca1"] = np.nan
    teacher_df["heavy_threshold_quantile"] = np.nan
    teacher_df["heavy_threshold_value"] = np.nan

    return teacher_df


def compute_table_inventory(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    mapping = {
        "dim_teachers": "unique_id",
        "entries": "unique_id",
        "interactions": "unique_id",
        "lessons": "id_aula",
        "formation": "unique_id_aprendizap",
        "mari_conv": "id_mari",
        "mari_reports": "id_mari",
        "mari_help": "user_id",
    }
    rows: List[Dict[str, Any]] = []
    for table, key in mapping.items():
        row_count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        distinct_key = int(conn.execute(f"SELECT COUNT(DISTINCT {key}) FROM {table}").fetchone()[0])
        rows.append(
            {
                "table": table,
                "row_count": row_count,
                "key_column": key,
                "distinct_key_count": distinct_key,
            }
        )
    return pd.DataFrame(rows)


def compute_join_coverage(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    joins = [
        (
            "teacher_entries",
            "SELECT COUNT(DISTINCT unique_id) FROM entries",
            "SELECT COUNT(DISTINCT e.unique_id) FROM entries e INNER JOIN dim_teachers d USING(unique_id)",
        ),
        (
            "teacher_interactions",
            "SELECT COUNT(DISTINCT unique_id) FROM interactions",
            "SELECT COUNT(DISTINCT i.unique_id) FROM interactions i INNER JOIN dim_teachers d USING(unique_id)",
        ),
        (
            "interaction_lessons",
            "SELECT COUNT(DISTINCT id_aula) FROM interactions WHERE id_aula IS NOT NULL",
            "SELECT COUNT(DISTINCT i.id_aula) FROM interactions i INNER JOIN lessons l ON CAST(i.id_aula AS VARCHAR)=CAST(l.id_aula AS VARCHAR)",
        ),
        (
            "teacher_formation",
            "SELECT COUNT(DISTINCT unique_id_aprendizap) FROM formation",
            "SELECT COUNT(DISTINCT f.unique_id_aprendizap) FROM formation f INNER JOIN dim_teachers d ON f.unique_id_aprendizap=d.unique_id",
        ),
        (
            "teacher_mari_conv",
            "SELECT COUNT(DISTINCT unique_id_aprendizap) FROM mari_conv",
            "SELECT COUNT(DISTINCT m.unique_id_aprendizap) FROM mari_conv m INNER JOIN dim_teachers d ON m.unique_id_aprendizap=d.unique_id",
        ),
        (
            "teacher_mari_help",
            "SELECT COUNT(DISTINCT user_id) FROM mari_help",
            "SELECT COUNT(DISTINCT h.user_id) FROM mari_help h INNER JOIN dim_teachers d ON h.user_id=d.unique_id",
        ),
    ]

    for join_name, source_sql, matched_sql in joins:
        source_distinct = int(conn.execute(source_sql).fetchone()[0] or 0)
        matched_distinct = int(conn.execute(matched_sql).fetchone()[0] or 0)
        coverage = float(matched_distinct / source_distinct) if source_distinct > 0 else np.nan
        rows.append(
            {
                "join_name": join_name,
                "source_distinct": source_distinct,
                "matched_distinct": matched_distinct,
                "coverage": coverage,
            }
        )

    return pd.DataFrame(rows)


def compute_identity_coverage(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    WITH base AS (
      SELECT
        (SELECT COUNT(DISTINCT unique_id) FROM dim_teachers) AS teachers,
        (SELECT COUNT(DISTINCT unique_id) FROM entries) AS entries_ids,
        (SELECT COUNT(DISTINCT unique_id) FROM interactions) AS interactions_ids,
        (SELECT COUNT(DISTINCT unique_id_aprendizap) FROM mari_conv) AS mari_conv_ids,
        (SELECT COUNT(DISTINCT user_id) FROM mari_help) AS mari_help_ids,
        (SELECT COUNT(DISTINCT e.unique_id) FROM entries e INNER JOIN dim_teachers d USING(unique_id)) AS entries_ids_in_teachers,
        (SELECT COUNT(DISTINCT i.unique_id) FROM interactions i INNER JOIN dim_teachers d USING(unique_id)) AS interactions_ids_in_teachers,
        (SELECT COUNT(DISTINCT m.unique_id_aprendizap) FROM mari_conv m INNER JOIN dim_teachers d ON m.unique_id_aprendizap=d.unique_id) AS mari_conv_ids_in_teachers,
        (SELECT COUNT(DISTINCT h.user_id) FROM mari_help h INNER JOIN dim_teachers d ON h.user_id=d.unique_id) AS mari_help_ids_in_teachers
    )
    SELECT * FROM base
    """
    df = conn.execute(sql).fetchdf()
    if df.empty:
        return df

    row = df.iloc[0].to_dict()
    records = []
    for source, total_key, in_key in [
        ("entries", "entries_ids", "entries_ids_in_teachers"),
        ("interactions", "interactions_ids", "interactions_ids_in_teachers"),
        ("mari_conv", "mari_conv_ids", "mari_conv_ids_in_teachers"),
        ("mari_help", "mari_help_ids", "mari_help_ids_in_teachers"),
    ]:
        total = int(row.get(total_key, 0) or 0)
        matched = int(row.get(in_key, 0) or 0)
        teachers = int(row.get("teachers", 0) or 0)
        records.append(
            {
                "source": source,
                "source_distinct_ids": total,
                "matched_teacher_ids": matched,
                "coverage_within_source": float(matched / total) if total > 0 else np.nan,
                "coverage_within_teachers": float(matched / teachers) if teachers > 0 else np.nan,
            }
        )
    return pd.DataFrame(records)


def compute_consistency_checks(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    checks: List[Dict[str, Any]] = []

    def push(check_name: str, metric_value: float, expected: str, details: str, status: str) -> None:
        checks.append(
            {
                "check_name": check_name,
                "metric_value": float(metric_value),
                "expected": expected,
                "details": details,
                "status": status,
            }
        )

    negative_duration = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE data_inicio IS NOT NULL AND data_fim IS NOT NULL AND data_fim < data_inicio"
    ).fetchone()[0]
    push(
        "entries_negative_duration",
        negative_duration,
        "expect_zero",
        "Sessões com data_fim antes de data_inicio.",
        "pass" if negative_duration == 0 else "fail",
    )

    short_sessions = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE data_inicio IS NOT NULL AND data_fim IS NOT NULL AND (epoch(data_fim)-epoch(data_inicio)) <= 5"
    ).fetchone()[0]
    push(
        "entries_zero_or_short_seconds",
        short_sessions,
        "monitor",
        "Sessões <= 5s (potencial ping técnico).",
        "warning" if short_sessions > 0 else "pass",
    )

    interactions_before_entry = conn.execute(
        """
        SELECT COUNT(*)
        FROM interactions i
        INNER JOIN dim_teachers d USING(unique_id)
        WHERE i.data_inicio IS NOT NULL
          AND d.data_entrada IS NOT NULL
          AND i.data_inicio < d.data_entrada
        """
    ).fetchone()[0]
    push(
        "interactions_before_teacher_entry",
        interactions_before_entry,
        "low",
        "Interação antes de data_entrada para IDs com match.",
        "warning" if interactions_before_entry > 0 else "pass",
    )

    missing_timestamp_rate = conn.execute(
        "SELECT AVG(CASE WHEN data_inicio IS NULL THEN 1.0 ELSE 0.0 END) FROM interactions"
    ).fetchone()[0]
    push(
        "interactions_missing_timestamp_rate",
        missing_timestamp_rate or 0.0,
        "expect_zero",
        "Taxa de interações sem data_inicio.",
        "pass" if (missing_timestamp_rate or 0.0) == 0 else "fail",
    )

    missing_event_type_rate = conn.execute(
        "SELECT AVG(CASE WHEN event_type IS NULL THEN 1.0 ELSE 0.0 END) FROM interactions"
    ).fetchone()[0]
    push(
        "interactions_missing_event_type_rate",
        missing_event_type_rate or 0.0,
        "low_rate",
        "Taxa de interações sem event_type.",
        "pass" if (missing_event_type_rate or 0.0) <= 0.05 else "warning",
    )

    negative_total_students_rate = conn.execute(
        "SELECT AVG(CASE WHEN total_alunos < 0 THEN 1.0 ELSE 0.0 END) FROM dim_teachers"
    ).fetchone()[0]
    push(
        "dim_negative_total_alunos_rate",
        negative_total_students_rate or 0.0,
        "expect_zero",
        "Taxa de total_alunos negativo.",
        "pass" if (negative_total_students_rate or 0.0) == 0 else "fail",
    )

    unmapped_lesson_rate = conn.execute(
        """
        WITH x AS (
          SELECT i.id_aula,
                 CASE WHEN l.id_aula IS NULL THEN 1 ELSE 0 END AS unmapped
          FROM interactions i
          LEFT JOIN lessons l ON CAST(i.id_aula AS VARCHAR)=CAST(l.id_aula AS VARCHAR)
          WHERE i.id_aula IS NOT NULL
        )
        SELECT AVG(unmapped::DOUBLE) FROM x
        """
    ).fetchone()[0]
    push(
        "interaction_id_aula_unmapped_rate",
        unmapped_lesson_rate or 0.0,
        "contextual",
        "Taxa de id_aula sem match em stg_lessons.",
        "info",
    )

    return pd.DataFrame(checks)


def compute_monthly_solution_usage(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    WITH interactions_dim AS (
      SELECT i.*
      FROM interactions i
      INNER JOIN dim_teachers d USING(unique_id)
      WHERE i.data_inicio IS NOT NULL
        AND lower(coalesce(i.user_type,''))='registered'
    ),
    x AS (
      SELECT
        date_trunc('month', data_inicio) AS month,
        SUM(CASE WHEN lower(coalesce(event_type,'')) LIKE '%aula%' THEN 1 ELSE 0 END) AS aula_events,
        SUM(CASE WHEN lower(coalesce(event_type,'')) LIKE '%prova%' THEN 1 ELSE 0 END) AS prova_events,
        COUNT(DISTINCT unique_id) AS active_users
      FROM interactions_dim
      GROUP BY 1
    )
    SELECT * FROM x WHERE month IS NOT NULL ORDER BY month
    """
    return conn.execute(sql).fetchdf()


def compute_users_panel(conn: duckdb.DuckDBPyConnection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users_panel = conn.execute(
        """
        WITH interactions_dim AS (
            SELECT i.* FROM interactions i INNER JOIN dim_teachers d USING(unique_id)
        ),
        new_users AS (
            SELECT date_trunc('month', data_entrada) AS month, COUNT(DISTINCT unique_id)::BIGINT AS new_users
            FROM dim_teachers
            WHERE data_entrada IS NOT NULL
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
            SELECT month FROM inter_reg
        )
        SELECT m.month, n.new_users, i.mau_registered_interactions
        FROM all_months m
        LEFT JOIN new_users n USING(month)
        LEFT JOIN inter_reg i USING(month)
        ORDER BY month
        """
    ).fetchdf()

    retention = conn.execute(
        """
        WITH interactions_dim AS (
            SELECT i.* FROM interactions i INNER JOIN dim_teachers d USING(unique_id)
        ),
        reg_month AS (
            SELECT DISTINCT unique_id, date_trunc('month', data_inicio) AS month
            FROM interactions_dim
            WHERE data_inicio IS NOT NULL AND lower(coalesce(user_type,''))='registered'
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

    return users_panel, retention


def compute_return_gap(conn: duckdb.DuckDBPyConnection, teacher_df: pd.DataFrame) -> Tuple[float, float, float]:
    overall = conn.execute(
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
        SELECT MEDIAN(gap_days) FROM clean
        """
    ).fetchone()[0]

    gap_rows = conn.execute(
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
          SELECT unique_id, (epoch(data_inicio)-epoch(prev_ts))/86400.0 AS gap_days
          FROM ordered
          WHERE prev_ts IS NOT NULL
        )
        SELECT unique_id, gap_days
        FROM gaps
        WHERE gap_days >= 0 AND gap_days <= 365
        """
    ).fetchdf()

    if gap_rows.empty or "unique_id" not in teacher_df.columns:
        return float(overall), float(np.nan), float(np.nan)

    flags = teacher_df[["unique_id"]].copy()
    if "heavy_user_flag" in teacher_df.columns:
        flags["heavy_user_flag"] = pd.to_numeric(teacher_df["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    else:
        flags["heavy_user_flag"] = 0
    flags = flags.drop_duplicates(subset=["unique_id"])
    merged = gap_rows.merge(flags, on="unique_id", how="left")
    merged["profile"] = np.where(pd.to_numeric(merged["heavy_user_flag"], errors="coerce").fillna(0) == 1, "heavy", "base_regular")
    gap_df = (
        merged.groupby("profile", dropna=False)["gap_days"]
        .median()
        .reset_index()
        .rename(columns={"gap_days": "median_gap_days"})
    )

    heavy_median = np.nan
    base_median = np.nan
    if not gap_df.empty:
        x = gap_df[gap_df["profile"] == "heavy"]
        y = gap_df[gap_df["profile"] == "base_regular"]
        if not x.empty:
            heavy_median = float(x.iloc[0]["median_gap_days"])
        if not y.empty:
            base_median = float(y.iloc[0]["median_gap_days"])

    return float(overall), float(heavy_median), float(base_median)


def compute_summary_metrics(
    conn: duckdb.DuckDBPyConnection,
    teacher_df: pd.DataFrame,
    users_panel: pd.DataFrame,
    retention: pd.DataFrame,
) -> Dict[str, Any]:
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
        WHERE data_inicio IS NOT NULL AND data_fim IS NOT NULL
        """
    ).fetchdf().iloc[0]

    return_gap_median_days, heavy_median, base_median = compute_return_gap(conn, teacher_df)

    users_panel = users_panel.copy()
    users_panel["month"] = pd.to_datetime(users_panel["month"], errors="coerce")
    users_panel = users_panel.sort_values("month")

    latest_new_month = None
    latest_new_count = None
    new_non_null = users_panel.dropna(subset=["new_users"]) if not users_panel.empty else pd.DataFrame()
    if not new_non_null.empty:
        lr = new_non_null.iloc[-1]
        latest_new_month = str(pd.to_datetime(lr["month"]).date())
        latest_new_count = int(lr["new_users"])

    cutoff = conn.execute(
        """
        WITH interactions_dim AS (
            SELECT i.* FROM interactions i INNER JOIN dim_teachers d USING(unique_id)
        )
        SELECT
            (SELECT MAX(data_inicio) FROM interactions_dim WHERE lower(coalesce(user_type,''))='registered') AS max_interactions_registered_ts
        """
    ).fetchdf()

    recent_slope = np.nan
    max_inter_ts = pd.NaT
    if not cutoff.empty:
        max_inter_ts = pd.to_datetime(cutoff.iloc[0]["max_interactions_registered_ts"], errors="coerce")

    if pd.notna(max_inter_ts):
        last_complete = (max_inter_ts.to_period("M") - 1).to_timestamp()
        recent = users_panel[
            (users_panel["month"] <= last_complete)
            & (users_panel["mau_registered_interactions"].notna())
        ].tail(6)
        if len(recent) >= 2:
            x = np.arange(len(recent), dtype=float)
            y = recent["mau_registered_interactions"].astype(float).to_numpy()
            recent_slope = float(np.polyfit(x, y, 1)[0])

    retention_recent_avg = np.nan
    if not retention.empty:
        r = retention.copy()
        r["month"] = pd.to_datetime(r["month"], errors="coerce")
        recent_ret = r.dropna(subset=["retention_rate"]).tail(6)
        if len(recent_ret) > 0:
            retention_recent_avg = float(recent_ret["retention_rate"].mean())

    total_sessions = int(sessions["total_sessions"]) if pd.notna(sessions["total_sessions"]) else 0
    short_sessions = int(sessions["le_5s"]) if pd.notna(sessions["le_5s"]) else 0

    return {
        "state_missing_pct": state_missing_pct,
        "utm_missing_pct": utm_missing_pct,
        "short_sessions_le_5s": short_sessions,
        "short_sessions_rate_le_5s": float(short_sessions / total_sessions) if total_sessions > 0 else np.nan,
        "return_gap_median_days": return_gap_median_days,
        "return_gap_heavy_median_days": heavy_median,
        "return_gap_base_median_days": base_median,
        "latest_new_users_month": latest_new_month,
        "latest_new_users_count": latest_new_count,
        "recent_6m_mau_interactions_slope_users_per_month": recent_slope,
        "retention_recent_avg_6m": retention_recent_avg,
    }


def compute_association_tables(conn: duckdb.DuckDBPyConnection, teacher_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = teacher_df.copy()

    state_stats = (
        df.groupby("estado_group", dropna=False)
        .agg(
            teachers=("unique_id", "count"),
            conversion_rate=("converted_within_window", "mean"),
            median_interactions=("interaction_count", "median"),
            median_session_min=("avg_session_min", "median"),
        )
        .reset_index()
        .sort_values("teachers", ascending=False)
    )

    utm_stats = (
        df.groupby("utm_group", dropna=False)
        .agg(
            teachers=("unique_id", "count"),
            conversion_rate=("converted_within_window", "mean"),
            median_interactions=("interaction_count", "median"),
            median_session_min=("avg_session_min", "median"),
        )
        .reset_index()
        .sort_values("teachers", ascending=False)
    )

    geo_rows: List[Dict[str, Any]] = []
    for target in ["converted_within_window", "heavy_user_flag"]:
        tab = pd.crosstab(df["estado_group"], df[target])
        if tab.shape[0] > 1 and tab.shape[1] > 1:
            _, p, _, _ = chi2_contingency(tab)
            geo_rows.append(
                {
                    "association": f"estado_group vs {target}",
                    "method": "chi2 + cramers_v",
                    "effect_size": cramers_v_from_table(tab),
                    "p_value": float(p),
                }
            )

    for num in ["interaction_count", "session_count", "total_alunos", "time_to_first_value_hours"]:
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
            geo_rows.append(
                {
                    "association": f"estado_group vs {num}",
                    "method": "kruskal + eta2",
                    "effect_size": float(eta2) if pd.notna(eta2) else np.nan,
                    "p_value": float(p),
                }
            )

    geo_associations = pd.DataFrame(geo_rows)

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
    top_corr_pairs = pd.DataFrame(pairs).sort_values("abs_spearman", ascending=False).head(30)

    cat_cols = ["estado_group", "utm_group", "primary_device", "currentstage", "activity_tier"]
    cat_pairs: List[Dict[str, Any]] = []
    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i + 1 :]:
            tab = pd.crosstab(df[c1], df[c2])
            if tab.shape[0] > 1 and tab.shape[1] > 1:
                cat_pairs.append({"var1": c1, "var2": c2, "cramers_v": cramers_v_from_table(tab)})
    cat_corr_pairs = pd.DataFrame(cat_pairs).sort_values("cramers_v", ascending=False)

    journey = conn.execute(
        """
        WITH agg AS (
            SELECT
                unique_id,
                MIN(CASE WHEN lower(coalesce(event_type,'')) LIKE '%aula%' THEN data_inicio ELSE NULL END) AS first_aula_ts,
                MIN(CASE WHEN lower(coalesce(event_type,'')) LIKE '%prova%' THEN data_inicio ELSE NULL END) AS first_prova_ts
            FROM interactions
            GROUP BY unique_id
        )
        SELECT * FROM agg
        """
    ).fetchdf()

    if journey.empty:
        journey_counts = pd.DataFrame(columns=["path", "teachers"])
    else:
        journey["first_aula_ts"] = pd.to_datetime(journey["first_aula_ts"], errors="coerce")
        journey["first_prova_ts"] = pd.to_datetime(journey["first_prova_ts"], errors="coerce")
        both = journey[journey["first_aula_ts"].notna() & journey["first_prova_ts"].notna()].copy()
        both["lag_days_prova_minus_aula"] = (both["first_prova_ts"] - both["first_aula_ts"]).dt.total_seconds() / 86400.0
        both["path"] = np.select(
            [both["lag_days_prova_minus_aula"] > 0, both["lag_days_prova_minus_aula"] < 0],
            ["aula_then_prova", "prova_then_aula"],
            default="same_day",
        )
        journey_counts = both["path"].value_counts(dropna=False).reset_index()
        journey_counts.columns = ["path", "teachers"]

    heavy_summary = (
        df.groupby("heavy_user_flag", dropna=False)
        .agg(
            teachers=("unique_id", "count"),
            avg_interactions=("interaction_count", "mean"),
            median_interactions=("interaction_count", "median"),
            conversion_rate=("converted_within_window", "mean"),
            avg_days_since_last_activity=("days_since_last_activity", "mean"),
        )
        .reset_index()
    )

    return {
        "state_stats": state_stats,
        "utm_stats": utm_stats,
        "geo_associations": geo_associations,
        "top_corr_pairs": top_corr_pairs,
        "cat_corr_pairs": cat_corr_pairs,
        "journey_path_counts": journey_counts,
        "heavy_summary": heavy_summary,
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
        return {"rows": 0, "feedback_repeat_users": 0, "source_counts": {}}

    hot = pd.concat(frames, ignore_index=True, sort=False)
    source_counts = hot["source_file"].value_counts(dropna=False).to_dict()

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
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
    }


def build_hypothesis_dataframe(
    teacher_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    hotjar: Dict[str, Any],
    alpha: float,
    min_segment_n: int,
    random_seed: int,
    max_cluster_sample: int,
    cluster_artifacts_summary: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    out: List[HypothesisResult] = []

    def push(item: HypothesisResult) -> None:
        if not item.interpretation:
            if item.status == "validated":
                item.interpretation = "Evidência compatível com a hipótese, dentro das premissas definidas."
            elif item.status == "rejected":
                item.interpretation = "Evidência na direção oposta da hipótese testada."
            elif item.status == "not_testable":
                item.interpretation = "Hipótese não testável com os dados disponíveis neste recorte."
            else:
                item.interpretation = "Evidência insuficiente para decisão conclusiva."
        out.append(item)

    # H1
    d = teacher_df[["aula_event_count", "prova_event_count"]].fillna(0)
    d = d[(d["aula_event_count"] + d["prova_event_count"]) > 0]
    if len(d) < 50:
        push(HypothesisResult(
            hypothesis_id="H1",
            statement="H1: Existe diferença de uso entre aula e prova.",
            status="inconclusive",
            evidence="Amostra insuficiente para teste pareado.",
            n_obs=len(d),
            what_was_tested="Diferença pareada aula_event_count vs prova_event_count por professor.",
            statistical_test="Wilcoxon signed-rank.",
            null_hypothesis="Mediana(aula - prova) = 0.",
            decision_rule="Validar se p < alpha e mediana(aula-prova) > 0.",
            alpha=alpha,
        ))
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
        push(HypothesisResult(
            hypothesis_id="H1",
            statement="H1: Existe diferença de uso entre aula e prova.",
            status=st,
            evidence=f"median_diff={med:.3f}; p={p:.3g}",
            p_value=float(p),
            effect_size=med,
            n_obs=len(d),
            assumptions="Classificação de evento por event_type.",
            caveats="Não mede causalidade ou preferência isolada de produto.",
            what_was_tested="Diferença pareada aula_event_count vs prova_event_count por professor.",
            statistical_test="Wilcoxon signed-rank.",
            null_hypothesis="Mediana(aula - prova) = 0.",
            decision_rule="Validar se p < alpha e mediana(aula-prova) > 0.",
            alpha=alpha,
        ))

    # H2
    d2 = monthly_df.copy()
    d2 = d2[(d2["prova_events"].fillna(0) > 0) & (d2["aula_events"].fillna(0) > 0)]
    if len(d2) < 6:
        push(HypothesisResult(
            hypothesis_id="H2",
            statement="H2: A sazonalidade difere entre aula e prova.",
            status="inconclusive",
            evidence="Meses insuficientes com uso simultâneo de aula e prova.",
            n_obs=len(d2),
            what_was_tested="Distribuição mensal de aula_events e prova_events.",
            statistical_test="Chi-quadrado de contingência + Cramér's V.",
            null_hypothesis="Distribuição mensal independe do tipo de solução.",
            decision_rule="Validar se p < alpha.",
            alpha=alpha,
        ))
    else:
        cont = np.vstack([d2["aula_events"].to_numpy(), d2["prova_events"].to_numpy()])
        chi2, p, _, _ = chi2_contingency(cont)
        n = cont.sum()
        r, c = cont.shape
        v = np.sqrt(chi2 / (n * max(1, min(r - 1, c - 1)))) if n > 0 else np.nan
        push(HypothesisResult(
            hypothesis_id="H2",
            statement="H2: A sazonalidade difere entre aula e prova.",
            status="validated" if p < alpha else "inconclusive",
            evidence=f"chi2={chi2:.2f}; p={p:.3g}; cramers_v={v:.3f}",
            p_value=float(p),
            effect_size=float(v),
            n_obs=len(d2),
            assumptions="Meses com ambos os tipos de uso são comparáveis no tempo.",
            caveats="Diferença pode misturar sazonalidade e ciclo de produto.",
            what_was_tested="Distribuição mensal de aula_events e prova_events.",
            statistical_test="Chi-quadrado de contingência + Cramér's V.",
            null_hypothesis="Distribuição mensal independe do tipo de solução.",
            decision_rule="Validar se p < alpha.",
            alpha=alpha,
        ))

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

    push(HypothesisResult(
        hypothesis_id="H4",
        statement="H4: Alguns segmentos convertem melhor que outros.",
        status="validated" if sig > 0 else "inconclusive",
        evidence=f"segments_with_signal={sig}",
        p_value=None if min_p is None else float(min_p),
        effect_size=float(max_eff),
        n_obs=len(teacher_df),
        assumptions="Conversão definida por first_value em até conversion_days.",
        caveats="Resultados dependem de missing e cobertura por segmento.",
        what_was_tested="Diferença de taxa de conversão entre segmentos.",
        statistical_test="Chi-quadrado + Cramér's V + filtro de spread prático.",
        null_hypothesis="Conversão independe de segmento.",
        decision_rule="Validar se pelo menos um segmento tiver p<alpha e spread>=3pp.",
        alpha=alpha,
    ))

    # H5 (comparação de device sem alvo de inatividade)
    x5 = teacher_df[["primary_device", "converted_within_window", "interaction_count"]].copy()
    x5 = x5[x5["primary_device"].isin(["mobile", "desktop"])]
    if len(x5) < 500:
        push(HypothesisResult(
            hypothesis_id="H5",
            statement="H5: Mobile e desktop têm comportamento diferente de uso/conversão.",
            status="inconclusive",
            evidence="Amostra insuficiente para comparação mobile vs desktop.",
            n_obs=len(x5),
            what_was_tested="Diferença em conversão e volume de interação por device.",
            statistical_test="Chi-quadrado + Mann-Whitney U.",
            null_hypothesis="Não há diferenças entre os grupos.",
            decision_rule="Validar se pelo menos um teste for significativo.",
            alpha=alpha,
        ))
    else:
        conv_tab = pd.crosstab(x5["primary_device"], x5["converted_within_window"])
        p_conv = 1.0
        v_conv = np.nan
        if conv_tab.shape == (2, 2):
            _, p_conv, _, _ = chi2_contingency(conv_tab.to_numpy())
            v_conv = cramers_v_from_table(conv_tab)
        m = x5[x5["primary_device"] == "mobile"]["interaction_count"].fillna(0)
        d = x5[x5["primary_device"] == "desktop"]["interaction_count"].fillna(0)
        _, p_inter = mannwhitneyu(m, d, alternative="two-sided")
        st = "validated" if (p_conv < alpha or p_inter < alpha) else "inconclusive"
        push(HypothesisResult(
            hypothesis_id="H5",
            statement="H5: Mobile e desktop têm comportamento diferente de uso/conversão.",
            status=st,
            evidence=f"p_conv={p_conv:.3g}; cramers_v_conv={v_conv:.3f}; p_inter={p_inter:.3g}",
            p_value=float(min(p_conv, p_inter)),
            effect_size=float(np.nan_to_num(v_conv)),
            n_obs=len(x5),
            assumptions="Primary device representa contexto dominante de uso.",
            caveats="Atribuição de device pode ser ruidosa quando há troca de dispositivo.",
            what_was_tested="Diferença em conversão e volume de interação por device.",
            statistical_test="Chi-quadrado + Mann-Whitney U.",
            null_hypothesis="Não há diferenças entre os grupos.",
            decision_rule="Validar se pelo menos um teste for significativo.",
            alpha=alpha,
        ))

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

    push(HypothesisResult(
        hypothesis_id="H8",
        statement="H8: Custo de ativação (TTFV) difere por segmento.",
        status="validated" if sig8 > 0 else "inconclusive",
        evidence=f"segments_with_signal={sig8}",
        p_value=None if minp8 is None else float(minp8),
        effect_size=float(best_eta),
        n_obs=int(teacher_df["time_to_first_value_hours"].notna().sum()),
        assumptions="TTFV calculado de data_entrada até first_value_ts.",
        caveats="Não isola qualidade de campanha ou coortes externas.",
        what_was_tested="Diferença de distribuição de TTFV entre segmentos.",
        statistical_test="Kruskal-Wallis + eta2.",
        null_hypothesis="Distribuições de TTFV são iguais entre grupos.",
        decision_rule="Validar se p<alpha e spread de medianas > 1h.",
        alpha=alpha,
    ))

    # H10
    cluster_ctx = cluster_artifacts_summary or {}
    best_k = cluster_ctx.get("best_k")
    best_score = pd.to_numeric(cluster_ctx.get("best_silhouette"), errors="coerce")
    cluster_features = cluster_ctx.get("cluster_feature_cols", [])
    train_sample_n = pd.to_numeric(cluster_ctx.get("cluster_train_sample_n"), errors="coerce")
    if best_k is None or pd.isna(best_score):
        push(HypothesisResult(
            hypothesis_id="H10",
            statement="H10: Existem perfis comportamentais distintos de professores.",
            status="inconclusive",
            evidence="Clusterização indisponível ou sem score válido.",
            what_was_tested="Separação de perfis de uso em espaço multivariado.",
            statistical_test="KMeans + silhouette.",
            null_hypothesis="Não há separação consistente de perfis.",
            decision_rule="Validar se melhor silhouette >= 0.20.",
            alpha=alpha,
        ))
    else:
        st = "validated" if float(best_score) >= 0.2 else "inconclusive"
        feat_txt = ", ".join([str(x) for x in cluster_features]) if isinstance(cluster_features, list) and cluster_features else "n/d"
        n_obs_h10 = int(train_sample_n) if pd.notna(train_sample_n) else int(max_cluster_sample)
        push(HypothesisResult(
            hypothesis_id="H10",
            statement="H10: Existem perfis comportamentais distintos de professores.",
            status=st,
            evidence=f"best_k={int(best_k)}; silhouette={float(best_score):.3f}; features={feat_txt}",
            effect_size=float(best_score),
            n_obs=n_obs_h10,
            assumptions="Vetores de comportamento representam uso de produto.",
            caveats="Resultado depende das features e da escala adotada.",
            what_was_tested="Separação de perfis de uso em espaço multivariado.",
            statistical_test="KMeans + silhouette.",
            null_hypothesis="Não há separação consistente de perfis.",
            decision_rule="Validar se melhor silhouette >= 0.20.",
            alpha=alpha,
        ))

    # H13
    req13 = {"visualizou_metodologia_ativa", "aula_event_count", "plano_event_count"}
    if not req13.issubset(teacher_df.columns):
        push(HypothesisResult(
            hypothesis_id="H13",
            statement="H13: Professores com metodologia ativa vista usam mais aula/plano.",
            status="inconclusive",
            evidence="Colunas obrigatórias ausentes.",
            what_was_tested="Diferença de uso aula+plano entre flag=1 e flag=0.",
            statistical_test="Mann-Whitney U.",
            null_hypothesis="Distribuições de uso são iguais.",
            decision_rule="Validar se p<alpha e diferença de medianas > 0.",
            alpha=alpha,
        ))
    else:
        x = teacher_df[["visualizou_metodologia_ativa", "aula_event_count", "plano_event_count"]].copy()
        x["flag"] = (x["visualizou_metodologia_ativa"].fillna(0) > 0).astype(int)
        x["usage_metric"] = x["aula_event_count"].fillna(0) + x["plano_event_count"].fillna(0)
        g1 = x.loc[x["flag"] == 1, "usage_metric"]
        g0 = x.loc[x["flag"] == 0, "usage_metric"]
        if len(g1) < 30 or len(g0) < 30:
            push(HypothesisResult(
                hypothesis_id="H13",
                statement="H13: Professores com metodologia ativa vista usam mais aula/plano.",
                status="inconclusive",
                evidence="Amostras insuficientes por grupo.",
                n_obs=len(x),
                what_was_tested="Diferença de uso aula+plano entre flag=1 e flag=0.",
                statistical_test="Mann-Whitney U.",
                null_hypothesis="Distribuições de uso são iguais.",
                decision_rule="Validar se p<alpha e diferença de medianas > 0.",
                alpha=alpha,
            ))
        else:
            _, p = mannwhitneyu(g1, g0, alternative="two-sided")
            med = float(np.median(g1) - np.median(g0))
            if p < alpha and med > 0:
                st = "validated"
            elif p < alpha and med < 0:
                st = "rejected"
            else:
                st = "inconclusive"
            push(HypothesisResult(
                hypothesis_id="H13",
                statement="H13: Professores com metodologia ativa vista usam mais aula/plano.",
                status=st,
                evidence=f"median_diff={med:.3f}; p={p:.3g}",
                p_value=float(p),
                effect_size=med,
                n_obs=len(x),
                assumptions="Flag visualizou_metodologia_ativa representa exposição real.",
                caveats="Pode haver auto-seleção entre grupos.",
                what_was_tested="Diferença de uso aula+plano entre flag=1 e flag=0.",
                statistical_test="Mann-Whitney U.",
                null_hypothesis="Distribuições de uso são iguais.",
                decision_rule="Validar se p<alpha e diferença de medianas > 0.",
                alpha=alpha,
            ))

    # H14
    req14 = {"total_alunos", "interaction_count"}
    if not req14.issubset(teacher_df.columns):
        push(HypothesisResult(
            hypothesis_id="H14",
            statement="H14: Número de alunos se relaciona com uso da plataforma.",
            status="inconclusive",
            evidence="Colunas obrigatórias ausentes.",
            what_was_tested="Correlação monotônica total_alunos vs interaction_count.",
            statistical_test="Spearman.",
            null_hypothesis="rho = 0.",
            decision_rule="Validar se p<alpha e |rho|>=0.10.",
            alpha=alpha,
        ))
    else:
        x = teacher_df[["total_alunos", "interaction_count"]].dropna()
        x = x[x["total_alunos"] >= 0]
        if len(x) < 100:
            push(HypothesisResult(
                hypothesis_id="H14",
                statement="H14: Número de alunos se relaciona com uso da plataforma.",
                status="inconclusive",
                evidence="Amostra insuficiente.",
                n_obs=len(x),
                what_was_tested="Correlação monotônica total_alunos vs interaction_count.",
                statistical_test="Spearman.",
                null_hypothesis="rho = 0.",
                decision_rule="Validar se p<alpha e |rho|>=0.10.",
                alpha=alpha,
            ))
        else:
            rho, p = spearmanr(x["total_alunos"], x["interaction_count"])
            st = "validated" if (p < alpha and abs(float(rho)) >= 0.1) else "inconclusive"
            push(HypothesisResult(
                hypothesis_id="H14",
                statement="H14: Número de alunos se relaciona com uso da plataforma.",
                status=st,
                evidence=f"rho={rho:.3f}; p={p:.3g}",
                p_value=float(p),
                effect_size=float(rho),
                n_obs=len(x),
                assumptions="total_alunos comparável entre professores.",
                caveats="Associação não implica mecanismo causal.",
                what_was_tested="Correlação monotônica total_alunos vs interaction_count.",
                statistical_test="Spearman.",
                null_hypothesis="rho = 0.",
                decision_rule="Validar se p<alpha e |rho|>=0.10.",
                alpha=alpha,
            ))

    # H15
    rows = int(hotjar.get("rows", 0) or 0)
    rep = int(hotjar.get("feedback_repeat_users", 0) or 0)
    push(HypothesisResult(
        hypothesis_id="H15",
        statement="H15: Heavy users deixam feedback mais recorrente e detalhado.",
        status="not_testable",
        evidence=f"Hotjar rows={rows}; repeat_hotjar_users={rep}; sem chave de identidade com professores.",
        assumptions="Necessária ponte de identidade entre Hotjar e professores.",
        caveats="Com os dados atuais, apenas leitura descritiva agregada é possível.",
        what_was_tested="Disponibilidade de bridge de identidade para ligar feedback a uso da plataforma.",
        statistical_test="Checagem de cobertura/bridge.",
        null_hypothesis="Não aplicável sem bridge.",
        decision_rule="Não testável sem chave comum entre sistemas.",
        alpha=alpha,
    ))

    return pd.DataFrame([h.to_dict() for h in out])


def _prepare_cluster_feature_series(series: pd.Series, col_name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if col_name == "time_to_first_value_hours":
        s = s.where(s >= 0, np.nan)
        fill_val = float(s.median()) if s.notna().any() else 0.0
        s = s.fillna(fill_val)
    else:
        s = s.fillna(0.0)
        s = s.clip(lower=0.0)
    return np.log1p(s.astype(float))


def _month_shift(month_ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (pd.Timestamp(month_ts).to_period("M") + int(n)).to_timestamp()


def _robust_scale_fit(series: pd.Series) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0, 1.0
    med = float(s.median())
    mad = float((s - med).abs().median())
    scale = float(1.4826 * mad) if mad > 1e-9 else float(s.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return med, scale


def _robust_scale_apply(series: pd.Series, med: float, scale: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    return (s - med) / scale


def _evaluate_heavy_threshold_candidate(
    threshold_quantile: float,
    threshold_value: float,
    active_scores: pd.DataFrame,
    um_all: pd.DataFrame,
    holdout_eval_months: List[pd.Timestamp],
    prevalence_eval_months: List[pd.Timestamp],
    target_share_min: float,
    target_share_max: float,
    max_prevalence_cv: float,
) -> Dict[str, Any]:
    cand = active_scores.copy()
    cand["is_heavy"] = (pd.to_numeric(cand["heavy_score_pca1"], errors="coerce") >= float(threshold_value)).astype(int)
    active_n = int(len(cand))
    heavy_n = int(cand["is_heavy"].sum())
    heavy_share = float(heavy_n / active_n) if active_n > 0 else np.nan

    flag_map = cand[["unique_id", "is_heavy"]].copy()
    panel = um_all.merge(flag_map, on="unique_id", how="left")
    panel["is_heavy"] = pd.to_numeric(panel["is_heavy"], errors="coerce").fillna(0).astype(int)

    prev_panel = panel[panel["month"].isin(prevalence_eval_months)].copy()
    monthly_prev = (
        prev_panel.groupby("month", as_index=False)
        .agg(active_users=("unique_id", "count"), heavy_users=("is_heavy", "sum"))
        .sort_values("month")
    )
    monthly_prev["heavy_prevalence"] = monthly_prev["heavy_users"] / monthly_prev["active_users"].replace(0, np.nan)
    prev_values = pd.to_numeric(monthly_prev["heavy_prevalence"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    prevalence_cv = float(prev_values.std(ddof=0) / prev_values.mean()) if len(prev_values) >= 2 and float(prev_values.mean()) > 0 else np.nan

    holdout = panel[panel["month"].isin(holdout_eval_months)].copy()
    holdout["segment"] = np.where(holdout["is_heavy"] == 1, "heavy_users", "base_regular")
    seg_rsva = (
        holdout.groupby("segment", as_index=False)
        .agg(
            active_rows=("unique_id", "count"),
            strict_users=("strict_flag", "sum"),
            strict_retained_users=("rsva_outcome", "sum"),
        )
        .reset_index(drop=True)
    )
    for col in ["active_rows", "strict_users", "strict_retained_users"]:
        if col in seg_rsva.columns:
            seg_rsva[col] = pd.to_numeric(seg_rsva[col], errors="coerce")
    seg_rsva["rsva_m1"] = seg_rsva["strict_retained_users"] / seg_rsva["active_rows"].replace(0, np.nan)

    holdout_user = holdout[["unique_id", "segment", "future_interactions", "future_value_any"]].drop_duplicates(subset=["unique_id"])
    seg_future = (
        holdout_user.groupby("segment", as_index=False)
        .agg(
            users=("unique_id", "count"),
            future_interactions=("future_interactions", "sum"),
            users_with_future_value=("future_value_any", "sum"),
        )
        .reset_index(drop=True)
    )
    seg_future["future_interactions_per_user"] = seg_future["future_interactions"] / seg_future["users"].replace(0, np.nan)
    seg_future["future_value_event_rate"] = seg_future["users_with_future_value"] / seg_future["users"].replace(0, np.nan)
    seg = seg_rsva.merge(
        seg_future[["segment", "users", "future_interactions_per_user", "future_value_event_rate"]],
        on="segment",
        how="left",
    )
    seg_map = {str(r["segment"]): r for _, r in seg.iterrows()}

    heavy_rsva = float(pd.to_numeric(seg_map.get("heavy_users", {}).get("rsva_m1"), errors="coerce")) if "heavy_users" in seg_map else np.nan
    base_rsva = float(pd.to_numeric(seg_map.get("base_regular", {}).get("rsva_m1"), errors="coerce")) if "base_regular" in seg_map else np.nan
    heavy_future = float(pd.to_numeric(seg_map.get("heavy_users", {}).get("future_interactions_per_user"), errors="coerce")) if "heavy_users" in seg_map else np.nan
    base_future = float(pd.to_numeric(seg_map.get("base_regular", {}).get("future_interactions_per_user"), errors="coerce")) if "base_regular" in seg_map else np.nan

    rsva_lift_ratio = float(heavy_rsva / base_rsva) if pd.notna(heavy_rsva) and pd.notna(base_rsva) and base_rsva > 0 else np.nan
    rsva_lift_diff = float(heavy_rsva - base_rsva) if pd.notna(heavy_rsva) and pd.notna(base_rsva) else np.nan
    future_lift_ratio = float(heavy_future / base_future) if pd.notna(heavy_future) and pd.notna(base_future) and base_future > 0 else np.nan

    share_ok = bool(pd.notna(heavy_share) and float(target_share_min) <= float(heavy_share) <= float(target_share_max))
    cv_ok = bool(pd.notna(prevalence_cv) and float(prevalence_cv) <= float(max_prevalence_cv))
    size_ok = bool(heavy_n >= 500)
    constraints_ok = bool(share_ok and cv_ok and size_ok and pd.notna(rsva_lift_ratio))

    share_penalty = 0.0
    if pd.notna(heavy_share):
        if heavy_share < target_share_min:
            share_penalty = float(target_share_min - heavy_share)
        elif heavy_share > target_share_max:
            share_penalty = float(heavy_share - target_share_max)
    cv_penalty = float(max(0.0, prevalence_cv - max_prevalence_cv)) if pd.notna(prevalence_cv) else 1.0
    lift_for_obj = float(rsva_lift_ratio) if pd.notna(rsva_lift_ratio) else -1.0
    objective = lift_for_obj - (4.0 * share_penalty) - (2.0 * cv_penalty) - (0.2 if not size_ok else 0.0)

    return {
        "threshold_quantile": float(threshold_quantile),
        "threshold_value": float(threshold_value),
        "active_users": int(active_n),
        "heavy_users": int(heavy_n),
        "heavy_share_active": float(heavy_share) if pd.notna(heavy_share) else np.nan,
        "prevalence_cv_recent_months": float(prevalence_cv) if pd.notna(prevalence_cv) else np.nan,
        "rsva_m1_heavy": heavy_rsva,
        "rsva_m1_base_regular": base_rsva,
        "rsva_m1_lift_ratio_heavy_vs_base": rsva_lift_ratio,
        "rsva_m1_lift_diff_heavy_minus_base": rsva_lift_diff,
        "future_interactions_lift_ratio_heavy_vs_base": future_lift_ratio,
        "constraint_share_ok": bool(share_ok),
        "constraint_cv_ok": bool(cv_ok),
        "constraint_min_size_ok": bool(size_ok),
        "constraints_ok": bool(constraints_ok),
        "selection_objective": float(objective),
    }


def build_fast_heavy_definition(
    conn: duckdb.DuckDBPyConnection,
    teacher_df: pd.DataFrame,
    random_seed: int,
    baseline_months: int = 6,
    holdout_months: int = 3,
    threshold_quantiles: Sequence[float] = FAST_HEAVY_THRESHOLD_QUANTILES,
    target_share_min: float = 0.08,
    target_share_max: float = 0.20,
    max_prevalence_cv: float = 0.25,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_df = teacher_df.copy()
    required_cols = {"unique_id", "interaction_count", "session_count", "total_session_min", "active_days", "recency_days"}
    missing = sorted(list(required_cols - set(out_df.columns)))
    if missing:
        summary = {
            "status": "missing_required_columns",
            "missing_columns": missing,
            "heavy_definition_rule": "heavy_score_fast_v1 indisponível",
        }
        return out_df, summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    months_df = conn.execute(
        """
        SELECT DISTINCT CAST(date_trunc('month', i.data_inicio) AS DATE) AS month
        FROM interactions i
        INNER JOIN dim_teachers d USING(unique_id)
        WHERE i.data_inicio IS NOT NULL
          AND lower(coalesce(i.user_type,''))='registered'
        ORDER BY 1
        """
    ).fetchdf()
    months = pd.to_datetime(months_df["month"], errors="coerce").dropna().sort_values().unique().tolist()
    if len(months) < 8:
        summary = {
            "status": "insufficient_months_for_fast_v1",
            "n_months_available": int(len(months)),
            "heavy_definition_rule": "Fallback para q90 de heavy_score em todos os ativos por falta de histórico.",
        }
    else:
        summary = {"status": "ok"}

    holdout_n = min(int(holdout_months), max(1, len(months) // 4))
    baseline_n = min(int(baseline_months), max(3, len(months) - holdout_n))
    holdout_month_list = months[-holdout_n:]
    baseline_month_list = months[max(0, len(months) - holdout_n - baseline_n):len(months) - holdout_n]
    if len(baseline_month_list) < 3:
        baseline_month_list = months[:-holdout_n]
    baseline_start = pd.Timestamp(min(baseline_month_list))
    holdout_start = pd.Timestamp(min(holdout_month_list))
    holdout_end_exclusive = _month_shift(pd.Timestamp(max(holdout_month_list)), 1)

    inter_base = conn.execute(
        f"""
        SELECT
            i.unique_id,
            COUNT(*)::DOUBLE AS interaction_count,
            COUNT(DISTINCT CAST(i.data_inicio AS DATE))::DOUBLE AS active_days,
            MAX(i.data_inicio) AS last_interaction_ts
        FROM interactions i
        INNER JOIN dim_teachers d USING(unique_id)
        WHERE i.data_inicio >= TIMESTAMP '{baseline_start:%Y-%m-%d}'
          AND i.data_inicio < TIMESTAMP '{holdout_start:%Y-%m-%d}'
          AND lower(coalesce(i.user_type,''))='registered'
        GROUP BY 1
        """
    ).fetchdf()

    entry_base = conn.execute(
        f"""
        SELECT
            e.unique_id,
            COUNT(*)::DOUBLE AS session_count,
            SUM(GREATEST(epoch(e.data_fim)-epoch(e.data_inicio),0))/60.0 AS total_session_min
        FROM entries e
        INNER JOIN dim_teachers d USING(unique_id)
        WHERE e.data_inicio >= TIMESTAMP '{baseline_start:%Y-%m-%d}'
          AND e.data_inicio < TIMESTAMP '{holdout_start:%Y-%m-%d}'
          AND lower(coalesce(e.user_type,''))='registered'
        GROUP BY 1
        """
    ).fetchdf()

    users = out_df[["unique_id"]].copy()
    users["unique_id"] = users["unique_id"].astype(str).str.strip()
    b = users.merge(inter_base, on="unique_id", how="left").merge(entry_base, on="unique_id", how="left")
    for col in ["interaction_count", "active_days", "session_count", "total_session_min"]:
        b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0.0)
    b["last_interaction_ts"] = pd.to_datetime(b["last_interaction_ts"], errors="coerce")
    b["recency_days"] = (pd.Timestamp(holdout_start) - b["last_interaction_ts"]).dt.total_seconds() / 86400.0
    b["recency_days"] = pd.to_numeric(b["recency_days"], errors="coerce")
    b.loc[b["recency_days"] < 0, "recency_days"] = np.nan
    b["active_user_heavy_window_flag"] = (b["interaction_count"] > 0).astype(int)

    active = b[b["active_user_heavy_window_flag"] == 1].copy()
    if active.empty:
        out_df["heavy_user_flag"] = 0
        out_df["active_user_heavy_window_flag"] = 0
        out_df["heavy_score_pca1"] = np.nan
        out_df["heavy_threshold_quantile"] = np.nan
        out_df["heavy_threshold_value"] = np.nan
        summary.update(
            {
                "status": "no_active_users_baseline_window",
                "baseline_start": str(baseline_start.date()),
                "holdout_start": str(holdout_start.date()),
                "holdout_end_exclusive": str(holdout_end_exclusive.date()),
            }
        )
        return out_df, summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = active[FAST_HEAVY_FEATURES].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in ["interaction_count", "session_count", "total_session_min", "active_days", "recency_days"]:
        x[col] = np.log1p(np.clip(pd.to_numeric(x[col], errors="coerce").fillna(0.0), a_min=0.0, a_max=None))

    robust_params: Dict[str, Dict[str, float]] = {}
    xz = pd.DataFrame(index=x.index)
    for col in FAST_HEAVY_FEATURES:
        med, scale = _robust_scale_fit(x[col])
        robust_params[col] = {"median": med, "scale": scale}
        xz[col] = _robust_scale_apply(x[col], med, scale)

    pca = PCA(n_components=1, random_state=random_seed)
    scores = pca.fit_transform(xz.to_numpy())[:, 0]
    orient_corr = np.corrcoef(scores, xz["interaction_count"].to_numpy())[0, 1] if len(scores) >= 3 else 1.0
    if pd.notna(orient_corr) and orient_corr < 0:
        scores = -scores

    active_scores = active[["unique_id"]].copy()
    active_scores["heavy_score_pca1"] = pd.to_numeric(scores, errors="coerce")

    holdout_inter_user = conn.execute(
        f"""
        SELECT
            i.unique_id,
            COUNT(*)::DOUBLE AS future_interactions,
            MAX(CASE WHEN lower(coalesce(i.event_type,'')) IN ('download_aula','download_plano_aula') THEN 1 ELSE 0 END)::INTEGER AS future_value_any
        FROM interactions i
        INNER JOIN dim_teachers d USING(unique_id)
        WHERE i.data_inicio >= TIMESTAMP '{holdout_start:%Y-%m-%d}'
          AND i.data_inicio < TIMESTAMP '{holdout_end_exclusive:%Y-%m-%d}'
          AND lower(coalesce(i.user_type,''))='registered'
        GROUP BY 1
        """
    ).fetchdf()
    holdout_inter_user["future_interactions"] = pd.to_numeric(holdout_inter_user["future_interactions"], errors="coerce").fillna(0.0)
    holdout_inter_user["future_value_any"] = pd.to_numeric(holdout_inter_user["future_value_any"], errors="coerce").fillna(0).astype(int)

    um_all = conn.execute(
        """
        WITH base AS (
            SELECT
                i.unique_id,
                CAST(date_trunc('month', i.data_inicio) AS DATE) AS month,
                MAX(CASE WHEN lower(coalesce(i.event_type,'')) IN ('download_aula','download_plano_aula') THEN 1 ELSE 0 END)::INTEGER AS strict_flag
            FROM interactions i
            INNER JOIN dim_teachers d USING(unique_id)
            WHERE i.data_inicio IS NOT NULL
              AND lower(coalesce(i.user_type,''))='registered'
            GROUP BY 1,2
        )
        SELECT unique_id, month, strict_flag
        FROM base
        """
    ).fetchdf()
    um_all["month"] = pd.to_datetime(um_all["month"], errors="coerce")
    um_all = um_all.dropna(subset=["month"]).sort_values(["unique_id", "month"]).reset_index(drop=True)
    um_all["next_month"] = um_all["month"].apply(lambda m: _month_shift(pd.Timestamp(m), 1))
    future_map = um_all[["unique_id", "month"]].copy().rename(columns={"month": "next_month"})
    future_map["next_active"] = 1
    um_all = um_all.merge(future_map, on=["unique_id", "next_month"], how="left")
    um_all["next_active"] = pd.to_numeric(um_all["next_active"], errors="coerce").fillna(0).astype(int)
    um_all["rsva_outcome"] = (pd.to_numeric(um_all["strict_flag"], errors="coerce").fillna(0).astype(int) * um_all["next_active"]).astype(int)

    holdout_month_set = set(pd.to_datetime(holdout_month_list))
    all_month_set = set(pd.to_datetime(months))
    holdout_eval_months = [m for m in sorted(holdout_month_set) if _month_shift(m, 1) in all_month_set]
    if not holdout_eval_months:
        holdout_eval_months = sorted(holdout_month_set)[:-1] if len(holdout_month_set) > 1 else sorted(holdout_month_set)
    prevalence_eval_months = sorted(set(pd.to_datetime(baseline_month_list + holdout_month_list)))

    um_all = um_all.merge(
        holdout_inter_user[["unique_id", "future_interactions", "future_value_any"]],
        on="unique_id",
        how="left",
    )
    um_all["future_interactions"] = pd.to_numeric(um_all["future_interactions"], errors="coerce").fillna(0.0)
    um_all["future_value_any"] = pd.to_numeric(um_all["future_value_any"], errors="coerce").fillna(0).astype(int)

    candidate_rows: List[Dict[str, Any]] = []
    score_vals = pd.to_numeric(active_scores["heavy_score_pca1"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    for q in threshold_quantiles:
        if score_vals.empty:
            continue
        thr = float(score_vals.quantile(float(q)))
        candidate_rows.append(
            _evaluate_heavy_threshold_candidate(
                threshold_quantile=float(q),
                threshold_value=thr,
                active_scores=active_scores,
                um_all=um_all,
                holdout_eval_months=holdout_eval_months,
                prevalence_eval_months=prevalence_eval_months,
                target_share_min=float(target_share_min),
                target_share_max=float(target_share_max),
                max_prevalence_cv=float(max_prevalence_cv),
            )
        )
    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        candidate_df = pd.DataFrame(
            [
                {
                    "threshold_quantile": 0.90,
                    "threshold_value": float(score_vals.quantile(0.90)) if not score_vals.empty else np.nan,
                    "constraints_ok": False,
                    "selection_objective": -1.0,
                }
            ]
        )

    valid = candidate_df[candidate_df["constraints_ok"] == True].copy() if "constraints_ok" in candidate_df.columns else pd.DataFrame()
    if not valid.empty:
        valid = valid.sort_values(["selection_objective", "rsva_m1_lift_ratio_heavy_vs_base", "rsva_m1_lift_diff_heavy_minus_base"], ascending=[False, False, False])
        best = valid.iloc[0]
    else:
        tmp = candidate_df.copy()
        if "selection_objective" in tmp.columns:
            tmp = tmp.sort_values("selection_objective", ascending=False)
        best = tmp.iloc[0]

    q_sel = float(pd.to_numeric(best.get("threshold_quantile"), errors="coerce"))
    thr_sel = float(pd.to_numeric(best.get("threshold_value"), errors="coerce"))
    active_scores["heavy_user_flag"] = (pd.to_numeric(active_scores["heavy_score_pca1"], errors="coerce") >= thr_sel).astype(int)

    all_users = out_df[["unique_id"]].copy()
    all_users["unique_id"] = all_users["unique_id"].astype(str).str.strip()
    merged_scores = all_users.merge(active_scores, on="unique_id", how="left").merge(
        b[["unique_id", "active_user_heavy_window_flag"]], on="unique_id", how="left"
    )
    merged_scores["heavy_user_flag"] = pd.to_numeric(merged_scores["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    merged_scores["active_user_heavy_window_flag"] = pd.to_numeric(merged_scores["active_user_heavy_window_flag"], errors="coerce").fillna(0).astype(int)
    merged_scores.loc[merged_scores["active_user_heavy_window_flag"] == 0, "heavy_user_flag"] = 0
    merged_scores["heavy_threshold_quantile"] = q_sel
    merged_scores["heavy_threshold_value"] = thr_sel

    out_df = out_df.merge(
        merged_scores[
            [
                "unique_id",
                "active_user_heavy_window_flag",
                "heavy_score_pca1",
                "heavy_user_flag",
                "heavy_threshold_quantile",
                "heavy_threshold_value",
            ]
        ],
        on="unique_id",
        how="left",
        suffixes=("", "_new"),
    )
    for col in ["active_user_heavy_window_flag", "heavy_user_flag", "heavy_threshold_quantile", "heavy_threshold_value", "heavy_score_pca1"]:
        new_col = f"{col}_new"
        if new_col in out_df.columns:
            out_df[col] = out_df[new_col]
            out_df = out_df.drop(columns=[new_col])
    out_df["heavy_user_flag"] = pd.to_numeric(out_df["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    out_df["active_user_heavy_window_flag"] = pd.to_numeric(out_df["active_user_heavy_window_flag"], errors="coerce").fillna(0).astype(int)

    score_map = active_scores[["unique_id", "heavy_score_pca1"]].copy()
    holdout_user = all_users.merge(holdout_inter_user, on="unique_id", how="left").merge(score_map, on="unique_id", how="left")
    holdout_user["future_interactions"] = pd.to_numeric(holdout_user["future_interactions"], errors="coerce").fillna(0.0)
    holdout_user["future_value_any"] = pd.to_numeric(holdout_user["future_value_any"], errors="coerce").fillna(0).astype(int)
    holdout_user = holdout_user.merge(merged_scores[["unique_id", "heavy_user_flag", "active_user_heavy_window_flag"]], on="unique_id", how="left")
    holdout_user = holdout_user[holdout_user["active_user_heavy_window_flag"] == 1].copy()

    decile_df = pd.DataFrame()
    monotonicity_spearman = np.nan
    if not holdout_user.empty and holdout_user["heavy_score_pca1"].notna().sum() >= 100:
        dec = holdout_user.copy()
        try:
            dec["score_decile"] = pd.qcut(pd.to_numeric(dec["heavy_score_pca1"], errors="coerce"), q=10, labels=False, duplicates="drop")
        except Exception:
            dec["score_decile"] = pd.cut(pd.to_numeric(dec["heavy_score_pca1"], errors="coerce"), bins=10, labels=False)
        dec["score_decile"] = pd.to_numeric(dec["score_decile"], errors="coerce")
        dec = dec.dropna(subset=["score_decile"])
        if not dec.empty:
            dec["score_decile"] = dec["score_decile"].astype(int) + 1
            decile_df = (
                dec.groupby("score_decile", as_index=False)
                .agg(
                    users=("unique_id", "count"),
                    mean_future_interactions=("future_interactions", "mean"),
                    future_value_event_rate=("future_value_any", "mean"),
                    mean_heavy_score=("heavy_score_pca1", "mean"),
                )
                .sort_values("score_decile")
                .reset_index(drop=True)
            )
            if len(decile_df) >= 4:
                monotonicity_spearman = float(spearmanr(decile_df["score_decile"], decile_df["mean_future_interactions"]).correlation)

    sel_flag_map = merged_scores[["unique_id", "heavy_user_flag"]].copy().rename(columns={"heavy_user_flag": "is_heavy"})
    prev_selected = um_all.merge(sel_flag_map, on="unique_id", how="left")
    prev_selected["is_heavy"] = pd.to_numeric(prev_selected["is_heavy"], errors="coerce").fillna(0).astype(int)
    prevalence_df = (
        prev_selected.groupby("month", as_index=False)
        .agg(active_users=("unique_id", "count"), heavy_users=("is_heavy", "sum"))
        .sort_values("month")
        .reset_index(drop=True)
    )
    prevalence_df["heavy_prevalence"] = prevalence_df["heavy_users"] / prevalence_df["active_users"].replace(0, np.nan)

    oot_rows = []
    if not candidate_df.empty:
        best_row = candidate_df[candidate_df["threshold_quantile"] == q_sel].head(1)
        if not best_row.empty:
            r = best_row.iloc[0]
            oot_rows.append(
                {
                    "selected_threshold_quantile": q_sel,
                    "selected_threshold_value": thr_sel,
                    "rsva_m1_heavy": r.get("rsva_m1_heavy"),
                    "rsva_m1_base_regular": r.get("rsva_m1_base_regular"),
                    "rsva_m1_lift_ratio_heavy_vs_base": r.get("rsva_m1_lift_ratio_heavy_vs_base"),
                    "rsva_m1_lift_diff_heavy_minus_base": r.get("rsva_m1_lift_diff_heavy_minus_base"),
                    "future_interactions_lift_ratio_heavy_vs_base": r.get("future_interactions_lift_ratio_heavy_vs_base"),
                    "prevalence_cv_recent_months": r.get("prevalence_cv_recent_months"),
                }
            )
    oot_lift_df = pd.DataFrame(oot_rows)

    summary.update(
        {
            "version": "heavy_score_fast_v1",
            "method": "PCA-1 em features de intensidade/consistência com escala robusta; threshold escolhido por grid com holdout.",
            "features_used": FAST_HEAVY_FEATURES,
            "transform": "log1p + robust_scale(median/MAD)",
            "baseline_months": [str(pd.Timestamp(m).date()) for m in baseline_month_list],
            "holdout_months": [str(pd.Timestamp(m).date()) for m in holdout_month_list],
            "holdout_eval_months": [str(pd.Timestamp(m).date()) for m in holdout_eval_months],
            "threshold_grid_quantiles": [float(x) for x in threshold_quantiles],
            "selected_threshold_quantile": float(q_sel),
            "selected_threshold_value": float(thr_sel),
            "selected_constraints_ok": bool(best.get("constraints_ok")) if "constraints_ok" in best.index else False,
            "selected_constraint_flags": {
                "share_ok": bool(best.get("constraint_share_ok")) if "constraint_share_ok" in best.index else False,
                "cv_ok": bool(best.get("constraint_cv_ok")) if "constraint_cv_ok" in best.index else False,
                "min_size_ok": bool(best.get("constraint_min_size_ok")) if "constraint_min_size_ok" in best.index else False,
            },
            "selected_prevalence_cv_recent_months": float(pd.to_numeric(best.get("prevalence_cv_recent_months"), errors="coerce"))
            if pd.notna(pd.to_numeric(best.get("prevalence_cv_recent_months"), errors="coerce"))
            else np.nan,
            "constraints": {
                "target_share_min": float(target_share_min),
                "target_share_max": float(target_share_max),
                "max_prevalence_cv": float(max_prevalence_cv),
                "min_heavy_users_active": 500,
            },
            "monotonicity_spearman_decile_future_interactions": float(monotonicity_spearman) if pd.notna(monotonicity_spearman) else np.nan,
            "heavy_users_total": int(out_df["heavy_user_flag"].sum()),
            "active_users_heavy_window": int((out_df["active_user_heavy_window_flag"] == 1).sum()),
            "heavy_share_active_heavy_window": float(
                out_df.loc[out_df["active_user_heavy_window_flag"] == 1, "heavy_user_flag"].mean()
            )
            if (out_df["active_user_heavy_window_flag"] == 1).any()
            else np.nan,
            "robust_scaler_params": robust_params,
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]) if hasattr(pca, "explained_variance_ratio_") else np.nan,
            "heavy_definition_rule": "heavy_user_flag=1 se active_user_heavy_window=1 e heavy_score_pca1 >= threshold selecionado por holdout no grid q85,q88,q90,q92,q95",
        }
    )
    return out_df, summary, candidate_df, prevalence_df, oot_lift_df, decile_df


def select_cluster_features(
    active_df: pd.DataFrame,
    candidate_cols: Sequence[str],
    corr_threshold: float = 0.95,
    min_non_zero_share: float = 0.01,
    max_zero_share: float = 0.98,
    min_non_zero_n: int = 100,
    min_features: int = 4,
) -> Tuple[List[str], pd.DataFrame]:
    if active_df.empty:
        return [], pd.DataFrame(columns=["feature", "selected", "drop_reason"])

    n_obs = float(len(active_df))
    feature_rows: List[Dict[str, Any]] = []
    transformed: Dict[str, pd.Series] = {}

    for col in candidate_cols:
        if col not in active_df.columns:
            continue
        raw = pd.to_numeric(active_df[col], errors="coerce")
        transformed_series = _prepare_cluster_feature_series(raw, col)
        transformed[col] = transformed_series

        missing_share = float(raw.isna().mean())
        zero_share = float((transformed_series == 0).mean())
        non_zero_share = float(1.0 - zero_share)
        unique_n = int(transformed_series.nunique(dropna=True))
        std = float(transformed_series.std(ddof=0)) if len(transformed_series) else 0.0
        non_zero_n = int((transformed_series > 0).sum())

        support_ok = bool(
            (unique_n >= 5)
            and (std > 0.0)
            and (non_zero_share >= float(min_non_zero_share))
            and (zero_share <= float(max_zero_share))
            and (non_zero_n >= int(min_non_zero_n))
        )
        feature_rows.append(
            {
                "feature": col,
                "missing_share": missing_share,
                "zero_share": zero_share,
                "non_zero_share": non_zero_share,
                "non_zero_n": non_zero_n,
                "n_unique_transformed": unique_n,
                "std_transformed": std,
                "support_ok": support_ok,
            }
        )

    audit = pd.DataFrame(feature_rows)
    if audit.empty:
        return [], pd.DataFrame(columns=["feature", "selected", "drop_reason"])

    anchor_order = {
        "interaction_count": 0,
        "session_count": 1,
        "download_event_count": 2,
        "aula_event_count": 3,
        "visualizacao_event_count": 4,
        "total_session_min": 5,
        "unique_lessons_count": 6,
    }
    audit["is_anchor"] = audit["feature"].map(lambda x: x in anchor_order)
    audit["anchor_priority"] = audit["feature"].map(lambda x: anchor_order.get(str(x), 99))
    audit["selection_score"] = (
        1.5 * audit["non_zero_share"]
        + 0.4 * np.log1p(np.clip(audit["std_transformed"], a_min=0.0, a_max=None))
        + 0.1 * (1.0 - audit["missing_share"])
    )
    audit["selected"] = False
    audit["drop_reason"] = ""
    audit["max_abs_corr_to_selected"] = np.nan

    eligible = audit[(audit["support_ok"] == True) | (audit["is_anchor"] == True)].copy()
    eligible = eligible.sort_values(
        ["anchor_priority", "selection_score", "non_zero_share", "std_transformed"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    pre_cols = eligible["feature"].astype(str).tolist()
    if not pre_cols:
        return [], audit.sort_values("anchor_priority").reset_index(drop=True)

    x_pre = pd.DataFrame({c: transformed[c] for c in pre_cols}, index=active_df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = x_pre.corr(method="spearman").abs().fillna(0.0)

    kept: List[str] = []
    for col in pre_cols:
        if not kept:
            kept.append(col)
            continue
        max_corr = float(corr.loc[col, kept].max()) if col in corr.index else 0.0
        if max_corr >= float(corr_threshold):
            audit.loc[audit["feature"] == col, "drop_reason"] = f"high_corr_ge_{corr_threshold:.2f}"
            audit.loc[audit["feature"] == col, "max_abs_corr_to_selected"] = max_corr
            continue
        kept.append(col)

    if len(kept) < int(min_features):
        pool = audit[~audit["feature"].isin(kept)].copy()
        pool = pool[pool["std_transformed"] > 0.0].sort_values(
            ["anchor_priority", "selection_score", "non_zero_share"],
            ascending=[True, False, False],
        )
        for col in pool["feature"].astype(str).tolist():
            kept.append(col)
            audit.loc[audit["feature"] == col, "drop_reason"] = "fallback_min_features"
            if len(kept) >= int(min_features):
                break

    kept = [c for c in kept if c in transformed]
    audit.loc[audit["feature"].isin(kept), "selected"] = True
    audit.loc[(audit["selected"] == False) & (audit["drop_reason"] == ""), "drop_reason"] = "low_support_or_redundant"
    audit = audit.sort_values(["selected", "anchor_priority", "selection_score"], ascending=[False, True, False]).reset_index(drop=True)
    audit["n_active_users"] = int(n_obs)
    return kept, audit


def compute_cluster_artifacts(
    teacher_df: pd.DataFrame,
    random_seed: int,
    max_cluster_sample: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    out_df = teacher_df.copy()
    if "heavy_user_flag" not in out_df.columns:
        out_df["heavy_user_flag"] = 0
    if "heavy_cluster_flag" not in out_df.columns:
        out_df["heavy_cluster_flag"] = 0
    if "behavior_cluster_id" not in out_df.columns:
        out_df["behavior_cluster_id"] = -1
    if "engagement_intensity_score" not in out_df.columns:
        out_df["engagement_intensity_score"] = np.nan
    if "cluster_fit_population" not in out_df.columns:
        out_df["cluster_fit_population"] = "inactive_or_unassigned"

    out_df["heavy_user_flag"] = pd.to_numeric(out_df["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    out_df["heavy_cluster_flag"] = pd.to_numeric(out_df["heavy_cluster_flag"], errors="coerce").fillna(0).astype(int)
    out_df["behavior_cluster_id"] = pd.to_numeric(out_df["behavior_cluster_id"], errors="coerce").fillna(-1).astype(int)
    out_df["engagement_intensity_score"] = pd.to_numeric(out_df["engagement_intensity_score"], errors="coerce")
    out_df["cluster_fit_population"] = out_df["cluster_fit_population"].astype(str)

    empty_profiles = pd.DataFrame(
        columns=[
            "cluster",
            "teachers",
            "share",
            "conversion_rate",
            "median_interactions",
            "median_session_min",
            "cluster_intensity_median",
            "cluster_is_heavy",
            "heavy_share",
        ]
    )
    empty_pca = pd.DataFrame(columns=["pca1", "pca2", "cluster"])
    empty_feature_audit = pd.DataFrame(columns=["feature", "selected", "drop_reason"])

    out_df["behavior_cluster_id"] = -1
    out_df["cluster_fit_population"] = np.where(
        pd.to_numeric(out_df["heavy_user_flag"], errors="coerce").fillna(0).astype(int) == 1,
        "heavy_users_only",
        "base_or_inactive",
    )
    out_df["heavy_cluster_flag"] = pd.to_numeric(out_df["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    out_df["engagement_intensity_score"] = pd.to_numeric(out_df.get("heavy_score_pca1"), errors="coerce")

    heavy_df = out_df[pd.to_numeric(out_df["heavy_user_flag"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if heavy_df.empty:
        summary = {
            "method": "kmeans_heavy_subtypes_fast_v1",
            "best_k": None,
            "best_silhouette": np.nan,
            "cluster_feature_cols": [],
            "cluster_feature_candidates": [c for c in CLUSTER_FEATURE_CANDIDATES if c in out_df.columns],
            "heavy_users_n": 0,
            "cluster_scope": "heavy_users_only",
            "cluster_feature_selection_note": "Sem heavy users para clusterização de subtipos.",
        }
        return out_df, empty_profiles, empty_pca, summary, empty_feature_audit

    denom_inter = pd.to_numeric(heavy_df.get("interaction_count"), errors="coerce").fillna(0.0).replace(0.0, np.nan)
    heavy_df["aula_event_share"] = pd.to_numeric(heavy_df.get("aula_event_count"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["prova_event_share"] = pd.to_numeric(heavy_df.get("prova_event_count"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["plano_event_share"] = pd.to_numeric(heavy_df.get("plano_event_count"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["download_event_share"] = pd.to_numeric(heavy_df.get("download_event_count"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["visualizacao_event_share"] = pd.to_numeric(heavy_df.get("visualizacao_event_count"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["ia_event_share"] = pd.to_numeric(heavy_df.get("ia_event_count"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["desktop_event_share"] = pd.to_numeric(heavy_df.get("desktop_events"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["mobile_event_share"] = pd.to_numeric(heavy_df.get("mobile_events"), errors="coerce").fillna(0.0) / denom_inter
    heavy_df["tablet_event_share"] = pd.to_numeric(heavy_df.get("tablet_events"), errors="coerce").fillna(0.0) / denom_inter
    for c in [
        "aula_event_share",
        "prova_event_share",
        "plano_event_share",
        "download_event_share",
        "visualizacao_event_share",
        "ia_event_share",
        "desktop_event_share",
        "mobile_event_share",
        "tablet_event_share",
    ]:
        heavy_df[c] = pd.to_numeric(heavy_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    selected_features, feature_audit = select_cluster_features(
        active_df=heavy_df,
        candidate_cols=[c for c in CLUSTER_FEATURE_CANDIDATES if c in heavy_df.columns],
        corr_threshold=0.95,
        min_non_zero_share=0.02,
        max_zero_share=0.98,
        min_non_zero_n=max(30, int(0.02 * len(heavy_df))),
        min_features=4,
    )
    if len(selected_features) < 2 or len(heavy_df) < 60:
        summary = {
            "method": "kmeans_heavy_subtypes_fast_v1",
            "best_k": None,
            "best_silhouette": np.nan,
            "cluster_feature_cols": selected_features,
            "cluster_feature_candidates": [c for c in CLUSTER_FEATURE_CANDIDATES if c in heavy_df.columns],
            "heavy_users_n": int(len(heavy_df)),
            "cluster_scope": "heavy_users_only",
            "cluster_feature_selection_note": "Features insuficientes ou amostra heavy pequena para subtipos.",
        }
        return out_df, empty_profiles, empty_pca, summary, feature_audit

    x_heavy = pd.DataFrame(
        {c: _prepare_cluster_feature_series(heavy_df[c], c) for c in selected_features},
        index=heavy_df.index,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(x_heavy) > max_cluster_sample:
        sample_idx = x_heavy.sample(max_cluster_sample, random_state=random_seed).index
    else:
        sample_idx = x_heavy.index
    x_sample = x_heavy.loc[sample_idx]
    if x_sample.shape[0] < 20:
        summary = {
            "method": "kmeans_heavy_subtypes_fast_v1",
            "best_k": None,
            "best_silhouette": np.nan,
            "cluster_feature_cols": selected_features,
            "cluster_feature_candidates": [c for c in CLUSTER_FEATURE_CANDIDATES if c in heavy_df.columns],
            "heavy_users_n": int(len(heavy_df)),
            "cluster_scope": "heavy_users_only",
            "cluster_feature_selection_note": "Amostra heavy insuficiente para clusterização robusta.",
        }
        return out_df, empty_profiles, empty_pca, summary, feature_audit

    scaler = StandardScaler()
    x_scaled_sample = scaler.fit_transform(x_sample)
    x_scaled_full = scaler.transform(x_heavy)
    max_k = max(2, min(6, x_sample.shape[0] - 1))
    min_cluster_n = max(10, int(np.ceil(0.05 * x_sample.shape[0])))

    best_k: Optional[int] = None
    best_score = -1.0
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = model.fit_predict(x_scaled_sample)
        counts = pd.Series(labels).value_counts()
        if counts.min() < min_cluster_n:
            continue
        score = silhouette_score(x_scaled_sample, labels)
        if score > best_score:
            best_score = float(score)
            best_k = int(k)
    if best_k is None:
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
            labels = model.fit_predict(x_scaled_sample)
            score = silhouette_score(x_scaled_sample, labels)
            if score > best_score:
                best_score = float(score)
                best_k = int(k)
    if best_k is None:
        best_k = 2
        best_score = np.nan

    final_model = KMeans(n_clusters=best_k, random_state=random_seed, n_init=10)
    labels_full = final_model.fit_predict(x_scaled_full)
    labels_sample_final = final_model.predict(x_scaled_sample)
    if len(np.unique(labels_sample_final)) >= 2:
        best_score = float(silhouette_score(x_scaled_sample, labels_sample_final))

    heavy_df["behavior_cluster_id"] = labels_full.astype(int)
    out_df.loc[heavy_df.index, "behavior_cluster_id"] = heavy_df["behavior_cluster_id"].astype(int)
    out_df["cluster_best_k"] = best_k
    out_df["cluster_silhouette"] = best_score
    out_df["cluster_train_sample_n"] = int(len(x_sample))
    out_df["cluster_feature_set"] = ",".join(selected_features)

    profiles = (
        heavy_df.groupby("behavior_cluster_id", dropna=False)
        .agg(
            teachers=("unique_id", "count"),
            conversion_rate=("converted_within_window", "mean"),
            median_interactions=("interaction_count", "median"),
            median_session_min=("avg_session_min", "median"),
            cluster_intensity_median=("heavy_score_pca1", "median"),
            heavy_share=("heavy_user_flag", "mean"),
        )
        .reset_index()
        .rename(columns={"behavior_cluster_id": "cluster"})
        .sort_values("teachers", ascending=False)
        .reset_index(drop=True)
    )
    profiles["share"] = profiles["teachers"] / profiles["teachers"].sum()
    profiles["cluster_is_heavy"] = True

    pca = PCA(n_components=2, random_state=random_seed)
    pca_points = pca.fit_transform(x_scaled_sample)
    pca_df = pd.DataFrame(
        {
            "pca1": pca_points[:, 0],
            "pca2": pca_points[:, 1],
            "cluster": labels_sample_final,
        }
    )

    summary = {
        "method": "kmeans_heavy_subtypes_fast_v1",
        "best_k": int(best_k),
        "best_silhouette": float(best_score) if pd.notna(best_score) else np.nan,
        "cluster_scope": "heavy_users_only",
        "cluster_feature_cols": selected_features,
        "cluster_feature_candidates": [c for c in CLUSTER_FEATURE_CANDIDATES if c in heavy_df.columns],
        "cluster_feature_selection_rule": "features comportamentais de mix/eventos e contexto; filtro por suporte e redundância (Spearman).",
        "cluster_train_sample_n": int(len(x_sample)),
        "heavy_users_n": int(len(heavy_df)),
    }
    return out_df, profiles, pca_df, summary, feature_audit


def status_counts(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or "status" not in df.columns:
        return {}
    vc = df["status"].value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_parquet_duckdb(df: pd.DataFrame, path: Path) -> None:
    """
    Escreve Parquet sem depender de pyarrow/fastparquet, usando DuckDB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(database=":memory:")
    try:
        conn.register("tmp_df", df)
        conn.execute(f"COPY tmp_df TO '{q(path)}' (FORMAT PARQUET)")
    finally:
        conn.close()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config(args)

    paths = ensure_dirs(cfg.output_dir)
    LOGGER.info("Starting etapa_01_base | data_dir=%s | output_dir=%s", cfg.data_dir, cfg.output_dir)

    conn = duckdb.connect(database=":memory:")
    build_views(conn, cfg.data_dir)

    snapshot_ts = conn.execute(
        """
        SELECT GREATEST(
          (SELECT max(data_fim) FROM entries),
          (SELECT max(data_inicio) FROM interactions),
          (SELECT max(updatedat) FROM mari_conv),
          (SELECT max(date) FROM mari_help)
        )
        """
    ).fetchone()[0]

    LOGGER.info("Building teacher analytical dataset")
    teacher_df = build_teacher_dataset(conn, conversion_days=cfg.conversion_days)

    LOGGER.info("Running fast heavy-score definition (PCA-1 + threshold grid)")
    (
        teacher_df,
        heavy_definition_summary,
        heavy_threshold_grid_df,
        heavy_prevalence_monthly_df,
        heavy_out_of_time_lift_df,
        heavy_score_deciles_df,
    ) = build_fast_heavy_definition(
        conn=conn,
        teacher_df=teacher_df,
        random_seed=cfg.random_seed,
        baseline_months=6,
        holdout_months=3,
        threshold_quantiles=FAST_HEAVY_THRESHOLD_QUANTILES,
        target_share_min=0.08,
        target_share_max=0.20,
        max_prevalence_cv=0.25,
    )

    LOGGER.info("Running clustering for heavy-user subtypes")
    teacher_df, cluster_profiles, cluster_pca_points, cluster_summary, cluster_feature_diagnostics = compute_cluster_artifacts(
        teacher_df=teacher_df,
        random_seed=cfg.random_seed,
        max_cluster_sample=cfg.max_cluster_sample,
    )

    LOGGER.info("Running data-quality and identity checks")
    table_inventory = compute_table_inventory(conn)
    join_coverage = compute_join_coverage(conn)
    identity_coverage = compute_identity_coverage(conn)
    consistency_checks = compute_consistency_checks(conn)

    LOGGER.info("Computing monthly panels and summary metrics")
    monthly_solution_usage = compute_monthly_solution_usage(conn)
    users_panel, retention = compute_users_panel(conn)
    summary = compute_summary_metrics(conn, teacher_df, users_panel, retention)

    LOGGER.info("Computing associations")
    assoc = compute_association_tables(conn, teacher_df)

    hotjar_summary = compute_hotjar_summary(cfg.data_dir)

    LOGGER.info("Evaluating hypotheses")
    hypothesis_df = build_hypothesis_dataframe(
        teacher_df=teacher_df,
        monthly_df=monthly_solution_usage,
        hotjar=hotjar_summary,
        alpha=cfg.alpha,
        min_segment_n=cfg.min_segment_n,
        random_seed=cfg.random_seed,
        max_cluster_sample=cfg.max_cluster_sample,
        cluster_artifacts_summary=cluster_summary,
    )

    # Export tabular artifacts
    teacher_df.to_csv(cfg.output_dir / "teacher_dataset.csv", index=False)
    write_parquet_duckdb(teacher_df, paths["parquet"] / "teacher_dataset.parquet")
    teacher_df.sample(min(cfg.teacher_dataset_sample_rows, len(teacher_df)), random_state=cfg.random_seed).to_csv(
        cfg.output_dir / "teacher_analytical_dataset_sample.csv", index=False
    )

    monthly_solution_usage.to_csv(cfg.output_dir / "eda_monthly_solution_usage.csv", index=False)
    users_panel.to_csv(cfg.output_dir / "users_monthly_panel.csv", index=False)
    retention.to_csv(cfg.output_dir / "retention_monthly_entries.csv", index=False)
    table_inventory.to_csv(cfg.output_dir / "data_quality_table_inventory.csv", index=False)
    join_coverage.to_csv(cfg.output_dir / "data_quality_join_coverage.csv", index=False)
    identity_coverage.to_csv(cfg.output_dir / "identity_coverage.csv", index=False)
    consistency_checks.to_csv(cfg.output_dir / "data_quality_consistency_checks.csv", index=False)

    hypothesis_df.to_csv(cfg.output_dir / "hypothesis_results.csv", index=False)
    for k, df in assoc.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(cfg.output_dir / f"{k}.csv", index=False)
            write_parquet_duckdb(df, paths["parquet"] / f"{k}.parquet")

    cluster_profiles.to_csv(cfg.output_dir / "cluster_profiles.csv", index=False)
    cluster_pca_points.to_csv(cfg.output_dir / "cluster_pca_points_sample.csv", index=False)
    cluster_feature_diagnostics.to_csv(cfg.output_dir / "cluster_feature_diagnostics.csv", index=False)
    heavy_threshold_grid_df.to_csv(cfg.output_dir / "heavy_threshold_grid_search.csv", index=False)
    heavy_prevalence_monthly_df.to_csv(cfg.output_dir / "heavy_prevalence_monthly.csv", index=False)
    heavy_out_of_time_lift_df.to_csv(cfg.output_dir / "heavy_out_of_time_lift.csv", index=False)
    heavy_score_deciles_df.to_csv(cfg.output_dir / "heavy_score_decile_diagnostics.csv", index=False)
    write_json(cfg.output_dir / "heavy_definition.json", heavy_definition_summary)

    # Consolidated payload
    trust_score = 100.0
    if not join_coverage.empty:
        mean_cov = float(join_coverage["coverage"].mean())
        fails = int((consistency_checks["status"] == "fail").sum()) if not consistency_checks.empty else 0
        warnings = int((consistency_checks["status"] == "warning").sum()) if not consistency_checks.empty else 0
        trust_score = max(0.0, min(100.0, 100.0 * mean_cov - 8.0 * fails - 2.5 * warnings))
    else:
        mean_cov = np.nan
        fails = 0
        warnings = 0

    validated_n = int((hypothesis_df["status"] == "validated").sum()) if not hypothesis_df.empty else 0
    not_testable_n = int((hypothesis_df["status"] == "not_testable").sum()) if not hypothesis_df.empty else 0

    consolidated: Dict[str, Any] = {
        "run_metadata": {
            "run_timestamp_utc": utc_now_iso(),
            "data_dir": str(cfg.data_dir),
            "output_dir": str(cfg.output_dir),
            "snapshot_ts": str(snapshot_ts),
            "config": {
                "random_seed": cfg.random_seed,
                "conversion_days": cfg.conversion_days,
                "alpha": cfg.alpha,
                "min_segment_n": cfg.min_segment_n,
                "max_cluster_sample": cfg.max_cluster_sample,
                "teacher_dataset_sample_rows": cfg.teacher_dataset_sample_rows,
            },
        },
        "data_quality": {
            "table_inventory": table_inventory.to_dict(orient="records"),
            "join_coverage": join_coverage.to_dict(orient="records"),
            "consistency_checks": consistency_checks.to_dict(orient="records"),
            "trust_assessment": {
                "trust_score_0_100": round(float(trust_score), 2),
                "mean_join_coverage": float(mean_cov) if pd.notna(mean_cov) else None,
                "consistency_fail_count": int(fails),
                "consistency_warning_count": int(warnings),
            },
            "identity_coverage_path": str(cfg.output_dir / "identity_coverage.csv"),
        },
        "eda": {
            **summary,
            "monthly_solution_usage_path": str(cfg.output_dir / "eda_monthly_solution_usage.csv"),
            "users_monthly_panel_path": str(cfg.output_dir / "users_monthly_panel.csv"),
            "retention_monthly_entries_path": str(cfg.output_dir / "retention_monthly_entries.csv"),
        },
        "hotjar": {
            "summary": hotjar_summary,
        },
        "hypotheses": {
            "status_counts": status_counts(hypothesis_df),
            "results": hypothesis_df.to_dict(orient="records"),
            "results_path": str(cfg.output_dir / "hypothesis_results.csv"),
        },
        "associations": {
            "state_stats_path": str(cfg.output_dir / "state_stats.csv"),
            "utm_stats_path": str(cfg.output_dir / "utm_stats.csv"),
            "geo_associations_path": str(cfg.output_dir / "geo_associations.csv"),
            "top_corr_pairs_path": str(cfg.output_dir / "top_corr_pairs.csv"),
            "cat_corr_pairs_path": str(cfg.output_dir / "cat_corr_pairs.csv"),
            "journey_path_counts_path": str(cfg.output_dir / "journey_path_counts.csv"),
        },
        "clustering": {
            "cluster_profiles_path": str(cfg.output_dir / "cluster_profiles.csv"),
            "cluster_pca_points_path": str(cfg.output_dir / "cluster_pca_points_sample.csv"),
            "cluster_feature_diagnostics_path": str(cfg.output_dir / "cluster_feature_diagnostics.csv"),
            "artifacts_summary": cluster_summary,
        },
        "heavy_definition": {
            "definition_json_path": str(cfg.output_dir / "heavy_definition.json"),
            "threshold_grid_path": str(cfg.output_dir / "heavy_threshold_grid_search.csv"),
            "prevalence_monthly_path": str(cfg.output_dir / "heavy_prevalence_monthly.csv"),
            "out_of_time_lift_path": str(cfg.output_dir / "heavy_out_of_time_lift.csv"),
            "score_deciles_path": str(cfg.output_dir / "heavy_score_decile_diagnostics.csv"),
            "summary": heavy_definition_summary,
        },
        "causal_diagnostic_assessment": {
            "causal_claim_allowed": False,
            "validated_hypotheses_n": validated_n,
            "not_testable_hypotheses_n": not_testable_n,
            "guardrails": [
                "Resultados são associacionais e não devem ser tratados como causalidade.",
                "Sem experimento ou quase-experimento, usar estes sinais para diagnóstico, não para prova de mecanismo.",
            ],
        },
        "limitations": [
            "Cobertura de identidade entre sistemas é parcial e varia por fonte.",
            "Sem ponte de identidade Hotjar->professor, hipóteses de feedback individual ficam não testáveis.",
            "Métricas refletem o snapshot disponível; devem ser reavaliadas a cada atualização de base.",
        ],
        "final_conclusions": [
            f"Score de confiança de qualidade: {round(float(trust_score), 2)}/100.",
            f"Distribuição de status das hipóteses: {status_counts(hypothesis_df)}.",
            f"Definição heavy (fast_v1): q={heavy_definition_summary.get('selected_threshold_quantile')} | heavy_share_active={heavy_definition_summary.get('heavy_share_active_heavy_window')}.",
            "Todas as métricas desta etapa são calculadas sem alvo/modelo de inatividade.",
        ],
    }

    write_json(cfg.output_dir / "consolidated_status.json", consolidated)

    md_lines = [
        "# Relatório técnico consolidado (etapa 01)",
        "",
        f"- Data/hora UTC: {consolidated['run_metadata']['run_timestamp_utc']}",
        f"- Snapshot: {consolidated['run_metadata']['snapshot_ts']}",
        f"- data_dir: `{cfg.data_dir}`",
        f"- output_dir: `{cfg.output_dir}`",
        "",
        "## Métricas principais",
        f"- state_missing_pct: {summary['state_missing_pct']:.4f}",
        f"- utm_missing_pct: {summary['utm_missing_pct']:.4f}",
        f"- short_sessions_rate_le_5s: {summary['short_sessions_rate_le_5s']:.4f}",
        f"- return_gap_median_days: {summary['return_gap_median_days']:.3f}",
        f"- recent_6m_mau_interactions_slope_users_per_month: {summary['recent_6m_mau_interactions_slope_users_per_month']:.3f}",
        f"- retention_recent_avg_6m: {summary['retention_recent_avg_6m']:.4f}",
        "",
        "## Hipóteses",
        f"- status_counts: {status_counts(hypothesis_df)}",
        "",
        "## Guardrails",
    ]
    for g in consolidated["causal_diagnostic_assessment"]["guardrails"]:
        md_lines.append(f"- {g}")

    write_markdown(cfg.output_dir / "consolidated_status.md", md_lines)
    write_markdown(cfg.output_dir / "relatorio_final_rigoroso.md", md_lines)

    LOGGER.info("Pipeline finished successfully | output=%s", cfg.output_dir)


if __name__ == "__main__":
    main()
