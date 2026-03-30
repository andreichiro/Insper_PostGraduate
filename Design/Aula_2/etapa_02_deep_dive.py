from __future__ import annotations

"""
Etapa 02 - Deep dive complementar 

Gera recortes adicionais a partir dos artefatos da etapa 01,
com foco em jornada, segmentos, e intensidade de uso
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import linregress


LOGGER = logging.getLogger("etapa_02_deep_dive")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")


@dataclass(frozen=True)
class DeepDiveConfig:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    random_seed: int = 42


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa deep dive complementar sem métricas de inatividade.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DeepDiveConfig:
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir if args.data_dir is not None else base_dir / "base_aprendizap").resolve()
    output_dir = (args.output_dir if args.output_dir is not None else base_dir / "analysis_output").resolve()
    return DeepDiveConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        random_seed=int(args.random_seed),
    )


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def load_teacher_dataset(output_dir: Path) -> pd.DataFrame:
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
    raise FileNotFoundError(
        f"teacher dataset não encontrado. Esperado em {p_parquet}, {p_csv_full} ou {p_csv_sample}. Execute etapa_01_base.py antes."
    )


def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def compute_session_quality_by_profile(teacher_df: pd.DataFrame) -> pd.DataFrame:
    df = teacher_df.copy()
    if "activity_tier" not in df.columns:
        df["activity_tier"] = "desconhecido"

    df["short_session_proxy"] = (pd.to_numeric(df.get("avg_session_min", np.nan), errors="coerce") <= (5.0 / 60.0)).astype(float)

    out = (
        df.groupby("activity_tier", dropna=False)
        .agg(
            teachers=("unique_id", "count"),
            median_avg_session_min=("avg_session_min", "median"),
            p90_total_session_min=("total_session_min", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), 90)) if len(s) else np.nan),
            avg_interactions=("interaction_count", "mean"),
            conversion_rate=("converted_within_window", "mean"),
            short_session_proxy_rate=("short_session_proxy", "mean"),
        )
        .reset_index()
        .sort_values("teachers", ascending=False)
    )
    return out


def compute_state_and_utm_rankings(state_stats: pd.DataFrame, utm_stats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    st = state_stats.copy()
    ut = utm_stats.copy()

    if not st.empty and "teachers" in st.columns:
        st = st[st["teachers"] >= 500].copy()
        st = st.sort_values(["conversion_rate", "teachers"], ascending=[False, False])
        if len(st) >= 2:
            st["delta_vs_median_conversion"] = st["conversion_rate"] - st["conversion_rate"].median()

    if not ut.empty and "teachers" in ut.columns:
        ut = ut[ut["teachers"] >= 300].copy()
        ut = ut.sort_values(["conversion_rate", "teachers"], ascending=[False, False])
        if len(ut) >= 2:
            ut["delta_vs_median_conversion"] = ut["conversion_rate"] - ut["conversion_rate"].median()

    return st, ut


def compute_journey_lag(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = duckdb.connect(database=":memory:")
    conn.execute(
        f"CREATE VIEW dim_teachers AS SELECT * FROM read_csv('{q(data_dir / 'dim_teachers.csv')}', delim=';', header=true, ignore_errors=true)"
    )
    conn.execute(
        f"CREATE VIEW interactions AS SELECT * FROM read_csv_auto('{q(data_dir / 'fct_teachers_contents_interactions.csv')}', header=true)"
    )

    journey = conn.execute(
        """
        WITH interactions_dim AS (
          SELECT i.*
          FROM interactions i
          INNER JOIN dim_teachers d USING(unique_id)
          WHERE i.data_inicio IS NOT NULL
            AND lower(coalesce(i.user_type,''))='registered'
        ),
        agg AS (
          SELECT
            unique_id,
            MIN(CASE WHEN lower(coalesce(event_type,'')) LIKE '%aula%' THEN data_inicio END) AS first_aula_ts,
            MIN(CASE WHEN lower(coalesce(event_type,'')) LIKE '%prova%' THEN data_inicio END) AS first_prova_ts
          FROM interactions_dim
          GROUP BY unique_id
        )
        SELECT
          unique_id,
          first_aula_ts,
          first_prova_ts,
          (epoch(first_prova_ts) - epoch(first_aula_ts))/86400.0 AS lag_days_prova_minus_aula
        FROM agg
        WHERE first_aula_ts IS NOT NULL AND first_prova_ts IS NOT NULL
        """
    ).fetchdf()
    conn.close()

    if journey.empty:
        return (
            pd.DataFrame(columns=["path", "teachers"]),
            pd.DataFrame(columns=["metric", "value"]),
        )

    journey["path"] = np.select(
        [journey["lag_days_prova_minus_aula"] > 0, journey["lag_days_prova_minus_aula"] < 0],
        ["aula_then_prova", "prova_then_aula"],
        default="same_day",
    )

    path_counts = journey["path"].value_counts(dropna=False).reset_index()
    path_counts.columns = ["path", "teachers"]

    lag_summary = pd.DataFrame(
        [
            {"metric": "teachers_with_both", "value": int(len(journey))},
            {"metric": "median_lag_days", "value": float(journey["lag_days_prova_minus_aula"].median())},
            {"metric": "p90_abs_lag_days", "value": float(np.percentile(np.abs(journey["lag_days_prova_minus_aula"]), 90))},
            {
                "metric": "share_aula_then_prova",
                "value": float((journey["path"] == "aula_then_prova").mean()),
            },
        ]
    )
    return path_counts, lag_summary


def compute_usage_trend(users_panel: pd.DataFrame) -> pd.DataFrame:
    panel = users_panel.copy()
    if panel.empty or "month" not in panel.columns:
        return pd.DataFrame(columns=["window", "slope_users_per_month", "r_value", "p_value", "n_months"])

    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    panel = panel.sort_values("month")
    panel = panel[panel["mau_registered_interactions"].notna()].copy()

    rows: List[Dict[str, float]] = []
    for window in [6, 9, 12]:
        win = panel.tail(window)
        if len(win) >= 3:
            x = np.arange(len(win), dtype=float)
            y = win["mau_registered_interactions"].astype(float).to_numpy()
            lr = linregress(x, y)
            rows.append(
                {
                    "window": window,
                    "slope_users_per_month": float(lr.slope),
                    "r_value": float(lr.rvalue),
                    "p_value": float(lr.pvalue),
                    "n_months": int(len(win)),
                }
            )
    return pd.DataFrame(rows)


def compute_heavy_mix(teacher_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = teacher_df.copy()
    if "heavy_user_flag" not in df.columns:
        df["heavy_user_flag"] = 0

    if "active_user_heavy_window_flag" in df.columns:
        df["active_user_heavy_window_flag"] = pd.to_numeric(df["active_user_heavy_window_flag"], errors="coerce").fillna(0).astype(int)
        scope = df[df["active_user_heavy_window_flag"] == 1].copy()
        population_base = "active_user_heavy_window_flag=1"
    elif "interaction_count" in df.columns:
        df["interaction_count"] = pd.to_numeric(df["interaction_count"], errors="coerce").fillna(0.0)
        scope = df[df["interaction_count"] > 0].copy()
        population_base = "interaction_count>0"
    else:
        scope = df.copy()
        population_base = "all_teacher_rows"

    heavy = scope[scope["heavy_user_flag"] == 1].copy()
    base = scope[scope["heavy_user_flag"] == 0].copy()

    def mix_by(col: str) -> pd.DataFrame:
        if col not in scope.columns:
            return pd.DataFrame(columns=[col, "teachers", "heavy_teachers", "heavy_share_within_group"])

        all_counts = scope[col].fillna("missing").astype(str).value_counts().rename_axis(col).reset_index(name="teachers")
        heavy_counts = heavy[col].fillna("missing").astype(str).value_counts().rename_axis(col).reset_index(name="heavy_teachers")
        out = all_counts.merge(heavy_counts, on=col, how="left")
        out["heavy_teachers"] = out["heavy_teachers"].fillna(0).astype(int)
        out["heavy_share_within_group"] = out["heavy_teachers"] / out["teachers"].replace(0, np.nan)
        return out.sort_values("teachers", ascending=False)

    summary = pd.DataFrame(
        [
            {
                "segment": "heavy",
                "teachers": int(len(heavy)),
                "share": float(len(heavy) / len(scope)) if len(scope) else np.nan,
                "median_interactions": float(heavy["interaction_count"].median()) if not heavy.empty else np.nan,
                "conversion_rate": float(heavy["converted_within_window"].mean()) if not heavy.empty else np.nan,
                "population_base": population_base,
            },
            {
                "segment": "base_regular",
                "teachers": int(len(base)),
                "share": float(len(base) / len(scope)) if len(scope) else np.nan,
                "median_interactions": float(base["interaction_count"].median()) if not base.empty else np.nan,
                "conversion_rate": float(base["converted_within_window"].mean()) if not base.empty else np.nan,
                "population_base": population_base,
            },
        ]
    )

    return {
        "heavy_summary": summary,
        "heavy_mix_device": mix_by("primary_device"),
        "heavy_mix_utm": mix_by("utm_group"),
        "heavy_mix_state": mix_by("estado_group"),
    }


def compute_cluster_profiles_detailed(
    teacher_df: pd.DataFrame,
    random_seed: int,
    max_cluster_sample: int = 50_000,
) -> pd.DataFrame:
    cluster_col = "behavior_cluster_id" if "behavior_cluster_id" in teacher_df.columns else ("cluster" if "cluster" in teacher_df.columns else None)
    if cluster_col is None:
        return pd.DataFrame(
            columns=[
                "cluster",
                "teachers",
                "share",
                "conversion_rate",
                "median_interactions",
                "median_session_min",
                "avg_time_to_first_value_hours",
                "heavy_share",
                "top_device",
                "top_utm",
                "top_estado",
                "cluster_intensity_median",
                "cluster_is_heavy",
                "silhouette_model",
                "best_k",
                "cluster_train_sample",
                "cluster_feature_cols",
            ]
        )

    df = teacher_df.copy()
    df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce").fillna(-1).astype(int)
    df = df[df[cluster_col] >= 0].copy()
    if df.empty:
        return pd.DataFrame()

    total = len(df)
    best_k_series = pd.to_numeric(df.get("cluster_best_k"), errors="coerce").dropna() if "cluster_best_k" in df.columns else pd.Series(dtype=float)
    silhouette_series = pd.to_numeric(df.get("cluster_silhouette"), errors="coerce").dropna() if "cluster_silhouette" in df.columns else pd.Series(dtype=float)
    train_sample_series = pd.to_numeric(df.get("cluster_train_sample_n"), errors="coerce").dropna() if "cluster_train_sample_n" in df.columns else pd.Series(dtype=float)
    feature_set = ""
    if "cluster_feature_set" in df.columns and df["cluster_feature_set"].notna().any():
        feature_set = str(df["cluster_feature_set"].dropna().iloc[0]).strip()

    best_k = int(best_k_series.iloc[0]) if not best_k_series.empty else int(df[cluster_col].nunique())
    best_score = float(silhouette_series.iloc[0]) if not silhouette_series.empty else np.nan
    cluster_train_sample = int(train_sample_series.iloc[0]) if not train_sample_series.empty else int(total)

    rows: List[Dict[str, object]] = []
    for cluster_id, grp in df.groupby(cluster_col):
        top_device = grp["primary_device"].fillna("missing").astype(str).value_counts().index[0] if "primary_device" in grp.columns and len(grp) else "missing"
        top_utm = grp["utm_group"].fillna("missing").astype(str).value_counts().index[0] if "utm_group" in grp.columns and len(grp) else "missing"
        top_estado = grp["estado_group"].fillna("missing").astype(str).value_counts().index[0] if "estado_group" in grp.columns and len(grp) else "missing"
        cluster_is_heavy = False
        if "heavy_cluster_flag" in grp.columns:
            cluster_is_heavy = bool(pd.to_numeric(grp["heavy_cluster_flag"], errors="coerce").fillna(0).mean() > 0.5)
        rows.append(
            {
                "cluster": int(cluster_id),
                "teachers": int(len(grp)),
                "share": float(len(grp) / total) if total else np.nan,
                "conversion_rate": float(grp["converted_within_window"].mean()) if "converted_within_window" in grp.columns else np.nan,
                "median_interactions": float(grp["interaction_count"].median()) if "interaction_count" in grp.columns else np.nan,
                "median_session_min": float(grp["avg_session_min"].median()) if "avg_session_min" in grp.columns else np.nan,
                "avg_time_to_first_value_hours": float(grp["time_to_first_value_hours"].mean()) if "time_to_first_value_hours" in grp.columns else np.nan,
                "heavy_share": float(grp["heavy_user_flag"].mean()) if "heavy_user_flag" in grp.columns else np.nan,
                "cluster_intensity_median": float(grp["engagement_intensity_score"].median()) if "engagement_intensity_score" in grp.columns else np.nan,
                "cluster_is_heavy": cluster_is_heavy,
                "top_device": top_device,
                "top_utm": top_utm,
                "top_estado": top_estado,
                "silhouette_model": best_score,
                "best_k": best_k,
                "cluster_train_sample": cluster_train_sample,
                "cluster_feature_cols": feature_set,
            }
        )

    return pd.DataFrame(rows).sort_values("teachers", ascending=False).reset_index(drop=True)


def main() -> None:
    setup_logging()
    cfg = build_config(parse_args())

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running deep dive | data_dir=%s | output_dir=%s", cfg.data_dir, cfg.output_dir)

    teacher_df = load_teacher_dataset(cfg.output_dir)

    state_stats = load_optional_csv(cfg.output_dir / "state_stats.csv")
    utm_stats = load_optional_csv(cfg.output_dir / "utm_stats.csv")
    users_panel = load_optional_csv(cfg.output_dir / "users_monthly_panel.csv")
    retention = load_optional_csv(cfg.output_dir / "retention_monthly_entries.csv")
    hypothesis_df = load_optional_csv(cfg.output_dir / "hypothesis_results.csv")
    geo_associations = load_optional_csv(cfg.output_dir / "geo_associations.csv")
    top_corr = load_optional_csv(cfg.output_dir / "top_corr_pairs.csv")
    cat_corr = load_optional_csv(cfg.output_dir / "cat_corr_pairs.csv")

    session_quality = compute_session_quality_by_profile(teacher_df)
    state_rank, utm_rank = compute_state_and_utm_rankings(state_stats, utm_stats)
    path_counts, lag_summary = compute_journey_lag(cfg.data_dir)
    trend = compute_usage_trend(users_panel)
    heavy_mix = compute_heavy_mix(teacher_df)
    cluster_detailed = compute_cluster_profiles_detailed(teacher_df, random_seed=cfg.random_seed)

    hyp_summary = pd.DataFrame()
    if not hypothesis_df.empty:
        cols = [
            c
            for c in [
                "hypothesis_id",
                "statement",
                "status",
                "p_value",
                "effect_size",
                "evidence",
                "interpretation",
            ]
            if c in hypothesis_df.columns
        ]
        hyp_summary = hypothesis_df[cols].copy()

    outputs: Dict[str, pd.DataFrame] = {
        "deep_dive_session_quality_by_profile": session_quality,
        "deep_dive_state_conversion_ranking": state_rank,
        "deep_dive_utm_conversion_ranking": utm_rank,
        "deep_dive_journey_path_counts": path_counts,
        "deep_dive_journey_lag_summary": lag_summary,
        "deep_dive_usage_trend_windows": trend,
        "deep_dive_cluster_profiles_detailed": cluster_detailed,
        "deep_dive_hypothesis_summary": hyp_summary,
        "deep_dive_geo_associations": geo_associations,
        "deep_dive_top_corr_pairs": top_corr,
        "deep_dive_cat_corr_pairs": cat_corr,
        "deep_dive_heavy_summary": heavy_mix["heavy_summary"],
        "deep_dive_heavy_mix_device": heavy_mix["heavy_mix_device"],
        "deep_dive_heavy_mix_utm": heavy_mix["heavy_mix_utm"],
        "deep_dive_heavy_mix_state": heavy_mix["heavy_mix_state"],
        "deep_dive_retention_monthly_entries": retention,
        "deep_dive_users_monthly_panel": users_panel,
    }

    produced_files: List[str] = []
    for name, df in outputs.items():
        path = cfg.output_dir / f"{name}.csv"
        if df is None:
            continue
        df.to_csv(path, index=False)
        produced_files.append(path.name)

    feature_cols_consistency: List[str] = []
    if not cluster_detailed.empty and "cluster_feature_cols" in cluster_detailed.columns and cluster_detailed["cluster_feature_cols"].notna().any():
        raw_feature_str = str(cluster_detailed["cluster_feature_cols"].dropna().iloc[0]).strip()
        if raw_feature_str:
            feature_cols_consistency = [x.strip() for x in raw_feature_str.split(",") if str(x).strip()]

    summary = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "output_dir": str(cfg.output_dir),
        "source_teacher_rows": int(len(teacher_df)),
        "files_generated": produced_files,
        "hypothesis_status_counts": hypothesis_df["status"].value_counts().to_dict() if (not hypothesis_df.empty and "status" in hypothesis_df.columns) else {},
        "cluster_definition_consistency": {
            "method": "Reuso da clusterização de subtipos heavy da etapa 01 (KMeans em heavy_users com features de mix comportamental).",
            "feature_columns": feature_cols_consistency,
            "best_k": int(pd.to_numeric(cluster_detailed["best_k"], errors="coerce").dropna().iloc[0])
            if (not cluster_detailed.empty and "best_k" in cluster_detailed.columns and cluster_detailed["best_k"].notna().any())
            else None,
            "silhouette_model": float(pd.to_numeric(cluster_detailed["silhouette_model"], errors="coerce").dropna().iloc[0])
            if (not cluster_detailed.empty and "silhouette_model" in cluster_detailed.columns and cluster_detailed["silhouette_model"].notna().any())
            else None,
        },
        "notes": [
            "Deep dive calculado sem métricas ou modelos de inatividade.",
            "Todos os artefatos derivam de transformações explícitas da base e da etapa 01.",
        ],
    }

    (cfg.output_dir / "deep_dive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Deep dive finished successfully.")


if __name__ == "__main__":
    main()
