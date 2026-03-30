#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
DEFAULT_SOURCE_DIR = DEFAULT_BASE_DIR / "analysis_output_v3_tmp"
DEFAULT_OUTPUT_NAME = "explainable_drift_prediction_v2"

RECENT_WINDOW_MONTHS = 6
PROFILE_CONTROL_VAR = "teacher_currentsubject_group"
CONTROL_MIN_GROUP_ROWS = 100

DRIFT_NUMERIC_HIGH_PSI = 0.25
DRIFT_NUMERIC_HIGH_SMD = 0.50
DRIFT_NUMERIC_MEDIUM_PSI = 0.10
DRIFT_NUMERIC_MEDIUM_SMD = 0.25
DRIFT_CATEGORICAL_HIGH_TV = 0.15
DRIFT_CATEGORICAL_HIGH_DIFF_PP = 10.0
DRIFT_CATEGORICAL_MEDIUM_TV = 0.08
DRIFT_CATEGORICAL_MEDIUM_DIFF_PP = 5.0

PUBLIC_TABLE_MAP = {
    "audit_base_modelada_validation": "audit_base_modelada_validation",
    "audit_persona_feature_readiness": "audit_persona_feature_readiness_final_v2",
    "base_modelada_v2": "base_modelada_v2",
    "dim_calendar": "dim_calendar_final_v2",
    "dim_device": "dim_device_final_v2",
    "dim_event": "dim_event_final_v2",
    "dim_persona_range_candidates": "dim_persona_range_candidates_final_v2",
    "dim_teacher": "dim_teacher_final_v2",
    "fct_formation_clean": "fct_formation_clean_final_v2",
    "fct_interaction_clean": "fct_interaction_clean_final_v2",
    "fct_session_clean": "fct_session_clean_final_v2",
    "fct_teacher_month": "fct_teacher_month_final_v2",
    "mart_teacher_cluster_ready": "mart_teacher_cluster_ready_final_v2",
    "mart_teacher_month_cluster_ready": "mart_teacher_month_cluster_ready_final_v2",
    "mart_teacher_month_panel": "mart_teacher_month_panel_final_v2",
    "mart_teacher_month_persona_ready": "mart_teacher_month_persona_ready_final_v2",
    "mart_teacher_persona_ready": "mart_teacher_persona_ready_final_v2",
}

TABLE_GRAINS = {
    "audit_base_modelada_validation": "check_name",
    "audit_persona_feature_readiness": "feature_name x feature_level",
    "base_modelada_v2": "teacher_unique_id x month",
    "dim_calendar": "month_start x uf x rede",
    "dim_device": "device_group",
    "dim_event": "event_type",
    "dim_persona_range_candidates": "feature_name x feature_level",
    "dim_teacher": "teacher_unique_id",
    "fct_formation_clean": "formation_row_hash",
    "fct_interaction_clean": "interaction_row_hash",
    "fct_session_clean": "session_row_hash",
    "fct_teacher_month": "teacher_unique_id x month",
    "mart_teacher_cluster_ready": "teacher_unique_id",
    "mart_teacher_month_cluster_ready": "teacher_unique_id x month",
    "mart_teacher_month_panel": "teacher_unique_id x month",
    "mart_teacher_month_persona_ready": "teacher_unique_id x month",
    "mart_teacher_persona_ready": "teacher_unique_id",
}

TABLE_ROLES = {
    "audit_base_modelada_validation": "quality gate for all downstream interpretation",
    "audit_persona_feature_readiness": "feature readiness and caveat dictionary",
    "base_modelada_v2": "upstream canonical monthly teacher base",
    "dim_calendar": "calendar enrichment and temporal context",
    "dim_device": "device taxonomy upstream of monthly device flags",
    "dim_event": "event taxonomy upstream of monthly event families",
    "dim_persona_range_candidates": "data-driven cut points for explainable thresholds",
    "dim_teacher": "teacher profile and control variable source",
    "fct_formation_clean": "secondary learning/formation source, reviewed but not used directly here",
    "fct_interaction_clean": "event-level source upstream of monthly behavioral aggregates",
    "fct_session_clean": "session telemetry source upstream of session aggregates",
    "fct_teacher_month": "upstream monthly fact used to build persona/panel marts",
    "mart_teacher_cluster_ready": "teacher-level clustering layer, reviewed for context only",
    "mart_teacher_month_cluster_ready": "monthly clustering layer, reviewed for context only",
    "mart_teacher_month_panel": "densified monthly panel for context and signal-gap interpretation",
    "mart_teacher_month_persona_ready": "main modeling table for drift and prediction",
    "mart_teacher_persona_ready": "teacher-level summary used for context and coverage review",
}

LEAKAGE_FEATURES = {
    "strict_user_flag",
    "returned_active_m1",
    "returned_any_download_m1",
    "returned_strict_value_m1",
    "strict_return_value_m1",
    "next_month_observed_flag",
    "target_return_active_m1",
    "target_churn_m1",
}

CONTEXT_ONLY_FEATURES = {
    "teacher_estado",
    "teacher_utm_group",
    "teacher_currentstage",
}

TELEMETRY_SUPPORT_FEATURES = {
    "raw_entry_session_count_month",
    "ping_entry_session_count_month",
    "clean_entry_session_count_month",
    "clean_entry_total_session_minutes_month",
    "clean_entry_avg_session_minutes_month",
    "entry_signal_flag",
    "clean_entry_signal_flag",
    "only_ping_entry_flag",
    "clean_entry_exposed_no_download_flag",
    "clean_entry_exposed_no_activity_no_download_flag",
    "clean_entry_exposed_activity_no_download_flag",
}


@dataclass(frozen=True)
class Config:
    base_dir: Path
    source_dir: Path
    source_duckdb_path: Path
    output_dir: Path
    output_duckdb_path: Path


def build_config(
    base_dir: Path | None = None,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Config:
    base = (base_dir or DEFAULT_BASE_DIR).resolve()
    source = (source_dir or DEFAULT_SOURCE_DIR).resolve()
    output = (output_dir or (source / DEFAULT_OUTPUT_NAME)).resolve()
    output_db_name = output.name or DEFAULT_OUTPUT_NAME
    return Config(
        base_dir=base,
        source_dir=source,
        source_duckdb_path=source / "duckdb" / "base_modelada_v2.duckdb",
        output_dir=output,
        output_duckdb_path=output / "duckdb" / f"{output_db_name}.duckdb",
    )


def ensure_output_dirs(output_dir: Path) -> None:
    for subdir in ["parquet", "json", "audit", "reports", "duckdb"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def connect_source(cfg: Config) -> duckdb.DuckDBPyConnection:
    if not cfg.source_duckdb_path.exists():
        raise FileNotFoundError(f"Source DuckDB not found: {cfg.source_duckdb_path}")
    return duckdb.connect(str(cfg.source_duckdb_path), read_only=True)


def connect_output(cfg: Config) -> duckdb.DuckDBPyConnection:
    ensure_output_dirs(cfg.output_dir)
    return duckdb.connect(str(cfg.output_duckdb_path))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def persist_table(conn_out: duckdb.DuckDBPyConnection, cfg: Config, table_name: str, df: pd.DataFrame) -> None:
    conn_out.register(f"_{table_name}_df", df)
    conn_out.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _{table_name}_df")
    parquet_path = cfg.output_dir / "parquet" / f"{table_name}.parquet"
    parquet_sql = str(parquet_path).replace("'", "''")
    conn_out.execute(f"COPY {table_name} TO '{parquet_sql}' (FORMAT PARQUET)")


def normalize_text(series: pd.Series, default: str = "missing") -> pd.Series:
    out = series.fillna(default).astype(str).str.strip()
    return out.replace({"": default, "None": default, "nan": default, "<missing>": default})


def attach_reference(
    df: pd.DataFrame,
    source_tables: Sequence[str],
    build_summary: str,
    rebuild_from_raw: str,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["source_tables"] = []
        out["build_summary"] = []
        out["rebuild_from_raw"] = []
        return out
    out = df.copy()
    out.insert(0, "rebuild_from_raw", rebuild_from_raw)
    out.insert(0, "build_summary", build_summary)
    out.insert(0, "source_tables", "; ".join(source_tables))
    return out


def strip_reference_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["source_tables", "build_summary", "rebuild_from_raw"], errors="ignore").copy()


def load_public_tables(conn: duckdb.DuckDBPyConnection) -> Dict[str, pd.DataFrame]:
    existing = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    missing = [physical for physical in PUBLIC_TABLE_MAP.values() if physical not in existing]
    if missing:
        raise RuntimeError(f"Missing required source tables: {', '.join(sorted(missing))}")
    return {
        public_name: conn.execute(f"SELECT * FROM {physical_name}").fetchdf()
        for public_name, physical_name in PUBLIC_TABLE_MAP.items()
    }


def build_input_map(
    tables: Dict[str, pd.DataFrame],
    direct_tables: Sequence[str],
    flag_column: str,
    analysis_summary: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    direct_set = set(direct_tables)
    for public_name, physical_name in PUBLIC_TABLE_MAP.items():
        rows.append(
            {
                "public_table_name": public_name,
                "physical_table_name": physical_name,
                "grain": TABLE_GRAINS[public_name],
                "row_count": int(len(tables[public_name])),
                flag_column: int(public_name in direct_set),
                "role_in_this_analysis": TABLE_ROLES[public_name],
                "why_not_direct_if_zero": "" if public_name in direct_set else "reviewed as upstream lineage/context only",
            }
        )
    df = pd.DataFrame(rows).sort_values("public_table_name").reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=list(PUBLIC_TABLE_MAP.keys()),
        build_summary=analysis_summary,
        rebuild_from_raw="Run raw_para_base_modelada_v4.py to rebuild the relevant-table layer, then rerun the explainable drift/prediction scripts.",
    )


def prepare_model_population(tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    month = tables["mart_teacher_month_persona_ready"].copy()
    validation = tables["audit_base_modelada_validation"].copy()

    month["month"] = pd.to_datetime(month["month"], errors="coerce")
    numeric_exceptions = {
        "teacher_unique_id",
        "month",
        "month_signal_class",
        "teacher_population_status",
        "teacher_estado",
        "teacher_currentsubject_group",
        "teacher_currentstage",
        "teacher_utm_group",
    }
    for col in [col for col in month.columns if col not in numeric_exceptions]:
        month[col] = pd.to_numeric(month[col], errors="coerce")
    for col in [
        "month_signal_class",
        "teacher_population_status",
        "teacher_estado",
        "teacher_currentsubject_group",
        "teacher_currentstage",
        "teacher_utm_group",
    ]:
        month[col] = normalize_text(month[col])

    population = month[
        (month["observed_month_flag"].fillna(0) == 1)
        & (month["persona_analysis_eligible_flag"].fillna(0) == 1)
        & (month["next_month_observed_flag"].fillna(0) == 1)
    ].copy()
    population["target_return_active_m1"] = (population["returned_active_m1"].fillna(0) == 1).astype(int)
    population["target_churn_m1"] = (population["returned_active_m1"].fillna(0) == 0).astype(int)

    failed_checks = validation[normalize_text(validation["status"]) == "fail"].copy()
    summary = pd.DataFrame(
        [
            {
                "population_name": "model_population",
                "definition": "Observed teacher-month rows with persona_analysis_eligible_flag = 1 and next_month_observed_flag = 1.",
                "rows": int(len(population)),
                "teachers": int(population["teacher_unique_id"].nunique()),
                "month_start": str(population["month"].min()),
                "month_end": str(population["month"].max()),
                "return_rate_m1": float(population["target_return_active_m1"].mean()),
                "churn_rate_m1": float(population["target_churn_m1"].mean()),
                "failed_validation_checks": int(len(failed_checks)),
            }
        ]
    )
    summary = attach_reference(
        summary,
        source_tables=["mart_teacher_month_persona_ready", "audit_base_modelada_validation"],
        build_summary="Modeling population summary built from the monthly persona-ready mart after filtering to rows with a valid next-month outcome.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, load mart_teacher_month_persona_ready, filter observed rows where persona_analysis_eligible_flag = 1 and next_month_observed_flag = 1, then recompute counts and outcome rates.",
    )
    return population, summary


def build_feature_candidates(
    tables: Dict[str, pd.DataFrame],
    population: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    readiness = tables["audit_persona_feature_readiness"].copy()
    ranges = tables["dim_persona_range_candidates"].copy()
    readiness["feature_name"] = normalize_text(readiness["feature_name"])
    readiness["feature_level"] = normalize_text(readiness["feature_level"])
    readiness["feature_role"] = normalize_text(readiness["feature_role"])
    ranges["feature_name"] = normalize_text(ranges["feature_name"])
    ranges["feature_level"] = normalize_text(ranges["feature_level"])

    candidate_rows: List[Dict[str, Any]] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = ["month_signal_class", PROFILE_CONTROL_VAR]
    context_drift_features: List[str] = [
        "teacher_estado",
        "teacher_utm_group",
        "teacher_currentstage",
        PROFILE_CONTROL_VAR,
        "month_signal_class",
    ]

    for _, row in readiness.iterrows():
        feature_name = row["feature_name"]
        if row["feature_level"] != "teacher_month":
            continue
        if feature_name not in population.columns:
            continue
        is_leakage = int(feature_name in LEAKAGE_FEATURES)
        is_context_only = int(feature_name in CONTEXT_ONLY_FEATURES)
        include_as_control = int(feature_name == PROFILE_CONTROL_VAR)
        include_in_model = int((not is_leakage and not is_context_only) or include_as_control)
        model_usage = "predictor"
        if is_leakage:
            model_usage = "leakage_excluded"
        elif is_context_only:
            model_usage = "context_only"
        elif include_as_control:
            model_usage = "control_only"

        dtype_is_numeric = pd.api.types.is_numeric_dtype(population[feature_name])
        if include_in_model and dtype_is_numeric and feature_name not in categorical_features:
            numeric_features.append(feature_name)

        candidate_rows.append(
            {
                "feature_name": feature_name,
                "feature_level": row["feature_level"],
                "feature_role": row["feature_role"],
                "definition": row["definition"],
                "caveat": row["caveat"],
                "missing_rate": float(row["missing_rate"]),
                "zero_share": float(row["zero_share"]),
                "std": float(row["std"]),
                "recommended_for_persona_analysis": int(row["recommended_for_persona_analysis"]),
                "recommended_for_persona_ranges": int(row["recommended_for_persona_ranges"]),
                "recommended_for_behavior_clustering": int(row["recommended_for_behavior_clustering"]),
                "is_leakage_feature": is_leakage,
                "is_context_only_feature": is_context_only,
                "is_telemetry_support_feature": int(feature_name in TELEMETRY_SUPPORT_FEATURES),
                "include_as_control": include_as_control,
                "include_in_model": include_in_model,
                "model_usage": model_usage,
            }
        )

    extra_context = [
        PROFILE_CONTROL_VAR,
        "month_signal_class",
        "teacher_estado",
        "teacher_utm_group",
        "teacher_currentstage",
    ]
    existing_features = {row["feature_name"] for row in candidate_rows}
    for feature_name in extra_context:
        if feature_name not in population.columns or feature_name in existing_features:
            continue
        candidate_rows.append(
            {
                "feature_name": feature_name,
                "feature_level": "teacher_month",
                "feature_role": "context_interpretation",
                "definition": "Context variable kept for control or drift interpretation.",
                "caveat": "Use as control or interpretation support, not as the main behavior story.",
                "missing_rate": float(normalize_text(population[feature_name]).eq("missing").mean()),
                "zero_share": 0.0,
                "std": 0.0,
                "recommended_for_persona_analysis": 0,
                "recommended_for_persona_ranges": 0,
                "recommended_for_behavior_clustering": 0,
                "is_leakage_feature": 0,
                "is_context_only_feature": int(feature_name in CONTEXT_ONLY_FEATURES),
                "is_telemetry_support_feature": 0,
                "include_as_control": int(feature_name == PROFILE_CONTROL_VAR),
                "include_in_model": int(feature_name in {PROFILE_CONTROL_VAR, "month_signal_class"}),
                "model_usage": "control_only" if feature_name == PROFILE_CONTROL_VAR else ("predictor" if feature_name == "month_signal_class" else "context_only"),
            }
        )

    df = pd.DataFrame(candidate_rows).sort_values(
        ["include_in_model", "is_leakage_feature", "feature_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    df = df.merge(
        ranges[["feature_name", "feature_level", "p25", "p50", "p75", "p90", "note"]],
        on=["feature_name", "feature_level"],
        how="left",
    )
    df = attach_reference(
        df,
        source_tables=["audit_persona_feature_readiness", "dim_persona_range_candidates", "mart_teacher_month_persona_ready"],
        build_summary="Candidate-feature table built from the persona-readiness audit, range candidates, and explicit leakage/context screening rules.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, load audit_persona_feature_readiness and dim_persona_range_candidates, keep teacher-month features present in mart_teacher_month_persona_ready, exclude future-derived targets, and mark context-only variables separately.",
    )
    numeric_features = sorted(set(numeric_features))
    categorical_features = sorted(set([feature for feature in categorical_features if feature in population.columns]))
    context_drift_features = sorted(set([feature for feature in context_drift_features if feature in population.columns]))
    return df, numeric_features, categorical_features, context_drift_features


def chi_square_with_cramers_v(
    frame: pd.DataFrame,
    feature_col: str,
    target_col: str,
) -> Tuple[float, float, int, int]:
    work = frame[[feature_col, target_col]].copy()
    work[feature_col] = normalize_text(work[feature_col])
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce").fillna(0).astype(int)
    contingency = pd.crosstab(work[feature_col], work[target_col])
    if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return float("nan"), float("nan"), int(contingency.shape[0]), int(work.shape[0])
    chi2, p_value, _, _ = chi2_contingency(contingency)
    n = contingency.values.sum()
    phi2 = chi2 / max(n, 1)
    r, k = contingency.shape
    denom = max(min(k - 1, r - 1), 1)
    return float(p_value), float(np.sqrt(phi2 / denom)), int(contingency.shape[0]), int(work.shape[0])


def psi_numeric(baseline: pd.Series, recent: pd.Series, bins: int = 10) -> float:
    base = pd.to_numeric(baseline, errors="coerce").dropna()
    rec = pd.to_numeric(recent, errors="coerce").dropna()
    if base.empty or rec.empty:
        return float("nan")
    quantiles = np.unique(np.nanquantile(base, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    edges = [-np.inf] + list(quantiles[1:-1]) + [np.inf]
    base_bins = pd.cut(base, bins=edges, include_lowest=True).value_counts(sort=False)
    rec_bins = pd.cut(rec, bins=edges, include_lowest=True).value_counts(sort=False)
    base_share = np.clip(base_bins / max(len(base), 1), 1e-6, None)
    rec_share = np.clip(rec_bins / max(len(rec), 1), 1e-6, None)
    return float(np.sum((rec_share - base_share) * np.log(rec_share / base_share)))


def numeric_drift_level(psi: float, smd: float) -> str:
    if pd.isna(psi) or pd.isna(smd):
        return "insufficient_data"
    if psi >= DRIFT_NUMERIC_HIGH_PSI or abs(smd) >= DRIFT_NUMERIC_HIGH_SMD:
        return "high_drift"
    if psi >= DRIFT_NUMERIC_MEDIUM_PSI or abs(smd) >= DRIFT_NUMERIC_MEDIUM_SMD:
        return "medium_drift"
    return "low_drift"


def categorical_drift_level(total_variation: float, max_diff_pp: float) -> str:
    if pd.isna(total_variation) or pd.isna(max_diff_pp):
        return "insufficient_data"
    if total_variation >= DRIFT_CATEGORICAL_HIGH_TV or max_diff_pp >= DRIFT_CATEGORICAL_HIGH_DIFF_PP:
        return "high_drift"
    if total_variation >= DRIFT_CATEGORICAL_MEDIUM_TV or max_diff_pp >= DRIFT_CATEGORICAL_MEDIUM_DIFF_PP:
        return "medium_drift"
    return "low_drift"


def drift_rank(level: str) -> int:
    order = {"high_drift": 0, "medium_drift": 1, "low_drift": 2, "insufficient_data": 3}
    return order.get(level, 99)


def safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = pd.Series(y_true)
    if y.nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y, y_score))


def safe_average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = pd.Series(y_true)
    if y.nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y, y_score))


def safe_brier(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = pd.Series(y_true)
    if y.empty:
        return float("nan")
    return float(brier_score_loss(y, y_score))


def safe_log_loss(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = pd.Series(y_true)
    if y.nunique(dropna=True) < 2:
        return float("nan")
    clipped = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y, clipped))


def top_decile_lift(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    work = pd.DataFrame({"y": y_true, "score": y_score}).dropna()
    if work.empty or work["y"].nunique() < 2:
        return float("nan")
    threshold = work["score"].quantile(0.90)
    top_rate = float(work.loc[work["score"] >= threshold, "y"].mean())
    base_rate = float(work["y"].mean())
    if base_rate == 0:
        return float("nan")
    return top_rate / base_rate


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers: List[Tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(categorical_features),
            )
        )
    return ColumnTransformer(transformers=transformers)


def build_output_reference(
    cfg: Config,
    table_names: Iterable[str],
    build_summary: str,
) -> pd.DataFrame:
    rows = []
    for table_name in table_names:
        rows.append(
            {
                "table_name": table_name,
                "parquet_path": str(cfg.output_dir / "parquet" / f"{table_name}.parquet"),
                "duckdb_path": str(cfg.output_duckdb_path),
                "how_to_rebuild": "Run raw_para_base_modelada_v4.py first, then rerun the corresponding explainable drift/prediction script.",
            }
        )
    df = pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=list(PUBLIC_TABLE_MAP.keys()),
        build_summary=build_summary,
        rebuild_from_raw="Run raw_para_base_modelada_v4.py first, then rerun the explainable drift/prediction scripts; this table is generated as an output manifest.",
    )


def load_output_parquet(output_dir: Path, table_name: str) -> pd.DataFrame:
    path = output_dir / "parquet" / f"{table_name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
