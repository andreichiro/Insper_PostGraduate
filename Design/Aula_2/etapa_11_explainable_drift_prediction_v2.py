#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analytics_v2_common import (
    PALETTE,
    build_card_html,
    build_table_html,
    figure_to_html,
    fmt_num,
    render_report_html,
)


DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
DEFAULT_SOURCE_DIR = DEFAULT_BASE_DIR / "analysis_output_v3_tmp"
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
    "fct_formation_clean": "secondary learning/formation source, reviewed but not needed directly here",
    "fct_interaction_clean": "event-level source upstream of monthly behavioral aggregates",
    "fct_session_clean": "session telemetry source upstream of session aggregates",
    "fct_teacher_month": "upstream monthly fact used to build persona/panel marts",
    "mart_teacher_cluster_ready": "teacher-level clustering layer, reviewed for scope not direct modeling",
    "mart_teacher_month_cluster_ready": "monthly clustering layer, reviewed for scope not direct modeling",
    "mart_teacher_month_panel": "densified monthly panel for drift context and signal-gap interpretation",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused, explainable drift and prediction review from the relevant base tables."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    base_dir = args.base_dir.resolve()
    source_dir = args.source_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else source_dir / "focused_prediction_drift_v2"
    )
    source_duckdb_path = source_dir / "duckdb" / "base_modelada_v2.duckdb"
    output_duckdb_path = output_dir / "duckdb" / "focused_prediction_drift_v2.duckdb"
    return Config(
        base_dir=base_dir,
        source_dir=source_dir,
        source_duckdb_path=source_duckdb_path,
        output_dir=output_dir,
        output_duckdb_path=output_duckdb_path,
    )


def ensure_output_dirs(output_dir: Path) -> None:
    for subdir in ["csv", "parquet", "json", "audit", "reports", "duckdb"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


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


def write_markdown(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def persist_table(conn_out: duckdb.DuckDBPyConnection, cfg: Config, table_name: str, df: pd.DataFrame) -> None:
    conn_out.register(f"_{table_name}_df", df)
    conn_out.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _{table_name}_df")
    csv_path = cfg.output_dir / "csv" / f"{table_name}.csv"
    parquet_path = cfg.output_dir / "parquet" / f"{table_name}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def normalize_text(series: pd.Series, default: str = "missing") -> pd.Series:
    out = series.fillna(default).astype(str).str.strip()
    out = out.replace({"": default, "None": default, "nan": default, "<missing>": default})
    return out


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


def chi_square_with_cramers_v(frame: pd.DataFrame, feature_col: str, target_col: str) -> Tuple[float, float, int, int]:
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
    cramers_v = np.sqrt(phi2 / denom)
    return float(p_value), float(cramers_v), int(contingency.shape[0]), int(work.shape[0])


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
    order = {
        "high_drift": 0,
        "medium_drift": 1,
        "low_drift": 2,
        "insufficient_data": 3,
    }
    return order.get(level, 99)


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


def load_public_tables(conn: duckdb.DuckDBPyConnection) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    existing = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    missing = [physical for physical in PUBLIC_TABLE_MAP.values() if physical not in existing]
    if missing:
        raise RuntimeError(f"Missing required source tables: {', '.join(sorted(missing))}")
    for public_name, physical_name in PUBLIC_TABLE_MAP.items():
        tables[public_name] = conn.execute(f"SELECT * FROM {physical_name}").fetchdf()
    return tables


def build_input_map(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    direct_drift = {
        "audit_base_modelada_validation",
        "audit_persona_feature_readiness",
        "dim_persona_range_candidates",
        "dim_teacher",
        "mart_teacher_month_persona_ready",
    }
    direct_prediction = {
        "audit_base_modelada_validation",
        "audit_persona_feature_readiness",
        "dim_persona_range_candidates",
        "dim_teacher",
        "mart_teacher_month_persona_ready",
    }
    for public_name, physical_name in PUBLIC_TABLE_MAP.items():
        frame = tables[public_name]
        rows.append(
            {
                "public_table_name": public_name,
                "physical_table_name": physical_name,
                "grain": TABLE_GRAINS[public_name],
                "row_count": int(len(frame)),
                "used_directly_for_drift": int(public_name in direct_drift),
                "used_directly_for_prediction": int(public_name in direct_prediction),
                "role_in_this_analysis": TABLE_ROLES[public_name],
                "why_not_direct_if_zero": (
                    "reviewed as upstream lineage/context only"
                    if public_name not in direct_drift.union(direct_prediction)
                    else ""
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("public_table_name").reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=list(PUBLIC_TABLE_MAP.keys()),
        build_summary="Inventory of the declared relevant tables, their physical names, grain, and why each one is or is not used directly in this focused analysis.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py to build the relevant-table layer, then rerun etapa_11_explainable_drift_prediction_v2.py.",
    )


def prepare_model_population(tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    month = tables["mart_teacher_month_persona_ready"].copy()
    validation = tables["audit_base_modelada_validation"].copy()

    month["month"] = pd.to_datetime(month["month"], errors="coerce")

    numeric_cols = [
        col
        for col in month.columns
        if col not in {"teacher_unique_id", "month", "month_signal_class", "teacher_population_status", "teacher_estado", "teacher_currentsubject_group", "teacher_currentstage", "teacher_utm_group"}
    ]
    for col in numeric_cols:
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
    population_rows = pd.DataFrame(
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
    population_rows = attach_reference(
        population_rows,
        source_tables=["mart_teacher_month_persona_ready", "audit_base_modelada_validation"],
        build_summary="Modeling population summary built from the monthly persona-ready mart after filtering to rows with a valid next-month outcome.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, use mart_teacher_month_persona_ready, filter observed rows where persona_analysis_eligible_flag = 1 and next_month_observed_flag = 1, then compute row/teacher counts and target rates.",
    )
    return population, population_rows


def build_assumptions_table() -> pd.DataFrame:
    rows = [
        {
            "section": "drift",
            "assumption_id": "A1",
            "assumption": "Drift is evaluated on the same modeling population used for prediction: active observed teacher-months with next month observable.",
            "justification": "This keeps drift relevant to the exact population where prediction claims are made.",
            "what_changes_if_false": "If a broader population is used, drift values may be diluted by non-modeled months and become less decision-relevant.",
        },
        {
            "section": "drift",
            "assumption_id": "A2",
            "assumption": f"Old vs recent means first {RECENT_WINDOW_MONTHS} modeling months versus last {RECENT_WINDOW_MONTHS} modeling months.",
            "justification": "A symmetric early-vs-recent comparison is easy to explain and avoids arbitrary hand-picked dates.",
            "what_changes_if_false": "A different baseline window changes the magnitude of reported drift, especially for acquisition and maturity variables.",
        },
        {
            "section": "drift",
            "assumption_id": "A3",
            "assumption": "Numeric drift relevance uses PSI and standardized mean difference together; categorical drift relevance uses total variation and max share difference.",
            "justification": "Using both shape and central-tendency measures is more robust than a single metric.",
            "what_changes_if_false": "A single metric can miss practically important shifts or overreact to small but noisy changes.",
        },
        {
            "section": "prediction",
            "assumption_id": "B1",
            "assumption": "Predictors must be available at month t; future-derived outcomes are excluded as leakage.",
            "justification": "A valid prediction model cannot use information from month t+1.",
            "what_changes_if_false": "Model performance will be inflated and not deployable in practice.",
        },
        {
            "section": "prediction",
            "assumption_id": "B2",
            "assumption": f"The control variable is {PROFILE_CONTROL_VAR} and must be checked for coverage, slice size, and association with the target before use.",
            "justification": "A control variable that is mostly missing or statistically disconnected from the target adds confusion instead of rigor.",
            "what_changes_if_false": "Profile adjustment may be unstable or misleading.",
        },
        {
            "section": "prediction",
            "assumption_id": "B3",
            "assumption": "Models are tested with temporal train/test splits, not random splits.",
            "justification": "Temporal splits better approximate real deployment and are the right way to test drift-sensitive behavior.",
            "what_changes_if_false": "Random splits can overestimate performance by leaking future distributional information into training.",
        },
    ]
    df = pd.DataFrame(rows)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "audit_base_modelada_validation"],
        build_summary="Method assumptions table for the focused drift and prediction review.",
        rebuild_from_raw="No raw recomputation needed; this table documents the assumptions used by etapa_11_explainable_drift_prediction_v2.py.",
    )


def build_control_variable_validity(population: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    months = sorted(population["month"].dropna().unique().tolist())
    windows = {
        "all_history": population.copy(),
        "recent_6m": population[population["month"].isin(months[-RECENT_WINDOW_MONTHS:])].copy(),
    }
    for window_label, subset in windows.items():
        feature = normalize_text(subset[PROFILE_CONTROL_VAR])
        group_counts = feature.value_counts(dropna=False)
        missing_rate = float((feature == "missing").mean())
        min_group_rows = int(group_counts.min()) if not group_counts.empty else 0
        valid_group_share = float((group_counts >= CONTROL_MIN_GROUP_ROWS).mean()) if not group_counts.empty else 0.0
        for target_col, target_label in [
            ("target_churn_m1", "abandonar_m1"),
            ("target_return_active_m1", "retornar_ativo_m1"),
        ]:
            p_value, cramers_v, n_groups, n_rows = chi_square_with_cramers_v(subset, PROFILE_CONTROL_VAR, target_col)
            rows.append(
                {
                    "window_label": window_label,
                    "target": target_label,
                    "control_variable": PROFILE_CONTROL_VAR,
                    "rows": int(n_rows),
                    "distinct_groups": int(n_groups),
                    "missing_rate": missing_rate,
                    "min_group_rows": min_group_rows,
                    "share_groups_ge_min_rows": valid_group_share,
                    "chi_square_p_value": p_value,
                    "cramers_v": cramers_v,
                    "is_statistically_supported": int(
                        pd.notna(p_value)
                        and p_value < 0.05
                        and min_group_rows >= CONTROL_MIN_GROUP_ROWS
                        and missing_rate <= 0.20
                    ),
                    "plain_english_readout": (
                        "usable control"
                        if pd.notna(p_value) and p_value < 0.05 and min_group_rows >= CONTROL_MIN_GROUP_ROWS and missing_rate <= 0.20
                        else "weak or unstable control"
                    ),
                }
            )
    df = pd.DataFrame(rows)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher", "audit_base_modelada_validation"],
        build_summary="Statistical validity check of the chosen control variable using coverage, slice size, and chi-square association with the target.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, take mart_teacher_month_persona_ready rows in the modeling population, then test teacher_currentsubject_group against each target with chi-square and report coverage/slice counts.",
    )


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
    context_drift_features: List[str] = ["teacher_estado", "teacher_utm_group", "teacher_currentstage", PROFILE_CONTROL_VAR, "month_signal_class"]

    for _, row in readiness.iterrows():
        feature_name = row["feature_name"]
        if row["feature_level"] != "teacher_month":
            continue
        if feature_name not in population.columns:
            continue
        is_leakage = int(feature_name in LEAKAGE_FEATURES)
        is_context_only = int(feature_name in CONTEXT_ONLY_FEATURES)
        include_as_control = int(feature_name == PROFILE_CONTROL_VAR)
        include_for_model = int((not is_leakage and not is_context_only) or include_as_control)
        model_usage = "predictor"
        if is_leakage:
            model_usage = "leakage_excluded"
        elif is_context_only:
            model_usage = "context_only"
        elif include_as_control:
            model_usage = "control_only"
        dtype_is_numeric = pd.api.types.is_numeric_dtype(population[feature_name])
        if include_for_model and dtype_is_numeric and feature_name not in categorical_features:
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
                "include_in_model": include_for_model,
                "model_usage": model_usage,
            }
        )
    for feature_name in [PROFILE_CONTROL_VAR, "month_signal_class", "teacher_estado", "teacher_utm_group", "teacher_currentstage"]:
        if feature_name not in population.columns:
            continue
        if feature_name not in {row["feature_name"] for row in candidate_rows}:
            candidate_rows.append(
                {
                    "feature_name": feature_name,
                    "feature_level": "teacher_month",
                    "feature_role": "context_interpretation",
                    "definition": "Context variable kept for control or drift interpretation.",
                    "caveat": "Use as control or drift dimension, not as the core product-behavior story.",
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
                    "include_in_model": int(feature_name == PROFILE_CONTROL_VAR or feature_name == "month_signal_class"),
                    "model_usage": "control_only" if feature_name == PROFILE_CONTROL_VAR else ("predictor" if feature_name == "month_signal_class" else "context_only"),
                }
            )

    df = pd.DataFrame(candidate_rows).sort_values(["include_in_model", "is_leakage_feature", "feature_name"], ascending=[False, True, True]).reset_index(drop=True)
    range_cols = ["feature_name", "feature_level", "p25", "p50", "p75", "p90", "note"]
    df = df.merge(ranges[range_cols], on=["feature_name", "feature_level"], how="left")
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


def supported_control_windows(control_validity: pd.DataFrame) -> set[str]:
    if control_validity.empty:
        return set()
    grouped = (
        control_validity.groupby("window_label", as_index=False)["is_statistically_supported"]
        .min()
        .rename(columns={"is_statistically_supported": "all_targets_supported"})
    )
    return set(grouped.loc[grouped["all_targets_supported"] == 1, "window_label"].tolist())


def build_feature_screening(
    population: pd.DataFrame,
    feature_candidates: pd.DataFrame,
) -> pd.DataFrame:
    features = feature_candidates[feature_candidates["include_in_model"] == 1]["feature_name"].tolist()
    rows: List[Dict[str, Any]] = []
    base = population.copy()
    for feature_name in features:
        if feature_name not in base.columns:
            continue
        series = pd.to_numeric(base[feature_name], errors="coerce")
        if series.notna().sum() < 100:
            continue
        for target_col, target_label in [
            ("target_churn_m1", "abandonar_m1"),
            ("target_return_active_m1", "retornar_ativo_m1"),
        ]:
            y = base[target_col].astype(int)
            valid = series.notna() & y.notna()
            if valid.sum() < 100 or y[valid].nunique() < 2:
                continue
            x = series[valid]
            y_valid = y[valid]
            raw_auc = safe_auc(y_valid, x)
            separation_auc = max(raw_auc, 1 - raw_auc) if pd.notna(raw_auc) else float("nan")
            pos = x[y_valid == 1]
            neg = x[y_valid == 0]
            if pos.empty or neg.empty:
                p_value = float("nan")
            else:
                try:
                    p_value = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = float("nan")
            rows.append(
                {
                    "feature_name": feature_name,
                    "target": target_label,
                    "rows_used": int(valid.sum()),
                    "positive_class_mean": float(pos.mean()) if not pos.empty else float("nan"),
                    "negative_class_mean": float(neg.mean()) if not neg.empty else float("nan"),
                    "effect_direction": "higher_in_positive_class" if pos.mean() >= neg.mean() else "lower_in_positive_class",
                    "raw_auc": raw_auc,
                    "separation_auc": separation_auc,
                    "mann_whitney_p_value": p_value,
                    "is_univariate_signal": int(pd.notna(p_value) and p_value < 0.05 and pd.notna(separation_auc) and separation_auc >= 0.55),
                }
            )
    df = pd.DataFrame(rows).sort_values(["target", "separation_auc"], ascending=[True, False]).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "audit_persona_feature_readiness"],
        build_summary="Univariate screening of each model candidate against each target using AUC-style separation and Mann-Whitney significance.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, screen each included numeric feature against each target, and compute separation_auc plus Mann-Whitney p-values.",
    )


def build_numeric_drift(
    population: pd.DataFrame,
    numeric_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(population["month"].dropna().unique().tolist())
    baseline_months = months[:RECENT_WINDOW_MONTHS]
    recent_months = months[-RECENT_WINDOW_MONTHS:]
    baseline = population[population["month"].isin(baseline_months)].copy()
    recent = population[population["month"].isin(recent_months)].copy()

    feature_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []
    for feature_name in numeric_features:
        base = pd.to_numeric(baseline[feature_name], errors="coerce")
        rec = pd.to_numeric(recent[feature_name], errors="coerce")
        if base.notna().sum() == 0 or rec.notna().sum() == 0:
            continue
        pooled_std = np.sqrt((np.nanvar(base, ddof=0) + np.nanvar(rec, ddof=0)) / 2.0)
        base_mean = float(base.mean())
        rec_mean = float(rec.mean())
        if (pd.isna(pooled_std) or pooled_std == 0) and np.isclose(base_mean, rec_mean, equal_nan=True):
            smd = 0.0
        elif pooled_std and not np.isnan(pooled_std):
            smd = (rec_mean - base_mean) / pooled_std
        else:
            smd = float("nan")
        psi = psi_numeric(base, rec)
        level = numeric_drift_level(psi, smd)
        feature_rows.append(
            {
                "feature_name": feature_name,
                "baseline_month_start": str(min(baseline_months)),
                "baseline_month_end": str(max(baseline_months)),
                "recent_month_start": str(min(recent_months)),
                "recent_month_end": str(max(recent_months)),
                "baseline_rows": int(base.notna().sum()),
                "recent_rows": int(rec.notna().sum()),
                "baseline_mean": base_mean,
                "recent_mean": rec_mean,
                "baseline_median": float(base.median()),
                "recent_median": float(rec.median()),
                "mean_delta": rec_mean - base_mean,
                "standardized_mean_diff": smd,
                "psi": psi,
                "drift_level": level,
                "drift_relevance": "relevant" if level in {"high_drift", "medium_drift"} else "limited",
            }
        )
    for target_col, target_name in [
        ("target_churn_m1", "target_churn_m1"),
        ("target_return_active_m1", "target_return_active_m1"),
    ]:
        base = pd.to_numeric(baseline[target_col], errors="coerce")
        rec = pd.to_numeric(recent[target_col], errors="coerce")
        outcome_rows.append(
            {
                "metric_name": target_name,
                "baseline_month_start": str(min(baseline_months)),
                "baseline_month_end": str(max(baseline_months)),
                "recent_month_start": str(min(recent_months)),
                "recent_month_end": str(max(recent_months)),
                "baseline_rate": float(base.mean()),
                "recent_rate": float(rec.mean()),
                "rate_diff_pp": (float(rec.mean()) - float(base.mean())) * 100,
            }
        )
    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df["_drift_rank"] = feature_df["drift_level"].map(drift_rank)
        feature_df = feature_df.sort_values(["_drift_rank", "psi"], ascending=[True, False]).drop(columns="_drift_rank").reset_index(drop=True)
    feature_df = attach_reference(
        feature_df,
        source_tables=["mart_teacher_month_persona_ready", "audit_persona_feature_readiness", "dim_persona_range_candidates"],
        build_summary="Numeric drift between the first and last six modeling months, measured with mean/median change, standardized mean difference, and PSI.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, split first six versus last six modeling months, then compute mean/median deltas, SMD, and PSI for each selected numeric feature.",
    )
    outcome_df = pd.DataFrame(outcome_rows)
    outcome_df = attach_reference(
        outcome_df,
        source_tables=["mart_teacher_month_persona_ready"],
        build_summary="Outcome-rate drift between the first and last six modeling months.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, split first six versus last six modeling months, then compare churn and return rates.",
    )
    return feature_df, outcome_df


def build_categorical_drift(
    population: pd.DataFrame,
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    months = sorted(population["month"].dropna().unique().tolist())
    baseline_months = months[:RECENT_WINDOW_MONTHS]
    recent_months = months[-RECENT_WINDOW_MONTHS:]
    baseline = population[population["month"].isin(baseline_months)].copy()
    recent = population[population["month"].isin(recent_months)].copy()
    rows: List[Dict[str, Any]] = []
    for feature_name in categorical_features:
        base_series = normalize_text(baseline[feature_name])
        rec_series = normalize_text(recent[feature_name])
        categories = sorted(set(base_series.unique()).union(set(rec_series.unique())))
        base_share = base_series.value_counts(normalize=True, dropna=False)
        rec_share = rec_series.value_counts(normalize=True, dropna=False)
        total_variation = 0.5 * float(sum(abs(float(base_share.get(cat, 0.0)) - float(rec_share.get(cat, 0.0))) for cat in categories))
        max_diff_pp = max(abs((float(rec_share.get(cat, 0.0)) - float(base_share.get(cat, 0.0))) * 100) for cat in categories) if categories else float("nan")
        level = categorical_drift_level(total_variation, max_diff_pp)
        for cat in categories:
            rows.append(
                {
                    "feature_name": feature_name,
                    "category_value": cat,
                    "baseline_month_start": str(min(baseline_months)),
                    "baseline_month_end": str(max(baseline_months)),
                    "recent_month_start": str(min(recent_months)),
                    "recent_month_end": str(max(recent_months)),
                    "baseline_share": float(base_share.get(cat, 0.0)),
                    "recent_share": float(rec_share.get(cat, 0.0)),
                    "share_diff_pp": (float(rec_share.get(cat, 0.0)) - float(base_share.get(cat, 0.0))) * 100,
                    "feature_total_variation": total_variation,
                    "feature_max_share_diff_pp": max_diff_pp,
                    "drift_level": level,
                    "drift_relevance": "relevant" if level in {"high_drift", "medium_drift"} else "limited",
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_drift_rank"] = df["drift_level"].map(drift_rank)
        df["_share_abs"] = df["share_diff_pp"].abs()
        df = df.sort_values(["_drift_rank", "_share_abs"], ascending=[True, False]).drop(columns=["_drift_rank", "_share_abs"]).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher"],
        build_summary="Categorical drift between the first and last six modeling months, measured with share changes and total variation distance.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, split first six versus last six modeling months, then compare category shares and total variation for each selected dimension.",
    )


def window_subset(df: pd.DataFrame, window_label: str) -> pd.DataFrame:
    months = sorted(df["month"].dropna().unique().tolist())
    if window_label == "all_history":
        return df.copy()
    if window_label == "recent_6m":
        return df[df["month"].isin(months[-RECENT_WINDOW_MONTHS:])].copy()
    raise ValueError(f"Unknown window_label: {window_label}")


def temporal_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    months = sorted(df["month"].dropna().unique().tolist())
    if len(months) < 3:
        return pd.DataFrame(), pd.DataFrame(), [], []
    split_idx = max(1, min(len(months) - 1, int(np.floor(len(months) * 0.67))))
    train_months = months[:split_idx]
    test_months = months[split_idx:]
    train = df[df["month"].isin(train_months)].copy()
    test = df[df["month"].isin(test_months)].copy()
    return train, test, train_months, test_months


def build_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> ColumnTransformer:
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_features),
            )
        )
    return ColumnTransformer(transformers=transformers)


def fit_models(
    population: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    control_validity: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    valid_control_windows = supported_control_windows(control_validity)

    model_specs = [
        {
            "model_name": "dummy_baseline",
            "feature_spec": "baseline",
            "numeric_features": [],
            "categorical_features": [],
            "estimator": DummyClassifier(strategy="prior"),
        },
        {
            "model_name": "profile_only_logistic",
            "feature_spec": "profile_only",
            "numeric_features": [],
            "categorical_features": [PROFILE_CONTROL_VAR],
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        },
        {
            "model_name": "behavior_plus_profile_logistic",
            "feature_spec": "behavior_plus_profile",
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        },
        {
            "model_name": "behavior_plus_profile_random_forest",
            "feature_spec": "behavior_plus_profile",
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "estimator": RandomForestClassifier(
                n_estimators=250,
                max_features="sqrt",
                min_samples_leaf=40,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=42,
            ),
        },
    ]

    for target_col, target_label in [
        ("target_churn_m1", "abandonar_m1"),
        ("target_return_active_m1", "retornar_ativo_m1"),
    ]:
        for window_label in ["all_history", "recent_6m"]:
            window_df = window_subset(population, window_label)
            train, test, train_months, test_months = temporal_train_test_split(window_df)
            if train.empty or test.empty or train[target_col].nunique() < 2 or test[target_col].nunique() < 2:
                continue
            control_is_supported = window_label in valid_control_windows

            best_model_name = None
            best_score = float("-inf")
            best_pipeline: Pipeline | None = None
            best_x_test: pd.DataFrame | None = None
            best_y_test: pd.Series | None = None
            best_features: List[str] = []

            for spec in model_specs:
                y_train = train[target_col].astype(int)
                y_test = test[target_col].astype(int)
                spec_numeric = list(spec["numeric_features"])
                spec_categorical = list(spec["categorical_features"])
                if not control_is_supported:
                    spec_categorical = [feature for feature in spec_categorical if feature != PROFILE_CONTROL_VAR]
                if spec["model_name"] == "profile_only_logistic" and not spec_categorical:
                    continue
                if spec["model_name"] == "dummy_baseline":
                    estimator = spec["estimator"]
                    estimator.fit(np.zeros((len(train), 1)), y_train)
                    score = estimator.predict_proba(np.zeros((len(test), 1)))[:, 1]
                else:
                    x_cols = spec_numeric + spec_categorical
                    X_train = train[x_cols].copy()
                    X_test = test[x_cols].copy()
                    pipeline = Pipeline(
                        steps=[
                            ("preprocess", build_preprocessor(spec_numeric, spec_categorical)),
                            ("model", spec["estimator"]),
                        ]
                    )
                    pipeline.fit(X_train, y_train)
                    score = pipeline.predict_proba(X_test)[:, 1]
                    if spec["feature_spec"] == "behavior_plus_profile":
                        roc_auc = safe_auc(y_test, score)
                        if pd.notna(roc_auc) and roc_auc > best_score:
                            best_score = roc_auc
                            best_model_name = spec["model_name"]
                            best_pipeline = pipeline
                            best_x_test = X_test
                            best_y_test = y_test
                            best_features = x_cols

                comparison_rows.append(
                    {
                        "target": target_label,
                        "window_label": window_label,
                        "model_name": spec["model_name"],
                        "feature_spec": spec["feature_spec"],
                        "control_variable": PROFILE_CONTROL_VAR,
                        "control_variable_supported": int(control_is_supported),
                        "control_variable_applied": int(control_is_supported and PROFILE_CONTROL_VAR in spec_categorical),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "train_month_start": str(min(train_months)),
                        "train_month_end": str(max(train_months)),
                        "test_month_start": str(min(test_months)),
                        "test_month_end": str(max(test_months)),
                        "positive_rate_test": float(y_test.mean()),
                        "roc_auc": safe_auc(y_test, score),
                        "average_precision": safe_average_precision(y_test, score),
                        "brier_score": safe_brier(y_test, score),
                        "log_loss": safe_log_loss(y_test, score),
                        "top_decile_lift": top_decile_lift(y_test, score),
                    }
                )

            if best_pipeline is not None and best_x_test is not None and best_y_test is not None:
                perm = permutation_importance(
                    best_pipeline,
                    best_x_test[best_features],
                    best_y_test,
                    n_repeats=3,
                    random_state=42,
                    n_jobs=1,
                    scoring="roc_auc",
                )
                order = np.argsort(perm.importances_mean)[::-1]
                for idx in order[:15]:
                    importance_rows.append(
                        {
                            "target": target_label,
                            "window_label": window_label,
                            "model_name": best_model_name,
                            "feature_name": best_features[idx],
                            "permutation_importance_mean": float(perm.importances_mean[idx]),
                            "permutation_importance_std": float(perm.importances_std[idx]),
                        }
                    )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["target", "window_label", "roc_auc"], ascending=[True, True, False]
    ).reset_index(drop=True)
    comparison_df = attach_reference(
        comparison_df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher", "audit_persona_feature_readiness"],
        build_summary="Temporal model comparison across all history and recent 6 months using a dummy baseline, a profile-only model, and two behavior-plus-profile models.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, choose non-leaky monthly features, split train/test by time, train the listed models, and compute common classification metrics on the held-out months.",
    )
    importance_df = pd.DataFrame(importance_rows).sort_values(
        ["target", "window_label", "permutation_importance_mean"], ascending=[True, True, False]
    ).reset_index(drop=True)
    importance_df = attach_reference(
        importance_df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher", "audit_persona_feature_readiness"],
        build_summary="Permutation importance for the best behavior-plus-profile model in each target/window pair.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, fit the best behavior-plus-profile model on the temporal train split, then compute permutation importance on the held-out test months using the raw input columns.",
    )
    return comparison_df, importance_df


def chart_block(
    artifact_name: str,
    title: str,
    subtitle: str,
    body_html: str,
    lineage: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "artifact_name": artifact_name,
        "title": title,
        "subtitle": subtitle,
        "body_html": body_html,
        "lineage": lineage,
    }


def strip_reference_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["source_tables", "build_summary", "rebuild_from_raw"]
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore").copy()


def build_output_reference(cfg: Config, output_tables: Iterable[str]) -> pd.DataFrame:
    rows = []
    for table_name in output_tables:
        rows.append(
            {
                "table_name": table_name,
                "csv_path": str(cfg.output_dir / "csv" / f"{table_name}.csv"),
                "parquet_path": str(cfg.output_dir / "parquet" / f"{table_name}.parquet"),
                "duckdb_path": str(cfg.output_duckdb_path),
                "how_to_rebuild": "Run raw_para_base_modelada_v4.py first, then run etapa_11_explainable_drift_prediction_v2.py.",
            }
        )
    df = pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=list(PUBLIC_TABLE_MAP.keys()),
        build_summary="Reference index for every final table generated by the focused drift/prediction path.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, then run etapa_11_explainable_drift_prediction_v2.py; this table is generated at the end as the manifest of outputs.",
    )


def summary_payload(
    population_summary: pd.DataFrame,
    control_validity: pd.DataFrame,
    numeric_drift: pd.DataFrame,
    categorical_drift: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> Dict[str, Any]:
    best_models = (
        model_comparison.sort_values(["target", "window_label", "roc_auc"], ascending=[True, True, False])
        .groupby(["target", "window_label"], as_index=False)
        .head(1)
    )
    top_numeric = numeric_drift.copy()
    if not top_numeric.empty:
        top_numeric = top_numeric.assign(_drift_rank=top_numeric["drift_level"].map(drift_rank)).sort_values(
            ["_drift_rank", "psi"], ascending=[True, False]
        ).drop(columns="_drift_rank").head(8)
    top_categorical = categorical_drift.copy()
    if not top_categorical.empty:
        top_categorical = top_categorical.assign(_drift_rank=top_categorical["drift_level"].map(drift_rank), _share_abs=top_categorical["share_diff_pp"].abs()).sort_values(
            ["_drift_rank", "_share_abs"], ascending=[True, False]
        ).drop(columns=["_drift_rank", "_share_abs"]).head(10)
    top_importance = feature_importance.groupby(["target", "window_label"], as_index=False).head(8)
    return {
        "generated_at_utc": utc_now_iso(),
        "population_summary": population_summary.drop(columns=["source_tables", "build_summary", "rebuild_from_raw"]).to_dict(orient="records"),
        "best_models": best_models.drop(columns=["source_tables", "build_summary", "rebuild_from_raw"]).to_dict(orient="records"),
        "control_validity": control_validity.drop(columns=["source_tables", "build_summary", "rebuild_from_raw"]).to_dict(orient="records"),
        "top_numeric_drift": top_numeric.drop(columns=["source_tables", "build_summary", "rebuild_from_raw"]).to_dict(orient="records"),
        "top_categorical_drift": top_categorical.drop(columns=["source_tables", "build_summary", "rebuild_from_raw"]).to_dict(orient="records"),
        "top_feature_importance": top_importance.drop(columns=["source_tables", "build_summary", "rebuild_from_raw"]).to_dict(orient="records"),
    }

def build_html_report(
    cfg: Config,
    input_map: pd.DataFrame,
    assumptions: pd.DataFrame,
    population_summary: pd.DataFrame,
    control_validity: pd.DataFrame,
    feature_candidates: pd.DataFrame,
    feature_screening: pd.DataFrame,
    numeric_drift: pd.DataFrame,
    categorical_drift: pd.DataFrame,
    outcome_drift: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_reference: pd.DataFrame,
) -> str:
    pop = strip_reference_cols(population_summary).iloc[0]
    best_models = (
        strip_reference_cols(model_comparison)
        .sort_values(["target", "window_label", "roc_auc"], ascending=[True, True, False])
        .groupby(["target", "window_label"], as_index=False)
        .head(1)
    )
    strongest_numeric = strip_reference_cols(numeric_drift).copy()
    if not strongest_numeric.empty:
        strongest_numeric = strongest_numeric.assign(_drift_rank=strongest_numeric["drift_level"].map(drift_rank)).sort_values(
            ["_drift_rank", "psi"], ascending=[True, False]
        ).drop(columns="_drift_rank").head(1)
    strongest_cat = strip_reference_cols(categorical_drift).copy()
    if not strongest_cat.empty:
        strongest_cat = strongest_cat.assign(_drift_rank=strongest_cat["drift_level"].map(drift_rank), _share_abs=strongest_cat["share_diff_pp"].abs()).sort_values(
            ["_drift_rank", "_share_abs"], ascending=[True, False]
        ).drop(columns=["_drift_rank", "_share_abs"]).head(1)
    numeric_drift_display = strip_reference_cols(numeric_drift)
    if not numeric_drift_display.empty:
        numeric_drift_display = numeric_drift_display.assign(
            _drift_rank=numeric_drift_display["drift_level"].map(drift_rank)
        ).sort_values(["_drift_rank", "psi"], ascending=[True, False]).drop(columns="_drift_rank")
    categorical_drift_display = strip_reference_cols(categorical_drift)
    if not categorical_drift_display.empty:
        categorical_drift_display = categorical_drift_display.assign(
            _drift_rank=categorical_drift_display["drift_level"].map(drift_rank),
            _share_abs=categorical_drift_display["share_diff_pp"].abs(),
        ).sort_values(["_drift_rank", "_share_abs"], ascending=[True, False]).drop(columns=["_drift_rank", "_share_abs"])
    summary_cards = "".join(
        [
            build_card_html("Relevant Tables Reviewed", fmt_num(len(PUBLIC_TABLE_MAP), 0), "Declared source layer"),
            build_card_html("Modeling Rows", fmt_num(pop["rows"], 0), "Observed teacher-months with month t+1 visible"),
            build_card_html("Teachers In Scope", fmt_num(pop["teachers"], 0), "Unique teachers in the modeling population"),
            build_card_html("Best ROC AUC", fmt_num(best_models["roc_auc"].max(), 3), "Best temporal test score"),
            build_card_html(
                "Strongest Numeric Drift",
                strongest_numeric.iloc[0]["feature_name"] if not strongest_numeric.empty else "n/a",
                strongest_numeric.iloc[0]["drift_level"] if not strongest_numeric.empty else "n/a",
            ),
            build_card_html(
                "Strongest Categorical Drift",
                strongest_cat.iloc[0]["feature_name"] if not strongest_cat.empty else "n/a",
                strongest_cat.iloc[0]["category_value"] if not strongest_cat.empty else "n/a",
            ),
        ]
    )

    sections: List[Dict[str, Any]] = []

    input_block = chart_block(
        "focused_input_map",
        "What Was Reviewed",
        "This is the isolated relevant-table layer and its exact role in the focused path.",
        build_table_html(strip_reference_cols(input_map), max_rows=25),
        {
            "raw_tables": "dim_teachers, fct_teachers_entries, fct_teachers_contents_interactions, stg_lessons, stg_formation, calendario_escolar_uf_rede",
            "population": "Declared relevant-table layer exported by raw_para_base_modelada_v4.py",
            "grain": "1 row per declared relevant table",
            "joins": "No new joins; inventory over the exported relevant tables",
            "filters": "None",
            "logic": "Maps public table names to physical tables, grain, direct-use flags, and role in the focused analysis",
            "caveats": "Tables marked as context-only were reviewed but not used directly in the drift or prediction calculations",
        },
    )
    sections.append(
        {
            "title": "Scope And Inputs",
            "description": "Start from the declared relevant-table layer, then separate direct analytical use from upstream context.",
            "blocks": [input_block],
        }
    )

    drift_blocks: List[Dict[str, Any]] = []
    drift_blocks.append(
        chart_block(
            "drift_assumptions",
            "Step 1. Freeze The Drift Assumptions",
            "Plain-English assumptions that make the drift measurement explainable and reproducible.",
            build_table_html(strip_reference_cols(assumptions), max_rows=12),
            {
                "raw_tables": "dim_teachers, fct_teachers_entries, fct_teachers_contents_interactions; materialized upstream into mart_teacher_month_persona_ready",
                "population": "Method definition only",
                "grain": "1 row per assumption",
                "joins": "Not applicable",
                "filters": "Not applicable",
                "logic": "Documents the population, comparison window, metrics, and leakage rules before any result is computed",
                "caveats": "These assumptions drive interpretation; if they change, the reported drift also changes",
            },
        )
    )
    drift_blocks.append(
        chart_block(
            "drift_population",
            "Step 2. Define The Modeling Population",
            "The same population is used for drift and prediction so the story stays aligned end to end.",
            build_table_html(strip_reference_cols(population_summary), max_rows=5),
            {
                "raw_tables": "dim_teachers, fct_teachers_entries, fct_teachers_contents_interactions; via raw_para_base_modelada_v4.py",
                "population": "Observed teacher-month rows with persona_analysis_eligible_flag = 1 and next_month_observed_flag = 1",
                "grain": "1 row summary for the final modeling population",
                "joins": "Already materialized in mart_teacher_month_persona_ready",
                "filters": "observed_month_flag = 1, persona_analysis_eligible_flag = 1, next_month_observed_flag = 1",
                "logic": "Counts rows, teachers, time range, and target prevalence after eligibility filters",
                "caveats": "This is the inferable teacher-level population, not the full anonymous/shadow raw universe",
            },
        )
    )
    if not numeric_drift_display.empty:
        numeric_chart_df = numeric_drift_display.head(12).copy()
        fig = px.bar(
            numeric_chart_df,
            x="psi",
            y="feature_name",
            color="drift_level",
            orientation="h",
            color_discrete_map={
                "high_drift": PALETTE[5],
                "medium_drift": PALETTE[3],
                "low_drift": PALETTE[2],
                "insufficient_data": PALETTE[4],
            },
            title="Top numeric drift features by PSI",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        drift_blocks.append(
            chart_block(
                "drift_numeric",
                "Step 3. Measure Numeric Drift",
                "Old versus recent means first 6 modeling months versus last 6 modeling months. Drift is measured with PSI plus standardized mean difference.",
                figure_to_html(fig) + build_table_html(numeric_chart_df, max_rows=12),
                {
                    "raw_tables": "dim_teachers, fct_teachers_entries, fct_teachers_contents_interactions; via mart_teacher_month_persona_ready",
                    "population": "Modeling population",
                    "grain": "1 row per numeric feature",
                    "joins": "Already materialized in mart_teacher_month_persona_ready",
                    "filters": "first 6 vs last 6 modeling months",
                    "logic": "Compute mean/median change, standardized mean difference, PSI, then classify low/medium/high drift",
                    "caveats": "High drift means the recent feature distribution differs materially from the old one; it does not imply a product problem by itself",
                },
            )
        )
    if not categorical_drift_display.empty:
        categorical_chart_df = categorical_drift_display.head(15).copy()
        categorical_chart_df["feature_category"] = categorical_chart_df["feature_name"] + "::" + categorical_chart_df["category_value"]
        fig = px.bar(
            categorical_chart_df,
            x="share_diff_pp",
            y="feature_category",
            color="drift_level",
            orientation="h",
            color_discrete_map={
                "high_drift": PALETTE[5],
                "medium_drift": PALETTE[3],
                "low_drift": PALETTE[2],
                "insufficient_data": PALETTE[4],
            },
            title="Top categorical drift slices by share difference (pp)",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        drift_blocks.append(
            chart_block(
                "drift_categorical",
                "Step 4. Measure Categorical Drift",
                "Category drift is measured with share changes and total variation distance, then read dimension by dimension.",
                figure_to_html(fig) + build_table_html(categorical_chart_df.drop(columns=["feature_category"]), max_rows=15),
                {
                    "raw_tables": "dim_teachers; via mart_teacher_month_persona_ready",
                    "population": "Modeling population",
                    "grain": "1 row per dimension x category",
                    "joins": "Already materialized in mart_teacher_month_persona_ready",
                    "filters": "first 6 vs last 6 modeling months",
                    "logic": "Compare category shares in the old and recent windows, compute total variation, and classify drift relevance",
                    "caveats": "Categorical drift often mixes acquisition change, profile mix change, and telemetry change",
                },
            )
        )
    drift_blocks.append(
        chart_block(
            "drift_outcome",
            "Step 5. Check Outcome Drift",
            "This tells us whether the business problem itself changed, not only the predictors.",
            build_table_html(strip_reference_cols(outcome_drift), max_rows=10),
            {
                "raw_tables": "dim_teachers, fct_teachers_entries, fct_teachers_contents_interactions; via mart_teacher_month_persona_ready",
                "population": "Modeling population",
                "grain": "1 row per target metric",
                "joins": "Already materialized in mart_teacher_month_persona_ready",
                "filters": "first 6 vs last 6 modeling months",
                "logic": "Compare churn and return rates across the old and recent windows",
                "caveats": "Outcome drift can be driven by product, acquisition mix, seasonality, or instrumentation change",
            },
        )
    )
    sections.append(
        {
            "title": "A. Drift",
            "description": "Step by step: define the valid population, compare old versus recent periods, then decide whether the change is practically relevant.",
            "blocks": drift_blocks,
        }
    )

    prediction_blocks: List[Dict[str, Any]] = []
    prediction_blocks.append(
        chart_block(
            "prediction_control_validity",
            "Step 1. Validate The Control Variable",
            f"The chosen control variable is {PROFILE_CONTROL_VAR}. It must have coverage, group size, and target association before it is used for adjustment.",
            build_table_html(strip_reference_cols(control_validity), max_rows=10),
            {
                "raw_tables": "dim_teachers; used through mart_teacher_month_persona_ready",
                "population": "Modeling population",
                "grain": "1 row per target x evaluation window",
                "joins": "Already materialized in mart_teacher_month_persona_ready",
                "filters": "all_history and recent_6m windows",
                "logic": "Check missingness, minimum group size, chi-square p-value, and Cramer's V before using the profile variable as control",
                "caveats": "Statistical support justifies adjustment; it does not make the control variable causal",
            },
        )
    )
    prediction_blocks.append(
        chart_block(
            "prediction_feature_candidates",
            "Step 2. Build The Candidate Feature Set",
            "Each variable is labeled as predictor, control-only, context-only, telemetry-support, or leakage-excluded.",
            build_table_html(strip_reference_cols(feature_candidates), max_rows=30),
            {
                "raw_tables": "audit_persona_feature_readiness, dim_persona_range_candidates, mart_teacher_month_persona_ready",
                "population": "Teacher-month features available at month t",
                "grain": "1 row per feature",
                "joins": "feature_name joins between the readiness audit and range candidates",
                "filters": "teacher_month features only; future-derived targets excluded",
                "logic": "Start from the readiness audit, then explicitly exclude leakage and separate interpretation-only fields from predictors",
                "caveats": "Telemetry-support features can be predictive but should be interpreted with tracking caveats",
            },
        )
    )
    screening_display = strip_reference_cols(feature_screening)
    if not screening_display.empty:
        screening_chart_df = screening_display.sort_values(["target", "separation_auc"], ascending=[True, False]).groupby("target", as_index=False).head(10).copy()
        screening_chart_df["target_feature"] = screening_chart_df["target"] + " :: " + screening_chart_df["feature_name"]
        fig = px.bar(
            screening_chart_df,
            x="separation_auc",
            y="target_feature",
            color="target",
            orientation="h",
            color_discrete_sequence=[PALETTE[0], PALETTE[3]],
            title="Top univariate signals before multivariate modeling",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        prediction_blocks.append(
            chart_block(
                "prediction_feature_screening",
                "Step 3. Screen Variables One By One",
                "This is not the final model. It is the first pass to show which numeric variables already separate churn and return on their own.",
                figure_to_html(fig) + build_table_html(screening_chart_df.drop(columns=["target_feature"]), max_rows=20),
                {
                    "raw_tables": "mart_teacher_month_persona_ready, audit_persona_feature_readiness",
                    "population": "Modeling population",
                    "grain": "1 row per feature x target",
                    "joins": "Feature list applied directly to the modeling mart",
                    "filters": "numeric included features with enough non-missing rows",
                    "logic": "Compute univariate AUC-style separation and Mann-Whitney significance for each target",
                    "caveats": "A strong univariate variable can lose importance once correlated features enter the model together",
                },
            )
        )
    model_display = strip_reference_cols(model_comparison)
    if not model_display.empty:
        fig = px.bar(
            model_display,
            x="target",
            y="roc_auc",
            color="model_name",
            barmode="group",
            facet_col="window_label",
            color_discrete_sequence=PALETTE,
            title="Temporal model comparison across windows",
        )
        prediction_blocks.append(
            chart_block(
                "prediction_model_comparison",
                "Step 4. Compare Models With Temporal Splits",
                "Train on older months, test on newer months, and compare common metrics across baseline, profile-only, and behavior-plus-profile models.",
                figure_to_html(fig) + build_table_html(model_display, max_rows=20),
                {
                    "raw_tables": "mart_teacher_month_persona_ready, dim_teacher, audit_persona_feature_readiness",
                    "population": "Modeling population",
                    "grain": "1 row per target x window x model",
                    "joins": "Already materialized in mart_teacher_month_persona_ready, with dim_teacher profile columns embedded upstream",
                    "filters": "Temporal split only; no random split",
                    "logic": "Fit dummy, profile-only logistic, behavior+profile logistic, and behavior+profile random forest, then score held-out future months",
                    "caveats": "AUC is not enough on its own; calibration and precision matter too, especially with imbalanced targets",
                },
            )
        )
    importance_display = strip_reference_cols(feature_importance)
    if not importance_display.empty:
        importance_chart_df = importance_display.groupby(["target", "window_label"], as_index=False).head(8).copy()
        importance_chart_df["target_window"] = importance_chart_df["target"] + " | " + importance_chart_df["window_label"]
        fig = px.bar(
            importance_chart_df,
            x="permutation_importance_mean",
            y="feature_name",
            color="target_window",
            orientation="h",
            facet_col="target_window",
            color_discrete_sequence=PALETTE,
            title="Held-out feature importance for the best behavior-plus-profile model",
        )
        prediction_blocks.append(
            chart_block(
                "prediction_feature_importance",
                "Step 5. Explain The Best Model",
                "Permutation importance is computed only on held-out months, so it reflects real out-of-time signal rather than in-sample fit.",
                figure_to_html(fig) + build_table_html(importance_display, max_rows=25),
                {
                    "raw_tables": "mart_teacher_month_persona_ready, dim_teacher, audit_persona_feature_readiness",
                    "population": "Held-out months from the modeling population",
                    "grain": "1 row per feature importance result",
                    "joins": "Same inputs used by the best behavior-plus-profile model",
                    "filters": "Best model per target x window only",
                    "logic": "Use permutation importance on the held-out test split to rank features by how much they matter to prediction",
                    "caveats": "Importance is about predictive contribution, not causality or product mechanism",
                },
            )
        )
    sections.append(
        {
            "title": "B. Prediction",
            "description": "Step by step: validate the control, screen the variables, fit temporally valid models, then interpret the strongest predictors carefully.",
            "blocks": prediction_blocks,
        }
    )

    rebuild_body = (
        "<div class='note'>"
        "<p><b>Step 1:</b> run <code>/Users/akatsurada/Documents/INSPER/Design/Aula_2/raw_para_base_modelada_v4.py</code> to rebuild the relevant-table layer from raw files.</p>"
        "<p><b>Step 2:</b> run <code>/Users/akatsurada/Documents/INSPER/Design/Aula_2/etapa_11_explainable_drift_prediction_v2.py</code> to rebuild the focused drift and prediction outputs.</p>"
        f"<p><b>Step 3:</b> open the CSV, Parquet, DuckDB, markdown, and HTML artifacts under <code>{cfg.output_dir}</code>.</p>"
        "</div>"
        + build_table_html(strip_reference_cols(output_reference), max_rows=20)
    )
    sections.append(
        {
            "title": "Rebuild And Output Reference",
            "description": "Every final table includes source references, a quick build summary, and rebuild instructions. This last section is the manifest and rebuild guide.",
            "blocks": [
                chart_block(
                    "output_reference",
                    "Outputs And Rebuild Guide",
                    "Use this block as the index for the final focused path.",
                    rebuild_body,
                    {
                        "raw_tables": "dim_teachers, fct_teachers_entries, fct_teachers_contents_interactions; via raw_para_base_modelada_v4.py",
                        "population": "Final focused drift and prediction outputs",
                        "grain": "1 row per generated output table",
                        "joins": "Not applicable",
                        "filters": "Not applicable",
                        "logic": "Lists every final table with its output paths and rebuild instructions",
                        "caveats": "Rebuild starts from raw only through raw_para_base_modelada_v4.py; etapa_11 assumes the relevant-table layer already exists",
                    },
                )
            ],
        }
    )

    return render_report_html(
        title="Relatório - Drift e Predição Explainable v2",
        subtitle="Camada focada, isolada e reprodutível a partir das tabelas relevantes da base_modelada_v2. Explica drift e predição passo a passo em plain English.",
        summary_cards_html=summary_cards,
        sections=sections,
    )


def write_summary_markdown(path: Path, cfg: Config, summary: Dict[str, Any]) -> None:
    lines = [
        "# Explainable Drift and Prediction Review v2",
        "",
        "## Paths",
        "",
        f"- Source DuckDB: `{cfg.source_duckdb_path}`",
        f"- Focused output directory: `{cfg.output_dir}`",
        f"- HTML report: `{cfg.output_dir / 'reports' / 'relatorio_drift_prediction_explainable_v2.html'}`",
        "",
        "## Step By Step",
        "",
        "### A. Drift",
        "",
        "1. Start from the relevant-table layer produced by `raw_para_base_modelada_v4.py`.",
        "2. Freeze the modeling population as observed teacher-month rows with `persona_analysis_eligible_flag = 1` and `next_month_observed_flag = 1`.",
        f"3. Define `old` as the first {RECENT_WINDOW_MONTHS} modeling months and `recent` as the last {RECENT_WINDOW_MONTHS} modeling months.",
        "4. Measure numeric drift with mean change, median change, standardized mean difference, and PSI.",
        "5. Measure categorical drift with category share difference and total variation distance.",
        "6. Check outcome drift separately so we know whether the business problem itself changed over time.",
        "",
        "### B. Prediction",
        "",
        f"1. Validate `{PROFILE_CONTROL_VAR}` before using it as a control variable.",
        "2. Build the candidate feature set from `audit_persona_feature_readiness` and exclude leakage features.",
        "3. Screen numeric features one by one to understand early signal before modeling.",
        "4. Train models with temporal train/test splits only, never random splits.",
        "5. Compare a dummy baseline, a profile-only logistic model, a behavior-plus-profile logistic model, and a behavior-plus-profile random forest.",
        "6. Use held-out permutation importance to explain the best multivariate model.",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Modeling rows: {summary['population_summary'][0]['rows'] if summary['population_summary'] else 'n/a'}",
        f"- Teachers in scope: {summary['population_summary'][0]['teachers'] if summary['population_summary'] else 'n/a'}",
        "",
        "## Best Models",
    ]
    for row in summary["best_models"]:
        lines.append(
            f"- `{row['target']}` | `{row['window_label']}` | `{row['model_name']}` | auc={row['roc_auc']:.4f} | ap={row['average_precision']:.4f} | brier={row['brier_score']:.4f}"
        )
    lines.append("")
    lines.append("## Strongest Numeric Drift")
    for row in summary["top_numeric_drift"][:6]:
        lines.append(
            f"- `{row['feature_name']}` | `{row['drift_level']}` | mean_old={row['baseline_mean']:.4f} | mean_recent={row['recent_mean']:.4f} | psi={row['psi']:.4f}"
        )
    lines.append("")
    lines.append("## Strongest Categorical Drift")
    for row in summary["top_categorical_drift"][:8]:
        lines.append(
            f"- `{row['feature_name']}::{row['category_value']}` | `{row['drift_level']}` | diff_pp={row['share_diff_pp']:.2f} | tv={row['feature_total_variation']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Rebuild From Raw",
            "",
            "1. Run `raw_para_base_modelada_v4.py` to rebuild the relevant-table layer.",
            "2. Run `etapa_11_explainable_drift_prediction_v2.py` to rebuild the focused outputs.",
            "3. Read the output manifest table `analytics_drift_prediction_output_reference_v2` for exact table paths.",
        ]
    )
    write_markdown(path, lines)


def main() -> None:
    cfg = build_config(parse_args())
    source_conn = connect_source(cfg)
    output_conn = connect_output(cfg)
    try:
        tables = load_public_tables(source_conn)
        input_map = build_input_map(tables)
        population, population_summary = prepare_model_population(tables)
        assumptions = build_assumptions_table()
        control_validity = build_control_variable_validity(population)
        feature_candidates, numeric_features, categorical_features, context_drift_features = build_feature_candidates(tables, population)
        feature_screening = build_feature_screening(population, feature_candidates)
        numeric_drift, outcome_drift = build_numeric_drift(population, numeric_features)
        categorical_drift = build_categorical_drift(population, context_drift_features)
        model_comparison, feature_importance = fit_models(population, numeric_features, categorical_features, control_validity)

        outputs: Dict[str, pd.DataFrame] = {
            "analytics_drift_prediction_input_map_v2": input_map,
            "analytics_drift_prediction_assumptions_v2": assumptions,
            "analytics_drift_prediction_population_summary_v2": population_summary,
            "analytics_prediction_control_variable_validity_v2": control_validity,
            "analytics_prediction_feature_candidates_v2": feature_candidates,
            "analytics_prediction_feature_screening_v2": feature_screening,
            "analytics_drift_numeric_explainable_v2": numeric_drift,
            "analytics_drift_outcome_explainable_v2": outcome_drift,
            "analytics_drift_categorical_explainable_v2": categorical_drift,
            "analytics_prediction_model_comparison_explainable_v2": model_comparison,
            "analytics_prediction_feature_importance_explainable_v2": feature_importance,
        }
        output_reference = build_output_reference(cfg, outputs.keys())
        outputs["analytics_drift_prediction_output_reference_v2"] = output_reference

        for table_name, df in outputs.items():
            persist_table(output_conn, cfg, table_name, df)

        summary = summary_payload(
            population_summary=population_summary,
            control_validity=control_validity,
            numeric_drift=numeric_drift,
            categorical_drift=categorical_drift,
            model_comparison=model_comparison,
            feature_importance=feature_importance,
        )
        write_json(cfg.output_dir / "json" / "explainable_drift_prediction_summary_v2.json", summary)
        write_summary_markdown(cfg.output_dir / "audit" / "explainable_drift_prediction_summary_v2.md", cfg, summary)

        html = build_html_report(
            cfg=cfg,
            input_map=input_map,
            assumptions=assumptions,
            population_summary=population_summary,
            control_validity=control_validity,
            feature_candidates=feature_candidates,
            feature_screening=feature_screening,
            numeric_drift=numeric_drift,
            categorical_drift=categorical_drift,
            outcome_drift=outcome_drift,
            model_comparison=model_comparison,
            feature_importance=feature_importance,
            output_reference=output_reference,
        )
        html_path = cfg.output_dir / "reports" / "relatorio_drift_prediction_explainable_v2.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
    finally:
        source_conn.close()
        output_conn.close()


if __name__ == "__main__":
    main()
