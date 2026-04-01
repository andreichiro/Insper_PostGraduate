"""Configuração e definição do estudo do pipeline modelled -> ml.

Este módulo concentra:
- constantes globais controladas por spec
- runtime config resolvida
- registries e políticas oficiais
- definição fixa B e utilitários de regra
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from targeted_ml.config.models import AnalysisSpec

def load_runtime_overrides() -> dict[str, Any]:
    raw = os.environ.get("TARGETED_ML_RUNTIME_OVERRIDES_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class RuntimeBuildConfig:
    analysis_kind: str
    enabled_tracks: list[str]
    label_window_days: int
    post_label_block_days: int
    post_label_block_count: int
    external_validators: list[str]
    definition_a_enabled: bool
    definition_a_strategy: str
    definition_a_candidate_metrics: list[str]
    definition_a_sql_file: str
    definition_a_python_strategy: str
    definition_b: dict[str, Any]
    definition_b_sql_file: str
    definition_b_python_strategy: str
    max_outer_test_months: int
    min_official_valid_outer_folds: int
    min_official_test_rows: int
    min_official_test_positives: int
    min_official_test_negatives: int
    tuning_enabled: bool
    tuning_n_iter: int
    tuning_max_inner_splits: int
    tuning_scoring: str
    model_family_scope: list[str]
    model_comparison_workers: int
    calibration_method: str
    feature_importance_permutation_repeats: int
    cluster_k_candidates: list[int]
    cluster_bootstrap_iterations: int
    cluster_sample_size: int
    registered_band_policies: list[dict[str, Any]]
    heavy_user_percentile_policies: list[dict[str, Any]]

    @classmethod
    def from_analysis_spec(cls, spec: AnalysisSpec) -> "RuntimeBuildConfig":
        return cls.from_payload(spec.runtime_overrides())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RuntimeBuildConfig":
        return cls(
            analysis_kind=str(payload.get("analysis_kind", "activity")),
            enabled_tracks=[str(value) for value in payload.get("enabled_tracks", ["S1", "S7", "S1_PLUS_S7", "STRICT_CONTEXT"])],
            label_window_days=int(payload.get("label_window_days", 30)),
            post_label_block_days=int(payload.get("post_label_block_days", 30)),
            post_label_block_count=int(payload.get("post_label_block_count", 3)),
            external_validators=[str(value) for value in payload.get("external_validators", [])],
            definition_a_enabled=bool(payload.get("definition_a_enabled", True)),
            definition_a_strategy=str(payload.get("definition_a_strategy", "univariate_exact")),
            definition_a_candidate_metrics=[str(value) for value in payload.get("definition_a_candidate_metrics", [])],
            definition_a_sql_file=str(payload.get("definition_a_sql_file", "") or ""),
            definition_a_python_strategy=str(payload.get("definition_a_python_strategy", "") or ""),
            definition_b=dict(payload.get("definition_b", {})),
            definition_b_sql_file=str(payload.get("definition_b_sql_file", "") or ""),
            definition_b_python_strategy=str(payload.get("definition_b_python_strategy", "") or ""),
            max_outer_test_months=int(payload.get("max_outer_test_months", 5)),
            min_official_valid_outer_folds=int(payload.get("min_official_valid_outer_folds", 2)),
            min_official_test_rows=int(payload.get("min_official_test_rows", 50)),
            min_official_test_positives=int(payload.get("min_official_test_positives", 5)),
            min_official_test_negatives=int(payload.get("min_official_test_negatives", 20)),
            tuning_enabled=bool(payload.get("tuning_enabled", True)),
            tuning_n_iter=int(payload.get("tuning_n_iter", 4)),
            tuning_max_inner_splits=int(payload.get("tuning_max_inner_splits", 3)),
            tuning_scoring=str(payload.get("tuning_scoring", "neg_brier_score")),
            model_family_scope=[str(value) for value in payload.get("model_family_scope", ["logistic_regression", "random_forest", "catboost"])],
            model_comparison_workers=int(payload.get("model_comparison_workers", 3)),
            calibration_method=str(payload.get("calibration_method", "sigmoid")),
            feature_importance_permutation_repeats=int(payload.get("feature_importance_permutation_repeats", 5)),
            cluster_k_candidates=[int(value) for value in payload.get("cluster_k_candidates", [2, 3, 4, 5, 6])],
            cluster_bootstrap_iterations=int(payload.get("cluster_bootstrap_iterations", 20)),
            cluster_sample_size=int(payload.get("cluster_sample_size", 10000)),
            registered_band_policies=[dict(row) for row in payload.get("registered_band_policies", [])],
            heavy_user_percentile_policies=[dict(row) for row in payload.get("heavy_user_percentile_policies", [])],
        )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELLED_DUCKDB = PROJECT_ROOT / "data" / "modelled" / "duckdb" / "base_modelada_v2.duckdb"
MODELLED_TABLES = [
    "base_modelada_v2",
    "dim_teacher",
    "dim_event",
    "dim_device",
    "dim_calendar",
    "fct_teacher_month",
    "fct_session_clean",
    "fct_interaction_clean",
    "fct_formation_clean",
    "fct_mari_conversation_resolved",
    "fct_mari_help_resolved",
    "fct_mari_reports_resolved",
    "dim_lesson",
    "mart_teacher_cluster_ready",
    "mart_teacher_month_cluster_ready",
    "mart_teacher_persona_ready",
    "mart_teacher_month_persona_ready",
    "dim_persona_range_candidates",
    "mart_teacher_month_panel",
]
OFFICIAL_DEFINITION_B = "definition_b"
ENABLED_TRACKS = ["S1", "S7", "S1_PLUS_S7", "STRICT_CONTEXT"]
EXTERNAL_VALIDATORS = [
    "returned_active_post_label_m1",
    "returned_active_post_label_m2",
    "returned_active_post_label_m3",
    "active_days_post_label_3m",
    "sustained_active_2of3_post_label",
]
DEFINITION_A_STRATEGY = "univariate_exact"
OFFICIAL_METRIC_OBJECTIVES = {
    "valid_folds": "max",
    "mean_ap": "max",
    "mean_roc_auc": "max",
    "mean_brier": "min",
    "mean_log_loss": "min",
    "mean_calibration_slope_error": "min",
    "mean_calibration_intercept_abs": "min",
}
TRAIN_DEFINITION_OBJECTIVES = {
    "gap_returned_active_post_label_m1": "max",
    "gap_returned_active_post_label_m2": "max",
    "gap_returned_active_post_label_m3": "max",
    "gap_active_days_post_label_3m": "max",
    "gap_sustained_active_2of3_post_label": "max",
    "prevalence_entropy": "max",
    "monthly_prevalence_std": "min",
    "bootstrap_prevalence_ci_width": "min",
}
TEST_DEFINITION_OBJECTIVES = {
    "folds": "max",
    "test_gap_returned_active_post_label_m1": "max",
    "test_gap_returned_active_post_label_m2": "max",
    "test_gap_returned_active_post_label_m3": "max",
    "test_gap_active_days_post_label_3m": "max",
    "test_gap_sustained_active_2of3_post_label": "max",
    "test_prevalence_entropy": "max",
    "test_monthly_prevalence_std": "min",
    "test_bootstrap_prevalence_ci_width": "min",
}
BOOTSTRAP_ITERATIONS = 200
LABEL_WINDOW_DAYS = 30
POST_LABEL_BLOCK_DAYS = 30
POST_LABEL_BLOCK_COUNT = 3
# Outer fold mensal: treina em meses acumulados e testa no mes seguinte.
# O limite de 5 meses continua sendo uma convencao de execucao/custo e aparece
# no registro de arbitrariedade; nao e uma "descoberta" dos dados.
MAX_OUTER_TEST_MONTHS = 5
# Para publicar media e dispersao entre folds, exigimos pelo menos 2 folds
# validos. Resultado com 1 fold pode aparecer como diagnostico, mas nao como
# resumo oficial.
MIN_OFFICIAL_VALID_OUTER_FOLDS = 2
# Para um fold contar como "oficial", o teste precisa ter suporte suficiente.
# Isso evita que folds minúsculos entrem no resumo principal como se tivessem o
# mesmo peso de um mês inteiro.
MIN_OFFICIAL_TEST_ROWS = 50
MIN_OFFICIAL_TEST_POSITIVES = 5
MIN_OFFICIAL_TEST_NEGATIVES = 20
TUNING_ENABLED = True
TUNING_N_ITER = 8
TUNING_MAX_INNER_SPLITS = 3
TUNING_SCORING = "neg_brier_score"
FEATURE_IMPORTANCE_PERMUTATION_REPEATS = 5
PUBLISHED_PVALUE_PERMUTATIONS = 1000
CLUSTER_K_CANDIDATES = [2, 3, 4, 5, 6]
CLUSTER_BOOTSTRAP_ITERATIONS = 20
CLUSTER_SAMPLE_SIZE = 10000
MODEL_COMPARISON_WORKERS = max(1, min(3, os.cpu_count() or 1))
CALIBRATION_METHOD = "sigmoid"
HEAVY_USER_PERCENTILE_POLICIES = [
    {
        "policy_name": "heavy_top_10_percent",
        "policy_type": "percentile_cutoff",
        "parameter_json": '{"top_share": 0.10}',
        "description": "Marca como heavy user os 10% maiores valores do heavy_intensity_score.",
    },
    {
        "policy_name": "heavy_top_20_percent",
        "policy_type": "percentile_cutoff",
        "parameter_json": '{"top_share": 0.20}',
        "description": "Marca como heavy user os 20% maiores valores do heavy_intensity_score.",
    },
]
REGISTERED_BAND_POLICIES = [
    {
        "policy_name": "top_10_percent",
        "policy_type": "quantile_risk_cutoff",
        "parameter_json": '{"top_share": 0.1}',
        "description": "Marca como alto risco os 10% maiores valores de risk_score no conjunto de predições externas concatenadas.",
    },
    {
        "policy_name": "tercis",
        "policy_type": "tertile_risk_band",
        "parameter_json": '{"q_high": 0.6666666666666666, "q_low": 0.3333333333333333}',
        "description": "Separa risk_score em baixo, medio e alto pelos tercis empíricos do conjunto de predições externas concatenadas.",
    },
    {
        "policy_name": "score_ge_0_70",
        "policy_type": "fixed_risk_cutoff",
        "parameter_json": '{"risk_threshold": 0.7}',
        "description": "Marca como alto risco os casos com risk_score maior ou igual a 0.70.",
    },
]

# Referência observacional do último build completo bem-sucedido de `official_build`.
# Serve só para ETA aproximada do progress bar; não altera lógica analítica.
BUILD_PROGRESS_REFERENCE_MINUTES = 1135.0
BUILD_PROGRESS_STAGE_SPECS = [
    {"key": "registries", "label": "registries", "weight": 1.0},
    {"key": "onboarding_mart", "label": "onboarding mart", "weight": 1.0},
    {"key": "first_session_journey", "label": "first session journey", "weight": 1.5},
    {"key": "future_metrics", "label": "future metrics", "weight": 2.0},
    {"key": "definition_search", "label": "definition search", "weight": 4.0},
    {"key": "definition_comparison", "label": "definition comparison", "weight": 2.0},
    {"key": "leakage_audit", "label": "leakage audit", "weight": 1.0},
    {"key": "model_evaluation", "label": "model evaluation", "weight": 34.0},
    {"key": "cv_score_robustness", "label": "cv score robustness", "weight": 1.0},
    {"key": "cv_metric_robustness", "label": "cv metric robustness", "weight": 1.0},
    {"key": "prediction_bootstrap", "label": "prediction bootstrap", "weight": 1.5},
    {"key": "definition_b_feature_block_gain", "label": "definition B feature blocks", "weight": 41.0},
    {"key": "definition_b_excessive_separation", "label": "definition B excessive separation", "weight": 1.0},
    {"key": "reference_scope", "label": "reference scope", "weight": 0.5},
    {"key": "post_model_refit", "label": "post-model refit", "weight": 6.0},
    {"key": "threshold_outputs", "label": "threshold outputs", "weight": 1.5},
    {"key": "cv_threshold_robustness", "label": "cv threshold robustness", "weight": 1.0},
    {"key": "cluster_outputs", "label": "cluster outputs", "weight": 1.0},
    {"key": "heavy_user_outputs", "label": "heavy-user outputs", "weight": 1.0},
    {"key": "navigation_outputs", "label": "navigation outputs", "weight": 0.5},
    {"key": "summary_write", "label": "summary write", "weight": 0.5},
]

RUNTIME_OVERRIDES = load_runtime_overrides()
DEFAULT_DEFINITION_B_SPEC = {
    "definition_name": OFFICIAL_DEFINITION_B,
    "metric_name": "future_business_active_weeks",
    "operator": ">=",
    "threshold": 1.0,
    "rule_text": "future_business_active_weeks >= 1",
}

RUNTIME_CONFIG = RuntimeBuildConfig.from_payload(
    {
        "analysis_kind": RUNTIME_OVERRIDES.get("analysis_kind", "activity"),
        "enabled_tracks": RUNTIME_OVERRIDES.get("enabled_tracks", ENABLED_TRACKS),
        "label_window_days": RUNTIME_OVERRIDES.get("label_window_days", LABEL_WINDOW_DAYS),
        "post_label_block_days": RUNTIME_OVERRIDES.get("post_label_block_days", POST_LABEL_BLOCK_DAYS),
        "post_label_block_count": RUNTIME_OVERRIDES.get("post_label_block_count", POST_LABEL_BLOCK_COUNT),
        "external_validators": RUNTIME_OVERRIDES.get("external_validators", EXTERNAL_VALIDATORS),
        "definition_a_enabled": RUNTIME_OVERRIDES.get("definition_a_enabled", True),
        "definition_a_strategy": RUNTIME_OVERRIDES.get("definition_a_strategy", DEFINITION_A_STRATEGY),
        "definition_a_candidate_metrics": RUNTIME_OVERRIDES.get("definition_a_candidate_metrics", []),
        "definition_a_sql_file": RUNTIME_OVERRIDES.get("definition_a_sql_file", ""),
        "definition_a_python_strategy": RUNTIME_OVERRIDES.get("definition_a_python_strategy", ""),
        "definition_b": RUNTIME_OVERRIDES.get("definition_b", DEFAULT_DEFINITION_B_SPEC),
        "definition_b_sql_file": RUNTIME_OVERRIDES.get("definition_b_sql_file", ""),
        "definition_b_python_strategy": RUNTIME_OVERRIDES.get("definition_b_python_strategy", ""),
        "max_outer_test_months": RUNTIME_OVERRIDES.get("max_outer_test_months", MAX_OUTER_TEST_MONTHS),
        "min_official_valid_outer_folds": RUNTIME_OVERRIDES.get("min_official_valid_outer_folds", MIN_OFFICIAL_VALID_OUTER_FOLDS),
        "min_official_test_rows": RUNTIME_OVERRIDES.get("min_official_test_rows", MIN_OFFICIAL_TEST_ROWS),
        "min_official_test_positives": RUNTIME_OVERRIDES.get("min_official_test_positives", MIN_OFFICIAL_TEST_POSITIVES),
        "min_official_test_negatives": RUNTIME_OVERRIDES.get("min_official_test_negatives", MIN_OFFICIAL_TEST_NEGATIVES),
        "tuning_enabled": RUNTIME_OVERRIDES.get("tuning_enabled", TUNING_ENABLED),
        "tuning_n_iter": RUNTIME_OVERRIDES.get("tuning_n_iter", TUNING_N_ITER),
        "tuning_max_inner_splits": RUNTIME_OVERRIDES.get("tuning_max_inner_splits", TUNING_MAX_INNER_SPLITS),
        "tuning_scoring": RUNTIME_OVERRIDES.get("tuning_scoring", TUNING_SCORING),
        "model_family_scope": RUNTIME_OVERRIDES.get(
            "model_family_scope",
            ["logistic_regression", "random_forest", "catboost"],
        ),
        "model_comparison_workers": RUNTIME_OVERRIDES.get("model_comparison_workers", MODEL_COMPARISON_WORKERS),
        "calibration_method": RUNTIME_OVERRIDES.get("calibration_method", CALIBRATION_METHOD),
        "feature_importance_permutation_repeats": RUNTIME_OVERRIDES.get(
            "feature_importance_permutation_repeats",
            FEATURE_IMPORTANCE_PERMUTATION_REPEATS,
        ),
        "cluster_k_candidates": RUNTIME_OVERRIDES.get("cluster_k_candidates", CLUSTER_K_CANDIDATES),
        "cluster_bootstrap_iterations": RUNTIME_OVERRIDES.get("cluster_bootstrap_iterations", CLUSTER_BOOTSTRAP_ITERATIONS),
        "cluster_sample_size": RUNTIME_OVERRIDES.get("cluster_sample_size", CLUSTER_SAMPLE_SIZE),
        "registered_band_policies": RUNTIME_OVERRIDES.get("registered_band_policies", REGISTERED_BAND_POLICIES),
        "heavy_user_percentile_policies": RUNTIME_OVERRIDES.get(
            "heavy_user_percentile_policies",
            HEAVY_USER_PERCENTILE_POLICIES,
        ),
    }
)



def apply_runtime_config(runtime_config: RuntimeBuildConfig) -> None:
    global RUNTIME_CONFIG
    global OFFICIAL_DEFINITION_B
    global ENABLED_TRACKS
    global EXTERNAL_VALIDATORS
    global DEFINITION_A_STRATEGY
    global LABEL_WINDOW_DAYS
    global POST_LABEL_BLOCK_DAYS
    global POST_LABEL_BLOCK_COUNT
    global MAX_OUTER_TEST_MONTHS
    global MIN_OFFICIAL_VALID_OUTER_FOLDS
    global MIN_OFFICIAL_TEST_ROWS
    global MIN_OFFICIAL_TEST_POSITIVES
    global MIN_OFFICIAL_TEST_NEGATIVES
    global TUNING_ENABLED
    global TUNING_N_ITER
    global TUNING_MAX_INNER_SPLITS
    global TUNING_SCORING
    global FEATURE_IMPORTANCE_PERMUTATION_REPEATS
    global CLUSTER_K_CANDIDATES
    global CLUSTER_BOOTSTRAP_ITERATIONS
    global CLUSTER_SAMPLE_SIZE
    global MODEL_COMPARISON_WORKERS
    global HEAVY_USER_PERCENTILE_POLICIES
    global REGISTERED_BAND_POLICIES
    global CALIBRATION_METHOD

    RUNTIME_CONFIG = runtime_config
    OFFICIAL_DEFINITION_B = str(runtime_config.definition_b.get("definition_name", OFFICIAL_DEFINITION_B))
    ENABLED_TRACKS = list(runtime_config.enabled_tracks)
    EXTERNAL_VALIDATORS = list(runtime_config.external_validators)
    DEFINITION_A_STRATEGY = str(runtime_config.definition_a_strategy)
    LABEL_WINDOW_DAYS = int(runtime_config.label_window_days)
    POST_LABEL_BLOCK_DAYS = int(runtime_config.post_label_block_days)
    POST_LABEL_BLOCK_COUNT = int(runtime_config.post_label_block_count)
    MAX_OUTER_TEST_MONTHS = int(runtime_config.max_outer_test_months)
    MIN_OFFICIAL_VALID_OUTER_FOLDS = int(runtime_config.min_official_valid_outer_folds)
    MIN_OFFICIAL_TEST_ROWS = int(runtime_config.min_official_test_rows)
    MIN_OFFICIAL_TEST_POSITIVES = int(runtime_config.min_official_test_positives)
    MIN_OFFICIAL_TEST_NEGATIVES = int(runtime_config.min_official_test_negatives)
    TUNING_ENABLED = bool(runtime_config.tuning_enabled)
    TUNING_N_ITER = int(runtime_config.tuning_n_iter)
    TUNING_MAX_INNER_SPLITS = int(runtime_config.tuning_max_inner_splits)
    TUNING_SCORING = str(runtime_config.tuning_scoring)
    FEATURE_IMPORTANCE_PERMUTATION_REPEATS = int(runtime_config.feature_importance_permutation_repeats)
    CLUSTER_K_CANDIDATES = list(runtime_config.cluster_k_candidates)
    CLUSTER_BOOTSTRAP_ITERATIONS = int(runtime_config.cluster_bootstrap_iterations)
    CLUSTER_SAMPLE_SIZE = int(runtime_config.cluster_sample_size)
    MODEL_COMPARISON_WORKERS = int(runtime_config.model_comparison_workers)
    HEAVY_USER_PERCENTILE_POLICIES = list(runtime_config.heavy_user_percentile_policies)
    REGISTERED_BAND_POLICIES = list(runtime_config.registered_band_policies)
    CALIBRATION_METHOD = str(runtime_config.calibration_method)

apply_runtime_config(RUNTIME_CONFIG)

def get_definition_b_spec() -> dict[str, Any]:
    spec = DEFAULT_DEFINITION_B_SPEC.copy()
    spec.update(RUNTIME_CONFIG.definition_b)
    spec["definition_name"] = str(spec.get("definition_name", OFFICIAL_DEFINITION_B))
    spec["metric_name"] = str(spec.get("metric_name", "future_business_active_weeks"))
    spec["operator"] = str(spec.get("operator", ">="))
    spec["threshold"] = float(spec.get("threshold", 1.0))
    spec["rule_text"] = str(spec.get("rule_text", f'{spec["metric_name"]} {spec["operator"]} {spec["threshold"]:g}'))
    return spec

def apply_operator(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if operator == ">=":
        return values >= threshold
    if operator == ">":
        return values > threshold
    if operator == "<=":
        return values <= threshold
    if operator == "<":
        return values < threshold
    if operator == "==":
        return values == threshold
    raise ValueError(f"Unsupported operator: {operator}")

def make_atomic_rule(metric_name: str, threshold: float, operator: str = ">=") -> dict[str, Any]:
    return {
        "kind": "atomic",
        "metric_name": str(metric_name),
        "operator": str(operator),
        "threshold": float(threshold),
    }

def canonicalize_rule(rule: dict[str, Any]) -> dict[str, Any]:
    kind = str(rule.get("kind", "atomic"))
    if kind == "atomic":
        return make_atomic_rule(
            metric_name=str(rule["metric_name"]),
            threshold=float(rule["threshold"]),
            operator=str(rule.get("operator", ">=")),
        )
    combiner = str(rule.get("combiner", "AND")).upper()
    normalized_children = [canonicalize_rule(dict(child)) for child in rule.get("rules", [])]
    flattened_children: list[dict[str, Any]] = []
    for child in normalized_children:
        if child.get("kind") == "compound" and child.get("combiner") == combiner:
            flattened_children.extend(child.get("rules", []))
        else:
            flattened_children.append(child)
    flattened_children = sorted(flattened_children, key=stable_json)
    return {"kind": "compound", "combiner": combiner, "rules": flattened_children}

def extract_rule_metric_names(rule: dict[str, Any]) -> list[str]:
    normalized = canonicalize_rule(rule)
    if normalized["kind"] == "atomic":
        return [str(normalized["metric_name"])]
    metric_names: list[str] = []
    for child in normalized["rules"]:
        metric_names.extend(extract_rule_metric_names(child))
    return sorted(set(metric_names))

def rule_size(rule: dict[str, Any]) -> int:
    normalized = canonicalize_rule(rule)
    if normalized["kind"] == "atomic":
        return 1
    return int(sum(rule_size(child) for child in normalized["rules"]))

def rule_operator_label(rule: dict[str, Any]) -> str:
    normalized = canonicalize_rule(rule)
    if normalized["kind"] == "atomic":
        return str(normalized.get("operator", ">="))
    return str(normalized.get("combiner", "AND"))

def rule_metric_signature(rule: dict[str, Any]) -> str:
    return " + ".join(extract_rule_metric_names(rule))

def build_rule_text(rule: dict[str, Any]) -> str:
    normalized = canonicalize_rule(rule)
    if normalized["kind"] == "atomic":
        return f'{normalized["metric_name"]} {normalized["operator"]} {float(normalized["threshold"]):g}'
    joiner = f' {normalized["combiner"]} '
    child_text = [build_rule_text(child) for child in normalized["rules"]]
    return "(" + joiner.join(child_text) + ")"

def build_definition_a_label_name(rule: dict[str, Any]) -> str:
    return f"definition_a::{build_rule_text(rule)}"

def apply_rule_to_frame(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    normalized = canonicalize_rule(rule)
    if normalized["kind"] == "atomic":
        metric_name = str(normalized["metric_name"])
        if metric_name not in frame.columns:
            raise KeyError(f"Metric {metric_name} not found in frame")
        return apply_operator(frame[metric_name], normalized["operator"], float(normalized["threshold"])).astype(int)
    child_labels = [apply_rule_to_frame(frame, child).astype(bool).to_numpy() for child in normalized["rules"]]
    if not child_labels:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
    if normalized["combiner"] == "AND":
        combined = np.logical_and.reduce(child_labels)
    elif normalized["combiner"] == "OR":
        combined = np.logical_or.reduce(child_labels)
    else:
        raise ValueError(f"Unsupported combiner: {normalized['combiner']}")
    return pd.Series(combined.astype(int), index=frame.index)

def normalize_text(value: Any, default: str = "missing") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<missing>"}:
        return default
    return text

def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)

def build_track_registry() -> pd.DataFrame:
    rows = [
        {
            "track_name": "S1",
            "score_window_end_day": 0,
            "score_moment_text": "Fim da primeira sessao observada.",
            "allowed_feature_classes_json": stable_json(["context", "s1"]),
            "official_flag": 1,
        },
        {
            "track_name": "S7",
            "score_window_end_day": 7,
            "score_moment_text": "Fim dos primeiros 7 dias corridos a partir da ancora de onboarding.",
            "allowed_feature_classes_json": stable_json(["context", "s7"]),
            "official_flag": 1,
        },
        {
            "track_name": "S1_PLUS_S7",
            "score_window_end_day": 7,
            "score_moment_text": "Fim dos primeiros 7 dias, com bloco da primeira sessao e bloco acumulado da semana.",
            "allowed_feature_classes_json": stable_json(["context", "s1", "s7"]),
            "official_flag": 1,
        },
        {
            "track_name": "STRICT_CONTEXT",
            "score_window_end_day": 7,
            "score_moment_text": "Fim dos primeiros 7 dias, mas com entrada restrita a contexto inicial e sem comportamento inicial de produto.",
            "allowed_feature_classes_json": stable_json(["context"]),
            "official_flag": 1,
        },
    ]
    track_df = pd.DataFrame(rows)
    if ENABLED_TRACKS:
        track_df = track_df[track_df["track_name"].isin(ENABLED_TRACKS)].copy()
    return track_df.reset_index(drop=True)

def build_arbitrariness_registry() -> pd.DataFrame:
    rows = [
        {
            "choice_name": "label_window_days",
            "choice_value": str(LABEL_WINDOW_DAYS),
            "choice_type": "arbitrary_required",
            "where_used": "Definition A, Definition B, external validators",
            "status": "kept",
            "why": "A binary future label needs a finite horizon. The official build uses a fixed 30-day label window and surfaces it explicitly as arbitrary.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "post_label_block_days",
            "choice_value": str(POST_LABEL_BLOCK_DAYS),
            "choice_type": "arbitrary_required",
            "where_used": "external validators",
            "status": "kept",
            "why": "Post-label validators are measured in three consecutive 30-day blocks after the label window.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "bootstrap_iterations",
            "choice_value": str(BOOTSTRAP_ITERATIONS),
            "choice_type": "arbitrary_required",
            "where_used": "definition diagnostics and prediction confidence intervals",
            "status": "kept",
            "why": "Bootstrap needs a finite number of resamples. The value is surfaced explicitly instead of being hidden in the code.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "temporal_splitter_test_periods",
            "choice_value": "1",
            "choice_type": "mechanical",
            "where_used": "ExpandingMonthSplit",
            "status": "kept",
            "why": "The splitter evaluates one unique month at a time to preserve month boundaries in the panel.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "temporal_splitter_max_outer_test_months",
            "choice_value": str(MAX_OUTER_TEST_MONTHS),
            "choice_type": "arbitrary_required",
            "where_used": "definition search and outer model backtest",
            "status": "kept",
            "why": "The published build limits the expanding outer backtest to the last 5 test months to keep the exact-threshold search and calibrated model comparison computationally feasible. This is surfaced explicitly instead of being hidden.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "minimum_valid_outer_folds_for_official_summary",
            "choice_value": str(MIN_OFFICIAL_VALID_OUTER_FOLDS),
            "choice_type": "mechanical",
            "where_used": "definition comparison and official model frontier",
            "status": "kept",
            "why": "The official build requires at least two valid outer folds before publishing a mean and dispersion across folds. Single-fold results remain diagnostic only.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "minimum_test_rows_for_official_fold",
            "choice_value": str(MIN_OFFICIAL_TEST_ROWS),
            "choice_type": "mechanical",
            "where_used": "definition comparison and official model frontier",
            "status": "kept",
            "why": "O resumo oficial ignora folds com teste muito pequeno, porque eles podem inflar AP, ROC AUC e calibração sem representar um mês estável de operação.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "minimum_test_positives_for_official_fold",
            "choice_value": str(MIN_OFFICIAL_TEST_POSITIVES),
            "choice_type": "mechanical",
            "where_used": "definition comparison and official model frontier",
            "status": "kept",
            "why": "O resumo oficial exige um número mínimo de positivos no teste para evitar folds com classe rara demais entrando como se fossem evidência forte.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "minimum_test_negatives_for_official_fold",
            "choice_value": str(MIN_OFFICIAL_TEST_NEGATIVES),
            "choice_type": "mechanical",
            "where_used": "definition comparison and official model frontier",
            "status": "kept",
            "why": "O resumo oficial também exige um número mínimo de negativos no teste, para não publicar métricas baseadas em contraste quase inexistente entre as classes.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "calibration_method",
            "choice_value": CALIBRATION_METHOD,
            "choice_type": "mechanical",
            "where_used": "CalibratedClassifierCV official path",
            "status": "kept",
            "why": "The official path follows the library-guided default for smaller calibration samples and preserves monotonic ranking.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "light_temporal_tuning_enabled",
            "choice_value": str(TUNING_ENABLED),
            "choice_type": "mechanical",
            "where_used": "official model comparison",
            "status": "kept",
            "why": "O build oficial aplica uma busca leve de hiperparâmetros dentro do treino do outer fold, sempre em ordem temporal e sem tocar o teste externo.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "light_temporal_tuning_iterations",
            "choice_value": str(TUNING_N_ITER),
            "choice_type": "arbitrary_required",
            "where_used": "official model comparison",
            "status": "kept",
            "why": "A busca de hiperparâmetros usa poucas iterações para manter o custo controlado. Esse limite continua explícito no relatório.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "light_temporal_tuning_max_inner_splits",
            "choice_value": str(TUNING_MAX_INNER_SPLITS),
            "choice_type": "mechanical",
            "where_used": "official model comparison",
            "status": "kept",
            "why": "A busca temporal de hiperparâmetros usa poucos inner splits mensais para continuar leve e preservar ordem temporal.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "light_temporal_tuning_scoring",
            "choice_value": TUNING_SCORING,
            "choice_type": "arbitrary_required",
            "where_used": "official model comparison",
            "status": "kept",
            "why": "A busca temporal leve precisa de uma métrica-guia; o build oficial usa Brier negativo para privilegiar qualidade probabilística já no tuning.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "model_family_scope",
            "choice_value": stable_json(RUNTIME_CONFIG.model_family_scope),
            "choice_type": "arbitrary_required",
            "where_used": "official model comparison",
            "status": "kept",
            "why": "The official comparison is intentionally limited to the three requested model families and does not claim that this scope is exhaustive.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "estimator_random_seed",
            "choice_value": "42",
            "choice_type": "mechanical",
            "where_used": "random_forest and catboost official path",
            "status": "kept",
            "why": "The official path fixes the estimator seed for reproducibility across reruns.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "estimator_parallel_workers",
            "choice_value": f"model_threads={MODEL_COMPARISON_WORKERS};random_forest_n_jobs=1;catboost_thread_count=1",
            "choice_type": "mechanical",
            "where_used": "official model comparison",
            "status": "kept",
            "why": "The official path compares the requested model families in parallel threads while constraining estimator-level parallelism to avoid nested oversubscription during calibrated temporal backtests.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "cluster_k_candidate_grid",
            "choice_value": stable_json(CLUSTER_K_CANDIDATES),
            "choice_type": "arbitrary_required",
            "where_used": "published descriptive cluster layer",
            "status": "kept",
            "why": "A grade finita de k ainda e uma escolha livre. O build oficial torna essa grade explicita e escolhe k dentro dela pelo melhor silhouette no mart cluster_ready.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "cluster_bootstrap_iterations",
            "choice_value": str(CLUSTER_BOOTSTRAP_ITERATIONS),
            "choice_type": "arbitrary_required",
            "where_used": "published descriptive cluster validation",
            "status": "kept",
            "why": "A validacao descritiva de cluster usa um numero finito de reamostragens para relatar estabilidade por ARI.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "cluster_sample_size",
            "choice_value": str(CLUSTER_SAMPLE_SIZE),
            "choice_type": "arbitrary_required",
            "where_used": "published descriptive cluster validation",
            "status": "kept",
            "why": "O silhouette e os refits bootstrap de cluster usam amostra maxima fixa para manter o tempo de execucao controlado e reprodutivel.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "heavy_user_pca_proxy",
            "choice_value": stable_json(
                [
                    "future_business_active_weeks",
                    "future_sessions",
                    "future_session_minutes",
                    "future_active_days",
                    "future_distinct_actions",
                    "future_activity_events",
                    "future_downloads",
                    "future_content_views",
                    "future_mapped_lessons",
                    "future_formation_events",
                ]
            ),
            "choice_type": "arbitrary_required",
            "where_used": "published heavy-user layer",
            "status": "kept",
            "why": "Heavy user continua sendo um proxy sintetico. No oficial ele passa a usar PCA sobre metricas nativas futuras, com a formula explicitada em vez de pesos escondidos.",
            "in_official_report_flag": 1,
        },
        {
            "choice_name": "registered_cutoff_and_band_policies",
            "choice_value": stable_json([row["policy_name"] for row in REGISTERED_BAND_POLICIES]),
            "choice_type": "arbitrary_required",
            "where_used": "published operational overlay",
            "status": "kept",
            "why": "Cutoff e bandas entram como politicas registradas e configuraveis. O oficial deixa claro que a politica usada e uma escolha de operacao, nao uma descoberta da modelagem.",
            "in_official_report_flag": 1,
        },
    ]
    return pd.DataFrame(rows)

def build_policy_registry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "policy_group": "feature_selection",
            "policy_name": "official_feature_policy_note",
            "policy_value_json": stable_json(
                {
                    "official_policy": [
                        "feature_in_contract",
                        "available_at_score_time",
                        "pit_safe_for_track",
                    ],
                    "supervised_statistical_selection": False,
                }
            ),
            "active_in_build_flag": 1,
            "official_flag": 1,
            "why": "No oficial, nao existe selecao supervisionada por teste estatistico. A politica oficial e contrato + disponibilidade no score time + PIT-safe. Importancia e permutacao entra como inspecao publicada, nunca como gate de inclusao.",
        }
    ]
    for row in REGISTERED_BAND_POLICIES:
        rows.append(
            {
                "policy_group": "risk_band_policy",
                "policy_name": row["policy_name"],
                "policy_value_json": row["parameter_json"],
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": f"{row['description']} A politica continua registrada como dependente de cutoff, mas passa a aparecer na camada oficial como overlay operacional configuravel.",
            }
        )
    for row in HEAVY_USER_PERCENTILE_POLICIES:
        rows.append(
            {
                "policy_group": "heavy_user_policy",
                "policy_name": row["policy_name"],
                "policy_value_json": row["parameter_json"],
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": f"{row['description']} O corte continua configuravel e explicitamente registrado.",
            }
        )
    rows.extend(
        [
            {
                "policy_group": "threshold_metrics",
                "policy_name": "precision_recall_f1_by_cutoff",
                "policy_value_json": stable_json({"event_of_interest": "future_inactivity", "score_used": "risk_score"}),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "Precision, recall, F1 e matriz de confusao por cutoff passam a ser publicados como camada operacional registrada, sempre com politica de corte explicitada.",
            },
            {
                "policy_group": "monthly_rate_metrics",
                "policy_name": "r2_and_mape_on_monthly_realized_risk",
                "policy_value_json": stable_json({"target": "monthly_realized_inactivity_rate", "prediction": "monthly_mean_risk_score"}),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "R2 e MAPE entram como leitura mensal agregada de risco realizado versus risco previsto; nao sao usados como criterio de selecao do modelo linha a linha.",
            },
            {
                "policy_group": "feature_inspection",
                "policy_name": "permutation_importance_neg_brier",
                "policy_value_json": stable_json(
                    {
                        "n_repeats": FEATURE_IMPORTANCE_PERMUTATION_REPEATS,
                        "scorer": "neg_brier_score",
                    }
                ),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "Importancia por permutacao entra como inspecao publicada de sinais e nao como gate oficial de selecao de variaveis.",
            },
            {
                "policy_group": "leakage_audit",
                "policy_name": "definition_b_structural_leakage_audit",
                "policy_value_json": stable_json(
                    {
                        "checks": [
                            "available_at_score_time",
                            "pit_safe",
                            "label_window_after_score",
                            "source_table_overlap",
                            "source_column_overlap",
                            "future_named_source_columns",
                        ]
                    }
                ),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "A Definition B recebe uma auditoria estrutural expandida para verificar score time, PIT-safety e qualquer toque de origem com a janela futura do label.",
            },
            {
                "policy_group": "leakage_audit",
                "policy_name": "definition_b_feature_block_gain_test",
                "policy_value_json": stable_json(
                    {
                        "baseline": "context_only",
                        "reference_model": "logistic_regression",
                        "candidate_blocks": ["feature_class", "behavior_family", "full_allowed_features"],
                        "abnormal_uplift_threshold": 0.90,
                    }
                ),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "A Definition B passa por um teste de ganho incremental por bloco de features em cima de um baseline de contexto. Saltos anormais viram diagnostico materializado de suspeita, nao prova definitiva.",
            },
            {
                "policy_group": "leakage_audit",
                "policy_name": "definition_b_excessive_separation_red_flag",
                "policy_value_json": stable_json(
                    {
                        "comparison_scope": "within_track",
                        "combined_score_quantile_flag": 0.95,
                        "metrics": ["ap", "roc_auc", "brier", "log_loss", "calibration", "stability"],
                    }
                ),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "Resultados bons demais para ser verdade entram como red flag relativa dentro da trilha, combinando performance, calibracao e estabilidade. Isso e diagnostico complementar e nao prova unica de leakage.",
            },
            {
                "policy_group": "cluster_policy",
                "policy_name": "kmeans_cluster_ready_grid",
                "policy_value_json": stable_json({"algorithm": "kmeans", "k_candidates": CLUSTER_K_CANDIDATES, "bootstrap_iterations": CLUSTER_BOOTSTRAP_ITERATIONS}),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "Clusters entram como camada descritiva sobre o mart cluster_ready, com algoritmo fixo e escolha data-driven de k dentro da grade registrada.",
            },
            {
                "policy_group": "heavy_user_policy",
                "policy_name": "pca_heavy_intensity_score",
                "policy_value_json": stable_json(
                    {
                        "algorithm": "pca_first_component",
                        "metrics": [
                            "future_business_active_weeks",
                            "future_sessions",
                            "future_session_minutes",
                            "future_active_days",
                            "future_distinct_actions",
                            "future_activity_events",
                            "future_downloads",
                            "future_content_views",
                            "future_mapped_lessons",
                            "future_formation_events",
                        ],
                    }
                ),
                "active_in_build_flag": 1,
                "official_flag": 1,
                "why": "Heavy user passa a usar um score continuo de intensidade baseado no primeiro componente principal das metricas nativas futuras, sem pesos manuais fixos.",
            },
        ]
    )
    return pd.DataFrame(rows)

def build_feature_registry() -> pd.DataFrame:
    rows = [
        {
            "feature_name": "months_after_entry",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["data_entrada_month", "first_month"]),
            "pit_class": "context",
            "behavior_family": "time_since_entry",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "teacher_population_status",
            "feature_type": "categorical",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["teacher_population_status"]),
            "pit_class": "context",
            "behavior_family": "registration_context",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "utm_group",
            "feature_type": "categorical",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["utm_group"]),
            "pit_class": "context",
            "behavior_family": "acquisition_context",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "first_session_entry_surface",
            "feature_type": "categorical",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_first_event_utm_source", "first_utm_source"]),
            "pit_class": "context",
            "behavior_family": "entry_surface",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "first_session_device_bucket",
            "feature_type": "categorical",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_first_event_device", "first_device"]),
            "pit_class": "context",
            "behavior_family": "device_context",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "first_event_missing_flag",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first_event_type"]),
            "pit_class": "context",
            "behavior_family": "data_quality",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "first_device_missing_flag",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first_device"]),
            "pit_class": "context",
            "behavior_family": "data_quality",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "first_utm_missing_flag",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first_utm_source"]),
            "pit_class": "context",
            "behavior_family": "data_quality",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "session_without_interaction_flag",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_interactions"]),
            "pit_class": "context",
            "behavior_family": "data_quality",
            "feature_class": "context",
            "allowed_in_S1": 1,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 1,
        },
        {
            "feature_name": "first_session_duration_min",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_duration_min"]),
            "pit_class": "behavioral_early",
            "behavior_family": "session_depth",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_interactions",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_interactions"]),
            "pit_class": "behavioral_early",
            "behavior_family": "session_depth",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_downloads",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_downloads"]),
            "pit_class": "behavioral_early",
            "behavior_family": "early_downloads",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_views",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_views"]),
            "pit_class": "behavioral_early",
            "behavior_family": "early_views",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_other_actions",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_other_actions"]),
            "pit_class": "behavioral_early",
            "behavior_family": "early_actions",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_navigation_events",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_navigation_events"]),
            "pit_class": "behavioral_early",
            "behavior_family": "early_navigation",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_meaningful_events",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_meaningful_events"]),
            "pit_class": "behavioral_early",
            "behavior_family": "early_meaningful_use",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_has_interaction_flag",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_interactions"]),
            "pit_class": "behavioral_early",
            "behavior_family": "early_presence",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_has_meaningful_action_flag",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_meaningful_events"]),
            "pit_class": "behavioral_early",
            "behavior_family": "early_presence",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "secs_to_first_interaction",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_start_ts", "first_session_first_event_ts"]),
            "pit_class": "behavioral_early",
            "behavior_family": "time_to_action",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "secs_to_first_meaningful_action",
            "feature_type": "numeric",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_start_ts", "first_session_first_meaningful_ts"]),
            "pit_class": "behavioral_early",
            "behavior_family": "time_to_action",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_first_event_action_group",
            "feature_type": "categorical",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_first_event_action"]),
            "pit_class": "behavioral_early",
            "behavior_family": "action_type",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_first_meaningful_action_group",
            "feature_type": "categorical",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_first_meaningful_action"]),
            "pit_class": "behavioral_early",
            "behavior_family": "action_type",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first_session_exit_state",
            "feature_type": "categorical",
            "source_table": "mart_first_session_journey_v1",
            "source_columns_json": stable_json(["first_session_downloads", "first_session_views", "first_session_other_actions", "first_session_navigation_events"]),
            "pit_class": "behavioral_early",
            "behavior_family": "exit_state",
            "feature_class": "s1",
            "allowed_in_S1": 1,
            "allowed_in_S7": 0,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first7d_events",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first7d_events"]),
            "pit_class": "behavioral_week",
            "behavior_family": "week_depth",
            "feature_class": "s7",
            "allowed_in_S1": 0,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first7d_active_days",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first7d_active_days"]),
            "pit_class": "behavioral_week",
            "behavior_family": "week_presence",
            "feature_class": "s7",
            "allowed_in_S1": 0,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first7d_sessions",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first7d_sessions"]),
            "pit_class": "behavioral_week",
            "behavior_family": "week_sessions",
            "feature_class": "s7",
            "allowed_in_S1": 0,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first7d_session_minutes",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first7d_session_minutes"]),
            "pit_class": "behavioral_week",
            "behavior_family": "week_minutes",
            "feature_class": "s7",
            "allowed_in_S1": 0,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first3_interaction_downloads",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first3_interaction_downloads"]),
            "pit_class": "behavioral_week",
            "behavior_family": "week_downloads",
            "feature_class": "s7",
            "allowed_in_S1": 0,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first3_interaction_views",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first3_interaction_views"]),
            "pit_class": "behavioral_week",
            "behavior_family": "week_views",
            "feature_class": "s7",
            "allowed_in_S1": 0,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
        {
            "feature_name": "first3_interaction_other_actions",
            "feature_type": "numeric",
            "source_table": "mart_onboarding_population_v1",
            "source_columns_json": stable_json(["first3_interaction_other_actions"]),
            "pit_class": "behavioral_week",
            "behavior_family": "week_actions",
            "feature_class": "s7",
            "allowed_in_S1": 0,
            "allowed_in_S7": 1,
            "allowed_in_S1_PLUS_S7": 1,
            "allowed_in_STRICT_CONTEXT": 0,
        },
    ]
    return pd.DataFrame(rows)

def build_candidate_metric_registry() -> pd.DataFrame:
    candidate_metric_rows = RUNTIME_OVERRIDES.get("candidate_metrics")
    if candidate_metric_rows:
        return pd.DataFrame(candidate_metric_rows)
    rows = [
        {
            "metric_name": "future_business_active_weeks",
            "metric_type": "count",
            "source_columns_json": stable_json(["session_start_ts", "interaction_ts", "is_activity_event"]),
            "definition_role": "definition_b_literal_comparator",
            "definition_a_candidate_flag": 1,
            "semantic_group": "recurrence",
        },
        {
            "metric_name": "future_sessions",
            "metric_type": "count",
            "source_columns_json": stable_json(["session_start_ts"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "access",
        },
        {
            "metric_name": "future_session_minutes",
            "metric_type": "continuous",
            "source_columns_json": stable_json(["session_start_ts", "duration_min"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "time_spent",
        },
        {
            "metric_name": "future_interactions",
            "metric_type": "count",
            "source_columns_json": stable_json(["interaction_ts"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "activity",
        },
        {
            "metric_name": "future_activity_events",
            "metric_type": "count",
            "source_columns_json": stable_json(["interaction_ts", "is_activity_event"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "activity",
        },
        {
            "metric_name": "future_active_days",
            "metric_type": "count",
            "source_columns_json": stable_json(["interaction_ts"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "recurrence",
        },
        {
            "metric_name": "future_distinct_actions",
            "metric_type": "count",
            "source_columns_json": stable_json(["interaction_ts", "event_action", "event_family", "event_type"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "diversity",
        },
        {
            "metric_name": "future_downloads",
            "metric_type": "count",
            "source_columns_json": stable_json(["interaction_ts", "is_download_event"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "value",
        },
        {
            "metric_name": "future_content_views",
            "metric_type": "count",
            "source_columns_json": stable_json(["interaction_ts", "is_content_view_event"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "consumption",
        },
        {
            "metric_name": "future_mapped_lessons",
            "metric_type": "count",
            "source_columns_json": stable_json(["interaction_ts", "lesson_mapped_flag"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "lesson_use",
        },
        {
            "metric_name": "future_formation_events",
            "metric_type": "count",
            "source_columns_json": stable_json(["formation_ts"]),
            "definition_a_candidate_flag": 1,
            "semantic_group": "formation",
        },
        {
            "metric_name": "future_mari_help_events",
            "metric_type": "count",
            "source_columns_json": stable_json(["help_ts"]),
            "definition_a_candidate_flag": 0,
            "semantic_group": "support",
        },
        {
            "metric_name": "future_mari_conversation_events",
            "metric_type": "count",
            "source_columns_json": stable_json(["mari_created_ts"]),
            "definition_a_candidate_flag": 0,
            "semantic_group": "support",
        },
    ]
    if not RUNTIME_CONFIG.definition_a_enabled:
        for row in rows:
            row["definition_a_candidate_flag"] = 0
        return pd.DataFrame(rows)
    allowed_candidate_metrics = set(RUNTIME_CONFIG.definition_a_candidate_metrics)
    if allowed_candidate_metrics:
        for row in rows:
            row["definition_a_candidate_flag"] = 1 if row["metric_name"] in allowed_candidate_metrics else 0
    return pd.DataFrame(rows)

def build_label_registry(
    official_definition_a_rows: pd.DataFrame,
    definition_b_row: pd.DataFrame,
) -> pd.DataFrame:
    def extract_metric_names(rule_payload: str) -> list[str]:
        try:
            rule = json.loads(rule_payload)
        except Exception:
            return []
        return extract_rule_metric_names(rule) if isinstance(rule, dict) else []

    rows: List[dict[str, Any]] = []
    for row in official_definition_a_rows.to_dict(orient="records"):
        rows.append(
            {
                "label_name": f'definition_a::{row["rule_text"]}',
                "label_group": "definition_a",
                "official_flag": 1,
                "source_table": "mart_future_metrics_v1",
                "source_columns_json": stable_json(extract_metric_names(row["rule_json"])),
                "rule_json": row["rule_json"],
                "window_start_day": 8,
                "window_end_day": LABEL_WINDOW_DAYS + 7,
            }
        )
    rows.append(
        {
            "label_name": "definition_b_label",
            "label_group": "definition_b",
            "official_flag": 1,
            "source_table": "mart_future_metrics_v1",
            "source_columns_json": stable_json([definition_b_row["metric_name"]]),
            "rule_json": definition_b_row["rule_json"],
            "window_start_day": 8,
            "window_end_day": LABEL_WINDOW_DAYS + 7,
        }
    )
    rows.extend(
        [
            {
                "label_name": "returned_active_post_label_m1",
                "label_group": "external_validator",
                "official_flag": 0,
                "source_table": "mart_future_metrics_v1",
                "source_columns_json": stable_json(["returned_active_post_label_m1"]),
                "rule_json": stable_json({"metric_name": "returned_active_post_label_m1", "operator": "=", "threshold": 1}),
                "window_start_day": LABEL_WINDOW_DAYS + 8,
                "window_end_day": LABEL_WINDOW_DAYS + 7 + POST_LABEL_BLOCK_DAYS,
            },
            {
                "label_name": "returned_active_post_label_m2",
                "label_group": "external_validator",
                "official_flag": 0,
                "source_table": "mart_future_metrics_v1",
                "source_columns_json": stable_json(["returned_active_post_label_m2"]),
                "rule_json": stable_json({"metric_name": "returned_active_post_label_m2", "operator": "=", "threshold": 1}),
                "window_start_day": LABEL_WINDOW_DAYS + 8 + POST_LABEL_BLOCK_DAYS,
                "window_end_day": LABEL_WINDOW_DAYS + 7 + POST_LABEL_BLOCK_DAYS * 2,
            },
            {
                "label_name": "returned_active_post_label_m3",
                "label_group": "external_validator",
                "official_flag": 0,
                "source_table": "mart_future_metrics_v1",
                "source_columns_json": stable_json(["returned_active_post_label_m3"]),
                "rule_json": stable_json({"metric_name": "returned_active_post_label_m3", "operator": "=", "threshold": 1}),
                "window_start_day": LABEL_WINDOW_DAYS + 8 + POST_LABEL_BLOCK_DAYS * 2,
                "window_end_day": LABEL_WINDOW_DAYS + 7 + POST_LABEL_BLOCK_DAYS * 3,
            },
            {
                "label_name": "active_days_post_label_3m",
                "label_group": "external_validator",
                "official_flag": 0,
                "source_table": "mart_future_metrics_v1",
                "source_columns_json": stable_json(["active_days_post_label_3m"]),
                "rule_json": stable_json({"metric_name": "active_days_post_label_3m", "operator": "identity"}),
                "window_start_day": LABEL_WINDOW_DAYS + 8,
                "window_end_day": LABEL_WINDOW_DAYS + 7 + POST_LABEL_BLOCK_DAYS * 3,
            },
            {
                "label_name": "sustained_active_2of3_post_label",
                "label_group": "external_validator",
                "official_flag": 0,
                "source_table": "mart_future_metrics_v1",
                "source_columns_json": stable_json(["returned_active_post_label_m1", "returned_active_post_label_m2", "returned_active_post_label_m3"]),
                "rule_json": stable_json({"metric_name": "sustained_active_2of3_post_label", "operator": "=", "threshold": 1}),
                "window_start_day": LABEL_WINDOW_DAYS + 8,
                "window_end_day": LABEL_WINDOW_DAYS + 7 + POST_LABEL_BLOCK_DAYS * 3,
            },
        ]
    )
    return pd.DataFrame(rows)
