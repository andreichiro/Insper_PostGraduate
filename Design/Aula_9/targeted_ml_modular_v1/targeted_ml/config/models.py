from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class DataSpec(BaseModel):
    dataset_root: Path
    modeled_source: str = "auto"
    rebuild_modelled_if_missing: bool = False
    modeled_duckdb_relative_path: Path = Path("data/modelled/duckdb/base_modelada_v2.duckdb")
    raw_relative_path: Path = Path("raw/base_aprendizap")

    @field_validator("modeled_source")
    @classmethod
    def validate_modeled_source(cls, value: str) -> str:
        allowed = {"auto", "duckdb", "parquet", "raw"}
        normalized = str(value).strip().lower()
        if normalized not in allowed:
            raise ValueError(f"modeled_source must be one of {sorted(allowed)}")
        return normalized


class PopulationSpec(BaseModel):
    official_population: str = "all_observed_first_use"

    @field_validator("official_population")
    @classmethod
    def validate_official_population(cls, value: str) -> str:
        allowed = {"all_observed_first_use", "same_month_entry_only"}
        normalized = str(value).strip().lower()
        if normalized not in allowed:
            raise ValueError(f"population.official_population must be one of {sorted(allowed)}")
        return normalized


class DefinitionASpec(BaseModel):
    enabled: bool = True
    strategy: str = "screened_pairwise_compound_weighted"
    candidate_metrics: list[str] = Field(default_factory=list)
    promoted_candidate_limit: int = 3
    sql_file: Path | None = None
    python_strategy: str | None = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        alias_map = {"univariate": "univariate_exact"}
        allowed = {"univariate_exact", "screened_pairwise_compound_weighted"}
        normalized = str(value).strip().lower()
        normalized = alias_map.get(normalized, normalized)
        if normalized not in allowed:
            raise ValueError(f"definition_a.strategy must be one of {sorted(allowed)}")
        return normalized

    @model_validator(mode="after")
    def validate_official_constraints(self) -> "DefinitionASpec":
        return self


class DefinitionBSpec(BaseModel):
    definition_name: str = "definition_b"
    metric_name: str = "future_business_active_weeks"
    operator: str = ">="
    threshold: float = 1.0
    rule_text: str | None = None
    sql_file: Path | None = None
    python_strategy: str | None = None


class LabelSpec(BaseModel):
    window_days: int = 30
    post_label_block_days: int = 30
    post_label_block_count: int = 3
    external_validators: list[str] = Field(default_factory=list)
    definition_a: DefinitionASpec = Field(default_factory=DefinitionASpec)
    definition_b: DefinitionBSpec = Field(default_factory=DefinitionBSpec)


class TrackSpec(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["S1", "S7", "S1_PLUS_S7", "STRICT_CONTEXT"])


class NumericGateSpec(BaseModel):
    column_name: str = "lock_gap_sustained_active_2of3_post_label_ci_low"
    operator: str = ">"
    threshold: float = 0.0

    @field_validator("column_name")
    @classmethod
    def validate_column_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("numeric gate column_name must be a non-empty string")
        return normalized

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, value: str) -> str:
        allowed = {">", ">=", "<", "<=", "==", "!="}
        normalized = str(value).strip()
        if normalized not in allowed:
            raise ValueError(f"numeric gate operator must be one of {sorted(allowed)}")
        return normalized


class ModelingSpec(BaseModel):
    max_outer_test_months: int = 6
    definition_selection_holdout_months: int = 6
    definition_lock_months: int = 6
    min_official_valid_outer_folds: int = 2
    min_official_test_rows: int = 50
    min_official_test_positives: int = 5
    min_official_test_negatives: int = 20
    definition_lock_bootstrap_gate: NumericGateSpec = Field(default_factory=NumericGateSpec)
    tuning_enabled: bool = True
    tuning_n_iter: int = 4
    tuning_max_inner_splits: int = 3
    tuning_scoring: str = "neg_brier_score"
    model_families: list[str] = Field(default_factory=lambda: ["logistic_regression", "random_forest", "catboost"])
    workers: int = 6
    calibration_method: str = "sigmoid"
    skip_post_model_refit: bool = False


class PolicySpec(BaseModel):
    policy_name: str
    policy_type: str
    parameter_json: str
    description: str


class PostModelOutputsSpec(BaseModel):
    feature_importance_permutation_repeats: int = 5
    cluster_k_candidates: list[int] = Field(default_factory=lambda: [2, 3, 4, 5, 6])
    cluster_bootstrap_iterations: int = 20
    cluster_sample_size: int = 10000
    band_policies: list[PolicySpec] = Field(default_factory=list)
    heavy_user_policies: list[PolicySpec] = Field(default_factory=list)


class ReportSpec(BaseModel):
    output_html_name: str = "targeted_ml_report_v1.html"
    render_single_report: bool = True


class AnalysisSpec(BaseModel):
    analysis_name: str
    analysis_kind: str
    data: DataSpec
    population: PopulationSpec = Field(default_factory=PopulationSpec)
    label: LabelSpec
    tracks: TrackSpec = Field(default_factory=TrackSpec)
    modeling: ModelingSpec = Field(default_factory=ModelingSpec)
    post_model_outputs: PostModelOutputsSpec = Field(default_factory=PostModelOutputsSpec)
    report: ReportSpec = Field(default_factory=ReportSpec)

    @field_validator("analysis_kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        allowed = {"activity", "churn", "return"}
        if value not in allowed:
            raise ValueError(f"analysis_kind must be one of {sorted(allowed)}")
        return value

    def runtime_overrides(self) -> dict[str, Any]:
        return {
            "label_window_days": self.label.window_days,
            "post_label_block_days": self.label.post_label_block_days,
            "post_label_block_count": self.label.post_label_block_count,
            "external_validators": self.label.external_validators,
            "enabled_tracks": self.tracks.enabled,
            "analysis_kind": self.analysis_kind,
            "data_modeled_source": self.data.modeled_source,
            "data_raw_relative_path": str(self.data.raw_relative_path),
            "data_modeled_duckdb_relative_path": str(self.data.modeled_duckdb_relative_path),
            "official_population_filter": self.population.official_population,
            "definition_b": self.label.definition_b.model_dump(),
            "definition_a_enabled": self.label.definition_a.enabled,
            "definition_a_strategy": self.label.definition_a.strategy,
            "definition_a_candidate_metrics": self.label.definition_a.candidate_metrics,
            "definition_a_promoted_candidate_limit": self.label.definition_a.promoted_candidate_limit,
            "definition_a_sql_file": str(self.label.definition_a.sql_file) if self.label.definition_a.sql_file else "",
            "definition_a_python_strategy": self.label.definition_a.python_strategy or "",
            "definition_b_sql_file": str(self.label.definition_b.sql_file) if self.label.definition_b.sql_file else "",
            "definition_b_python_strategy": self.label.definition_b.python_strategy or "",
            "max_outer_test_months": self.modeling.max_outer_test_months,
            "definition_selection_holdout_months": self.modeling.definition_selection_holdout_months,
            "definition_lock_months": self.modeling.definition_lock_months,
            "min_official_valid_outer_folds": self.modeling.min_official_valid_outer_folds,
            "min_official_test_rows": self.modeling.min_official_test_rows,
            "min_official_test_positives": self.modeling.min_official_test_positives,
            "min_official_test_negatives": self.modeling.min_official_test_negatives,
            "definition_lock_bootstrap_gate": self.modeling.definition_lock_bootstrap_gate.model_dump(),
            "tuning_enabled": self.modeling.tuning_enabled,
            "tuning_n_iter": self.modeling.tuning_n_iter,
            "tuning_max_inner_splits": self.modeling.tuning_max_inner_splits,
            "tuning_scoring": self.modeling.tuning_scoring,
            "model_family_scope": self.modeling.model_families,
            "model_comparison_workers": self.modeling.workers,
            "calibration_method": self.modeling.calibration_method,
            "feature_importance_permutation_repeats": self.post_model_outputs.feature_importance_permutation_repeats,
            "cluster_k_candidates": self.post_model_outputs.cluster_k_candidates,
            "cluster_bootstrap_iterations": self.post_model_outputs.cluster_bootstrap_iterations,
            "cluster_sample_size": self.post_model_outputs.cluster_sample_size,
            "registered_band_policies": [row.model_dump() for row in self.post_model_outputs.band_policies],
            "heavy_user_percentile_policies": [row.model_dump() for row in self.post_model_outputs.heavy_user_policies],
        }
