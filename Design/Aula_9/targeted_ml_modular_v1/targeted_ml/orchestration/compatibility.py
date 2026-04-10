from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field
from pyarrow import parquet as pq


class CompatibilityError(RuntimeError):
    """Raised when the zero-loss contract is broken."""


class CompatibilityResult(BaseModel):
    baseline_contract: str
    current_contract: str
    missing_artifact_tables: list[str] = Field(default_factory=list)
    missing_report_sections: list[str] = Field(default_factory=list)
    extra_artifact_tables: list[str] = Field(default_factory=list)
    extra_report_sections: list[str] = Field(default_factory=list)
    missing_columns_by_table: dict[str, list[str]] = Field(default_factory=dict)
    extra_columns_by_table: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def is_compatible(self) -> bool:
        return (
            not self.missing_artifact_tables
            and not self.missing_report_sections
            and not self.missing_columns_by_table
        )


CURRENT_TO_BASELINE_ARTIFACT_NAME_MAP = {
    "core_scoring_scenarios_v1.parquet": "core_problem_catalog_v1.parquet",
    "post_model_threshold_metrics_v1.parquet": "overlay_threshold_metrics_v1.parquet",
    "post_model_confusion_matrix_v1.parquet": "overlay_confusion_matrix_v1.parquet",
    "post_model_band_summary_v1.parquet": "overlay_band_summary_v1.parquet",
    "post_model_monthly_fit_v1.parquet": "overlay_monthly_fit_v1.parquet",
    "post_model_cv_threshold_folds_v1.parquet": "overlay_cv_threshold_folds_v1.parquet",
    "post_model_cv_confusion_folds_v1.parquet": "overlay_cv_confusion_folds_v1.parquet",
    "post_model_cv_threshold_summary_v1.parquet": "overlay_cv_threshold_summary_v1.parquet",
    "post_model_cv_confusion_summary_v1.parquet": "overlay_cv_confusion_summary_v1.parquet",
    "post_model_feature_importance_v1.parquet": "overlay_feature_importance_v1.parquet",
    "post_model_reference_selection_v1.parquet": "overlay_reference_scope_v1.parquet",
    "post_model_cluster_assignment_v1.parquet": "overlay_cluster_assignment_v1.parquet",
    "post_model_cluster_profile_v1.parquet": "overlay_cluster_profile_v1.parquet",
    "post_model_cluster_summary_v1.parquet": "overlay_cluster_summary_v1.parquet",
    "post_model_cluster_validation_v1.parquet": "overlay_cluster_validation_v1.parquet",
    "post_model_heavy_user_scores_v1.parquet": "overlay_heavy_user_scores_v1.parquet",
    "post_model_heavy_user_profile_v1.parquet": "overlay_heavy_user_profile_v1.parquet",
    "post_model_heavy_user_summary_v1.parquet": "overlay_heavy_user_summary_v1.parquet",
    "governance_post_model_output_status_v1.parquet": "governance_overlay_status_v1.parquet",
}


def _normalize_artifact_names(artifact_names: set[str]) -> set[str]:
    return {CURRENT_TO_BASELINE_ARTIFACT_NAME_MAP.get(name, name) for name in artifact_names}


def _load_contract(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_contract_payloads(baseline: dict, current: dict, baseline_path: Path | None = None, current_path: Path | None = None) -> CompatibilityResult:
    baseline_artifacts = set(baseline.get("artifact_tables", []))
    current_artifacts = _normalize_artifact_names(set(current.get("artifact_tables", [])))
    baseline_sections = set(baseline.get("report_sections", []))
    current_sections = set(current.get("report_sections", []))
    return CompatibilityResult(
        baseline_contract=str(baseline_path or "<memory>"),
        current_contract=str(current_path or "<memory>"),
        missing_artifact_tables=sorted(baseline_artifacts - current_artifacts),
        missing_report_sections=sorted(baseline_sections - current_sections),
        extra_artifact_tables=sorted(current_artifacts - baseline_artifacts),
        extra_report_sections=sorted(current_sections - baseline_sections),
    )


def _parquet_columns(build_dir: Path) -> dict[str, list[str]]:
    tables_dir = build_dir / "tables"
    legacy_tables_dir = build_dir / "parquet"
    source_dir = tables_dir if tables_dir.exists() else legacy_tables_dir
    if not source_dir.exists():
        return {}
    output: dict[str, list[str]] = {}
    for path in sorted(source_dir.glob("*.parquet")):
        normalized_name = CURRENT_TO_BASELINE_ARTIFACT_NAME_MAP.get(path.name, path.name)
        output[normalized_name] = pq.read_schema(path).names
    return output


def _compare_parquet_columns(baseline_build_dir: Path, current_build_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    baseline_columns = _parquet_columns(baseline_build_dir)
    current_columns = _parquet_columns(current_build_dir)
    missing: dict[str, list[str]] = {}
    extra: dict[str, list[str]] = {}
    for table_name in sorted(set(baseline_columns) & set(current_columns)):
        baseline_set = set(baseline_columns[table_name])
        current_set = set(current_columns[table_name])
        if baseline_set - current_set:
            missing[table_name] = sorted(baseline_set - current_set)
        if current_set - baseline_set:
            extra[table_name] = sorted(current_set - baseline_set)
    return missing, extra


def compare_contracts(baseline_path: Path, current_path: Path, baseline_build_dir: Path | None = None, current_build_dir: Path | None = None) -> CompatibilityResult:
    result = compare_contract_payloads(
        _load_contract(baseline_path),
        _load_contract(current_path),
        baseline_path=baseline_path,
        current_path=current_path,
    )
    if baseline_build_dir and current_build_dir and baseline_build_dir.exists() and current_build_dir.exists():
        missing_cols, extra_cols = _compare_parquet_columns(baseline_build_dir, current_build_dir)
        result.missing_columns_by_table = missing_cols
        result.extra_columns_by_table = extra_cols
    if not result.is_compatible:
        raise CompatibilityError(
            "zero-loss compatibility failed: "
            f"missing tables={result.missing_artifact_tables}, "
            f"missing sections={result.missing_report_sections}, "
            f"missing columns={result.missing_columns_by_table}"
        )
    return result
