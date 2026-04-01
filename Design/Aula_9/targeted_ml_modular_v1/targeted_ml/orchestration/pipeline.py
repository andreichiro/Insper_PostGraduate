from __future__ import annotations

from datetime import datetime
from pathlib import Path

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.data.modelled import build_parquets_from_duckdb, is_complete_modelled_duckdb, prepare_modelled_base
import pandas as pd

from targeted_ml.inference.service import export_reference_models, score_modelled_duckdb, score_scoring_frame
from targeted_ml.orchestration.artifacts import ProjectPaths, build_project_paths, ensure_dirs
from targeted_ml.orchestration.artifacts import write_json
from targeted_ml.orchestration.compatibility import compare_contracts
from targeted_ml.orchestration.manifests import write_compatibility_result, write_compatibility_snapshot, write_run_manifest
from targeted_ml.pipelines.modelled_to_ml.runner import run_build_for_spec
from targeted_ml.reporting.render import build_report
from targeted_ml.reporting.views import OFFICIAL_REPORT_SECTIONS


def resolve_paths(project_root: Path, output_root: Path | None = None) -> ProjectPaths:
    paths = build_project_paths(project_root=project_root, output_root=output_root)
    ensure_dirs(paths)
    return paths


def build_modelled(spec: AnalysisSpec, paths: ProjectPaths) -> Path:
    if is_complete_modelled_duckdb(paths.modelled_duckdb) and not spec.data.rebuild_modelled_if_missing:
        copied_parquets = 0
        if not any(paths.modelled_parquet_dir.glob("*.parquet")):
            copied_parquets = build_parquets_from_duckdb(paths.modelled_duckdb, paths.modelled_parquet_dir)
        extra = {
            "dataset_root": str(spec.data.dataset_root),
            "modeled_source": spec.data.modeled_source,
            "copied_parquets": copied_parquets,
            "modelled_duckdb": str(paths.modelled_duckdb),
            "modelled_parquet_dir": str(paths.modelled_parquet_dir),
            "reused_existing_modelled": True,
        }
    else:
        extra = prepare_modelled_base(spec, paths)
    write_run_manifest(paths, spec, "build_modelled", extra=extra)
    return paths.modelled_duckdb


def build_ml(spec: AnalysisSpec, paths: ProjectPaths) -> Path:
    run_build_for_spec(
        spec=spec,
        modelled_duckdb=paths.modelled_duckdb,
        output_dir=paths.build_dir,
        compute_post_model_refit=not spec.modeling.skip_post_model_refit,
    )
    write_run_manifest(
        paths,
        spec,
        "build_ml",
        extra={
            "build_dir": str(paths.build_dir),
            "output_tables": sorted(f.name for f in paths.tables_dir.glob("*.parquet")),
        },
    )
    return paths.build_dir


def build_html(spec: AnalysisSpec, paths: ProjectPaths) -> Path:
    output_html = build_report(spec, paths)
    current_contract = write_compatibility_snapshot(paths, OFFICIAL_REPORT_SECTIONS)
    baseline_contract = paths.project_root / "baseline" / "compatibility_contract_v2.json"
    baseline_build_dir = paths.project_root / "baseline" / "build_v2"
    if baseline_contract.exists():
        result = compare_contracts(
            baseline_contract,
            current_contract,
            baseline_build_dir=baseline_build_dir,
            current_build_dir=paths.build_dir,
        )
        write_compatibility_result(paths, result)
    write_run_manifest(
        paths,
        spec,
        "build_report",
        extra={
            "output_html": str(output_html),
            "report_sections": OFFICIAL_REPORT_SECTIONS,
        },
    )
    return output_html


def export_serving(
    spec: AnalysisSpec,
    paths: ProjectPaths,
    problem_keys: list[str] | None = None,
    model_names: list[str] | None = None,
) -> Path:
    manifest_path = export_reference_models(
        spec=spec,
        paths=paths,
        problem_keys=problem_keys,
        model_names=model_names,
    )
    write_run_manifest(
        paths,
        spec,
        "export_serving",
        extra={
            "serving_manifest": str(manifest_path),
            "problem_keys": problem_keys or [],
            "model_names": model_names or [],
        },
    )
    return manifest_path


def score_modelled_input(
    spec: AnalysisSpec,
    paths: ProjectPaths,
    modelled_duckdb: Path,
    problem_keys: list[str] | None = None,
    model_names: list[str] | None = None,
    run_name: str | None = None,
) -> Path:
    requested_modelled = modelled_duckdb.resolve()
    if not requested_modelled.exists():
        if requested_modelled == paths.modelled_duckdb.resolve():
            build_modelled(spec, paths)
        else:
            raise FileNotFoundError(f"modelled_duckdb not found: {requested_modelled}")
    run_dir = score_modelled_duckdb(
        paths=paths,
        modelled_duckdb=requested_modelled,
        problem_keys=problem_keys,
        model_names=model_names,
        run_name=run_name,
    )
    write_run_manifest(
        paths,
        spec,
        "score_modelled",
        extra={
            "inference_run_dir": str(run_dir),
            "modelled_duckdb": str(requested_modelled),
            "problem_keys": problem_keys or [],
            "model_names": model_names or [],
            "run_name": run_name or "",
        },
    )
    return run_dir


def score_scoring_frame_input(
    spec: AnalysisSpec,
    paths: ProjectPaths,
    scoring_frame_path: Path,
    latest_observed_ts: str | None = None,
    problem_keys: list[str] | None = None,
    model_names: list[str] | None = None,
    run_name: str | None = None,
) -> Path:
    parsed_latest_ts = pd.to_datetime(latest_observed_ts, errors="coerce") if latest_observed_ts else None
    if latest_observed_ts and pd.isna(parsed_latest_ts):
        raise ValueError(f"Invalid latest_observed_ts: {latest_observed_ts}")
    run_dir = score_scoring_frame(
        paths=paths,
        scoring_frame_path=scoring_frame_path,
        latest_observed_ts=parsed_latest_ts,
        problem_keys=problem_keys,
        model_names=model_names,
        run_name=run_name,
    )
    write_run_manifest(
        paths,
        spec,
        "score_scoring_frame",
        extra={
            "inference_run_dir": str(run_dir),
            "scoring_frame_path": str(scoring_frame_path),
            "latest_observed_ts": latest_observed_ts or "",
            "problem_keys": problem_keys or [],
            "model_names": model_names or [],
            "run_name": run_name or "",
        },
    )
    return run_dir


def score_raw_input(
    spec: AnalysisSpec,
    paths: ProjectPaths,
    dataset_root: Path,
    problem_keys: list[str] | None = None,
    model_names: list[str] | None = None,
    run_name: str | None = None,
) -> Path:
    staging_slug = run_name.strip().replace(" ", "_") if run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_root = paths.inference_runs_dir / "_raw_modelled_staging" / staging_slug
    staging_paths = resolve_paths(paths.project_root, staging_root)
    spec_for_raw = spec.model_copy(deep=True)
    spec_for_raw.data.dataset_root = Path(dataset_root).resolve()
    spec_for_raw.data.modeled_source = "raw"
    build_modelled(spec_for_raw, staging_paths)
    run_dir = score_modelled_duckdb(
        paths=paths,
        modelled_duckdb=staging_paths.modelled_duckdb,
        problem_keys=problem_keys,
        model_names=model_names,
        run_name=run_name,
    )
    write_json(
        run_dir / "raw_input_context.json",
        {
            "dataset_root": str(Path(dataset_root).resolve()),
            "raw_relative_path": str(spec_for_raw.data.raw_relative_path),
            "staging_output_root": str(staging_root),
            "staging_modelled_duckdb": str(staging_paths.modelled_duckdb),
        },
    )
    write_run_manifest(
        paths,
        spec,
        "score_raw",
        extra={
            "inference_run_dir": str(run_dir),
            "dataset_root": str(Path(dataset_root).resolve()),
            "raw_relative_path": str(spec_for_raw.data.raw_relative_path),
            "staging_output_root": str(staging_root),
            "staging_modelled_duckdb": str(staging_paths.modelled_duckdb),
            "problem_keys": problem_keys or [],
            "model_names": model_names or [],
            "run_name": run_name or "",
        },
    )
    return run_dir


def build_all(spec: AnalysisSpec, project_root: Path, output_root: Path | None = None, skip_modelled: bool = False, skip_post_model_refit: bool = False) -> ProjectPaths:
    paths = resolve_paths(project_root=project_root, output_root=output_root)
    if not skip_modelled:
        build_modelled(spec, paths)
    if skip_post_model_refit:
        spec.modeling.skip_post_model_refit = True
    build_ml(spec, paths)
    build_html(spec, paths)
    return paths
