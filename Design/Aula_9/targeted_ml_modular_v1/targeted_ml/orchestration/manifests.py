from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.orchestration.compatibility import CompatibilityResult
from targeted_ml.orchestration.artifacts import ProjectPaths, write_json


def hash_spec(spec: AnalysisSpec) -> str:
    raw = json.dumps(spec.model_dump(mode="json"), ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_revision(project_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def write_run_manifest(paths: ProjectPaths, spec: AnalysisSpec, stage_name: str, extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "analysis_name": spec.analysis_name,
        "analysis_kind": spec.analysis_kind,
        "dataset_root": str(spec.data.dataset_root),
        "spec_hash": hash_spec(spec),
        "git_revision": git_revision(paths.project_root),
        "stage_name": stage_name,
        "runtime_overrides": spec.runtime_overrides(),
    }
    if extra:
        payload.update(extra)
    path = paths.metadata_dir / f"run_manifest_{stage_name}.json"
    write_json(path, payload)
    return path


def write_compatibility_snapshot(paths: ProjectPaths, sections: list[str]) -> Path:
    parquet_tables = sorted(f.name for f in paths.tables_dir.glob("*.parquet"))
    payload = {
        "artifact_tables": parquet_tables,
        "summary_files": sorted(f.name for f in paths.metadata_dir.glob("*.json")),
        "report_sections": sections,
    }
    path = paths.metadata_dir / "compatibility_contract_current.json"
    write_json(path, payload)
    return path


def write_compatibility_result(paths: ProjectPaths, result: CompatibilityResult) -> Path:
    path = paths.metadata_dir / "compatibility_check_v1.json"
    write_json(path, result.model_dump())
    return path
