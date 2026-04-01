from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    build_dir: Path
    staging_dir: Path
    serving_dir: Path
    inference_runs_dir: Path
    reports_dir: Path
    metadata_dir: Path
    tables_dir: Path
    duckdb_dir: Path
    modelled_dir: Path
    modelled_parquet_dir: Path
    modelled_duckdb: Path


def build_project_paths(project_root: Path, output_root: Path | None = None) -> ProjectPaths:
    build_dir = (output_root or (project_root / "build")).resolve()
    modelled_dir = (build_dir / "modelled").resolve()
    return ProjectPaths(
        project_root=project_root.resolve(),
        build_dir=build_dir,
        staging_dir=build_dir / "staging",
        serving_dir=build_dir / "serving",
        inference_runs_dir=build_dir / "inference_runs",
        reports_dir=build_dir / "reports",
        metadata_dir=build_dir / "metadata",
        tables_dir=build_dir / "tables",
        duckdb_dir=build_dir / "duckdb",
        modelled_dir=modelled_dir,
        modelled_parquet_dir=modelled_dir / "parquet",
        modelled_duckdb=modelled_dir / "duckdb" / "base_modelada_v2.duckdb",
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    for path in [
        paths.build_dir,
        paths.staging_dir,
        paths.serving_dir,
        paths.inference_runs_dir,
        paths.reports_dir,
        paths.metadata_dir,
        paths.tables_dir,
        paths.duckdb_dir,
        paths.modelled_dir,
        paths.modelled_parquet_dir,
        paths.modelled_duckdb.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def _json_default(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
