"""Persistência, caminhos e staging incremental do pipeline modelled -> ml."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd


@dataclass(frozen=True)
class EnginePaths:
    project_root: Path
    modelled_duckdb: Path
    output_dir: Path
    output_duckdb: Path
    compute_post_model_refit: bool


TASK_TABLES = {
    "fold_metrics": "fold_metrics.parquet",
    "predictions": "predictions.parquet",
    "inner_audit": "inner_audit.parquet",
    "importance": "importance.parquet",
    "post_model_output_status": "post_model_output_status.parquet",
}


@dataclass(frozen=True)
class ModelTaskKey:
    problem_key: str
    model_name: str
    task_scope: str = "core"

    @property
    def task_id(self) -> str:
        return f"{self.task_scope}__{self.problem_key}__{self.model_name}".replace("/", "_")


class TaskArtifactStore:
    def __init__(self, staging_dir: Path) -> None:
        self.root = staging_dir / "model_tasks"
        self.root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, key: ModelTaskKey) -> Path:
        return self.root / key.task_id

    def manifest_path(self, key: ModelTaskKey) -> Path:
        return self.task_dir(key) / "task_manifest.json"

    def is_completed(self, key: ModelTaskKey, expected_signature: str | None = None) -> bool:
        path = self.manifest_path(key)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if payload.get("status") != "completed":
            return False
        if expected_signature is not None and payload.get("task_signature") != expected_signature:
            return False
        return True

    def write_running(self, key: ModelTaskKey, metadata: dict[str, Any] | None = None) -> None:
        payload = {"status": "running", "task_id": key.task_id}
        if metadata:
            payload.update(metadata)
        self._write_manifest_atomic(key, payload)

    def write_failed(self, key: ModelTaskKey, error_payload: dict[str, Any]) -> None:
        payload = {"status": "failed", "task_id": key.task_id}
        payload.update(error_payload)
        self._write_manifest_atomic(key, payload)

    def write_completed(self, key: ModelTaskKey, tables: dict[str, pd.DataFrame], metadata: dict[str, Any] | None = None) -> None:
        task_dir = self.task_dir(key)
        task_dir.mkdir(parents=True, exist_ok=True)
        for table_name, file_name in TASK_TABLES.items():
            self._write_parquet_atomic(task_dir / file_name, tables.get(table_name, pd.DataFrame()))
        payload = {"status": "completed", "task_id": key.task_id}
        if metadata:
            payload.update(metadata)
        self._write_manifest_atomic(key, payload)

    def load_completed(self, key: ModelTaskKey) -> dict[str, pd.DataFrame]:
        task_dir = self.task_dir(key)
        return {
            table_name: pd.read_parquet(task_dir / file_name)
            for table_name, file_name in TASK_TABLES.items()
            if (task_dir / file_name).exists()
        }

    def clear_task(self, key: ModelTaskKey) -> None:
        task_dir = self.task_dir(key)
        if task_dir.exists():
            shutil.rmtree(task_dir)

    def _write_manifest_atomic(self, key: ModelTaskKey, payload: dict[str, Any]) -> None:
        path = self.manifest_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        tmp_path.replace(path)

    def _write_parquet_atomic(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix('.tmp.parquet')
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)

def ensure_output_dirs(output_dir: Path) -> None:
    for name in ("tables", "metadata", "reports", "duckdb"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

def clear_previous_outputs(output_dir: Path) -> None:
    for subdir in ("tables", "metadata", "reports", "parquet", "json"):
        path = output_dir / subdir
        if not path.exists():
            continue
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
    for duckdb_name in ("duckdb", "local_runtime_duckdb"):
        duckdb_dir = output_dir / duckdb_name
        if duckdb_dir.exists():
            for child in duckdb_dir.iterdir():
                if child.is_file():
                    child.unlink()
    for staging_name in ("staging", "local_runtime_staging"):
        staging_dir = output_dir / staging_name
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

def attach_modelled_views(conn: duckdb.DuckDBPyConnection, modelled_duckdb: Path, table_names: Iterable[str]) -> None:
    duckdb_sql = str(modelled_duckdb).replace("'", "''")
    conn.execute(f"ATTACH '{duckdb_sql}' AS modelled_base (READ_ONLY)")
    existing = {row[0] for row in conn.execute("SHOW TABLES FROM modelled_base").fetchall()}
    missing = [name for name in table_names if name not in existing]
    if missing:
        raise RuntimeError("Missing required tables in modelled base: " + ", ".join(sorted(missing)))
    for table_name in table_names:
        conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM modelled_base.{table_name}")

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")

def persist_table(conn: duckdb.DuckDBPyConnection, output_dir: Path, table_name: str, df: pd.DataFrame) -> None:
    conn.register("_tmp_df", df)
    try:
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
        parquet_path = output_dir / "tables" / f"{table_name}.parquet"
        parquet_sql = str(parquet_path).replace("'", "''")
        conn.execute(f"COPY {table_name} TO '{parquet_sql}' (FORMAT PARQUET)")
    finally:
        conn.unregister("_tmp_df")
