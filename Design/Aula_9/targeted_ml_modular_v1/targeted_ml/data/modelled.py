from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.data.raw_to_modelled import MODELED_TABLES, rebuild_modelled_from_raw, resolve_raw_source_dir
from targeted_ml.data.sources import resolve_dataset_root
from targeted_ml.orchestration.artifacts import ProjectPaths


def _copy_tree_files(src: Path, dst: Path, pattern: str) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in sorted(src.glob(pattern)):
        target = dst / item.name
        shutil.copy2(item, target)
        count += 1
    return count


def build_duckdb_from_parquets(parquet_dir: Path, duckdb_path: Path) -> None:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    if duckdb_path.exists():
        duckdb_path.unlink()
    conn = duckdb.connect(str(duckdb_path))
    try:
        for parquet_path in sorted(parquet_dir.glob("*.parquet")):
            table_name = parquet_path.stem
            quoted = str(parquet_path).replace("'", "''")
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{quoted}')")
    finally:
        conn.close()


def build_parquets_from_duckdb(duckdb_path: Path, parquet_dir: Path) -> int:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for existing in parquet_dir.glob("*.parquet"):
        existing.unlink()
    conn = duckdb.connect(str(duckdb_path))
    try:
        table_names = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        for table_name in sorted(table_names):
            quoted = str((parquet_dir / f"{table_name}.parquet")).replace("'", "''")
            conn.execute(f"COPY {table_name} TO '{quoted}' (FORMAT PARQUET)")
    finally:
        conn.close()
    return len(list(parquet_dir.glob("*.parquet")))


def is_complete_modelled_duckdb(duckdb_path: Path) -> bool:
    if not duckdb_path.exists():
        return False
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        existing_tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if not set(MODELED_TABLES).issubset(existing_tables):
            return False
        base_rows = int(conn.execute("SELECT COUNT(*) FROM base_modelada_v2").fetchone()[0] or 0)
        fact_rows = int(conn.execute("SELECT COUNT(*) FROM fct_teacher_month").fetchone()[0] or 0)
        persona_month_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_month_persona_ready").fetchone()[0] or 0)
        distinct_teachers = int(conn.execute("SELECT COUNT(DISTINCT teacher_unique_id) FROM base_modelada_v2").fetchone()[0] or 0)
        dim_teacher_rows = int(conn.execute("SELECT COUNT(*) FROM dim_teacher").fetchone()[0] or 0)
        cluster_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_cluster_ready").fetchone()[0] or 0)
        persona_teacher_rows = int(conn.execute("SELECT COUNT(*) FROM mart_teacher_persona_ready").fetchone()[0] or 0)
    finally:
        conn.close()
    return (
        base_rows == fact_rows
        and persona_month_rows == fact_rows
        and dim_teacher_rows == distinct_teachers
        and cluster_rows == dim_teacher_rows
        and persona_teacher_rows == dim_teacher_rows
    )


def _resolve_modelled_duckdb_candidates(spec: AnalysisSpec, paths: ProjectPaths, root: Path) -> list[Path]:
    configured = spec.data.modeled_duckdb_relative_path
    candidates: list[Path] = []
    if configured.is_absolute():
        candidates.append(configured.resolve())
    else:
        candidates.append((root / configured).resolve())
        candidates.append((paths.project_root / configured).resolve())
    candidates.extend(
        [
            (root / "modelled" / "duckdb" / "base_modelada_v2.duckdb").resolve(),
            (root / "duckdb" / "base_modelada_v2.duckdb").resolve(),
        ]
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def prepare_modelled_base(spec: AnalysisSpec, paths: ProjectPaths) -> dict[str, str | int]:
    root = resolve_dataset_root(Path(spec.data.dataset_root))
    modeled_source = str(spec.data.modeled_source).strip().lower()
    raw_source_dir = resolve_raw_source_dir(spec)
    if modeled_source == "raw":
        return rebuild_modelled_from_raw(spec, paths)
    if modeled_source == "auto" and raw_source_dir is not None:
        return rebuild_modelled_from_raw(spec, paths)
    modelled_root = root / "modelled"
    parquet_src_candidates = [
        modelled_root / "parquet",
        root,
    ] if modeled_source in {"auto", "parquet"} else []
    duckdb_src_candidates = _resolve_modelled_duckdb_candidates(spec, paths, root) if modeled_source in {"auto", "duckdb"} else []
    parquet_src = next((p for p in parquet_src_candidates if p.exists() and any(p.glob("*.parquet"))), None)
    duckdb_src = next((p for p in duckdb_src_candidates if p.exists()), None)
    if parquet_src is None and duckdb_src is None:
        if modeled_source == "auto":
            return rebuild_modelled_from_raw(spec, paths)
        raise FileNotFoundError(f"Could not find modeled parquet, duckdb, or raw source under {root}")
    copied_parquets = 0
    parquet_src_same_as_target = parquet_src is not None and parquet_src.resolve() == paths.modelled_parquet_dir.resolve()
    if parquet_src is not None and not parquet_src_same_as_target:
        if paths.modelled_parquet_dir.exists():
            shutil.rmtree(paths.modelled_parquet_dir)
        copied_parquets = _copy_tree_files(parquet_src, paths.modelled_parquet_dir, "*.parquet")
    elif parquet_src_same_as_target:
        copied_parquets = len(list(paths.modelled_parquet_dir.glob("*.parquet")))
    duckdb_src_same_as_target = duckdb_src is not None and duckdb_src.resolve() == paths.modelled_duckdb.resolve()
    resolved_modeled_source = modeled_source
    if duckdb_src is not None and not duckdb_src_same_as_target:
        paths.modelled_duckdb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(duckdb_src, paths.modelled_duckdb)
        resolved_modeled_source = "duckdb"
    elif duckdb_src_same_as_target:
        paths.modelled_duckdb.parent.mkdir(parents=True, exist_ok=True)
        resolved_modeled_source = "duckdb"
    elif copied_parquets > 0:
        build_duckdb_from_parquets(paths.modelled_parquet_dir, paths.modelled_duckdb)
        resolved_modeled_source = "parquet"
    elif parquet_src_same_as_target:
        resolved_modeled_source = "parquet"
    if paths.modelled_duckdb.exists() and not any(paths.modelled_parquet_dir.glob("*.parquet")):
        copied_parquets = build_parquets_from_duckdb(paths.modelled_duckdb, paths.modelled_parquet_dir)
    return {
        "dataset_root": str(root),
        "modeled_source": resolved_modeled_source,
        "copied_parquets": copied_parquets,
        "modelled_duckdb": str(paths.modelled_duckdb),
        "modelled_parquet_dir": str(paths.modelled_parquet_dir),
        "reused_same_target_parquet_source": parquet_src_same_as_target,
        "reused_same_target_duckdb_source": duckdb_src_same_as_target,
    }
