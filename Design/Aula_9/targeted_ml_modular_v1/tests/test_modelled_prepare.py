from __future__ import annotations

from pathlib import Path

import pandas as pd

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.data.modelled import is_complete_modelled_duckdb, prepare_modelled_base
from targeted_ml.orchestration.artifacts import build_project_paths, ensure_dirs


def _make_spec(dataset_root: Path, rebuild: bool = True, modeled_source: str = "auto") -> AnalysisSpec:
    return AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {
                "dataset_root": str(dataset_root),
                "modeled_source": modeled_source,
                "rebuild_modelled_if_missing": rebuild,
                "modeled_duckdb_relative_path": "data/modelled/duckdb/base_modelada_v2.duckdb",
            },
            "label": {},
        }
    )


def test_prepare_modelled_base_copies_duckdb_into_output_root_and_exports_parquets(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    source_duckdb = project_root / "data" / "modelled" / "duckdb" / "base_modelada_v2.duckdb"
    source_duckdb.parent.mkdir(parents=True, exist_ok=True)
    import duckdb

    conn = duckdb.connect(str(source_duckdb))
    conn.execute("CREATE TABLE dim_teacher AS SELECT 'u1' AS teacher_unique_id")
    conn.close()

    paths = build_project_paths(project_root=project_root)
    ensure_dirs(paths)
    spec = _make_spec(project_root / "data", rebuild=True)

    result = prepare_modelled_base(spec, paths)

    assert Path(result["modelled_duckdb"]) == paths.modelled_duckdb
    assert paths.modelled_duckdb.exists()
    assert paths.modelled_duckdb != source_duckdb
    assert (paths.modelled_parquet_dir / "dim_teacher.parquet").exists()
    assert result["modeled_source"] == "duckdb"
    assert result["reused_same_target_duckdb_source"] is False


def test_prepare_modelled_base_copies_parquets_into_output_root_and_builds_duckdb(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    parquet_dir = project_root / "data" / "modelled" / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / "dim_teacher.parquet"
    source_df = pd.DataFrame({"teacher_unique_id": ["u1"], "value": [1]})
    source_df.to_parquet(parquet_path, index=False)
    paths = build_project_paths(project_root=project_root)
    ensure_dirs(paths)
    spec = _make_spec(project_root / "data", rebuild=True)

    result = prepare_modelled_base(spec, paths)

    assert result["copied_parquets"] == 1
    assert paths.modelled_duckdb.exists()
    assert (paths.modelled_parquet_dir / "dim_teacher.parquet").exists()
    pd.testing.assert_frame_equal(pd.read_parquet(paths.modelled_parquet_dir / "dim_teacher.parquet"), source_df)


def test_prepare_modelled_base_falls_back_to_raw_rebuild_when_requested(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    dataset_root = project_root / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    paths = build_project_paths(project_root=project_root)
    ensure_dirs(paths)
    spec = _make_spec(dataset_root, rebuild=True, modeled_source="raw")

    def _fake_rebuild(spec_arg: AnalysisSpec, paths_arg):
        paths_arg.modelled_duckdb.parent.mkdir(parents=True, exist_ok=True)
        paths_arg.modelled_duckdb.write_bytes(b"rebuilt")
        return {
            "build_mode": "raw_to_modelled_rebuild",
            "modelled_duckdb": str(paths_arg.modelled_duckdb),
        }

    monkeypatch.setattr("targeted_ml.data.modelled.rebuild_modelled_from_raw", _fake_rebuild)

    result = prepare_modelled_base(spec, paths)

    assert result["build_mode"] == "raw_to_modelled_rebuild"
    assert paths.modelled_duckdb.read_bytes() == b"rebuilt"


def test_prepare_modelled_base_auto_prefers_raw_when_available(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    dataset_root = project_root / "datasets"
    raw_dir = dataset_root / "raw" / "base_aprendizap"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for file_name in [
        "dim_teachers.csv",
        "fct_teachers_entries.csv",
        "fct_teachers_contents_interactions.csv",
        "stg_lessons.csv",
        "stg_formation.csv",
        "stg_mari_ia_conversation.csv",
        "stg_mari_ia_reports.csv",
        "fct_mari_ia_eventos_isso_ajudou.csv",
        "calendario_escolar_uf_rede.csv",
    ]:
        (raw_dir / file_name).write_text("dummy", encoding="utf-8")
    source_duckdb = dataset_root / "data" / "modelled" / "duckdb" / "base_modelada_v2.duckdb"
    source_duckdb.parent.mkdir(parents=True, exist_ok=True)
    source_duckdb.write_bytes(b"seed")
    paths = build_project_paths(project_root=project_root)
    ensure_dirs(paths)
    spec = _make_spec(dataset_root, rebuild=True, modeled_source="auto")

    def _fake_rebuild(spec_arg: AnalysisSpec, paths_arg):
        paths_arg.modelled_duckdb.parent.mkdir(parents=True, exist_ok=True)
        paths_arg.modelled_duckdb.write_bytes(b"raw-first")
        return {
            "build_mode": "raw_to_modelled_rebuild",
            "modelled_duckdb": str(paths_arg.modelled_duckdb),
        }

    monkeypatch.setattr("targeted_ml.data.modelled.rebuild_modelled_from_raw", _fake_rebuild)

    result = prepare_modelled_base(spec, paths)

    assert result["build_mode"] == "raw_to_modelled_rebuild"
    assert paths.modelled_duckdb.read_bytes() == b"raw-first"


def test_is_complete_modelled_duckdb_rejects_partial_package(tmp_path: Path) -> None:
    import duckdb

    duckdb_path = tmp_path / "partial.duckdb"
    conn = duckdb.connect(str(duckdb_path))
    conn.execute("CREATE TABLE base_modelada_v2 AS SELECT 1 AS teacher_unique_id, DATE '2025-01-01' AS month")
    conn.execute("CREATE TABLE fct_teacher_month AS SELECT 1 AS teacher_unique_id, DATE '2025-01-01' AS month")
    conn.close()

    assert is_complete_modelled_duckdb(duckdb_path) is False
