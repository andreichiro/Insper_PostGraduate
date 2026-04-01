from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.data.raw_to_modelled import REQUIRED_RAW_FILES
from targeted_ml.modeling.calibration import build_temporal_calibrator
from targeted_ml.orchestration.artifacts import ProjectPaths, write_json
from targeted_ml.orchestration.manifests import git_revision, hash_spec
from targeted_ml.pipelines.modelled_to_ml import analysis_setup as setup
from targeted_ml.pipelines.modelled_to_ml.analysis_setup import RuntimeBuildConfig
from targeted_ml.pipelines.modelled_to_ml.dataset_builder import (
    build_first_session_journey_mart,
    build_official_frame,
    build_onboarding_mart,
    select_active_features,
)
from targeted_ml.pipelines.modelled_to_ml.modeling import (
    build_model_specs,
    build_preprocessor,
    build_temporal_calibration_holdout,
    probability_metrics,
    tune_temporal_estimator,
)
from targeted_ml.pipelines.modelled_to_ml.selection import (
    candidate_definition_group,
    select_serving_scope,
)
from targeted_ml.pipelines.modelled_to_ml.storage import attach_modelled_views


def _artifact_id(problem_key: str, model_name: str) -> str:
    digest = hashlib.sha256(f"{problem_key}::{model_name}".encode("utf-8")).hexdigest()[:16]
    return f"{model_name}__{digest}"


def _read_build_table(paths: ProjectPaths, table_name: str) -> pd.DataFrame:
    path = paths.tables_dir / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required build table not found: {path}")
    return pd.read_parquet(path)


def _load_build_context(spec: AnalysisSpec, paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    runtime_config = RuntimeBuildConfig.from_analysis_spec(spec)
    setup.apply_runtime_config(runtime_config)
    journey = _read_build_table(paths, "mart_first_session_journey_v1")
    future_metrics = _read_build_table(paths, "mart_future_metrics_v1")
    definition_frontier = _read_build_table(paths, "core_definition_frontier_v1")
    selection_df = _read_build_table(paths, "core_definition_selection_v1")
    feature_registry = _read_build_table(paths, "governance_feature_registry_v1")
    scoring_scenarios = _read_build_table(paths, "core_scoring_scenarios_v1")
    model_frontier = _read_build_table(paths, "core_model_frontier_v1")
    model_predictions = _read_build_table(paths, "core_model_predictions_v1")
    frame = build_official_frame(journey, future_metrics, selection_df)
    return {
        "journey": journey,
        "future_metrics": future_metrics,
        "definition_frontier": definition_frontier,
        "selection_df": selection_df,
        "feature_registry": feature_registry,
        "scoring_scenarios": scoring_scenarios,
        "model_frontier": model_frontier,
        "model_predictions": model_predictions,
        "frame": frame,
    }


def _build_model_input(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    cols = [name for name in feature_names if name in frame.columns]
    if "first_month" in frame.columns:
        cols = cols + ["first_month"]
    return frame[cols].copy()


def _fit_single_reference_model(
    frame: pd.DataFrame,
    feature_registry: pd.DataFrame,
    scenario: dict[str, Any],
    model_name: str,
    spec_hash_value: str,
    git_revision_value: str,
    source_build_dir: str,
    source_build_summary: str,
    export_id: str,
) -> tuple[Any, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    working = frame.dropna(subset=["first_month"]).copy()
    working["y_true"] = pd.to_numeric(working[scenario["label_col"]], errors="coerce")
    working = working[working["y_true"].notna()].copy()
    working["y_true"] = working["y_true"].astype(int)
    if working.empty:
        raise ValueError(f"Training frame is empty for {scenario['problem_key']} | {model_name}")
    fit_idx, calibration_idx, calibration_audit = build_temporal_calibration_holdout(
        working,
        month_col="first_month",
        target_col="y_true",
    )
    if fit_idx is None or calibration_idx is None:
        raise ValueError(f"No valid temporal calibration holdout for {scenario['problem_key']} | {model_name}")
    fit_train = working.iloc[fit_idx].copy()
    calibration_holdout = working.iloc[calibration_idx].copy()
    requested_feature_names = json.loads(scenario["feature_names_json"])
    active_feature_names = select_active_features(
        fit_train=fit_train,
        feature_names=requested_feature_names,
        calibration_holdout=calibration_holdout,
    )
    if not active_feature_names:
        raise ValueError(f"No active features available for {scenario['problem_key']} | {model_name}")
    model_spec = next((row for row in build_model_specs() if row["model_name"] == model_name), None)
    if model_spec is None:
        raise ValueError(f"Model spec not found for {model_name}")
    preprocessor = build_preprocessor(feature_registry, active_feature_names)
    estimator = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_spec["estimator"]),
        ]
    )
    tuning_train = fit_train[active_feature_names + ["first_month", "y_true"]].copy()
    train_for_calibration = working[active_feature_names + ["first_month", "y_true"]].copy()
    tuned_estimator, tuning_audit, tuning_meta = tune_temporal_estimator(
        estimator=estimator,
        model_spec=model_spec,
        tuning_train=tuning_train,
        feature_names=active_feature_names,
        target_col="y_true",
    )
    calibrated = build_temporal_calibrator(
        estimator=tuned_estimator,
        train=train_for_calibration,
        target_col="y_true",
        fit_idx=fit_idx,
        calibration_idx=calibration_idx,
        method=setup.CALIBRATION_METHOD,
    )
    in_sample_score = calibrated.predict_proba(_build_model_input(working, active_feature_names))[:, 1]
    in_sample_metrics = probability_metrics(working["y_true"].to_numpy(dtype=int), in_sample_score)
    metadata = {
        "problem_key": scenario["problem_key"],
        "definition_name": scenario["definition_name"],
        "track_name": scenario["track_name"],
        "label_col": scenario["label_col"],
        "model_name": model_name,
        "score_window_end_day": int(scenario["score_window_end_day"]),
        "requested_feature_count": int(len(requested_feature_names)),
        "active_feature_count": int(len(active_feature_names)),
        "active_feature_names": active_feature_names,
        "training_rows": int(len(working)),
        "training_positives": int(working["y_true"].sum()),
        "training_negatives": int(len(working) - int(working["y_true"].sum())),
        "fit_rows": int(len(fit_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "tuning_meta": tuning_meta,
        "in_sample_metrics": in_sample_metrics,
        "spec_hash": spec_hash_value,
        "git_revision": git_revision_value,
        "source_build_dir": source_build_dir,
        "source_build_summary": source_build_summary,
        "export_id": export_id,
    }
    return calibrated, metadata, calibration_audit, tuning_audit


def _build_feature_schema_records(feature_registry: pd.DataFrame, feature_names: list[str]) -> list[dict[str, Any]]:
    registry = feature_registry.set_index("feature_name", drop=False)
    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        row = registry.loc[feature_name]
        rows.append(
            {
                "feature_name": feature_name,
                "feature_type": str(row["feature_type"]),
                "feature_class": str(row["feature_class"]),
                "behavior_family": str(row["behavior_family"]),
                "source_table": str(row["source_table"]),
                "source_columns": json.loads(row["source_columns_json"]),
            }
        )
    return rows


def _build_model_schema(metadata: dict[str, Any], feature_registry: pd.DataFrame) -> dict[str, Any]:
    feature_names = list(metadata["active_feature_names"])
    numeric_features = [
        row["feature_name"]
        for row in _build_feature_schema_records(feature_registry, feature_names)
        if row["feature_type"] == "numeric"
    ]
    categorical_features = [
        row["feature_name"]
        for row in _build_feature_schema_records(feature_registry, feature_names)
        if row["feature_type"] == "categorical"
    ]
    return {
        "input_kind": "modelled_duckdb_or_journey_frame",
        "required_modelled_tables": list(setup.MODELLED_TABLES),
        "required_scoring_columns": ["teacher_unique_id", "first_month", "onboarding_anchor_ts"] + feature_names,
        "required_feature_columns": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "score_window_end_day": int(metadata["score_window_end_day"]),
        "eligibility_rule": f"onboarding_anchor_ts + {int(metadata['score_window_end_day'])} dias <= latest_observed_ts",
        "positive_class_meaning": "probabilidade calibrada de pertencer ao label positivo do problema exportado",
        "risk_score_meaning": "1 - probabilidade calibrada do label positivo",
    }


def _next_available_dir(parent: Path, slug: str) -> Path:
    candidate = parent / slug
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = parent / f"{slug}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def _build_serving_contract(spec: AnalysisSpec, reference_scope_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    required_keys = ["teacher_unique_id", "first_month", "onboarding_anchor_ts"]
    required_modelled_tables: set[str] = set()
    union_features: set[str] = set()
    union_numeric: set[str] = set()
    union_categorical: set[str] = set()
    model_rows: list[dict[str, Any]] = []
    for row in reference_scope_rows:
        schema = json.loads(Path(row["schema_path"]).read_text(encoding="utf-8"))
        required_modelled_tables.update(schema.get("required_modelled_tables", []))
        union_features.update(schema.get("required_feature_columns", []))
        union_numeric.update(schema.get("numeric_features", []))
        union_categorical.update(schema.get("categorical_features", []))
        model_rows.append(
            {
                "artifact_id": row["artifact_id"],
                "problem_key": row["problem_key"],
                "definition_name": row["definition_name"],
                "track_name": row["track_name"],
                "model_name": row["model_name"],
                "schema_path": row["schema_path"],
                "feature_list_path": row["feature_path"],
            }
        )
    scoring_frame_columns = required_keys + sorted(union_features)
    contract = {
        "supported_input_kinds": ["modelled_duckdb", "scoring_frame_file", "raw_dataset_root"],
        "supported_scoring_frame_formats": ["csv", "parquet"],
        "required_modelled_tables": sorted(required_modelled_tables),
        "raw_dataset_root_contract": {
            "supported": True,
            "required_relative_path": str(spec.data.raw_relative_path),
            "required_files": list(REQUIRED_RAW_FILES),
            "behavior": "Reconstrói modeled localmente em staging e depois aplica o modelo salvo.",
        },
        "required_key_columns": required_keys,
        "required_key_column_rules": {
            "teacher_unique_id": "Obrigatório, não vazio e estável por professor.",
            "first_month": "Obrigatório e parseável como data; representa o primeiro mês observado do professor.",
            "onboarding_anchor_ts": "Obrigatório e parseável como timestamp; usado para calcular elegibilidade temporal quando score_window_ready_flag não vier pronto.",
        },
        "required_scoring_columns_union": scoring_frame_columns,
        "required_feature_columns_union": sorted(union_features),
        "numeric_features_union": sorted(union_numeric),
        "categorical_features_union": sorted(union_categorical),
        "optional_scoring_columns": ["score_window_ready_flag"],
        "latest_observed_ts_rule": "Obrigatório no score de scoring_frame quando score_window_ready_flag nao vier no arquivo.",
        "model_contracts": model_rows,
    }
    template_df = pd.DataFrame(columns=scoring_frame_columns + ["score_window_ready_flag"])
    return contract, template_df


def _sync_latest_export_to_root(export_dir: Path, serving_dir: Path) -> None:
    root_models_dir = serving_dir / "models"
    export_models_dir = export_dir / "models"
    root_models_dir.mkdir(parents=True, exist_ok=True)
    for existing in root_models_dir.glob("*"):
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)
    for artifact in export_models_dir.glob("*"):
        target = root_models_dir / artifact.name
        shutil.copy2(artifact, target)


def export_reference_models(
    spec: AnalysisSpec,
    paths: ProjectPaths,
    problem_keys: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
) -> Path:
    context = _load_build_context(spec, paths)
    selected_scope, selection_candidates, selection_meta = select_serving_scope(
        model_frontier=context["model_frontier"],
        model_predictions=context["model_predictions"],
        definition_selection=context["selection_df"],
        definition_frontier=context["definition_frontier"],
        scoring_scenarios=context["scoring_scenarios"],
        problem_keys=problem_keys,
        model_names=model_names,
    )
    scoring_scenarios = context["scoring_scenarios"]
    feature_registry = context["feature_registry"]
    model_frontier = context["model_frontier"]
    frame = context["frame"]
    spec_hash_value = hash_spec(spec)
    git_revision_value = git_revision(paths.project_root)
    source_build_dir = str(paths.build_dir)
    source_build_summary = str(paths.metadata_dir / "build_summary_v1.json")
    exported_at = datetime.now().isoformat(timespec="seconds")
    export_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = _next_available_dir(paths.serving_dir / "exports", export_slug)
    models_dir = export_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    exported_rows: list[dict[str, Any]] = []
    print(f"[serving] export_id={export_dir.name} | candidates={len(selection_candidates)}", flush=True)
    selection_candidates.to_parquet(export_dir / "serving_selection_candidates.parquet", index=False)
    for ref_row in selected_scope.to_dict(orient="records"):
        print(
            f"[serving] fitting problem={ref_row['problem_key']} | model={ref_row['model_name']}",
            flush=True,
        )
        scenario_match = scoring_scenarios[scoring_scenarios["problem_key"] == ref_row["problem_key"]]
        if scenario_match.empty:
            raise ValueError(f"Scoring scenario not found for {ref_row['problem_key']}")
        scenario = scenario_match.iloc[0].to_dict()
        calibrated, metadata, calibration_audit, tuning_audit = _fit_single_reference_model(
            frame=frame,
            feature_registry=feature_registry,
            scenario=scenario,
            model_name=str(ref_row["model_name"]),
            spec_hash_value=spec_hash_value,
            git_revision_value=git_revision_value,
            source_build_dir=source_build_dir,
            source_build_summary=source_build_summary,
            export_id=export_dir.name,
        )
        artifact_id = _artifact_id(str(ref_row["problem_key"]), str(ref_row["model_name"]))
        model_path = models_dir / f"{artifact_id}.joblib"
        feature_path = models_dir / f"{artifact_id}.feature_list.json"
        schema_path = models_dir / f"{artifact_id}.schema.json"
        manifest_path = models_dir / f"{artifact_id}.manifest.json"
        calibration_audit_path = models_dir / f"{artifact_id}.calibration_audit.parquet"
        tuning_audit_path = models_dir / f"{artifact_id}.tuning_audit.parquet"
        frontier_match = model_frontier[
            (model_frontier["problem_key"] == ref_row["problem_key"])
            & (model_frontier["model_name"] == ref_row["model_name"])
        ]
        frontier_metrics = frontier_match.iloc[0].to_dict() if not frontier_match.empty else {}
        bundle = {
            "predictor": calibrated,
            "problem_key": metadata["problem_key"],
            "model_name": metadata["model_name"],
            "active_feature_names": metadata["active_feature_names"],
            "score_window_end_day": metadata["score_window_end_day"],
        }
        joblib.dump(bundle, model_path)
        write_json(feature_path, {"active_feature_names": metadata["active_feature_names"]})
        write_json(schema_path, _build_model_schema(metadata, feature_registry))
        model_manifest = {
            "export_id": export_dir.name,
            "artifact_id": artifact_id,
            "problem_key": metadata["problem_key"],
            "definition_name": metadata["definition_name"],
            "definition_group": candidate_definition_group(metadata["problem_key"], metadata["definition_name"]),
            "track_name": metadata["track_name"],
            "label_col": metadata["label_col"],
            "model_name": metadata["model_name"],
            "selection_reason": str(ref_row["selection_reason"]),
            "serving_rank": int(ref_row.get("serving_rank", 1)),
            "score_window_end_day": metadata["score_window_end_day"],
            "feature_schema": _build_feature_schema_records(feature_registry, metadata["active_feature_names"]),
            "training_rows": metadata["training_rows"],
            "training_positives": metadata["training_positives"],
            "training_negatives": metadata["training_negatives"],
            "fit_rows": metadata["fit_rows"],
            "calibration_rows": metadata["calibration_rows"],
            "tuning_meta": metadata["tuning_meta"],
            "in_sample_metrics": metadata["in_sample_metrics"],
            "spec_hash": metadata["spec_hash"],
            "git_revision": metadata["git_revision"],
            "source_build_dir": metadata["source_build_dir"],
            "source_build_summary": metadata["source_build_summary"],
            "frontier_metrics": frontier_metrics,
            "selection_diagnostics": {
                "mean_brier": ref_row.get("mean_brier"),
                "mean_log_loss": ref_row.get("mean_log_loss"),
                "mean_calibration_slope_error": ref_row.get("mean_calibration_slope_error"),
                "mean_calibration_intercept_abs": ref_row.get("mean_calibration_intercept_abs"),
                "mean_ap": ref_row.get("mean_ap"),
                "mean_roc_auc": ref_row.get("mean_roc_auc"),
                "max_probability_metric_std": ref_row.get("max_probability_metric_std"),
                "max_operational_metric_std": ref_row.get("max_operational_metric_std"),
                "max_operational_metric_jump": ref_row.get("max_operational_metric_jump"),
                "max_confusion_share_std": ref_row.get("max_confusion_share_std"),
                "max_confusion_share_jump": ref_row.get("max_confusion_share_jump"),
            },
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "schema_path": str(schema_path),
            "calibration_audit_path": str(calibration_audit_path),
            "tuning_audit_path": str(tuning_audit_path),
            "exported_at": exported_at,
        }
        write_json(manifest_path, model_manifest)
        calibration_audit.to_parquet(calibration_audit_path, index=False)
        tuning_audit.to_parquet(tuning_audit_path, index=False)
        exported_rows.append(model_manifest)
        print(f"[serving] saved artifact_id={artifact_id}", flush=True)

    inference_contract, scoring_template = _build_serving_contract(spec, exported_rows)
    write_json(export_dir / "inference_contract.json", inference_contract)
    scoring_template.to_csv(export_dir / "scoring_frame_template.csv", index=False)
    top_manifest = {
        "analysis_name": spec.analysis_name,
        "analysis_kind": spec.analysis_kind,
        "spec_hash": spec_hash_value,
        "git_revision": git_revision_value,
        "source_build_dir": source_build_dir,
        "source_build_summary": source_build_summary,
        "export_id": export_dir.name,
        "export_dir": str(export_dir),
        "serving_status": "unique_primary_model" if len(exported_rows) == 1 else "filtered_multi_model_export",
        "primary_model_artifact_id": exported_rows[0]["artifact_id"] if len(exported_rows) == 1 else None,
        "primary_model_manifest_path": str(export_dir / "models" / f"{exported_rows[0]['artifact_id']}.manifest.json") if len(exported_rows) == 1 else None,
        "exported_model_count": len(exported_rows),
        "exported_at": exported_at,
        "selection_meta": selection_meta,
        "inference_contract_path": str(export_dir / "inference_contract.json"),
        "reference_scope_rows": exported_rows,
    }
    export_manifest_path = export_dir / "serving_manifest.json"
    write_json(export_manifest_path, top_manifest)
    write_json(export_dir / "serving_scope.json", {"serving_scope": exported_rows, "selection_meta": selection_meta})
    write_json(export_dir / "reference_scope.json", {"reference_scope": exported_rows})
    _sync_latest_export_to_root(export_dir, paths.serving_dir)

    manifest_path = paths.serving_dir / "serving_manifest.json"
    write_json(manifest_path, top_manifest)
    write_json(paths.serving_dir / "serving_scope.json", {"serving_scope": exported_rows, "selection_meta": selection_meta})
    write_json(paths.serving_dir / "reference_scope.json", {"reference_scope": exported_rows})
    write_json(paths.serving_dir / "inference_contract.json", inference_contract)
    scoring_template.to_csv(paths.serving_dir / "scoring_frame_template.csv", index=False)
    selection_candidates.to_parquet(paths.serving_dir / "serving_selection_candidates.parquet", index=False)
    write_json(
        paths.serving_dir / "latest.json",
        {
            "latest_export_dir": str(export_dir),
            "latest_serving_manifest": str(export_manifest_path),
        },
    )
    print(f"[serving] manifest={manifest_path}", flush=True)
    return manifest_path


def load_reference_models(paths: ProjectPaths) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = paths.serving_dir / "serving_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Serving manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundles: list[dict[str, Any]] = []
    for row in manifest.get("reference_scope_rows", []):
        model_payload = joblib.load(row["model_path"])
        bundles.append(
            {
                "manifest": row,
                "predictor": model_payload["predictor"],
                "active_feature_names": list(model_payload["active_feature_names"]),
                "score_window_end_day": int(model_payload["score_window_end_day"]),
            }
        )
    return manifest, bundles


def _compute_latest_observed_ts(conn: duckdb.DuckDBPyConnection) -> pd.Timestamp:
    query = """
    WITH latest AS (
      SELECT MAX(session_start_ts) AS ts FROM fct_session_clean
      UNION ALL
      SELECT MAX(interaction_ts) AS ts FROM fct_interaction_clean
      UNION ALL
      SELECT MAX(formation_ts) AS ts FROM fct_formation_clean
      UNION ALL
      SELECT MAX(mari_created_ts) AS ts FROM fct_mari_conversation_resolved
      UNION ALL
      SELECT MAX(help_ts) AS ts FROM fct_mari_help_resolved
      UNION ALL
      SELECT MAX(report_ts) AS ts FROM fct_mari_reports_resolved
    )
    SELECT MAX(ts) AS latest_observed_ts
    FROM latest
    """
    value = conn.execute(query).fetchone()[0]
    return pd.to_datetime(value, errors="coerce")


def _build_scoring_frame_from_modelled(modelled_duckdb: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    conn = duckdb.connect()
    try:
        attach_modelled_views(conn, modelled_duckdb.resolve(), setup.MODELLED_TABLES)
        onboarding = build_onboarding_mart(conn)
        conn.register("_tmp_onboarding", onboarding)
        conn.execute("CREATE OR REPLACE TABLE mart_onboarding_population_v1 AS SELECT * FROM _tmp_onboarding")
        conn.unregister("_tmp_onboarding")
        journey = build_first_session_journey_mart(conn)
        latest_observed_ts = _compute_latest_observed_ts(conn)
    finally:
        conn.close()
    return journey, latest_observed_ts


def _count_datetime_parse_issues(series: pd.Series) -> int:
    parsed = pd.to_datetime(series, errors="coerce")
    return int(parsed.isna().sum() - series.isna().sum())


def _count_blank_key_issues(series: pd.Series) -> int:
    values = series.astype("string")
    return int(values.isna().sum() + values.str.strip().eq("").fillna(False).sum())


def validate_inference_input_schema(scoring_frame: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle["manifest"]
    required_features = list(bundle["active_feature_names"])
    missing_required = [name for name in required_features if name not in scoring_frame.columns]
    feature_schema = manifest.get("feature_schema", [])
    numeric_features = [row["feature_name"] for row in feature_schema if row["feature_type"] == "numeric"]
    categorical_features = [row["feature_name"] for row in feature_schema if row["feature_type"] == "categorical"]
    numeric_cast_issues: dict[str, int] = {}
    for feature_name in numeric_features:
        if feature_name not in scoring_frame.columns:
            continue
        coerced = pd.to_numeric(scoring_frame[feature_name], errors="coerce")
        numeric_cast_issues[feature_name] = int(coerced.isna().sum() - scoring_frame[feature_name].isna().sum())
    required_keys = ["teacher_unique_id", "first_month", "onboarding_anchor_ts"]
    missing_keys = [name for name in required_keys if name not in scoring_frame.columns]
    key_column_issues: dict[str, int] = {}
    if "teacher_unique_id" in scoring_frame.columns:
        key_column_issues["teacher_unique_id_blank_or_missing"] = _count_blank_key_issues(scoring_frame["teacher_unique_id"])
    if "first_month" in scoring_frame.columns:
        key_column_issues["first_month_parse_issues"] = _count_datetime_parse_issues(scoring_frame["first_month"])
    if "onboarding_anchor_ts" in scoring_frame.columns:
        key_column_issues["onboarding_anchor_ts_parse_issues"] = _count_datetime_parse_issues(scoring_frame["onboarding_anchor_ts"])
    total_key_issues = int(sum(key_column_issues.values()))
    return {
        "problem_key": manifest["problem_key"],
        "model_name": manifest["model_name"],
        "required_feature_count": len(required_features),
        "required_feature_columns": required_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_required_features": missing_required,
        "missing_key_columns": missing_keys,
        "key_column_issues": key_column_issues,
        "numeric_cast_issues": numeric_cast_issues,
        "valid_input_flag": int(not missing_required and not missing_keys and total_key_issues == 0),
    }


def _score_bundle_on_frame(
    scoring_frame: pd.DataFrame,
    latest_observed_ts: pd.Timestamp | None,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = validate_inference_input_schema(scoring_frame, bundle)
    if not validation["valid_input_flag"]:
        raise ValueError(
            f"Invalid input for {bundle['manifest']['problem_key']} | {bundle['manifest']['model_name']}: "
            f"missing={validation['missing_required_features'] or validation['missing_key_columns']} "
            f"key_issues={validation.get('key_column_issues', {})}"
        )
    manifest = bundle["manifest"]
    work = scoring_frame.copy()
    score_window_end_day = int(bundle["score_window_end_day"])
    if "score_window_ready_flag" in work.columns:
        work["score_window_ready_flag"] = (
            pd.to_numeric(work["score_window_ready_flag"], errors="coerce").fillna(0).clip(lower=0, upper=1).astype(int)
        )
    else:
        if latest_observed_ts is None:
            raise ValueError(
                "latest_observed_ts is required when score_window_ready_flag is not provided in the scoring frame."
            )
        work["latest_observed_ts"] = latest_observed_ts
        work["score_window_ready_flag"] = (
            pd.to_datetime(work["onboarding_anchor_ts"], errors="coerce")
            + pd.to_timedelta(score_window_end_day, unit="D")
            <= latest_observed_ts
        ).astype(int)
    eligible = work[work["score_window_ready_flag"] == 1].copy()
    if eligible.empty:
        scored = work[["teacher_unique_id", "first_month", "onboarding_anchor_ts", "score_window_ready_flag"]].copy()
        scored["problem_key"] = manifest["problem_key"]
        scored["definition_name"] = manifest["definition_name"]
        scored["track_name"] = manifest["track_name"]
        scored["model_name"] = manifest["model_name"]
        scored["score_positive"] = np.nan
        scored["risk_score"] = np.nan
        scored["eligibility_reason"] = "insufficient_observation_window"
        scored["risk_rank"] = pd.Series([pd.NA] * len(scored), dtype="Int64")
        return scored, validation
    X = _build_model_input(eligible, list(bundle["active_feature_names"]))
    eligible["score_positive"] = bundle["predictor"].predict_proba(X)[:, 1]
    eligible["risk_score"] = 1.0 - eligible["score_positive"]
    eligible["eligibility_reason"] = ""
    eligible["problem_key"] = manifest["problem_key"]
    eligible["definition_name"] = manifest["definition_name"]
    eligible["track_name"] = manifest["track_name"]
    eligible["model_name"] = manifest["model_name"]
    keep_cols = [
        "teacher_unique_id",
        "first_month",
        "onboarding_anchor_ts",
        "problem_key",
        "definition_name",
        "track_name",
        "model_name",
        "score_window_ready_flag",
        "score_positive",
        "risk_score",
        "eligibility_reason",
    ]
    ineligible = work[work["score_window_ready_flag"] == 0].copy()
    if not ineligible.empty:
        ineligible["problem_key"] = manifest["problem_key"]
        ineligible["definition_name"] = manifest["definition_name"]
        ineligible["track_name"] = manifest["track_name"]
        ineligible["model_name"] = manifest["model_name"]
        ineligible["score_positive"] = np.nan
        ineligible["risk_score"] = np.nan
        ineligible["eligibility_reason"] = "insufficient_observation_window"
    frames = []
    if not eligible.empty:
        frames.append(eligible[keep_cols])
    if not ineligible.empty:
        frames.append(ineligible[keep_cols])
    scored = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keep_cols)
    scored["risk_rank"] = (
        scored["risk_score"]
        .astype("float64")
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    return scored.sort_values(["problem_key", "risk_score"], ascending=[True, False], na_position="last"), validation


def score_modelled_duckdb(
    paths: ProjectPaths,
    modelled_duckdb: Path,
    problem_keys: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
    run_name: str | None = None,
) -> Path:
    manifest, bundles = load_reference_models(paths)
    filtered_bundles = [
        bundle
        for bundle in bundles
        if (not problem_keys or bundle["manifest"]["problem_key"] in set(problem_keys))
        and (not model_names or bundle["manifest"]["model_name"] in set(model_names))
    ]
    if not filtered_bundles:
        raise ValueError("No exported serving models matched the requested filters.")
    scoring_frame, latest_observed_ts = _build_scoring_frame_from_modelled(modelled_duckdb)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = run_name.strip().replace(" ", "_") if run_name else run_id
    run_dir = _next_available_dir(paths.inference_runs_dir, run_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inference] mode=modelled_duckdb | run_dir={run_dir}", flush=True)
    all_scores: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    for bundle in filtered_bundles:
        print(
            f"[inference] scoring problem={bundle['manifest']['problem_key']} | model={bundle['manifest']['model_name']}",
            flush=True,
        )
        scored, validation = _score_bundle_on_frame(scoring_frame, latest_observed_ts, bundle)
        all_scores.append(scored)
        validation_rows.append(validation)
    scored_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    validation_df = pd.DataFrame(validation_rows)
    high_risk = scored_df[scored_df["risk_score"].notna()].sort_values(
        ["problem_key", "risk_score", "teacher_unique_id"],
        ascending=[True, False, True],
    )
    scored_df.to_parquet(run_dir / "scores_all_models.parquet", index=False)
    high_risk.to_parquet(run_dir / "high_risk_users.parquet", index=False)
    validation_df.to_parquet(run_dir / "validation_report.parquet", index=False)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "run_name": run_name or "",
            "modelled_duckdb": str(modelled_duckdb.resolve()),
            "latest_observed_ts": str(latest_observed_ts),
            "serving_status": manifest["serving_status"],
            "scored_model_count": len(filtered_bundles),
            "scored_rows": int(len(scored_df)),
            "eligible_rows": int(pd.to_numeric(scored_df["score_window_ready_flag"], errors="coerce").fillna(0).sum()) if not scored_df.empty else 0,
            "artifacts": {
                "scores_all_models": str(run_dir / "scores_all_models.parquet"),
                "high_risk_users": str(run_dir / "high_risk_users.parquet"),
                "validation_report": str(run_dir / "validation_report.parquet"),
            },
        },
    )
    write_json(paths.inference_runs_dir / "latest.json", {"latest_run_dir": str(run_dir)})
    print(f"[inference] completed run_dir={run_dir}", flush=True)
    return run_dir


def _load_scoring_frame(scoring_frame_path: Path) -> pd.DataFrame:
    suffix = scoring_frame_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(scoring_frame_path)
    if suffix == ".csv":
        return pd.read_csv(scoring_frame_path)
    raise ValueError(f"Unsupported scoring frame format: {scoring_frame_path}")


def score_scoring_frame(
    paths: ProjectPaths,
    scoring_frame_path: Path,
    latest_observed_ts: pd.Timestamp | None = None,
    problem_keys: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
    run_name: str | None = None,
) -> Path:
    manifest, bundles = load_reference_models(paths)
    filtered_bundles = [
        bundle
        for bundle in bundles
        if (not problem_keys or bundle["manifest"]["problem_key"] in set(problem_keys))
        and (not model_names or bundle["manifest"]["model_name"] in set(model_names))
    ]
    if not filtered_bundles:
        raise ValueError("No exported serving models matched the requested filters.")
    scoring_frame = _load_scoring_frame(scoring_frame_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = run_name.strip().replace(" ", "_") if run_name else run_id
    run_dir = _next_available_dir(paths.inference_runs_dir, run_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inference] mode=scoring_frame_file | run_dir={run_dir}", flush=True)
    copied_input = run_dir / f"input_frame{scoring_frame_path.suffix.lower()}"
    copied_input.write_bytes(scoring_frame_path.read_bytes())
    all_scores: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    for bundle in filtered_bundles:
        print(
            f"[inference] scoring problem={bundle['manifest']['problem_key']} | model={bundle['manifest']['model_name']}",
            flush=True,
        )
        scored, validation = _score_bundle_on_frame(scoring_frame, latest_observed_ts, bundle)
        all_scores.append(scored)
        validation_rows.append(validation)
    scored_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    validation_df = pd.DataFrame(validation_rows)
    high_risk = scored_df[scored_df["risk_score"].notna()].sort_values(
        ["problem_key", "risk_score", "teacher_unique_id"],
        ascending=[True, False, True],
    )
    scored_df.to_parquet(run_dir / "scores_all_models.parquet", index=False)
    high_risk.to_parquet(run_dir / "high_risk_users.parquet", index=False)
    validation_df.to_parquet(run_dir / "validation_report.parquet", index=False)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "run_name": run_name or "",
            "input_kind": "scoring_frame_file",
            "scoring_frame_path": str(scoring_frame_path.resolve()),
            "copied_input_path": str(copied_input),
            "latest_observed_ts": str(latest_observed_ts) if latest_observed_ts is not None else "",
            "serving_status": manifest["serving_status"],
            "scored_model_count": len(filtered_bundles),
            "scored_rows": int(len(scored_df)),
            "eligible_rows": int(pd.to_numeric(scored_df["score_window_ready_flag"], errors="coerce").fillna(0).sum()) if not scored_df.empty else 0,
            "artifacts": {
                "scores_all_models": str(run_dir / "scores_all_models.parquet"),
                "high_risk_users": str(run_dir / "high_risk_users.parquet"),
                "validation_report": str(run_dir / "validation_report.parquet"),
            },
        },
    )
    write_json(paths.inference_runs_dir / "latest.json", {"latest_run_dir": str(run_dir)})
    print(f"[inference] completed run_dir={run_dir}", flush=True)
    return run_dir
