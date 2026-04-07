from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from targeted_ml.config.loader import load_analysis_spec
from targeted_ml.inference.service import (
    _build_delivery_outputs,
    _build_serving_contract,
    _next_available_dir,
    _score_bundle_on_frame,
    _write_inference_run_manifest,
    validate_inference_input_schema,
)


class DummyPredictor:
    def predict_proba(self, frame: pd.DataFrame):
        values = pd.to_numeric(frame["x"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        positive = values.clip(0.0, 1.0)
        return pd.DataFrame({"p0": 1.0 - positive, "p1": positive}).to_numpy()


def test_score_bundle_accepts_explicit_score_window_ready_flag() -> None:
    scoring_frame = pd.DataFrame(
        [
            {"teacher_unique_id": "u1", "first_month": "2024-01-01", "onboarding_anchor_ts": "2024-01-03", "x": 0.8, "score_window_ready_flag": 1},
            {"teacher_unique_id": "u2", "first_month": "2024-01-01", "onboarding_anchor_ts": "2024-01-04", "x": 0.2, "score_window_ready_flag": 1},
        ]
    )
    bundle = {
        "manifest": {
            "problem_key": "definition_a::rule__S1_PLUS_S7",
            "definition_name": "definition_a::rule",
            "track_name": "S1_PLUS_S7",
            "model_name": "catboost",
            "feature_schema": [{"feature_name": "x", "feature_type": "numeric"}],
        },
        "predictor": DummyPredictor(),
        "active_feature_names": ["x"],
        "score_window_end_day": 7,
    }

    scored, validation = _score_bundle_on_frame(scoring_frame, None, bundle)

    assert validation["valid_input_flag"] == 1
    scored_by_teacher = scored.set_index("teacher_unique_id")
    assert int(scored_by_teacher.loc["u2", "risk_rank"]) == 1
    assert int(scored_by_teacher.loc["u1", "risk_rank"]) == 2
    assert round(float(scored_by_teacher.loc["u1", "score_positive"]), 4) == 0.8
    assert round(float(scored_by_teacher.loc["u2", "score_positive"]), 4) == 0.2
    assert round(float(scored_by_teacher.loc["u1", "risk_score"]), 4) == 0.2
    assert round(float(scored_by_teacher.loc["u2", "risk_score"]), 4) == 0.8


def test_score_bundle_requires_latest_ts_when_ready_flag_missing() -> None:
    scoring_frame = pd.DataFrame(
        [{"teacher_unique_id": "u1", "first_month": "2024-01-01", "onboarding_anchor_ts": "2024-01-03", "x": 0.8}]
    )
    bundle = {
        "manifest": {
            "problem_key": "definition_a::rule__S1_PLUS_S7",
            "definition_name": "definition_a::rule",
            "track_name": "S1_PLUS_S7",
            "model_name": "catboost",
            "feature_schema": [{"feature_name": "x", "feature_type": "numeric"}],
        },
        "predictor": DummyPredictor(),
        "active_feature_names": ["x"],
        "score_window_end_day": 7,
    }

    with pytest.raises(ValueError, match="latest_observed_ts"):
        _score_bundle_on_frame(scoring_frame, None, bundle)


def test_validate_inference_input_schema_rejects_invalid_key_columns() -> None:
    scoring_frame = pd.DataFrame(
        [{"teacher_unique_id": " ", "first_month": "not-a-date", "onboarding_anchor_ts": "bad-ts", "x": 0.5}]
    )
    bundle = {
        "manifest": {
            "problem_key": "definition_a::rule__S1_PLUS_S7",
            "definition_name": "definition_a::rule",
            "track_name": "S1_PLUS_S7",
            "model_name": "catboost",
            "feature_schema": [{"feature_name": "x", "feature_type": "numeric"}],
        },
        "predictor": DummyPredictor(),
        "active_feature_names": ["x"],
        "score_window_end_day": 7,
    }

    validation = validate_inference_input_schema(scoring_frame, bundle)

    assert validation["valid_input_flag"] == 0
    assert validation["key_column_issues"]["teacher_unique_id_blank_or_missing"] == 1
    assert validation["key_column_issues"]["first_month_parse_issues"] == 1
    assert validation["key_column_issues"]["onboarding_anchor_ts_parse_issues"] == 1


def test_next_available_dir_appends_suffix(tmp_path) -> None:
    first = _next_available_dir(tmp_path, "run")
    first.mkdir()
    second = _next_available_dir(tmp_path, "run")
    assert second.name == "run_2"


def test_inference_run_manifest_tracks_requested_and_resolved_names(tmp_path) -> None:
    run_dir = tmp_path / "example_modelled_inference"
    run_dir.mkdir()
    _write_inference_run_manifest(
        run_dir=run_dir,
        requested_run_name="final_modelled_latest",
        payload={"run_id": "123", "serving_status": "unique_primary_model"},
    )

    payload = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["run_dir"] == str(run_dir)
    assert payload["run_name"] == "example_modelled_inference"
    assert payload["requested_run_name"] == "final_modelled_latest"


def test_build_delivery_outputs_adds_flags_and_filtered_views() -> None:
    scored = pd.DataFrame(
        [
            {
                "teacher_unique_id": "u1",
                "first_month": "2024-01-01",
                "onboarding_anchor_ts": "2024-01-03",
                "problem_key": "definition_a::rule__S1_PLUS_S7",
                "definition_name": "definition_a::rule",
                "track_name": "S1_PLUS_S7",
                "model_name": "catboost",
                "score_window_ready_flag": 1,
                "score_positive": 0.1,
                "risk_score": 0.9,
                "eligibility_reason": "",
                "risk_rank": 1,
            },
            {
                "teacher_unique_id": "u2",
                "first_month": "2024-01-01",
                "onboarding_anchor_ts": "2024-01-03",
                "problem_key": "definition_a::rule__S1_PLUS_S7",
                "definition_name": "definition_a::rule",
                "track_name": "S1_PLUS_S7",
                "model_name": "catboost",
                "score_window_ready_flag": 1,
                "score_positive": 0.4,
                "risk_score": 0.6,
                "eligibility_reason": "",
                "risk_rank": 2,
            },
            {
                "teacher_unique_id": "u3",
                "first_month": "2024-01-01",
                "onboarding_anchor_ts": "2024-01-03",
                "problem_key": "definition_a::rule__S1_PLUS_S7",
                "definition_name": "definition_a::rule",
                "track_name": "S1_PLUS_S7",
                "model_name": "catboost",
                "score_window_ready_flag": 1,
                "score_positive": 0.8,
                "risk_score": 0.2,
                "eligibility_reason": "",
                "risk_rank": 3,
            },
        ]
    )

    delivery, filtered = _build_delivery_outputs(scored)

    assert {"flag_top_10_percent", "flag_tercis", "flag_score_ge_0_70"}.issubset(delivery.columns)
    assert int(delivery.loc[delivery["teacher_unique_id"] == "u1", "flag_top_10_percent"].iloc[0]) == 1
    assert int(delivery.loc[delivery["teacher_unique_id"] == "u1", "flag_tercis"].iloc[0]) == 1
    assert int(delivery.loc[delivery["teacher_unique_id"] == "u2", "flag_tercis"].iloc[0]) == 0
    assert int(delivery.loc[delivery["teacher_unique_id"] == "u1", "flag_score_ge_0_70"].iloc[0]) == 1
    assert int(delivery.loc[delivery["teacher_unique_id"] == "u2", "flag_score_ge_0_70"].iloc[0]) == 0
    assert list(filtered["top_10_percent"]["teacher_unique_id"]) == ["u1"]
    assert list(filtered["tercis"]["teacher_unique_id"]) == ["u1"]
    assert list(filtered["score_ge_0_70"]["teacher_unique_id"]) == ["u1"]


def test_build_serving_contract_declares_raw_dataset_root_support(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_analysis_spec(repo_root / "specs" / "activity.yaml")
    schema_path = tmp_path / "artifact.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "required_modelled_tables": ["base_modelada_v2", "fct_session_clean"],
                "required_feature_columns": ["x"],
                "numeric_features": ["x"],
                "categorical_features": [],
            }
        ),
        encoding="utf-8",
    )
    contract, template = _build_serving_contract(
        spec,
        [
            {
                "artifact_id": "artifact",
                "problem_key": "definition_a::rule__S1_PLUS_S7",
                "definition_name": "definition_a::rule",
                "track_name": "S1_PLUS_S7",
                "model_name": "catboost",
                "schema_path": str(schema_path),
                "feature_path": str(tmp_path / "artifact.feature_list.json"),
            }
        ],
    )

    assert "raw_dataset_root" in contract["supported_input_kinds"]
    assert contract["raw_dataset_root_contract"]["supported"] is True
    assert contract["raw_dataset_root_contract"]["required_relative_path"] == "raw/base_aprendizap"
    assert "dim_teachers.csv" in contract["raw_dataset_root_contract"]["required_files"]
    assert list(template.columns) == ["teacher_unique_id", "first_month", "onboarding_anchor_ts", "x", "score_window_ready_flag"]
