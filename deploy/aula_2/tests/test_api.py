"""End-to-end tests for the FastAPI inference API."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from insper_deploy_kedro.api import _artifacts, app
from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    add_split_column,
    clean_data,
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)
from insper_deploy_kedro.pipelines.modelling.nodes import train_model


def _build_production_artifacts(sample_raw_data, raw_columns_config, columns_config):
    """Build real (small) production artifacts from test fixtures."""
    split_ratio = {"train": 0.7, "validation": 0.15, "test": 0.15}
    fit_cfg = {"split_to_fit": ["train"]}

    cleaned = clean_data(sample_raw_data, raw_columns_config)
    featured = add_features(cleaned)
    split = add_split_column(featured, split_ratio, random_state=42)
    encoders = fit_encoders(split, columns_config, fit_cfg)
    encoded = transform_encoders(split, encoders)
    scalers = fit_scalers(encoded, columns_config, fit_cfg)
    master = transform_scalers(encoded, scalers)

    model = train_model(
        master,
        columns_config,
        {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        },
    )

    return {
        "encoders": encoders,
        "scalers": scalers,
        "model": model,
        "inference_raw_columns": {
            k: v for k, v in raw_columns_config.items() if k != "target"
        },
        "model_version": "sklearn.linear_model.LogisticRegression",
    }


@pytest.fixture()
def client(sample_raw_data, raw_columns_config, columns_config):
    """TestClient with pre-loaded artifacts (no disk I/O)."""
    artifacts = _build_production_artifacts(
        sample_raw_data, raw_columns_config, columns_config
    )
    _artifacts.update(artifacts)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc
    _artifacts.clear()


VALID_PAYLOAD = {
    "instances": [
        {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50,
        }
    ]
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_health_includes_model_version(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert body["model_version"] is not None


class TestInferenceEndpoint:
    def test_valid_request_returns_predictions(self, client):
        resp = client.post("/inference", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 1
        assert "prediction" in body["predictions"][0]

    def test_batch_request(self, client):
        payload = {"instances": VALID_PAYLOAD["instances"] * 3}
        resp = client.post("/inference", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 3

    def test_missing_field_returns_422(self, client):
        bad = {"instances": [{"Pregnancies": 6}]}
        resp = client.post("/inference", json=bad)
        assert resp.status_code == 422

    def test_empty_instances_returns_422(self, client):
        resp = client.post("/inference", json={"instances": []})
        assert resp.status_code == 422

    def test_extra_field_rejected(self, client):
        bad = {"instances": [{**VALID_PAYLOAD["instances"][0], "extra": 99}]}
        resp = client.post("/inference", json=bad)
        assert resp.status_code == 422


class TestAPISecurity:
    def test_no_key_required_when_env_unset(self, client):
        with mock.patch.dict(os.environ, {}, clear=True):
            resp = client.post("/inference", json=VALID_PAYLOAD)
            assert resp.status_code == 200

    def test_valid_key_accepted(self, client):
        with mock.patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/inference",
                json=VALID_PAYLOAD,
                headers={"X-API-Key": "test-key"},
            )
            assert resp.status_code == 200

    def test_invalid_key_rejected(self, client):
        with mock.patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/inference",
                json=VALID_PAYLOAD,
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 401
