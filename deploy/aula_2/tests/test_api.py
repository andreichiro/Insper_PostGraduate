"""Testes end-to-end da API FastAPI de inferência"""

from __future__ import annotations

import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from insper_deploy_kedro.api import _artifacts, app


@pytest.fixture()
def client(
    master_table,
    columns_config,
    raw_columns_config,
    trained_model,
    preprocessing_config,
):
    """TestClient com artefatos pré-carregados reusando fixtures do conftest"""
    from insper_deploy_kedro.pipelines.data_engineering.nodes import (
        fit_encoders,
        fit_scalers,
    )

    fit_cfg = {"split_to_fit": ["train"]}
    encoders = fit_encoders(
        master_table, columns_config, fit_cfg, preprocessing_config
    )
    scalers = fit_scalers(
        master_table, columns_config, fit_cfg, preprocessing_config
    )

    artifacts = {
        "encoders": encoders,
        "scalers": scalers,
        "model": trained_model,
        "inference_raw_columns": {
            k: v for k, v in raw_columns_config.items() if k != "target"
        },
        "model_version": trained_model["class_path"],
    }
    with TestClient(app, raise_server_exceptions=False) as tc:
        _artifacts.update(artifacts)
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

# Parte extra do health check, agr no teste
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

# Inferencia
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
