"""End-to-end tests for the FastAPI inference API."""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from insper_deploy_kedro.api import _artifacts, app, get_artifacts, verify_api_key

VALID_PAYLOAD = {
    "instances": [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
        }
    ]
}


@pytest.fixture()
def client(production_artifacts):
    """TestClient with artifacts injected via dependency overrides."""
    _artifacts.update(production_artifacts)
    app.dependency_overrides[get_artifacts] = lambda: production_artifacts
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc
    app.dependency_overrides.clear()
    _artifacts.clear()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_health_shows_model_version(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert body["model_version"] is not None

    def test_health_reports_not_loaded_when_empty(self):
        """Health should still respond 200 but report model_loaded=False."""
        app.dependency_overrides[verify_api_key] = lambda: None
        with TestClient(app, raise_server_exceptions=False) as tc:
            _artifacts.clear()
            resp = tc.get("/health")
        app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False


class TestInferenceEndpoint:
    def test_valid_request_returns_predictions(self, client):
        resp = client.post("/inference", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 1
        pred = body["predictions"][0]
        assert "prediction" in pred
        assert pred["prediction_proba"] is not None

    def test_prediction_proba_in_valid_range(self, client):
        resp = client.post("/inference", json=VALID_PAYLOAD)
        proba = resp.json()["predictions"][0]["prediction_proba"]
        assert 0.0 <= proba <= 1.0

    def test_batch_request(self, client):
        payload = {"instances": VALID_PAYLOAD["instances"] * 3}
        resp = client.post("/inference", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 3

    def test_missing_field_returns_422(self, client):
        bad = {"instances": [{"gender": "Female"}]}
        resp = client.post("/inference", json=bad)
        assert resp.status_code == 422

    def test_empty_instances_returns_422(self, client):
        resp = client.post("/inference", json={"instances": []})
        assert resp.status_code == 422

    def test_extra_field_rejected(self, client):
        payload = {
            "instances": [{**VALID_PAYLOAD["instances"][0], "unknown_field": 42}]
        }
        resp = client.post("/inference", json=payload)
        assert resp.status_code == 422

    def test_negative_tenure_rejected(self, client):
        payload = {"instances": [{**VALID_PAYLOAD["instances"][0], "tenure": -5}]}
        resp = client.post("/inference", json=payload)
        assert resp.status_code == 422

    def test_senior_citizen_out_of_range_rejected(self, client):
        payload = {"instances": [{**VALID_PAYLOAD["instances"][0], "SeniorCitizen": 3}]}
        resp = client.post("/inference", json=payload)
        assert resp.status_code == 422

    def test_negative_charges_rejected(self, client):
        payload = {
            "instances": [{**VALID_PAYLOAD["instances"][0], "MonthlyCharges": -10.0}]
        }
        resp = client.post("/inference", json=payload)
        assert resp.status_code == 422


class TestArtifactsNotLoaded:
    def test_inference_returns_503_when_no_model(self):
        def _no_artifacts():
            raise HTTPException(status_code=503, detail="Model artifacts not loaded")

        app.dependency_overrides[get_artifacts] = _no_artifacts
        app.dependency_overrides[verify_api_key] = lambda: None
        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post("/inference", json=VALID_PAYLOAD)
        app.dependency_overrides.clear()
        assert resp.status_code == 503


class TestApiKeySecurity:
    def test_returns_401_when_key_required_but_missing(self, production_artifacts):
        app.dependency_overrides[get_artifacts] = lambda: production_artifacts
        os.environ["API_KEY"] = "test-secret-key"
        try:
            with TestClient(app, raise_server_exceptions=False) as tc:
                resp = tc.post("/inference", json=VALID_PAYLOAD)
            assert resp.status_code == 401
        finally:
            os.environ.pop("API_KEY", None)
            app.dependency_overrides.clear()

    def test_returns_401_when_wrong_key(self, production_artifacts):
        app.dependency_overrides[get_artifacts] = lambda: production_artifacts
        os.environ["API_KEY"] = "correct-key"
        try:
            with TestClient(app, raise_server_exceptions=False) as tc:
                resp = tc.post(
                    "/inference",
                    json=VALID_PAYLOAD,
                    headers={"X-API-Key": "wrong-key"},
                )
            assert resp.status_code == 401
        finally:
            os.environ.pop("API_KEY", None)
            app.dependency_overrides.clear()

    def test_returns_200_when_correct_key_provided(self, production_artifacts):
        _artifacts.update(production_artifacts)
        app.dependency_overrides[get_artifacts] = lambda: production_artifacts
        os.environ["API_KEY"] = "test-secret-key"
        try:
            with TestClient(app, raise_server_exceptions=False) as tc:
                resp = tc.post(
                    "/inference",
                    json=VALID_PAYLOAD,
                    headers={"X-API-Key": "test-secret-key"},
                )
            assert resp.status_code == 200
        finally:
            os.environ.pop("API_KEY", None)
            app.dependency_overrides.clear()
            _artifacts.clear()

    def test_health_does_not_require_key(self, production_artifacts):
        """Health endpoint should always be public (for Cloud Run probes)."""
        _artifacts.update(production_artifacts)
        os.environ["API_KEY"] = "test-secret-key"
        try:
            with TestClient(app, raise_server_exceptions=False) as tc:
                resp = tc.get("/health")
            assert resp.status_code == 200
        finally:
            os.environ.pop("API_KEY", None)
            _artifacts.clear()
