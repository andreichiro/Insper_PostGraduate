"""FastAPI serving layer — diabetes prediction with API-key security."""

from __future__ import annotations

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    clean_data,
    transform_encoders,
    transform_scalers,
)
from insper_deploy_kedro.pipelines.inference.nodes import predict

logger = logging.getLogger(__name__)

_artifacts: dict[str, Any] = {}

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """Validate API key when API_KEY env var is set."""
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Pydantic schemas ─────────────────────────────────────────────────


class DiabetesFeatures(BaseModel):
    """Input features for a single patient."""

    model_config = ConfigDict(extra="forbid")

    Pregnancies: float = Field(..., ge=0, examples=[6])
    Glucose: float = Field(..., ge=0, examples=[148])
    BloodPressure: float = Field(..., ge=0, examples=[72])
    SkinThickness: float = Field(..., ge=0, examples=[35])
    Insulin: float = Field(..., ge=0, examples=[0])
    BMI: float = Field(..., ge=0, examples=[33.6])
    DiabetesPedigreeFunction: float = Field(..., ge=0, examples=[0.627])
    Age: float = Field(..., ge=0, examples=[50])


class InferenceRequest(BaseModel):
    """Batch inference request."""

    model_config = ConfigDict(extra="forbid")

    instances: list[DiabetesFeatures] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    """Single prediction output."""

    prediction: str
    prediction_proba: float | None = None


class InferenceResponse(BaseModel):
    """Batch inference response."""

    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    model_loaded: bool
    model_version: str | None = None


# ── Artifact loading ─────────────────────────────────────────────────


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load production artifacts once on startup."""
    project_root = Path.cwd()
    models_dir = project_root / "data" / "06_models"
    conf_path = project_root / "conf" / "base" / "parameters" / "inference.yml"

    with open(conf_path) as fh:
        inference_params = yaml.safe_load(fh)

    _artifacts["encoders"] = _load_pickle(models_dir / "production_encoders.pkl")
    _artifacts["scalers"] = _load_pickle(models_dir / "production_scalers.pkl")
    _artifacts["model"] = _load_pickle(models_dir / "production_model.pkl")
    _artifacts["inference_raw_columns"] = inference_params["inference_raw_columns"]

    model_info = _artifacts["model"]
    _artifacts["model_version"] = model_info.get("class_path", "unknown")

    logger.info("Production artifacts loaded from %s", models_dir)
    yield
    _artifacts.clear()


# ── Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes outcome using a production ML pipeline.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse(
        status="healthy",
        model_loaded=bool(_artifacts),
        model_version=_artifacts.get("model_version"),
    )


@app.post("/inference", response_model=InferenceResponse)
def run_inference(
    request: InferenceRequest,
    _key: str | None = Depends(_verify_api_key),
) -> InferenceResponse:
    """Run the full inference pipeline on a batch of patients."""
    if not _artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    try:
        raw_df = pd.DataFrame([inst.model_dump() for inst in request.instances])

        cleaned = clean_data(raw_df, _artifacts["inference_raw_columns"])
        featured = add_features(cleaned)
        encoded = transform_encoders(featured, _artifacts["encoders"])
        scaled = transform_scalers(encoded, _artifacts["scalers"])
        result = predict(scaled, _artifacts["model"])

        predictions = [
            PredictionResult(
                prediction=str(row["prediction"]),
                prediction_proba=(
                    float(row["prediction_proba"])
                    if "prediction_proba" in row
                    else None
                ),
            )
            for _, row in result.iterrows()
        ]
        return InferenceResponse(predictions=predictions)

    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
