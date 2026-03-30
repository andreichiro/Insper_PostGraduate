"""FastAPI serving layer — churn prediction.

Artifacts are loaded via Kedro's DataCatalog, so switching from local
filesystem to GCS only requires a catalog overlay:

    KEDRO_ENV=cloud  →  conf/cloud/catalog.yml (gs:// paths)
    KEDRO_ENV=local  →  conf/base/catalog.yml  (local paths, default)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog
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


# ── Pydantic schemas ─────────────────────────────────────────────────


class ChurnFeatures(BaseModel):
    """Input features for a single customer."""

    model_config = ConfigDict(extra="forbid")

    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: float = Field(..., ge=0, examples=[1])
    PhoneService: str = Field(..., examples=["No"])
    MultipleLines: str = Field(..., examples=["No phone service"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[29.85])
    TotalCharges: float = Field(..., ge=0, examples=[29.85])


class InferenceRequest(BaseModel):
    """Batch inference request."""

    model_config = ConfigDict(extra="forbid")

    instances: list[ChurnFeatures] = Field(..., min_length=1)


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


# ── Security ─────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Validate API key when ``API_KEY`` env var is set.

    When no key is configured the check is skipped (development mode).
    """
    expected = os.environ.get("API_KEY")
    if expected is None:
        return
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Dependency injection ─────────────────────────────────────────────


def get_artifacts() -> dict[str, Any]:
    """Return loaded artifacts or raise 503 if the model is not ready."""
    if not _artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return _artifacts


# ── Artifact loading via Kedro catalog ───────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load production artifacts via Kedro's DataCatalog.

    ``KEDRO_ENV`` controls which catalog overlay is used:
    - ``local`` (default): reads from local ``data/06_models/``
    - ``cloud``: reads from ``gs://BUCKET/...`` (conf/cloud/catalog.yml)
    """
    project_root = Path.cwd()
    kedro_env = os.environ.get("KEDRO_ENV", "local")

    config_loader = OmegaConfigLoader(
        conf_source=str(project_root / "conf"),
        base_env="base",
        default_run_env=kedro_env,
    )

    params = config_loader["parameters"]
    _artifacts["inference_raw_columns"] = params["inference_raw_columns"]

    catalog = DataCatalog.from_config(config_loader["catalog"])
    _artifacts["encoders"] = catalog.load("production_encoders")
    _artifacts["scalers"] = catalog.load("production_scalers")
    _artifacts["model"] = catalog.load("production_model")

    logger.info("Production artifacts loaded (env=%s)", kedro_env)
    yield
    _artifacts.clear()


# ── Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using a production ML pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness / readiness probe (no auth required)."""
    model = _artifacts.get("model")
    return HealthResponse(
        status="healthy",
        model_loaded=bool(_artifacts),
        model_version=model.get("class_path") if isinstance(model, dict) else None,
    )


@app.post(
    "/inference",
    response_model=InferenceResponse,
    dependencies=[Depends(verify_api_key)],
)
def run_inference(
    request: InferenceRequest,
    artifacts: Annotated[dict[str, Any], Depends(get_artifacts)],
) -> InferenceResponse:
    """Run the full inference pipeline on a batch of customers."""
    try:
        raw_df = pd.DataFrame([inst.model_dump() for inst in request.instances])

        cleaned = clean_data(raw_df, artifacts["inference_raw_columns"])
        featured = add_features(cleaned)
        encoded = transform_encoders(featured, artifacts["encoders"])
        scaled = transform_scalers(encoded, artifacts["scalers"])
        result = predict(scaled, artifacts["model"])

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
