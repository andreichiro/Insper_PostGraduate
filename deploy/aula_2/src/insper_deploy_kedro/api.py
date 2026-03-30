"""Camada de serving FastAPI, predição de diabetes"""

from __future__ import annotations

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
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
    """Valida API key quando a env var API_KEY tá setada."""
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


#Schema
class DiabetesFeatures(BaseModel):
    """Features de entrada pra um paciente."""

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
    """Request de inferência em batch."""

    model_config = ConfigDict(extra="forbid")

    instances: list[DiabetesFeatures] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    """Resultado de uma predição."""

    prediction: str
    prediction_proba: float | None = None


class InferenceResponse(BaseModel):
    """Response de inferência em batch."""

    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Response do health check."""

    status: str
    model_loaded: bool
    model_version: str | None = None


#Artefatos (treinados localmente, inferencia pode ser pelo google cloud run)

def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carrega artefatos de produção uma vez na inicialização."""
    project_root = Path.cwd()
    models_dir = project_root / "data" / "06_models"

    try:
        _artifacts["encoders"] = _load_pickle(models_dir / "production_encoders.pkl")
        _artifacts["scalers"] = _load_pickle(models_dir / "production_scalers.pkl")
        _artifacts["model"] = _load_pickle(models_dir / "production_model.pkl")
        _artifacts["inference_raw_columns"] = {
            "categorical": [],
            "numerical": list(DiabetesFeatures.model_fields.keys()),
        }
        _artifacts["model_version"] = _artifacts["model"].get("class_path", "unknown")
        logger.info("Artefatos de produção carregados de %s", models_dir)
    except FileNotFoundError:
        logger.warning("Artefatos não encontrados em %s — rode kedro run primeiro", models_dir)

    yield
    _artifacts.clear()


# Camada de Aplicação

app = FastAPI(
    title="API de Predição de Diabetes",
    description="Prever diabetes usando um pipeline ML de produção.",
    version="0.2.0",
    lifespan=lifespan,
)

# Health check q estava faltando
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Probe de liveness / readiness."""
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
    """Roda o pipeline completo de inferência num batch de pacientes"""
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
        logger.exception("Inferência falhou")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
