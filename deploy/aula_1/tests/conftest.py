"""Shared fixtures for churn project tests."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

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


@pytest.fixture()
def raw_columns_config() -> dict[str, list[str]]:
    """Raw column config — what clean_data selects from raw CSV."""
    return {
        "target": ["Churn"],
        "categorical": [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
        "numerical": [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ],
    }


@pytest.fixture()
def columns_config() -> dict[str, list[str]]:
    """Full column config — includes derived features (used by encoders/scalers/model)."""
    return {
        "target": ["Churn"],
        "categorical": [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
        "numerical": [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "avg_charge_per_month",
        ],
    }


@pytest.fixture()
def inference_raw_columns(raw_columns_config: dict) -> dict[str, list[str]]:
    """Inference raw columns (no target)."""
    return {k: v for k, v in raw_columns_config.items() if k != "target"}


@pytest.fixture()
def sample_raw_data() -> pd.DataFrame:
    """Minimal Telco-Churn dataframe (8 rows)."""
    return pd.DataFrame(
        {
            "gender": [
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
            ],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0, 1, 0],
            "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "tenure": [1, 34, 2, 45, 12, 20, 5, 72],
            "PhoneService": ["No", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"],
            "MultipleLines": [
                "No phone service",
                "No",
                "Yes",
                "No phone service",
                "No",
                "No phone service",
                "Yes",
                "No",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "DSL",
                "No",
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "DSL",
            ],
            "OnlineSecurity": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "Yes",
                "No",
                "No",
                "Yes",
            ],
            "OnlineBackup": [
                "Yes",
                "No",
                "No",
                "No internet service",
                "Yes",
                "No",
                "Yes",
                "Yes",
            ],
            "DeviceProtection": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "No",
                "Yes",
                "No",
                "Yes",
            ],
            "TechSupport": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "Yes",
                "No",
                "No",
                "Yes",
            ],
            "StreamingTV": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "Yes",
                "No",
                "Yes",
                "Yes",
            ],
            "StreamingMovies": [
                "No",
                "No",
                "Yes",
                "No internet service",
                "Yes",
                "Yes",
                "No",
                "Yes",
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
                "Two year",
                "One year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 20.05, 70.70, 45.50, 80.25, 35.10],
            "TotalCharges": [
                29.85,
                1889.5,
                108.15,
                459.25,
                948.55,
                910.0,
                401.25,
                2500.0,
            ],
            "Churn": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


@pytest.fixture()
def split_ratio() -> dict[str, float]:
    return {"train": 0.7, "validation": 0.15, "test": 0.15}


@pytest.fixture()
def fit_transform_config() -> dict[str, list[str]]:
    return {"split_to_fit": ["train"]}


# ── Composite fixtures (DRY: single pipeline chain definition) ───────


@pytest.fixture()
def master_table(
    sample_raw_data,
    raw_columns_config,
    columns_config,
    split_ratio,
    fit_transform_config,
) -> pd.DataFrame:
    """Full DE pipeline: raw -> clean -> features -> split -> encode -> scale."""
    cleaned = clean_data(sample_raw_data, raw_columns_config)
    featured = add_features(cleaned)
    split = add_split_column(featured, split_ratio, random_state=42)
    encoders = fit_encoders(split, columns_config, fit_transform_config)
    encoded = transform_encoders(split, encoders)
    scalers = fit_scalers(encoded, columns_config, fit_transform_config)
    return transform_scalers(encoded, scalers)


@pytest.fixture()
def trained_model(master_table, columns_config) -> dict[str, Any]:
    """Baseline LogisticRegression trained on test data."""
    return train_model(
        master_table,
        columns_config,
        {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        },
    )


@pytest.fixture()
def production_artifacts(
    sample_raw_data,
    raw_columns_config,
    columns_config,
    split_ratio,
    fit_transform_config,
) -> dict[str, Any]:
    """Complete set of production artifacts for API tests."""
    cleaned = clean_data(sample_raw_data, raw_columns_config)
    featured = add_features(cleaned)
    split = add_split_column(featured, split_ratio, random_state=42)
    encoders = fit_encoders(split, columns_config, fit_transform_config)
    encoded = transform_encoders(split, encoders)
    scalers = fit_scalers(encoded, columns_config, fit_transform_config)
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
    }
