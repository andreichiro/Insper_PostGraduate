"""Shared fixtures for diabetes project tests."""

from __future__ import annotations

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
        "target": ["Outcome"],
        "categorical": [],
        "numerical": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    }


@pytest.fixture()
def columns_config() -> dict[str, list[str]]:
    """Full column config — includes derived features (used by encoders/scalers/model)."""
    return {
        "target": ["Outcome"],
        "categorical": [],
        "numerical": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "glucose_bmi_interaction",
        ],
    }


@pytest.fixture()
def inference_raw_columns(raw_columns_config: dict) -> dict[str, list[str]]:
    """Inference raw columns (no target)."""
    return {k: v for k, v in raw_columns_config.items() if k != "target"}


@pytest.fixture()
def sample_raw_data() -> pd.DataFrame:
    """Minimal Pima diabetes dataframe (10 rows)."""
    return pd.DataFrame(
        {
            "Pregnancies": [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
            "Glucose": [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
            "BloodPressure": [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
            "SkinThickness": [35, 29, 0, 23, 35, 0, 32, 0, 45, 0],
            "Insulin": [0, 0, 0, 94, 168, 0, 88, 0, 543, 0],
            "BMI": [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0],
            "DiabetesPedigreeFunction": [
                0.627,
                0.351,
                0.672,
                0.167,
                2.288,
                0.201,
                0.248,
                0.134,
                0.158,
                0.232,
            ],
            "Age": [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
            "Outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        }
    )


@pytest.fixture()
def split_ratio() -> dict[str, float]:
    return {"train": 0.7, "validation": 0.15, "test": 0.15}


@pytest.fixture()
def fit_transform_config() -> dict[str, list[str]]:
    return {"split_to_fit": ["train"]}


@pytest.fixture()
def master_table(
    sample_raw_data,
    raw_columns_config,
    columns_config,
    split_ratio,
    fit_transform_config,
) -> pd.DataFrame:
    """Full master table through the DE pipeline."""
    cleaned = clean_data(sample_raw_data, raw_columns_config)
    featured = add_features(cleaned)
    split = add_split_column(featured, split_ratio, random_state=42)
    encoders = fit_encoders(split, columns_config, fit_transform_config)
    encoded = transform_encoders(split, encoders)
    scalers = fit_scalers(encoded, columns_config, fit_transform_config)
    return transform_scalers(encoded, scalers)


@pytest.fixture()
def trained_model(master_table, columns_config) -> dict:
    """A trained LogReg model artifact for reuse."""
    params = {
        "class_path": "sklearn.linear_model.LogisticRegression",
        "train_splits": ["train"],
        "init_args": {"max_iter": 1000},
    }
    return train_model(master_table, columns_config, params)
