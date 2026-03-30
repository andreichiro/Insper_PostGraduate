"""Fixtures compartilhadas pros testes do projeto diabetes."""

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

# Espelha conf/base/parameters/*.yml — testes ficam alinhados ao config declarativo.
PREPROCESSING_FIXTURE: dict = {
    "train_test_split_function": "sklearn.model_selection.train_test_split",
    "min_rows_for_stratify": 20,
    "categorical_encoder": {
        "class_path": "sklearn.preprocessing.OrdinalEncoder",
        "init_args": {
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
            "dtype": int,
        },
    },
    "numerical_scaler": {
        "class_path": "sklearn.preprocessing.StandardScaler",
        "init_args": {},
    },
}

ML_RUNTIME_FIXTURE: dict = {
    "target_encoder": {
        "class_path": "sklearn.preprocessing.LabelEncoder",
        "init_args": {},
    },
    "cross_validation": {
        "class_path": "sklearn.model_selection.StratifiedKFold",
        "init_args": {"shuffle": True, "random_state": 42},
    },
    "optuna_study": {"direction": "maximize"},
    "optuna_sampler": {
        "class_path": "optuna.samplers.TPESampler",
        "init_args": {"seed": 42},
    },
}

EVALUATION_FIXTURE: dict = {
    "confusion_matrix": {
        "function_path": "sklearn.metrics.confusion_matrix",
        "kwargs": {},
    },
    "metrics": [
        {
            "key": "accuracy",
            "function_path": "sklearn.metrics.accuracy_score",
            "prediction_input": "y_pred",
            "kwargs": {},
        },
        {
            "key": "precision",
            "function_path": "sklearn.metrics.precision_score",
            "prediction_input": "y_pred",
            "kwargs": {"zero_division": 0},
        },
        {
            "key": "recall",
            "function_path": "sklearn.metrics.recall_score",
            "prediction_input": "y_pred",
            "kwargs": {"zero_division": 0},
        },
        {
            "key": "f1",
            "function_path": "sklearn.metrics.f1_score",
            "prediction_input": "y_pred",
            "kwargs": {"zero_division": 0},
        },
        {
            "key": "roc_auc",
            "function_path": "sklearn.metrics.roc_auc_score",
            "prediction_input": "y_proba",
            "kwargs": {},
        },
    ],
    "derived": {
        "r2": {
            "function_path": "sklearn.metrics.r2_score",
            "prediction_input": "y_proba",
            "kwargs": {},
        },
        "mape": {"type": "mae_as_percent_of_mean_label"},
    },
}


@pytest.fixture()
def preprocessing_config() -> dict:
    return PREPROCESSING_FIXTURE


@pytest.fixture()
def ml_runtime_config() -> dict:
    return ML_RUNTIME_FIXTURE


@pytest.fixture()
def evaluation_config() -> dict:
    return EVALUATION_FIXTURE


@pytest.fixture()
def raw_columns_config() -> dict[str, list[str]]:
    """Config de colunas brutas, o que clean_data seleciona do arquivo"""
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
    """Config completa de colunas, incluindo features derivadas"""
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
    """Colunas brutas de inferência (sem target)"""
    return {k: v for k, v in raw_columns_config.items() if k != "target"}


@pytest.fixture()
def sample_raw_data() -> pd.DataFrame:
    """Sample hardcoded do dataframe diabetes Pima (10 linhas)"""
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
    preprocessing_config,
) -> pd.DataFrame:
    """Master table completa passando por todo o pipeline DE"""
    cleaned = clean_data(sample_raw_data, raw_columns_config)
    featured = add_features(cleaned)
    split = add_split_column(
        featured,
        split_ratio,
        random_state=42,
        stratify_column="Outcome",
        preprocessing=preprocessing_config,
    )
    encoders = fit_encoders(
        split, columns_config, fit_transform_config, preprocessing_config
    )
    encoded = transform_encoders(split, encoders)
    scalers = fit_scalers(
        encoded, columns_config, fit_transform_config, preprocessing_config
    )
    return transform_scalers(encoded, scalers)


@pytest.fixture()
def trained_model(master_table, columns_config, ml_runtime_config) -> dict:
    """Artefato de modelo treinado pra reuso nos testes"""
    params = {
        "class_path": "sklearn.linear_model.LogisticRegression",
        "train_splits": ["train"],
        "init_args": {"max_iter": 1000},
    }
    return train_model(master_table, columns_config, params, ml_runtime_config)
