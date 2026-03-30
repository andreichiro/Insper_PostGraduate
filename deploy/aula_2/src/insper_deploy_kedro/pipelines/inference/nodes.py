"""Inference nodes -- only new functions (to_dataframe, predict)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def to_dataframe(raw_input: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw input (single dict or list of dicts) into a DataFrame."""
    if isinstance(raw_input, dict):
        raw_input = [raw_input]
    dataframe = pd.DataFrame(raw_input)
    logger.info(
        "to_dataframe: %d rows, %d columns", len(dataframe), len(dataframe.columns)
    )
    return dataframe


def predict(
    features_dataframe: pd.DataFrame,
    model_artifact: dict[str, Any],
) -> pd.DataFrame:
    """Run production model on prepared features. Returns predictions with decoded labels."""
    estimator = model_artifact["estimator"]
    target_encoder = model_artifact["target_encoder"]
    feature_columns = model_artifact["feature_columns"]

    x_inference = features_dataframe[feature_columns]

    predicted_codes = estimator.predict(x_inference)
    predicted_labels = target_encoder.inverse_transform(predicted_codes)

    predictions = features_dataframe.copy()
    predictions["prediction"] = predicted_labels

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_inference)[:, 1]
        predictions["prediction_proba"] = probabilities

    logger.info("predict: %d predictions made", len(predictions))
    return predictions
