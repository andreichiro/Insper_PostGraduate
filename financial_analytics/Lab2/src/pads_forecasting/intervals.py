"""Prediction interval helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_bootstrap_intervals(
    point_forecast: np.ndarray | pd.Series,
    residuals: np.ndarray | pd.Series,
    *,
    levels: tuple[int, ...] = (80, 95),
) -> pd.DataFrame:
    """Create symmetric empirical residual intervals around point forecasts."""

    forecast = np.asarray(point_forecast, dtype=float)
    residual_arr = pd.Series(residuals).dropna().astype(float).to_numpy()
    if residual_arr.size == 0:
        residual_arr = np.array([0.0])

    data: dict[str, np.ndarray] = {}
    for level in levels:
        alpha = (100 - level) / 2
        lo_q = np.percentile(residual_arr, alpha)
        hi_q = np.percentile(residual_arr, 100 - alpha)
        data[f"lo_{level}"] = forecast + lo_q
        data[f"hi_{level}"] = forecast + hi_q
    return pd.DataFrame(data)
