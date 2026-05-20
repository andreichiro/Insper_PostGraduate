"""Small MLflow helpers used in addition to kedro-mlflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash for a nested config dictionary."""

    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Stable-enough fingerprint for a small dataframe."""

    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def mlflow_log_artifacts(paths: list[str | Path]) -> None:
    """Log artifacts if MLflow has an active run; otherwise no-op."""

    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        for path in paths:
            artifact_path = Path(path)
            if artifact_path.exists():
                if artifact_path.is_dir():
                    mlflow.log_artifacts(str(artifact_path))
                else:
                    mlflow.log_artifact(str(artifact_path))
    except Exception:
        return


def mlflow_log_metrics(metrics: dict[str, Any], prefix: str | None = None) -> None:
    """Log numeric metrics if MLflow is active."""

    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and pd.notna(value):
                mlflow.log_metric(f"{prefix}.{key}" if prefix else key, float(value))
    except Exception:
        return


def mlflow_log_params(params: dict[str, Any], prefix: str | None = None) -> None:
    """Log small run parameters if MLflow is active."""

    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        for key, value in params.items():
            mlflow.log_param(f"{prefix}.{key}" if prefix else key, str(value))
    except Exception:
        return
