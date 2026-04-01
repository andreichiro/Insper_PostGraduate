from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd

from targeted_ml.pipelines.modelled_to_ml.analysis_setup import RuntimeBuildConfig


KEY_COLUMNS = ["teacher_unique_id", "first_month"]


def _load_python_callable(reference: str):
    module_ref, callable_name = reference.split(":", 1)
    module_path = Path(module_ref)
    if module_path.exists():
        spec = importlib.util.spec_from_file_location("targeted_ml_custom_strategy", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load python strategy from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    return getattr(module, callable_name)


def _run_sql_strategy(sql_file: str, frame: pd.DataFrame) -> pd.DataFrame:
    sql_path = Path(sql_file)
    query = sql_path.read_text(encoding="utf-8")
    conn = duckdb.connect()
    conn.register("input_frame", frame)
    try:
        return conn.execute(query).fetchdf()
    finally:
        conn.close()


def _run_python_strategy(reference: str, frame: pd.DataFrame) -> pd.DataFrame:
    callable_obj = _load_python_callable(reference)
    result = callable_obj(frame.copy())
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Python custom metric strategy must return a pandas DataFrame.")
    return result


def _validate_custom_metric_frame(custom_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in KEY_COLUMNS if column not in custom_df.columns]
    if missing:
        raise ValueError(f"Custom metric output is missing key columns: {missing}")
    metric_columns = [column for column in custom_df.columns if column not in KEY_COLUMNS]
    if not metric_columns:
        raise ValueError("Custom metric output must contain at least one metric column besides teacher_unique_id and first_month.")
    return custom_df[KEY_COLUMNS + metric_columns].copy()


def apply_custom_metric_overrides(frame: pd.DataFrame, runtime_config: RuntimeBuildConfig) -> pd.DataFrame:
    custom_frames: list[pd.DataFrame] = []
    if runtime_config.definition_a_sql_file:
        custom_frames.append(_validate_custom_metric_frame(_run_sql_strategy(runtime_config.definition_a_sql_file, frame)))
    if runtime_config.definition_a_python_strategy:
        custom_frames.append(_validate_custom_metric_frame(_run_python_strategy(runtime_config.definition_a_python_strategy, frame)))
    if runtime_config.definition_b_sql_file:
        custom_frames.append(_validate_custom_metric_frame(_run_sql_strategy(runtime_config.definition_b_sql_file, frame)))
    if runtime_config.definition_b_python_strategy:
        custom_frames.append(_validate_custom_metric_frame(_run_python_strategy(runtime_config.definition_b_python_strategy, frame)))

    merged = frame.copy()
    for custom_df in custom_frames:
        merged = merged.merge(custom_df, on=KEY_COLUMNS, how="left", suffixes=("", "__custom"))
        custom_columns = [column for column in merged.columns if column.endswith("__custom")]
        for custom_column in custom_columns:
            base_column = custom_column.removesuffix("__custom")
            merged[base_column] = merged[custom_column].combine_first(merged.get(base_column))
            merged = merged.drop(columns=[custom_column])
    return merged
