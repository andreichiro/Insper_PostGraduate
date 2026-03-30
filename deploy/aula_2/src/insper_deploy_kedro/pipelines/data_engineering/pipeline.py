"""Pipeline DE: raw -> limpar -> validar -> features -> split -> validar -> encode -> scale -> master_table."""

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_features,
    add_split_column,
    clean_data,
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)
from .validations import validate_clean_data, validate_split_data


def create_pipeline(**kwargs: Any) -> Pipeline:  # noqa: ARG001
    """Monta o DAG de data engineering."""
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["raw_data", "params:raw_columns"],
                outputs="cleaned_data_raw",
                name="clean_data_node",
                tags=["data_engineering", "cleaning"],
            ),
            node(
                func=validate_clean_data,
                inputs=[
                    "cleaned_data_raw",
                    "params:raw_columns",
                    "params:data_quality",
                ],
                outputs="cleaned_data",
                name="validate_clean_data_node",
                tags=["data_engineering", "validation"],
            ),
            node(
                func=add_features,
                inputs=["cleaned_data"],
                outputs="featured_data",
                name="add_features_node",
                tags=["data_engineering", "feature_engineering"],
            ),
            node(
                func=add_split_column,
                inputs=[
                    "featured_data",
                    "params:split_ratio",
                    "params:random_state",
                    "params:stratify_column",
                    "params:preprocessing",
                ],
                outputs="split_data_raw",
                name="add_split_column_node",
                tags=["data_engineering", "splitting"],
            ),
            node(
                func=validate_split_data,
                inputs=[
                    "split_data_raw",
                    "params:split_ratio",
                    "params:stratify_column",
                    "params:data_quality",
                ],
                outputs="split_data",
                name="validate_split_data_node",
                tags=["data_engineering", "validation"],
            ),
            node(
                func=fit_encoders,
                inputs=[
                    "split_data",
                    "params:columns",
                    "params:fit_transform",
                    "params:preprocessing",
                ],
                outputs="encoders",
                name="fit_encoders_node",
                tags=["data_engineering", "encoding"],
            ),
            node(
                func=transform_encoders,
                inputs=["split_data", "encoders"],
                outputs="encoded_data",
                name="transform_encoders_node",
                tags=["data_engineering", "encoding"],
            ),
            node(
                func=fit_scalers,
                inputs=[
                    "encoded_data",
                    "params:columns",
                    "params:fit_transform",
                    "params:preprocessing",
                ],
                outputs="scalers",
                name="fit_scalers_node",
                tags=["data_engineering", "scaling"],
            ),
            node(
                func=transform_scalers,
                inputs=["encoded_data", "scalers"],
                outputs="master_table",
                name="transform_scalers_node",
                tags=["data_engineering", "scaling"],
            ),
        ]
    )
