"""DE pipeline: raw -> clean -> features -> split -> encode -> scale -> master_table."""

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


def create_pipeline(**kwargs: Any) -> Pipeline:  # noqa: ARG001
    """Wire up the data engineering DAG."""
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["raw_data", "params:raw_columns"],
                outputs="cleaned_data",
                name="clean_data_node",
            ),
            node(
                func=add_features,
                inputs=["cleaned_data"],
                outputs="featured_data",
                name="add_features_node",
            ),
            node(
                func=add_split_column,
                inputs=[
                    "featured_data",
                    "params:split_ratio",
                    "params:random_state",
                    "params:stratify_column",
                ],
                outputs="split_data",
                name="add_split_column_node",
            ),
            node(
                func=fit_encoders,
                inputs=[
                    "split_data",
                    "params:columns",
                    "params:fit_transform",
                ],
                outputs="encoders",
                name="fit_encoders_node",
            ),
            node(
                func=transform_encoders,
                inputs=["split_data", "encoders"],
                outputs="encoded_data",
                name="transform_encoders_node",
            ),
            node(
                func=fit_scalers,
                inputs=[
                    "encoded_data",
                    "params:columns",
                    "params:fit_transform",
                ],
                outputs="scalers",
                name="fit_scalers_node",
            ),
            node(
                func=transform_scalers,
                inputs=["encoded_data", "scalers"],
                outputs="master_table",
                name="transform_scalers_node",
            ),
        ]
    )
