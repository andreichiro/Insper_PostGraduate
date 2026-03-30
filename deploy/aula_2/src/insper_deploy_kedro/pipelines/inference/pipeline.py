"""Inference pipeline: clean -> features -> transform (no fit) -> predict.

Reuses clean_data, add_features, transform_encoders, transform_scalers from DE.
Only predict is new. No fitting happens here.
"""

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    clean_data,
    transform_encoders,
    transform_scalers,
)

from .nodes import predict


def create_pipeline(**kwargs: Any) -> Pipeline:  # noqa: ARG001
    """Wire up the inference DAG -- transform only, no fitting."""
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["raw_data_inference", "params:inference_raw_columns"],
                outputs="cleaned_inference",
                name="clean_inference_node",
            ),
            node(
                func=add_features,
                inputs=["cleaned_inference"],
                outputs="featured_inference",
                name="add_features_inference_node",
            ),
            node(
                func=transform_encoders,
                inputs=["featured_inference", "production_encoders"],
                outputs="encoded_inference",
                name="encode_inference_node",
            ),
            node(
                func=transform_scalers,
                inputs=["encoded_inference", "production_scalers"],
                outputs="scaled_inference",
                name="scale_inference_node",
            ),
            node(
                func=predict,
                inputs=["scaled_inference", "production_model"],
                outputs="predictions",
                name="predict_node",
            ),
        ]
    )
