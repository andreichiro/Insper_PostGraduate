"""Refit pipeline: re-fit encoders/scalers/model on ALL data for production.

Reuses DE and modelling nodes with split_to_fit: [train, validation, test].
Adds probability calibration as a post-processing step.
"""

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)
from insper_deploy_kedro.pipelines.modelling.nodes import calibrate_model, train_model


def create_pipeline(**kwargs: Any) -> Pipeline:  # noqa: ARG001
    """Wire up the refit DAG -- same functions, all data, with calibration."""
    return pipeline(
        [
            node(
                func=fit_encoders,
                inputs=[
                    "split_data",
                    "params:columns",
                    "params:refit_fit_transform",
                ],
                outputs="production_encoders",
                name="refit_encoders_node",
            ),
            node(
                func=transform_encoders,
                inputs=["split_data", "production_encoders"],
                outputs="production_encoded_data",
                name="refit_transform_encoders_node",
            ),
            node(
                func=fit_scalers,
                inputs=[
                    "production_encoded_data",
                    "params:columns",
                    "params:refit_fit_transform",
                ],
                outputs="production_scalers",
                name="refit_scalers_node",
            ),
            node(
                func=transform_scalers,
                inputs=["production_encoded_data", "production_scalers"],
                outputs="production_master_table",
                name="refit_transform_scalers_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "production_master_table",
                    "params:columns",
                    "best_model_config",
                ],
                outputs="raw_production_model",
                name="refit_model_node",
            ),
            node(
                func=calibrate_model,
                inputs=[
                    "production_master_table",
                    "raw_production_model",
                    "params:columns",
                    "params:calibration",
                ],
                outputs="production_model",
                name="calibrate_model_node",
            ),
        ]
    )
