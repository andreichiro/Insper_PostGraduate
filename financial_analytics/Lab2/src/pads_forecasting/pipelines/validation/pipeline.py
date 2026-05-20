"""Validation pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.validation.nodes import build_folds_metadata


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=build_folds_metadata,
                inputs="params:validation",
                outputs="folds_metadata",
                name="build_folds_metadata",
            )
        ]
    )
