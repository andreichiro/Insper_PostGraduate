"""Diagnostics pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.diagnostics.nodes import run_residual_diagnostics


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=run_residual_diagnostics,
                inputs=[
                    "target_strategies",
                    "model_selection",
                    "folds_metadata",
                    "params:project",
                    "params:validation",
                    "params:models",
                    "params:outputs",
                ],
                outputs=[
                    "residual_diagnostics",
                    "interval_coverage_proxy",
                    "interval_validation_predictions",
                ],
                name="run_residual_diagnostics",
            )
        ]
    )
