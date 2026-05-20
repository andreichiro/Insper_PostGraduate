"""Reconstruction pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.reconstruction.nodes import build_target_strategies


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=build_target_strategies,
                inputs=[
                    "canonical_panel",
                    "params:data",
                    "params:reconstruction",
                    "params:validation",
                    "params:outputs",
                ],
                outputs=[
                    "target_strategies",
                    "target_strategy_summary",
                    "alpha_sensitivity",
                    "leave_one_fold_alpha",
                ],
                name="build_target_strategies",
            )
        ]
    )
