"""Old-data gate pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.old_data_gate.nodes import run_old_data_gate


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=run_old_data_gate,
                inputs=[
                    "target_strategies",
                    "folds_metadata",
                    "params:project",
                    "params:validation",
                    "params:models",
                    "params:selection",
                ],
                outputs="old_data_gate",
                name="run_old_data_gate",
            )
        ]
    )
