"""EDA pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.eda.nodes import generate_eda_outputs


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=generate_eda_outputs,
                inputs=["canonical_panel", "target_strategies", "params:data", "params:outputs"],
                outputs=["eda_summary", "stationarity_tests"],
                name="generate_eda_outputs",
            )
        ]
    )
