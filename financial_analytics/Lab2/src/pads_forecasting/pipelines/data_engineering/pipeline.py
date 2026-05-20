"""Data engineering pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.data_engineering.nodes import build_canonical_panel


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=build_canonical_panel,
                inputs=[
                    "main_raw",
                    "acquired_raw",
                    "params:project",
                    "params:data",
                    "params:interventions",
                    "params:reconstruction",
                    "params:validation",
                    "params:selection",
                    "params:outputs",
                    "params:models",
                    "params:hpo",
                    "params:metrics",
                ],
                outputs=["canonical_panel", "data_validation"],
                name="build_canonical_panel",
            )
        ]
    )
