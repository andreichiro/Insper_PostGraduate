"""Final forecast pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.final_forecast.nodes import run_final_forecast


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=run_final_forecast,
                inputs=[
                    "target_strategies",
                    "model_selection",
                    "robust_alpha_summary",
                    "params:project",
                    "params:data",
                    "params:validation",
                    "params:interventions",
                    "params:models",
                    "params:outputs",
                ],
                outputs=[
                    "forecast_intervals",
                    "previsao",
                    "challenger_forecasts",
                    "final_model_metadata",
                ],
                name="run_final_forecast",
            )
        ]
    )
