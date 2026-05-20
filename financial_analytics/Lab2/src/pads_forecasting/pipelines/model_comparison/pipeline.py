"""Model comparison pipeline definition."""

from kedro.pipeline import node, pipeline

from pads_forecasting.pipelines.model_comparison.nodes import run_model_comparison


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=run_model_comparison,
                inputs=[
                    "target_strategies",
                    "folds_metadata",
                    "old_data_gate",
                    "params:project",
                    "params:validation",
                    "params:models",
                    "params:selection",
                ],
                outputs=[
                    "cv_fold_results",
                    "cv_summary",
                    "train_valid_gap",
                    "model_selection",
                    "horizon_metrics",
                    "horizon_summary",
                    "mase_uncertainty",
                    "nested_selection_audit",
                    "nested_cv_results",
                    "nested_cv_summary",
                    "rolling_origin_robustness",
                    "robust_alpha_results",
                    "robust_alpha_summary",
                    "selection_objective_audit",
                    "covid_adjustment_coefficients",
                    "covid_adjustment_audit",
                    "covid_mode_comparison",
                ],
                name="run_model_comparison",
            )
        ]
    )
