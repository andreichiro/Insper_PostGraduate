"""Modelling pipeline: optimize 3 models → evaluate → select best → test report."""

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_all_on_test,
    evaluate_model,
    optimize_model,
    select_best_model,
)


def create_pipeline(**kwargs: Any) -> Pipeline:  # noqa: ARG001
    """Wire up optimize/evaluate/select/test-report DAG."""
    return pipeline(
        [
            # ── Optimize all models with Optuna ─────────────────────────
            node(
                func=optimize_model,
                inputs=["master_table", "params:columns", "params:baseline"],
                outputs="baseline_model",
                name="optimize_baseline_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "master_table",
                    "baseline_model",
                    "params:columns",
                    "params:evaluation",
                ],
                outputs="baseline_metrics",
                name="evaluate_baseline_node",
            ),
            node(
                func=optimize_model,
                inputs=[
                    "master_table",
                    "params:columns",
                    "params:optimization",
                ],
                outputs="optimized_model",
                name="optimize_catboost_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "master_table",
                    "optimized_model",
                    "params:columns",
                    "params:evaluation",
                ],
                outputs="optimized_metrics",
                name="evaluate_optimized_node",
            ),
            node(
                func=optimize_model,
                inputs=["master_table", "params:columns", "params:xgboost"],
                outputs="xgboost_model",
                name="optimize_xgboost_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "master_table",
                    "xgboost_model",
                    "params:columns",
                    "params:evaluation",
                ],
                outputs="xgboost_metrics",
                name="evaluate_xgboost_node",
            ),
            # ── Select best model ───────────────────────────────────────
            node(
                func=select_best_model,
                inputs=[
                    "baseline_model",
                    "baseline_metrics",
                    "optimized_model",
                    "optimized_metrics",
                    "xgboost_model",
                    "xgboost_metrics",
                    "params:model_selection",
                ],
                outputs="best_model_config",
                name="select_best_model_node",
            ),
            # ── Final held-out test evaluation ──────────────────────────
            node(
                func=evaluate_all_on_test,
                inputs=[
                    "master_table",
                    "baseline_model",
                    "optimized_model",
                    "xgboost_model",
                    "params:columns",
                ],
                outputs="test_report",
                name="test_evaluation_node",
            ),
        ]
    )
