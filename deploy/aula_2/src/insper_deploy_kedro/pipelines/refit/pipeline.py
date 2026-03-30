"""Pipeline de refit: re-fita encoders/scalers/modelo com TODOS os dados pra produção.

Reutiliza nodes de DE e modelagem com split_to_fit: [train, validation, test].
Add calibração de probabilidade como pós-processamento
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


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Monta o DAG de refit: mesmas funções, todos os dados, c/ calibração."""
    return pipeline(
        [
            node(
                func=fit_encoders,
                inputs=[
                    "split_data",
                    "params:columns",
                    "params:refit_fit_transform",
                    "params:preprocessing",
                ],
                outputs="production_encoders",
                name="refit_encoders_node",
                tags=["refit", "encoding"],
            ),
            node(
                func=transform_encoders,
                inputs=["split_data", "production_encoders"],
                outputs="production_encoded_data",
                name="refit_transform_encoders_node",
                tags=["refit", "encoding"],
            ),
            node(
                func=fit_scalers,
                inputs=[
                    "production_encoded_data",
                    "params:columns",
                    "params:refit_fit_transform",
                    "params:preprocessing",
                ],
                outputs="production_scalers",
                name="refit_scalers_node",
                tags=["refit", "scaling"],
            ),
            node(
                func=transform_scalers,
                inputs=["production_encoded_data", "production_scalers"],
                outputs="production_master_table",
                name="refit_transform_scalers_node",
                tags=["refit", "scaling"],
            ),
            node(
                func=train_model,
                inputs=[
                    "production_master_table",
                    "params:columns",
                    "best_model_config",
                    "params:ml_runtime",
                ],
                outputs="raw_production_model",
                name="refit_model_node",
                tags=["refit", "training"],
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
                tags=["refit", "calibration"],
            ),
        ]
    )
