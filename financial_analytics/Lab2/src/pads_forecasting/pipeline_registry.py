"""Project pipeline registry."""

from kedro.pipeline import Pipeline

from pads_forecasting.pipelines.data_engineering.pipeline import (
    create_pipeline as create_data_engineering_pipeline,
)
from pads_forecasting.pipelines.diagnostics.pipeline import (
    create_pipeline as create_diagnostics_pipeline,
)
from pads_forecasting.pipelines.eda.pipeline import create_pipeline as create_eda_pipeline
from pads_forecasting.pipelines.final_forecast.pipeline import (
    create_pipeline as create_final_forecast_pipeline,
)
from pads_forecasting.pipelines.model_comparison.pipeline import (
    create_pipeline as create_model_comparison_pipeline,
)
from pads_forecasting.pipelines.old_data_gate.pipeline import (
    create_pipeline as create_old_data_gate_pipeline,
)
from pads_forecasting.pipelines.reconstruction.pipeline import (
    create_pipeline as create_reconstruction_pipeline,
)
from pads_forecasting.pipelines.reporting.pipeline import (
    create_pipeline as create_reporting_pipeline,
)
from pads_forecasting.pipelines.validation.pipeline import (
    create_pipeline as create_validation_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register Kedro pipelines."""

    data_engineering = create_data_engineering_pipeline()
    reconstruction = create_reconstruction_pipeline()
    eda = create_eda_pipeline()
    validation = create_validation_pipeline()
    old_data_gate = create_old_data_gate_pipeline()
    model_comparison = create_model_comparison_pipeline()
    diagnostics = create_diagnostics_pipeline()
    final_forecast = create_final_forecast_pipeline()
    reporting = create_reporting_pipeline()

    full = (
        data_engineering
        + reconstruction
        + eda
        + validation
        + old_data_gate
        + model_comparison
        + diagnostics
        + final_forecast
        + reporting
    )

    old_data_gate_flow = data_engineering + reconstruction + validation + old_data_gate
    model_comparison_flow = old_data_gate_flow + model_comparison
    final_forecast_flow = model_comparison_flow + diagnostics + final_forecast

    return {
        "__default__": full,
        "full": full,
        "data_engineering": data_engineering,
        "reconstruction": reconstruction,
        "eda": eda,
        "validation": validation,
        "old_data_gate": old_data_gate_flow,
        "old_data_gate_only": old_data_gate,
        "model_comparison": model_comparison_flow,
        "model_comparison_only": model_comparison,
        "diagnostics": diagnostics,
        "final_forecast": final_forecast_flow,
        "final_forecast_only": final_forecast,
        "reporting": full,
        "reporting_only": reporting,
    }
