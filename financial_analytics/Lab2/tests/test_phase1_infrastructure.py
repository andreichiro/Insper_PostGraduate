from pathlib import Path

import yaml

from pads_forecasting.pipeline_registry import register_pipelines
from pads_forecasting.schemas import validate_parameter_groups

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "conf/base"


def _load_yaml(name: str) -> dict:
    with (CONF / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_all_parameter_groups_have_pydantic_contracts():
    base = _load_yaml("parameters.yml")
    models = _load_yaml("parameters_models.yml")
    validation = _load_yaml("parameters_validation.yml")
    outputs = _load_yaml("parameters_outputs.yml")

    validate_parameter_groups(
        base["project"],
        base["data"],
        base["interventions"],
        base["reconstruction"],
        validation["validation"],
        validation["selection"],
        outputs["outputs"],
        models["models"],
        models["hpo"],
        validation["metrics"],
    )

    assert "mlruns_dir" in outputs["outputs"]
    assert "html_report_path" in outputs["outputs"]
    assert "decision_html_report_path" in outputs["outputs"]
    assert "notebook_path" not in outputs["outputs"]


def test_catalog_declares_phase1_core_datasets():
    catalog = _load_yaml("catalog.yml")

    expected = {
        "main_raw",
        "acquired_raw",
        "canonical_panel",
        "data_validation",
        "target_strategies",
        "folds_metadata",
        "old_data_gate",
        "cv_fold_results",
        "cv_summary",
        "train_valid_gap",
        "model_selection",
        "residual_diagnostics",
        "interval_coverage_proxy",
        "interval_validation_predictions",
        "forecast_intervals",
        "challenger_forecasts",
        "final_model_metadata",
        "previsao",
        "assignment_checklist",
        "shap_feature_importance",
        "shap_values_sample",
        "html_report",
        "decision_html_report",
    }

    assert expected.issubset(catalog)


def test_mlflow_config_declares_local_tracking_and_experiment():
    mlflow_config = _load_yaml("mlflow.yml")

    assert mlflow_config["server"]["mlflow_tracking_uri"] == "mlruns"
    assert mlflow_config["tracking"]["experiment"]["name"] == "pads_forecasting_lab2"
    assert mlflow_config["tracking"]["run"]["nested"] is True


def test_pipeline_registry_declares_all_phase1_pipelines_with_nodes():
    pipelines = register_pipelines()
    expected = {
        "full",
        "data_engineering",
        "reconstruction",
        "eda",
        "validation",
        "old_data_gate",
        "model_comparison",
        "diagnostics",
        "final_forecast",
        "reporting",
    }

    assert expected.issubset(pipelines)
    for name in expected:
        assert len(pipelines[name].nodes) > 0


def test_reporting_pipeline_consumes_phase7_interval_coverage_proxy():
    reporting_inputs = register_pipelines()["reporting_only"].inputs()

    assert "interval_coverage_proxy" in reporting_inputs
    assert "interval_validation_predictions" in reporting_inputs
    assert "challenger_forecasts" in reporting_inputs
    assert "final_model_metadata" in reporting_inputs
    assert "target_strategies" in reporting_inputs
    assert "model_selection" in reporting_inputs
