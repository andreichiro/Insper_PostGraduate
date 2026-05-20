from pathlib import Path

import pandas as pd

from pads_forecasting.pipeline_registry import register_pipelines

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_registry_contains_composite_commands():
    pipelines = register_pipelines()

    assert "full" in pipelines
    assert "old_data_gate" in pipelines
    assert "model_comparison" in pipelines
    assert "final_forecast" in pipelines


def test_previsao_shape_if_pipeline_has_run():
    path = ROOT / "outputs/previsao.csv"
    if not path.exists():
        return
    previsao = pd.read_csv(path)

    assert list(previsao.columns) == ["data", "previsao"]
    assert len(previsao) == 12


def test_forecast_intervals_shape_if_pipeline_has_run():
    path = ROOT / "data/08_reporting/forecast_intervals.parquet"
    if not path.exists():
        return
    forecast = pd.read_parquet(path)

    assert list(forecast.columns) == ["data", "previsao", "lo_80", "hi_80", "lo_95", "hi_95"]
    assert len(forecast) == 12


def test_html_and_shap_artifacts_if_pipeline_has_run():
    report = ROOT / "data/08_reporting/pads_forecasting_report.html"
    decision_report = ROOT / "data/08_reporting/pads_decision_report.html"
    shap_importance = ROOT / "data/08_reporting/shap_feature_importance.parquet"
    shap_values = ROOT / "data/08_reporting/shap_values_sample.parquet"
    if (
        not report.exists()
        or not decision_report.exists()
        or not shap_importance.exists()
        or not shap_values.exists()
    ):
        return

    html = report.read_text(encoding="utf-8")
    decision_html = decision_report.read_text(encoding="utf-8")
    importance = pd.read_parquet(shap_importance)
    values = pd.read_parquet(shap_values)

    assert "PADS Forecasting" in html
    assert "Explainability SHAP" in html
    assert "Decisoes para MASE 2024" in decision_html
    assert "O que foi descartado" in decision_html
    assert {"feature", "mean_abs_shap", "rank", "method"}.issubset(importance.columns)
    assert importance["status"].eq("ok").any()
    assert {"feature", "feature_value", "shap_value"}.issubset(values.columns)
    assert not values.empty
