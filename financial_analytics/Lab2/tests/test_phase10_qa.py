from pathlib import Path

import pandas as pd
import yaml

from pads_forecasting.pipelines import reporting
from pads_forecasting.pipelines.reporting import nodes as reporting_nodes

ROOT = Path(__file__).resolve().parents[1]


def _frame(columns: list[str], rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{column: 1 for column in columns} for _ in range(rows)])


def _final_mlflow_artifact_dir(root: Path) -> Path:
    artifact_dir = root / "mlruns/1/run-id/artifacts"
    artifact_dir.mkdir(parents=True)
    for name in reporting_nodes.FINAL_MLFLOW_ARTIFACTS:
        (artifact_dir / name).write_text("ok", encoding="utf-8")
    return artifact_dir


def test_phase10_final_mlflow_artifact_bundle_requires_one_complete_run(tmp_path):
    assert not reporting_nodes._final_mlflow_artifacts_present(tmp_path / "mlruns")

    incomplete = tmp_path / "mlruns/1/incomplete/artifacts"
    incomplete.mkdir(parents=True)
    (incomplete / "forecast_intervals.parquet").write_text("ok", encoding="utf-8")
    assert not reporting_nodes._final_mlflow_artifacts_present(tmp_path / "mlruns")

    _final_mlflow_artifact_dir(tmp_path)
    assert reporting_nodes._final_mlflow_artifacts_present(tmp_path / "mlruns")


def test_phase10_qa_environment_keeps_all_model_lanes_but_is_artifact_separated():
    with (ROOT / "conf/qa/parameters_models.yml").open(encoding="utf-8") as handle:
        models_config = yaml.safe_load(handle)
    with (ROOT / "conf/qa/parameters_outputs.yml").open(encoding="utf-8") as handle:
        outputs_config = yaml.safe_load(handle)
    with (ROOT / "conf/qa/catalog.yml").open(encoding="utf-8") as handle:
        catalog_config = yaml.safe_load(handle)

    enabled_lanes = {
        name
        for name, config in models_config["models"].items()
        if isinstance(config, dict) and config.get("enabled")
    }
    assert enabled_lanes == {
        "seasonal_naive",
        "ets",
        "sarimax",
        "prophet",
        "lightgbm",
        "ridge",
        "elasticnet",
        "bvar",
    }
    assert outputs_config["outputs"]["mlruns_dir"] == "mlruns_qa"
    assert outputs_config["outputs"]["html_report_path"].startswith("data/99_qa/")
    assert outputs_config["outputs"]["decision_html_report_path"].startswith("data/99_qa/")
    assert "notebook_path" not in outputs_config["outputs"]
    assert catalog_config["forecast_intervals"]["filepath"].startswith("data/99_qa/")
    assert catalog_config["forecast_intervals"]["filepath"].endswith(".parquet")
    assert catalog_config["previsao"]["filepath"] == "outputs/previsao_qa.csv"
    assert catalog_config["shap_feature_importance"]["filepath"].startswith("data/99_qa/")
    assert catalog_config["shap_feature_importance"]["filepath"].endswith(".parquet")
    assert catalog_config["html_report"]["filepath"].endswith("pads_forecasting_report.html")
    assert catalog_config["decision_html_report"]["filepath"].endswith("pads_decision_report.html")


def test_phase10_reporting_is_artifact_only_and_has_no_notebook_requirement(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reporting_nodes, "_log_to_mlflow", lambda **kwargs: None)
    _final_mlflow_artifact_dir(tmp_path)

    forecast_intervals = pd.DataFrame(
        {
            "data": pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d"),
            "previsao": range(12),
            "lo_80": range(12),
            "hi_80": range(12),
            "lo_95": range(12),
            "hi_95": range(12),
        }
    )
    previsao = forecast_intervals[["data", "previsao"]].copy()
    html_report_path = tmp_path / "data/08_reporting/pads_forecasting_report.html"
    html_report_path.parent.mkdir(parents=True)
    html_report_path.write_text(
        "PADS Forecasting Explainability SHAP Common MASE",
        encoding="utf-8",
    )
    decision_html_report_path = tmp_path / "data/08_reporting/pads_decision_report.html"
    decision_html_report_path.write_text(
        "PADS Forecasting Decisoes para MASE 2024 O que foi descartado ajuste COVID",
        encoding="utf-8",
    )
    shap_feature_importance = pd.DataFrame(
        [
            {
                "status": "ok",
                "feature": "lag_12",
                "mean_abs_shap": 10.0,
                "rank": 1,
                "method": "tree_shap_lightgbm",
            }
        ]
    )
    shap_values_sample = pd.DataFrame(
        [
            {
                "status": "ok",
                "feature": "lag_12",
                "feature_value": 400.0,
                "shap_value": 10.0,
            }
        ]
    )
    cv_summary = _frame(
        [
            "normal_mean_mase",
            "mean_mase",
            "mean_common_mase",
            "normal_mean_common_mase",
            "cv_common_mase",
            "covid_mode",
        ],
        rows=2,
    )
    cv_summary["covid_mode"] = ["none", "adjusted_target"]
    checklist = reporting_nodes.build_assignment_checklist(
        data_validation=_frame(["check", "passed"]),
        eda_summary=_frame(["metric", "value"]),
        stationarity_tests=pd.DataFrame({"test": ["ADF", "KPSS"]}),
        old_data_gate=_frame(["record_type"]),
        cv_summary=cv_summary,
        horizon_metrics=_frame(["horizon_index", "common_mase"]),
        horizon_summary=_frame(["horizon_index", "horizon_common_mase"]),
        mase_uncertainty=_frame(
            [
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_probability_beats_seasonal_naive",
            ]
        ),
        nested_selection_audit=_frame(["selection_scope", "inner_fold_count", "outer_common_mase"]),
        nested_cv_results=_frame(
            [
                "selection_scope",
                "inner_candidate_count",
                "inner_mean_common_mase",
                "outer_common_mase",
            ]
        ),
        nested_cv_summary=_frame(
            ["summary_scope", "nested_outer_folds", "nested_mean_common_mase"]
        ),
        rolling_origin_robustness=_frame(["mean_common_mase", "robustness_fold_count"]),
        robust_alpha_results=_frame(
            [
                "model_family",
                "selected_alpha",
                "inner_mean_common_mase",
                "outer_common_mase",
                "alpha_beats_one_by_common_mase",
            ]
        ),
        robust_alpha_summary=_frame(
            [
                "model_family",
                "selected_alpha_mode",
                "best_alpha_normal_folds_only_grid",
                "best_alpha_stress_downweighted_grid",
                "mean_common_mase_delta_vs_alpha_one",
            ]
        ),
        selection_objective_audit=_frame(
            ["objective", "objective_metric", "target_strategy", "model_id"]
        ),
        covid_adjustment_coefficients=_frame(
            [
                "covid_beta_covid_shock",
                "covid_beta_covid_recovery",
                "covid_beta_covid_aftershock_2021",
            ]
        ),
        covid_adjustment_audit=pd.DataFrame(
            [
                {
                    "train_end_before_valid_start": True,
                    "future_covid_assumed_zero": True,
                }
            ]
        ),
        covid_mode_comparison=_frame(
            ["target_strategy", "model_family", "adjusted_improvement_pct"]
        ),
        residual_diagnostics=_frame(["residual_mean"]),
        interval_coverage_proxy=_frame(
            ["coverage_80", "coverage_95", "mean_width_80", "mean_width_95"]
        ),
        interval_validation_predictions=_frame(
            [
                "data",
                "y_true",
                "yhat",
                "lo_80",
                "hi_80",
                "lo_95",
                "hi_95",
                "covered_80",
                "covered_95",
            ]
        ),
        forecast_intervals=forecast_intervals,
        challenger_forecasts=_frame(
            ["candidate_role", "data", "previsao", "lo_80", "hi_80", "lo_95", "hi_95"]
        ),
        final_model_metadata=_frame(
            [
                "selected_target_strategy",
                "selected_final_alpha",
                "selected_final_alpha_source",
                "selected_model_id",
                "train_start",
                "train_end",
                "forecast_start",
                "forecast_end",
                "future_covid_shock_sum",
                "future_covid_recovery_sum",
            ]
        ),
        shap_feature_importance=shap_feature_importance,
        shap_values_sample=shap_values_sample,
        decision_html_report=(
            "PADS Forecasting Decisoes para MASE 2024 O que foi descartado ajuste COVID"
        ),
        html_report="PADS Forecasting Explainability SHAP Common MASE",
        previsao=previsao,
        project={"run_id": "test", "seed": 42},
        outputs={
            "reporting_dir": "data/08_reporting",
            "figures_dir": "data/08_reporting/figures",
            "previsao_path": "outputs/previsao.csv",
            "forecast_intervals_path": "data/08_reporting/forecast_intervals.parquet",
            "mlruns_dir": "mlruns",
            "html_report_path": "data/08_reporting/pads_forecasting_report.html",
            "decision_html_report_path": "data/08_reporting/pads_decision_report.html",
        },
    )

    requirements = dict(zip(checklist["requirement"], checklist["passed"], strict=True))
    assert "notebook_shell_created" not in requirements
    assert requirements["no_hidden_notebook_decisions"]
    assert requirements["mlflow_final_artifacts_complete"]
    assert requirements["shap_explainability_complete"]
    assert requirements["decision_html_report_complete"]
    assert requirements["html_report_complete"]
    assert requirements["fixed_target_mase_complete"]
    assert requirements["horizon_mase_complete"]
    assert requirements["mase_uncertainty_complete"]
    assert requirements["nested_selection_audit_complete"]
    assert requirements["formal_nested_cv_complete"]
    assert requirements["rolling_origin_robustness_complete"]
    assert requirements["robust_alpha_model_family_complete"]
    assert requirements["selection_objective_audit_complete"]
    assert requirements["covid_adjusted_target_fairness_complete"]
    assert requirements["covid_adjustment_coefficients_complete"]
    assert requirements["covid_adjustment_no_leakage_audit_complete"]
    assert checklist["passed"].all()
    assert not list(Path("notebooks").glob("*.ipynb"))
    assert hasattr(reporting, "__name__")
