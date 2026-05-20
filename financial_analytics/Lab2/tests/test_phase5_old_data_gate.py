import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pads_forecasting.modeling import model_specs
from pads_forecasting.pipelines.old_data_gate.nodes import (
    FOLD_METRIC_COLUMNS,
    _log_old_data_gate_to_mlflow,
    run_old_data_gate,
)
from pads_forecasting.pipelines.validation.nodes import build_folds_metadata
from pads_forecasting.selection import old_data_gate_decision


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _strategy_frame(name: str, *, pre_adjustment: float, alpha: float = 1.0) -> pd.DataFrame:
    dates = pd.Series(pd.date_range("2014-01-01", "2023-12-01", freq="MS"))
    trend = np.arange(len(dates), dtype=float) * 0.75
    seasonal = 12.0 * np.sin(2 * np.pi * dates.dt.month / 12)
    observed_consolidated = 100.0 + trend + seasonal + 24.0
    pre_mask = dates < pd.Timestamp("2019-07-01")
    y = observed_consolidated.copy()
    y.loc[pre_mask] = 100.0 + trend[pre_mask] + seasonal[pre_mask] + pre_adjustment
    frame = pd.DataFrame(
        {
            "data": dates,
            "y": y,
            "target_strategy": name,
            "strategy_family": name,
            "alpha": alpha,
            "beta": 1.0,
            "target_source": np.where(
                pre_mask,
                "reconstructed_or_raw_pre_acquisition",
                "observed_consolidated",
            ),
            "covid_shock": np.where(
                (dates >= "2020-03-01") & (dates <= "2020-06-01"),
                1,
                0,
            ),
            "covid_recovery": np.where(
                (dates >= "2020-07-01") & (dates <= "2020-12-01"),
                1,
                0,
            ),
            "month": dates.dt.month,
            "trend_index": np.arange(len(dates), dtype=int),
        }
    )
    if name == "post_only":
        return frame.loc[~pre_mask].reset_index(drop=True)
    return frame.reset_index(drop=True)


def _target_strategies() -> dict:
    return {
        "selected_alpha": 1.05,
        "strategies": {
            "post_only": _strategy_frame("post_only", pre_adjustment=24.0),
            "raw_full": _strategy_frame("raw_full", pre_adjustment=0.0),
            "proforma_sum": _strategy_frame("proforma_sum", pre_adjustment=24.0),
            "calibrated_alpha": _strategy_frame(
                "calibrated_alpha",
                pre_adjustment=25.0,
                alpha=1.05,
            ),
        },
    }


def test_phase5_old_data_gate_outputs_fold_summary_and_decision_records():
    project = {"seed": 42}
    validation = _load_yaml("conf/base/parameters_validation.yml")["validation"]
    selection = _load_yaml("conf/base/parameters_validation.yml")["selection"]
    models = _load_yaml("conf/base/parameters_models.yml")
    folds = build_folds_metadata(validation)
    specs = model_specs(models, stage="old_data_gate", include_optional=False)

    gate = run_old_data_gate(
        _target_strategies(),
        folds,
        project,
        validation,
        models,
        selection,
    )
    fold_rows = gate[gate["record_type"].eq("fold")]
    summary_rows = gate[gate["record_type"].eq("summary")]
    decision_rows = gate[gate["record_type"].eq("decision")]

    assert set(gate["record_type"]) == {"fold", "summary", "decision"}
    assert len(fold_rows) == 4 * len(specs) * len(folds)
    assert set(fold_rows["target_strategy"]) == {
        "post_only",
        "raw_full",
        "proforma_sum",
        "calibrated_alpha",
    }
    assert set(fold_rows["model_family"]) == {
        "seasonal_naive",
        "ets",
        "ridge",
        "elasticnet",
        "lightgbm",
    }
    assert not {"sarimax", "prophet", "bvar"} & set(fold_rows["model_family"])
    assert set(decision_rows["target_strategy"]) == {"proforma_sum", "calibrated_alpha"}
    assert not summary_rows.empty
    assert fold_rows["status"].eq("ok").all()
    assert set(FOLD_METRIC_COLUMNS).issubset(fold_rows.columns)
    assert "model_params" in fold_rows.columns
    assert any(col.startswith("model_param_") for col in fold_rows.columns)
    assert {"alpha", "beta", "horizon"}.issubset(fold_rows.columns)
    assert "mean_train_residual_abs_mean" in summary_rows.columns
    assert "residual_diagnostics_no_worse_than_post_only" in decision_rows.columns
    assert gate["selected_alpha"].eq(1.05).all()


def test_phase5_old_data_gate_rejects_materially_worse_residual_diagnostics():
    summary = pd.DataFrame(
        [
            {
                "target_strategy": "post_only",
                "normal_mean_mase": 1.0,
                "mean_mase": 1.0,
                "cv_mase": 0.1,
                "mean_relative_mae_vs_seasonal_naive": 0.9,
                "mean_train_valid_ratio": 1.1,
                "mean_train_residual_abs_mean": 4.0,
            },
            {
                "target_strategy": "raw_full",
                "normal_mean_mase": 0.9,
                "mean_mase": 0.9,
                "cv_mase": 0.1,
                "mean_relative_mae_vs_seasonal_naive": 0.9,
                "mean_train_valid_ratio": 1.1,
                "mean_train_residual_abs_mean": 4.0,
            },
            {
                "target_strategy": "proforma_sum",
                "normal_mean_mase": 0.7,
                "mean_mase": 0.7,
                "cv_mase": 0.1,
                "mean_relative_mae_vs_seasonal_naive": 0.8,
                "mean_train_valid_ratio": 1.1,
                "mean_train_residual_abs_mean": 8.0,
            },
        ]
    )

    decision = old_data_gate_decision(
        summary,
        {
            "old_data_min_improvement_pct": 3.0,
            "max_cv_mase": 0.40,
            "train_valid_ratio_reject": 3.0,
            "residual_diagnostics_max_ratio": 1.25,
        },
    )
    proforma = decision[decision["target_strategy"].eq("proforma_sum")].iloc[0]

    assert bool(proforma["passed"]) is False
    assert bool(proforma["residual_diagnostics_no_worse_than_post_only"]) is False
    assert proforma["residual_ratio_vs_post_only"] == 2.0


class _FakeRun:
    def __init__(self, mlflow, run: dict) -> None:
        self.mlflow = mlflow
        self.run = run
        self.previous = None

    def __enter__(self):
        self.previous = self.mlflow.current
        self.mlflow.current = self.run
        self.mlflow.runs.append(self.run)
        return self.run

    def __exit__(self, exc_type, exc, tb):
        self.mlflow.current = self.previous
        return False


class _FakeMlflow(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.current = None
        self.runs: list[dict] = []
        self.parent_params: dict[str, object] = {}
        self.parent_metrics: dict[str, float] = {}
        self.artifacts: list[str] = []

    def active_run(self):
        return object()

    def start_run(self, run_name: str, nested: bool = False):
        return _FakeRun(
            self,
            {
                "run_name": run_name,
                "nested": nested,
                "params": {},
                "metrics": {},
            },
        )

    def log_param(self, key: str, value: object) -> None:
        if self.current is None:
            self.parent_params[key] = value
        else:
            self.current["params"][key] = value

    def log_metric(self, key: str, value: float) -> None:
        if self.current is None:
            self.parent_metrics[key] = value
        else:
            self.current["metrics"][key] = value

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(Path(path).name)


def test_phase5_old_data_gate_logs_nested_mlflow_runs(monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    fold_results = pd.DataFrame(
        [
            {
                "stage": "old_data_gate",
                "target_strategy": "proforma_sum",
                "model_id": "seasonal_naive",
                "model_family": "seasonal_naive",
                "model_params": '{"season_length": 12}',
                "model_param_season_length": 12,
                "covid_mode": "none",
                "complexity": "simple",
                "fold_name": "fold_2022_normal",
                "fold_role": "normal",
                "train_end": "2021-12-01",
                "valid_start": "2022-01-01",
                "valid_end": "2022-12-01",
                "horizon": 12,
                "status": "ok",
                "alpha": 1.0,
                "beta": 1.0,
                "mae": 10.0,
                "rmse": 12.0,
                "mase": 0.8,
                "relative_mae_vs_seasonal_naive": 0.9,
                "train_valid_mae_ratio": 1.2,
                "train_residual_abs_mean": 4.0,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "stage": "old_data_gate",
                "target_strategy": "proforma_sum",
                "model_id": "seasonal_naive",
                "model_family": "seasonal_naive",
                "model_params": '{"season_length": 12}',
                "model_param_season_length": 12,
                "covid_mode": "none",
                "complexity": "simple",
                "mean_mase": 0.8,
                "normal_mean_mase": 0.8,
                "cv_mase": 0.0,
                "mean_train_residual_abs_mean": 4.0,
                "folds": 1,
            }
        ]
    )
    decision = pd.DataFrame(
        [
            {
                "target_strategy": "proforma_sum",
                "passed": True,
                "normal_mean_mase": 0.8,
                "improvement_vs_post_only_pct": 5.0,
                "cv_mase": 0.0,
                "train_valid_ratio": 1.2,
                "residual_ratio_vs_post_only": 0.9,
            }
        ]
    )

    gate_table = pd.concat(
        [
            decision.assign(record_type="decision"),
            summary.assign(record_type="summary"),
            fold_results.assign(record_type="fold"),
        ],
        ignore_index=True,
        sort=False,
    )

    _log_old_data_gate_to_mlflow(
        fold_results,
        summary,
        decision,
        selected_alpha=1.0,
        gate_table=gate_table,
        selection_params={
            "old_data_min_improvement_pct": 3.0,
            "train_valid_ratio_reject": 3.0,
        },
    )

    assert (
        fake_mlflow.parent_params["old_data_gate.selection.old_data_min_improvement_pct"] == "3.0"
    )
    assert fake_mlflow.parent_metrics["old_data_gate.proforma_sum.passed"] == 1.0
    assert fake_mlflow.artifacts == ["old_data_gate.parquet"]
    assert {
        "old_data_gate/proforma_sum/seasonal_naive/summary",
        "old_data_gate/proforma_sum/seasonal_naive/fold_2022_normal",
    } == {run["run_name"] for run in fake_mlflow.runs}
    fold_run = next(
        run for run in fake_mlflow.runs if run["run_name"].endswith("/fold_2022_normal")
    )
    assert fold_run["nested"] is True
    assert fold_run["params"]["target_strategy"] == "proforma_sum"
    assert fold_run["params"]["selected_alpha"] == "1.0"
    assert fold_run["params"]["model_param_season_length"] == "12"
    assert fold_run["params"]["beta"] == "1.0"
    assert fold_run["params"]["horizon"] == "12"
    assert fold_run["metrics"]["mase"] == 0.8
