import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pads_forecasting.modeling import model_specs
from pads_forecasting.pipelines.model_comparison.nodes import (
    ARTIFACT_NAMES,
    FOLD_METRIC_COLUMNS,
    MLFLOW_MAX_FOLD_NESTED_CANDIDATES,
    MLFLOW_MAX_SUMMARY_NESTED_RUNS,
    SUMMARY_METRIC_COLUMNS,
    TRAIN_VALID_GAP_COLUMNS,
    run_model_comparison,
)
from pads_forecasting.pipelines.validation.nodes import build_folds_metadata


def _reduced_all_lane_models() -> dict[str, Any]:
    return {
        "models": {
            "seasonal_naive": {"enabled": True, "season_length": 12},
            "ets": {
                "enabled": True,
                "grid": {
                    "trend": ["add"],
                    "seasonal": ["add"],
                    "damped_trend": [False],
                    "use_boxcox": [False],
                },
            },
            "sarimax": {
                "enabled": True,
                "grid": {
                    "p": [0],
                    "d": [1],
                    "q": [1],
                    "P": [0],
                    "D": [1],
                    "Q": [1],
                    "m": [12],
                    "trend": ["n"],
                },
                "covid_modes": ["none"],
            },
            "prophet": {
                "enabled": True,
                "optional": True,
                "grid": {
                    "yearly_seasonality": [3],
                    "seasonality_mode": ["additive"],
                    "changepoint_prior_scale": [0.05],
                    "seasonality_prior_scale": [5.0],
                    "covid_regressors": [False],
                },
            },
            "ridge": {"enabled": True, "alpha_grid": [1.0], "lags": [1, 2, 12]},
            "elasticnet": {
                "enabled": True,
                "alpha_grid": [0.1],
                "l1_ratio_grid": [0.5],
                "lags": [1, 2, 12],
            },
            "lightgbm": {
                "enabled": True,
                "grid": {
                    "lags": [[1, 2, 12]],
                    "rolling_windows": [[3]],
                    "max_depth": [2],
                    "num_leaves": [4],
                    "n_estimators": [25],
                    "learning_rate": [0.05],
                    "min_data_in_leaf": [8],
                    "lambda_l1": [0.0],
                    "lambda_l2": [0.0],
                    "forecast_strategy": ["recursive"],
                },
            },
            "bvar": {
                "enabled": True,
                "optional": True,
                "grid": {
                    "lags": [1],
                    "minnesota_lambda": [0.2],
                    "cross_lag_shrinkage": [0.1],
                    "covid_exog": [True],
                    "draws": [8],
                    "tune": [8],
                },
            },
        }
    }


def _old_data_gate_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_type": "decision",
                "target_strategy": "proforma_sum",
                "passed": True,
            },
            {
                "record_type": "decision",
                "target_strategy": "calibrated_alpha",
                "passed": True,
            },
            {
                "record_type": "decision",
                "target_strategy": "raw_full",
                "passed": False,
            },
        ]
    )


def _fold_metadata() -> pd.DataFrame:
    validation = {
        "horizon": 12,
        "season_length": 12,
        "folds": [
            {
                "name": "fold_2021_stress",
                "role": "stress",
                "train_end": "2020-12-01",
                "valid_start": "2021-01-01",
                "valid_end": "2021-12-01",
            },
            {
                "name": "fold_2022_normal",
                "role": "normal",
                "train_end": "2021-12-01",
                "valid_start": "2022-01-01",
                "valid_end": "2022-12-01",
            },
            {
                "name": "fold_2023_normal",
                "role": "normal",
                "train_end": "2022-12-01",
                "valid_start": "2023-01-01",
                "valid_end": "2023-12-01",
            },
        ],
    }
    return build_folds_metadata(validation)


def _target_strategies() -> dict[str, Any]:
    dates = pd.Series(pd.date_range("2014-01-01", "2023-12-01", freq="MS"))
    base = pd.DataFrame(
        {
            "data": dates,
            "y": 100 + np.arange(len(dates)) * 0.5,
            "target_strategy": "post_only",
            "strategy_family": "post_only",
            "alpha": 1.0,
            "beta": 1.0,
            "target_source": "observed_consolidated",
            "covid_shock": 0,
            "covid_recovery": 0,
            "month": dates.dt.month,
            "trend_index": np.arange(len(dates)),
        }
    )
    return {
        "selected_alpha": 1.05,
        "strategies": {
            "post_only": base.assign(target_strategy="post_only"),
            "raw_full": base.assign(target_strategy="raw_full"),
            "proforma_sum": base.assign(target_strategy="proforma_sum"),
            "calibrated_alpha": base.assign(target_strategy="calibrated_alpha", alpha=1.05),
        },
        "alpha_candidates": {
            1.0: base.assign(target_strategy="calibrated_alpha", alpha=1.0),
            1.5: base.assign(target_strategy="calibrated_alpha", alpha=1.5),
        },
    }


def _fake_cv_rows(
    *,
    stage: str,
    strategy_names: list[str],
    specs: list[dict[str, Any]],
    folds_metadata: pd.DataFrame,
) -> pd.DataFrame:
    alpha_grid_mode = all(str(strategy).startswith("alpha_") for strategy in strategy_names)
    if alpha_grid_mode:
        assert stage == "robust_alpha"
    else:
        assert strategy_names == ["post_only", "proforma_sum", "calibrated_alpha"]
        assert "raw_full" not in strategy_names
    assert {spec["family"] for spec in specs} == {
        "seasonal_naive",
        "ets",
        "sarimax",
        "prophet",
        "ridge",
        "elasticnet",
        "lightgbm",
        "bvar",
    } or alpha_grid_mode

    family_mase = {
        "seasonal_naive": 1.0,
        "ets": 0.88,
        "sarimax": 0.82,
        "ridge": 0.78,
        "elasticnet": 0.77,
        "prophet": 0.74,
        "bvar": 0.73,
        "lightgbm": 0.70,
    }
    strategy_offset = {"post_only": 0.03, "proforma_sum": 0.0, "calibrated_alpha": 0.005}
    fold_offset = {"stress": 0.1, "normal": 0.0}
    rows: list[dict[str, Any]] = []
    for strategy in strategy_names:
        if alpha_grid_mode:
            alpha = float(str(strategy).removeprefix("alpha_").replace("_", "."))
            strategy_offset[strategy] = 0.02 if alpha == 1.0 else -0.01
        else:
            alpha = 1.05 if strategy == "calibrated_alpha" else 1.0
        for spec in specs:
            for fold in folds_metadata.to_dict("records"):
                mase = (
                    family_mase[spec["family"]]
                    + strategy_offset[strategy]
                    + fold_offset[fold["fold_role"]]
                )
                mae = mase * 10.0
                row = {
                    "stage": stage,
                    "target_strategy": strategy,
                    "alpha": alpha,
                    "beta": 1.0,
                    "model_id": spec["model_id"],
                    "model_family": spec["family"],
                    "model_params": json.dumps(spec["params"], sort_keys=True, default=str),
                    "covid_mode": spec["covid_mode"],
                    "complexity": spec["complexity"],
                    "fold_name": fold["fold_name"],
                    "fold_role": fold["fold_role"],
                    "train_end": fold["train_end"],
                    "valid_start": fold["valid_start"],
                    "valid_end": fold["valid_end"],
                    "horizon": fold["horizon"],
                    "status": "ok",
                    "mae": mae,
                    "rmse": mae + 1.5,
                    "mase": mase,
                    "mase_denominator": 10.0,
                    "common_mase": mase,
                    "common_mase_denominator": 10.0,
                    "bias": -0.2 + mase / 10.0,
                    "relative_mae_vs_seasonal_naive": mase,
                    "train_mae": mae / 1.1,
                    "validation_mae": mae,
                    "train_valid_mae_gap": mae - mae / 1.1,
                    "train_valid_mae_ratio": 1.1,
                    "train_residual_mean": 0.0,
                    "train_residual_abs_mean": mae / 2.0,
                    "train_residual_std": mae / 3.0,
                }
                for key, value in spec["params"].items():
                    row[f"model_param_{key}"] = json.dumps(value, sort_keys=True, default=str)
                rows.append(row)
    return pd.DataFrame(rows)


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


def test_phase6_model_comparison_runs_all_dev_lanes_and_ranks_candidates(monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    models = _reduced_all_lane_models()
    specs = model_specs(models, stage="model_comparison", include_optional=True)
    folds = _fold_metadata()

    def fake_evaluate_cv(**kwargs):
        return _fake_cv_rows(
            stage=kwargs["stage"],
            strategy_names=kwargs["strategy_names"],
            specs=kwargs["specs"],
            folds_metadata=kwargs["folds_metadata"],
        )

    monkeypatch.setattr(
        "pads_forecasting.pipelines.model_comparison.nodes.evaluate_cv",
        fake_evaluate_cv,
    )

    outputs = run_model_comparison(
        target_strategies=_target_strategies(),
        folds_metadata=folds,
        old_data_gate=_old_data_gate_decisions(),
        project={"seed": 42},
        validation={"season_length": 12},
        models=models,
        selection={
            "old_data_min_improvement_pct": 0.0,
            "stability_iqr_multiplier": 1.5,
            "stability_min_fold_variation_epsilon": 1e-12,
        },
    )
    fold_results, summary, train_valid_gap, model_selection = outputs[:4]
    robust_alpha_results, robust_alpha_summary, selection_objective_audit = outputs[11:14]

    expected_rows = 3 * len(specs) * len(folds)
    assert len(fold_results) == expected_rows
    assert len(summary) == 3 * len(specs)
    assert len(train_valid_gap) == expected_rows
    assert set(fold_results["target_strategy"]) == {
        "post_only",
        "proforma_sum",
        "calibrated_alpha",
    }
    assert "raw_full" not in set(fold_results["target_strategy"])
    assert {spec["family"] for spec in specs} == set(summary["model_family"])
    assert set(FOLD_METRIC_COLUMNS).issubset(fold_results.columns)
    assert set(SUMMARY_METRIC_COLUMNS).issubset(summary.columns)
    assert set(TRAIN_VALID_GAP_COLUMNS).issubset(train_valid_gap.columns)
    assert not robust_alpha_results.empty
    assert not robust_alpha_summary.empty
    assert not selection_objective_audit.empty
    assert {"selected_alpha", "alpha_beats_one_by_common_mase"}.issubset(
        robust_alpha_results.columns
    )
    assert {"best_alpha_normal_folds_only_grid", "selected_alpha_mode"}.issubset(
        robust_alpha_summary.columns
    )
    assert any(col.startswith("model_param_") for col in fold_results.columns)
    assert any(col.startswith("model_param_") for col in train_valid_gap.columns)

    selected = model_selection[model_selection["selected"].astype(str).str.lower().isin(["true"])]
    assert len(selected) == 1
    selected_row = selected.iloc[0]
    assert selected_row["target_strategy"] == "proforma_sum"
    assert selected_row["model_family"] == "lightgbm"
    assert bool(selected_row["eligible_for_selection"]) is True
    assert selected_row["normal_mean_mase"] < 0.75

    assert set(fake_mlflow.artifacts) == set(ARTIFACT_NAMES.values())
    assert fake_mlflow.parent_params["model_comparison.admissible_strategies"] == (
        "post_only,proforma_sum,calibrated_alpha"
    )
    assert fake_mlflow.parent_params["model_comparison.selection.stability_iqr_multiplier"] == (
        "1.5"
    )
    assert (
        fake_mlflow.parent_params["model_comparison.selected.model_id"]
        == (selected_row["model_id"])
    )
    assert (
        fake_mlflow.parent_metrics["model_comparison.selected.normal_mean_mase"]
        == (selected_row["normal_mean_mase"])
    )
    expected_summary_runs = min(len(summary), MLFLOW_MAX_SUMMARY_NESTED_RUNS)
    expected_logged_candidate_keys = set(
        summary.sort_values(
            ["normal_mean_common_mase", "mean_common_mase"],
            ascending=[True, True],
            kind="mergesort",
        )[["target_strategy", "model_id"]]
        .head(MLFLOW_MAX_FOLD_NESTED_CANDIDATES)
        .itertuples(index=False, name=None)
    )
    expected_fold_runs = sum(
        (row.target_strategy, row.model_id) in expected_logged_candidate_keys
        for row in fold_results.itertuples(index=False)
    )
    assert len(fake_mlflow.runs) == expected_summary_runs + expected_fold_runs
    assert (
        fake_mlflow.parent_params["model_comparison.mlflow_logging_mode"]
        == "compact_full_csv_artifacts"
    )

    lightgbm_fold = next(
        run
        for run in fake_mlflow.runs
        if run["run_name"].endswith("/fold_2022_normal") and "/lightgbm_" in run["run_name"]
    )
    assert lightgbm_fold["nested"] is True
    assert lightgbm_fold["params"]["target_strategy"] in {
        "post_only",
        "proforma_sum",
        "calibrated_alpha",
    }
    assert lightgbm_fold["params"]["fold_role"] == "normal"
    assert lightgbm_fold["params"]["horizon"] == "12"
    assert lightgbm_fold["params"]["model_param_num_leaves"] == "4"
    assert lightgbm_fold["metrics"]["mase"] < 0.8

    summary_run = next(run for run in fake_mlflow.runs if run["run_name"].endswith("/summary"))
    assert "normal_mean_mase" in summary_run["metrics"]
    assert "folds" in summary_run["metrics"]


def test_phase6_production_config_expands_the_full_stage_b_model_suite():
    models = yaml.safe_load(Path("conf/base/parameters_models.yml").read_text())
    specs = model_specs(models, stage="model_comparison", include_optional=True)
    counts = pd.Series([spec["family"] for spec in specs]).value_counts().to_dict()

    assert len(specs) == 3595
    assert counts == {
        "lightgbm": 2916,
        "sarimax": 432,
        "prophet": 162,
        "bvar": 36,
        "elasticnet": 27,
        "ets": 8,
        "ridge": 12,
        "seasonal_naive": 2,
    }
