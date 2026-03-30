"""Nodes de modelagem — Optuna, métricas e calibração guiados por YAML (class_path / callables)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from insper_deploy_kedro.class_loading import load_callable, load_class
from insper_deploy_kedro.constants import SPLIT_COLUMN, ModelArtifact

logger = logging.getLogger(__name__)


def _get_feature_and_target_columns(
    columns: dict[str, list[str]],
) -> tuple[list[str], str]:
    """Extrai lista de features e nome da coluna target do config."""
    feature_columns = columns["categorical"] + columns["numerical"]
    target_column = columns["target"][0]
    return feature_columns, target_column


def _build_target_encoder(ml_runtime: dict[str, Any]) -> Any:
    cfg = ml_runtime["target_encoder"]
    cls = load_class(cfg["class_path"])
    return cls(**dict(cfg.get("init_args") or {}))


def train_model(
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    model_params: dict[str, Any],
    ml_runtime: dict[str, Any],
) -> ModelArtifact:
    """Treina modelo com classe e hiperparâmetros do YAML."""
    feature_columns, target_column = _get_feature_and_target_columns(columns)

    train_splits: list[str] = model_params.get("train_splits", ["train"])
    train_data = master_table[master_table[SPLIT_COLUMN].isin(train_splits)]

    target_encoder = _build_target_encoder(ml_runtime)
    target_encoder.fit(train_data[target_column])

    x_train = train_data[feature_columns]
    y_train = target_encoder.transform(train_data[target_column])

    model_class = load_class(model_params["class_path"])
    estimator = model_class(**model_params.get("init_args", {}))
    estimator.fit(x_train, y_train)

    logger.info(
        "train_model: %s on %d rows (%s)",
        model_params["class_path"],
        len(x_train),
        train_splits,
    )

    return {
        "estimator": estimator,
        "target_encoder": target_encoder,
        "feature_columns": feature_columns,
        "class_path": model_params["class_path"],
        "init_args": model_params.get("init_args", {}),
    }


def optimize_model(
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    optimization_params: dict[str, Any],
    ml_runtime: dict[str, Any],
) -> ModelArtifact:
    """Optuna + CV — classes CV e sampler vêm do `ml_runtime` no YAML."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    search_space = optimization_params.get("search_space", {})
    if not search_space:
        return train_model(master_table, columns, optimization_params, ml_runtime)

    feature_columns, target_column = _get_feature_and_target_columns(columns)
    train_data = master_table[master_table[SPLIT_COLUMN] == "train"]

    target_encoder = _build_target_encoder(ml_runtime)
    target_encoder.fit(train_data[target_column])

    x_train = train_data[feature_columns].values
    y_train = target_encoder.transform(train_data[target_column])

    class_path = optimization_params["class_path"]
    init_args = optimization_params.get("init_args", {})
    n_trials = optimization_params.get("n_trials", 30)
    cv_folds = optimization_params.get("cv", 5)
    scoring = optimization_params.get("scoring", "roc_auc")
    seed = optimization_params.get("random_state", 42)

    model_class = load_class(class_path)

    cv_cfg = ml_runtime["cross_validation"]
    cv_class = load_class(cv_cfg["class_path"])
    cv_kwargs = {**dict(cv_cfg.get("init_args") or {}), "n_splits": cv_folds}
    cv = cv_class(**cv_kwargs)

    sampler_cfg = ml_runtime["optuna_sampler"]
    sampler_class = load_class(sampler_cfg["class_path"])
    sampler_kwargs = {**dict(sampler_cfg.get("init_args") or {}), "seed": seed}
    sampler = sampler_class(**sampler_kwargs)

    study_cfg = ml_runtime.get("optuna_study") or {}
    direction = study_cfg.get("direction", "maximize")

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for name, config in search_space.items():
            stype = config["type"]
            if stype == "int":
                params[name] = trial.suggest_int(name, config["low"], config["high"])
            elif stype == "float":
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"], log=config.get("log", False)
                )
            elif stype == "categorical":
                params[name] = trial.suggest_categorical(name, config["choices"])

        estimator = model_class(**{**init_args, **params})
        try:
            scores = cross_val_score(
                estimator, x_train, y_train, cv=cv, scoring=scoring
            )
            valid_scores = scores[~np.isnan(scores)]
            return float(valid_scores.mean()) if len(valid_scores) > 0 else 0.0
        except ValueError:
            return 0.0

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning(
            "optimize_model: all %d trials failed for %s, falling back to train_model",
            n_trials,
            class_path,
        )
        return train_model(master_table, columns, optimization_params, ml_runtime)

    best_params = {**init_args, **study.best_params}
    estimator = model_class(**best_params)
    estimator.fit(x_train, y_train)

    logger.info(
        "optimize_model: %s best_%s=%.4f, best_params=%s (n_trials=%d)",
        class_path,
        scoring,
        study.best_value,
        study.best_params,
        n_trials,
    )

    return {
        "estimator": estimator,
        "target_encoder": target_encoder,
        "feature_columns": feature_columns,
        "class_path": class_path,
        "init_args": init_args,
        "best_params": study.best_params,
        "best_cv_score": float(study.best_value),
    }


def _metric_keys(evaluation_params: dict[str, Any]) -> list[str]:
    keys = [m["key"] for m in evaluation_params.get("metrics", [])]
    derived = evaluation_params.get("derived") or {}
    if isinstance(derived, dict):
        if derived.get("r2"):
            keys.append("r2")
        if derived.get("mape"):
            keys.append("mape")
    return list(dict.fromkeys(keys))


def evaluate_model(
    master_table: pd.DataFrame,
    model_artifact: ModelArtifact,
    columns: dict[str, list[str]],
    evaluation_params: dict[str, Any],
) -> dict[str, Any]:
    """Métricas e matriz de confusão declaradas no YAML (`evaluation`)."""
    split_name: str = evaluation_params["split"]
    _, target_column = _get_feature_and_target_columns(columns)

    feature_columns = model_artifact["feature_columns"]
    target_encoder = model_artifact["target_encoder"]
    estimator = model_artifact["estimator"]

    split_data = master_table[master_table[SPLIT_COLUMN] == split_name]
    metric_keys = _metric_keys(evaluation_params)

    if len(split_data) == 0:
        logger.warning("evaluate_model (%s): no samples in split", split_name)
        empty: dict[str, Any] = {
            "split": split_name,
            "n_samples": 0,
            "confusion_matrix": [],
        }
        for k in metric_keys:
            empty[k] = 0.0
        return empty

    x_split = split_data[feature_columns]
    y_true = target_encoder.transform(split_data[target_column])

    y_pred = estimator.predict(x_split)
    y_proba = (
        estimator.predict_proba(x_split)[:, 1]
        if hasattr(estimator, "predict_proba")
        else y_pred.astype(float)
    )

    n_classes = len(np.unique(y_true))

    cm_cfg = evaluation_params.get("confusion_matrix") or {}
    cm_fn = load_callable(
        cm_cfg.get("function_path", "sklearn.metrics.confusion_matrix")
    )
    cm_kwargs = dict(cm_cfg.get("kwargs") or {})
    cm = cm_fn(y_true, y_pred, **cm_kwargs)

    metrics: dict[str, Any] = {
        "split": split_name,
        "n_samples": len(x_split),
        "confusion_matrix": cm.tolist(),
    }

    for m in evaluation_params.get("metrics", []) or []:
        fn = load_callable(m["function_path"])
        pred_kind = m.get("prediction_input", "y_pred")
        y_second = y_proba if pred_kind == "y_proba" else y_pred
        kwargs = dict(m.get("kwargs") or {})
        metrics[m["key"]] = float(fn(y_true, y_second, **kwargs))

    derived = evaluation_params.get("derived") or {}
    if isinstance(derived, dict):
        r2_cfg = derived.get("r2")
        if r2_cfg and n_classes > 1:
            r2_fn = load_callable(r2_cfg["function_path"])
            pred_kind = r2_cfg.get("prediction_input", "y_proba")
            y_second = y_proba if pred_kind == "y_proba" else y_pred
            rk = dict(r2_cfg.get("kwargs") or {})
            metrics["r2"] = float(r2_fn(y_true, y_second, **rk))
        elif "r2" in metric_keys and "r2" not in metrics:
            metrics["r2"] = 0.0

        mape_cfg = derived.get("mape")
        if mape_cfg and mape_cfg.get("type") == "mae_as_percent_of_mean_label":
            mae = float(mean_absolute_error(y_true, y_proba))
            metrics["mape"] = float(mae / max(y_true.mean(), 1e-8)) * 100.0
        elif "mape" in metric_keys and "mape" not in metrics:
            metrics["mape"] = 0.0

    logger.info(
        "evaluate_model (%s): f1=%.4f, roc_auc=%.4f, r2=%.4f, mape=%.2f%%, cm=%s",
        split_name,
        metrics.get("f1", 0),
        metrics.get("roc_auc", 0),
        metrics.get("r2", 0),
        metrics.get("mape", 0),
        cm.tolist(),
    )
    return metrics


def select_best_model(  # noqa: PLR0913
    baseline_model: ModelArtifact,
    baseline_metrics: dict[str, float],
    optimized_model: ModelArtifact,
    optimized_metrics: dict[str, float],
    xgboost_model: ModelArtifact,
    xgboost_metrics: dict[str, float],
    selection_params: dict[str, Any],
) -> dict[str, Any]:
    """Escolhe vencedor e monta config de refit (splits também no YAML)."""
    metric = selection_params.get("metric", "roc_auc")
    refit_splits = selection_params.get(
        "refit_train_splits",
        ["train", "validation", "test"],
    )

    candidates = [
        ("baseline", baseline_model, baseline_metrics),
        ("optimized", optimized_model, optimized_metrics),
        ("xgboost", xgboost_model, xgboost_metrics),
    ]

    for name, _model, metrics in candidates:
        logger.info(
            "select_best_model: %-10s  %s=%.4f  f1=%.4f  recall=%.4f  accuracy=%.4f",
            name,
            metric,
            metrics.get(metric, 0),
            metrics.get("f1", 0),
            metrics.get("recall", 0),
            metrics.get("accuracy", 0),
        )

    best_name, best_model, best_metrics = max(
        candidates, key=lambda c: c[2].get(metric, 0)
    )

    init_args = dict(best_model.get("init_args", {}))
    if "best_params" in best_model:
        init_args.update(best_model["best_params"])

    config: dict[str, Any] = {
        "class_path": best_model["class_path"],
        "train_splits": list(refit_splits),
        "init_args": init_args,
    }

    logger.info(
        "select_best_model: WINNER = %s (%s=%.4f) → refit with %s %s",
        best_name,
        metric,
        best_metrics[metric],
        config["class_path"],
        init_args,
    )

    return config


def evaluate_all_on_test(  # noqa: PLR0913
    master_table: pd.DataFrame,
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    columns: dict[str, list[str]],
    evaluation_params: dict[str, Any],
) -> dict[str, Any]:
    """Relatório no teste — reusa o mesmo bloco `evaluation` com split=test."""
    test_eval = {**evaluation_params, "split": "test"}
    report: dict[str, Any] = {}

    for name, model in [
        ("baseline", baseline_model),
        ("optimized", optimized_model),
        ("xgboost", xgboost_model),
    ]:
        metrics = evaluate_model(master_table, model, columns, test_eval)
        report[name] = metrics
        logger.info(
            "test_report: %-10s roc_auc=%.4f f1=%.4f recall=%.4f",
            name,
            metrics.get("roc_auc", 0),
            metrics.get("f1", 0),
            metrics.get("recall", 0),
        )

    return report


def calibrate_model(
    production_master_table: pd.DataFrame,
    model_artifact: ModelArtifact,
    columns: dict[str, list[str]],
    calibration_params: dict[str, Any],
) -> ModelArtifact:
    """Calibração — classe e init_args no YAML (`refit.calibration`)."""
    feature_columns = model_artifact["feature_columns"]
    target_encoder = model_artifact["target_encoder"]
    _, target_column = _get_feature_and_target_columns(columns)

    x = production_master_table[feature_columns]
    y = target_encoder.transform(production_master_table[target_column])

    cal_class_path = calibration_params.get(
        "class_path",
        "sklearn.calibration.CalibratedClassifierCV",
    )
    cal_class = load_class(cal_class_path)
    cal_kwargs = dict(calibration_params.get("init_args") or {})
    calibrated = cal_class(estimator=model_artifact["estimator"], **cal_kwargs)
    calibrated.fit(x, y)

    logger.info(
        "calibrate_model: %s init_args=%s",
        cal_class_path,
        cal_kwargs,
    )

    return {**model_artifact, "estimator": calibrated}
