"""Modelling nodes -- Optuna optimization, evaluation, selection, calibration."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as compute_confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

SPLIT_COLUMN = "split"


def _load_class(class_path: str) -> type:
    """Import a class from a dotted path like 'sklearn.linear_model.LogisticRegression'."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_feature_and_target_columns(
    columns: dict[str, list[str]],
) -> tuple[list[str], str]:
    """Extract feature column list and target column name from params."""
    feature_columns = columns["categorical"] + columns["numerical"]
    target_column = columns["target"][0]
    return feature_columns, target_column


def train_model(
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    model_params: dict[str, Any],
) -> dict[str, Any]:
    """Train a model via dynamic class loading. Returns artifact dict with estimator + metadata."""
    feature_columns, target_column = _get_feature_and_target_columns(columns)

    train_splits: list[str] = model_params.get("train_splits", ["train"])
    train_data = master_table[master_table[SPLIT_COLUMN].isin(train_splits)]

    target_encoder = LabelEncoder()
    target_encoder.fit(train_data[target_column])

    x_train = train_data[feature_columns]
    y_train = target_encoder.transform(train_data[target_column])

    model_class = _load_class(model_params["class_path"])
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
) -> dict[str, Any]:
    """Optimize hyperparameters using Optuna with stratified cross-validation.

    Falls back to ``train_model`` when ``search_space`` is empty or missing.
    Accepts a YAML-defined search space where each parameter specifies its
    type (``int``, ``float``, ``categorical``) and bounds.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    search_space = optimization_params.get("search_space", {})
    if not search_space:
        return train_model(master_table, columns, optimization_params)

    feature_columns, target_column = _get_feature_and_target_columns(columns)
    train_data = master_table[master_table[SPLIT_COLUMN] == "train"]

    target_encoder = LabelEncoder()
    target_encoder.fit(train_data[target_column])

    x_train = train_data[feature_columns].values
    y_train = target_encoder.transform(train_data[target_column])

    class_path = optimization_params["class_path"]
    init_args = optimization_params.get("init_args", {})
    n_trials = optimization_params.get("n_trials", 30)
    cv_folds = optimization_params.get("cv", 5)
    scoring = optimization_params.get("scoring", "roc_auc")

    model_class = _load_class(class_path)

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
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(
                estimator, x_train, y_train, cv=cv, scoring=scoring
            )
            valid_scores = scores[~np.isnan(scores)]
            return float(valid_scores.mean()) if len(valid_scores) > 0 else 0.0
        except ValueError:
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning(
            "optimize_model: all %d trials failed for %s, falling back to train_model",
            n_trials,
            class_path,
        )
        return train_model(master_table, columns, optimization_params)

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


def evaluate_model(
    master_table: pd.DataFrame,
    model_artifact: dict[str, Any],
    columns: dict[str, list[str]],
    evaluation_params: dict[str, Any],
) -> dict[str, Any]:
    """Compute metrics + confusion matrix on a given split."""
    split_name: str = evaluation_params["split"]
    _, target_column = _get_feature_and_target_columns(columns)

    feature_columns = model_artifact["feature_columns"]
    target_encoder = model_artifact["target_encoder"]
    estimator = model_artifact["estimator"]

    split_data = master_table[master_table[SPLIT_COLUMN] == split_name]
    if len(split_data) == 0:
        logger.warning("evaluate_model (%s): no samples in split", split_name)
        return {
            "split": split_name,
            "n_samples": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "r2": 0.0,
            "mape": 0.0,
            "confusion_matrix": [],
        }

    x_split = split_data[feature_columns]
    y_true = target_encoder.transform(split_data[target_column])

    y_pred = estimator.predict(x_split)
    y_proba = (
        estimator.predict_proba(x_split)[:, 1]
        if hasattr(estimator, "predict_proba")
        else y_pred.astype(float)
    )

    n_classes = len(np.unique(y_true))
    cm = compute_confusion_matrix(y_true, y_pred)

    r2 = float(r2_score(y_true, y_proba)) if n_classes > 1 else 0.0
    mae = float(mean_absolute_error(y_true, y_proba))
    mape = float(mae / max(y_true.mean(), 1e-8)) * 100.0

    metrics: dict[str, Any] = {
        "split": split_name,
        "n_samples": len(x_split),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (float(roc_auc_score(y_true, y_proba)) if n_classes > 1 else 0.0),
        "r2": r2,
        "mape": mape,
        "confusion_matrix": cm.tolist(),
    }

    logger.info(
        "evaluate_model (%s): f1=%.4f, roc_auc=%.4f, r2=%.4f, mape=%.2f%%, cm=%s",
        split_name,
        metrics["f1"],
        metrics["roc_auc"],
        metrics["r2"],
        metrics["mape"],
        cm.tolist(),
    )
    return metrics


def select_best_model(  # noqa: PLR0913
    baseline_model: dict[str, Any],
    baseline_metrics: dict[str, float],
    optimized_model: dict[str, Any],
    optimized_metrics: dict[str, float],
    xgboost_model: dict[str, Any],
    xgboost_metrics: dict[str, float],
    selection_params: dict[str, str],
) -> dict[str, Any]:
    """Compare all trained models on the validation split and produce the refit config.

    Returns a model_params dict that ``train_model`` can consume directly,
    with ``train_splits`` set to all splits (for the refit stage).
    """
    metric = selection_params.get("metric", "roc_auc")

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
        "train_splits": ["train", "validation", "test"],
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


def evaluate_all_on_test(
    master_table: pd.DataFrame,
    baseline_model: dict[str, Any],
    optimized_model: dict[str, Any],
    xgboost_model: dict[str, Any],
    columns: dict[str, list[str]],
) -> dict[str, Any]:
    """Produce a held-out test evaluation report for all candidate models.

    Returns a dict keyed by model name, each containing metrics + confusion matrix.
    This node runs independently of ``select_best_model`` to give a final,
    unbiased estimate of all three models before deployment.
    """
    test_params: dict[str, str] = {"split": "test"}
    report: dict[str, Any] = {}

    for name, model in [
        ("baseline", baseline_model),
        ("optimized", optimized_model),
        ("xgboost", xgboost_model),
    ]:
        metrics = evaluate_model(master_table, model, columns, test_params)
        report[name] = metrics
        logger.info(
            "test_report: %-10s roc_auc=%.4f f1=%.4f recall=%.4f",
            name,
            metrics["roc_auc"],
            metrics["f1"],
            metrics["recall"],
        )

    return report


def calibrate_model(
    production_master_table: pd.DataFrame,
    model_artifact: dict[str, Any],
    columns: dict[str, list[str]],
    calibration_params: dict[str, Any],
) -> dict[str, Any]:
    """Wrap the refitted model with CalibratedClassifierCV (Platt scaling).

    Uses ``cv='prefit'`` by default — the model is already trained on all
    data, so only the calibration curve (sigmoid or isotonic) is fitted.
    This lets the business use predicted probabilities for thresholding
    (e.g. "contact the top 20% most likely to churn").
    """
    feature_columns = model_artifact["feature_columns"]
    target_encoder = model_artifact["target_encoder"]
    _, target_column = _get_feature_and_target_columns(columns)

    x = production_master_table[feature_columns]
    y = target_encoder.transform(production_master_table[target_column])

    method = calibration_params.get("method", "sigmoid")
    cv = calibration_params.get("cv", "prefit")

    calibrated = CalibratedClassifierCV(
        estimator=model_artifact["estimator"],
        cv=cv,
        method=method,
    )
    calibrated.fit(x, y)

    logger.info("calibrate_model: method=%s, cv=%s", method, cv)

    return {**model_artifact, "estimator": calibrated}
