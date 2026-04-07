from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _build_logistic_param_distributions() -> list[dict[str, Any]]:
    common_c_values = [0.01, 0.1, 1.0, 10.0]
    common_class_weights = [None, "balanced"]
    return [
        {
            "model__solver": ["lbfgs"],
            "model__penalty": ["l2"],
            "model__C": common_c_values,
            "model__class_weight": common_class_weights,
            "model__max_iter": [10000],
        },
        {
            "model__solver": ["liblinear"],
            "model__penalty": ["l1", "l2"],
            "model__C": common_c_values,
            "model__class_weight": common_class_weights,
            "model__max_iter": [10000],
        },
        {
            "model__solver": ["saga"],
            "model__penalty": ["elasticnet"],
            "model__C": [0.1, 1.0, 10.0],
            "model__l1_ratio": [0.25, 0.5, 0.75],
            "model__class_weight": common_class_weights,
            "model__max_iter": [10000],
        },
    ]


def _build_random_forest_param_distributions() -> dict[str, Any]:
    return {
        "model__n_estimators": [150, 300, 500],
        "model__max_depth": [6, 10, 14],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", 0.5, 0.75],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }


def _build_catboost_param_distributions() -> list[dict[str, Any]]:
    common = {
        "model__iterations": [150, 300, 500, 700],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "model__l2_leaf_reg": [1.0, 3.0, 5.0, 9.0],
    }
    return [
        common,
        {
            **common,
            "model__auto_class_weights": ["Balanced", "SqrtBalanced"],
        },
    ]


def build_model_specs(model_families: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    allowed = set(model_families)
    if "logistic_regression" in allowed:
        specs.append(
            {
                "model_name": "logistic_regression",
                "estimator": LogisticRegression(max_iter=10000, solver="lbfgs", random_state=42),
                "param_distributions": _build_logistic_param_distributions(),
            }
        )
    if "random_forest" in allowed:
        specs.append(
            {
                "model_name": "random_forest",
                "estimator": RandomForestClassifier(random_state=42, n_jobs=1),
                "param_distributions": _build_random_forest_param_distributions(),
            }
        )
    if "catboost" in allowed:
        specs.append(
            {
                "model_name": "catboost",
                "estimator": CatBoostClassifier(
                    loss_function="Logloss",
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                ),
                "param_distributions": _build_catboost_param_distributions(),
            }
        )
    return specs
