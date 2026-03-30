"""Unit tests for modelling nodes."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV

from insper_deploy_kedro.pipelines.modelling.nodes import (
    calibrate_model,
    evaluate_all_on_test,
    evaluate_model,
    optimize_model,
    select_best_model,
    train_model,
)


class TestTrainModel:
    def test_returns_artifact_dict(self, master_table, columns_config):
        model_params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = train_model(master_table, columns_config, model_params)
        assert "estimator" in artifact
        assert "target_encoder" in artifact
        assert "feature_columns" in artifact

    def test_feature_columns_include_derived(self, master_table, columns_config):
        model_params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = train_model(master_table, columns_config, model_params)
        assert "avg_charge_per_month" in artifact["feature_columns"]

    def test_stores_class_path_and_init_args(self, master_table, columns_config):
        model_params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = train_model(master_table, columns_config, model_params)
        assert artifact["class_path"] == "sklearn.linear_model.LogisticRegression"
        assert artifact["init_args"] == {"max_iter": 1000}

    def test_target_encoder_fit_on_train_only(self, master_table, columns_config):
        model_params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = train_model(master_table, columns_config, model_params)
        train_labels = set(
            master_table[master_table["split"] == "train"]["Churn"].unique()
        )
        encoder_classes = set(artifact["target_encoder"].classes_)
        assert encoder_classes.issubset(train_labels)


class TestOptimizeModel:
    def test_returns_best_params(self, master_table, columns_config):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "search_space": {
                "C": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
            },
            "init_args": {"max_iter": 1000, "class_weight": "balanced"},
            "n_trials": 3,
            "cv": 2,
            "scoring": "roc_auc",
        }
        artifact = optimize_model(master_table, columns_config, params)
        assert "best_params" in artifact
        assert "C" in artifact["best_params"]
        assert "best_cv_score" in artifact
        assert isinstance(artifact["best_cv_score"], float)

    def test_falls_back_to_train_without_search_space(
        self, master_table, columns_config
    ):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = optimize_model(master_table, columns_config, params)
        assert "estimator" in artifact
        assert "best_params" not in artifact

    def test_trained_model_can_predict(self, master_table, columns_config):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "search_space": {
                "C": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
            },
            "init_args": {"max_iter": 1000},
            "n_trials": 2,
            "cv": 2,
            "scoring": "roc_auc",
        }
        artifact = optimize_model(master_table, columns_config, params)
        x = master_table[artifact["feature_columns"]].iloc[:2]
        predictions = artifact["estimator"].predict(x)
        assert len(predictions) == 2


class TestEvaluateModel:
    def test_returns_standard_metrics(
        self, master_table, trained_model, columns_config
    ):
        eval_params = {"split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )

        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "r2"):
            assert key in metrics
        assert "mape" in metrics
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_metrics_include_sample_count(
        self, master_table, trained_model, columns_config
    ):
        eval_params = {"split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )
        assert metrics["n_samples"] > 0

    def test_includes_confusion_matrix(
        self, master_table, trained_model, columns_config
    ):
        eval_params = {"split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert isinstance(cm, list)
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_includes_r2_and_mape(self, master_table, trained_model, columns_config):
        eval_params = {"split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )
        assert isinstance(metrics["r2"], float)
        assert isinstance(metrics["mape"], float)
        assert metrics["mape"] >= 0.0

    def test_handles_empty_split_gracefully(
        self, master_table, trained_model, columns_config
    ):
        eval_params = {"split": "nonexistent_split"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )
        assert metrics["n_samples"] == 0
        assert metrics["confusion_matrix"] == []
        assert metrics["r2"] == 0.0
        assert metrics["mape"] == 0.0


class TestSelectBestModel:
    def test_picks_highest_metric(self, master_table, columns_config):
        params_a = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        model_a = train_model(master_table, columns_config, params_a)
        metrics_a = {"roc_auc": 0.80, "f1": 0.60, "accuracy": 0.75}
        metrics_b = {"roc_auc": 0.85, "f1": 0.65, "accuracy": 0.78}
        metrics_c = {"roc_auc": 0.82, "f1": 0.62, "accuracy": 0.76}

        config = select_best_model(
            model_a,
            metrics_a,
            model_a,
            metrics_b,
            model_a,
            metrics_c,
            {"metric": "roc_auc"},
        )
        assert config["train_splits"] == ["train", "validation", "test"]
        assert "class_path" in config
        assert "init_args" in config

    def test_merges_best_params_for_optimized(self, master_table, columns_config):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        model = train_model(master_table, columns_config, params)

        optimized = dict(model)
        optimized["best_params"] = {"C": 0.5}
        optimized["init_args"] = {"max_iter": 1000}

        config = select_best_model(
            model,
            {"roc_auc": 0.70},
            optimized,
            {"roc_auc": 0.90},
            model,
            {"roc_auc": 0.75},
            {"metric": "roc_auc"},
        )
        assert config["init_args"]["C"] == 0.5
        assert config["init_args"]["max_iter"] == 1000


class TestEvaluateAllOnTest:
    def test_returns_report_for_all_models(
        self, master_table, trained_model, columns_config
    ):
        report = evaluate_all_on_test(
            master_table, trained_model, trained_model, trained_model, columns_config
        )
        assert set(report.keys()) == {"baseline", "optimized", "xgboost"}
        for model_name in report:
            assert "roc_auc" in report[model_name]
            assert "confusion_matrix" in report[model_name]
            assert report[model_name]["split"] == "test"


class TestCalibrateModel:
    def test_returns_calibrated_estimator(
        self, master_table, trained_model, columns_config
    ):
        calibrated = calibrate_model(
            master_table,
            trained_model,
            columns_config,
            {"method": "sigmoid", "cv": "prefit"},
        )
        assert isinstance(calibrated["estimator"], CalibratedClassifierCV)

    def test_calibrated_model_can_predict_proba(
        self, master_table, trained_model, columns_config
    ):
        calibrated = calibrate_model(
            master_table,
            trained_model,
            columns_config,
            {"method": "sigmoid", "cv": "prefit"},
        )
        x = master_table[calibrated["feature_columns"]].iloc[:3]
        proba = calibrated["estimator"].predict_proba(x)
        assert proba.shape == (3, 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_preserves_metadata(self, master_table, trained_model, columns_config):
        calibrated = calibrate_model(
            master_table,
            trained_model,
            columns_config,
            {"method": "sigmoid", "cv": "prefit"},
        )
        assert calibrated["class_path"] == trained_model["class_path"]
        assert calibrated["feature_columns"] == trained_model["feature_columns"]
        assert calibrated["target_encoder"] is trained_model["target_encoder"]
