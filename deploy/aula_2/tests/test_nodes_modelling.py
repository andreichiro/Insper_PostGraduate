"""Unit tests for modelling nodes."""

from __future__ import annotations

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
        assert "init_args" in artifact

    def test_feature_columns_include_derived(self, master_table, columns_config):
        model_params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = train_model(master_table, columns_config, model_params)
        assert "glucose_bmi_interaction" in artifact["feature_columns"]


class TestOptimizeModel:
    def test_returns_best_params(self, master_table, columns_config):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "init_args": {"max_iter": 1000},
            "search_space": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "n_trials": 3,
            "cv": 2,
            "scoring": "roc_auc",
        }
        artifact = optimize_model(master_table, columns_config, params)
        assert "estimator" in artifact
        assert "best_params" in artifact or "class_path" in artifact

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
            "init_args": {"max_iter": 1000},
            "search_space": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "n_trials": 3,
            "cv": 2,
            "scoring": "roc_auc",
        }
        artifact = optimize_model(master_table, columns_config, params)
        fc = artifact["feature_columns"]
        x = master_table[master_table["split"] == "train"][fc]
        preds = artifact["estimator"].predict(x)
        assert len(preds) == len(x)


class TestEvaluateModel:
    def test_returns_standard_metrics(self, master_table, columns_config, trained_model):
        eval_params = {"split": "train"}
        metrics = evaluate_model(master_table, trained_model, columns_config, eval_params)

        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_includes_r2_and_mape(self, master_table, columns_config, trained_model):
        eval_params = {"split": "train"}
        metrics = evaluate_model(master_table, trained_model, columns_config, eval_params)
        assert "r2" in metrics
        assert "mape" in metrics

    def test_includes_confusion_matrix(
        self, master_table, columns_config, trained_model
    ):
        eval_params = {"split": "train"}
        metrics = evaluate_model(master_table, trained_model, columns_config, eval_params)
        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], list)

    def test_handles_empty_split_gracefully(
        self, master_table, columns_config, trained_model
    ):
        eval_params = {"split": "nonexistent"}
        metrics = evaluate_model(master_table, trained_model, columns_config, eval_params)
        assert metrics["n_samples"] == 0
        assert metrics["f1"] == 0.0
        assert metrics["r2"] == 0.0
        assert metrics["mape"] == 0.0


class TestSelectBestModel:
    def test_returns_refit_config(self, master_table, columns_config, trained_model):
        eval_params = {"split": "train"}
        metrics = evaluate_model(master_table, trained_model, columns_config, eval_params)

        config = select_best_model(
            trained_model,
            metrics,
            trained_model,
            metrics,
            trained_model,
            metrics,
            {"metric": "roc_auc"},
        )
        assert "class_path" in config
        assert "train_splits" in config
        assert "init_args" in config


class TestEvaluateAllOnTest:
    def test_returns_report_for_all_models(
        self, master_table, columns_config, trained_model
    ):
        report = evaluate_all_on_test(
            master_table,
            trained_model,
            trained_model,
            trained_model,
            columns_config,
        )
        for name in ("baseline", "optimized", "xgboost"):
            assert name in report
            assert "f1" in report[name]
            assert "confusion_matrix" in report[name]


class TestCalibrateModel:
    def test_returns_calibrated_estimator(
        self, master_table, columns_config, trained_model
    ):
        cal_params = {"method": "sigmoid", "cv": "prefit"}
        calibrated = calibrate_model(
            master_table, trained_model, columns_config, cal_params
        )
        assert "estimator" in calibrated
        assert hasattr(calibrated["estimator"], "predict_proba")

    def test_calibrated_model_can_predict_proba(
        self, master_table, columns_config, trained_model
    ):
        cal_params = {"method": "sigmoid", "cv": "prefit"}
        calibrated = calibrate_model(
            master_table, trained_model, columns_config, cal_params
        )
        fc = calibrated["feature_columns"]
        x = master_table[master_table["split"] == "train"][fc]
        proba = calibrated["estimator"].predict_proba(x)
        assert proba.shape[1] == 2

    def test_preserves_metadata(self, master_table, columns_config, trained_model):
        cal_params = {"method": "sigmoid", "cv": "prefit"}
        calibrated = calibrate_model(
            master_table, trained_model, columns_config, cal_params
        )
        assert calibrated["class_path"] == trained_model["class_path"]
        assert calibrated["feature_columns"] == trained_model["feature_columns"]
