"""Testes dos nodes de inferência — to_dataframe e predict."""

from __future__ import annotations

import pandas as pd

from insper_deploy_kedro.pipelines.inference.nodes import predict, to_dataframe


class TestToDataframe:
    def test_single_dict(self):
        result = to_dataframe({"a": 1, "b": 2})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ["a", "b"]

    def test_list_of_dicts(self):
        result = to_dataframe([{"a": 1}, {"a": 2}])
        assert len(result) == 2

    def test_empty_list_returns_empty_dataframe(self):
        result = to_dataframe([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestPredict:
    def test_returns_prediction_column(self, master_table, trained_model):
        result = predict(master_table, trained_model)
        assert "prediction" in result.columns
        assert len(result) == len(master_table)

    def test_returns_proba_for_sklearn(self, master_table, trained_model):
        result = predict(master_table, trained_model)
        assert "prediction_proba" in result.columns
        assert result["prediction_proba"].between(0, 1).all()

    def test_predictions_are_valid_labels(
        self, sample_raw_data, master_table, trained_model
    ):
        result = predict(master_table, trained_model)
        valid_labels = set(sample_raw_data["Outcome"].unique())
        assert set(result["prediction"].unique()).issubset(valid_labels)

    def test_preserves_original_columns(self, master_table, trained_model):
        original_cols = set(master_table.columns)
        result = predict(master_table, trained_model)
        assert original_cols.issubset(set(result.columns))
