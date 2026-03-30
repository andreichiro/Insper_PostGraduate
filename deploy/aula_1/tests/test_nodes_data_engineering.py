"""Unit tests for data-engineering nodes."""

from __future__ import annotations

import pandas as pd
import pytest

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    add_split_column,
    clean_data,
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)


class TestCleanData:
    def test_selects_configured_columns(self, sample_raw_data, raw_columns_config):
        result = clean_data(sample_raw_data, raw_columns_config)
        expected_cols = (
            raw_columns_config["target"]
            + raw_columns_config["categorical"]
            + raw_columns_config["numerical"]
        )
        assert set(result.columns) == set(expected_cols)

    def test_coerces_numerical_to_float(self, sample_raw_data, raw_columns_config):
        result = clean_data(sample_raw_data, raw_columns_config)
        for col in raw_columns_config["numerical"]:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_raises_on_missing_column(self, raw_columns_config):
        df = pd.DataFrame({"gender": ["Male"]})
        with pytest.raises(KeyError, match="columns not found"):
            clean_data(df, raw_columns_config)

    def test_coerces_non_numeric_strings_to_zero(
        self, sample_raw_data, raw_columns_config
    ):
        dirty = sample_raw_data.copy()
        dirty.loc[0, "TotalCharges"] = " "
        result = clean_data(dirty, raw_columns_config)
        assert result.loc[0, "TotalCharges"] == 0.0


class TestAddFeatures:
    def test_adds_derived_column(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        assert "avg_charge_per_month" in featured.columns

    def test_derived_column_is_numeric(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        assert pd.api.types.is_numeric_dtype(featured["avg_charge_per_month"])

    def test_derived_column_formula(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        expected = cleaned["TotalCharges"] / (cleaned["tenure"] + 1)
        pd.testing.assert_series_equal(
            featured["avg_charge_per_month"], expected, check_names=False
        )

    def test_does_not_mutate_input(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        original_cols = set(cleaned.columns)
        add_features(cleaned)
        assert set(cleaned.columns) == original_cols


class TestAddSplitColumn:
    def test_adds_split_column(self, sample_raw_data, raw_columns_config, split_ratio):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        result = add_split_column(featured, split_ratio, random_state=42)
        assert "split" in result.columns

    def test_split_names_match_config(
        self, sample_raw_data, raw_columns_config, split_ratio
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        result = add_split_column(featured, split_ratio, random_state=42)
        assert set(result["split"].unique()).issubset(set(split_ratio.keys()))

    def test_reproducible_with_seed(
        self, sample_raw_data, raw_columns_config, split_ratio
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        r1 = add_split_column(featured, split_ratio, random_state=42)
        r2 = add_split_column(featured, split_ratio, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)


class TestFitTransformEncoders:
    def test_fit_returns_one_encoder_per_categorical(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(featured, split_ratio, random_state=42)
        encoders = fit_encoders(split, columns_config, fit_transform_config)
        assert set(encoders.keys()) == set(columns_config["categorical"])

    def test_transform_replaces_strings_with_ints(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(featured, split_ratio, random_state=42)
        encoders = fit_encoders(split, columns_config, fit_transform_config)
        encoded = transform_encoders(split, encoders)
        for col in columns_config["categorical"]:
            assert pd.api.types.is_integer_dtype(encoded[col])

    def test_unseen_category_becomes_minus_one(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(featured, split_ratio, random_state=42)
        encoders = fit_encoders(split, columns_config, fit_transform_config)

        unseen_row = split.iloc[[0]].copy()
        unseen_row["gender"] = "UNSEEN_VALUE"
        encoded = transform_encoders(unseen_row, encoders)
        assert encoded["gender"].iloc[0] == -1


class TestFitTransformScalers:
    def test_fit_returns_one_scaler_per_numerical(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(featured, split_ratio, random_state=42)
        encoders = fit_encoders(split, columns_config, fit_transform_config)
        encoded = transform_encoders(split, encoders)
        scalers = fit_scalers(encoded, columns_config, fit_transform_config)
        assert set(scalers.keys()) == set(columns_config["numerical"])

    def test_transform_changes_scale(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(featured, split_ratio, random_state=42)
        encoders = fit_encoders(split, columns_config, fit_transform_config)
        encoded = transform_encoders(split, encoders)
        scalers = fit_scalers(encoded, columns_config, fit_transform_config)
        scaled = transform_scalers(encoded, scalers)
        for col in columns_config["numerical"]:
            assert scaled[col].std() != encoded[col].std() or len(scaled) == 1

    def test_train_split_approximately_standardized(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
    ):
        """Train split should have approximately mean=0, std=1 after scaling."""
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(featured, split_ratio, random_state=42)
        encoders = fit_encoders(split, columns_config, fit_transform_config)
        encoded = transform_encoders(split, encoders)
        scalers = fit_scalers(encoded, columns_config, fit_transform_config)
        scaled = transform_scalers(encoded, scalers)

        train_mask = scaled["split"] == "train"
        train_data = scaled[train_mask]
        if len(train_data) > 1:
            for col in columns_config["numerical"]:
                assert abs(train_data[col].mean()) < 0.5
