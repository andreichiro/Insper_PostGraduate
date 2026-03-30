"""Testes unitários dos nodes de DE """

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
        df = pd.DataFrame({"Pregnancies": [6]})
        with pytest.raises(KeyError, match="columns not found"):
            clean_data(df, raw_columns_config)


class TestAddFeatures:
    def test_adds_derived_column(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        assert "glucose_bmi_interaction" in featured.columns

    def test_derived_column_is_numeric(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        assert pd.api.types.is_numeric_dtype(featured["glucose_bmi_interaction"])

    def test_derived_column_formula(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        expected = cleaned["Glucose"] * cleaned["BMI"] / 1000
        pd.testing.assert_series_equal(
            featured["glucose_bmi_interaction"], expected, check_names=False
        )


class TestAddSplitColumn:
    def test_adds_split_column(
        self, sample_raw_data, raw_columns_config, split_ratio, preprocessing_config
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        result = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        assert "split" in result.columns

    def test_split_names_match_config(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        result = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        assert set(result["split"].unique()).issubset(set(split_ratio.keys()))

    def test_reproducible_with_seed(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        r1 = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        r2 = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_stratified_split(
        self, sample_raw_data, raw_columns_config, split_ratio, preprocessing_config
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        bigger = pd.concat([featured] * 5, ignore_index=True)
        result = add_split_column(
            bigger,
            split_ratio,
            random_state=42,
            stratify_column="Outcome",
            preprocessing=preprocessing_config,
        )
        assert "split" in result.columns
        assert set(result["split"].unique()) == set(split_ratio.keys())


class TestFitTransformEncoders:
    def test_fit_returns_empty_dict_no_categoricals(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        """Diabetes não tem colunas categóricas, então o dict de encoders fica vazio."""
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        assert encoders == {}

    def test_transform_is_noop_without_encoders(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        encoded = transform_encoders(split, encoders)
        pd.testing.assert_frame_equal(encoded, split)


class TestFitTransformScalers:
    def test_fit_returns_one_scaler_per_numerical(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        encoded = transform_encoders(split, encoders)
        scalers = fit_scalers(
            encoded, columns_config, fit_transform_config, preprocessing_config
        )
        assert set(scalers.keys()) == set(columns_config["numerical"])

    def test_transform_changes_scale(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        encoded = transform_encoders(split, encoders)
        scalers = fit_scalers(
            encoded, columns_config, fit_transform_config, preprocessing_config
        )
        scaled = transform_scalers(encoded, scalers)
        for col in columns_config["numerical"]:
            assert scaled[col].std() != encoded[col].std() or len(scaled) == 1
