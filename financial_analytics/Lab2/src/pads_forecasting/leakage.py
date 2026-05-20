"""Leakage checks that fail loudly."""

from __future__ import annotations

import pandas as pd


class LeakageError(ValueError):
    """Raised when a leakage invariant is violated."""


def assert_no_fold_leakage(
    train_end: str | pd.Timestamp, valid_start: str | pd.Timestamp, valid_end: str | pd.Timestamp
) -> None:
    """Assert temporal ordering of train and validation windows."""

    train_end_ts = pd.Timestamp(train_end)
    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)
    if train_end_ts >= valid_start_ts:
        raise LeakageError(
            f"train_end {train_end_ts.date()} must be before valid_start {valid_start_ts.date()}"
        )
    if valid_end_ts < valid_start_ts:
        raise LeakageError("valid_end must be >= valid_start")


def assert_no_2024_used(df: pd.DataFrame, date_col: str = "data") -> None:
    """Assert no hidden 2024 values are present in training/evaluation data."""

    if (pd.to_datetime(df[date_col]) >= pd.Timestamp("2024-01-01")).any():
        raise LeakageError("2024 hidden dates are present in a dataset.")


def assert_acquired_only_pre_acquisition(
    df: pd.DataFrame,
    acquisition_date: str | pd.Timestamp,
    acquired_col: str = "adquirida_separada",
) -> None:
    """Assert acquired-company standalone values are not used after acquisition."""

    acquisition_ts = pd.Timestamp(acquisition_date)
    post = df[pd.to_datetime(df["data"]) >= acquisition_ts]
    if acquired_col in post and post[acquired_col].notna().any():
        raise LeakageError("Standalone acquired-company values exist after acquisition.")


def assert_validation_target_observed(valid_df: pd.DataFrame) -> None:
    """Assert validation target rows are observed consolidated values."""

    if (
        "target_source" in valid_df
        and not valid_df["target_source"].eq("observed_consolidated").all()
    ):
        raise LeakageError("Validation target must be observed post-acquisition consolidated C_t.")


def assert_shifted_lag_columns(
    feature_df: pd.DataFrame, lag_cols: list[str], date_col: str = "data"
) -> None:
    """Guardrail hook for lag feature tests."""

    if feature_df[date_col].duplicated().any():
        raise LeakageError("Feature frame has duplicated dates.")
    missing = [col for col in lag_cols if col not in feature_df.columns]
    if missing:
        raise LeakageError(f"Missing lag columns: {missing}")


def assert_fold_datasets_no_leakage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    valid_start: str | pd.Timestamp,
    valid_end: str | pd.Timestamp,
    acquisition_date: str | pd.Timestamp = "2019-07-01",
    date_col: str = "data",
) -> None:
    """Assert fold train/validation slices obey all temporal leakage invariants."""

    assert_no_fold_leakage(train_end, valid_start, valid_end)
    assert_no_2024_used(train_df, date_col=date_col)
    assert_no_2024_used(valid_df, date_col=date_col)

    train_end_ts = pd.Timestamp(train_end)
    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)
    train_dates = pd.to_datetime(train_df[date_col])
    valid_dates = pd.to_datetime(valid_df[date_col])

    if train_dates.empty:
        raise LeakageError("Training fold is empty.")
    if valid_dates.empty:
        raise LeakageError("Validation fold is empty.")
    if (train_dates > train_end_ts).any():
        raise LeakageError("Training fold contains dates after train_end.")
    if (valid_dates < valid_start_ts).any() or (valid_dates > valid_end_ts).any():
        raise LeakageError("Validation fold contains dates outside validation window.")
    if set(train_dates).intersection(set(valid_dates)):
        raise LeakageError("Training and validation folds overlap.")

    assert_validation_target_observed(valid_df)
    if "adquirida_separada" in train_df:
        assert_acquired_only_pre_acquisition(train_df, acquisition_date)
    if "adquirida_separada" in valid_df:
        assert_acquired_only_pre_acquisition(valid_df, acquisition_date)
