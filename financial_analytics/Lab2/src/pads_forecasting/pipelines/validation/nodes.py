"""Validation fold and leakage metadata nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pads_forecasting.leakage import (
    LeakageError,
    assert_fold_datasets_no_leakage,
    assert_no_fold_leakage,
)


def build_folds_metadata(validation: dict[str, Any]) -> pd.DataFrame:
    """Construct rolling-origin annual fold metadata."""

    rows = []
    seen_names = set()
    for fold in validation["folds"]:
        if fold["name"] in seen_names:
            raise LeakageError(f"Duplicate fold name: {fold['name']}")
        seen_names.add(fold["name"])
        assert_no_fold_leakage(fold["train_end"], fold["valid_start"], fold["valid_end"])
        valid_dates = pd.date_range(fold["valid_start"], fold["valid_end"], freq="MS")
        if len(valid_dates) != int(validation["horizon"]):
            raise LeakageError(
                f"Fold {fold['name']} has horizon {len(valid_dates)}, "
                f"expected {validation['horizon']}"
            )
        rows.append(
            {
                "fold_name": fold["name"],
                "train_end": fold["train_end"],
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "fold_role": fold["role"],
                "horizon": len(valid_dates),
                "expected_horizon": validation["horizon"],
                "season_length": validation["season_length"],
            }
        )
    return pd.DataFrame(rows)


def make_fold_slices(
    series_df: pd.DataFrame,
    fold: dict[str, Any] | pd.Series,
    *,
    acquisition_date: str = "2019-07-01",
    date_col: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-checked train/validation slices for one temporal fold."""

    fold_record = fold.to_dict() if isinstance(fold, pd.Series) else dict(fold)
    train_end = fold_record["train_end"]
    valid_start = fold_record["valid_start"]
    valid_end = fold_record["valid_end"]

    frame = series_df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    train = frame[frame[date_col] <= pd.Timestamp(train_end)].reset_index(drop=True)
    valid = frame[
        (frame[date_col] >= pd.Timestamp(valid_start))
        & (frame[date_col] <= pd.Timestamp(valid_end))
    ].reset_index(drop=True)

    assert_fold_datasets_no_leakage(
        train,
        valid,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        acquisition_date=acquisition_date,
        date_col=date_col,
    )
    return train, valid
