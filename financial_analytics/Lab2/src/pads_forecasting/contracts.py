"""Data contracts and compact validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DataContractError(ValueError):
    """Raised when a data contract is violated."""


@dataclass(frozen=True)
class ValidationRow:
    check: str
    passed: bool
    observed: str
    expected: str


def validation_frame(rows: Iterable[ValidationRow]) -> pd.DataFrame:
    """Return a validation table suitable for artifact/reporting output."""

    return pd.DataFrame([row.__dict__ for row in rows])


def normalize_monthly_frame(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    value_name: str,
) -> pd.DataFrame:
    """Normalize a two-column monthly data frame."""

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.rename(columns={date_col: "data", value_col: value_name})
    out = out.sort_values("data").reset_index(drop=True)
    out[value_name] = pd.to_numeric(out[value_name], errors="raise")
    return out


def _expected_months(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def validate_monthly_series(
    df: pd.DataFrame,
    *,
    name: str,
    value_col: str,
    expected_start: str,
    expected_end: str,
    expected_rows: int,
) -> list[ValidationRow]:
    """Validate monthly ordering, bounds, row counts, and values."""

    rows: list[ValidationRow] = []
    expected_start_ts = pd.Timestamp(expected_start)
    expected_end_ts = pd.Timestamp(expected_end)
    expected_index = _expected_months(expected_start_ts, expected_end_ts)

    rows.append(
        ValidationRow(
            f"{name}_row_count",
            len(df) == expected_rows,
            str(len(df)),
            str(expected_rows),
        )
    )
    rows.append(
        ValidationRow(
            f"{name}_start",
            df["data"].min() == expected_start_ts,
            str(df["data"].min().date()),
            str(expected_start_ts.date()),
        )
    )
    rows.append(
        ValidationRow(
            f"{name}_end",
            df["data"].max() == expected_end_ts,
            str(df["data"].max().date()),
            str(expected_end_ts.date()),
        )
    )
    rows.append(
        ValidationRow(
            f"{name}_first_day_of_month",
            bool((df["data"].dt.day == 1).all()),
            str(sorted(df.loc[df["data"].dt.day != 1, "data"].astype(str).unique())[:3]),
            "all dates on first day of month",
        )
    )
    rows.append(
        ValidationRow(
            f"{name}_no_duplicates",
            not df["data"].duplicated().any(),
            str(int(df["data"].duplicated().sum())),
            "0 duplicate dates",
        )
    )
    missing = expected_index.difference(pd.DatetimeIndex(df["data"]))
    rows.append(
        ValidationRow(
            f"{name}_no_missing_months",
            len(missing) == 0,
            ",".join(str(ts.date()) for ts in missing[:5]),
            "no missing monthly periods",
        )
    )
    rows.append(
        ValidationRow(
            f"{name}_non_negative",
            bool((df[value_col] >= 0).all()),
            str(int((df[value_col] < 0).sum())),
            "0 negative values",
        )
    )
    rows.append(
        ValidationRow(
            f"{name}_no_2024",
            bool((df["data"] < pd.Timestamp("2024-01-01")).all()),
            str(df["data"].max().date()),
            "max date before 2024-01-01",
        )
    )
    return rows


def assert_validation_passed(rows: Iterable[ValidationRow]) -> None:
    """Raise if any validation row failed."""

    failed = [row for row in rows if not row.passed]
    if failed:
        details = "; ".join(
            f"{row.check}: observed={row.observed}, expected={row.expected}" for row in failed
        )
        raise DataContractError(details)


def ensure_dir(path: str | Path) -> None:
    """Create a directory if needed."""

    Path(path).mkdir(parents=True, exist_ok=True)


def validate_previsao_shape(df: pd.DataFrame) -> None:
    """Validate final assignment forecast file shape."""

    expected_cols = ["data", "previsao"]
    if list(df.columns) != expected_cols:
        raise DataContractError(
            f"previsao.csv columns must be {expected_cols}, got {list(df.columns)}"
        )
    if len(df) != 12:
        raise DataContractError(f"previsao.csv must have 12 rows, got {len(df)}")
