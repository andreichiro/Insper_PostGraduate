import pandas as pd
import pytest

from pads_forecasting.leakage import LeakageError, assert_no_2024_used, assert_no_fold_leakage


def test_fold_overlap_fails():
    with pytest.raises(LeakageError):
        assert_no_fold_leakage("2021-01-01", "2021-01-01", "2021-12-01")


def test_2024_values_fail():
    df = pd.DataFrame({"data": pd.to_datetime(["2023-12-01", "2024-01-01"])})
    with pytest.raises(LeakageError):
        assert_no_2024_used(df)
