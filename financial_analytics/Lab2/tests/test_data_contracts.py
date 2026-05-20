from pathlib import Path

import pandas as pd

from pads_forecasting.contracts import normalize_monthly_frame, validate_monthly_series

ROOT = Path(__file__).resolve().parents[1]


def test_raw_data_contracts_match_assignment_counts():
    main = pd.read_csv(ROOT / "data/01_raw/distribr_serie.txt")
    acquired = pd.read_csv(ROOT / "data/01_raw/distribr_adquirida.txt")
    main = normalize_monthly_frame(main, date_col="data", value_col="valor", value_name="valor")
    acquired = normalize_monthly_frame(
        acquired, date_col="data", value_col="valor", value_name="valor"
    )

    main_rows = validate_monthly_series(
        main,
        name="main",
        value_col="valor",
        expected_start="2014-01-01",
        expected_end="2023-12-01",
        expected_rows=120,
    )
    acquired_rows = validate_monthly_series(
        acquired,
        name="acquired",
        value_col="valor",
        expected_start="2014-01-01",
        expected_end="2019-06-01",
        expected_rows=66,
    )

    assert all(row.passed for row in main_rows + acquired_rows)
