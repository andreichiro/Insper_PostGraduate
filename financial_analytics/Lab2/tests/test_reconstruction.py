import pandas as pd

from pads_forecasting.pipelines.reconstruction.nodes import _strategy_frame


def _panel():
    dates = pd.date_range("2019-05-01", periods=4, freq="MS")
    return pd.DataFrame(
        {
            "data": dates,
            "br_publicado": [100.0, 110.0, 200.0, 210.0],
            "adquirida_separada": [10.0, 20.0, pd.NA, pd.NA],
            "target_source": ["observed_br_standalone"] * 2 + ["observed_consolidated"] * 2,
            "covid_shock": 0,
            "covid_recovery": 0,
            "month": [5, 6, 7, 8],
            "trend_index": [0, 1, 2, 3],
        }
    )


def test_proforma_never_adds_acquired_after_acquisition():
    out = _strategy_frame(
        _panel(),
        name="proforma_sum",
        acquisition_date="2019-07-01",
        alpha=1.0,
    )

    assert out.loc[out["data"].eq(pd.Timestamp("2019-06-01")), "y"].iloc[0] == 130.0
    assert out.loc[out["data"].eq(pd.Timestamp("2019-07-01")), "y"].iloc[0] == 200.0


def test_calibrated_alpha_multiplies_only_acquired_company():
    out = _strategy_frame(
        _panel(),
        name="calibrated_alpha",
        acquisition_date="2019-07-01",
        alpha=0.5,
    )

    assert out.loc[out["data"].eq(pd.Timestamp("2019-06-01")), "y"].iloc[0] == 120.0


def test_post_only_starts_at_acquisition():
    out = _strategy_frame(_panel(), name="post_only", acquisition_date="2019-07-01")

    assert out["data"].min() == pd.Timestamp("2019-07-01")
