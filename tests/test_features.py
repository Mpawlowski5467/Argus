"""Tests for fundamental ratio + growth computation."""

import pandas as pd

from stockscan.features import compute_features


def _wide():
    return pd.DataFrame(
        {
            "cik": [1, 1],
            "adsh": ["a", "b"],
            "sic": [3571, 3571],
            "form": ["10-K", "10-K"],
            "fy": [2022, 2023],
            "filed_date": pd.to_datetime(["2023-02-01", "2024-02-01"]),
            "period_end": pd.to_datetime(["2022-12-31", "2023-12-31"]),
            "revenue": [100.0, 120.0],
            "cost_of_revenue": [60.0, 70.0],
            "gross_profit": [40.0, 50.0],
            "operating_income": [20.0, 25.0],
            "net_income": [10.0, 12.0],
            "cfo": [15.0, 14.0],
            "assets": [200.0, 220.0],
            "assets_current": [80.0, 90.0],
            "liabilities": [120.0, 130.0],
            "liabilities_current": [50.0, 55.0],
            "equity": [80.0, 90.0],
            "cash": [30.0, 33.0],
            "equity_tag": [80.0, 90.0],
        }
    )


def test_ratios_and_growth():
    f = compute_features(_wide()).set_index("fy")
    assert abs(f.loc[2023, "roa"] - 12 / 220) < 1e-9
    assert abs(f.loc[2023, "gross_profitability"] - 50 / 220) < 1e-9
    assert abs(f.loc[2023, "accruals"] - (12 - 14) / 220) < 1e-9
    assert abs(f.loc[2023, "asset_growth"] - (220 / 200 - 1)) < 1e-9
    assert abs(f.loc[2023, "revenue_growth"] - (120 / 100 - 1)) < 1e-9
    # first year has no prior filing -> growth is undefined
    assert pd.isna(f.loc[2022, "asset_growth"])
