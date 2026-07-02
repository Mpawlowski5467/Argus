"""Test the survivorship correction: dead names excluded, dying names re-injected."""

import numpy as np
import pandas as pd

from stockscan.features import FEATURES
from stockscan.fundamental_panel import build_fundamental_panel


def _features():
    rows = []
    for cik in (1, 2, 3):
        r = {"cik": cik, "filed_date": pd.Timestamp("2023-06-01"), "sic": 3571}
        r.update({f: 0.1 * cik for f in FEATURES})
        rows.append(r)
    return pd.DataFrame(rows)


def _close():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    return pd.DataFrame({"AAA": np.linspace(100.0, 130.0, len(idx))}, index=idx)


def test_panel_imputes_dying_name_and_excludes_dead():
    delistings = pd.DataFrame(
        {
            "cik": [2, 3],
            "delist_date": pd.to_datetime(["2024-02-15", "2023-12-01"]),
            "reason": ["dereg", "delist"],
        }
    )
    ticker_map = {1: "AAA", 2: "BBB", 3: "CCC"}  # only AAA has prices

    panel = build_fundamental_panel(
        _features(), _close(), delistings=delistings, ticker_map=ticker_map,
        horizon=21, min_names=1,
    )
    jan = panel[panel["date"] == pd.Timestamp("2024-01-31")]
    ciks = set(jan["cik"])
    assert 1 in ciks   # priced survivor -> real label
    assert 2 in ciks   # dies within the window, no price -> re-injected via imputation
    assert 3 not in ciks  # delisted before the date -> excluded (was alive nowhere in-sample)

    bbb = jan.loc[jan["cik"] == 2].iloc[0]
    assert abs(bbb["label"] - (-1.00)) < 1e-9  # 'dereg' = going-dark -> -1.00 (from config)
    assert "coverage" in panel.attrs  # survivorship gap is recorded
