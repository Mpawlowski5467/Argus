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


def _close_with_death():
    """AAA trades throughout; DDD~2 collapses and stops trading mid-February."""
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    close = pd.DataFrame({"AAA": np.linspace(100.0, 130.0, len(idx))}, index=idx)
    dead = pd.Series(np.linspace(50.0, 5.0, 34), index=idx[:34])  # last trade 2024-02-15
    close["DDD~2"] = dead.reindex(idx)
    return close


def test_no_impute_uses_real_terminal_returns_and_never_imputes():
    """delistings=None (the --no-impute path): failures enter ONLY via real prices."""
    ticker_map = {1: "AAA", 2: "DDD~2", 3: "CCC"}  # cik 3 has no prices at all
    close = _close_with_death()

    panel = build_fundamental_panel(
        _features(), close, delistings=None, ticker_map=ticker_map,
        horizon=21, min_names=1,
    )
    assert not panel["imputed"].any()  # nothing imputed, ever
    jan = panel[panel["date"] == pd.Timestamp("2024-01-31")]
    assert set(jan["cik"]) == {1, 2}   # unpriced cik 3 drops; dying cik 2 stays via real prices

    # the dying name's label is its REAL terminal return: last trade / entry - 1
    entry = close.loc[pd.Timestamp("2024-01-31"), "DDD~2"]
    terminal = 5.0  # its final print
    ddd = jan.loc[jan["cik"] == 2].iloc[0]
    assert abs(ddd["label"] - (terminal / entry - 1)) < 1e-9
    assert ddd["label"] < -0.5  # the decline is captured, not dropped


def test_unadjusted_liquidity_price_changes_membership_not_return_labels():
    """A controlled rebaseline can use raw price for the liquidity floor while labels
    still come from adjusted total-return closes. Defaults remain adjusted-only."""
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    close = pd.DataFrame({
        "AAA": np.linspace(100.0, 110.0, len(idx)),
        "LOW": np.linspace(0.50, 0.60, len(idx)),  # adjusted below $1, raw will be $30
    }, index=idx)
    raw_price = close.copy()
    raw_price["LOW"] = 30.0
    dv = pd.DataFrame(2_000_000.0, index=idx, columns=close.columns)
    ticker_map = {1: "AAA", 2: "LOW", 3: "CCC"}

    default = build_fundamental_panel(
        _features(), close, delistings=None, ticker_map=ticker_map,
        horizon=21, min_names=1, dollar_volume=dv, min_dollar_volume=1_000_000,
    )
    rebased = build_fundamental_panel(
        _features(), close, delistings=None, ticker_map=ticker_map,
        horizon=21, min_names=1, dollar_volume=dv, min_dollar_volume=1_000_000,
        liquidity_price=raw_price,
    )

    jan_default = default[default["date"] == pd.Timestamp("2024-01-31")]
    jan_rebased = rebased[rebased["date"] == pd.Timestamp("2024-01-31")]
    assert set(jan_default["cik"]) == {1}
    assert set(jan_rebased["cik"]) == {1, 2}

    low = jan_rebased.loc[jan_rebased["cik"] == 2].iloc[0]
    expected_label = (
        close["LOW"].shift(-21).loc[pd.Timestamp("2024-01-31")]
        / close.loc[pd.Timestamp("2024-01-31"), "LOW"] - 1
    )
    assert abs(low["label"] - expected_label) < 1e-9
