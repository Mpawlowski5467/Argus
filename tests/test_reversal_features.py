"""Reversal / low-vol / Amihud price features: PIT + parity + NaN-safety.

These three are the Tier-0 price-feature candidates tested after momentum (see the
reversal section of RESULTS.md). Like momentum they attach as-of the rebalance date
from the shared price matrices and flow through the SAME attach + sector-rank
transforms train and serve use — so the two safety properties are pinned here too:
point-in-time (only PAST prices) and train/serve parity (one shared code path).

NaN-safety carries more weight here than for momentum: ``low_vol`` needs a long
history and ``amihud`` divides by dollar volume, so zero-volume days must be masked
(never a divide-by-zero) and a name with no price/volume column must map to NaN, not
crash. ``amihud`` also depends on the dollar-volume matrix, so the plumbing keeps it
OPTIONAL — omitting ``dv`` simply skips it.
"""

import numpy as np
import pandas as pd

from stockscan.features import FEATURES
from stockscan.fundamental_panel import (
    PRICE_FEATURES,
    attach_price_features,
    build_fundamental_panel,
    price_feature_matrices,
)
from stockscan.panel import amihud, low_vol, short_term_reversal

TMAP = {1: "AAA", 2: "BBB", 3: "CCC"}
DMID = 300  # a mid-window as-of position: >=126 past rows (low_vol) AND future rows after


def _features(filed="2023-01-02", sic=3571):
    rows = []
    for cik in (1, 2, 3):
        r = {"cik": cik, "filed_date": pd.Timestamp(filed), "sic": sic}
        r.update({f: 0.1 * cik for f in FEATURES})
        rows.append(r)
    return pd.DataFrame(rows)


def _close_long():
    """~450 trading days so low_vol (126d) and the 21d features are defined late."""
    idx = pd.bdate_range("2022-06-01", "2024-03-29")
    n = len(idx)
    return pd.DataFrame(
        {
            "AAA": np.linspace(50.0, 150.0, n),                    # smooth climber -> low vol, +st_rev
            "BBB": np.linspace(150.0, 60.0, n),                    # smooth decliner -> -st_rev
            "CCC": 100.0 + 20.0 * np.sin(np.linspace(0, 30, n)),   # choppy -> high vol
        },
        index=idx,
    )


def _dv(close):
    """Positive dollar-volume matrix aligned to ``close`` (CCC the thinnest -> most illiquid)."""
    return pd.DataFrame({"AAA": 5e6, "BBB": 3e6, "CCC": 1e6}, index=close.index)


def test_price_feature_matrices_route_to_the_right_functions():
    close = _close_long()
    dv = _dv(close)
    mats = price_feature_matrices(close, dv)
    assert set(mats) == set(PRICE_FEATURES)  # all five built when dv is supplied
    d = close.index[-1]
    pd.testing.assert_series_equal(mats["st_rev"].loc[d], short_term_reversal(close).loc[d])
    pd.testing.assert_series_equal(mats["low_vol"].loc[d], low_vol(close).loc[d])
    pd.testing.assert_series_equal(mats["amihud"].loc[d], amihud(close, dv).loc[d])


def test_short_term_reversal_construction_sign_and_pit():
    close = _close_long()
    sr = short_term_reversal(close)
    d = close.index[DMID]

    # exact construction: trailing 21-row return, only PAST closes
    expected = close["AAA"].iloc[DMID] / close["AAA"].iloc[DMID - 21] - 1.0
    assert abs(sr.loc[d, "AAA"] - expected) < 1e-12
    # raw sign: recent climber positive, recent decliner negative (the model learns the flip)
    assert sr.loc[d, "AAA"] > 0 > sr.loc[d, "BBB"]
    # PIT: dropping every row AFTER d does not change the as-of value
    assert abs(short_term_reversal(close.loc[:d]).loc[d, "AAA"] - sr.loc[d, "AAA"]) < 1e-12


def test_low_vol_is_nonpositive_orders_by_smoothness_and_pit():
    close = _close_long()
    lv = low_vol(close)
    d = close.index[DMID]

    assert (lv.loc[d].dropna() <= 0).all()          # negative of a std -> non-positive
    assert lv.loc[d, "AAA"] > lv.loc[d, "CCC"]       # smoother name ranks higher (less-negative)
    # PIT: only past daily returns feed the window
    assert abs(low_vol(close.loc[:d]).loc[d, "AAA"] - lv.loc[d, "AAA"]) < 1e-12


def test_amihud_masks_zero_volume_orders_by_illiquidity_and_pit():
    close = _close_long()
    dv = _dv(close)
    dv.iloc[DMID - 5, dv.columns.get_loc("AAA")] = 0.0  # a zero-volume day inside AAA's window
    am = amihud(close, dv)
    d = close.index[DMID]

    assert np.isfinite(am.loc[d, "AAA"])                 # zero-vol day masked, NOT a divide-by-zero
    assert int(np.isinf(am.to_numpy()).sum()) == 0       # no infinities anywhere in the matrix
    assert am.loc[d, "CCC"] > am.loc[d, "AAA"]           # thin + choppy = most illiquid
    # PIT: past closes + past dollar volume only
    assert abs(amihud(close.loc[:d], dv.loc[:d]).loc[d, "CCC"] - am.loc[d, "CCC"]) < 1e-12


def test_attach_new_features_key_by_price_column_and_nan_safe():
    """Shared attach maps by price COLUMN (dead names = TICKER~CIK) and never crashes."""
    close = _close_long()
    dv = _dv(close)
    close["DDD~9"] = close["AAA"] * 0.5  # a dead name's column
    dv["DDD~9"] = dv["AAA"]
    mats = price_feature_matrices(close, dv)
    price_date = close.index[-1]
    cross = pd.DataFrame({"cik": [1, 9, 7], "ticker": ["AAA", "DDD~9", "ZZZ"]})
    out = attach_price_features(cross, price_date, mats)

    for feat in ("st_rev", "low_vol", "amihud"):
        assert abs(out.loc[0, feat] - mats[feat].loc[price_date, "AAA"]) < 1e-12
        assert abs(out.loc[1, feat] - mats[feat].loc[price_date, "DDD~9"]) < 1e-12
        assert np.isnan(out.loc[2, feat])  # ZZZ has no price column -> NaN, not a KeyError


def test_panel_attaches_and_ranks_all_price_features_when_dv_present():
    close = _close_long()
    panel = build_fundamental_panel(
        _features(), close, delistings=None, ticker_map=TMAP, dollar_volume=_dv(close),
        horizon=21, min_names=1, price_features=True,
    )
    assert not panel.empty
    for f in ("mom_12_1", "mom_6_1", "st_rev", "low_vol", "amihud"):
        assert f in panel.columns and f"{f}_rank" in panel.columns
        r = panel[f"{f}_rank"].dropna()
        assert ((r >= 0) & (r <= 1)).all()  # ranks are proper percentiles


def test_amihud_skipped_without_dollar_volume():
    """amihud is dv-dependent: without dv it is silently absent (never ranked, never crashes)."""
    close = _close_long()
    mats = price_feature_matrices(close)  # no dv
    assert "amihud" not in mats
    assert set(mats) == {"mom_12_1", "mom_6_1", "st_rev", "low_vol"}

    panel = build_fundamental_panel(  # price_features on, dollar_volume=None
        _features(), close, delistings=None, ticker_map=TMAP,
        horizon=21, min_names=1, price_features=True,
    )
    assert not panel.empty
    assert "amihud" not in panel.columns and "amihud_rank" not in panel.columns
    assert "st_rev_rank" in panel.columns  # the price-only features still attach + rank


def test_new_price_features_off_by_default():
    """Default build carries none of the new columns — the shipped panel is unchanged."""
    panel = build_fundamental_panel(
        _features(), _close_long(), delistings=None, ticker_map=TMAP,
        dollar_volume=_dv(_close_long()), horizon=21, min_names=1,
    )
    assert not panel.empty
    for c in ("st_rev", "low_vol", "amihud", "st_rev_rank", "low_vol_rank", "amihud_rank"):
        assert c not in panel.columns
