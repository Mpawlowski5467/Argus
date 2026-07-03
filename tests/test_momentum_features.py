"""Price-feature (momentum) wiring: off by default, PIT + survivorship-safe when on.

Momentum is the first PRICE-derived feature (attached as-of the rebalance date from
the shared close matrix, not carried on the filing row). These tests pin the two
things that make it safe to add: it is point-in-time (only past closes) and it flows
through the SAME shared transforms (attach + sector-rank) both train and serve use.
"""

import numpy as np
import pandas as pd

from stockscan.features import FEATURES
from stockscan.fundamental_panel import (
    attach_price_features,
    build_fundamental_panel,
    price_feature_matrices,
)
from stockscan.panel import momentum_12_1


def _features(filed="2023-01-02", sic=3571):
    rows = []
    for cik in (1, 2, 3):
        r = {"cik": cik, "filed_date": pd.Timestamp(filed), "sic": sic}
        r.update({f: 0.1 * cik for f in FEATURES})
        rows.append(r)
    return pd.DataFrame(rows)


def _close_long():
    """~450 trading days so 12-1 momentum (252d lookback) is defined late in the window."""
    idx = pd.bdate_range("2022-06-01", "2024-03-29")
    n = len(idx)
    return pd.DataFrame(
        {
            "AAA": np.linspace(50.0, 150.0, n),                   # climber -> high momentum
            "BBB": np.linspace(150.0, 60.0, n),                   # decliner -> low momentum
            "CCC": 100.0 + 20.0 * np.sin(np.linspace(0, 6, n)),   # choppy
        },
        index=idx,
    )


def test_price_features_off_by_default():
    """Baseline build is byte-identical: no momentum columns unless explicitly enabled."""
    panel = build_fundamental_panel(
        _features(), _close_long(), delistings=None,
        ticker_map={1: "AAA", 2: "BBB", 3: "CCC"}, horizon=21, min_names=1,
    )
    assert not panel.empty
    for c in ("mom_12_1", "mom_6_1", "mom_12_1_rank", "mom_6_1_rank"):
        assert c not in panel.columns


def test_price_features_attached_and_pit():
    close = _close_long()
    panel = build_fundamental_panel(
        _features(), close, delistings=None, ticker_map={1: "AAA", 2: "BBB", 3: "CCC"},
        horizon=21, min_names=1, price_features=True,
    )
    assert not panel.empty
    for c in ("mom_12_1", "mom_6_1", "mom_12_1_rank", "mom_6_1_rank"):
        assert c in panel.columns
    r = panel["mom_12_1_rank"].dropna()
    assert ((r >= 0) & (r <= 1)).all()  # ranks are proper percentiles

    # PIT: the attached value equals the as-of-date matrix lookup (only PAST closes).
    d = panel["date"].max()
    aaa_row = panel[(panel["cik"] == 1) & (panel["date"] == d)].iloc[0]
    assert abs(aaa_row["mom_12_1"] - momentum_12_1(close).loc[d, "AAA"]) < 1e-12

    # sanity on the signal's direction: the steady climber outranks the steady decliner
    late = panel[panel["date"] == d]
    assert late.loc[late["cik"] == 1, "mom_12_1_rank"].iloc[0] > \
        late.loc[late["cik"] == 2, "mom_12_1_rank"].iloc[0]


def test_attach_price_features_keys_by_price_column_and_nan_safe():
    """The shared attach maps by price COLUMN (dead names = TICKER~CIK) and never crashes."""
    close = _close_long()
    close["DDD~9"] = close["AAA"] * 0.5  # a dead name's column
    mats = price_feature_matrices(close)
    price_date = close.index[-1]
    cross = pd.DataFrame({"cik": [1, 9, 7], "ticker": ["AAA", "DDD~9", "ZZZ"]})
    out = attach_price_features(cross, price_date, mats)

    assert abs(out.loc[0, "mom_12_1"] - mats["mom_12_1"].loc[price_date, "AAA"]) < 1e-12
    assert abs(out.loc[1, "mom_12_1"] - mats["mom_12_1"].loc[price_date, "DDD~9"]) < 1e-12
    assert np.isnan(out.loc[2, "mom_12_1"])  # ZZZ has no price column -> NaN, not a KeyError
