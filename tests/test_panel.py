"""Tests for the panel builder: shift math and cross-sectional excess."""

import numpy as np
import pandas as pd

from stockscan.panel import build_panel, forward_return, forward_return_to_last, momentum_12_1


def _close(n=400, n_tickers=8):
    idx = pd.bdate_range("2023-01-02", periods=n)
    tickers = [chr(65 + i) for i in range(n_tickers)]
    data = {t: np.linspace(10, 20, n) * (1 + 0.01 * k) for k, t in enumerate(tickers)}
    return pd.DataFrame(data, index=idx)


def test_forward_return_matches_manual_shift():
    c = _close()
    fr = forward_return(c, horizon=63)
    t = 100
    for col in c.columns:
        expected = c[col].iloc[t + 63] / c[col].iloc[t] - 1
        assert abs(fr[col].iloc[t] - expected) < 1e-9


def test_forward_return_to_last_uses_terminal_price_for_dying_names():
    c = _close(n=200, n_tickers=2)
    # ticker B stops trading at position 100 (delisting) -- its real terminal price
    death, last_price = 100, c["B"].iloc[100]
    c.loc[c.index[death + 1]:, "B"] = np.nan

    fr = forward_return_to_last(c, horizon=63)
    plain = forward_return(c, horizon=63)
    t = 80  # window [80, 143] straddles the death at 100
    assert np.isnan(plain["B"].iloc[t])                      # plain shift: no label
    expected = last_price / c["B"].iloc[t] - 1
    assert abs(fr["B"].iloc[t] - expected) < 1e-9            # real last-trade return
    # continuously-traded names are untouched
    assert abs(fr["A"].iloc[t] - plain["A"].iloc[t]) < 1e-12
    # long-dead names get no label (no entry price at t)
    assert np.isnan(fr["B"].iloc[180])
    # a name whose LAST-EVER trade is the sampling date itself must get NaN, not a
    # fabricated 0.0 (there is no price strictly inside the forward window)
    assert np.isnan(fr["B"].iloc[death])


def test_momentum_matches_manual_shift():
    c = _close()
    m = momentum_12_1(c, lookback=252, skip=21)
    t, col = 300, "A"
    expected = c[col].iloc[t - 21] / c[col].iloc[t - 252] - 1
    assert abs(m[col].iloc[t] - expected) < 1e-9


def test_build_panel_columns_and_zero_mean_excess():
    p = build_panel(_close(n=400), horizon=63)
    assert {"date", "ticker", "feature", "label", "label_excess"}.issubset(p.columns)
    if len(p):
        per_date_mean = p.groupby("date")["label_excess"].mean().abs()
        assert (per_date_mean < 1e-9).all()  # excess is de-meaned within each date
