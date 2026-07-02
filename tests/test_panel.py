"""Tests for the panel builder: shift math and cross-sectional excess."""

import numpy as np
import pandas as pd

from stockscan.panel import build_panel, forward_return, momentum_12_1


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
