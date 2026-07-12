"""Market-context reads: display-only facts, honest about insufficient history."""

import numpy as np
import pandas as pd

from stockscan.view.regime import compute_regime, regime_line


def _matrix(days=600, cols=("UP", "DOWN", "FLAT")):
    """Deterministic shapes: a riser (always above its trailing mean), a faller
    (always below), and an oscillator (nonzero vol, no trend)."""
    idx = pd.bdate_range("2024-01-02", periods=days)
    t = np.arange(days)
    wiggle = 1 + 0.002 * np.sin(t / 5.0)     # nonzero vol without changing the trend
    shapes = {
        "UP": 100 * 1.002 ** t * wiggle,
        "DOWN": 100 * 0.998 ** t * wiggle,
        "FLAT": 100 + np.sin(t / 3.0),
    }
    return pd.DataFrame({c: shapes[c] for c in cols}, index=idx)


def test_regime_reads_are_sane():
    px = _matrix()
    r = compute_regime(px)
    assert r is not None and r["n_names"] == 3
    # a steady up-drift name sits above its 200d mean; a down-drift one below
    assert 0.0 < r["breadth_above_200d"] < 1.0
    assert r["median_vol_ann"] > 0
    assert -1.0 <= r["ew_drawdown"] <= 0.0
    assert r["as_of"] == str(px.index[-1].date())
    assert "not a model input" in r["note"]


def test_regime_respects_ticker_subset_and_as_of():
    px = _matrix()
    r_all = compute_regime(px)
    r_up = compute_regime(px, tickers=["UP"])
    assert r_up["n_names"] == 1 and r_up["breadth_above_200d"] == 1.0
    r_old = compute_regime(px, as_of=px.index[-50])
    assert r_old["as_of"] == str(px.index[-50].date())
    assert r_all["as_of"] != r_old["as_of"]
    assert compute_regime(px, tickers=["ABSENT"]) is None


def test_regime_refuses_thin_history():
    assert compute_regime(_matrix(days=100)) is None
    assert compute_regime(pd.DataFrame()) is None
    assert compute_regime(None) is None


def test_regime_line_wording():
    r = {"breadth_above_200d": 0.62, "median_vol_ann": 0.28, "vol_pctile_5y": 71,
         "ew_drawdown": -0.041}
    line = regime_line(r)
    assert line == ("62% of names above their 200d · median vol 28%/yr "
                    "(71st pctile, 5y) · equal-weight universe 4% off its high")
    at_high = regime_line({**r, "ew_drawdown": -0.001})
    assert "at its high" in at_high
    assert regime_line(None) == ""
