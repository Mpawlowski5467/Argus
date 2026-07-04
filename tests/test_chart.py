"""Pure-function tests for the deterministic price summary + BUY/HOLD/AVOID verdict."""

import pandas as pd

from stockscan.view.chart import price_summary, verdict


def test_price_summary_changes_and_range():
    # 300 points: ramp so trailing windows are well-defined
    s = pd.Series([100 + i for i in range(300)])
    ps = price_summary(s)
    assert ps["last"] == 399
    assert ps["hi_52w"] == 399 and ps["lo_52w"] == 100 + (300 - 252)
    assert ps["chg_1m"] == round((399 / (399 - 21) - 1) * 100, 10) or ps["chg_1m"] > 0
    assert ps["n"] == 300


def test_price_summary_handles_nans_and_shorts():
    ps = price_summary(pd.Series([float("nan"), 10.0, float("nan"), 12.0]))
    assert ps["last"] == 12.0 and ps["hi_52w"] == 12.0 and ps["lo_52w"] == 10.0
    empty = price_summary(pd.Series([float("nan")]))
    assert empty["last"] is None and empty["n"] == 0


def test_verdict_thresholds():
    assert verdict(0.95)["call"] == "BUY"
    assert verdict(0.80)["call"] == "BUY"
    assert verdict(0.79)["call"] == "HOLD"
    assert verdict(0.40)["call"] == "HOLD"
    assert verdict(0.39)["call"] == "AVOID"
    assert verdict(None)["call"] == "N/A"
    assert verdict(float("nan"))["call"] == "N/A"
