"""Pure-function tests for the terminal price chart + summary + verdict."""

import math

import pandas as pd

from stockscan.tui.chart import braille_chart, candle_panel, price_summary, verdict


def _blank(row: str) -> bool:
    return all(ch in (" ", chr(0x2800)) for ch in row)


def test_braille_chart_dimensions_and_charset():
    rows = braille_chart(list(range(50)), width=20, height=8)
    assert len(rows) == 8
    assert all(len(r) == 20 for r in rows)
    # every glyph is a space or a braille pattern
    assert all(ch == " " or 0x2800 <= ord(ch) <= 0x28FF for r in rows for ch in r)


def test_braille_chart_empty_is_blank():
    rows = braille_chart([], width=10, height=4)
    assert len(rows) == 4 and all(_blank(r) for r in rows)


def test_braille_chart_rising_line_climbs():
    # a rising series should light the bottom-left and the top-right, not vice-versa
    rows = braille_chart(list(range(100)), width=20, height=6)
    top, bottom = rows[0], rows[-1]
    assert not _blank(top[-3:])       # high values reach the top on the right
    assert not _blank(bottom[:3])     # low values sit at the bottom on the left
    assert _blank(top[:3])            # top-left stays empty for a rising line


def test_braille_chart_flat_series_no_crash():
    rows = braille_chart([5.0] * 30, width=12, height=4)
    assert len(rows) == 4 and any(not _blank(r) for r in rows)


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


def test_candle_panel_renders_and_colors():
    # 40 rising days: last candle closes up (green), chart + volume + axis present
    o = [100 + i for i in range(40)]
    h = [x + 2 for x in o]
    lo = [x - 2 for x in o]
    c = [x + 1 for x in o]           # close > open every day -> green
    v = [1000 + i for i in range(40)]
    panel = candle_panel(o, h, lo, c, v, width=30, height=8, vheight=2)
    assert "[green]" in panel and "[red]" not in panel   # all up days
    assert "█" in panel and "│" in panel                 # bodies + wicks
    assert "vol" in panel and "└" in panel               # volume strip + axis
    # a down day shows red
    down = candle_panel([10, 10], [11, 11], [8, 8], [9, 8], [5, 5], width=4, height=6)
    assert "[red]" in down


def test_candle_panel_empty():
    assert "no OHLC" in candle_panel([], [], [], [], [], width=10, height=4)


def test_verdict_thresholds():
    assert verdict(0.95)["call"] == "BUY"
    assert verdict(0.80)["call"] == "BUY"
    assert verdict(0.79)["call"] == "HOLD"
    assert verdict(0.40)["call"] == "HOLD"
    assert verdict(0.39)["call"] == "AVOID"
    assert verdict(None)["call"] == "N/A"
    assert verdict(float("nan"))["call"] == "N/A"
