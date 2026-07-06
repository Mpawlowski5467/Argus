"""The shared horizon table is the single source of truth for the 1w/1m/3m/1y chips.

Three surfaces must agree on the horizon set or they silently drift: view.chart's
price lookbacks, assist.move's coincidence windows, and the static/app.js chips.
These tests pin all three to stockscan.horizons so a change in one can't slip past."""

import re

from stockscan.assist import move
from stockscan.config import REPO_ROOT
from stockscan.horizons import BY_KEY, HORIZONS, KEYS
from stockscan.view.chart import price_summary


def test_table_is_self_consistent():
    assert KEYS == tuple(h.key for h in HORIZONS)
    assert set(BY_KEY) == set(KEYS) and all(BY_KEY[k].key == k for k in KEYS)
    # calendar windows are looser than the trading-day lookbacks (weekend/holiday
    # slack) and both strictly increase across horizons
    for h in HORIZONS:
        assert h.calendar_days > h.trading_days > 0
    assert [h.trading_days for h in HORIZONS] == sorted(h.trading_days for h in HORIZONS)
    assert [h.calendar_days for h in HORIZONS] == sorted(h.calendar_days for h in HORIZONS)


def test_move_horizons_derive_from_the_table():
    # assist.move keeps its {key: {label, days, window}} shape, but every value now
    # comes from the shared table — days is the CALENDAR window
    assert list(move.HORIZONS) == list(KEYS)
    for k, h in BY_KEY.items():
        assert move.HORIZONS[k] == {"label": h.label, "days": h.calendar_days,
                                    "window": h.window}


def test_price_summary_keys_and_lookbacks_track_the_table():
    ramp = [100.0 + i for i in range(300)]          # strictly increasing, no NaNs
    ps = price_summary(ramp)
    # exactly one chg_<key> per horizon, and nothing else shaped like a change field
    assert {k for k in ps if k.startswith("chg_")} == {f"chg_{k}" for k in KEYS}
    last = ramp[-1]
    for h in HORIZONS:
        old = ramp[-h.trading_days - 1]              # the TRADING-day lookback
        assert ps[f"chg_{h.key}"] == (last / old - 1.0) * 100.0
    # the empty-series path exposes the same key set (all None)
    empty = price_summary([float("nan")])
    assert {k for k in empty if k.startswith("chg_")} == {f"chg_{k}" for k in KEYS}
    assert all(empty[f"chg_{k}"] is None for k in KEYS)


def test_app_js_chips_match_the_table():
    """static/app.js can't import the Python table, so pin its chip keys here —
    a chip added/removed/reordered in the JS without touching horizons.py fails."""
    js = (REPO_ROOT / "static" / "app.js").read_text()
    pairs = re.findall(r'chip\("(\w+)",\s*s\.chg_(\w+)\)', js)
    assert pairs, "no explain-move chips found in app.js — did the render change?"
    keys = [k for k, _ in pairs]
    assert keys == list(KEYS)                        # same set AND order as the table
    assert all(k == field for k, field in pairs)     # chip('1w', s.chg_1w) stays aligned
