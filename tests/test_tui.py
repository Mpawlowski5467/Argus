"""argus TUI: pure row-shapers (no textual) + a headless boot/navigation smoke test."""

import asyncio

import pandas as pd
import pytest

from stockscan.tui.data import scan_rows, sectors_in, status_dict, watch_rows


# --- pure facade helpers (no textual, no real data) -----------------------------

def _cross(**over):
    base = {"cik": [1, 2, 3], "ticker": ["A", "B", "C"], "name": ["Alpha", "Beta", "Gamma"],
            "sector": ["Tech", "Fin", "Tech"], "score": [0.1, 0.3, 0.2],
            "pct": [0.40, 1.00, 0.70], "fy": [2025, 2025, 2025]}
    base.update(over)
    return pd.DataFrame(base)


def test_scan_rows_sort_and_shape():
    rows = scan_rows(_cross())
    assert [r["ticker"] for r in rows] == ["B", "C", "A"]  # score desc
    assert rows[0]["decile"] == 10 and rows[0]["pct"] == 100
    assert rows[-1]["decile"] == 4


def test_scan_rows_sector_filter():
    tech = scan_rows(_cross(), "Tech")
    assert {r["ticker"] for r in tech} == {"A", "C"}
    assert scan_rows(_cross(), "all") == scan_rows(_cross())


def test_sectors_in():
    assert sectors_in(_cross()) == ["all", "Fin", "Tech"]


def test_watch_rows_delta_and_missing_flag():
    cross = _cross(cik=[1], ticker=["A"], name=["Alpha"], sector=["Tech"],
                   score=[0.2], pct=[0.90], fy=[2025])
    feats = pd.DataFrame({"cik": [1, 2],
                          "available_date": pd.to_datetime(["2025-03-01", "2024-03-01"])})
    wl = [{"cik": 1, "column": "A"}, {"cik": 2, "column": "DEAD~2"}]
    prev = {1: {"percentile": 80}, 2: {}}
    rows = watch_rows(wl, cross, prev, feats)
    assert rows[0]["delta"] == 10 and rows[0]["pct"] == 90     # 90 - 80
    assert rows[0]["last_filing"] == "2025-03-01"
    assert rows[1]["pct"] is None and rows[1]["flag"] is not None  # not in liquid cross


def test_status_dict_shape(tmp_path):
    from stockscan.ops.state import OpsState

    class _Art:
        pass
    with OpsState(tmp_path / "s.sqlite") as st:
        st.add_alert("x", "msg")
        s = status_dict(st, _Art(), pd.Timestamp("2026-06-30"))
    assert s["as_of"] == "2026-06-30"
    assert s["unseen_alerts"] == 1
    assert "vintage" in s and "nightly_status" in s


# --- headless app smoke test (injected fake facade) -----------------------------

class FakeData:
    def sectors(self):
        return ["all", "Tech", "Fin"]

    def scan(self, sector=None):
        rows = [
            {"rank": 1, "cik": 320193, "ticker": "AAPL", "name": "Apple", "sector": "Tech",
             "pct": 96, "decile": 10, "fy": 2025},
            {"rank": 2, "cik": 19617, "ticker": "JPM", "name": "JPMorgan", "sector": "Fin",
             "pct": 85, "decile": 9, "fy": 2025},
        ]
        return [r for r in rows if not sector or sector == "all" or r["sector"] == sector]

    def watch(self):
        return {"rows": [{"cik": 320193, "ticker": "AAPL", "pct": 96, "decile": 10,
                          "delta": 2, "last_filing": "2025-11-01", "flag": None}],
                "alerts": [{"created": "2026-07-01T00:00", "kind": "filing_detected",
                            "message": "cik 320193 filed", "seen": False}]}

    def paper(self):
        return None

    def status(self):
        return {"as_of": "2026-06-30", "fund_quarter": "2026q1", "vintage": "b50bc6d9",
                "artifact_registered": True, "nightly_status": "ok", "unseen_alerts": 1}

    def ticker(self, cik, as_of=None):
        return {
            "packet": {
                "meta": {"name": "Apple", "ticker": "AAPL", "sector": "Tech",
                         "fiscal_year": 2025},
                "model": {"percentile": 96, "decile": 10, "score": 0.0142,
                          "trained_through": "2026-03-31",
                          "drivers": [{"label": "roa", "contribution": 0.018,
                                       "direction": "supports"}]},
                "signals": [{"label": "roa", "value": 31.2, "unit": "%",
                             "pct_rank": 98, "read": "supports"}]},
            "column": "AAPL", "narrative": "Apple ranks high on profitability.",
            "flags": {"in_sample": True, "liquidity_pass": True, "filed_date": "2025-11-01",
                      "available_date": "2025-11-02", "staleness_days": 240}}

    def refresh(self, as_of=None):
        pass

    def close(self):
        pass


def test_argus_boots_switches_and_themes():
    pytest.importorskip("textual")
    from textual.widgets import ContentSwitcher, DataTable

    from stockscan.tui.app import ArgusApp, TickerView

    async def scenario():
        app = ArgusApp(adata=FakeData())
        async with app.run_test() as pilot:
            await pilot.pause()
            sw = app.query_one(ContentSwitcher)
            assert sw.current == "scan"
            assert app.query_one("#scan-table", DataTable).row_count == 2

            before = app.theme
            await pilot.press("t")
            assert app.theme != before          # light/dark toggle works

            for key, view in (("3", "watch"), ("4", "paper"), ("2", "ticker"), ("1", "scan")):
                await pilot.press(key)
                assert sw.current == view

            app.show_ticker(320193)
            assert sw.current == "ticker"
            tv = app.query_one(TickerView)
            assert tv._res["packet"]["meta"]["name"] == "Apple"
            assert "Apple" in tv._build(tv._res)   # renders without error

    asyncio.run(scenario())
