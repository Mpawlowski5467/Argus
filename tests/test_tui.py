"""argus TUI: pure row-shapers (no textual) + a headless boot/navigation smoke test."""

import asyncio

import pandas as pd
import pytest

from stockscan.tui.data import scan_rows, search_rows, sectors_in, status_dict, watch_rows


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


def test_search_rows_by_ticker_and_name():
    cross = _cross(ticker=["AAPL", "MSFT", "JPM"], name=["Apple", "Microsoft", "JPMorgan"])
    assert [r["ticker"] for r in search_rows(cross, "app")] == ["AAPL"]     # ticker/name substr
    assert [r["ticker"] for r in search_rows(cross, "micro")] == ["MSFT"]   # by name
    assert search_rows(cross, "") == scan_rows(cross)                        # empty = full scan
    assert search_rows(cross, "zzz") == []                                  # no match


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

    def search(self, query, limit=40):
        return [r for r in self.scan("all")
                if query.upper() in r["ticker"] or query.upper() in r["name"].upper()]

    def resolve(self, query):
        return {"cik": 320193, "column": "AAPL"} if "AAP" in str(query).upper() else None

    def price(self, cik):
        import pandas as _pd
        from stockscan.tui.chart import price_summary
        s = _pd.Series([100 + (i % 20) + i * 0.5 for i in range(300)])
        return {"column": "AAPL", "series": s, "summary": price_summary(s, adv=9.9e8)}

    def events(self, cik, limit=8):
        return [{"filed_date": "2026-05-01", "form": "10-Q", "period_end": "2026-03-31",
                 "label": "quarterly report"}]

    def ohlc(self, cik, tail=252):
        import pandas as _pd
        n = 120
        base = [100 + (i % 15) + i * 0.4 for i in range(n)]
        return _pd.DataFrame({
            "date": _pd.date_range("2025-01-01", periods=n, freq="B"),
            "open": base, "high": [b + 2 for b in base], "low": [b - 2 for b in base],
            "close": [b + 0.5 for b in base], "volume": [1000 + i * 3 for i in range(n)]})

    def news(self, cik, limit=6):
        return [{"title": "Apple unveils a new thing", "date": "2026-07-01",
                 "url": "https://www.reuters.com/x", "source": "reuters.com"}]

    def live_quote(self, cik, refresh=False):
        return {"last": 309.12, "time": "2026-07-02T23:57:00Z", "bid": 309.0, "ask": 309.2,
                "prev_close": 308.63, "chg_pct": 0.16}

    def is_watched(self, cik):
        return int(cik) in self.__dict__.setdefault("_w", set())

    def toggle_watch(self, cik):
        s = self.__dict__.setdefault("_w", set())
        if int(cik) in s:
            s.discard(int(cik))
            return False
        s.add(int(cik))
        return True

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
                await pilot.pause()
                assert sw.current == view

            app.show_ticker(320193)
            assert sw.current == "ticker"
            tv = app.query_one(TickerView)
            assert tv._res["packet"]["meta"]["name"] == "Apple"
            page = tv._build(tv._res)
            assert "Apple" in page                 # renders without error
            assert "BUY" in page                   # deterministic verdict (96th pct)
            assert "1y" in page                    # price header + chart present
            assert "candle chart" in page          # candles are the default view
            # the enrichment worker is async; drive the callback deterministically
            tv.set_enrichment(320193, app.adata.events(320193),
                              app.adata.news(320193), app.adata.live_quote(320193))
            page2 = tv._build(tv._res)
            assert "10-Q" in page2                 # EDGAR filing/event
            assert "reuters.com" in page2          # Intrinio news headline
            assert "live" in page2                 # live quote line
            tv.toggle_chart()                      # candle <-> line
            assert "line chart" in tv._build(tv._res)

            # live search filters the scan table down to the match
            app.query_one(ContentSwitcher).current = "scan"
            app.query_one("#search").focus()
            await pilot.press(*"AAP")
            await pilot.pause()
            assert app.query_one("#scan-table", DataTable).row_count == 1

            # watchlist toggle from the ticker page (w)
            app.show_ticker(320193)
            await pilot.pause()
            assert "w to watch" in tv._build(tv._res)
            await pilot.press("w")
            await pilot.pause()
            assert "watching" in tv._build(tv._res)

            # live auto-refresh toggle (a)
            assert app._auto_timer is None
            await pilot.press("a")
            await pilot.pause()
            assert app._auto_timer is not None
            await pilot.press("a")
            await pilot.pause()
            assert app._auto_timer is None

            # help modal (? opens, esc closes)
            from stockscan.tui.app import HelpScreen
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpScreen)

    asyncio.run(scenario())
