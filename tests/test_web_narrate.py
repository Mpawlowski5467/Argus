"""The web caching layer = a per-cik analyze cache + the tiered narration wiring.

The narration machinery itself is covered in test_narrate_cache; here we check the
adapters: ArgusData.ticker memoizes serve.analyze per cik (deep copies in AND out,
so no caller can poison the stored result) and refresh()/reload() retire it; and
ArgusData.narrate wires the web button through NarrationCache/narrate_smart —
cache hit -> instant, tiers by materiality, template never cached, current news
attached without busting the durable hash."""

import json

import numpy as np
import pandas as pd

from stockscan.narrate.cache import NarrationCache
from stockscan.view.data import ArgusData


# -- the per-cik analyze cache ------------------------------------------------------

class _Stub:
    """Minimal ServeData stand-in — enough for refresh() and the cache paths."""

    def __init__(self):
        self.ticker_map = {5: "AAA"}
        self.close = pd.DataFrame({"AAA": [10.0, 20.0]},
                                  index=pd.to_datetime(["2026-07-02", "2026-07-03"]))


def _analyze_res(cik=5):
    return {"cik": cik, "percentile": 91, "decile": 10,
            "packet": {"meta": {"cik": cik, "ticker": "AAA"}}}


def _patched_facade(monkeypatch, calls):
    import stockscan.serve as serve_mod

    def fake_analyze(query, as_of=None, data=None, artifact=None, llm=None):
        calls.append((query, as_of))
        return _analyze_res(int(query))

    monkeypatch.setattr(serve_mod, "analyze", fake_analyze)
    return ArgusData(data=_Stub(), artifact=None, as_of=pd.Timestamp("2026-07-04"))


def test_facade_ticker_memoizes_analyze_per_cik(monkeypatch):
    calls = []
    ad = _patched_facade(monkeypatch, calls)
    r1 = ad.ticker(5)
    r2 = ad.ticker(5)
    assert len(calls) == 1 and r2 == r1        # one cross-section scoring, same answer
    ad.ticker(7)
    assert len(calls) == 2                     # per cik, not one global entry


def test_facade_ticker_hands_out_copies_never_the_cached_dict(monkeypatch):
    ad = _patched_facade(monkeypatch, [])
    r1 = ad.ticker(5)
    r1["packet"]["context"] = {"news": ["live-view leak"]}   # narrate mutates in place
    r2 = ad.ticker(5)
    assert "context" not in r2["packet"]       # the cache stays pristine (firewall)
    r2["percentile"] = 1
    assert ad.ticker(5)["percentile"] == 91    # and hits are copies too


def test_facade_ticker_never_caches_failures_or_explicit_as_of(monkeypatch):
    import pytest
    import stockscan.serve as serve_mod

    calls = []
    ad = _patched_facade(monkeypatch, calls)
    ad.ticker(5, as_of="2026-06-30")           # a historical read is not this view
    ad.ticker(5, as_of="2026-06-30")
    assert len(calls) == 2                     # bypasses: analyze runs each time

    def boom(query, as_of=None, data=None, artifact=None, llm=None):
        raise ValueError("no 10-K available point-in-time")

    monkeypatch.setattr(serve_mod, "analyze", boom)
    with pytest.raises(ValueError):
        ad.ticker(9)
    monkeypatch.setattr(serve_mod, "analyze",
                        lambda q, as_of=None, data=None, artifact=None, llm=None:
                        calls.append((q, as_of)) or _analyze_res(int(q)))
    assert ad.ticker(9)["cik"] == 9            # the error was never cached


def test_refresh_drops_the_analyze_cache(monkeypatch):
    import stockscan.serve as serve_mod

    calls = []
    ad = _patched_facade(monkeypatch, calls)
    monkeypatch.setattr(serve_mod, "build_cross_section",
                        lambda data, as_of: pd.DataFrame({"cik": [5]}))

    class _Art:
        def score(self, cross):
            return np.zeros(len(cross))

    ad.artifact = _Art()
    ad.ticker(5)
    ad.refresh()                               # rebuilt ranks -> stale answers dropped
    ad.ticker(5)
    assert len(calls) == 2


def test_state_reload_swaps_in_a_fresh_facade(monkeypatch):
    """POST /reload rebuilds ArgusData from scratch — every session cache (the
    analyze cache above all) dies with the old instance, so chat can never answer
    from the pre-update cross-section."""
    import stockscan.view.data as data_mod
    from stockscan.web.state import AppState

    monkeypatch.setattr(data_mod.ArgusData, "load",
                        classmethod(lambda cls, as_of=None: cls(data=None, artifact=None)))
    monkeypatch.setattr(AppState, "start_load", lambda self: self._load())  # no thread
    st = AppState()
    st.start_load()
    old = st.adata
    old.__dict__["_analyze_cache"] = {5: "stale"}
    st.reload()
    assert st.status == "ready" and st.adata is not old
    assert "_analyze_cache" not in st.adata.__dict__


# -- the narration wiring (facade -> narrate_smart + NarrationCache) ----------------

def _packet(pct=80, period_end="2025-09-30", value=31.2):
    return {
        "meta": {"ticker": "AAA", "name": "ALPHA", "cik": 5, "fiscal_year": 2025,
                 "period_end": period_end, "sector": "Tech"},
        "signals": [{"id": "roa", "label": "Return on assets", "value": value,
                     "unit": "%", "pct_rank": 90, "direction": "higher-is-better"}],
        "composite": {"label": "x", "percentile": 75},
        "model": {"label": "x", "score": 0.01, "percentile": pct, "decile": 9,
                  "n_names": 100, "as_of": "2026-06-30",
                  "trained_through": "2026-03-31",
                  "drivers": [{"id": "roa", "label": "roa", "contribution": 0.001,
                               "direction": "supports"}]},
        "disclaimer": "Not investment advice.",
    }


def _counting_llm(tag, calls, seen=None):
    def llm(system, user):
        calls.append(tag)
        if seen is not None:
            seen["packet"] = json.loads(user)
        pkt = json.loads(user)
        s = pkt["signals"][0]
        return json.dumps({
            "reasoning": "",
            "summary": f"[{tag}] {s['label']} {s['value']}{s['unit']} at "
                       f"{s['pct_rank']}th pct.",
            "citations": [{"id": s["id"], "direction": "supports"}],
        })
    return llm


def _narr_facade(news=None):
    ad = ArgusData(data=None, artifact=None, as_of=pd.Timestamp("2026-07-04"))
    ad._news_context = lambda cik: news or []
    return ad


def test_facade_narrate_serves_cache_then_tiers_by_materiality(tmp_path):
    cache = NarrationCache(tmp_path / "narr.sqlite")
    calls = []
    full, light = _counting_llm("full", calls), _counting_llm("light", calls)
    ad = _narr_facade()

    r1 = ad.narrate(_packet(), llm_full=full, llm_light=light, cache=cache)
    assert r1["tier"] == "full" and calls == ["full"]          # cold start
    r2 = ad.narrate(_packet(), llm_full=full, llm_light=light, cache=cache)
    assert r2["tier"] == "cache" and calls == ["full"]         # 30-90s -> instant
    assert r2["narrative"] == r1["narrative"] and r2["source"] == "llm"
    r3 = ad.narrate(_packet(pct=82, value=31.4), llm_full=full, llm_light=light,
                    cache=cache)
    assert r3["tier"] == "light" and calls == ["full", "light"]  # wiggle -> 14B tier
    cache.close()


def test_facade_narrate_attaches_news_without_busting_the_cache(tmp_path):
    cache = NarrationCache(tmp_path / "narr.sqlite")
    calls, seen = [], {}
    a1 = {"id": "a1", "date": "2026-07-01", "source": "wire",
          "event_type": "guidance", "takeaway": "raised guidance"}
    ad = _narr_facade(news=[a1])

    r1 = ad.narrate(_packet(), llm_full=_counting_llm("full", calls, seen), cache=cache)
    assert r1["tier"] == "full+news"                           # fresh read sees news
    assert seen["packet"]["context"]["news"][0]["id"] == "a1"  # ...in the prompt

    # tomorrow's headline churn: different news, same fundamentals -> still a hit
    b2 = {**a1, "id": "b2", "takeaway": "analyst day scheduled"}
    r2 = _narr_facade(news=[b2]).narrate(_packet(), llm_full=_counting_llm("full", calls),
                                         cache=cache)
    assert r2["tier"] == "cache" and calls == ["full"]
    cache.close()


def test_facade_narrate_dead_endpoint_degrades_and_never_caches(tmp_path):
    cache = NarrationCache(tmp_path / "narr.sqlite")

    def dead(system, user):
        raise TimeoutError()

    ad = _narr_facade()
    r = ad.narrate(_packet(), llm_full=dead, llm_light=dead, cache=cache)
    assert r["tier"] == "template" and r["source"] == "template-fallback"
    assert r["narrative"]                       # grounded template, never a crash
    assert cache.get(5) is None                 # the degraded text never enters
    cache.close()
