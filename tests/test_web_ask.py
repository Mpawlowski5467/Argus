"""The web ask surface = a context builder + a facade method + a thin route.

The grounded-answer machinery itself is covered in test_assist; here we check the
three adapters: build_chat_context widens the packet with CITABLE display numbers
(display-rounded twins so UI phrasings ground) plus honest notes, without mutating
anything; the facade assembles packet + news + price + verdict and delegates; and
the route validates input, single-flights the LLM, and wires convert.jsonable."""

import json

import pandas as pd
import pytest

from stockscan.assist.qa import build_chat_context
from stockscan.narrate.ground import check_grounding
from stockscan.view.data import ArgusData


def _res():
    return {
        "packet": {
            "meta": {"cik": 1, "ticker": "AAA", "name": "Alpha", "sector": "Tech"},
            "model": {"score": 0.9123, "percentile": 91, "decile": 10},
            "signals": [{"label": "return on assets", "value": 12.0, "pct_rank": 80,
                         "read": "supports"}],
        },
        "percentile": 91, "decile": 10,
        "confidence": {"score": 62, "hit_rate": 0.551, "n": 1200},
        "distress": {"prob": 0.034, "percentile": 88, "flag": "elevated",
                     "horizon_months": 12},
        "drawdown": {"prob": 0.41, "percentile": 77, "flag": "elevated",
                     "horizon_months": 6, "threshold": -0.30},
        "flags": {"staleness_days": 41, "in_sample": False},
    }


# -- build_chat_context ------------------------------------------------------------

def test_chat_context_makes_display_numbers_citable():
    ctx = build_chat_context(_res(),
                             price_summary={"last": 101.2489, "chg_1y": 12.3456,
                                            "adv": 23_400_000.0},
                             verdict={"call": "BUY"})
    # the phrasings the UI shows must trace: "55.1% hit-rate", "P≈3.4%", "ADV $23.4M"
    text = ("confidence 62 out of 100 on a 55.1% hit-rate (n=1,200); distress P about "
            "3.4%; drawdown P 41% (a 30%+ fall); last close 101.25, up 12.3% on the "
            "year, ADV 23.4 million")
    assert check_grounding(text, ctx) == []
    # while a fabricated figure still violates
    assert check_grounding("revenue grew 47%", ctx) == [47.0]


def test_chat_context_carries_honest_notes_and_does_not_mutate():
    res = _res()
    before = json.dumps(res, sort_keys=True, default=str)
    ctx = build_chat_context(res, price_summary={"last": 1.0}, verdict={"call": "BUY"})
    assert json.dumps(res, sort_keys=True, default=str) == before      # pure
    d = ctx["display"]
    for k in ("verdict", "confidence", "distress", "drawdown", "price"):
        assert "note" in d[k]
    assert "never a trade input" in d["distress"]["note"]
    assert "not a return forecast" in d["verdict"]["note"]
    assert ctx["meta"]["ticker"] == "AAA"          # packet spread into the top level


def test_chat_context_omits_absent_blocks():
    res = _res()
    res["confidence"] = None
    res["drawdown"] = None
    ctx = build_chat_context(res)                  # no price, no verdict
    assert "confidence" not in ctx["display"] and "drawdown" not in ctx["display"]
    assert "price" not in ctx["display"] and "verdict" not in ctx["display"]
    assert "distress" in ctx["display"]


# -- the facade method -------------------------------------------------------------

def _stub_facade():
    ad = ArgusData(data=None, artifact=None, as_of=pd.Timestamp("2026-07-04"))
    ad.ticker = lambda cik, as_of=None: _res()
    ad.price = lambda cik: {"summary": {"last": 101.25, "chg_1y": 12.3}}
    ad._news_context = lambda cik: []
    return ad


def test_facade_ask_grounds_and_reports_meta():
    ad = _stub_facade()
    seen = {}

    def llm(system, user):
        seen["system"], seen["user"] = system, user
        return "ranked 91st percentile, decile 10 — confidence 62 out of 100"

    r = ad.ask(1, "why is it ranked here?", llm=llm)
    assert r["grounded"] and not r["refused"]
    assert r["cik"] == 1 and r["ticker"] == "AAA" and r["n_news"] == 0
    assert '"display"' in seen["user"]                  # the widened context went out
    assert "out-of-sample hit-rate" in seen["system"]   # CHAT_SYSTEM, not bare QA_SYSTEM


def test_facade_ask_dead_llm_refuses_never_crashes():
    ad = _stub_facade()

    def llm(system, user):
        raise TimeoutError()

    r = ad.ask(1, "why?", llm=llm)
    assert r["refused"] is True
    assert r["violations"] == ["llm-error:TimeoutError"]


def test_facade_ask_weaves_history_without_widening_grounding():
    ad = _stub_facade()
    seen = {}

    def llm(system, user):
        seen["user"] = user
        return "as said, the rank is unchanged"

    r = ad.ask(1, "and the decile?", llm=llm,
               history=[{"role": "user", "content": "why?"},
                        {"role": "assistant", "content": "ranked 91st"}])
    assert not r["refused"]
    assert "CONVERSATION SO FAR" in seen["user"] and "ranked 91st" in seen["user"]


# -- the route ---------------------------------------------------------------------

def test_route_ask_validates_cleans_history_and_wires_jsonable(monkeypatch):
    pytest.importorskip("fastapi")
    import numpy as np
    from fastapi import HTTPException

    from stockscan.web import routes
    from stockscan.web.state import STATE

    class _Facade:
        def ask(self, cik, question, history=None):
            assert history == [{"role": "user", "content": "a"}]   # junk stripped
            return {"answer": "ok", "refused": False, "cik": np.int64(cik)}

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    out = routes.ask(5, {"question": "why?", "history": [
        "junk", {"role": "assistant"}, {"role": "user", "content": "a"}]})
    assert out["answer"] == "ok"
    assert out["cik"] == 5 and isinstance(out["cik"], int)         # numpy coerced

    with pytest.raises(HTTPException) as e:
        routes.ask(5, {"question": "   "})
    assert e.value.status_code == 422
    with pytest.raises(HTTPException) as e:
        routes.ask(5, {"question": "x" * 2001})
    assert e.value.status_code == 422


def test_route_ask_single_flights_the_llm(monkeypatch):
    pytest.importorskip("fastapi")
    from stockscan.web import routes
    from stockscan.web.state import STATE

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", object())   # must never be reached
    assert routes._LLM_GATE.acquire(blocking=False)
    try:
        assert routes.ask(5, {"question": "busy?"}) == {"busy": True}
    finally:
        routes._LLM_GATE.release()
