"""The book ask surface = a context builder + a facade method + a thin route.

The grounded-answer machinery is covered in test_assist and the scorecard math in
test_portfolio; here we check the three adapters: build_book_context widens the
scorecard with CITABLE display numbers (rounded percentile / percent twins, the
flagged-value sum the UI prints) plus honest notes, without mutating anything; the
facade builds the scorecard and delegates under the PORTFOLIO system prompt; and
the route validates input, single-flights the LLM, wires convert.jsonable, and is
registered before /ask/{cik} so the literal path wins."""

import json

import pandas as pd
import pytest

from stockscan.assist.book import build_book_context
from stockscan.narrate.ground import check_grounding
from stockscan.view.data import ArgusData


def _sc():
    return {
        "as_of": "2026-07-04",
        "n_total": 3, "n_owned": 2, "n_watch": 1, "n_listed": 2, "n_unlisted": 1,
        "total_value": 3512.5678, "total_cost": 3000.0,
        "unrealized_pl": 512.5678, "unrealized_pl_pct": 17.0856,
        "percentile_equal": 57.5, "percentile_value": 62.5,
        "distress": {"known": True,
                     "count": {"high": 0, "elevated": 1, "normal": 1},
                     "value": {"high": 250.0, "elevated": 1500.0, "normal": 2000.0},
                     "at_risk": 1},
        "industry_concentration": [
            {"name": "Semiconductors", "count": 1,
             "weight_count": 0.5, "weight_value": 0.5714},
            {"name": "Banks", "count": 1, "weight_count": 0.5, "weight_value": 0.4286},
        ],
        "sector_concentration": [
            {"name": "Tech", "count": 1, "weight_count": 0.5, "weight_value": 0.5714},
        ],
        "holdings": [
            {"cik": 1, "owned": True, "in_universe": True, "status": "listed",
             "ticker": "AAA", "name": "Alpha", "industry": "Semiconductors",
             "shares": 100.0, "value": 2000.0, "cost": 1000.0,
             "unrealized_pl": 1000.0, "unrealized_pl_pct": 100.0,
             "pct": 90, "decile": 10, "dflag": "normal", "dprob": None},
            {"cik": 2, "owned": True, "in_universe": True, "status": "listed",
             "ticker": "BBB", "name": "Beta", "industry": "Banks",
             "shares": 50.0, "value": 1512.5678, "cost": 2000.0,
             "unrealized_pl": -487.4322, "unrealized_pl_pct": -24.3721,
             "pct": 30, "decile": 3, "dflag": "elevated", "dprob": 0.034},
            {"cik": 9, "owned": False, "in_universe": False,
             "status": "not in liquid universe / lapsed filer",
             "ticker": "—", "name": "", "industry": "Unknown",
             "shares": None, "value": None, "cost": None,
             "unrealized_pl": None, "unrealized_pl_pct": None,
             "pct": None, "decile": None, "dflag": None, "dprob": None},
        ],
    }


# -- build_book_context --------------------------------------------------------

def test_book_context_makes_display_numbers_citable():
    ctx = build_book_context(_sc())
    # the phrasings the book tab shows must trace: rounded percentiles ("58th"),
    # 1-dp P/L percents, whole-percent concentration, dprob as a percent, and the
    # flagged-value sum the distress line prints (250 + 1500 = 1750)
    text = ("the book ranks 58th equal-weight and 63rd value-weight; unrealized "
            "P/L up 17.1% on 3,512.57 of value; 57% of the money sits in "
            "Semiconductors; BBB is down 24.4% and carries distress P about 3.4%; "
            "1,750.00 of book value sits in flagged names")
    assert check_grounding(text, ctx) == []
    # while a fabricated figure still violates
    assert check_grounding("the book should return 8% a year", ctx) == [8.0]


def test_book_context_carries_honest_notes_and_does_not_mutate():
    sc = _sc()
    before = json.dumps(sc, sort_keys=True, default=str)
    ctx = build_book_context(sc)
    assert json.dumps(sc, sort_keys=True, default=str) == before        # pure
    assert "nothing here is a portfolio forecast" in ctx["note"]
    assert "cite both together" in ctx["weighting_note"]
    assert "never a trade input" in ctx["distress"]["note"]
    assert ctx["distress"]["value_at_risk"] == 1750.0
    assert ctx["percentile_equal_round"] == 58                # JS Math.round(57.5)
    assert ctx["percentile_value_round"] == 63
    assert ctx["holdings"][1]["dprob_pct"] == 3.4
    assert ctx["holdings"][1]["unrealized_pl_pct_round"] == 24.4        # abs, 1dp
    assert ctx["industry_concentration"][0]["weight_value_pct"] == 57


def test_book_context_omits_twins_for_absent_numbers():
    sc = _sc()
    sc["percentile_value"] = None                  # nothing held with a price yet
    sc["unrealized_pl_pct"] = None
    sc["distress"] = {"known": False, "count": {}, "value": None, "at_risk": 0}
    ctx = build_book_context(sc)
    assert "percentile_value_round" not in ctx
    assert "unrealized_pl_pct_round" not in ctx
    assert "value_at_risk" not in ctx["distress"]  # no per-flag values to sum
    assert "note" in ctx["distress"]               # the framing still rides along


def test_book_context_handles_an_empty_book():
    ctx = build_book_context({"n_total": 0, "holdings": [],
                              "industry_concentration": [], "sector_concentration": []})
    assert ctx["holdings"] == [] and ctx["industry_concentration"] == []
    assert "nothing here is a portfolio forecast" in ctx["note"]


# -- the facade method -----------------------------------------------------------

def _stub_facade():
    ad = ArgusData(data=None, artifact=None, as_of=pd.Timestamp("2026-07-04"))
    ad.scorecard = lambda: _sc()
    return ad


def test_facade_ask_book_grounds_and_reports_meta():
    ad = _stub_facade()
    seen = {}

    def llm(system, user):
        seen["system"], seen["user"] = system, user
        return "58th percentile equal-weight, 63rd value-weight — 3 names tracked"

    r = ad.ask_book("where does my book rank?", llm=llm)
    assert r["grounded"] and not r["refused"]
    assert r["n_names"] == 3 and r["as_of"] == "2026-07-04"
    assert '"percentile_equal_round"' in seen["user"]   # the widened scorecard went out
    assert "both weightings" in seen["system"]          # PORTFOLIO_SYSTEM, not QA_SYSTEM
    assert "ONE company" not in seen["system"]


def test_facade_ask_book_dead_llm_refuses_never_crashes():
    ad = _stub_facade()

    def llm(system, user):
        raise TimeoutError()

    r = ad.ask_book("how concentrated am I?", llm=llm)
    assert r["refused"] is True
    assert r["violations"] == ["llm-error:TimeoutError"]


def test_facade_ask_book_weaves_history_without_widening_grounding():
    ad = _stub_facade()
    seen = {}

    def llm(system, user):
        seen["user"] = user
        return "as said, the rank is unchanged"

    r = ad.ask_book("and value-weighted?", llm=llm,
                    history=[{"role": "user", "content": "where does it rank?"},
                             {"role": "assistant", "content": "58th equal-weight"}])
    assert not r["refused"]
    assert "CONVERSATION SO FAR" in seen["user"] and "58th equal-weight" in seen["user"]


# -- the route -------------------------------------------------------------------

def test_route_ask_book_validates_cleans_history_and_wires_jsonable(monkeypatch):
    pytest.importorskip("fastapi")
    import numpy as np
    from fastapi import HTTPException

    from stockscan.web import routes
    from stockscan.web.state import STATE

    class _Facade:
        def ask_book(self, question, history=None):
            assert history == [{"role": "user", "content": "a"}]       # junk stripped
            return {"answer": "ok", "refused": False, "n_names": np.int64(3)}

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    out = routes.ask_book({"question": "rank?", "history": [
        "junk", {"role": "assistant"}, {"role": "user", "content": "a"}]})
    assert out["answer"] == "ok"
    assert out["n_names"] == 3 and isinstance(out["n_names"], int)     # numpy coerced

    with pytest.raises(HTTPException) as e:
        routes.ask_book({"question": "   "})
    assert e.value.status_code == 422
    with pytest.raises(HTTPException) as e:
        routes.ask_book({"question": "x" * 2001})
    assert e.value.status_code == 422


def test_route_ask_book_single_flights_the_llm(monkeypatch):
    pytest.importorskip("fastapi")
    from stockscan.web import routes
    from stockscan.web.state import STATE

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", object())   # must never be reached
    assert routes._LLM_GATE.acquire(blocking=False)
    try:
        assert routes.ask_book({"question": "busy?"}) == {"busy": True}
    finally:
        routes._LLM_GATE.release()


def test_route_ask_book_registered_before_the_int_typed_cik_route():
    pytest.importorskip("fastapi")
    from stockscan.web.routes import router

    paths = [r.path for r in router.routes]
    assert paths.index("/ask/book") < paths.index("/ask/{cik}")   # literal path wins
