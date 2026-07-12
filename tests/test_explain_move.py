"""'Explain this move' = a window-filtered grounded context + honest fallbacks.

The grounded-answer machinery is covered in test_assist; here we check what this
surface adds: the horizon window actually filters news/filings (COINCIDING items
only), every number the price chips show is citable under the grounding guard, an
empty window or missing history answers deterministically WITHOUT the LLM, a
fabricated numeral still refuses, and the facade wires price/news/filings in."""

import pandas as pd

from stockscan.assist.move import (
    MOVE_SYSTEM,
    build_move_context,
    explain_move,
)
from stockscan.narrate.ground import check_grounding
from stockscan.view.data import ArgusData

_META = {"ticker": "AAA", "name": "Alpha"}
_SUM = {"last": 123.456, "chg_1w": -3.21, "chg_1m": 12.34, "chg_3m": None,
        "chg_1y": 45.6, "hi_52w": 150.2, "lo_52w": 80.7, "adv": 5e6, "n": 400}
_AS_OF = "2026-07-03"
_NEWS = [
    {"id": "n1", "date": "2026-07-01", "source": "reuters.com",
     "event_type": "guidance", "takeaway": "cut full-year guidance on weak demand"},
    {"id": "n2", "date": "2026-05-15", "source": "wsj.com",
     "event_type": "mna", "takeaway": "reported takeover interest"},
]
_FILINGS = [
    {"form": "8-K", "filed_date": "2026-06-30", "period_end": "", "label": "Current report"},
    {"form": "10-K", "filed_date": "2025-09-01", "period_end": "2025-06-30", "label": "Annual report"},
]


# -- build_move_context ----------------------------------------------------------

def test_context_keeps_only_coinciding_items():
    ctx = build_move_context(_META, "1w", _SUM, _AS_OF, news=_NEWS, filings=_FILINGS)
    assert [n["id"] for n in ctx["news"]] == ["n1"]              # 05-15 is outside 1w
    assert [f["form"] for f in ctx["filings"]] == ["8-K"]        # the old 10-K too
    wide = build_move_context(_META, "1y", _SUM, _AS_OF, news=_NEWS, filings=_FILINGS)
    assert len(wide["news"]) == 2 and len(wide["filings"]) == 2  # both inside 372d


def test_context_drops_items_dated_after_the_move_ended():
    # news memory can hold items PUBLISHED AFTER the price series' last close —
    # they cannot have coincided with a move that already ended
    late = [{"id": "n3", "date": "2026-07-05", "source": "x.com",
             "event_type": "other", "takeaway": "something after the close"}]
    ctx = build_move_context(_META, "1w", _SUM, _AS_OF, news=_NEWS + late,
                             filings=_FILINGS)
    assert [n["id"] for n in ctx["news"]] == ["n1"]


def test_context_numbers_match_the_chip_phrasing():
    ctx = build_move_context(_META, "1m", _SUM, _AS_OF, news=_NEWS, filings=_FILINGS)
    assert ctx["move"]["chg_pct"] == 12.3 and ctx["move"]["direction"] == "up"
    assert ctx["move"]["last"] == 123.46
    assert "hi_52w" not in ctx["move"]           # range left out on purpose
    text = ("AAA is up 12.3% over the last month of trading, closing at 123.46; "
            "an 8-K landed 2026-06-30 and reuters.com reported a guidance cut")
    assert check_grounding(text, ctx) == []
    assert check_grounding("it moved because earnings rose 40%", ctx) == [40.0]


def test_context_notes_frame_coincidence_not_cause():
    ctx = build_move_context(_META, "1w", _SUM, _AS_OF, news=_NEWS, filings=_FILINGS)
    assert "coincidence is not cause" in ctx["news_note"].lower()
    assert "not explanations" in ctx["filings_note"]
    assert "neither predicts nor explains" in ctx["model_note"]
    assert "COINCIDENCE IS NOT CAUSE" in MOVE_SYSTEM


# -- explain_move: the deterministic honest paths (LLM never woken) ---------------

def _no_llm(system, prompt):
    raise AssertionError("the LLM must not be called on a deterministic path")


def test_empty_window_answers_deterministically_and_grounds():
    r = explain_move(_META, "1w", _SUM, _AS_OF,
                     news=[_NEWS[1]], filings=[_FILINGS[1]], llm=_no_llm)
    assert r["deterministic"] is True and r["refused"] is False
    assert "down 3.2%" in r["answer"] and "coincidence" in r["answer"].lower()
    ctx = build_move_context(_META, "1w", _SUM, _AS_OF,
                             news=[_NEWS[1]], filings=[_FILINGS[1]])
    assert check_grounding(r["answer"], ctx) == []   # the canned text traces too


def test_missing_history_is_an_honest_answer_not_a_crash():
    r = explain_move(_META, "3m", _SUM, _AS_OF, news=_NEWS, filings=_FILINGS,
                     llm=_no_llm)                     # chg_3m is None
    assert r["deterministic"] is True and "price history" in r["answer"]
    r = explain_move(_META, "1m", {}, None, llm=_no_llm)   # no summary at all
    assert r["deterministic"] is True and "no move here to explain" in r["answer"]


def test_unknown_horizon_raises():
    import pytest
    with pytest.raises(ValueError):
        explain_move(_META, "6m", _SUM, _AS_OF)


# -- explain_move: the grounded LLM path ------------------------------------------

def test_clean_answer_passes_through():
    def llm(system, prompt):
        # the ISO-date rule now lives in core.grounded_answer (deduped from
        # MOVE_SYSTEM) — it must still reach this surface's system prompt
        assert "COINCIDENCE" in system and "YYYY-MM-DD" in system and "12.3" in prompt
        return ("AAA is up 12.3% over the last month of trading; reuters.com "
                "reported a guidance cut dated 2026-07-01, which coincided with "
                "the window.")
    r = explain_move(_META, "1m", _SUM, _AS_OF, news=_NEWS, filings=_FILINGS, llm=llm)
    assert r["refused"] is False and r["deterministic"] is False
    assert "coincided" in r["answer"]


def test_fabricated_numeral_retries_then_refuses():
    calls = []

    def llm(system, prompt):
        calls.append(prompt)
        return "it fell because earnings dropped 40%"
    r = explain_move(_META, "1m", _SUM, _AS_OF, news=_NEWS, filings=_FILINGS, llm=llm)
    assert r["refused"] is True and len(calls) == 2   # one retry, then refusal
    assert 40.0 in r["violations"]


# -- the facade method -------------------------------------------------------------

def _stub_facade():
    ad = ArgusData(data=None, artifact=None, as_of=pd.Timestamp(_AS_OF))
    ad._cross = pd.DataFrame({"cik": [7], "ticker": ["AAA"], "name": ["Alpha"]})
    idx = pd.date_range("2026-06-01", _AS_OF, freq="B")
    ad.price = lambda cik: {"column": "AAA",
                            "series": pd.Series(range(len(idx)), index=idx),
                            "summary": dict(_SUM)}
    # the move surface fetches WINDOWED news (recall since=cutoff), not the curated
    # 6-row packet context — stub that path; client-side window filtering still applies
    ad._news_window = lambda cik, since, limit=12: [
        dict(r, publication_date=r["date"]) for r in _NEWS]
    ad.events = lambda cik: list(_FILINGS)
    return ad


def test_facade_wires_price_news_and_filings():
    ad = _stub_facade()
    seen = {}

    def llm(system, prompt):
        seen["prompt"] = prompt
        return "AAA is up 12.3% over the last month of trading."
    r = ad.explain_move(7, "1m", llm=llm)
    assert r["cik"] == 7 and r["ticker"] == "AAA" and r["horizon"] == "1m"
    assert r["refused"] is False
    assert "guidance" in seen["prompt"]          # windowed news made the context
    assert "8-K" in seen["prompt"]               # so did the coinciding filing


def test_facade_windows_news_to_the_horizon():
    # move_context must pass the horizon's cutoff to the news fetch so a short
    # window queries the store for its own window, not filter a truncated recall
    ad = _stub_facade()
    seen = {}
    ad._news_window = lambda cik, since, limit=12: (seen.__setitem__("since", since), [])[1]
    ad.move_context(7, "1w")
    assert seen["since"] == "2026-06-23"         # 2026-07-03 − 10 calendar days


def test_facade_splits_context_from_answer_for_the_gate():
    # move_context does the network/context work + the code-only decision with NO
    # llm; move_answer is the only part that needs the model (route gates just it)
    ad = _stub_facade()
    bundle, det = ad.move_context(7, "1m")
    assert det is None and "ctx" in bundle       # coinciding items → needs the model
    r = ad.move_answer(7, "1m", bundle,
                       llm=lambda s, u: "AAA is up 12.3% over the last month of trading.")
    assert r["refused"] is False and r["ticker"] == "AAA"


def test_facade_deterministic_path_needs_no_llm():
    ad = _stub_facade()
    ad._news_window = lambda cik, since, limit=12: []
    ad.events = lambda cik: []                    # empty window → code-only answer
    bundle, det = ad.move_context(7, "1w")
    assert det is not None and det["deterministic"] is True
    assert det["cik"] == 7 and det["ticker"] == "AAA"


def test_facade_falls_back_to_universe_name_when_unscored():
    # an unscored name (absent from _cross) still gets ticker AND name from the
    # universe pick — the fallback must not blank the company name
    ad = _stub_facade()
    ad._cross = pd.DataFrame({"cik": [], "ticker": [], "name": []})
    ad._pick = lambda cik, field: {"ticker": "ZZZ", "name": "Zeta Corp"}[field]
    bundle, _ = ad.move_context(7, "1m")
    assert bundle["ctx"]["company"] == {"ticker": "ZZZ", "name": "Zeta Corp"}


def test_facade_rejects_unknown_horizon():
    import pytest
    with pytest.raises(ValueError):
        _stub_facade().explain_move(7, "2w")
    with pytest.raises(ValueError):
        _stub_facade().move_context(7, "2w")
