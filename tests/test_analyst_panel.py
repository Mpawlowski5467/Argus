"""Analyst panel: four grounded memos, judge-suppressed on drift, cached per context.

The panel is commentary machinery — these tests pin the honesty properties: every
memo passes the numeral guard or refuses, a judge-flagged memo is withheld (never
silently shown or dropped), refused/suppressed memos never feed forward as prior
prose, and the cache retires a name's old panel when its context hash moves.
"""

import json

import pytest

from stockscan.assist.analyst import (
    PANEL_SYSTEMS,
    ROLES,
    WITHHELD,
    PanelCache,
    build_panel,
    panel_role,
    role_question,
)

CTX = {"meta": {"ticker": "AAPL", "name": "Apple Inc"},
       "model": {"percentile": 96, "decile": 10},
       "display": {"confidence": {"score": 16, "hit_rate_pct": 49.3}}}


def make_llm(text="the rank sits at the 96th percentile among scored peers"):
    def llm(system, user):
        return text
    return llm


def judge_ok(system, user):
    return json.dumps({"issues": []})


def judge_bad(system, user):
    return json.dumps({"issues": [{"type": "direction", "quote": "x", "why": "y"}]})


# --- prompts carry the non-negotiables --------------------------------------------

def test_every_role_prompt_carries_the_honesty_rules():
    for role, system in PANEL_SYSTEMS.items():
        assert "MONTHLY CROSS-SECTIONAL PEER RANK" in system, role
        assert "COINCIDED" in system, role
        assert "No buy/sell/hold language" in system, role


def test_role_question_visibility():
    prior = {"bull": "BULLTEXT", "bear": "BEARTEXT", "risk": "RISKTEXT"}
    assert "BULLTEXT" not in role_question("bull", prior)      # bull sees nothing
    assert "BULLTEXT" in role_question("bear", prior)          # bear rebuts bull
    assert "RISKTEXT" not in role_question("bear", prior)
    assert "BULLTEXT" not in role_question("risk", prior)      # risk argues nobody
    syn = role_question("synthesis", prior)
    assert all(m in syn for m in ("BULLTEXT", "BEARTEXT", "RISKTEXT"))


# --- generation: guard, judge, refusal ---------------------------------------------

def test_panel_role_grounds_and_passes_judge():
    r = panel_role("bull", CTX, make_llm(), judge_llm=judge_ok)
    assert r["refused"] is False and r["suppressed"] is False
    assert r["shown"] == r["answer"] and "96th" in r["shown"]


def test_fabricated_numeral_refuses_not_guesses():
    r = panel_role("bull", CTX, make_llm("revenue grew 34% this quarter"))
    assert r["refused"] is True
    assert "34" not in r["answer"]                     # the refusal text, not the leak


def test_judge_flagged_memo_is_withheld_never_shown():
    r = panel_role("bear", CTX, make_llm(), judge_llm=judge_bad)
    assert r["suppressed"] is True and r["shown"] == WITHHELD
    assert r["answer"] != WITHHELD                     # raw kept for review, not display
    assert r["judge_issues"]


def test_unknown_role_is_loud():
    with pytest.raises(ValueError):
        panel_role("trader", CTX, make_llm())          # TradingAgents' trader stays out


def test_build_panel_never_feeds_forward_bad_memos():
    calls = []

    def llm(system, user):
        calls.append(user)
        if "BULL analyst" in system:
            return "revenue grew 34%"                  # fabrication -> refused
        return "the rank sits at the 96th percentile among scored peers"

    panel = build_panel(CTX, llm)
    assert panel["roles"]["bull"]["refused"] is True
    assert panel["roles"]["synthesis"]["refused"] is False
    joined = "\n".join(calls)
    assert "BULL MEMO" not in joined                   # refused memo never rode along


# --- cache -------------------------------------------------------------------------

def test_panel_cache_roundtrip_and_eviction(tmp_path):
    p = tmp_path / "panel.sqlite"
    with PanelCache(p) as pc:
        pc.put(1, "hashA", "bull", {"answer": "a", "refused": False})
        pc.put(1, "hashA", "bear", {"answer": "b", "refused": False})
        assert set(pc.get(1, "hashA")) == {"bull", "bear"}
        assert pc.get(1, "hashB") == {}
        # the context moved (new filing / rank change): old panel retires wholesale
        pc.put(1, "hashB", "bull", {"answer": "a2", "refused": False})
        assert pc.get(1, "hashA") == {}
        assert set(pc.get(1, "hashB")) == {"bull"}
        # other names are untouched by an eviction
        pc.put(2, "hashZ", "risk", {"answer": "r", "refused": False})
        pc.put(1, "hashB", "bear", {"answer": "b2", "refused": False})
        assert set(pc.get(2, "hashZ")) == {"risk"}


def test_roles_are_the_four_report_roles():
    assert ROLES == ("bull", "bear", "risk", "synthesis")
