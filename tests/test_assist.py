"""Tests for the read-only AI assist layer (LLM mocked; firewall scan on real tree)."""

import json

from stockscan.assist.audit import firewall_review_diff, firewall_scan
from stockscan.assist.brief import build_brief_context, nightly_brief
from stockscan.assist.core import REFUSAL, grounded_answer
from stockscan.assist.judge import judge_narration
from stockscan.assist.qa import answer_from_packet

_PACKET = {
    "meta": {"cik": 1, "name": "ALPHA", "ticker": "ALPHA"},
    "signals": [{"id": "roa", "label": "Return on assets", "value": 19.8,
                 "pct_rank": 95, "read": "supports"}],
    "model": {"percentile": 73, "decile": 8},
}


# --- core: grounded_answer -------------------------------------------------------

def test_grounded_answer_accepts_context_numbers():
    def llm(system, user):
        return "Return on assets is 19.8% (95th percentile), a strength."
    r = grounded_answer(_PACKET, "how is profitability?", llm, "SYS")
    assert r["grounded"] and not r["refused"] and r["attempts"] == 1


def test_grounded_answer_refuses_a_fabricated_number():
    def llm(system, user):
        return "Its secret margin is 42.0%."      # 42 is nowhere in the packet
    r = grounded_answer(_PACKET, "margin?", llm, "SYS", max_retries=1)
    assert r["refused"] and r["answer"] == REFUSAL
    assert 42.0 in r["violations"]


def test_grounded_answer_degrades_on_llm_error():
    def llm(system, user):
        raise TimeoutError("down")
    r = grounded_answer(_PACKET, "q", llm, "SYS")
    assert r["refused"] and any(str(v).startswith("llm-error") for v in r["violations"])


# --- A: qa -----------------------------------------------------------------------

def test_qa_answers_from_packet_grounded():
    def llm(system, user):
        assert "CONTEXT" in user and "QUESTION" in user
        return "The model ranks it in the 73rd percentile (decile 8)."
    r = answer_from_packet(_PACKET, "why ranked here?", llm)
    assert r["grounded"] and not r["refused"]


def test_qa_refuses_when_it_would_fabricate():
    r = answer_from_packet(_PACKET, "q", lambda s, u: "It returned 88.8% last year.")
    assert r["refused"]


def test_qa_history_is_woven_in():
    seen = {}

    def llm(system, user):
        seen["u"] = user
        return "Return on assets 19.8%."
    answer_from_packet(_PACKET, "and now?", llm,
                       history=[{"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"}])
    assert "CONVERSATION SO FAR" in seen["u"] and "hello" in seen["u"]


# --- B: judge --------------------------------------------------------------------

def test_judge_reports_issues():
    def llm(system, user):
        return json.dumps({"issues": [{"type": "direction", "quote": "strong leverage",
                                       "why": "high leverage is a weakness"}]})
    r = judge_narration("… strong leverage …", _PACKET, llm)
    assert not r["faithful"] and r["issues"][0]["type"] == "direction"


def test_judge_passes_clean_and_fails_open():
    assert judge_narration("fine", _PACKET, lambda s, u: '{"issues": []}')["faithful"]
    # unparseable / error -> advisory, so fail-open (deterministic guard is the guarantee)
    assert judge_narration("fine", _PACKET, lambda s, u: "not json")["faithful"]
    assert judge_narration("fine", _PACKET, lambda s, u: (_ for _ in ()).throw(RuntimeError()))["faithful"]


# --- C: audit (firewall) ---------------------------------------------------------

def test_firewall_scan_flags_forbidden_import(tmp_path):
    root = tmp_path / "stockscan"
    root.mkdir()
    (root / "model.py").write_text("from .newsmem import NewsStore\nx = 1\n")
    (root / "backtest.py").write_text("import stockscan.narrate as n\n")
    (root / "features.py").write_text("import pandas as pd\nfrom .concepts import w\n")  # clean
    (root / "tui").mkdir()
    (root / "tui" / "app.py").write_text("from ..newsmem import x\n")  # live-side, not protected
    v = firewall_scan(root)
    by_mod = {x["module"]: x["imports"] for x in v}
    assert by_mod.get("model") == ["newsmem"]
    assert by_mod.get("backtest") == ["narrate"]
    assert "features" not in by_mod          # clean core module
    assert "tui" not in by_mod               # tui is live-side, not protected


def test_firewall_denylist_auto_covers_new_modules_and_allows_bridges(tmp_path):
    """The denylist design: a brand-new core module (a future model head) is protected
    with no list edit; the sanctioned bridges (serve/ops/config) may import the live
    side. This is the property that keeps the firewall correct as the project grows."""
    root = tmp_path / "stockscan"
    root.mkdir()
    (root / "distress2.py").write_text("from .newsmem import x\n")     # unlisted new head
    (root / "serve.py").write_text("from .narrate.narrator import narrate_packet\n")
    (root / "ops").mkdir()
    (root / "ops" / "monitor.py").write_text("from ..narrate import x\n")
    (root / "config.py").write_text("from .assist import y\n")         # infra bridge
    v = {x["module"]: x["imports"] for x in firewall_scan(root)}
    assert v.get("distress2") == ["newsmem"]     # auto-protected without listing it
    assert "serve" not in v and "ops" not in v and "config" not in v   # bridges allowed


def test_real_codebase_firewall_is_intact():
    """The guarantee, enforced in CI: the signal/data core imports nothing from the
    live-view/AI side (news, newsmem, narrate, assist, quote, tui)."""
    from stockscan.config import REPO_ROOT
    v = firewall_scan(REPO_ROOT / "src" / "stockscan")
    assert v == [], f"FIREWALL BREACH: {v}"


def test_firewall_review_diff_parses_issues():
    def llm(system, user):
        return "Here: " + json.dumps({"issues": [{"severity": "critical", "file": "x.py",
                                                   "line_hint": "10", "why": "leak"}]})
    r = firewall_review_diff("diff --git a/x.py", llm)
    assert r["issues"][0]["severity"] == "critical"
    # transport failure is non-fatal (deterministic scan is the guarantee)
    assert firewall_review_diff("d", lambda s, u: (_ for _ in ()).throw(OSError()))["issues"] == []


# --- D: brief --------------------------------------------------------------------

class _FakeState:
    def last_run(self, job):
        return {"nightly": {"status": "ok", "finished": "2026-07-03T05:00:00",
                            "deltas": {"prices_failed_frac": 0.0}},
                "news": {"status": "ok", "finished": "2026-07-03T05:02:00",
                         "deltas": {"new": 12, "extracted": 12}}}.get(job)

    def alerts(self, unseen_only=True, limit=50):
        return [{"kind": "pctile_move", "message": "NFLX +11", "cik": 1065280}]


def test_build_brief_context_gathers_runs_and_alerts():
    ctx = build_brief_context(_FakeState())
    assert ctx["jobs"]["news"]["deltas"]["new"] == 12
    assert ctx["n_unseen_alerts"] == 1 and ctx["unseen_alerts"][0]["kind"] == "pctile_move"


def test_nightly_brief_is_grounded():
    ctx = build_brief_context(_FakeState())

    def llm(system, user):
        return "Overnight: prices refreshed, 12 new articles extracted, 1 alert (NFLX +11)."
    r = nightly_brief(ctx, llm)
    assert r["grounded"] and not r["refused"]
