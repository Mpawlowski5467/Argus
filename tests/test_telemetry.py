"""LLM telemetry: fail-open logging of shown turns + the nightly judge sample."""

import json

import pytest

import stockscan.ops.state as state_mod
from stockscan.assist.telemetry import context_hash, judge_sample, record_turn
from stockscan.ops.state import OpsState


@pytest.fixture
def state(tmp_path, monkeypatch):
    # record_turn opens OpsState() with the module default — point it at a tmp store
    monkeypatch.setattr(state_mod, "OPS_STATE_PATH", tmp_path / "state.sqlite")
    with OpsState() as st:
        yield st


def _res(answer="the percentile is 96", refused=False, attempts=1):
    return {"answer": answer, "refused": refused, "attempts": attempts,
            "grounded": True, "violations": []}


def test_context_hash_is_stable_and_order_free():
    a = context_hash({"x": 1, "y": [2, 3]})
    b = context_hash({"y": [2, 3], "x": 1})
    assert a == b and len(a) == 12
    assert context_hash({"x": 2}) != a


def test_record_turn_roundtrips_through_the_store(state):
    ctx = {"meta": {"ticker": "AAPL"}, "pct": 96}
    record_turn("ask", ctx, "why is it ranked here?", _res(), 1.25,
                usage={"prompt_tokens": 900, "completion_tokens": 120})
    rows = state.unjudged_turns(since="2000-01-01", limit=10)
    assert len(rows) == 1
    t = rows[0]
    assert t["surface"] == "ask" and "AAPL" in t["context"]
    stats = state.llm_turn_stats(since="2000-01-01")
    assert stats == {"turns": 1, "refused": 0, "avg_latency_ms": 1250,
                     "max_latency_ms": 1250, "judged_with_issues": 0}


def test_record_turn_never_raises(monkeypatch):
    # a broken store must not break the answer the user already has
    monkeypatch.setattr(state_mod, "OPS_STATE_PATH", "/dev/null/nope/state.sqlite")
    record_turn("ask", {"x": 1}, "q", _res(), 0.1)   # swallowed


def test_refused_turns_are_logged_but_never_judged(state):
    record_turn("ask", {"x": 1}, "q", _res(refused=True), 0.1)
    assert state.llm_turn_stats(since="2000-01-01")["refused"] == 1
    assert state.unjudged_turns(since="2000-01-01") == []


def test_judge_sample_flags_and_alerts(state):
    record_turn("ask", {"pct": 96}, "q1", _res("rank is 96"), 0.2)
    record_turn("move", {"pct": 96}, "q2", _res("it moved on earnings"), 0.2)

    def bad_judge_llm(system, user):
        return json.dumps({"issues": [{"type": "direction", "quote": "x", "why": "y"}]})

    out = judge_sample(state, bad_judge_llm, since="2000-01-01", limit=5)
    assert out == {"sampled": 2, "judged": 2, "flagged": 2}
    alerts = state.alerts(unseen_only=True)
    assert len(alerts) == 1 and alerts[0]["kind"] == "judge_flag"
    assert "2 of 2" in alerts[0]["message"]
    # judged turns leave the sampling pool; stats see the issues
    assert state.unjudged_turns(since="2000-01-01") == []
    assert state.llm_turn_stats(since="2000-01-01")["judged_with_issues"] == 2


def test_judge_sample_faithful_run_stays_silent(state):
    record_turn("ask", {"pct": 96}, "q", _res(), 0.2)

    def good_judge_llm(system, user):
        return json.dumps({"issues": []})

    out = judge_sample(state, good_judge_llm, since="2000-01-01")
    assert out["flagged"] == 0
    assert state.alerts(unseen_only=True) == []


def test_judge_sample_fails_open_on_judge_errors(state):
    record_turn("ask", {"pct": 96}, "q", _res(), 0.2)

    def dead_llm(system, user):
        raise TimeoutError("model down")

    out = judge_sample(state, dead_llm, since="2000-01-01")
    assert out["flagged"] == 0                     # error counts as faithful (advisory)
    assert state.unjudged_turns(since="2000-01-01") == []   # still marked, no re-judge loop
