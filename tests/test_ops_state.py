"""Ops state store: idempotent writes, the honesty trail, the paper book."""

import pytest

from stockscan.ops.state import OpsState


@pytest.fixture
def state(tmp_path):
    with OpsState(tmp_path / "state.sqlite") as st:
        yield st


def test_job_run_logging(state):
    run_id = state.job_start("prices")
    assert state.last_run("prices")["status"] == "running"
    state.job_finish(run_id, "ok", {"written": 3})
    last = state.last_run("prices")
    assert last["status"] == "ok"
    assert last["deltas"] == {"written": 3}
    assert state.last_run("prices", status="failed") is None
    assert state.last_run("universe") is None


def test_watchlist_add_remove_idempotent(state):
    state.watch_add(320193, "AAPL", "core")
    state.watch_add(320193, "AAPL", "core")  # replay is safe
    assert [w["cik"] for w in state.watchlist()] == [320193]
    state.watch_remove(320193)
    assert state.watchlist() == []
    state.watch_add(320193, "AAPL")  # re-adding reactivates
    assert len(state.watchlist()) == 1


def test_signal_state_roundtrip(state):
    assert state.get_signal(1) is None
    state.record_signal(1, 85, 9, "2026-06-30")
    state.record_signal(1, 70, 7, "2026-07-01")  # upsert, not append
    sig = state.get_signal(1)
    assert (sig["percentile"], sig["decile"], sig["as_of"]) == (70, 7, "2026-07-01")


def test_add_filings_returns_only_news(state):
    rows = [
        {"cik": 1, "form": "10-K", "filed_date": "2026-02-01",
         "period_end": "2025-12-31", "source": "fsds"},
        {"cik": 1, "form": "10-Q", "filed_date": "2026-05-01",
         "period_end": "2026-03-31", "source": "edgar"},
    ]
    assert len(state.add_filings(rows)) == 2
    assert state.add_filings(rows) == []  # replay: nothing new
    assert state.has_filings(1)
    assert state.has_filings(1, source="fsds")
    assert not state.has_filings(2)
    assert state.latest_filing_date(1) == "2026-05-01"


def test_same_day_multi_period_filings_all_recorded(state):
    """A delinquent filer catching up files several same-form docs on one day —
    period_end is part of the key, so none of them is swallowed."""
    rows = [
        {"cik": 9, "form": "10-Q", "filed_date": "2026-06-01",
         "period_end": pe, "source": "edgar"}
        for pe in ("2025-09-30", "2025-12-31", "2026-03-31")
    ]
    assert len(state.add_filings(rows)) == 3


def test_alerts_flow(state):
    state.add_alert("percentile_move", "cik 1 moved", cik=1, payload={"from": 80, "to": 60})
    state.add_alert("filing_detected", "cik 2 filed", cik=2)
    unseen = state.alerts()
    assert len(unseen) == 2
    assert state.mark_alerts_seen([unseen[0]["id"]]) == 1
    assert len(state.alerts()) == 1
    assert len(state.alerts(unseen_only=False)) == 2


def test_book_transitions_idempotent(state):
    state.book_apply({1: "A", 2: "B"}, set(), "2026-06-30")
    assert set(state.book()) == {1, 2}
    state.book_apply({3: "C"}, {1}, "2026-07-31")
    assert set(state.book()) == {2, 3}
    # replaying the same rebalance (crash-heal path) changes nothing
    state.book_apply({3: "C"}, {1}, "2026-07-31")
    assert set(state.book()) == {2, 3}
    # re-entry after exit reactivates
    state.book_apply({1: "A"}, set(), "2026-08-31")
    assert set(state.book()) == {1, 2, 3}
