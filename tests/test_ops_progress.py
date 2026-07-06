"""Paper-progress alerts + position-aware alert annotation.

Both are display/alert plumbing over already-computed numbers: paper_progress_alerts
diffs a compare() report against the last recorded check, book_weights turns the
positions store into alert-text annotations. Neither ever feeds a signal."""

import pandas as pd

from stockscan.ops.monitor import _held_note, book_weights
from stockscan.ops.paper import paper_progress_alerts
from stockscan.ops.state import OpsState


# -- paper progress ---------------------------------------------------------------

class _State:
    """Duck-typed OpsState: last_run + add_alert capture (test_assist convention)."""

    def __init__(self, prev_deltas=None):
        self._prev = None if prev_deltas is None else {"deltas": prev_deltas}
        self.alerts = []

    def last_run(self, job, status=None):
        return self._prev

    def add_alert(self, kind, message, cik=None, payload=None):
        self.alerts.append({"kind": kind, "message": message, "payload": payload})


def _rep(oos=1, ic=0.031, degraded=None, floor=None):
    rep = {"months_scored_oos": oos, "live_mean_ic": ic, "degraded": degraded,
           "baseline": {"expected_ic": 0.0375}}
    if floor is not None:
        rep["degradation_floor"] = floor
    return rep


def test_first_run_seeds_silently():
    st = _State(prev_deltas=None)
    d = paper_progress_alerts(st, _rep(oos=1))
    assert st.alerts == [] and d["note"] == "seeded" and d["months_scored_oos"] == 1


def test_new_oos_month_alerts_once_and_is_idempotent():
    st = _State(prev_deltas={"months_scored_oos": 1, "degraded": None})
    d = paper_progress_alerts(st, _rep(oos=2))
    assert d["alerts"] == 1
    assert st.alerts[0]["kind"] == "paper_graded"
    assert "2 total" in st.alerts[0]["message"] and "0.0375" in st.alerts[0]["message"]
    # same report against the now-recorded state → no re-alert
    st2 = _State(prev_deltas=d)
    assert paper_progress_alerts(st2, _rep(oos=2))["alerts"] == 0


def test_degradation_flip_alerts_both_ways():
    st = _State(prev_deltas={"months_scored_oos": 6, "degraded": False})
    d = paper_progress_alerts(st, _rep(oos=6, ic=0.001, degraded=True, floor=0.0187))
    assert d["alerts"] == 1 and st.alerts[0]["kind"] == "paper_degraded"
    assert "DEGRADED" in st.alerts[0]["message"]

    st = _State(prev_deltas={"months_scored_oos": 6, "degraded": True})
    d = paper_progress_alerts(st, _rep(oos=6, ic=0.04, degraded=False, floor=0.0187))
    assert st.alerts[0]["kind"] == "paper_recovered"


def test_verdict_not_yet_applicable_never_flips():
    st = _State(prev_deltas={"months_scored_oos": 1, "degraded": None})
    d = paper_progress_alerts(st, _rep(oos=1, degraded=None))
    assert d["alerts"] == 0 and st.alerts == []


# -- position-aware annotation ------------------------------------------------------

class _Data:
    def __init__(self):
        idx = pd.bdate_range("2026-06-01", periods=10)
        self.close = pd.DataFrame({"AAA": [10.0] * 10, "BBB": [30.0] * 10}, index=idx)
        self.ticker_map = {1: "AAA", 2: "BBB", 3: "CCC"}


def test_book_weights_values_from_last_close(tmp_path):
    with OpsState(tmp_path / "s.sqlite") as st:
        st.position_set(1, shares=100, cost_basis=5.0)   # 100×10 = 1000
        st.position_set(2, shares=100, cost_basis=20.0)  # 100×30 = 3000
        st.position_set(3, shares=50, cost_basis=1.0)    # no price column → skipped
        w = book_weights(st, _Data())
    assert w[1] == 0.25 and w[2] == 0.75 and 3 not in w
    assert _held_note(w, 2) == " — you hold ≈75% of your book"
    assert _held_note(w, 99) == ""


def test_book_weights_empty_without_positions(tmp_path):
    with OpsState(tmp_path / "s.sqlite") as st:
        assert book_weights(st, _Data()) == {}
