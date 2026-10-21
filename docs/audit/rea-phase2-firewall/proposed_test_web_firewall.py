"""PROPOSED as tests/test_web_firewall.py (REA Phase 2) — not yet added to the suite.

HTTP-level firewall for the ticker call: the BUY/HOLD/AVOID verdict (and every signal
field /api/ticker serves) is a function of the frozen model's ranks ONLY.

test_serve.py proves the risk heads and confidence never move score/percentile/decile/
packet inside analyze(). These close the two gaps above that layer, through the
ASSEMBLED app on a REAL ArgusData over the synthetic world (no fake facade):

1. the personal book — positions, watchlist, alerts, all in the ops store — cannot reach
   /api/ticker: the route never even opens the store, and rewriting the book through the
   app's own write endpoints leaves the response byte-identical;
2. the verdict the route attaches (routes.py) is exactly view.chart.verdict(percentile),
   with and without the display-only drawdown head + confidence table attached — the
   head may only LOWER confidence (the one sanctioned interlock), never touch the call.
"""

import copy

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from test_serve import world  # noqa: E402,F401  (module-scoped synthetic world fixture)

from stockscan.ops import state as ops_state  # noqa: E402
from stockscan.view.chart import verdict  # noqa: E402
from stockscan.view.data import ArgusData  # noqa: E402
from stockscan.web.state import STATE  # noqa: E402

AS_OF = pd.Timestamp("2024-06-28")
SIGNAL = ("score", "percentile", "decile", "verdict", "ranks", "packet")


@pytest.fixture()
def app(world, tmp_path, monkeypatch):
    """The assembled app over a real facade + a throwaway ops store."""
    monkeypatch.setattr(ops_state, "OPS_STATE_PATH", tmp_path / "ops.sqlite")
    monkeypatch.setattr(STATE, "start_load", lambda: None)
    facade = ArgusData(data=copy.copy(world["data"]), artifact=world["artifact"])
    facade.refresh(AS_OF)
    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", facade)
    from stockscan.web.server import app as asgi

    with TestClient(asgi) as client:
        ciks = [int(c) for c in facade._cross["cik"]]
        served = [c for c in ciks if client.get(f"/api/ticker/{c}").status_code == 200]
        assert len(served) >= 5, "synthetic world should serve several names"
        facade.refresh(AS_OF)                     # forget the probe's analyze cache
        yield client, facade, served


def test_ticker_route_never_opens_the_personal_store(app, monkeypatch):
    """Positions, watchlist and alerts all live in the ops store; the signal route must
    not read it at all — not merely produce the same bytes by coincidence."""
    client, _, served = app

    def _boom(*a, **k):
        raise AssertionError("/api/ticker opened the personal ops store")

    monkeypatch.setattr(ops_state.OpsState, "__init__", _boom)
    for cik in served:
        assert client.get(f"/api/ticker/{cik}").status_code == 200


def test_ticker_bytes_are_independent_of_the_book(app):
    client, facade, served = app
    before = {c: client.get(f"/api/ticker/{c}").content for c in served}

    # rewrite the book through the app's OWN write endpoints, plus risk-flavoured alerts
    a, b = served[0], served[1]
    assert client.post(f"/api/positions/{a}", json={"shares": 5000, "cost": 1.0}).status_code == 200
    assert client.post(f"/api/positions/{b}", json={"shares": 3, "cost": 999.0}).status_code == 200
    assert client.post(f"/api/watch/{a}/toggle").json()["watched"] is True
    with ops_state.OpsState() as st:
        st.add_alert("distress_risk", "test: distress HIGH", cik=a)
        st.add_alert("drawdown_risk", "test: drawdown HIGH", cik=b)
    assert {p["cik"] for p in client.get("/api/positions").json()} >= {a, b}   # it landed

    facade.refresh(AS_OF)                         # make the server truly recompute
    after = {c: client.get(f"/api/ticker/{c}").content for c in served}
    assert after == before                        # byte-identical, every served name


def test_served_verdict_is_a_pure_function_of_percentile(app):
    client, _, served = app
    for cik in served:
        j = client.get(f"/api/ticker/{cik}").json()
        assert j["verdict"] == verdict(j["percentile"] / 100.0), cik


def test_risk_head_and_confidence_never_move_the_call_over_http(app, world, tmp_path):
    """Attach the display-only drawdown head + a confidence table: the call is unchanged
    and confidence may only go DOWN (the sanctioned drawdown->confidence interlock)."""
    from stockscan.drawdown import fit_drawdown, load_drawdown_artifact, save_drawdown_artifact

    client, facade, served = app
    cal = {"deciles": {str(d): {"hit_rate": 0.40 + 0.02 * d, "mean_excess": 0.0, "n": 1000,
                                "ci_low": 0.38, "ci_high": 0.62} for d in range(1, 11)}}
    facade.data.confidence_calibration = cal
    facade.refresh(AS_OF)
    plain = {c: client.get(f"/api/ticker/{c}").json() for c in served}

    wp = world["panel"].copy()
    wp["y"] = (wp["cik"] % 3 == 0).astype(float)
    wp.attrs["censor_date"] = wp["date"].max()
    wp.attrs["horizon_months"] = 6
    wmodel = fit_drawdown(wp, params=dict(n_estimators=15, min_child_samples=5))
    facade.data.drawdown_artifact = load_drawdown_artifact(
        save_drawdown_artifact(wmodel, wp, out_dir=tmp_path / "w"))
    facade.refresh(AS_OF)
    headed = {c: client.get(f"/api/ticker/{c}").json() for c in served}

    for cik in served:
        p, h = plain[cik], headed[cik]
        assert h["drawdown"] is not None and p["drawdown"] is None
        for k in SIGNAL:
            assert h[k] == p[k], (cik, k)
        if p["confidence"] and h["confidence"]:
            assert h["confidence"]["score"] <= p["confidence"]["score"], cik
