"""HTTP-level smoke test of the ASSEMBLED app — the one layer direct route calls
can't cover: router registration order, the /api prefix, static mounting, and the
loading handshake, exercised through a real TestClient."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from stockscan.web import routes  # noqa: E402
from stockscan.web.state import STATE  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    # the real lifespan kicks off the heavy background data load — not in tests
    monkeypatch.setattr(STATE, "start_load", lambda: None)
    from stockscan.web.server import app

    with TestClient(app) as c:
        yield c


def test_status_handshake_503_until_ready(client, monkeypatch):
    monkeypatch.setattr(STATE, "status", "loading")
    monkeypatch.setattr(STATE, "adata", None)
    r = client.get("/api/status")
    assert r.status_code == 503 and r.json()["loading"] is True

    class _Facade:
        def status(self):
            return {"as_of": "2026-07-03"}

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    r = client.get("/api/status")
    assert r.status_code == 200 and r.json()["as_of"] == "2026-07-03"


def test_ask_book_route_order_beats_the_int_path(client, monkeypatch):
    """POST /api/ask/book must reach ask_book — if the int-typed /ask/{cik} were
    registered first, "book" would 422 at path-param parsing."""

    class _Facade:
        def ask_book(self, question, history=None):
            return {"answer": "both weightings shown", "refused": False}

        def ask(self, cik, question, history=None):   # must NOT be hit
            raise AssertionError("fell through to /ask/{cik}")

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    r = client.post("/api/ask/book", json={"question": "where does my book rank?"})
    assert r.status_code == 200 and r.json()["answer"] == "both weightings shown"


def test_new_endpoints_are_wired(client, monkeypatch):
    class _Facade:
        def watched_ciks(self):
            return [1, 2]

        def digest(self):
            return {"jobs": {}, "n_unseen_alerts": 0, "unseen_alerts": []}

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    assert client.get("/api/watch-ids").json() == {"ciks": [1, 2]}
    assert client.get("/api/digest").json()["n_unseen_alerts"] == 0


def test_explain_move_route_validates_then_delegates(client, monkeypatch):
    class _Facade:
        def move_context(self, cik, horizon):
            # deterministic (code-only) answer: no gate, no model
            return {"ctx": {}, "ticker": "AAA"}, {
                "answer": "nothing coincided", "refused": False,
                "deterministic": True, "cik": cik, "horizon": horizon}

        def move_answer(self, cik, horizon, bundle, llm=None):
            raise AssertionError("deterministic path must not call the model")

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    r = client.post("/api/explain-move/7", json={"horizon": "6m"})
    assert r.status_code == 422                       # unknown chip never hits the facade
    r = client.post("/api/explain-move/7", json={"horizon": "1m"})
    assert r.status_code == 200 and r.json()["horizon"] == "1m"


def test_explain_move_route_gates_only_the_model_call(client, monkeypatch):
    """When move_context has no deterministic answer, the route runs move_answer
    under the single-flight gate — context assembly already happened gate-free."""
    class _Facade:
        def move_context(self, cik, horizon):
            return {"ctx": {"x": 1}, "ticker": "AAA"}, None      # needs the model

        def move_answer(self, cik, horizon, bundle, llm=None):
            return {"answer": "reuters.com reported a guidance cut that coincided",
                    "refused": False, "deterministic": False, "cik": cik,
                    "horizon": horizon, "ticker": bundle["ticker"]}

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    r = client.post("/api/explain-move/7", json={"horizon": "1m"})
    assert r.status_code == 200 and r.json()["deterministic"] is False
    assert "coincided" in r.json()["answer"]


def test_static_index_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "app.js" in r.text and "scan" in r.text   # the real index.html
