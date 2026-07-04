"""The scorecard web surface = a thin facade method + a thin route. The book-level
math is covered in test_portfolio; here we check the two adapters: the facade
builds the price map from the last close and delegates, and the route wires the
facade through convert.jsonable (numpy-safe)."""

import numpy as np
import pandas as pd

from stockscan.view.data import ArgusData


class _Stub:
    """Minimal ServeData stand-in: a close matrix + ticker_map (last row = latest)."""

    def __init__(self):
        self.ticker_map = {1: "AAA", 2: "BBB"}
        self.close = pd.DataFrame({"AAA": [10.0, 20.0], "BBB": [40.0, 30.0]})


class _FakeOps:
    rows = [
        {"cik": 1, "shares": 100, "cost_basis": 10.0, "added_at": "2026-01-01"},
        {"cik": 2, "shares": 50, "cost_basis": 40.0, "added_at": "2026-02-01"},
    ]

    wl: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def positions(self):
        return self.rows

    def watchlist(self):
        return self.wl


def _cross():
    return pd.DataFrame({
        "cik": [1, 2], "ticker": ["AAA", "BBB"], "name": ["Alpha", "Beta"],
        "sector": ["Tech", "Fin"], "sic": [3674, 6021],
        "score": [0.9, 0.1], "pct": [0.90, 0.30],
        "dprob": [np.nan, 0.05], "dflag": ["normal", "elevated"],
    })


def test_facade_scorecard_prices_from_last_close_and_delegates(monkeypatch):
    import stockscan.ops.state as state_mod
    monkeypatch.setattr(state_mod, "OpsState", _FakeOps)

    ad = ArgusData(data=_Stub(), artifact=None,
                   as_of=pd.Timestamp("2026-07-04"), _cross=_cross())
    sc = ad.scorecard()

    assert sc["as_of"] == "2026-07-04"
    assert sc["n_total"] == 2 and sc["n_owned"] == 2 and sc["n_listed"] == 2
    # value uses the LAST close: AAA 100×20=2000, BBB 50×30=1500
    by = {h["cik"]: h for h in sc["holdings"]}
    assert by[1]["value"] == 2000.0 and by[2]["value"] == 1500.0
    assert sc["total_value"] == 3500.0
    assert sc["percentile_equal"] == 60.0                      # mean(90,30)


def test_facade_scorecard_merges_watchlist_only_names(monkeypatch):
    import stockscan.ops.state as state_mod

    class _Ops(_FakeOps):
        rows = [{"cik": 1, "shares": 100, "cost_basis": 10.0, "added_at": "2026-01-01"}]
        wl = [{"cik": 1, "column": "AAA", "added": "x"},   # already held → not doubled
              {"cik": 2, "column": "BBB", "added": "x"}]   # followed-only → no shares

    monkeypatch.setattr(state_mod, "OpsState", _Ops)
    ad = ArgusData(data=_Stub(), artifact=None,
                   as_of=pd.Timestamp("2026-07-04"), _cross=_cross())
    sc = ad.scorecard()
    assert sc["n_total"] == 2 and sc["n_owned"] == 1 and sc["n_watch"] == 1
    by = {h["cik"]: h for h in sc["holdings"]}
    assert by[1]["owned"] is True and by[1]["value"] == 2000.0
    assert by[2]["owned"] is False and by[2]["value"] is None  # watchlist-only
    assert by[2]["pct"] == 30                                  # standing still resolves
    assert sc["percentile_equal"] == 60.0                     # both count: mean(90,30)


def test_facade_scorecard_skips_price_for_unknown_column(monkeypatch):
    import stockscan.ops.state as state_mod

    class _Ops(_FakeOps):
        rows = [{"cik": 9, "shares": 5, "cost_basis": 1.0, "added_at": "2026-01-01"}]

    monkeypatch.setattr(state_mod, "OpsState", _Ops)
    ad = ArgusData(data=_Stub(), artifact=None,
                   as_of=pd.Timestamp("2026-07-04"), _cross=_cross())
    sc = ad.scorecard()
    assert sc["n_total"] == 1 and sc["n_unlisted"] == 1        # cik 9 not in cross
    assert sc["holdings"][0]["value"] is None                  # and no price column


def test_route_scorecard_wires_facade_through_jsonable(monkeypatch):
    import pytest
    pytest.importorskip("fastapi")
    from stockscan.web import routes
    from stockscan.web.state import STATE

    class _Facade:
        def scorecard(self):
            return {"n_holdings": 1, "percentile_equal": np.float64(60.0),
                    "holdings": [{"cik": 1, "value": np.int64(2000)}]}

    monkeypatch.setattr(STATE, "status", "ready")
    monkeypatch.setattr(STATE, "adata", _Facade())
    out = routes.scorecard()
    assert out["n_holdings"] == 1
    assert out["percentile_equal"] == 60.0 and isinstance(out["percentile_equal"], float)
    assert out["holdings"][0]["value"] == 2000                 # numpy coerced to native
