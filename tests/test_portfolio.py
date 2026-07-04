"""The portfolio scorecard is a pure, display-only aggregation — test it over tiny
DataFrames with no data store. Covers the holdings→cross join, the two weightings
(equal + position-value), distress exposure, concentration ordering, and the
honesty guarantees (unlisted names kept, full holdings list always returned)."""

import numpy as np
import pandas as pd

from stockscan.portfolio import UNLISTED, holdings_join, scorecard


def _cross():
    # cik 1 top-decile semi (normal), cik 2 low-pct bank (elevated distress),
    # cik 3 mid software (high distress). cik 9 is deliberately absent.
    return pd.DataFrame({
        "cik":    [1, 2, 3],
        "ticker": ["AAA", "BBB", "CCC"],
        "name":   ["Alpha", "Beta", "Gamma"],
        "sector": ["Tech", "Fin", "Tech"],
        "sic":    [3674, 6021, 7372],          # Semiconductors, Banks, Software
        "score":  [0.9, 0.1, 0.5],
        "pct":    [0.90, 0.30, 0.60],
        "dprob":  [np.nan, 0.05, 0.12],
        "dflag":  ["normal", "elevated", "high"],
    })


def _positions():
    return [
        {"cik": 1, "shares": 100, "cost_basis": 10.0, "added_at": "2026-01-01"},
        {"cik": 2, "shares": 50, "cost_basis": 40.0, "added_at": "2026-02-01"},
        {"cik": 3, "shares": 10, "cost_basis": 100.0, "added_at": "2026-03-01"},
        {"cik": 9, "shares": 5, "cost_basis": 50.0, "added_at": "2026-04-01"},  # unlisted
    ]


_PRICES = {1: 20.0, 2: 30.0, 3: 100.0}          # cik 9 has no price


# --- holdings_join -------------------------------------------------------------

def test_join_shapes_listed_holding_with_standing_value_and_pl():
    rows = {h["cik"]: h for h in holdings_join(_positions(), _cross(), _PRICES)}
    a = rows[1]
    assert a["in_universe"] and a["status"] == "listed" and a["owned"] is True
    assert a["ticker"] == "AAA" and a["industry"] == "Semiconductors"
    assert a["pct"] == 90 and a["decile"] == 9         # ceil(0.90*10)=9; decile 10 needs >0.9
    assert a["dflag"] == "normal" and a["dprob"] is None        # NaN -> None
    assert a["value"] == 2000.0 and a["cost"] == 1000.0
    assert a["unrealized_pl"] == 1000.0 and a["unrealized_pl_pct"] == 100.0


def test_join_keeps_unlisted_holding_flagged_not_dropped():
    rows = {h["cik"]: h for h in holdings_join(_positions(), _cross(), _PRICES)}
    u = rows[9]
    assert u["in_universe"] is False and u["status"] == UNLISTED
    assert u["pct"] is None and u["decile"] is None and u["dflag"] is None
    assert u["value"] is None                                   # no price supplied


def test_join_without_prices_leaves_value_none_but_keeps_standing():
    rows = {h["cik"]: h for h in holdings_join(_positions(), _cross())}
    assert rows[1]["value"] is None and rows[1]["unrealized_pl"] is None
    assert rows[1]["pct"] == 90                                 # standing still resolves


def test_join_reads_distress_prob_when_present():
    rows = {h["cik"]: h for h in holdings_join(_positions(), _cross(), _PRICES)}
    assert rows[2]["dflag"] == "elevated" and rows[2]["dprob"] == 0.05
    assert rows[3]["dflag"] == "high" and rows[3]["dprob"] == 0.12


def test_join_watchlist_only_name_has_standing_but_no_value():
    # a followed-but-not-held name: shares/cost None -> not owned, no value/P&L,
    # but its model standing still resolves from the cross-section.
    rows = holdings_join([{"cik": 1, "shares": None, "cost_basis": None, "added_at": "x"}],
                         _cross(), _PRICES)
    r = rows[0]
    assert r["owned"] is False
    assert r["in_universe"] is True and r["pct"] == 90 and r["dflag"] == "normal"
    assert r["value"] is None and r["cost"] is None and r["unrealized_pl"] is None


# --- scorecard aggregates ------------------------------------------------------

def test_scorecard_counts_and_book_value():
    sc = scorecard(_positions(), _cross(), _PRICES, as_of="2026-07-04")
    assert sc["as_of"] == "2026-07-04"
    assert sc["n_total"] == 4 and sc["n_owned"] == 4 and sc["n_watch"] == 0
    assert sc["n_listed"] == 3 and sc["n_unlisted"] == 1
    assert sc["total_value"] == 4500.0                          # cik9 (no price) excluded
    assert sc["total_cost"] == 4000.0                           # same priced set as value → P/L consistent
    assert sc["unrealized_pl"] == 500.0
    assert len(sc["holdings"]) == 4                             # full list always returned


def test_scorecard_mixes_owned_holdings_and_watchlist_only_names():
    # cik1,2 held; cik3 followed-only (no shares). All three feed model standing;
    # only the held+priced ones feed value / value-weighting.
    entries = _positions()[:2] + [{"cik": 3, "shares": None, "cost_basis": None,
                                   "added_at": "x"}]
    sc = scorecard(entries, _cross(), _PRICES)
    assert sc["n_total"] == 3 and sc["n_owned"] == 2 and sc["n_watch"] == 1
    assert sc["percentile_equal"] == 60.0                       # mean(90,30,60) — all 3
    # value-weight only over the 2 held+priced: (90*2000 + 30*1500)/3500
    assert sc["percentile_value"] == round((90 * 2000 + 30 * 1500) / 3500, 1)
    assert sc["total_value"] == 3500.0                          # cik3 has no shares
    assert sc["distress"]["count"]["high"] == 1                 # cik3 still counts (high)


def test_scorecard_shows_both_weightings_and_they_differ():
    sc = scorecard(_positions(), _cross(), _PRICES)
    assert sc["percentile_equal"] == 60.0                       # mean(90,30,60)
    # value-weighted: (90*2000 + 30*1500 + 60*1000)/4500 = 63.33
    assert sc["percentile_value"] == 63.3
    # the two differ — the whole point of showing both (no single number stands in)
    assert sc["percentile_equal"] != sc["percentile_value"]


def test_scorecard_value_weight_falls_back_to_none_without_prices():
    sc = scorecard(_positions(), _cross())                      # no prices
    assert sc["percentile_equal"] == 60.0                       # still available
    assert sc["percentile_value"] is None                       # can't value-weight
    assert sc["total_value"] is None and sc["unrealized_pl"] is None


def test_scorecard_distress_exposure_counts_and_at_risk():
    sc = scorecard(_positions(), _cross(), _PRICES)
    d = sc["distress"]
    assert d["known"] is True
    assert d["count"] == {"high": 1, "elevated": 1, "normal": 1}
    assert d["at_risk"] == 2                                    # high + elevated
    assert d["value"]["high"] == 1000.0 and d["value"]["elevated"] == 1500.0


def test_scorecard_distress_unknown_when_no_flag_column():
    cross = _cross().drop(columns=["dprob", "dflag"])
    sc = scorecard(_positions(), cross, _PRICES)
    assert sc["distress"]["known"] is False
    assert sc["distress"]["at_risk"] == 0 and sc["distress"]["value"] is None


def test_scorecard_concentration_is_value_weighted_and_biggest_first():
    sc = scorecard(_positions(), _cross(), _PRICES)
    ind = sc["industry_concentration"]
    assert [b["name"] for b in ind][0] == "Semiconductors"     # biggest by value (2000)
    top = ind[0]
    assert top["count"] == 1 and abs(top["weight_value"] - 2000 / 4500) < 1e-9
    # sector view collapses Tech (cik1+cik3) ahead of Fin
    sec = sc["sector_concentration"]
    assert sec[0]["name"] == "Tech" and sec[0]["count"] == 2
    assert abs(sec[0]["weight_value"] - 3000 / 4500) < 1e-9


def test_scorecard_concentration_excludes_watchlist_only_names():
    # cik1 held (Tech), cik2 held (Fin), cik3 watch-only (Tech). Concentration reflects
    # HELD money only — cik3 must not add a phantom bucket or inflate Tech's count.
    entries = _positions()[:2] + [{"cik": 3, "shares": None, "cost_basis": None,
                                   "added_at": "x"}]
    sc = scorecard(entries, _cross(), _PRICES)
    sectors = {b["name"]: b for b in sc["sector_concentration"]}
    assert set(sectors) == {"Tech", "Fin"}
    assert sectors["Tech"]["count"] == 1                 # only cik1 (held), not cik3 (watched)


def test_scorecard_concentration_uses_count_when_no_value():
    sc = scorecard(_positions(), _cross())                      # no prices
    for b in sc["sector_concentration"]:
        assert b["weight_value"] is None and b["weight_count"] is not None
    assert sc["sector_concentration"][0]["name"] == "Tech"      # 2 names > 1


def test_scorecard_empty_book_is_all_zeros_not_a_crash():
    sc = scorecard([], _cross())
    assert sc["n_total"] == 0 and sc["n_owned"] == 0 and sc["holdings"] == []
    assert sc["percentile_equal"] is None and sc["percentile_value"] is None
    assert sc["total_value"] is None
    assert sc["distress"]["known"] is False and sc["distress"]["at_risk"] == 0
    assert sc["industry_concentration"] == [] and sc["sector_concentration"] == []
