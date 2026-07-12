"""Row shapers: the risk-head chips on browsing tables (firewalled display flags)."""

import pandas as pd

from stockscan.view.rows import scan_rows, watch_rows


def _cross(with_heads=True):
    d = {
        "cik": [1, 2, 3],
        "ticker": ["AAA", "BBB", "CCC"],
        "name": ["Aaa Inc", "Bbb Corp", "Ccc Ltd"],
        "sector": ["tech", "tech", "energy"],
        "score": [3.0, 2.0, 1.0],
        "pct": [0.95, 0.50, 0.05],
        "fy": [2025, 2025, 2024],
    }
    if with_heads:
        d["dprob"] = [0.02, 0.31, 0.72]
        d["dflag"] = ["none", "elevated", "high"]
        d["wprob"] = [0.20, 0.61, 0.44]
        d["wflag"] = ["none", "elevated", "none"]
    return pd.DataFrame(d)


def test_scan_rows_carry_risk_chips_only_when_flagged():
    rows = scan_rows(_cross())
    by_tk = {r["ticker"]: r for r in rows}
    assert by_tk["AAA"]["risk"] == []                       # clean name: no chips
    assert by_tk["BBB"]["risk"] == [
        {"kind": "distress", "level": "elevated", "prob_pct": 31},
        {"kind": "drawdown", "level": "elevated", "prob_pct": 61},
    ]
    assert by_tk["CCC"]["risk"] == [
        {"kind": "distress", "level": "high", "prob_pct": 72},
    ]


def test_scan_rows_without_head_columns_stay_chipless():
    rows = scan_rows(_cross(with_heads=False))
    assert all(r["risk"] == [] for r in rows)               # heads unfrozen: no crash


def test_watch_rows_carry_risk_chips_and_absent_names_stay_flagged():
    feats = pd.DataFrame({"cik": [3], "available_date": [pd.Timestamp("2025-04-01")]})
    rows = watch_rows(
        [{"cik": 3, "column": "CCC"}, {"cik": 99, "column": "GONE"}],
        _cross(), {}, feats, as_of="2026-07-01",
    )
    assert rows[0]["risk"][0]["kind"] == "distress"
    assert "distress high" in rows[0]["flag"]               # the text flag still rides
    assert rows[1]["risk"] == [] and "not in liquid universe" in rows[1]["flag"]
