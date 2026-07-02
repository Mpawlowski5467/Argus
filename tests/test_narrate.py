"""Tests for packet assembly + grounded narration (LLM mocked -- no model needed)."""

import json

import pandas as pd

from stockscan.features import FEATURES
from stockscan.narrate.ground import is_grounded
from stockscan.narrate.narrator import narrate
from stockscan.narrate.packet import build_packet


def _feats():
    rows = []
    for cik, name, val in [(1, "ALPHA", 0.12), (2, "BETA", 0.35)]:
        for fy in (2024, 2025):
            r = {"cik": cik, "name": name, "sic": 3571, "fy": fy,
                 "period_end": pd.Timestamp(f"{fy}-09-30")}
            for f in FEATURES:
                r[f] = val + (0.01 if fy == 2025 else 0.0)
            rows.append(r)
    return pd.DataFrame(rows)


def test_build_packet_structure_and_self_grounding():
    pkt = build_packet(1, features_df=_feats())
    assert pkt["meta"]["name"] == "ALPHA"
    assert pkt["meta"]["fiscal_year"] == 2025
    assert pkt["signals"] and all(isinstance(s["value"], (int, float)) for s in pkt["signals"])
    assert pkt["composite"]["percentile"] is not None
    s = pkt["signals"][0]
    assert is_grounded(f"{s['label']} was {s['value']}{s['unit']} at {s['pct_rank']}th pct", pkt)


def test_narrate_template_without_llm():
    r = narrate(1, features_df=_feats())
    assert r["source"] == "template"
    assert r["grounded"]


def test_narrate_accepts_grounded_llm_output():
    feats = _feats()

    def mock_llm(system, user):
        pkt = json.loads(user)
        s = pkt["signals"][0]
        return f"{pkt['meta']['name']} shows {s['label']} of {s['value']}{s['unit']} at the {s['pct_rank']}th percentile."

    r = narrate(1, llm=mock_llm, features_df=feats)
    assert r["source"] == "llm"
    assert r["grounded"]


def test_narrate_falls_back_on_hallucinated_number():
    def bad_llm(system, user):
        return "This company has a spectacular secret ROA of 999999%."

    r = narrate(1, llm=bad_llm, features_df=_feats())
    assert r["source"] == "template-fallback"
    assert r["rejected_numbers"]
