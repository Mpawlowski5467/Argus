"""Tests for packet assembly + the cited-JSON grounded narration (LLM mocked)."""

import json

import pandas as pd

from stockscan.features import FEATURES
from stockscan.narrate.ground import is_grounded
from stockscan.narrate.narrator import (
    SYSTEM,
    expected_directions,
    narrate,
    narrate_packet,
    parse_llm_json,
    validate_narration,
)
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


_NEWS = [
    {"id": "new_ABC123", "date": "2026-03-15", "source": "reuters.com",
     "event_type": "M&A", "takeaway": "Reported acquisition of a rival announced"},
]


def _good_llm(system, user):
    pkt = json.loads(user)
    s = pkt["signals"][0]
    exp = expected_directions(pkt)
    return json.dumps({
        "reasoning": f"{s['label']} sits at the {s['pct_rank']}th percentile.",
        "summary": (f"{pkt['meta']['name']} shows {s['label']} of {s['value']}"
                    f"{s['unit']} at the {s['pct_rank']}th percentile."),
        "citations": [{"id": s["id"], "direction": exp[s["id"]]}],
    })


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
    # template citations agree with the packet by construction
    assert validate_narration(
        {"reasoning": "", "summary": r["narrative"], "citations": r["citations"]},
        r["packet"],
    ) == []


def test_narrate_accepts_cited_json_output():
    r = narrate(1, llm=_good_llm, features_df=_feats())
    assert r["source"] == "llm"
    assert r["grounded"] and r["first_pass_ok"] and r["attempts"] == 1
    assert r["citations"]


def test_narrator_system_carries_the_iso_date_rule():
    """Narration runs its OWN loop (not core.grounded_answer), so it can't inherit
    _DATE_RULE — the keep-dates-verbatim instruction must live in SYSTEM directly,
    or a reworded date leaks a day-number to the same guard and drops to template."""
    assert "YYYY-MM-DD" in SYSTEM and "reworded date" in SYSTEM


def test_narrate_falls_back_on_hallucinated_number():
    def bad_llm(system, user):
        return json.dumps({"reasoning": "", "summary":
                           "This company has a spectacular secret ROA of 999999%.",
                           "citations": []})

    r = narrate(1, llm=bad_llm, features_df=_feats())
    assert r["source"] == "template-fallback"
    assert r["grounded"]  # the fallback is grounded by construction
    assert any(isinstance(v, float) for v in r["violations"])


def test_llm_transport_error_degrades_to_template_not_crash():
    def flaky_llm(system, user):
        raise TimeoutError("simulated read timeout")

    r = narrate(1, llm=flaky_llm, features_df=_feats())
    assert r["source"] == "template-fallback"
    assert r["grounded"]
    assert any(str(v).startswith("llm-error") for v in r["violations"])


def test_narrate_falls_back_on_unparseable_reply():
    r = narrate(1, llm=lambda s, u: "Sure! Here's my analysis: it's great.",
                features_df=_feats())
    assert r["source"] == "template-fallback"
    assert "unparseable-json" in r["violations"]


def test_direction_disagreement_is_caught():
    """The LLM cannot call a weakness a strength: a citation whose direction
    contradicts the packet's own sign is a violation (outside the median band)."""
    pkt = build_packet(1, features_df=_feats())
    pkt["signals"][0]["pct_rank"] = 90  # decisively strong, not in the 45-55 band
    sid = pkt["signals"][0]["id"]
    right = expected_directions(pkt)[sid]
    wrong = "detracts" if right == "supports" else "supports"
    v = validate_narration(
        {"reasoning": "", "summary": "A fine company.", "citations":
         [{"id": sid, "direction": wrong}]}, pkt)
    assert any(str(x).startswith("direction-disagrees") for x in v)
    v_ok = validate_narration(
        {"reasoning": "", "summary": "A fine company.", "citations":
         [{"id": sid, "direction": right}]}, pkt)
    assert v_ok == []


def test_unknown_citation_id_is_caught():
    pkt = build_packet(1, features_df=_feats())
    v = validate_narration(
        {"reasoning": "", "summary": "Fine.", "citations":
         [{"id": "made_up_signal", "direction": "supports"}]}, pkt)
    assert any(str(x).startswith("unknown-citation-id") for x in v)


def test_parse_llm_json_tolerates_fences_and_prose():
    obj = {"reasoning": "r", "summary": "s", "citations": []}
    assert parse_llm_json(json.dumps(obj)) == obj
    assert parse_llm_json("```json\n" + json.dumps(obj) + "\n```") == obj
    assert parse_llm_json("Here you go:\n" + json.dumps(obj)) == obj
    assert parse_llm_json("no json at all") is None
    # braces in surrounding prose must not corrupt the extraction (review finding)
    assert parse_llm_json(json.dumps(obj) + "\nNote: {caveat} applies.") == obj
    assert parse_llm_json("Using format {id, direction}:\n" + json.dumps(obj)) == obj
    # a preamble example object without a summary cannot shadow the real reply
    assert parse_llm_json('{"id": "x"}\n' + json.dumps(obj)) == obj


def test_uncited_mention_of_a_signal_is_rejected():
    """The direction guard is not opt-out: mentioning a signal by name without
    citing it (the wrong-direction evasion route) is a violation."""
    pkt = build_packet(1, features_df=_feats())
    label = pkt["signals"][0]["label"]
    sid = pkt["signals"][0]["id"]
    v = validate_narration(
        {"reasoning": "", "summary": f"The company's {label} is remarkable.",
         "citations": []}, pkt)
    assert any(str(x).startswith("uncited-mention") for x in v)
    exp = expected_directions(pkt)
    v_ok = validate_narration(
        {"reasoning": "", "summary": f"The company's {label} is remarkable.",
         "citations": [{"id": sid, "direction": exp[sid]}]}, pkt)
    assert v_ok == []


def test_median_signals_accept_either_direction():
    pkt = build_packet(1, features_df=_feats())
    # force a signal to sit exactly at the median
    pkt["signals"][0]["pct_rank"] = 50
    sid = pkt["signals"][0]["id"]
    for direction in ("supports", "detracts"):
        v = validate_narration(
            {"reasoning": "", "summary": "Fine.", "citations":
             [{"id": sid, "direction": direction}]}, pkt)
        assert not any(str(x).startswith("direction-disagrees") for x in v)


def test_template_never_repeats_a_signal_in_strong_and_weak():
    pkt = build_packet(1, features_df=_feats())
    pkt["signals"] = pkt["signals"][:4]  # thin packet (e.g. a financials filer)
    from stockscan.narrate.narrator import _template
    text = _template(pkt)
    mentioned = [s["label"] for s in pkt["signals"] if s["label"] in text]
    for label in mentioned:
        assert text.count(label) == 1, f"{label} listed as both strong and weak"


def test_template_huge_values_stay_grounded():
    pkt = build_packet(1, features_df=_feats())
    pkt["signals"][0]["value"] = 2.34e16  # junk ratio from a near-zero denominator
    r = narrate_packet(pkt)  # template path re-checks grounding honestly
    assert "e+" not in r["narrative"] and "E+" not in r["narrative"]
    assert r["grounded"], r.get("template_leaks")


def _news_llm(system, user):
    pkt = json.loads(user)
    s = pkt["signals"][0]
    exp = expected_directions(pkt)
    a = pkt["context"]["news"][0]
    return json.dumps({
        "reasoning": f"{s['label']} sits at the {s['pct_rank']}th percentile.",
        "summary": (f"{pkt['meta']['name']} shows {s['label']} of {s['value']}{s['unit']} "
                    f"at the {s['pct_rank']}th percentile. Separately, {a['source']} "
                    f"reported a {a['event_type']} development."),
        "citations": [{"id": s["id"], "direction": exp[s["id"]]},
                      {"id": a["id"], "direction": "reported"}],
    })


def test_build_packet_attaches_number_free_news_context():
    pkt = build_packet(1, features_df=_feats(), news=[
        {"id": "new_X", "date": "2026-03-15", "source": "reuters.com",
         "event_type": "guidance", "takeaway": "Guidance cut to $873 million for the year"}])
    entry = pkt["context"]["news"][0]
    assert entry["id"] == "new_X" and entry["event_type"] == "guidance"
    assert "873" not in entry["takeaway"]          # every numeral stripped
    # a news date's YEAR still grounds (via the date field); a fabricated figure does not
    assert is_grounded("Reported in March 2026.", pkt)
    assert not is_grounded("A fabricated 873% jump.", pkt)


def test_news_free_packet_is_unchanged():
    """A packet built without news has no context key — byte-identical to before."""
    assert "context" not in build_packet(1, features_df=_feats())
    assert "context" not in build_packet(1, features_df=_feats(), news=[])


def test_fabricated_number_still_caught_with_news_context():
    """The acceptance guard: news context present must NOT weaken the fabrication
    check. A number that appeared only in the RAW article (stripped from the packet
    takeaway) is still a hallucination when the narration emits it."""
    pkt = build_packet(1, features_df=_feats(), news=[
        {"id": "new_ACQ", "date": "2026-03-15", "source": "reuters.com",
         "event_type": "M&A", "takeaway": "Reported buyback of 873 million shares"}])

    def bad_llm(system, user):
        return json.dumps({
            "reasoning": "",
            "summary": "Following the reported buyback, ROA jumped an incredible 873%.",
            "citations": [{"id": "new_ACQ", "direction": "reported"}],
        })

    r = narrate_packet(pkt, llm=bad_llm)
    assert r["source"] == "template-fallback"
    assert 873.0 in r["violations"]                # the article's own number is NOT blessed


def test_narrate_accepts_valid_news_citation():
    pkt = build_packet(1, features_df=_feats(), news=_NEWS)
    r = narrate_packet(pkt, llm=_news_llm)
    assert r["source"] == "llm" and r["grounded"] and r["first_pass_ok"]
    assert any(c["id"] == _NEWS[0]["id"] and c["direction"] == "reported"
               for c in r["citations"])


def test_unknown_news_citation_is_caught():
    pkt = build_packet(1, features_df=_feats(), news=_NEWS)
    v = validate_narration(
        {"reasoning": "", "summary": "Fine.",
         "citations": [{"id": "new_DOESNOTEXIST", "direction": "reported"}]}, pkt)
    assert any(str(x).startswith("unknown-citation-id") for x in v)


def test_news_citation_must_be_reported_not_a_signal_direction():
    """News carries no supports/detracts sign — that would be the firewall leaking a
    signal into the score's framing. Only the neutral 'reported' direction is valid."""
    pkt = build_packet(1, features_df=_feats(), news=_NEWS)
    nid = _NEWS[0]["id"]
    v = validate_narration(
        {"reasoning": "", "summary": "Fine.",
         "citations": [{"id": nid, "direction": "supports"}]}, pkt)
    assert any(str(x).startswith("news-bad-direction") for x in v)
    v_ok = validate_narration(
        {"reasoning": "", "summary": "Fine.",
         "citations": [{"id": nid, "direction": "reported"}]}, pkt)
    assert v_ok == []


def test_packet_hash_ignores_news_context():
    """Live headlines must never invalidate a narration cached on unchanged funds."""
    from stockscan.narrate.cache import packet_hash

    assert packet_hash(build_packet(1, features_df=_feats())) == \
        packet_hash(build_packet(1, features_df=_feats(), news=_NEWS))


def test_driver_directions_enter_expected_and_template():
    pkt = build_packet(1, features_df=_feats())
    pkt["model"] = {
        "label": "x", "score": 0.01, "percentile": 88, "decile": 9, "n_names": 100,
        "as_of": "2026-06-30", "trained_through": "2026-03-31",
        "drivers": [
            {"id": "driver:roa", "label": "Return on assets", "contribution": 0.003,
             "direction": "supports"},
            {"id": "driver:leverage", "label": "Leverage (liabilities/assets)",
             "contribution": -0.002, "direction": "detracts"},
        ],
    }
    exp = expected_directions(pkt)
    assert exp["driver:roa"] == "supports"
    assert exp["driver:leverage"] == "detracts"
    assert exp["model"] == "supports"
    r = narrate_packet(pkt)  # template mode
    assert "Model drivers:" in r["narrative"]
    assert "supported by Return on assets" in r["narrative"]
    assert "held back by Leverage" in r["narrative"]
    assert {"id": "driver:roa", "direction": "supports"} in r["citations"]


def test_make_llm_tiers():
    """The factory is the ONE construction seam: chat is capped + reasoning-off,
    narration tiers stay uncapped, unknown tiers are loud."""
    import pytest

    from stockscan.config import LLM_CHAT_MAX_TOKENS, LLM_LIGHT_MODEL, LLM_MODEL
    from stockscan.narrate.llm import make_llm

    full, light, chat = make_llm("full"), make_llm("light"), make_llm("chat")
    assert full.model == LLM_MODEL and full.max_tokens is None
    assert light.model == LLM_LIGHT_MODEL and light.max_tokens is None
    assert chat.max_tokens == LLM_CHAT_MAX_TOKENS
    with pytest.raises(ValueError):
        make_llm("cloud")   # no cloud tier exists, by decision
