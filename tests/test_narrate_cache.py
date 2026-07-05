"""Materiality gate + narration cache: right tier, right reuse, never stale-material."""

import json

import pytest

from stockscan.narrate.cache import NarrationCache, materiality, narrate_smart, packet_hash


def _packet(pct=80, period_end="2025-09-30", drivers=("roa", "op_margin", "leverage"),
            value=31.2):
    return {
        "meta": {"ticker": "T1", "name": "ALPHA", "cik": 1, "fiscal_year": 2025,
                 "period_end": period_end, "sector": "Manufacturing"},
        "signals": [{"id": "roa", "label": "Return on assets", "value": value,
                     "unit": "%", "pct_rank": 90, "direction": "higher-is-better"}],
        "composite": {"label": "x", "percentile": 75},
        "model": {"label": "x", "score": 0.01, "percentile": pct, "decile": 9,
                  "n_names": 100, "as_of": "2026-06-30",
                  "trained_through": "2026-03-31",
                  "drivers": [{"id": d, "label": d, "contribution": 0.001,
                               "direction": "supports"} for d in drivers]},
        "disclaimer": "Not investment advice.",
    }


@pytest.fixture
def cache(tmp_path):
    c = NarrationCache(tmp_path / "cache.sqlite")
    yield c
    c.close()


def _counting_llm(tag, calls):
    def llm(system, user):
        calls.append(tag)
        pkt = json.loads(user)
        s = pkt["signals"][0]
        return json.dumps({
            "reasoning": "",
            "summary": f"[{tag}] {s['label']} {s['value']}{s['unit']} at "
                       f"{s['pct_rank']}th pct.",
            "citations": [{"id": s["id"], "direction": "supports"}],
        })
    return llm


def test_materiality_classes():
    p = _packet()
    assert materiality(None, p) == "material"                # cold start
    prev = {"packet_hash": packet_hash(p), "period_end": p["meta"]["period_end"],
            "percentile": 80, "drivers": ["roa", "op_margin", "leverage"]}
    assert materiality(prev, p) == "unchanged"
    assert materiality(prev, _packet(pct=82, value=31.4)) == "minor"     # wiggle
    assert materiality(prev, _packet(pct=95)) == "material"              # big move
    assert materiality(prev, _packet(period_end="2026-09-30")) == "material"  # new 10-K
    assert materiality(prev, _packet(drivers=("accruals", "roe", "roa"))) == "material"


def test_volatile_fields_do_not_bust_the_cache():
    """A daily re-query (new as-of, score wiggle, cross-section size change) with
    identical fundamentals must be 'unchanged' — else the cache never hits."""
    p1 = _packet()
    p2 = _packet(pct=81)  # small percentile drift, same fundamentals
    p2["meta"]["as_of"] = "2026-07-01"
    p2["model"]["as_of"] = "2026-07-01"
    p2["model"]["score"] = 0.0123
    p2["model"]["n_names"] = 101
    assert packet_hash(p1) == packet_hash(p2)
    prev = {"packet_hash": packet_hash(p1), "period_end": p1["meta"]["period_end"],
            "percentile": 80, "drivers": ["roa", "op_margin", "leverage"]}
    assert materiality(prev, p2) == "unchanged"
    # ...but a big percentile move is material even with the same durable hash
    p3 = _packet(pct=95)
    p3["meta"]["as_of"] = "2026-07-01"
    assert materiality(prev, p3) == "material"


def test_light_tier_does_not_ratchet_the_materiality_baseline(cache):
    """Successive minor drifts must be judged against the last FULL narration's
    baseline — a slow large move cannot creep past the threshold in small steps."""
    calls = []
    full = _counting_llm("full", calls)
    light = _counting_llm("light", calls)
    narrate_smart(_packet(pct=80), llm_full=full, llm_light=light, cache=cache)
    # drift +8 (minor), then +8 more: cumulative +16 vs the FULL baseline = material
    r2 = narrate_smart(_packet(pct=88, value=32.0), llm_full=full, llm_light=light,
                       cache=cache)
    assert r2["tier"] == "light"
    r3 = narrate_smart(_packet(pct=96, value=33.0), llm_full=full, llm_light=light,
                       cache=cache)
    assert r3["tier"] == "full"          # judged vs 80, not vs 88
    assert calls == ["full", "light", "full"]


def test_narrate_smart_tiers_and_cache_roundtrip(cache):
    calls = []
    full = _counting_llm("full", calls)
    light = _counting_llm("light", calls)

    r1 = narrate_smart(_packet(), llm_full=full, llm_light=light, cache=cache)
    assert r1["tier"] == "full" and calls == ["full"]        # cold start -> full

    r2 = narrate_smart(_packet(), llm_full=full, llm_light=light, cache=cache)
    assert r2["tier"] == "cache" and calls == ["full"]       # unchanged -> no LLM call
    assert r2["narrative"] == r1["narrative"]

    r3 = narrate_smart(_packet(pct=82, value=31.4), llm_full=full, llm_light=light,
                       cache=cache)
    assert r3["tier"] == "light" and calls == ["full", "light"]  # minor -> 14B tier

    r4 = narrate_smart(_packet(pct=40, value=12.0), llm_full=full, llm_light=light,
                       cache=cache)
    assert r4["tier"] == "full" and calls == ["full", "light", "full"]  # material


def test_cache_result_is_still_validated_output(cache):
    calls = []
    r = narrate_smart(_packet(), llm_full=_counting_llm("full", calls), cache=cache)
    assert r["grounded"]
    again = narrate_smart(_packet(), llm_full=_counting_llm("full", calls), cache=cache)
    assert again["tier"] == "cache" and again["grounded"]
    assert again["citations"]  # citations survive the cache round-trip


def test_template_tier_is_never_cached(cache):
    """A template run (no LLM available) must NOT enter the cache: caching it
    would let a later --no-llm invocation overwrite a real full-tier narration
    and, via the materiality baseline reset, serve the template forever."""
    r = narrate_smart(_packet(), cache=cache)
    assert r["tier"] == "template" and r["grounded"]
    assert cache.get(_packet()["meta"]["cik"]) is None  # nothing cached
    r2 = narrate_smart(_packet(), cache=cache)
    assert r2["tier"] == "template"  # still template, not a cache hit


def test_dead_endpoint_fallback_is_never_cached(cache):
    """An llm that is passed but DOWN degrades inside narrate_packet to the
    template fallback — that is a template run too: tier must say so and it must
    stay out of the cache, or the next healthy run would classify 'unchanged'
    and serve the degraded text forever."""
    def dead(system, user):
        raise ConnectionError("endpoint down")

    r = narrate_smart(_packet(), llm_full=dead, cache=cache)
    assert r["tier"] == "template" and r["source"] == "template-fallback"
    assert r["grounded"]                                # degraded, never broken
    assert cache.get(_packet()["meta"]["cik"]) is None  # the fallback never enters
    # endpoint back up -> a real full narration runs (no false 'unchanged' hit)
    calls = []
    r2 = narrate_smart(_packet(), llm_full=_counting_llm("full", calls), cache=cache)
    assert r2["tier"] == "full" and calls == ["full"]
