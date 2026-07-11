"""Confidence score — derived, bounded, honest, and never inventing certainty.

The score is a transparent function of the calibration table + the call's decile /
percentile / drivers / data-quality flags; these tests pin the properties that keep it
honest (edge-anchored, capped, penalized by bad data / incoherent drivers).
"""

import json

import pytest

from stockscan.confidence import (
    CEILING,
    load_calibration,
    load_calibration_optional,
    score_confidence,
)

CLEAN = {"liquidity_pass": True, "in_sample": False, "staleness_days": 100}
COHERENT = [{"contribution": 0.10}, {"contribution": 0.05}]


def cal(hit_rate: float, decile: int = 10, n: int = 1000) -> dict:
    return {"deciles": {str(decile): {
        "hit_rate": hit_rate, "mean_excess": 0.0, "n": n,
        "ci_low": max(0.0, hit_rate - 0.01), "ci_high": min(1.0, hit_rate + 0.01),
    }}}


def score(hit_rate=0.60, decile=10, percentile=95, drivers=None, flags=None, n=1000,
          downside=None):
    return score_confidence(
        decile, percentile, COHERENT if drivers is None else drivers,
        CLEAN if flags is None else flags, cal(hit_rate, decile, n), downside,
    )


# --- graceful absence -----------------------------------------------------------

def test_none_without_calibration_or_missing_decile():
    assert score_confidence(7, 95, COHERENT, CLEAN, None) is None
    assert score_confidence(None, 95, COHERENT, CLEAN, cal(0.6)) is None
    assert score_confidence(3, 95, COHERENT, CLEAN, cal(0.6, decile=7)) is None  # decile 3 absent


# --- edge-anchored + capped -----------------------------------------------------

def test_bigger_buy_side_directional_edge_scores_higher():
    strong = score(hit_rate=0.60)        # edge 0.10
    weak = score(hit_rate=0.52)          # edge 0.02
    flat = score(hit_rate=0.50)          # edge 0 -> no conviction
    assert strong["score"] > weak["score"] > flat["score"]
    assert flat["score"] == 0


def test_avoid_side_edge_is_symmetric():
    """A reliably-bad bottom decile earns the same conviction as a reliably-good top one."""
    buy = score(hit_rate=0.60, decile=10, percentile=95)
    avoid = score(hit_rate=0.40, decile=1, percentile=5)
    assert buy["score"] == avoid["score"]        # |0.60-0.5| == |0.40-0.5|


def test_wrong_direction_hit_rate_gets_no_confidence():
    """A high decile below 50% hit-rate, or a low decile above 50%, is not convincing."""
    assert score(hit_rate=0.48, decile=10, percentile=95)["score"] == 0
    assert score(hit_rate=0.52, decile=1, percentile=5)["score"] == 0


def test_hold_deciles_get_no_directional_confidence():
    assert score(hit_rate=0.60, decile=6, percentile=60)["score"] == 0


def test_never_exceeds_ceiling():
    # extreme edge + best margin + clean data + fully coherent -> still capped
    s = score(hit_rate=0.95, decile=10, percentile=100)
    assert s["score"] == CEILING <= 85


# --- bounded modifiers only ever lower it ---------------------------------------

def test_data_quality_penalties_lower_it():
    base = score()["score"]
    assert score(flags={"staleness_days": 600})["score"] < base
    assert score(flags={"liquidity_pass": False})["score"] < base
    assert score(flags={"in_sample": True})["score"] < base


def test_downside_risk_penalties_lower_confidence_without_erasing_track_record():
    base = score(hit_rate=0.56)
    elevated = score(hit_rate=0.56, downside={"drawdown": {"flag": "elevated", "prob": 0.56}})
    high = score(hit_rate=0.56, downside={"drawdown": {"flag": "high", "prob": 0.72}})

    assert base["score"] > elevated["score"] > high["score"]
    assert high["hit_rate"] == base["hit_rate"] and high["n"] == base["n"]
    assert high["components"]["downside_risk"] < elevated["components"]["downside_risk"] < 1.0


def test_incoherent_drivers_lower_it():
    aligned = score(drivers=[{"contribution": 0.10}, {"contribution": 0.08}])
    cancel = score(drivers=[{"contribution": 0.10}, {"contribution": -0.09}])
    assert aligned["score"] > cancel["score"]


def test_deeper_in_zone_scores_higher():
    # modest edge so the score sits below the ceiling and the margin effect is visible
    deep = score(hit_rate=0.55, decile=10, percentile=99)
    shallow = score(hit_rate=0.55, decile=10, percentile=81)
    assert deep["score"] > shallow["score"]


# --- the track record always rides along ----------------------------------------

def test_hit_rate_and_n_pass_through():
    s = score(hit_rate=0.58, n=1234)
    assert s["hit_rate"] == 0.58 and s["n"] == 1234
    assert 0 <= s["score"] <= CEILING
    assert set(s["components"]) == {
        "basis", "t_excess", "edge", "conviction_base", "margin", "data_quality",
        "coherence", "downside_risk"
    }
    assert s["components"]["basis"] == "hit_rate"  # legacy artifact: no t_excess


# --- t_excess anchor (magnitude-carried edge) ------------------------------------

def tcal(t_excess, hit_rate=0.49, decile=10, n=1000) -> dict:
    c = cal(hit_rate, decile, n)
    c["deciles"][str(decile)]["t_excess"] = t_excess
    return c


def test_t_anchor_awards_buy_confidence_despite_sub_half_hit_rate():
    """The real shape of this model's edge: decile 10 beats the median <50% of the
    time but its AVERAGE excess is reliably positive — conviction must not be zero."""
    s = score_confidence(10, 95, COHERENT, CLEAN, tcal(t_excess=2.0, hit_rate=0.49))
    assert s["score"] > 0
    assert s["components"]["basis"] == "t_excess"
    assert s["hit_rate"] == 0.49  # the frequency truth still rides along


def test_t_anchor_is_directional():
    # wrong-direction average excess earns nothing, either side
    assert score_confidence(10, 95, COHERENT, CLEAN, tcal(t_excess=-1.5))["score"] == 0
    assert score_confidence(
        1, 5, COHERENT, CLEAN, tcal(t_excess=1.5, hit_rate=0.43, decile=1))["score"] == 0
    # AVOID side: reliably NEGATIVE mean excess is convincing
    s = score_confidence(1, 5, COHERENT, CLEAN, tcal(t_excess=-6.0, hit_rate=0.43, decile=1))
    assert s["score"] > 0


def test_t_anchor_scales_and_caps():
    weak = score_confidence(10, 95, COHERENT, CLEAN, tcal(t_excess=1.0))
    strong = score_confidence(10, 95, COHERENT, CLEAN, tcal(t_excess=4.0))
    extreme = score_confidence(10, 95, COHERENT, CLEAN, tcal(t_excess=40.0))
    assert weak["score"] < strong["score"] <= extreme["score"] <= CEILING


def test_hold_deciles_ignore_t_excess():
    assert score_confidence(6, 60, COHERENT, CLEAN, tcal(t_excess=9.0, decile=6))["score"] == 0


# --- production artifact regression (the 2026-07-11 BUY-dead bug) -----------------

def test_production_calibration_not_buy_dead():
    """Guards the live artifact shape: a top-decile BUY at a deep percentile must earn
    nonzero confidence, and the AVOID side must stay strong. Skips when the local
    artifact is absent (CI has no artifacts/)."""
    from stockscan.confidence import CALIBRATION_PATH, load_calibration_optional

    calib = load_calibration_optional(CALIBRATION_PATH)
    if calib is None:
        pytest.skip("no local calibration artifact")
    buy = score_confidence(10, 95, COHERENT, CLEAN, calib)
    avoid = score_confidence(1, 5, COHERENT, CLEAN, calib)
    assert buy is not None and avoid is not None
    assert buy["score"] > 0, "BUY side of confidence is dead again"
    assert avoid["score"] > 0
    assert avoid["score"] >= buy["score"]  # this model's edge is avoid-heavier; if this
    # ever flips it is worth a fresh look at the calibration, not a silent pass


# --- artifact I/O ---------------------------------------------------------------

def test_optional_loader_returns_none_when_absent(tmp_path):
    assert load_calibration_optional(tmp_path / "nope.json") is None
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "nope.json")


def test_load_calibration_roundtrip(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(cal(0.6)))
    assert load_calibration(p)["deciles"]["10"]["hit_rate"] == 0.6
