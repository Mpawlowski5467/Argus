"""Confidence score — how much to trust a BUY/HOLD/AVOID call, DERIVED not trained.

This is not a second model. It is a transparent read of the return model's own
out-of-sample track record: an offline calibration artifact records, per prediction
decile, how often that decile ACTUALLY beat the cross-section OOS (its ``hit_rate``).
At serve time :func:`score_confidence` turns that hit-rate into a 0-100 conviction and
bounds it by how deep the name sits in its zone, the data quality, and whether the SHAP
drivers agree — never letting the number imply a certainty the small edge (IC ~0.03-0.05)
cannot support.

FIREWALL: like the distress head, confidence is display/risk ONLY. It reads the frozen
score's decile/percentile/drivers/flags and never writes back into score, percentile,
decile, drivers, the packet, paper, or any trade rule. It is attached in
``serve.analyze`` AFTER the model block is fixed, and degrades to ``None`` when no
calibration artifact is present (serve then behaves exactly as before).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import ARTIFACTS_DIR

CALIBRATION_DIR = ARTIFACTS_DIR / "confidence_cal"
CALIBRATION_PATH = CALIBRATION_DIR / "calibration.json"

# --- honesty knobs --------------------------------------------------------------
# A decile that beat the cross-section 60% of the time (edge 0.10) reads as full base
# conviction; nothing beyond that raises it. Given the model's true edge is small, the
# ceiling caps the displayed number well under "certain" no matter how the modifiers land.
EDGE_FULL = 0.10
CEILING = 85


# --- bounded modifiers (each in roughly [0.7, 1.0]) -----------------------------

def _margin_factor(percentile: float | None) -> float:
    """How deep the name sits in its call's zone (BUY >=80th, AVOID <40th, HOLD between).

    Depth 0 at the zone boundary, 1 at the extreme; a HOLD (no strong call) gets the
    floor. Only a mild adjustment (0.85..1.0) so it qualifies, never dominates.
    """
    if percentile is None or not math.isfinite(percentile):
        return 0.85
    p = float(percentile)
    if p >= 80:
        depth = (p - 80) / 20.0           # 80th -> 0, 100th -> 1
    elif p < 40:
        depth = (40 - p) / 40.0           # 40th -> 0, 0th -> 1
    else:
        depth = 0.0                        # HOLD band: weak call by construction
    return 0.85 + 0.15 * max(0.0, min(1.0, depth))


def _data_quality_factor(flags: dict | None) -> float:
    """Stale filing / below the liquidity floor / in-sample as-of each knock confidence
    down (multiplicatively, floored). All three come straight from the serve packet."""
    if not flags:
        return 1.0
    f = 1.0
    if flags.get("in_sample"):
        f *= 0.85                          # as-of inside the training window: score may be optimistic
    if flags.get("liquidity_pass") is False:
        f *= 0.80                          # thin name: served off-universe
    stale = flags.get("staleness_days")
    if stale is not None:
        if stale > 550:                    # ~18 months: filing nearly lapsed
            f *= 0.85
        elif stale > 400:
            f *= 0.93
    return max(0.5, f)


def _coherence(drivers) -> float:
    """|Σ contribution| / Σ|contribution| over the SHAP drivers: 1 when they all point one
    way, ~0 when big +/- contributions cancel. Neutral (1.0) when there are no drivers."""
    if not drivers:
        return 1.0
    contribs = [float(d.get("contribution", 0.0)) for d in drivers]
    denom = sum(abs(c) for c in contribs)
    if denom <= 0:
        return 1.0
    return abs(sum(contribs)) / denom


def _coherence_factor(drivers) -> float:
    return 0.80 + 0.20 * _coherence(drivers)


# --- calibration artifact I/O ---------------------------------------------------

def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    """Load the frozen per-decile OOS hit-rate table. Raises if none has been built."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no confidence calibration at {path}; run "
            f"scripts/build_confidence_calibration.py first"
        )
    return json.loads(path.read_text())


def load_calibration_optional(path: Path = CALIBRATION_PATH) -> dict | None:
    """Calibration as an OPTIONAL layer: the table if built, else ``None`` (serve/TUI/web
    run unchanged without it, exactly like the distress head)."""
    try:
        return load_calibration(path)
    except (FileNotFoundError, ValueError):
        return None


# --- the score ------------------------------------------------------------------

def score_confidence(
    decile: int | None,
    percentile: float | None,
    drivers,
    flags: dict | None,
    calibration: dict | None,
) -> dict | None:
    """Derive a 0-100 confidence for one name's call. ``None`` when it cannot be grounded.

    The number is anchored on the decile's directional edge (|hit_rate - 0.5|) from the
    calibration table, then bounded by margin / data-quality / coherence and capped at
    ``CEILING``. The raw ``hit_rate`` + sample size ride along so a caller can always show
    the track record the number is built from.
    """
    if not calibration or decile is None:
        return None
    stats = (calibration.get("deciles") or {}).get(str(int(decile)))
    if not stats or stats.get("hit_rate") is None:
        return None

    hit_rate = float(stats["hit_rate"])
    # Directional, not absolute: a BUY-side decile is convincing only when names in
    # that decile beat the cross-section more than half the time; an AVOID-side
    # decile is convincing only when they beat it less than half the time. HOLD
    # deciles deliberately earn no directional confidence — the model has no strong
    # call there. Using ``abs(hit_rate - 0.5)`` would falsely award confidence to a
    # top-decile BUY bucket whose hit-rate is below 50%.
    d = int(decile)
    if d >= 8:
        edge = max(0.0, hit_rate - 0.5)
    elif d <= 4:
        edge = max(0.0, 0.5 - hit_rate)
    else:
        edge = 0.0
    conviction_base = max(0.0, min(1.0, edge / EDGE_FULL))
    margin = _margin_factor(percentile)
    data_quality = _data_quality_factor(flags)
    coherence = _coherence_factor(drivers)

    raw = 100.0 * conviction_base * margin * data_quality * coherence
    score = int(round(min(raw, float(CEILING))))

    return {
        "score": score,
        "hit_rate": round(hit_rate, 4),
        "n": int(stats.get("n", 0)),
        "ci": [stats.get("ci_low"), stats.get("ci_high")],
        "decile": d,
        "components": {
            "edge": round(edge, 4),
            "conviction_base": round(conviction_base, 4),
            "margin": round(margin, 4),
            "data_quality": round(data_quality, 4),
            "coherence": round(coherence, 4),
        },
    }
