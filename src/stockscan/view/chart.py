"""Deterministic price-summary + the BUY/HOLD/AVOID verdict — pure functions the
web UI reads. Both are computed in code (the LLM never sets these).

``price_summary`` (last / trailing change / 52-week range) and ``verdict`` (the
cross-sectional call). No plotting lives here anymore — the browser draws the
charts client-side (static/charts.js).
"""

from __future__ import annotations

import math


def _finite(values) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _pct(new: float, old: float) -> float | None:
    if old in (None, 0) or new is None or not math.isfinite(old) or not math.isfinite(new):
        return None
    return (new / old - 1.0) * 100.0


def price_summary(series, adv: float | None = None) -> dict:
    """Last price, trailing % changes, and the 52-week range from a close Series.

    ``series`` is a pandas Series (date-indexed) or any ordered sequence; NaNs are
    dropped. ~252 trading days ≈ 1 year, 63 ≈ 3 months, 21 ≈ 1 month.
    """
    try:
        vals = _finite(series.tolist())  # pandas Series
    except AttributeError:
        vals = _finite(list(series))
    if not vals:
        return {"last": None, "chg_1m": None, "chg_3m": None, "chg_1y": None,
                "hi_52w": None, "lo_52w": None, "adv": adv, "n": 0}
    last = vals[-1]
    yr = vals[-252:] if len(vals) >= 2 else vals

    def back(n):
        return vals[-n - 1] if len(vals) > n else vals[0]

    return {
        "last": last,
        "chg_1m": _pct(last, back(21)),
        "chg_3m": _pct(last, back(63)),
        "chg_1y": _pct(last, back(252)),
        "hi_52w": max(yr),
        "lo_52w": min(yr),
        "adv": adv,
        "n": len(vals),
    }


# --- the deterministic call -----------------------------------------------------
# Long-only per the Phase-3 verdict (the short book was dropped: it died to borrow).
# Hysteresis from the backtest: enter the book in the top quintile, hold to the 40th
# percentile. We expose a single-snapshot read of that rule (there is no prior-state
# here), plus the raw percentile so nothing is hidden.

def verdict(percentile: float, decile: int | None = None) -> dict:
    """BUY / HOLD / AVOID from the cross-sectional percentile (0..1). Deterministic."""
    if percentile is None or not math.isfinite(percentile):
        return {"call": "N/A", "color": "dim", "reason": "no score"}
    pct = float(percentile)
    if pct >= 0.80:
        return {"call": "BUY", "color": "green",
                "reason": f"top-quintile signal ({round(pct * 100)}th pct) — enters the long book"}
    if pct >= 0.40:
        return {"call": "HOLD", "color": "yellow",
                "reason": f"mid signal ({round(pct * 100)}th pct) — hold if owned, no new buy"}
    return {"call": "AVOID", "color": "red",
            "reason": f"bottom signal ({round(pct * 100)}th pct) — long-only, so no position"}
