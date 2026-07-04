"""Book-level aggregation of the names the user tracks — the "portfolio scorecard".

Pure and DISPLAY-ONLY. It reads the user's tracked names — HELD positions (shares +
cost basis) and WATCHLIST-only names (followed, no shares) — and joins them onto the
ALREADY-SCORED cross-section (percentile, decile, distress flag) to summarise the
book. It writes nothing and feeds nothing — never the score, the paper book, the
backtest, or a trade. It lives on the live-view side of the firewall (registered in
``stockscan.assist.audit.FORBIDDEN``), so the signal core can never import it.

Honesty — the project's inviolable rule — the model signal is ONE monthly
cross-sectional PEER RANK, not a return forecast. So every aggregate here is a
same-day peer-rank snapshot, and the scorecard always ships the full per-holding
list plus BOTH weightings (equal-weight and value-weight) so a single collapsed
number can never masquerade as a portfolio outlook.

Weighting note: for a personal book the meaningful weight is your POSITION VALUE
(shares × latest price), i.e. where your money actually sits — not the company's
market cap (which measures company size, not your exposure). Value weighting needs
a price map; without it we fall back to equal weight and say so.

These are pure functions over plain dicts / a scored DataFrame, so they unit-test
without a data store or the web app.
"""

from __future__ import annotations

import math

import pandas as pd

from .sector import sic_industry

# A holding present in the book but absent from the scored cross-section (dead,
# stale filer, below the liquidity floor). Mirrors the watch view's wording so the
# book never silently drops a name — it's shown, flagged, and kept out of the ranks.
UNLISTED = "not in liquid universe / lapsed filer"


def _decile(pct: float) -> int:
    """Decile 1..10 from a 0..1 rank percentile (matches tui.data._decile)."""
    return int(min(10, max(1, math.ceil(pct * 10))))


def holdings_join(
    positions: list[dict],
    cross: pd.DataFrame,
    prices: dict[int, float] | None = None,
) -> list[dict]:
    """Join the user's tracked names onto the scored cross-section — one row each.

    ``positions``  : ``[{cik, shares, cost_basis, added_at}]``. ``shares``/``cost_basis``
                     may be ``None`` for a WATCHLIST-only name (followed, not held) —
                     it still gets model standing, just no value / P&L. ``owned`` marks
                     the ones with shares.
    ``cross``      : the scored cross-section (``cik, ticker, name, sector, sic, pct``;
                     ``dprob``/``dflag`` present only when the distress artifact is loaded).
    ``prices``     : optional ``{cik: latest_price}`` for value & P/L (live-view close).

    A name absent from the cross is returned with ``in_universe=False`` and
    ``status=UNLISTED`` — kept, never dropped. Order follows ``positions``.
    """
    prices = prices or {}
    by_cik = cross.set_index(cross["cik"].astype(int)) if len(cross) else None
    rows: list[dict] = []
    for p in positions:
        cik = int(p["cik"])
        shares = float(p["shares"]) if p.get("shares") is not None else None
        cost_basis = float(p["cost_basis"]) if p.get("cost_basis") is not None else None
        owned = shares is not None and shares > 0
        price = prices.get(cik)
        value = shares * price if (owned and price is not None) else None
        cost = shares * cost_basis if (owned and cost_basis is not None) else None
        row = {
            "cik": cik,
            "owned": owned,
            "shares": shares,
            "cost_basis": cost_basis,
            "added_at": p.get("added_at"),
            "price": price,
            "value": value,
            "cost": cost,
            "unrealized_pl": (value - cost) if (value is not None and cost is not None) else None,
            "unrealized_pl_pct": ((value / cost - 1.0) * 100.0)
            if (value is not None and cost) else None,
        }
        in_universe = by_cik is not None and cik in by_cik.index
        if in_universe:
            r = by_cik.loc[cik]
            pct01 = float(r["pct"])
            dflag = r.get("dflag") if "dflag" in cross.columns else None
            dprob = (float(r["dprob"]) if "dprob" in cross.columns
                     and pd.notna(r.get("dprob")) else None)
            row.update({
                "in_universe": True,
                "status": "listed",
                "ticker": str(r.get("ticker") or "—"),
                "name": str(r.get("name") or ""),
                "sector": str(r.get("sector") or "—"),
                "industry": sic_industry(r.get("sic")),
                "pct": int(round(pct01 * 100)),
                "decile": _decile(pct01),
                "dflag": dflag,
                "dprob": dprob,
            })
        else:
            row.update({
                "in_universe": False,
                "status": UNLISTED,
                "ticker": "—",
                "name": "",
                "sector": "—",
                "industry": "Unknown",
                "pct": None,
                "decile": None,
                "dflag": None,
                "dprob": None,
            })
        rows.append(row)
    return rows


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    """Weight-mean of (value, weight); None if no positive weight."""
    tot = sum(w for _, w in pairs if w and w > 0)
    if tot <= 0:
        return None
    return sum(v * w for v, w in pairs if w and w > 0) / tot


def _concentration(listed: list[dict], key: str) -> list[dict]:
    """Group listed holdings by ``key`` (e.g. 'industry' / 'sector') with count- and
    value-based weights. Value weight only spans holdings that have a value; if none
    do it is None so the display doesn't imply a $ split it can't compute."""
    total_value = sum(h["value"] for h in listed if h["value"] is not None)
    buckets: dict[str, dict] = {}
    for h in listed:
        b = buckets.setdefault(h[key], {"name": h[key], "count": 0, "value": 0.0,
                                        "has_value": False})
        b["count"] += 1
        if h["value"] is not None:
            b["value"] += h["value"]
            b["has_value"] = True
    n = len(listed)
    out = []
    for b in buckets.values():
        out.append({
            "name": b["name"],
            "count": b["count"],
            "weight_count": (b["count"] / n) if n else None,
            "weight_value": (b["value"] / total_value)
            if (total_value > 0 and b["has_value"]) else None,
        })
    # biggest exposure first — by value when we have it, else by count
    out.sort(key=lambda x: (x["weight_value"] if x["weight_value"] is not None
                            else x["weight_count"] or 0.0), reverse=True)
    return out


def scorecard(
    positions: list[dict],
    cross: pd.DataFrame,
    prices: dict[int, float] | None = None,
    as_of=None,
) -> dict:
    """Book-level scorecard: aggregate the names the user tracks into a peer-rank snapshot.

    ``positions`` covers HELD names (with shares) AND WATCHLIST-only names (no shares) —
    the latter contribute model standing / distress / concentration but no value / P&L.
    Returns the full ``holdings`` list (never hidden) plus aggregates: counts, equal- and
    value-weighted model percentile, distress exposure, and industry/sector concentration.
    Every aggregate is a same-day peer-rank snapshot ``as_of`` — not a return forecast.
    An empty book returns zeros and empty lists.
    """
    holdings = holdings_join(positions, cross, prices)
    listed = [h for h in holdings if h["in_universe"]]
    unlisted = [h for h in holdings if not h["in_universe"]]
    owned = [h for h in holdings if h["owned"]]

    valued = [h for h in owned if h["value"] is not None]
    total_value = sum(h["value"] for h in valued) if valued else None
    costed = [h for h in valued if h["cost"] is not None]
    total_cost = sum(h["cost"] for h in costed) if costed else None
    pl = [h for h in holdings if h["unrealized_pl"] is not None]
    unrealized_pl = sum(h["unrealized_pl"] for h in pl) if pl else None

    # model standing — equal-weight over every tracked name in the universe; value-weight
    # only over the HELD ones that have a price. Both shown so neither stands in alone.
    pct_equal = (round(sum(h["pct"] for h in listed) / len(listed), 1)
                 if listed else None)
    valued_listed = [h for h in listed if h["value"] is not None]
    pct_value = _weighted_mean([(h["pct"], h["value"]) for h in valued_listed])
    if pct_value is not None:
        pct_value = round(pct_value, 1)

    # concentration reflects your actual money: HELD names in the universe. With
    # nothing held yet, fall back to all tracked names (count-weighted) so a pure
    # watchlist still shows what you follow.
    held_listed = [h for h in owned if h["in_universe"]]
    conc_set = held_listed if held_listed else listed

    # distress exposure — counts by flag, plus the book value sitting in each flag
    flags = ("high", "elevated", "normal")
    distress_count = {f: 0 for f in flags}
    distress_value = {f: 0.0 for f in flags}
    distress_known = False
    for h in listed:
        f = h["dflag"]
        if f in distress_count:
            distress_known = True
            distress_count[f] += 1
            if h["value"] is not None:
                distress_value[f] += h["value"]

    return {
        "as_of": str(pd.Timestamp(as_of).date()) if as_of is not None else None,
        "n_total": len(holdings),
        "n_owned": len(owned),
        "n_watch": len(holdings) - len(owned),
        "n_listed": len(listed),
        "n_unlisted": len(unlisted),
        "total_value": total_value,
        "total_cost": total_cost,
        "unrealized_pl": unrealized_pl,
        "unrealized_pl_pct": ((total_value / total_cost - 1.0) * 100.0)
        if (total_value is not None and total_cost) else None,
        "percentile_equal": pct_equal,
        "percentile_value": pct_value,
        "distress": {
            "known": distress_known,
            "count": distress_count,
            "value": distress_value if distress_known else None,
            "at_risk": distress_count["high"] + distress_count["elevated"],
        },
        "industry_concentration": _concentration(conc_set, "industry"),
        "sector_concentration": _concentration(conc_set, "sector"),
        "holdings": holdings,
    }
