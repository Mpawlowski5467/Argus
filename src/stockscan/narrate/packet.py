"""Assemble a compact, grounded signal packet for one company (fundamentals only, no prices).

The packet is the COMPUTE -> NARRATE contract: pre-computed numbers with cross-sectional
peer percentiles and year-over-year deltas. The LLM narrates strictly from this; nothing
here is a prediction — it's a fundamental profile + peer screen.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

from ..concepts import WIDE_PATH
from ..edgar.tickers import cik_for
from ..features import FEATURE_SIGN, FEATURES, compute_features
from ..sector import sic_division

# features shown as percentages (x100) vs. plain ratios ("x")
_PCT = {
    "roa", "op_margin", "gross_profitability", "roe", "accruals",
    "asset_growth", "revenue_growth", "cash_to_assets",
}
LABELS = {
    "gross_profitability": "Gross profitability (GP/assets)",
    "roa": "Return on assets",
    "op_margin": "Operating margin",
    "roe": "Return on equity",
    "leverage": "Leverage (liabilities/assets)",
    "current_ratio": "Current ratio",
    "accruals": "Accruals (NI-CFO)/assets",
    "cash_to_assets": "Cash/assets",
    "asset_growth": "Asset growth YoY",
    "revenue_growth": "Revenue growth YoY",
}
_LABELS = LABELS  # backwards-compat alias


def _load_features() -> pd.DataFrame:
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    return compute_features(wide)


# --- news context (LIVE-VIEW + NARRATION ONLY — never scoring/backtest/panel) ------
#
# The packet is downstream of the model; nothing read here can reach the score. News
# rides along ONLY so the narrator may reference themes/past events and cite the
# article. The firewall on FABRICATION is kept honest by keeping takeaways NUMBER-FREE:
# every numeral is stripped, so the grounding guard's numeral domain stays the
# fundamentals packet and a fabricated figure is still caught even with news present.
# A date's YEAR is the only news figure that survives, and only via the ``date`` field.
# The raw article summary (the real numbers) lives in news.sqlite and the TUI, never here.
_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _strip_numbers(text: str) -> str:
    return " ".join(_DIGITS.sub(" ", str(text or "")).split())


def news_context(articles) -> list[dict]:
    """Shape recall()/extraction rows into packet-safe, number-free news takeaways.

    Each entry: ``{id, date, source, event_type, takeaway}``. ``takeaway`` is
    defensively stripped of ALL numerals here (belt-and-suspenders over the
    extraction contract) so no news-derived figure can ever enter ``allowed_numbers``.
    """
    out = []
    for a in articles or []:
        aid = str(a.get("id") or "").strip()
        if not aid:
            continue
        out.append({
            "id": aid,
            "date": (str(a.get("date") or a.get("publication_date") or ""))[:10],
            "source": str(a.get("source") or ""),
            "event_type": str(a.get("event_type") or "other"),
            "takeaway": _strip_numbers(a.get("takeaway") or a.get("title") or "")[:160],
        })
    return out


def build_packet(
    company,
    features_df: pd.DataFrame | None = None,
    snapshot: pd.DataFrame | None = None,
    as_of=None,
    news=None,
) -> dict:
    """Build the signal packet for a ticker or CIK.

    ``snapshot`` (optional): a pre-built one-row-per-company cross-section (with
    ``sector`` + FEATURES) to use as the peer-percentile universe -- the serve path
    passes its point-in-time, liquidity-filtered cross-section here so every
    percentile in the narration refers to the SAME universe the model scored.
    Without it, the universe is the latest filing per company in ``features_df``.
    ``as_of``: recorded in meta; ``features_df`` must already be PIT-filtered by the
    caller when an as-of date is in play (the serve path does this).
    ``news`` (optional): recalled article takeaways (LIVE-VIEW ONLY) attached under
    ``context.news`` for the narrator to reference/cite. Omitted key when empty, so a
    news-free packet is byte-identical to before; NEVER touches signals/scoring.
    """
    feats = (features_df if features_df is not None else _load_features()).copy()
    feats["period_end"] = pd.to_datetime(feats["period_end"])
    cik = company if isinstance(company, int) else cik_for(company)
    if cik is None:
        raise ValueError(f"unknown ticker/cik: {company}")

    if snapshot is not None:
        snap = snapshot.copy()
    else:
        # latest filing per company -> the cross-section for peer percentiles
        snap = feats.sort_values("period_end").drop_duplicates("cik", keep="last").copy()
        snap["sector"] = snap["sic"].map(sic_division)
    for f in FEATURES:
        snap[f"{f}_pct"] = snap.groupby("sector")[f].rank(pct=True) * 100
    comp = sum(FEATURE_SIGN[f] * (snap[f"{f}_pct"].fillna(50) / 100 - 0.5) for f in FEATURES)
    snap["_comp_pct"] = comp.groupby(snap["sector"]).rank(pct=True) * 100

    hit = snap[snap["cik"] == cik]
    if hit.empty:
        raise ValueError(f"no fundamentals for cik {cik}")
    row = hit.iloc[0]
    # Prior = the latest filing for an EARLIER period than the snapshot row's — never
    # positional. iloc[-2] would pair a delinquent re-filing against itself (YoY 0.0)
    # whenever the latest-available filing isn't the latest-period one.
    history = feats[feats["cik"] == cik].sort_values("period_end")
    earlier = history[history["period_end"] < pd.Timestamp(row["period_end"])]
    prior = earlier.iloc[-1] if len(earlier) else None

    signals = []
    for f in FEATURES:
        v = row[f]
        if pd.isna(v):
            continue
        pct = row.get(f"{f}_pct")
        pct_int = int(round(pct)) if pd.notna(pct) else None
        effective = None if pct_int is None else (
            pct_int if FEATURE_SIGN[f] > 0 else 100 - pct_int
        )
        sig = {
            "id": f,
            "label": _LABELS[f],
            "value": round(v * 100, 1) if f in _PCT else round(v, 2),
            "unit": "%" if f in _PCT else "x",
            "pct_rank": pct_int,
            "direction": "higher-is-better" if FEATURE_SIGN[f] > 0 else "lower-is-better",
            # the deterministic verdict — the narrator copies it, never derives it
            # (a 32nd-pct asset growth is a PLUS under the asset-growth anomaly;
            # deriving that from 'lower-is-better' is a known LLM stumble)
            "read": None if effective is None else
                    ("supports" if effective >= 50 else "detracts"),
        }
        if prior is not None and pd.notna(prior[f]) and f in _PCT:
            sig["yoy_change_pp"] = round((v - prior[f]) * 100, 1)
        signals.append(sig)

    meta = {
        "ticker": company if isinstance(company, str) else None,
        "name": row["name"],
        "cik": int(cik),
        "fiscal_year": int(row["fy"]) if pd.notna(row["fy"]) else None,
        "period_end": str(pd.Timestamp(row["period_end"]).date()),
        "sector": row["sector"],
    }
    if as_of is not None:
        meta["as_of"] = str(pd.Timestamp(as_of).date())
    packet = {
        "meta": meta,
        "signals": signals,
        "composite": {
            "label": "Composite quality score (a peer screen, NOT a return prediction)",
            "percentile": int(round(row["_comp_pct"])) if pd.notna(row["_comp_pct"]) else None,
        },
        "disclaimer": "Fundamental analysis / peer screening only; not investment advice.",
    }
    ctx = news_context(news)
    if ctx:
        packet["context"] = {"news": ctx}
    return packet
