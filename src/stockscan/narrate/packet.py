"""Assemble a compact, grounded signal packet for one company (fundamentals only, no prices).

The packet is the COMPUTE -> NARRATE contract: pre-computed numbers with cross-sectional
peer percentiles and year-over-year deltas. The LLM narrates strictly from this; nothing
here is a prediction — it's a fundamental profile + peer screen.
"""

from __future__ import annotations

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
_LABELS = {
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


def _load_features() -> pd.DataFrame:
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    return compute_features(wide)


def build_packet(company, features_df: pd.DataFrame | None = None) -> dict:
    """Build the signal packet for a ticker or CIK."""
    feats = (features_df if features_df is not None else _load_features()).copy()
    feats["period_end"] = pd.to_datetime(feats["period_end"])
    cik = company if isinstance(company, int) else cik_for(company)
    if cik is None:
        raise ValueError(f"unknown ticker/cik: {company}")

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
    history = feats[feats["cik"] == cik].sort_values("period_end")
    prior = history.iloc[-2] if len(history) >= 2 else None

    signals = []
    for f in FEATURES:
        v = row[f]
        if pd.isna(v):
            continue
        pct = row.get(f"{f}_pct")
        sig = {
            "id": f,
            "label": _LABELS[f],
            "value": round(v * 100, 1) if f in _PCT else round(v, 2),
            "unit": "%" if f in _PCT else "x",
            "pct_rank": int(round(pct)) if pd.notna(pct) else None,
            "direction": "higher-is-better" if FEATURE_SIGN[f] > 0 else "lower-is-better",
        }
        if prior is not None and pd.notna(prior[f]) and f in _PCT:
            sig["yoy_change_pp"] = round((v - prior[f]) * 100, 1)
        signals.append(sig)

    return {
        "meta": {
            "ticker": company if isinstance(company, str) else None,
            "name": row["name"],
            "cik": int(cik),
            "fiscal_year": int(row["fy"]) if pd.notna(row["fy"]) else None,
            "period_end": str(row["period_end"].date()),
            "sector": row["sector"],
        },
        "signals": signals,
        "composite": {
            "label": "Composite quality score (a peer screen, NOT a return prediction)",
            "percentile": int(round(row["_comp_pct"])) if pd.notna(row["_comp_pct"]) else None,
        },
        "disclaimer": "Fundamental analysis / peer screening only; not investment advice.",
    }
