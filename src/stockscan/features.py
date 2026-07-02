"""Fundamental ratio features from the wide concept table.

Fundamentals-only for now (no market-cap ratios like E/P -- those need point-in-time
shares x price, wired separately). Growth features use each company's prior 10-K.
FEATURE_SIGN encodes the expected direction (+1 higher-is-better, -1 higher-is-worse)
for building a composite score; the IC harness itself is sign-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "gross_profitability",
    "roa",
    "op_margin",
    "roe",
    "leverage",
    "current_ratio",
    "accruals",
    "cash_to_assets",
    "asset_growth",
    "revenue_growth",
]

FEATURE_SIGN = {
    "gross_profitability": 1,
    "roa": 1,
    "op_margin": 1,
    "roe": 1,
    "leverage": -1,
    "current_ratio": 1,
    "accruals": -1,       # Sloan: high accruals predict lower returns
    "cash_to_assets": 1,
    "asset_growth": -1,   # asset-growth anomaly: fast growers underperform
    "revenue_growth": 1,
}


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def compute_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Add fundamental ratio + growth columns to the wide per-filing table."""
    df = wide.copy()
    df["period_end"] = pd.to_datetime(df["period_end"])
    df = df.sort_values(["cik", "period_end"])
    prior_assets = df.groupby("cik")["assets"].shift(1)
    prior_revenue = df.groupby("cik")["revenue"].shift(1)

    df["gross_profitability"] = _div(df["gross_profit"], df["assets"])
    df["roa"] = _div(df["net_income"], df["assets"])
    df["op_margin"] = _div(df["operating_income"], df["revenue"])
    df["roe"] = _div(df["net_income"], df["equity"])
    df["leverage"] = _div(df["liabilities"], df["assets"])
    df["current_ratio"] = _div(df["assets_current"], df["liabilities_current"])
    df["accruals"] = _div(df["net_income"] - df["cfo"], df["assets"])
    df["cash_to_assets"] = _div(df["cash"], df["assets"])
    df["asset_growth"] = _div(df["assets"], prior_assets) - 1.0
    df["revenue_growth"] = _div(df["revenue"], prior_revenue) - 1.0

    return df.replace([np.inf, -np.inf], np.nan)
