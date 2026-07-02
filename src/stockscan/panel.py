"""Build the cross-sectional evaluation panel from the price matrix.

Phase 0 uses a price-only feature (12-1 momentum) and a price-only label (forward
return), both derived by positional shifts over the shared trading-day index -- so
the panel is inherently point-in-time: a close on date T is known at T, the feature
never peeks forward, and the label deliberately does. Fundamental features and their
filing-date PIT join (see stockscan.pit) arrive in Phase 1.
"""

from __future__ import annotations

import glob

import duckdb
import pandas as pd

from .config import LABEL_HORIZON_DAYS
from .prices import PRICES_DIR


def load_close_matrix(tickers=None, prices_dir=PRICES_DIR) -> pd.DataFrame:
    """Wide matrix of adjusted closes: index = trading date, columns = ticker."""
    files = sorted(glob.glob(str(prices_dir / "*.parquet")))
    if not files:
        return pd.DataFrame()
    src = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "])"
    df = duckdb.query(f"select ticker, date, close from {src}").df()
    if tickers is not None:
        df = df[df["ticker"].isin({t.upper() for t in tickers})]
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="ticker", values="close").sort_index()


def load_matrices(tickers=None, prices_dir=PRICES_DIR):
    """Return (close, dollar_volume) wide matrices for the liquidity filter."""
    files = sorted(glob.glob(str(prices_dir / "*.parquet")))
    if not files:
        return pd.DataFrame(), pd.DataFrame()
    src = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "])"
    df = duckdb.query(f"select ticker, date, close, close*volume as dv from {src}").df()
    if tickers is not None:
        df = df[df["ticker"].isin({t.upper() for t in tickers})]
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dv = df.pivot_table(index="date", columns="ticker", values="dv").sort_index()
    return close, dv


def momentum_12_1(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum: return from ~12 months ago to ~1 month ago (skips the last month)."""
    return close.shift(skip) / close.shift(lookback) - 1.0


def forward_return(close: pd.DataFrame, horizon: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """Forward total return over ``horizon`` trading days (uses future prices -- it's the label)."""
    return close.shift(-horizon) / close - 1.0


def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each month present in ``index`` (monthly rebalance grid)."""
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("M")).last())


def build_panel(
    close: pd.DataFrame,
    feature: pd.DataFrame | None = None,
    horizon: int = LABEL_HORIZON_DAYS,
    min_names: int = 5,
) -> pd.DataFrame:
    """Sample feature + forward label at monthly dates into a long panel.

    Columns: ``date, ticker, feature, label, label_excess`` where ``label_excess`` is
    the forward return minus the cross-sectional mean of that date (market-excess;
    sector-excess bucketing is a Phase-1 refinement).
    """
    if feature is None:
        feature = momentum_12_1(close)
    fwd = forward_return(close, horizon)
    frames = []
    for d in month_end_dates(close.index):
        if d not in feature.index or d not in fwd.index:
            continue
        sub = pd.DataFrame({"feature": feature.loc[d], "label": fwd.loc[d]}).dropna()
        if len(sub) < min_names:
            continue
        sub["date"] = d
        sub["ticker"] = sub.index
        frames.append(sub.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "feature", "label", "label_excess"])
    panel = pd.concat(frames, ignore_index=True)
    panel["label_excess"] = panel["label"] - panel.groupby("date")["label"].transform("mean")
    return panel
