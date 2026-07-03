"""Build the cross-sectional evaluation panel from the price matrix.

Phase 0 uses a price-only feature (12-1 momentum) and a price-only label (forward
return), both derived by positional shifts over the shared trading-day index -- so
the panel is inherently point-in-time: a close on date T is known at T, the feature
never peeks forward, and the label deliberately does. Fundamental features and their
filing-date PIT join (see stockscan.pit) arrive in Phase 1.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from .config import LABEL_HORIZON_DAYS
from .prices import PRICES_DIR


def load_close_matrix(tickers=None, prices_dir=PRICES_DIR) -> pd.DataFrame:
    """Wide matrix of adjusted closes: index = trading date, columns = ticker."""
    files = sorted(glob.glob(str(prices_dir / "*.parquet")))
    if not files:
        return pd.DataFrame()
    src = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "], union_by_name=true)"
    df = duckdb.query(f"select ticker, date, close from {src}").df()
    if tickers is not None:
        df = df[df["ticker"].isin({t.upper() for t in tickers})]
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="ticker", values="close").sort_index()


def load_matrices(tickers=None, prices_dir=PRICES_DIR, with_open: bool = False):
    """Return (close, dollar_volume[, open]) wide matrices.

    ``with_open=True`` adds the adjusted-open matrix (the backtester executes at
    next-bar open, DESIGN.md §6); default stays two-tuple for existing callers.
    union_by_name: files written after the Phase-5 schema addition carry extra
    unadjusted columns (uclose/uvolume) that older files lack.
    """
    files = sorted(glob.glob(str(prices_dir / "*.parquet")))
    if not files:
        return (pd.DataFrame(),) * (3 if with_open else 2)
    src = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "], union_by_name=true)"
    cols = "ticker, date, open, close, close*volume as dv" if with_open else \
           "ticker, date, close, close*volume as dv"
    df = duckdb.query(f"select {cols} from {src}").df()
    if tickers is not None:
        df = df[df["ticker"].isin({t.upper() for t in tickers})]
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dv = df.pivot_table(index="date", columns="ticker", values="dv").sort_index()
    if not with_open:
        return close, dv
    opn = df.pivot_table(index="date", columns="ticker", values="open").sort_index()
    return close, dv, opn


# --- matrix cache (Phase 5) ------------------------------------------------------
# The per-column store is 11k small parquets; pivoting them takes ~2 minutes, which
# is fine for offline research but not for a daily monitor loop. The nightly price
# job persists the two wide matrices; loaders fall back to the slow path whenever
# the cache is missing or older than the newest per-column file (never silently
# stale). Cache content is exactly the load_matrices output — verified by test.

def matrix_cache_paths(cache_dir=None):
    from .config import MATRIX_CACHE_DIR

    d = Path(cache_dir) if cache_dir is not None else MATRIX_CACHE_DIR
    return d / "close.parquet", d / "dv.parquet", d / "meta.json"


def _prices_manifest(prices_dir=PRICES_DIR) -> str:
    """Content fingerprint of the per-column store: sorted (name, size, mtime_ns).

    Filenames are part of the hash because renames (universe refresh: a death
    turns FOO.parquet into FOO~CIK.parquet) preserve mtime — an mtime-only check
    would call the cache fresh while its column set is stale.
    """
    import hashlib

    entries = []
    for f in sorted(glob.glob(str(Path(prices_dir) / "*.parquet"))):
        st = os.stat(f)
        entries.append(f"{os.path.basename(f)}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]


def save_matrix_cache(close: pd.DataFrame, dv: pd.DataFrame, cache_dir=None,
                      prices_dir=PRICES_DIR) -> None:
    close_p, dv_p, meta_p = matrix_cache_paths(cache_dir)
    close_p.parent.mkdir(parents=True, exist_ok=True)
    for frame, path in ((close, close_p), (dv, dv_p)):
        tmp = path.with_name("." + path.name + ".tmp")
        frame.to_parquet(tmp)
        os.replace(tmp, path)
    tmp = meta_p.with_name("." + meta_p.name + ".tmp")
    tmp.write_text(json.dumps({
        "built_at": str(pd.Timestamp.now("UTC")),
        "source_manifest": _prices_manifest(prices_dir),
        "max_date": str(close.index.max().date()) if len(close.index) else None,
        "n_columns": int(close.shape[1]),
    }))
    os.replace(tmp, meta_p)


def matrix_cache_fresh(cache_dir=None, prices_dir=PRICES_DIR) -> bool:
    _, _, meta_p = matrix_cache_paths(cache_dir)
    if not meta_p.exists():
        return False
    try:
        meta = json.loads(meta_p.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("source_manifest") == _prices_manifest(prices_dir)


def load_matrices_cached(prices_dir=PRICES_DIR, cache_dir=None):
    """(close, dv) from the cache when fresh, else the slow path (never stale)."""
    close_p, dv_p, _ = matrix_cache_paths(cache_dir)
    if close_p.exists() and dv_p.exists() and matrix_cache_fresh(cache_dir, prices_dir):
        return pd.read_parquet(close_p), pd.read_parquet(dv_p)
    return load_matrices(prices_dir=prices_dir)


def momentum_12_1(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum: return from ~12 months ago to ~1 month ago (skips the last month)."""
    return close.shift(skip) / close.shift(lookback) - 1.0


def momentum_6_1(close: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """6-1 momentum: return from ~6 months ago to ~1 month ago (skips the last month).

    Same skip-the-last-month construction as :func:`momentum_12_1` (the 1-month gap
    sidesteps short-term reversal); only the lookback shortens to ~6 months. Both use
    only PAST closes relative to each row's date, so the matrix is point-in-time by
    construction — the value at date T is knowable at T.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def forward_return(close: pd.DataFrame, horizon: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """Forward total return over ``horizon`` trading days (uses future prices -- it's the label)."""
    return close.shift(-horizon) / close - 1.0


def forward_return_to_last(close: pd.DataFrame, horizon: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """Forward return that uses the LAST traded price for series ending mid-window.

    Identical to :func:`forward_return` for continuously-traded names. A name whose
    series ends inside the horizon (delisting) gets its real terminal return
    (last trade / entry - 1) instead of NaN -- with delisted-inclusive price data this
    captures the actual death decline, replacing the imputed-haircut convention.
    A mid-window trading halt is likewise labeled with the return to the halt price
    (you could not have traded past it). Uses future prices -- it's the label.

    The fill limit is ``horizon - 1`` so the terminal price must lie STRICTLY inside
    (d, d+horizon]: at limit=horizon the fill source can be close[d] itself, which
    would fabricate an information-free 0.0 label for every name whose last-ever
    trade falls exactly on a sampling date.
    """
    filled = close.ffill(limit=horizon - 1) if horizon > 1 else close
    return filled.shift(-horizon) / close - 1.0


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
