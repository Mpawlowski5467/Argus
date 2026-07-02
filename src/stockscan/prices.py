"""Daily price ingest via yfinance -> per-ticker Parquet (long format).

Stooq's free bulk/CSV access is gated behind an anti-bot proof-of-work as of
mid-2026, so yfinance (unofficial Yahoo endpoints, no API key) is the pragmatic
free price source. It is survivorship-biased (delisted tickers mostly return
empty) and occasionally wrong on corporate actions; we cache every ticker to
Parquet and keep the fetch behind this small interface so a cleaner (paid) feed
can replace it later without touching the rest of the pipeline. See DESIGN.md.

Prices are stored auto-adjusted (splits + dividends) — one Parquet per ticker:
columns ``ticker, date, open, high, low, close, volume``.
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

from .config import PARQUET_DIR, PRICE_PROVIDER, TIINGO_TOKEN

PRICES_DIR = PARQUET_DIR / "prices"
_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
_COLS = ["open", "high", "low", "close", "volume"]


def _tidy(sub: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single ticker's yfinance frame to long tidy rows (drops NaN closes)."""
    sub = sub.rename(columns=_RENAME)
    keep = [c for c in _COLS if c in sub.columns]
    sub = sub[keep].reset_index()
    sub = sub.rename(columns={sub.columns[0]: "date"})
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    if "close" in sub.columns:
        sub = sub.dropna(subset=["close"])
    return sub.reset_index(drop=True)


def _iter_long(df: pd.DataFrame, tickers: list[str]):
    """Yield ``(ticker, tidy_df)`` from a yf.download frame (grouped by ticker)."""
    if isinstance(df.columns, pd.MultiIndex):
        for tk in dict.fromkeys(df.columns.get_level_values(0)):
            yield tk, _tidy(df[tk])
    else:  # single ticker -> flat columns
        yield tickers[0], _tidy(df)


def download_prices_yfinance(
    tickers,
    start,
    end=None,
    batch_size: int = 40,
    pause: float = 1.0,
    skip_cached: bool = True,
    out_dir=PRICES_DIR,
) -> list[str]:
    """Download daily auto-adjusted prices from yfinance, one Parquet per ticker.

    Returns the list of tickers actually written this call. Already-cached tickers
    are skipped unless ``skip_cached=False``. (Free but survivorship-biased.)
    """
    import yfinance as yf

    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = [t.upper() for t in tickers]
    todo = [t for t in tickers if not (skip_cached and (out_dir / f"{t}.parquet").exists())]
    written: list[str] = []
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        df = yf.download(
            batch, start=start, end=end, auto_adjust=True,
            progress=False, threads=False, group_by="ticker",
        )
        if df is None or len(df) == 0:
            continue
        for tk, tidy in _iter_long(df, batch):
            if tidy is None or tidy.empty:
                continue
            tidy.insert(0, "ticker", tk)
            tidy.to_parquet(out_dir / f"{tk}.parquet", index=False)
            written.append(tk)
        if i + batch_size < len(todo):
            time.sleep(pause)
    return written


def _tiingo_to_tidy(rows: list[dict], ticker: str) -> pd.DataFrame:
    """Normalize Tiingo daily-price JSON to the shared tidy schema (split/div-adjusted)."""
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", *_COLS])
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "date": pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize(),
            "open": df.get("adjOpen"),
            "high": df.get("adjHigh"),
            "low": df.get("adjLow"),
            "close": df.get("adjClose"),
            "volume": df.get("adjVolume"),
        }
    )
    return out.dropna(subset=["close"]).reset_index(drop=True)


def download_prices_tiingo(
    tickers,
    start,
    end=None,
    token: str = "",
    pause: float = 0.0,
    skip_cached: bool = True,
    out_dir=PRICES_DIR,
) -> list[str]:
    """Download daily adjusted prices from Tiingo (delisted-inclusive), one Parquet per ticker.

    Delisted names are retained via Tiingo's permaTicker, so symbols yfinance drops are
    kept. Writes the SAME on-disk schema as the yfinance adapter, so nothing downstream
    changes. Requires a Tiingo token (STOCKSCAN_TIINGO_TOKEN or the ``token`` arg).
    """
    token = token or TIINGO_TOKEN
    if not token:
        raise ValueError("Tiingo token missing; set STOCKSCAN_TIINGO_TOKEN or pass token=")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with httpx.Client(
        base_url="https://api.tiingo.com",
        headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
        timeout=30.0,
    ) as client:
        for raw in tickers:
            t = raw.upper()
            if skip_cached and (out_dir / f"{t}.parquet").exists():
                continue
            params = {"startDate": str(start), "format": "json"}
            if end:
                params["endDate"] = str(end)
            try:
                resp = client.get(f"/tiingo/daily/{t.lower()}/prices", params=params)
            except httpx.HTTPError:
                continue
            if resp.status_code != 200:  # unknown/unavailable symbol -> skip
                continue
            tidy = _tiingo_to_tidy(resp.json(), t)
            if not tidy.empty:
                tidy.to_parquet(out_dir / f"{t}.parquet", index=False)
                written.append(t)
            if pause:
                time.sleep(pause)
    return written


def download_prices(tickers, start, end=None, provider: str | None = None, **kwargs) -> list[str]:
    """Dispatch to the configured price provider (STOCKSCAN_PRICE_PROVIDER: yfinance|tiingo)."""
    provider = (provider or PRICE_PROVIDER).lower()
    if provider == "tiingo":
        return download_prices_tiingo(tickers, start, end, **kwargs)
    return download_prices_yfinance(tickers, start, end, **kwargs)
