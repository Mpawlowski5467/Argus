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

from .config import INTRINIO_API_KEY, PARQUET_DIR, PRICE_PROVIDER, TIINGO_TOKEN

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


class IntrinioTransientError(Exception):
    """A retryable failure (network, 5xx, 429/401 exhaustion) — NOT 'no data'.

    Callers must not treat this as an empty/unentitled security: writing partial
    results for it would let a transient outage masquerade as a delisting and be
    pinned forever by the file checkpoint.
    """


def _retry_after_seconds(resp: httpx.Response, attempt: int) -> float:
    try:
        return float(resp.headers.get("retry-after") or 0) or min(60, 5 * 2**attempt)
    except ValueError:  # RFC-7231 HTTP-date form -> just back off
        return min(60, 5 * 2**attempt)


def intrinio_get_json(client: httpx.Client, path: str, params: dict, max_tries: int = 6):
    """GET with 429/5xx/transport backoff.

    Returns parsed JSON on 200, or None on a PERMANENT condition (403 not entitled,
    404 unknown — genuinely no data for this identifier). Transient conditions
    (network errors, 5xx, 429, 401 auth blips) are retried; on exhaustion raises
    IntrinioTransientError so callers can leave the work item to be retried later
    instead of caching a wrong 'no data' answer.
    """
    for attempt in range(max_tries):
        try:
            resp = client.get(path, params=params)
        except httpx.HTTPError:
            time.sleep(min(30, 2**attempt))
            continue
        if resp.status_code in (429, 401) or resp.status_code >= 500:
            time.sleep(_retry_after_seconds(resp, attempt))
            continue
        if resp.status_code != 200:
            return None  # 4xx: permanently unavailable for this identifier
        return resp.json()
    raise IntrinioTransientError(f"retries exhausted for {path}")


def _intrinio_price_rows(client, identifier: str, start, end, api_key: str) -> list[dict] | None:
    """All daily price rows for one security (id or ticker), paged.

    Returns None when the security is permanently unavailable (403/404). Raises
    IntrinioTransientError if any page fails transiently — a truncated series
    would masquerade as a delisting, so partial data is refused.
    """
    rows: list[dict] = []
    next_page = None
    while True:
        params = {"api_key": api_key, "start_date": str(start),
                  "frequency": "daily", "page_size": 10000}
        if end:
            params["end_date"] = str(end)
        if next_page:
            params["next_page"] = next_page
        data = intrinio_get_json(client, f"/securities/{identifier}/prices", params)
        if data is None:
            return None
        rows.extend(data.get("stock_prices", []))
        next_page = data.get("next_page")
        if not next_page:
            return rows


def _intrinio_to_tidy(rows: list[dict], ticker: str) -> pd.DataFrame:
    """Normalize Intrinio /securities/{id}/prices rows to the shared tidy schema (adjusted)."""
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", *_COLS])
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "date": pd.to_datetime(df["date"]).dt.normalize(),
            "open": df.get("adj_open"),
            "high": df.get("adj_high"),
            "low": df.get("adj_low"),
            "close": df.get("adj_close"),
            "volume": df.get("adj_volume"),
        }
    )
    return out.dropna(subset=["close"]).reset_index(drop=True)


def download_prices_intrinio(
    tickers,
    start,
    end=None,
    api_key: str = "",
    pause: float = 0.0,
    skip_cached: bool = True,
    out_dir=PRICES_DIR,
) -> list[str]:
    """Download daily adjusted prices from Intrinio (delisted-inclusive), one Parquet per ticker.

    Intrinio retains delisted/inactive securities in its security master, so names yfinance
    drops are covered. Same on-disk schema as the other adapters, so nothing downstream
    changes. Requires an Intrinio key (STOCKSCAN_INTRINIO_KEY or ``api_key``); handles paging.
    """
    api_key = api_key or INTRINIO_API_KEY
    if not api_key:
        raise ValueError("Intrinio key missing; set STOCKSCAN_INTRINIO_KEY or pass api_key=")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with httpx.Client(base_url="https://api-v2.intrinio.com", timeout=60.0) as client:
        for raw in tickers:
            t = raw.upper()
            if skip_cached and (out_dir / f"{t}.parquet").exists():
                continue
            try:
                rows = _intrinio_price_rows(client, t, start, end, api_key)
            except IntrinioTransientError:
                continue  # nothing cached -> a later run retries this ticker
            tidy = _intrinio_to_tidy(rows or [], t)
            if not tidy.empty:
                tidy.to_parquet(out_dir / f"{t}.parquet", index=False)
                written.append(t)
            if pause:
                time.sleep(pause)
    return written


def _fetch_universe_column(client, col, g, start, end, api_key, pause, out_dir) -> str:
    """Fetch+splice one company's candidates, clip, write (atomically).

    Returns 'written' | 'unavailable' (permanently no data) | 'failed' (transient —
    no file is written, so the resume checkpoint retries it; writing a partial
    splice here would permanently cache a truncated series).
    """
    import os

    frames = []
    try:
        for prio, sec_id in zip(g["priority"], g["security_id"]):
            rows = _intrinio_price_rows(client, sec_id, start, end, api_key)
            if pause:
                time.sleep(pause)
            if not rows:  # None (403/404) or [] -> this candidate has no data
                continue
            tidy = _intrinio_to_tidy(rows, col)
            tidy["ticker"] = col
            tidy["_prio"] = prio
            frames.append(tidy)
    except IntrinioTransientError:
        return "failed"
    if not frames:
        return "unavailable"
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "_prio"]).drop_duplicates("date", keep="first")
    df = df.drop(columns="_prio").sort_values("date").reset_index(drop=True)
    clip = g["clip_date"].dropna()
    if len(clip):
        df = df[df["date"] <= clip.min()]
    if df.empty:
        return "unavailable"
    tmp = out_dir / f".{col}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_dir / f"{col}.parquet")  # atomic: never a half-written checkpoint
    return "written"


def download_intrinio_universe(
    universe: "pd.DataFrame",
    start,
    end=None,
    api_key: str = "",
    pause: float = 0.25,
    skip_cached: bool = True,
    out_dir=PRICES_DIR,
    log_every: int = 200,
    workers: int = 1,
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    """Fetch the survivorship-free universe: one spliced Parquet per company column.

    ``universe`` rows (from stockscan.intrinio_universe) carry ``column, security_id,
    priority, clip_date``. Every request goes BY SECURITY ID so a recycled ticker can
    never inject another company's prices. Candidates for one column are spliced by
    date (lowest priority number wins on overlap); ledger-dead names are clipped at
    ``clip_date``. Files are the checkpoint: existing columns are skipped, so the job
    is resumable. ``workers`` threads fetch concurrently (each throttled by ``pause``;
    429s back off in intrinio_get_json).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = api_key or INTRINIO_API_KEY
    if not api_key:
        raise ValueError("Intrinio key missing; set STOCKSCAN_INTRINIO_KEY or pass api_key=")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    unavailable = failed = 0
    groups = [
        (col, g.sort_values("priority"))
        for col, g in universe.groupby("column", sort=True)
        if not (skip_cached and (out_dir / f"{col}.parquet").exists())
    ]
    with httpx.Client(base_url="https://api-v2.intrinio.com", timeout=60.0,
                      transport=transport) as client:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {
                pool.submit(_fetch_universe_column, client, col, g, start, end,
                            api_key, pause, out_dir): col
                for col, g in groups
            }
            for i, fut in enumerate(as_completed(futs)):
                try:
                    status = fut.result()
                except Exception as e:  # one bad column must not strand the whole run
                    print(f"column {futs[fut]} raised {type(e).__name__}: {e}", flush=True)
                    status = "failed"
                if status == "written":
                    written.append(futs[fut])
                elif status == "unavailable":
                    unavailable += 1
                else:
                    failed += 1
                if log_every and (i + 1) % log_every == 0:
                    print(f"[{i + 1}/{len(groups)}] written={len(written)} "
                          f"unavailable={unavailable} failed={failed} last={futs[fut]}",
                          flush=True)
    print(f"done: {len(written)} columns written, {unavailable} unavailable, "
          f"{failed} FAILED (transient; re-run to retry) of {len(groups)} companies",
          flush=True)
    return written


def download_prices(tickers, start, end=None, provider: str | None = None, **kwargs) -> list[str]:
    """Dispatch to the configured price provider (yfinance | tiingo | intrinio)."""
    provider = (provider or PRICE_PROVIDER).lower()
    if provider == "tiingo":
        return download_prices_tiingo(tickers, start, end, **kwargs)
    if provider == "intrinio":
        return download_prices_intrinio(tickers, start, end, **kwargs)
    return download_prices_yfinance(tickers, start, end, **kwargs)
