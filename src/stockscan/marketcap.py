"""Live market capitalization (Intrinio data point) — company size for the markets view.

A LIVE-view convenience only, firewalled exactly like ``profile`` / ``quote``: it
sizes companies on the markets overview and NEVER feeds the model or the backtest
(the signal is a quarterly fundamental one). Point-in-time market cap would need
shares-outstanding ingested into the panel — this is deliberately the *current*
number, for display sizing, nothing more.

Fetched BY CIK (recycle-proof, see ``profile``) via Intrinio's ``marketcap`` data
point and cached with an hours-long TTL, because the cap moves every trading day
but a screen doesn't need it fresher than daily.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import httpx

from .config import INTRINIO_API_KEY, MARKETCAP_DB_PATH, MARKETCAP_REFETCH_HOURS
from .prices import intrinio_get_json

_BASE = "https://api-v2.intrinio.com"

_SCHEMA = """
create table if not exists market_caps (
    cik integer primary key,
    fetched_at text not null,   -- when WE fetched it (ISO, UTC)
    mktcap real                 -- USD; null = fetched but no value available
);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fetch_market_cap(cik: int, api_key: str | None = None,
                     client: httpx.Client | None = None) -> float | None:
    """Current market cap (USD) for a CIK from Intrinio. None on missing key/data/error."""
    api_key = api_key or INTRINIO_API_KEY
    if not api_key or not cik:
        return None
    own = client is None
    client = client or httpx.Client(base_url=_BASE, timeout=20.0)
    try:
        try:
            d = intrinio_get_json(
                client, f"/companies/{int(cik):010d}/data_point/marketcap/number",
                {"api_key": api_key})
        except Exception:
            return None
        try:
            return float(d) if d is not None else None
        except (TypeError, ValueError):
            return None
    finally:
        if own:
            client.close()


# -- cache (small WAL SQLite store; short-lived connections, thread-safe) ----------

def _connect() -> sqlite3.Connection:
    MARKETCAP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(MARKETCAP_DB_PATH), timeout=30.0)
    db.execute("pragma journal_mode=wal")
    db.executescript(_SCHEMA)
    return db


def _cache_get(cik: int) -> tuple[float | None, datetime] | None:
    """(mktcap, fetched_at) for a CIK, or None if never cached / row unreadable."""
    db = _connect()
    try:
        row = db.execute("select mktcap, fetched_at from market_caps where cik = ?",
                         (int(cik),)).fetchone()
    finally:
        db.close()
    if not row:
        return None
    try:
        return row[0], datetime.fromisoformat(row[1])
    except (ValueError, TypeError):
        return None


def _cache_put(cik: int, mktcap: float | None) -> None:
    db = _connect()
    try:
        db.execute(
            "insert or replace into market_caps (cik, fetched_at, mktcap) values (?, ?, ?)",
            (int(cik), _utcnow().isoformat(timespec="seconds"), mktcap),
        )
        db.commit()
    finally:
        db.close()


def get_market_cap(cik: int, refresh: bool = False,
                   api_key: str | None = None,
                   client: httpx.Client | None = None) -> float | None:
    """Cached market cap for a CIK. Fetches on a miss / stale row, else serves local.

    A fresh cached row (even a cached ``None`` 'no value') is honored so a screen of
    ~70 names doesn't re-hit the network within a day. On a fetch failure a stale
    cached value is served rather than nothing.
    """
    cached = None if refresh else _cache_get(int(cik))
    if cached is not None:
        cap, fetched_at = cached
        if _utcnow() - fetched_at < timedelta(hours=MARKETCAP_REFETCH_HOURS):
            return cap
    fresh = fetch_market_cap(int(cik), api_key=api_key, client=client)
    if fresh is not None:
        _cache_put(int(cik), fresh)
        return fresh
    if cached is not None:
        return cached[0]
    _cache_put(int(cik), None)  # remember the miss so we don't hammer it all session
    return None
