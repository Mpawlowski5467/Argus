"""Company profile (Intrinio /companies) — what a company does + where it's based.

A LIVE-view convenience only: qualitative metadata (business description, HQ
location, sector/industry, headcount). It NEVER feeds the model or the backtest —
the signal is a quarterly fundamental one, computed on end-of-day data. Firewalled
exactly like the news layer: display-only, never a feature, never point-in-time
joined into the panel.

Fetched BY CIK, not by ticker. The company endpoint takes a 10-digit zero-padded
CIK (``/companies/0000320193``), which is the one identifier that is a company
forever — tickers are recycled aggressively (see intrinio_universe), so a
by-ticker profile lookup for a delisted name could silently return a different,
currently-listed company. Results are cached in a small SQLite store so a screen
never re-burns quota on a profile that changes maybe once a year.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx

from .config import INTRINIO_API_KEY, PROFILE_DB_PATH, PROFILE_REFETCH_DAYS
from .prices import intrinio_get_json

_BASE = "https://api-v2.intrinio.com"

_SCHEMA = """
create table if not exists profiles (
    cik integer primary key,
    fetched_at text not null,   -- when WE fetched it (ISO, UTC)
    name text,
    data text not null          -- JSON: the normalized display dict
);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_profile(raw: dict) -> dict:
    """Map an Intrinio /companies record to a compact display dict (pure, no I/O).

    Keeps only the fields the ticker view shows. ``description`` prefers the
    one-sentence ``short_description`` over the paragraph-length long one; missing
    fields come back as ``None`` so the renderer can drop them cleanly.
    """
    def s(key: str) -> str | None:
        v = raw.get(key)
        v = str(v).strip() if v is not None else ""
        return v or None

    employees = raw.get("employees")
    try:
        employees = int(employees) if employees not in (None, "") else None
    except (TypeError, ValueError):
        employees = None

    # Intrinio hands back some cities all-caps ("AUSTIN"); title-case those so they
    # read consistently next to already-mixed-case names ("Cupertino").
    city = s("hq_address_city")
    if city and city.isupper():
        city = city.title()

    return {
        "cik": raw.get("cik"),
        "name": s("name"),
        "legal_name": s("legal_name"),
        "description": s("short_description") or s("long_description"),
        "description_long": s("long_description"),   # fuller text for theme tagging
        "sector": s("sector"),
        "industry": s("industry_category") or s("industry_group"),
        "city": city,
        "state": s("hq_state"),
        "country": s("hq_country"),
        "employees": employees,
        "ceo": s("ceo"),
        "url": s("company_url"),
    }


def fetch_profile(cik: int, api_key: str | None = None,
                  client: httpx.Client | None = None) -> dict | None:
    """Fetch + normalize one company's profile from Intrinio. None on any failure.

    Looks the company up by its 10-digit zero-padded CIK. Returns None on a missing
    key/CIK, a permanent 4xx (no such company / not entitled), or any transport
    error — the caller treats None as 'no profile available', never as an error.
    """
    api_key = api_key or INTRINIO_API_KEY
    if not api_key or not cik:
        return None
    own = client is None
    client = client or httpx.Client(base_url=_BASE, timeout=20.0)
    try:
        try:
            d = intrinio_get_json(client, f"/companies/{int(cik):010d}", {"api_key": api_key})
        except Exception:
            return None
        if not d:
            return None
        return normalize_profile(d)
    finally:
        if own:
            client.close()


# -- cache (small WAL SQLite store; short-lived connections, thread-safe) ----------

def _connect() -> sqlite3.Connection:
    PROFILE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(PROFILE_DB_PATH), timeout=30.0)
    db.execute("pragma journal_mode=wal")
    db.executescript(_SCHEMA)
    return db


def _cache_get(cik: int) -> tuple[dict, datetime] | None:
    """(profile, fetched_at) for a CIK, or None if never cached / row unreadable."""
    db = _connect()
    try:
        row = db.execute("select data, fetched_at from profiles where cik = ?",
                         (int(cik),)).fetchone()
    finally:
        db.close()
    if not row:
        return None
    try:
        return json.loads(row[0]), datetime.fromisoformat(row[1])
    except (ValueError, TypeError):
        return None


def _cache_put(cik: int, prof: dict) -> None:
    db = _connect()
    try:
        db.execute(
            "insert or replace into profiles (cik, fetched_at, name, data) values (?, ?, ?, ?)",
            (int(cik), _utcnow().isoformat(timespec="seconds"),
             prof.get("name"), json.dumps(prof)),
        )
        db.commit()
    finally:
        db.close()


def get_profile(cik: int, refresh: bool = False,
                api_key: str | None = None,
                client: httpx.Client | None = None) -> dict | None:
    """Cached company profile for a CIK. Fetches on a miss / stale row, else local.

    On a fetch failure a stale cached copy (if any) is served rather than nothing —
    a month-old 'what it does' beats an empty block. Returns None only when there is
    neither a fetch nor any cache to fall back on.
    """
    cached = None if refresh else _cache_get(int(cik))
    if cached is not None:
        prof, fetched_at = cached
        if _utcnow() - fetched_at < timedelta(days=PROFILE_REFETCH_DAYS):
            return prof
    fresh = fetch_profile(int(cik), api_key=api_key, client=client)
    if fresh is not None:
        _cache_put(int(cik), fresh)
        return fresh
    return cached[0] if cached is not None else None
