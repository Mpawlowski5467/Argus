"""The events feed: recent SEC EDGAR filings for a company.

For a fundamentals-driven scanner the most honest, point-in-time "news" is the
company's own filing stream — an 8-K *is* a material-event press release, a 10-K/Q
is the numbers landing. It is free, timestamped, survivorship-safe, and already the
spine of this project (the monitor watches the same submissions feed). A paid
headline/press feed can layer on top later; this needs no new provider.

``shape_filings`` (pure, testable) turns EDGAR's parallel-array ``recent`` block into
labelled rows; ``recent_filings`` does the throttled fetch.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from .config import INTRINIO_API_KEY
from .edgar.client import EdgarClient
from .prices import intrinio_get_json

_INTRINIO_BASE = "https://api-v2.intrinio.com"

FORM_LABELS = {
    "8-K": "material event",
    "8-K/A": "material event (amended)",
    "10-K": "annual report",
    "10-K/A": "annual report (amended)",
    "10-Q": "quarterly report",
    "10-Q/A": "quarterly report (amended)",
    "DEF 14A": "proxy statement",
    "DEFA14A": "proxy solicitation",
    "SC 13D": "activist stake (13D)",
    "SC 13D/A": "activist stake (amended)",
    "SC 13G": "passive stake (13G)",
    "SC 13G/A": "passive stake (amended)",
    "425": "merger communication",
    "S-1": "securities registration",
    "S-1/A": "securities registration (amended)",
    "25-NSE": "delisting notice",
    "15-12B": "deregistration",
    "15-12G": "deregistration",
}

# Newsworthy forms — the material-event + periodic + ownership/M&A set. Deliberately
# excludes the high-frequency low-signal noise (Form 3/4/5 insider filings) so the
# panel reads like a headline stream, not a filing dump.
NEWSWORTHY = frozenset({
    "8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A", "DEF 14A", "DEFA14A",
    "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A", "425", "S-1", "S-1/A",
    "25-NSE", "15-12B", "15-12G",
})


def shape_filings(recent: dict, limit: int = 8, forms=NEWSWORTHY) -> list[dict]:
    """EDGAR's ``filings.recent`` (parallel arrays) → newest-first labelled rows."""
    keep = set(forms) if forms else None
    rows = []
    for form, filed, period in zip(
        recent.get("form", []), recent.get("filingDate", []), recent.get("reportDate", [])
    ):
        if keep is not None and form not in keep:
            continue
        rows.append({
            "form": form,
            "filed_date": filed or "",
            "period_end": period or "",
            "label": FORM_LABELS.get(form, form),
        })
    rows.sort(key=lambda r: r["filed_date"], reverse=True)
    return rows[:limit]


def _source(url: str) -> str:
    """Publisher host from an article URL, e.g. 'www.reuters.com' -> 'reuters.com'."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def shape_article(a: dict) -> dict:
    """One Intrinio news article -> the fields the view + news store need.

    Carries the stable Intrinio ``id`` (the dedup key), the raw ``summary`` (the
    news store's ground truth), and the article's ``cik`` (from the embedded company
    record) alongside the display fields. Full-text is never fetched — headline +
    summary only, by decision."""
    url = a.get("url") or ""
    cik_raw = ((a.get("company") or {}).get("cik") or "").lstrip("0")
    return {
        "id": str(a.get("id") or "").strip(),
        "title": " ".join((a.get("title") or "").split()),
        "summary": " ".join((a.get("summary") or "").split()),
        "date": (a.get("publication_date") or "")[:10],
        "publication_date": a.get("publication_date") or "",
        "url": url,
        "source": _source(url) or str(a.get("source") or ""),
        "cik": int(cik_raw) if cik_raw.isdigit() else None,
    }


def company_news(ticker: str, limit: int = 6, api_key: str | None = None,
                 client: httpx.Client | None = None) -> list[dict]:
    """Recent press headlines for a ticker (Intrinio). [] on missing key or any error.

    Live-view only — never used for scoring/backtest (no point-in-time guarantee)."""
    api_key = api_key or INTRINIO_API_KEY
    if not api_key or not ticker:
        return []
    own = client is None
    client = client or httpx.Client(base_url=_INTRINIO_BASE, timeout=20.0)
    try:
        try:
            d = intrinio_get_json(
                client, f"/companies/{ticker}/news", {"api_key": api_key, "page_size": limit})
        except Exception:
            return []
        arts = (d or {}).get("news", []) or []
        return [shape_article(a) for a in arts[:limit]]
    finally:
        if own:
            client.close()


def company_news_pages(ticker: str, pages: int = 5, page_size: int = 100,
                       api_key: str | None = None,
                       client: httpx.Client | None = None) -> list[dict]:
    """Paginate a ticker's news back in time (Intrinio ``next_page``) to SEED the memory.

    Returns shaped articles across up to ``pages`` pages (deduped by id), oldest pull
    bounded by ``pages * page_size``. Stops early when the feed runs out (no next_page).
    [] on missing key or any error. Live-view only — never used for scoring/backtest."""
    api_key = api_key or INTRINIO_API_KEY
    if not api_key or not ticker:
        return []
    own = client is None
    client = client or httpx.Client(base_url=_INTRINIO_BASE, timeout=20.0)
    out: list[dict] = []
    seen: set[str] = set()
    token: str | None = None
    try:
        for _ in range(max(1, pages)):
            params = {"api_key": api_key, "page_size": page_size}
            if token:
                params["next_page"] = token
            try:
                d = intrinio_get_json(client, f"/companies/{ticker}/news", params)
            except Exception:
                break
            arts = (d or {}).get("news", []) or []
            for a in arts:
                shaped = shape_article(a)
                if shaped["id"] and shaped["id"] not in seen:
                    seen.add(shaped["id"])
                    out.append(shaped)
            token = (d or {}).get("next_page")
            if not arts or not token:
                break
        return out
    finally:
        if own:
            client.close()


def recent_filings(cik: int, limit: int = 8, client: EdgarClient | None = None,
                   forms=NEWSWORTHY) -> list[dict]:
    """Recent newsworthy EDGAR filings for a CIK. [] on any fetch error (never raises)."""
    own = client is None
    client = client or EdgarClient()
    try:
        try:
            data = client.get_json(
                f"{client.DATA_HOST}/submissions/CIK{int(cik):010d}.json")
        except Exception:
            return []
        recent = (data or {}).get("filings", {}).get("recent", {})
        return shape_filings(recent, limit=limit, forms=forms)
    finally:
        if own:
            client.close()
