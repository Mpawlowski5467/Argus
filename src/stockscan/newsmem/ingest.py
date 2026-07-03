"""Ingest orchestration: fetch → idempotent upsert → extract-missing.

One company at a time (lazy on ticker-open) or a watchlist batch (the nightly job).
Quota discipline: the Intrinio network pull is capped to ``limit`` articles and is
SKIPPED entirely for a company fetched within ``refetch_hours`` (the store's fetch
throttle) — so re-opening a ticker or a re-run nightly job costs no quota. Extraction
still runs for any stored article missing the current version, so a later model/prompt
bump regenerates offline without re-fetching.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from ..config import NEWS_FETCH_LIMIT, NEWS_REFETCH_HOURS
from ..news import _INTRINIO_BASE, company_news, company_news_pages
from .extract import EXTRACT_VERSION, extract_article
from .store import NewsStore


def watchlist_targets(watchlist_rows: list[dict], universe) -> list[tuple]:
    """[(cik, ticker)] for each watched cik, ticker = its top-priority universe row.

    Pure (no I/O) so the nightly job's company resolution is testable without the ops
    DB or a live universe. A watched cik absent from the universe maps to ticker None
    (the ingest then records it as no_ticker rather than fetching)."""
    tmap = ((universe.sort_values("priority").drop_duplicates("cik")
             .set_index("cik")["ticker"].to_dict()) if len(universe) else {})
    return [(int(w["cik"]), tmap.get(int(w["cik"]))) for w in watchlist_rows]


def _recently_fetched(store: NewsStore, cik: int, refetch_hours: float) -> bool:
    last = store.last_fetch(cik)
    if not last or not last.get("last_fetched"):
        return False
    try:
        when = datetime.fromisoformat(last["last_fetched"])
    except ValueError:
        return False
    return datetime.now(timezone.utc) - when < timedelta(hours=refetch_hours)


def ingest_company_news(cik: int, ticker: str | None, store: NewsStore, llm=None,
                        limit: int = NEWS_FETCH_LIMIT, version: str = EXTRACT_VERSION,
                        force: bool = False, refetch_hours: float = NEWS_REFETCH_HOURS,
                        client: httpx.Client | None = None) -> dict:
    """Fetch (quota-capped) + upsert + extract-missing for one company.

    Returns a deltas dict: ``{cik, ticker, fetched, new, extracted, skipped_fetch}``.
    """
    skipped = not force and _recently_fetched(store, cik, refetch_hours)
    new: list = []
    fetched: list = []
    if not skipped and ticker:
        fetched = company_news(ticker, limit=limit, client=client)
        new = store.upsert_articles(cik, ticker, fetched)
        store.record_fetch(cik, ticker, len(new))

    # with an LLM present, also upgrade any heuristic placeholders left by a lazy
    # TUI-open; without one, only fill truly-missing extractions (no churn).
    extracted = 0
    for a in store.articles_missing_extraction(cik, version, upgrade_heuristic=llm is not None):
        store.put_extraction(a["id"], version, extract_article(a, llm=llm))
        extracted += 1

    return {"cik": int(cik), "ticker": ticker, "fetched": len(fetched),
            "new": len(new), "extracted": extracted, "skipped_fetch": bool(skipped)}


def backfill_company_news(cik: int, ticker: str | None, store: NewsStore, llm=None,
                          pages: int = 5, page_size: int = 100,
                          version: str = EXTRACT_VERSION,
                          client: httpx.Client | None = None) -> dict:
    """One-shot deep seed: paginate a company's news back in time, upsert, extract.

    Unlike the throttled nightly ingest, this deliberately pulls history (bounded by
    ``pages``) so recall's 'notable past' has depth on day one instead of accruing over
    weeks. Idempotent — already-stored ids are skipped, so re-running only extends the
    tail. Defaults to heuristic extraction (a deep LLM pass over months of history would
    be huge); the nightly LLM job upgrades material placeholders over time."""
    fetched = company_news_pages(ticker, pages=pages, page_size=page_size,
                                 client=client) if ticker else []
    new = store.upsert_articles(cik, ticker, fetched)
    store.record_fetch(cik, ticker, len(new))
    extracted = 0
    for a in store.articles_missing_extraction(cik, version, upgrade_heuristic=llm is not None):
        store.put_extraction(a["id"], version, extract_article(a, llm=llm))
        extracted += 1
    return {"cik": int(cik), "ticker": ticker, "fetched": len(fetched),
            "new": len(new), "extracted": extracted, "pages": pages}


def backfill_watchlist(store: NewsStore, ciks_tickers: list[tuple], llm=None,
                       pages: int = 5, page_size: int = 100, version: str = EXTRACT_VERSION,
                       transport: httpx.BaseTransport | None = None) -> dict:
    """Seed the memory for every watched name (bounded, idempotent). Aggregate deltas."""
    totals = {"companies": 0, "fetched": 0, "new": 0, "extracted": 0, "no_ticker": 0,
              "pages": pages}
    with httpx.Client(base_url=_INTRINIO_BASE, timeout=20.0, transport=transport) as client:
        for cik, ticker in ciks_tickers:
            totals["companies"] += 1
            if not ticker:
                totals["no_ticker"] += 1
            d = backfill_company_news(cik, ticker, store, llm=llm, pages=pages,
                                      page_size=page_size, version=version, client=client)
            for k in ("fetched", "new", "extracted"):
                totals[k] += d[k]
    return totals


def ingest_watchlist(store: NewsStore, ciks_tickers: list[tuple], llm=None,
                     limit: int = NEWS_FETCH_LIMIT, version: str = EXTRACT_VERSION,
                     force: bool = False, refetch_hours: float = NEWS_REFETCH_HOURS,
                     transport: httpx.BaseTransport | None = None) -> dict:
    """Nightly batch: ingest news for every (cik, ticker) on the watchlist.

    One shared HTTP client; each name quota-capped independently. Idempotent — a
    re-run inside the refetch window does no network work and only backfills any
    still-missing extractions. Returns aggregate deltas for the job log."""
    totals = {"companies": 0, "fetched": 0, "new": 0, "extracted": 0,
              "skipped_fetch": 0, "no_ticker": 0}
    per: list[dict] = []
    with httpx.Client(base_url=_INTRINIO_BASE, timeout=20.0, transport=transport) as client:
        for cik, ticker in ciks_tickers:
            totals["companies"] += 1
            if not ticker:
                totals["no_ticker"] += 1
            d = ingest_company_news(cik, ticker, store, llm=llm, limit=limit,
                                    version=version, force=force,
                                    refetch_hours=refetch_hours, client=client)
            per.append(d)
            for k in ("fetched", "new", "extracted"):
                totals[k] += d[k]
            totals["skipped_fetch"] += int(d["skipped_fetch"])
    totals["names"] = [d["cik"] for d in per if d["new"] or d["extracted"]][:50]
    return totals
