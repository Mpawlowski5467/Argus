"""Ticker <-> CIK map from EDGAR, cached to disk.

This is also the seed of the universe: ``company_tickers.json`` is the list of
currently-listed filers. (Delisted names get layered in later from the
survivorship ledger; see DESIGN.md.)
"""

from __future__ import annotations

import json

from ..config import RAW_DIR
from .client import EdgarClient

_CACHE = RAW_DIR / "company_tickers.json"


def load_ticker_map(refresh: bool = False, client: EdgarClient | None = None) -> dict[str, int]:
    """Return ``{TICKER: cik}``. Cached on disk; fetched from EDGAR on first use/refresh."""
    if refresh or not _CACHE.exists():
        own = client is None
        client = client or EdgarClient()
        try:
            data = client.company_tickers()
        finally:
            if own:
                client.close()
        _CACHE.write_text(json.dumps(data))
    else:
        data = json.loads(_CACHE.read_text())
    return {str(row["ticker"]).upper(): int(row["cik_str"]) for row in data.values()}


def cik_for(ticker: str, **kwargs) -> int | None:
    return load_ticker_map(**kwargs).get(ticker.upper())


def cik_to_ticker(**kwargs) -> dict[int, str]:
    """Return ``{cik: TICKER}``. A CIK with several share classes keeps the first seen."""
    out: dict[int, str] = {}
    for ticker, cik in load_ticker_map(**kwargs).items():
        out.setdefault(cik, ticker)
    return out
