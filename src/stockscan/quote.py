"""Live/intraday quote for a security (Intrinio realtime price endpoint).

This is a LIVE-view convenience only — it never feeds the model or the backtest
(the signal is a quarterly fundamental one, computed on end-of-day data). It is a
manual/lazy fetch so it doesn't burn quota on every screen.
"""

from __future__ import annotations

import httpx

from .config import INTRINIO_API_KEY
from .prices import intrinio_get_json

_BASE = "https://api-v2.intrinio.com"


def realtime_price(security_id: str, api_key: str | None = None,
                   client: httpx.Client | None = None) -> dict | None:
    """Latest available trade for a security. None on missing key/id or any error."""
    api_key = api_key or INTRINIO_API_KEY
    if not api_key or not security_id:
        return None
    own = client is None
    client = client or httpx.Client(base_url=_BASE, timeout=20.0)
    try:
        try:
            d = intrinio_get_json(
                client, f"/securities/{security_id}/prices/realtime", {"api_key": api_key})
        except Exception:
            return None
        if not d:
            return None
        last = d.get("last_price")
        prev = d.get("close_price")
        chg = None
        if last is not None and prev not in (None, 0):
            chg = (last / prev - 1.0) * 100.0
        return {
            "last": last if last is not None else prev,
            "time": d.get("last_time") or d.get("updated_on"),
            "bid": d.get("bid_price"),
            "ask": d.get("ask_price"),
            "prev_close": prev,
            "chg_pct": chg,
        }
    finally:
        if own:
            client.close()
