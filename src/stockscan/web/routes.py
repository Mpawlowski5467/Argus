"""The /api endpoints — thin wrappers: request → ArgusData method → JSON.

Every handler is a sync ``def`` so Starlette runs the blocking, network-touching
facade calls in its threadpool (never make these ``async def``).

FIREWALL: the signal packet (/api/ticker) returns score/percentile/decile/
drivers/verdict with ZERO live-view data. Profile, news, live quote, market cap
and themes each have their own endpoint the browser fetches AFTER the packet.
Nothing here may attach a live-view field to the ticker response. The user's
saved position (shares + cost basis, /api/positions) is PERSONAL live-view data:
it is stored and shown back to the user only, and is NEVER read into the score,
the paper book, or any signal computation.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..assist.move import HORIZONS
from ..view.chart import verdict
from ..view.treemap import squarify
from . import convert
from .state import STATE

router = APIRouter()


def _facade():
    """The loaded facade, or a 503 while the background load is still running."""
    if STATE.status != "ready" or STATE.adata is None:
        raise HTTPException(status_code=503, detail="loading")
    return STATE.adata


def _safe(fn, default):
    """Live-view calls fail open — one dead vendor never 500s a page."""
    try:
        return fn()
    except Exception:
        return default


# -- status (the loader's poll target) ---------------------------------------
@router.get("/status")
def status():
    if STATE.status == "ready" and STATE.adata is not None:
        return {"loading": False, "error": None, **STATE.adata.status()}
    return JSONResponse({"loading": True, "error": STATE.error}, status_code=503)


# -- scan / search / resolve --------------------------------------------------
@router.get("/sectors")
def sectors():
    return _facade().sectors()


@router.get("/scan")
def scan(sector: str = "all"):
    return _facade().scan(sector)


@router.get("/search")
def search(q: str = "", limit: int = 40):
    return _facade().search(q, limit)


@router.get("/resolve")
def resolve(q: str):
    hit = _facade().resolve(q)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"no match for {q!r}")
    return hit


# -- ticker: the signal packet (no live-view) --------------------------------
@router.get("/ticker/{cik}")
def ticker(cik: int):
    a = _facade()
    try:
        res = dict(a.ticker(cik))
    except ValueError as exc:            # no 10-K / stale / illiquid — surface it
        raise HTTPException(status_code=422, detail=str(exc))
    res["verdict"] = verdict((res.get("percentile") or 0) / 100.0)
    return convert.jsonable(res)


@router.get("/price/{cik}")
def price(cik: int):
    pr = _facade().price(cik)
    if pr is None:
        return {"price": None}
    return {
        "column": pr.get("column"),
        "points": convert.series_to_points(pr.get("series")),
        "summary": convert.jsonable(pr.get("summary")),
    }


@router.get("/ohlc/{cik}")
def ohlc(cik: int, tail: int = 252):
    df = _facade().ohlc(cik, tail)
    return {"ohlc": convert.ohlc_to_arrays(df)}


# -- live-view enrichment (own endpoints, fail open) -------------------------
@router.get("/events/{cik}")
def events(cik: int):
    a = _facade()
    return convert.jsonable(_safe(lambda: a.events(cik), []))


@router.get("/news/{cik}")
def news(cik: int):
    a = _facade()
    return convert.jsonable(_safe(lambda: a.news(cik), []))


@router.get("/profile/{cik}")
def profile(cik: int):
    a = _facade()
    return _safe(lambda: a.profile(cik), None)


@router.get("/live/quote/{cik}")
def live_quote(cik: int, refresh: bool = False):
    a = _facade()
    return convert.jsonable(_safe(lambda: a.live_quote(cik, refresh=refresh), None))


@router.get("/market-cap/{cik}")
def market_cap(cik: int):
    a = _facade()
    return {"cik": cik, "cap": convert.jsonable(_safe(lambda: a.market_cap(cik), None))}


@router.post("/market-caps")
def market_caps(body: dict):
    a = _facade()
    ciks = list(dict.fromkeys(body.get("ciks", [])))   # dedupe, keep order

    def one(cik):
        try:
            return cik, a.market_cap(int(cik))
        except Exception:
            return cik, None

    caps: dict = {}
    if ciks:
        with ThreadPoolExecutor(max_workers=6) as ex:
            for cik, cap in ex.map(one, ciks):
                caps[str(cik)] = convert.jsonable(cap)
    return {"caps": caps}


# -- markets ------------------------------------------------------------------
@router.get("/markets")
def markets(top_k: int = 6):
    a = _facade()
    return {
        "themes": _safe(lambda: a.theme_markets(top_k), []),
        "industries": a.markets(top_k),
    }


# treemap drill-in: one market's names sized by live cap, packed by squarify.
# ``kind`` is "theme" | "ind"; query params so names with spaces/slashes work.
_MAP_ASPECT = 2.3   # squarify in a wide box so tiles stay near-square in the browser


@router.get("/market")
def market(kind: str, name: str, top: int = 18):
    a = _facade()
    cons = _safe(lambda: a.market_constituents(kind, name), [])

    def one(it):
        try:
            return {**it, "cap": a.market_cap(int(it["cik"]))}
        except Exception:
            return {**it, "cap": None}

    items = []
    if cons:
        with ThreadPoolExecutor(max_workers=6) as ex:
            items = [it for it in ex.map(one, cons) if it.get("cap")]
    items.sort(key=lambda it: it["cap"], reverse=True)
    items = items[:top]

    rects = squarify([it["cap"] for it in items], 0.0, 0.0, _MAP_ASPECT, 1.0)
    tiles = []
    for it, (x, y, w, h) in zip(items, rects):
        tiles.append({
            "cik": it["cik"], "ticker": it["ticker"], "name": it.get("name"),
            "pct": it.get("pct"), "decile": it.get("decile"), "cap": it["cap"],
            "x": x / _MAP_ASPECT, "y": y, "w": w / _MAP_ASPECT, "h": h,   # normalized 0..1
        })
    return {"kind": kind, "name": name, "aspect": _MAP_ASPECT, "tiles": convert.jsonable(tiles)}


# -- watchlist ----------------------------------------------------------------
@router.get("/watch")
def watch():
    return convert.jsonable(_facade().watch())


@router.get("/watch-ids")
def watch_ids():
    """Just the watched CIKs — cheap payload for the scan page's star column."""
    return {"ciks": convert.jsonable(_safe(lambda: _facade().watched_ciks(), []))}


@router.get("/watch/{cik}")
def is_watched(cik: int):
    return {"cik": cik, "watched": bool(_facade().is_watched(cik))}


@router.post("/watch/{cik}/toggle")
def toggle_watch(cik: int):
    return {"cik": cik, "watched": bool(_facade().toggle_watch(cik))}


# -- positions (PERSONAL holdings — DISPLAY-ONLY live-view; never a signal input) --
@router.get("/positions")
def positions():
    return convert.jsonable(_facade().positions())


@router.post("/positions/{cik}")
def set_position(cik: int, body: dict):
    shares = float(body.get("shares") or 0)
    cost = float(body.get("cost") or 0)
    return convert.jsonable(_facade().set_position(cik, shares, cost))


@router.delete("/positions/{cik}")
def remove_position(cik: int):
    _facade().remove_position(cik)
    return {"cik": cik, "removed": True}


# -- portfolio scorecard: book-level aggregation of the user's holdings -------
# DISPLAY-ONLY, firewalled. A same-day peer-rank snapshot of the book (equal- AND
# value-weighted percentile, distress exposure, concentration) + the full holdings
# list — never a portfolio forecast, never read into the score or paper book.
@router.get("/scorecard")
def scorecard():
    return convert.jsonable(_facade().scorecard())


# -- paper-forward ------------------------------------------------------------
@router.get("/paper")
def paper():
    return convert.jsonable(_facade().paper())


# -- narration + grounded chat (slow / optional LLM) + refresh ----------------
# One local model serves both routes and serializes anyway, so LLM work is
# single-flight: narrate WAITS its turn (one-click, the button already shows
# progress); ask returns {"busy": true} instead of queueing threadpool threads.
_LLM_GATE = threading.Lock()


@contextmanager
def _llm_single_flight():
    """Yield True holding the LLM gate, or False when another request holds it.

    One shared lock, four LLM endpoints — a hand-rolled acquire/finally per route
    is one missed release away from deadlocking every AI surface at once, so the
    lock discipline lives here once. Gate ONLY the model call: context assembly and
    any code-only answer must run before/outside the ``with`` so a slow fetch or an
    LLM-free response never holds the gate (mirrors how /narrate keeps facade work
    outside it)."""
    got = _LLM_GATE.acquire(blocking=False)
    try:
        yield got
    finally:
        if got:
            _LLM_GATE.release()


@router.post("/narrate/{cik}")
def narrate(cik: int):
    a = _facade()
    try:
        res = a.ticker(cik)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    with _LLM_GATE:
        return convert.jsonable(_safe(lambda: a.narrate(res["packet"]), {"narrative": "", "tier": "?"}))


def _ask_input(body: dict) -> tuple[str, list]:
    """Validate an ask body: a bounded question + the browser's cleaned history."""
    question = str(body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="empty question")
    if len(question) > 2000:
        raise HTTPException(status_code=422, detail="question too long")
    history = [t for t in (body.get("history") or [])
               if isinstance(t, dict) and t.get("content")][-8:]
    return question, history


@router.post("/ask/book")
def ask_book(body: dict):
    """Grounded chat about the BOOK — the scorecard made interactive. Same contract
    as /ask/{cik} one level up: answers come ONLY from the scorecard the tab shows
    (widened with display-rounded citable twins), the grounding guard catches any
    fabricated numeral, and the aggregate honesty rules (both weightings, snapshot
    not outlook, no portfolio forecast, no advice) live in the system prompt
    (assist.book.PORTFOLIO_SYSTEM). Registered BEFORE /ask/{cik} so the literal
    path wins — the int-typed {cik} would 422 "book" without falling through."""
    a = _facade()
    question, history = _ask_input(body)
    with _llm_single_flight() as got:
        if not got:
            return {"busy": True}
        return convert.jsonable(a.ask_book(question, history=history))


@router.post("/ask/{cik}")
def ask(cik: int, body: dict):
    """Grounded chat about one name — the narration made interactive. Answers come
    ONLY from the company's computed context (packet + display reads + recalled
    news); a fabricated numeral is caught by the grounding guard and the assistant
    REFUSES rather than guesses (assist.core.grounded_answer). Firewalled like
    /narrate: everything is assembled AFTER scoring, nothing feeds back. History is
    the browser's (stateless server) and never expands the grounding domain."""
    a = _facade()
    question, history = _ask_input(body)
    with _llm_single_flight() as got:
        if not got:
            return {"busy": True}
        try:
            return convert.jsonable(a.ask(cik, question, history=history))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))


@router.post("/explain-move/{cik}")
def explain_move(cik: int, body: dict):
    """Grounded "explain this move" — one trailing-change chip, one shot. The
    context is the move plus ONLY the news/filings dated inside its window; the
    system prompt (assist.move) bans causal claims — items COINCIDED, nothing
    more. Context assembly (price read + windowed news + EDGAR fetch) and the
    code-only answer for an empty window run OUTSIDE the gate, so a slow fetch
    or an LLM-free response never blocks ask/narrate; only the model call is
    single-flighted, returning {"busy": true} when contended."""
    a = _facade()
    horizon = str(body.get("horizon") or "").strip()
    if horizon not in HORIZONS:
        raise HTTPException(status_code=422, detail=f"unknown horizon {horizon!r}")
    bundle, deterministic = a.move_context(cik, horizon)
    if deterministic is not None:
        return convert.jsonable(deterministic)
    with _llm_single_flight() as got:
        if not got:
            return {"busy": True}
        return convert.jsonable(a.move_answer(cik, horizon, bundle))


# -- alerts ---------------------------------------------------------------------
@router.post("/alerts/seen")
def alerts_seen(body: dict | None = None):
    """Acknowledge alerts from the UI — the CLI was the only way to clear the
    statusbar badge, which contradicts the web-only decision. Body {"ids": [...]}
    marks specific alerts; empty/absent body marks all unseen."""
    a = _facade()
    ids = (body or {}).get("ids")
    if ids is not None:
        ids = [int(i) for i in ids]
    return convert.jsonable(a.mark_alerts_seen(ids))


# -- market context (breadth / vol / drawdown — display-only facts) -----------
@router.get("/regime")
def regime():
    """Three market-context reads over the liquid universe. Deliberately not a
    named 'regime' and never a model input — a header line for the human."""
    a = _facade()
    return convert.jsonable(_safe(lambda: a.regime(), None) or {})


# -- system health (the nightly's stored screen + job history) ----------------
@router.get("/health")
def health():
    """Last stored health screen + recent job strip. Read-only: the checks are run
    by the nightly (they probe the web UI and LLM — a live re-run here would
    self-probe), so a stale as_of is itself the signal that ops has stalled."""
    a = _facade()
    return convert.jsonable(_safe(lambda: a.health(), {}))


# -- AI analyst panel (bull/bear/risk/synthesis — commentary, never the signal) --
@router.get("/panel/{cik}")
def panel_cached(cik: int):
    """Whatever the panel cache holds for this name's LIVE context — no LLM, no
    gate. A new filing or rank move changes the context hash and empties this."""
    a = _facade()
    return convert.jsonable(a.panel_cached(cik))


@router.post("/panel/{cik}")
def panel_role(cik: int, body: dict):
    """Generate ONE memo (the frontend chains bull -> bear -> risk -> synthesis so
    each request stays bounded and the page paints progressively). Single-flight
    like every LLM surface."""
    from ..assist.analyst import ROLES

    role = (body or {}).get("role")
    if role not in ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {ROLES}")
    a = _facade()
    with _llm_single_flight() as got:
        if not got:
            return {"busy": True}
        return convert.jsonable(a.panel_role(cik, role))


# -- overnight digest (deterministic card; grounded LLM brief on demand) ------
@router.get("/digest")
def digest():
    """Last night's ops record: job statuses, unseen alerts, paper progress —
    the number-true card payload. The LLM prose is the POST (slow, optional)."""
    a = _facade()
    return convert.jsonable(_safe(lambda: a.digest(), {}))


@router.post("/digest")
def digest_brief():
    """The grounded morning brief (assist.brief) over the SAME context the card
    shows — uses only numbers in that record or refuses. Single-flight like ask."""
    a = _facade()
    with _llm_single_flight() as got:
        if not got:
            return {"busy": True}
        return convert.jsonable(_safe(
            lambda: a.digest_brief(),
            {"answer": "", "grounded": True, "refused": True, "violations": []}))


@router.post("/refresh")
def refresh():
    if STATE.status != "ready":
        raise HTTPException(status_code=503, detail="loading")
    STATE.refresh()
    return {"ok": True}


# -- on-demand data update ("update data" button) ----------------------------
# Runs the SAME nightly dispatcher launchd runs, as a self-guarded subprocess: pull fresh
# prices / filings / news now, then POST /reload swaps in the new data. The scheduled nightly
# is unaffected — both take the repo-wide ops flock, so they can never overlap.
@router.post("/nightly")
def nightly_run():
    if STATE.status != "ready":
        raise HTTPException(status_code=503, detail="loading")
    return STATE.start_nightly()


@router.get("/nightly")
def nightly_status():
    return STATE.nightly_status()


@router.post("/reload")
def reload_facade():
    """Full reload from disk (after a nightly update) — rebuilds the scored cross-section
    from freshly-ingested data. The loader/poll handshake covers the reload."""
    STATE.reload()
    return {"ok": True}
