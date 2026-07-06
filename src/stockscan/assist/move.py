"""Grounded "explain this move" — what REPORTEDLY coincided with a price change.

The ticker page's trailing-change chips (1w/1m/3m/1y), made explainable: one click
composes the move itself (display-rounded, citable), the news-memory takeaways whose
dates fall inside that window (number-free by construction), and the SEC filings
that landed in it, then asks the local model what reportedly happened around the
move. The rules are strict where the fabrication risk is worst: news COINCIDES with
a move, it never proves cause; the model's monthly fundamentals rank neither
predicts nor explains short-horizon price action; and when nothing coincided, a
deterministic path says exactly that without waking the LLM — refusing to invent a
story is the feature.

Read-only and firewalled like the rest of ``assist``: everything here is assembled
AFTER scoring from display-side reads; nothing feeds back toward the score, the
paper book, or a trade.
"""

from __future__ import annotations

from datetime import date, timedelta

from ..horizons import BY_KEY as _HORIZON
from .core import grounded_answer, isnum as _num, pct1 as _pct1

MOVE_SYSTEM = (
    "You are a careful equity analyst explaining ONE price move for ONE company, "
    "using ONLY the JSON CONTEXT you are given (the move itself, news-memory "
    "takeaways whose dates fall inside the move's window, and SEC filings from the "
    "same window).\n"
    "RULES:\n"
    "- Use ONLY numbers that appear in the context (the move's chg_pct, the last "
    "close, dates). Never invent, estimate, or compute figures. News takeaways are "
    "deliberately number-free — describe them in words.\n"
    "- COINCIDENCE IS NOT CAUSE: the news and filings in the context merely fell "
    "inside the move's window. Say an item 'coincided with' or 'landed during' the "
    "move; NEVER present it as the reason — no 'because of', 'driven by', 'on the "
    "news', or any phrasing that asserts cause.\n"
    "- News items are REPORTED claims: attribute them ('reported', 'according to "
    "<source>'), name the source, and never treat a takeaway as established fact.\n"
    "- The move is a trailing close-to-close change — history, not momentum, a "
    "signal, or a forecast. Do not predict what comes next, and give no advice.\n"
    "- The model's rank is a monthly fundamentals peer rank; it neither predicts "
    "nor explains short-horizon price moves — never imply it does.\n"
    "- If the context holds nothing that coincided, say so plainly rather than "
    "reach for a story.\n"
    "- Answer in two to four sentences, concise and direct."
)

# horizon -> the price_summary field suffix, a plain-English window, and the
# calendar-day lookback (``days``) used to keep only COINCIDING news/filings — the
# trading-day and calendar-day spans are pinned together in the shared horizon table
# (``stockscan.horizons``) so this window can't drift from the chip's price lookback.
HORIZONS: dict[str, dict] = {
    k: {"label": h.label, "days": h.calendar_days, "window": h.window}
    for k, h in _HORIZON.items()
}

_QUESTION = ("Explain this {label} move: state it, then what reported news or "
             "filings coincided with its window (with dates and sources), without "
             "claiming any of them caused it.")


def _cutoff(as_of, days: int) -> str | None:
    try:
        return (date.fromisoformat(str(as_of)[:10]) - timedelta(days=days)).isoformat()
    except (TypeError, ValueError):
        return None


def window_cutoff(horizon: str, as_of) -> str | None:
    """The lower date bound of a horizon's window (``as_of − days``), so a caller
    can pre-fetch news to the SAME window ``build_move_context`` filters to —
    instead of window-filtering an already-truncated recall."""
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}")
    return _cutoff(as_of, HORIZONS[horizon]["days"])


def _within(rows, key: str, cutoff: str | None, end: str | None) -> list[dict]:
    """Rows whose ISO date at ``key`` falls inside [cutoff, end] (kept when no
    anchor date exists — recalled rows are already recent-or-notable). The upper
    bound matters: news memory can hold items PUBLISHED AFTER the price series'
    last close, and an item dated after the move ended cannot have coincided."""
    out = []
    for r in rows or []:
        d = str(r.get(key) or "")[:10]
        if d and (cutoff is None or d >= cutoff) and (end is None or d <= end):
            out.append(dict(r))
    return out


def build_move_context(meta: dict, horizon: str, summary: dict, as_of,
                       news=None, filings=None) -> dict:
    """The grounding context for one chip: the move (display-rounded citable
    numbers ONLY — chg_pct as the chip's abs 1-dp phrasing, last close 2-dp),
    window-filtered news takeaways and filings, and number-free honesty notes.
    The 52-week range is deliberately left out: it invites range phrasings whose
    numerals aren't citable, and the range is not what this feature explains."""
    h = HORIZONS[horizon]
    chg = summary.get("chg_" + horizon)
    move: dict = {
        "horizon": h["label"],
        "window": h["window"],
        "as_of": str(as_of)[:10] if as_of else None,
        "note": ("a trailing close-to-close percent change — what already "
                 "happened, never a signal, a forecast, or a reason"),
    }
    if _num(chg):
        move["direction"] = "up" if chg > 0 else "down" if chg < 0 else "flat"
        move["chg_pct"] = _pct1(chg)          # the chip's display twin: abs, 1dp
    if _num(summary.get("last")):
        move["last"] = round(float(summary["last"]), 2)
    cut, end = _cutoff(as_of, h["days"]), move["as_of"]
    return {
        "company": {"ticker": str(meta.get("ticker") or "—"),
                    "name": str(meta.get("name") or "")},
        "move": move,
        "news": _within(news, "date", cut, end),
        "news_note": ("reported claims from news memory dated inside the move's "
                      "window — they COINCIDED with the move; coincidence is not "
                      "cause"),
        "filings": [{"form": f.get("form"), "filed_date": f.get("filed_date"),
                     "label": f.get("label")}
                    for f in _within(filings, "filed_date", cut, end)],
        "filings_note": ("SEC filings that landed inside the window — events on "
                         "the record, not explanations"),
        "model_note": ("the model's rank is a monthly fundamentals peer rank; it "
                       "neither predicts nor explains short-horizon price moves"),
    }


def deterministic_answer(ctx: dict) -> dict | None:
    """The code-only honest answer for a chip, or ``None`` when the LLM is needed.

    Two cases never wake the model — an unmeasurable move (no history for the
    horizon) and an empty window (nothing coincided). Kept separate from the LLM
    path so a caller can decide WITHOUT holding the single-flight gate: a
    guaranteed-instant answer must never queue behind an unrelated narration."""
    move, tk = ctx["move"], ctx["company"]["ticker"]
    base = {"grounded": True, "violations": [], "attempts": 0, "refused": False,
            "deterministic": True}
    if "chg_pct" not in move:
        return {**base, "answer": (
            f"Not enough price history to measure a {move['horizon']} change for "
            f"{tk}, so there is no move here to explain.")}
    if not ctx["news"] and not ctx["filings"]:
        word = {"up": "up", "down": "down"}.get(move["direction"], "about flat, at")
        return {**base, "answer": (
            f"{tk} is {word} {move['chg_pct']}% over {move['window']}"
            + (f" (as of {move['as_of']})" if move.get("as_of") else "") +
            ". Nothing in this name's news memory or recent SEC filings falls "
            "inside that window, so there is no reported company event here to "
            "point to — and a move without a matching headline is common. Even "
            "when items do coincide, coincidence would not prove cause; the "
            "model's monthly fundamentals rank does not explain short-horizon "
            "price moves either.")}
    return None


def answer_from_context(ctx: dict, llm, max_retries: int = 1) -> dict:
    """The grounded LLM shot over an already-built (non-deterministic) move context.

    Split from :func:`explain_move` so the web facade can hold the single-flight
    gate around ONLY this call — the context assembly and the code-only paths run
    gate-free. The caller must have checked :func:`deterministic_answer` first."""
    r = grounded_answer(ctx, _QUESTION.format(label=ctx["move"]["horizon"]), llm,
                        MOVE_SYSTEM, max_retries=max_retries)
    return {**r, "deterministic": False}


def explain_move(meta: dict, horizon: str, summary: dict, as_of,
                 news=None, filings=None, llm=None, max_retries: int = 1) -> dict:
    """One chip, one grounded shot. Same result contract as the ask surfaces
    (``{answer, grounded, violations, attempts, refused}``) plus ``deterministic``:
    True when the answer was built in code — no measurable change, or an empty
    window — so no LLM was woken and nothing could be invented."""
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}")
    ctx = build_move_context(meta, horizon, summary, as_of,
                             news=news, filings=filings)
    det = deterministic_answer(ctx)
    if det is not None:
        return det
    return answer_from_context(ctx, llm, max_retries=max_retries)
