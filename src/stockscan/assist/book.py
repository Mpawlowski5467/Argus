"""Grounded conversational Q&A over the user's BOOK — the portfolio scorecard.

The book tab, made interactive: ask "where does my book rank?", "how concentrated
am I?", "which names are flagged?" — answered strictly from the scorecard the tab
already shows (:func:`stockscan.portfolio.scorecard`), with every numeral checked
by the same grounding guard. Read-only and firewalled like the rest of ``assist``:
the scorecard is a display-side join built AFTER scoring, so nothing here can
touch the signal, the paper book, or a trade.

The honesty rules are re-derived here for AGGREGATES, where the drift risk is
worst: one collapsed number over a whole book reads like a portfolio outlook. So
the system prompt requires both weightings together, snapshot-not-outlook framing,
and bans any portfolio forecast or advice outright.
"""

from __future__ import annotations

from .core import grounded_answer, isnum as _num, jsround as _jsround, pct1 as _pct1

PORTFOLIO_SYSTEM = (
    "You are a careful equity fundamentals analyst answering questions about the "
    "user's BOOK — the set of names they hold or watch — using ONLY the JSON CONTEXT "
    "you are given (a same-day scorecard: per-name model standing, position values "
    "and P/L from the last close, distress/drawdown risk flags, and concentration).\n"
    "RULES:\n"
    "- Use ONLY numbers that appear in the context. Never invent, estimate, round "
    "differently, or compute new figures — no arithmetic, which at book level means "
    "NO summing, averaging, weighting, or netting holdings yourself; the pre-computed "
    "aggregates are the only aggregates. If a figure isn't in the context, describe "
    "it in words or say you don't have it.\n"
    "- Everything here is a SAME-DAY SNAPSHOT of a relative peer rank from ONE frozen "
    "monthly cross-sectional model: where the names stand among scored peers TODAY. "
    "It is NEVER a portfolio forecast, an expected return, or a price target, and you "
    "must never imply one — not even hedged ('likely', 'should', 'poised to').\n"
    "- When you cite the book's model standing, ALWAYS give both weightings together: "
    "the equal-weighted percentile (every tracked name counts the same) and the "
    "value-weighted percentile (weighted by position value — where the money actually "
    "sits). If one is missing from the context, say so. Never let a single collapsed "
    "number stand in for the book.\n"
    "- A percentile is a peer rank — the share of scored names ranked below today — "
    "not a probability of gains. Distress and drawdown flags are display-only risk "
    "flags from separate firewalled models, never trade inputs.\n"
    "- Values, prices, and P/L come from the last close: they describe what already "
    "happened, not an outlook.\n"
    "- A name marked outside the liquid universe has NO model standing — say that "
    "plainly; never fill one in.\n"
    "- No advice of any kind: never suggest buying, selling, trimming, adding, "
    "hedging, diversifying, or rebalancing anything.\n"
    "- If the context does not answer the question, say so plainly rather than guess.\n"
    "- Answer concisely and directly."
)


def build_book_context(sc: dict) -> dict:
    """The grounding context for the book chat: the scorecard dict widened with
    display-rounded citable twins and number-free honesty notes.

    Pure and non-mutating — ``sc`` and its nested rows are copied, never touched.
    Every number the book tab can show must trace: rounded percentile twins
    (``percentile_equal_round``), 1-dp P/L-percent twins, per-bucket concentration
    percents, per-holding risk probability percents, and the flagged-value sums the UI
    prints (``value_at_risk``) — code owns the numbers, the model only frames them."""
    ctx = dict(sc)
    ctx["note"] = (
        "a same-day peer-rank snapshot of the names the user tracks — the model makes "
        "one monthly cross-sectional peer ranking, so nothing here is a portfolio "
        "forecast, an expected return, or advice")
    ctx["weighting_note"] = (
        "equal-weight counts every tracked name the same; value-weight leans on where "
        "the money actually sits — always cite both together, neither stands in alone")
    for src, dst in (("percentile_equal", "percentile_equal_round"),
                     ("percentile_value", "percentile_value_round")):
        if _num(ctx.get(src)):
            ctx[dst] = _jsround(ctx[src])
    if _num(ctx.get("unrealized_pl_pct")):
        ctx["unrealized_pl_pct_round"] = _pct1(ctx["unrealized_pl_pct"])

    holdings = []
    for h in sc.get("holdings") or []:
        row = dict(h)
        if _num(row.get("unrealized_pl_pct")):
            row["unrealized_pl_pct_round"] = _pct1(row["unrealized_pl_pct"])
        if _num(row.get("dprob")):
            row["dprob_pct"] = _pct1(row["dprob"] * 100)
        if _num(row.get("wprob")):
            row["wprob_pct"] = _pct1(row["wprob"] * 100)
        holdings.append(row)
    ctx["holdings"] = holdings

    for key, label in (("distress", "distress / delist"),
                       ("drawdown", "large drawdown")):
        risk = sc.get(key)
        if not isinstance(risk, dict):
            continue
        d = dict(risk)
        val = d.get("value")
        if isinstance(val, dict):
            d["value_at_risk"] = (val.get("high") or 0.0) + (val.get("elevated") or 0.0)
        d["note"] = (f"learned {label} flags on individual names — display-only risk "
                     "exposure, never a trade input and never a portfolio forecast")
        ctx[key] = d

    for key in ("industry_concentration", "sector_concentration"):
        buckets = []
        for b in sc.get(key) or []:
            row = dict(b)
            for src, dst in (("weight_count", "weight_count_pct"),
                             ("weight_value", "weight_value_pct")):
                if _num(row.get(src)):
                    row[dst] = _jsround(row[src] * 100)
            buckets.append(row)
        ctx[key] = buckets
    return ctx


def answer_about_book(sc: dict, question: str, llm, history: list | None = None,
                      max_retries: int = 1) -> dict:
    """The book chat turn: grounded answer over the widened scorecard context. Same
    contract as :func:`stockscan.assist.qa.answer_about_company` — refuse over
    fabricate; trivially testable with a mock ``llm`` and a hand-built scorecard."""
    context = build_book_context(sc)
    return grounded_answer(context, question, llm, PORTFOLIO_SYSTEM,
                           max_retries=max_retries, history=history)
