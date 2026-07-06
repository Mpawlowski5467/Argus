"""Grounded conversational Q&A over ONE company's narration packet + news memory.

The narration you already have, made interactive: ask "why is it ranked here?", "what
changed since last quarter?", "what's the news history?" — answered strictly from the
packet (fundamentals, the frozen model's signal + SHAP drivers, recalled news), with
every numeral checked by the same grounding guard. Read-only; the packet is built
AFTER scoring, so nothing here can move the signal.
"""

from __future__ import annotations

from .core import grounded_answer, isnum, pct1

QA_SYSTEM = (
    "You are a careful equity fundamentals analyst answering questions about ONE "
    "company, using ONLY the JSON CONTEXT you are given (pre-computed fundamentals with "
    "sector percentiles and YoY changes, the frozen model's cross-sectional signal and "
    "its SHAP drivers, and recalled news takeaways).\n"
    "RULES:\n"
    "- Use ONLY numbers that appear in the context. Never invent, estimate, round "
    "differently, or compute new figures (no arithmetic). If a figure isn't in the "
    "context, describe it in words or say you don't have it.\n"
    "- Respect each signal's 'read'/'direction': a HIGH percentile on a lower-is-better "
    "signal (leverage, accruals, asset growth) is a weakness, not a strength — the "
    "packet already decided; don't re-derive it.\n"
    "- The model signal is a relative cross-sectional RANK from a frozen statistical "
    "model, never a guaranteed or predicted return. The composite is a peer screen.\n"
    "- News items in context.news are REPORTED claims: reference them as such "
    "('reported', 'according to <source>'), name the article, and never treat a "
    "takeaway as established fact or let it change the model's verdict.\n"
    "- If the context does not answer the question, say so plainly rather than guess.\n"
    "- Answer concisely and directly. No buy/sell advice, price targets, or predictions."
)


def answer_from_packet(packet: dict, question: str, llm, history: list | None = None,
                       max_retries: int = 1) -> dict:
    """Answer ``question`` about a company from its narration ``packet``.

    Thin wrapper over :func:`stockscan.assist.core.grounded_answer` with the Q&A
    system prompt — trivially testable with a mock ``llm`` and a hand-built packet."""
    return grounded_answer(packet, question, llm, QA_SYSTEM,
                           max_retries=max_retries, history=history)


# -- the web chat: packet + the firewalled display reads ---------------------------
# The ticker page shows verdict / confidence / distress / drawdown / price — all
# computed AFTER scoring and deliberately kept OUT of the narration packet. A chat
# that can't see them refuses on exactly the numbers a person most wants explained,
# so the web context is the packet WIDENED with those display blocks. Display-side
# only: nothing here writes back toward the score/packet/paper/trade path.

CHAT_SYSTEM = QA_SYSTEM + (
    "\n- The context may carry a 'display' section of firewalled display-only reads "
    "computed after scoring: 'verdict' (the BUY/HOLD/AVOID call — nothing more than "
    "the percentile band), 'confidence' (conviction out of 100 derived from the frozen "
    "model's out-of-sample hit-rate for this decile — cite its hit_rate_pct and n with "
    "it), 'distress' and 'drawdown' (learned event probabilities — risk flags for "
    "display, never trade inputs), 'price' (trailing close summary — history, not a "
    "forecast), and 'flags' (filing age / in-sample). When you use a block's numbers, "
    "carry its 'note' caveat in your answer."
)


def _rounded(block: dict | None, pct_fields: dict[str, str]) -> dict | None:
    """Copy ``block`` adding display-rounded percentage twins (``prob`` 0.034 →
    ``prob_pct`` 3.4) so the phrasing the UI shows is citable under the grounding
    guard's exact-integer / ±0.02-fraction matching (core.pct1, shared with the
    book context builder)."""
    if not block:
        return None
    out = dict(block)
    for src, dst in pct_fields.items():
        v = out.get(src)
        if isnum(v):
            out[dst] = pct1(v * 100)
    return out


def build_chat_context(res: dict, price_summary: dict | None = None,
                       verdict: dict | None = None) -> dict:
    """The grounding context for the web chat: the narration packet widened with the
    display-only blocks from an :func:`stockscan.serve.analyze` result.

    Pure and non-mutating — ``res`` and its packet are copied, never touched. Each
    display block carries a number-free ``note`` stating its honest framing so the
    model can echo the caveat instead of paraphrasing it loose."""
    packet = res.get("packet") or {}
    display: dict = {}
    if verdict:
        display["verdict"] = {**verdict, "note": (
            "the call is only the percentile band — a relative peer rank among "
            "scored names today, not a return forecast and not advice")}
    conf = _rounded(res.get("confidence"), {"hit_rate": "hit_rate_pct"})
    if conf:
        display["confidence"] = {**conf, "note": (
            "conviction out of 100, derived from the frozen model's out-of-sample "
            "hit-rate for this decile; capped — never certainty")}
    dist = _rounded(res.get("distress"), {"prob": "prob_pct"})
    if dist:
        display["distress"] = {**dist, "note": (
            "learned probability of distress or delisting within the horizon — a "
            "display-only risk flag, never a trade input")}
    draw = _rounded(res.get("drawdown"), {"prob": "prob_pct", "threshold": "threshold_pct"})
    if draw:
        display["drawdown"] = {**draw, "note": (
            "learned probability of a deep peak-to-trough fall within the horizon — "
            "a display-only risk flag, never a trade input")}
    if res.get("flags"):
        display["flags"] = dict(res["flags"])
    if price_summary:
        pr = {k: (round(float(v), 2 if k in ("last", "hi_52w", "lo_52w") else 1)
                  if isinstance(v, (int, float)) else v)
              for k, v in price_summary.items() if k != "adv"}
        adv = price_summary.get("adv")
        if isinstance(adv, (int, float)):
            pr["adv_musd"] = round(float(adv) / 1e6, 1)
        display["price"] = {**pr, "note": (
            "trailing close-price summary — what already happened, not a forecast")}
    return {**packet, "display": display}


def answer_about_company(res: dict, question: str, llm, history: list | None = None,
                         price_summary: dict | None = None, verdict: dict | None = None,
                         max_retries: int = 1) -> dict:
    """The web chat turn: grounded answer over the WIDENED context (packet + display
    reads). Same contract as :func:`answer_from_packet` — refuse over fabricate."""
    context = build_chat_context(res, price_summary=price_summary, verdict=verdict)
    return grounded_answer(context, question, llm, CHAT_SYSTEM,
                           max_retries=max_retries, history=history)
