"""Grounded conversational Q&A over ONE company's narration packet + news memory.

The narration you already have, made interactive: ask "why is it ranked here?", "what
changed since last quarter?", "what's the news history?" — answered strictly from the
packet (fundamentals, the frozen model's signal + SHAP drivers, recalled news), with
every numeral checked by the same grounding guard. Read-only; the packet is built
AFTER scoring, so nothing here can move the signal.
"""

from __future__ import annotations

from .core import grounded_answer

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
