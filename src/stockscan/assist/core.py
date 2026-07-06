"""The shared grounded-generation loop: answer STRICTLY from a context dict, or refuse.

This is the narration contract generalized to arbitrary read-only Q&A. The model is
handed a JSON context of already-computed facts and must answer using ONLY numbers that
appear in it. A deterministic post-check (the SAME grounding guard the narrator uses,
:func:`stockscan.narrate.ground.check_grounding`) verifies every numeral traces to the
context; a fabricated figure triggers a bounded retry, then an honest refusal rather
than a plausible guess. No arithmetic, no outside facts — code owns the numbers.
"""

from __future__ import annotations

import json
import math

from ..narrate.ground import check_grounding

REFUSAL = ("I can't answer that from the available data without guessing — the numbers "
           "you'd need aren't in what I was given.")


# --- display-rounding twins (shared by the chat context builders) -----------------
# The UI rounds for display (Math.round percentiles, toFixed(1) percents) and the
# grounding guard matches exact ints / ±0.02 fractions — so every displayed phrasing
# needs a citable twin in the context. abs() because prose says "down 24.4%", and
# the UI's − is U+2212, which the number extractor reads as positive anyway.

def isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def jsround(x) -> int:
    """JS ``Math.round`` (half-up) of ``abs(x)`` — what the UI's int display shows."""
    return int(math.floor(abs(float(x)) + 0.5))


def pct1(x) -> float:
    """1-decimal display twin of ``abs(x)`` — what ``sign()`` / toFixed(1) shows."""
    return round(abs(float(x)), 1)


def grounded_answer(context: dict, question: str, llm, system: str,
                    max_retries: int = 1, history: list | None = None) -> dict:
    """Answer ``question`` about ``context`` (a JSON-able dict of computed facts).

    Returns ``{answer, grounded, violations, attempts, refused}``. ``llm`` is a
    callable(system, user) -> str. Every numeral in the answer must trace to
    ``context``; otherwise the reply is rejected (retry, then refusal). ``history``
    is an optional list of prior ``{role, content}`` turns woven into the prompt so a
    conversation stays coherent without ever expanding the grounding domain."""
    ctx_json = json.dumps(context, indent=2, default=str, sort_keys=True)
    convo = ""
    for turn in (history or []):
        role = "You" if turn.get("role") == "user" else "Assistant"
        convo += f"{role}: {turn.get('content','')}\n"
    base = (f"CONTEXT (the only facts you may use):\n{ctx_json}\n\n"
            + (f"CONVERSATION SO FAR:\n{convo}\n" if convo else "")
            + f"QUESTION:\n{question}")

    log: list = []
    for attempt in range(max_retries + 1):
        prompt = base if not log else (
            base + "\n\nYour previous answer used numbers that are NOT in the context: "
            + json.dumps(log[-1]) + ". Redo it using only numbers that appear in the "
            "context (or describe them in words); do not invent or compute figures.")
        try:
            text = llm(system, prompt)
        except Exception as exc:  # transport/timeout: degrade, never crash — and never
            # retry: a dead endpoint fails again instantly, a TIMEOUT would double the
            # wait for nothing. Retries are for fabrication (the model answered fast
            # but leaked a number), not for transport.
            log.append([f"llm-error:{type(exc).__name__}"])
            break
        leaks = check_grounding(text or "", context)
        if not leaks:
            return {"answer": (text or "").strip(), "grounded": True,
                    "violations": [], "attempts": attempt + 1, "refused": False}
        log.append(leaks)
    return {"answer": REFUSAL, "grounded": True, "violations": log[-1] if log else [],
            "attempts": max_retries + 1, "refused": True}
