"""Narrate a company's signals with a local LLM, grounded so it invents no numbers.

Flow: build packet -> prompt the LLM -> verify every number traces to the packet ->
retry once -> deterministic template fallback. The verdict/score is never the LLM's;
it only frames the pre-computed signals.
"""

from __future__ import annotations

import json

from .ground import check_grounding
from .packet import build_packet

SYSTEM = (
    "You are a careful equity fundamentals analyst. You are given a JSON packet of "
    "already-computed signals for one company (values, sector percentiles, YoY changes). "
    "Write a concise (120-180 word) plain-English read of its fundamental profile and how "
    "it ranks versus sector peers.\n"
    "HARD RULES:\n"
    "- Use ONLY numbers that appear in the packet. Never invent, estimate, or compute new "
    "figures — including counts. If you need a number that isn't in the packet, describe it "
    "in words instead.\n"
    "- Quote values and percentiles exactly as given (e.g. '31%', '97th percentile').\n"
    "- No buy/sell advice, price targets, or predictions. This is analysis only.\n"
    "- The composite is a peer screen, not a forecast — say so if you mention it.\n"
    "- Lead with the strongest and weakest signals; note any material YoY change."
)


def _ord(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _template(packet: dict) -> str:
    m = packet["meta"]
    ranked = sorted(
        [s for s in packet["signals"] if s.get("pct_rank") is not None],
        key=lambda s: s["pct_rank"],
        reverse=True,
    )
    def fmt(s):
        return f"{s['label']} {s['value']}{s['unit']} ({_ord(s['pct_rank'])} pct)"
    strengths = "; ".join(fmt(s) for s in ranked[:3])
    weaker = "; ".join(fmt(s) for s in ranked[-3:])
    comp = packet["composite"]["percentile"]
    comp_txt = f"composite quality {_ord(comp)} percentile vs sector" if comp is not None else "composite n/a"
    return (
        f"{m['name']} (FY{m['fiscal_year']}, {m['sector']}): {comp_txt}. "
        f"Strongest: {strengths}. Weakest: {weaker}. {packet['disclaimer']}"
    )


def narrate(company, llm=None, features_df=None, max_retries: int = 1) -> dict:
    """Return {packet, narrative, grounded, source}. ``llm`` is a callable(system, user)->str."""
    packet = build_packet(company, features_df=features_df)
    if llm is None:
        return {"packet": packet, "narrative": _template(packet), "grounded": True, "source": "template"}

    user = json.dumps(packet, indent=2, default=str)
    violations: list[float] = []
    for _ in range(max_retries + 1):
        text = llm(SYSTEM, user)
        violations = check_grounding(text, packet)
        if not violations:
            return {"packet": packet, "narrative": text, "grounded": True, "source": "llm"}
    return {
        "packet": packet,
        "narrative": _template(packet),
        "grounded": True,
        "source": "template-fallback",
        "rejected_numbers": violations,
    }
