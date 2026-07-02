"""Grounding guard: every number in the narration must trace to the signal packet.

Deterministic anti-hallucination check. The LLM is handed a packet of pre-computed
numbers and told to use only those; this verifies it and catches any invented figure
before the narration is shown. It is not a semantic check -- purely "did every number
in the text come from the packet" -- which is exactly the guarantee we need: the LLM
frames and explains, but fabricates no data.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_FORM = re.compile(r"10-[KQ]s?\b")  # strip form types ("10-K", "10-Ks") — not numbers
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def extract_numbers(text: str) -> list[float]:
    """All numerals in ``text``. ISO dates decompose into POSITIVE (year, month, day)
    components — never signed fragments like -03 — so a date field can't whitelist
    fabricated negatives, and a re-formatted date ("March 31, 2026") still traces."""
    text = _FORM.sub("", text)
    out: list[float] = []

    def _date_parts(m: re.Match) -> str:
        out.extend((float(m.group(1)), float(int(m.group(2))), float(int(m.group(3)))))
        return " "

    text = _DATE.sub(_date_parts, text)
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _walk_numbers(obj) -> list[float]:
    nums: list[float] = []
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, bool):
            continue
        if isinstance(x, (int, float)):
            nums.append(float(x))
        elif isinstance(x, str):
            nums.extend(extract_numbers(x))
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    return nums


def allowed_numbers(packet: dict) -> set[float]:
    return {round(n, 4) for n in _walk_numbers(packet)}


def check_grounding(text: str, packet: dict, tol: float = 0.02) -> list[float]:
    """Return numbers in ``text`` that don't match any number in the packet (hallucinations).

    Matching is strict: integer packet values (years, percentiles, deciles, counts,
    identifiers) must match EXACTLY; fractional values allow only the small absolute
    ``tol`` for display rounding. No relative tolerance — a magnitude-scaled window
    around large numbers (a CIK, a year) would let fabrications hide near them
    (e.g. cik 886158 would bless any number within ±4,431 at 0.5% relative).
    """
    allowed = _walk_numbers(packet)
    violations = []
    for n in extract_numbers(text):
        ok = any(
            n == a if float(a).is_integer() else abs(n - a) <= tol
            for a in allowed
        )
        if not ok:
            violations.append(n)
    return violations


def is_grounded(text: str, packet: dict, **kwargs) -> bool:
    return not check_grounding(text, packet, **kwargs)
