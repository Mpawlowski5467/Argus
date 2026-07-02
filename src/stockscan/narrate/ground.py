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
_FORM = re.compile(r"10-[KQ]\b")  # strip form types so "10-K" doesn't read as the number 10


def extract_numbers(text: str) -> list[float]:
    text = _FORM.sub("", text)
    out = []
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
    """Return numbers in ``text`` that don't match any number in the packet (hallucinations)."""
    allowed = _walk_numbers(packet)
    violations = []
    for n in extract_numbers(text):
        if not any(abs(n - a) <= tol + 0.005 * abs(a) for a in allowed):
            violations.append(n)
    return violations


def is_grounded(text: str, packet: dict, **kwargs) -> bool:
    return not check_grounding(text, packet, **kwargs)
