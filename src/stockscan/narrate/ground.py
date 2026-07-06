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
# A day/month slot is RANGE-BOUNDED so only a real calendar value is ever mistaken for a
# date component. This is the crux of the guard's safety: an out-of-range figure — "March
# 99", "the 45th", "95/5/2025" — is NOT a date, so it is left in the text to be grounded
# (a fabricated 32-99 can no longer hide in a date's day slot). An IN-range value disguised
# as a real date ("March 25" / "25/12/2025") stays indistinguishable from a date and is
# stripped — the same deliberate, pre-existing tradeoff (you cannot tell "the 25th" from a
# fabricated 25), now bounded to 1-31 instead of 0-99.
_DAY = r"(?:3[01]|[12]\d|0?[1-9])"   # 1-31
_MON = r"(?:1[0-2]|0?[1-9])"         # 1-12
# slash dates anchored on a 4-digit YEAR, month-first ("07/01/2026" / "2026/07/01").
# The 4-digit year rules out a bare ratio ("1/2"); the bounded month/day slots rule out a
# fabricated out-of-range pair ("95/5/2025", "99/99/2025") — those no longer read as a
# date, so they still have to ground. Only the year survives as a traceable numeral.
_SLASH_DATE = re.compile(rf"\b(?:(\d{{4}})/{_MON}/{_DAY}|{_MON}/{_DAY}/(\d{{4}}))\b")
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
           "November|December")
_ORD = r"(?:st|nd|rd|th)"  # ordinal day suffix ("1st", "31st")
# natural-language date forms. A MONTH NAME anchors every form, and a bounded day number
# (1-31) is only stripped when a YEAR or an explicit ORDINAL disambiguates it as a date:
#   "March 31, 2026" / "July 1st, 2026" / "31 March 2026" / "1st of July 2026"
#   "July 1st" / "1st July"  (ordinal, no year)         "March 2026"  (month + year)
# A lone "97th" (a percentile) has no month; a bare "June 30" (no year, no ordinal) is
# ambiguous with "June" + a figure; and an out-of-range "March 45" / "the 45th" is not a
# date at all — all are left to ground rather than risk blessing a fabricated number.
_TEXT_DATE = re.compile(
    rf"\b(?:"
    rf"(?:{_MONTHS})\s+{_DAY}{_ORD}?,?\s+\d{{4}}"
    rf"|{_DAY}{_ORD}?\s+(?:of\s+)?(?:{_MONTHS})\s+\d{{4}}"
    rf"|(?:{_MONTHS})\s+{_DAY}{_ORD}"
    rf"|{_DAY}{_ORD}\s+(?:of\s+)?(?:{_MONTHS})"
    rf"|(?:{_MONTHS})\s+\d{{4}}"
    rf")\b",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[float]:
    """All numerals in ``text``, with DATES removed first.

    Dates (ISO, slash, or natural language — including ordinal "July 1st, 2026" and
    slash "07/01/2026" forms) are stripped from both sides rather than decomposed:
    whitelisting a date's day/month as bare integers would bless fabricated figures
    like "up 12%" or "31% share" for any Dec-31 filer. Only the YEAR survives as a
    traceable numeral (so "fiscal 2025" still grounds; a bare fabricated "31" no
    longer does). Stripping is deliberately conservative — a month name plus a
    year or an explicit ordinal — so a lone "97th" or a bare "June 30" still has to
    trace to the context."""
    text = _FORM.sub("", text)
    out: list[float] = []

    def _keep_year_iso(m: re.Match) -> str:
        out.append(float(m.group(1)))
        return " "

    def _keep_year_slash(m: re.Match) -> str:
        y = m.group(1) or m.group(2)
        if y:
            out.append(float(y))
        return " "

    def _keep_year_text(m: re.Match) -> str:
        y = re.search(r"\b(\d{4})\b", m.group(0))
        if y:
            out.append(float(y.group(1)))
        return " "

    text = _DATE.sub(_keep_year_iso, text)
    text = _SLASH_DATE.sub(_keep_year_slash, text)
    text = _TEXT_DATE.sub(_keep_year_text, text)
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
