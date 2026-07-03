"""Structured extraction over a headline + summary (LIVE-VIEW ONLY, never a feature).

An article's raw title+summary is the ground truth; an *extraction* is a derived,
VERSIONED, regenerable read of it: ``{event_type, entities, keywords, takeaway,
sentiment, materiality}``. Two integrity rules, both enforced here deterministically:

1. An extraction may NOT assert a number that isn't in the raw article — the same
   grounding numeral-check the narrator uses, applied to the takeaway. A fabricated
   figure fails validation → one retry with feedback → deterministic (number-free,
   heuristic) fallback. The store never holds a hallucinated number.
2. The takeaway that feeds the narration packet is number-free by construction (the
   packet sanitizer strips it again — belt and suspenders), so news can never smuggle
   a figure past the narrator's grounding guard.

Runs on the same local Ollama tiers as narration; ``llm=None`` yields the heuristic
fallback so the store is always populated (the ``--no-llm`` ops path, and tests).
"""

from __future__ import annotations

import json
import re

from ..narrate.ground import extract_numbers

# Bump when the prompt/schema contract changes so a re-extract is a new, distinct row.
EXTRACT_VERSION = "v1"

EVENT_TYPES = ("earnings", "M&A", "guidance", "litigation", "exec", "product", "other")
SENTIMENTS = ("positive", "neutral", "negative")

EXTRACT_SYSTEM = (
    "You classify a single financial news item from its HEADLINE and SUMMARY only. "
    "Respond with ONLY a JSON object, no code fences, exactly:\n"
    '{"event_type": "earnings|M&A|guidance|litigation|exec|product|other",\n'
    ' "entities": ["<company/person/ticker named in the item>", ...],\n'
    ' "keywords": ["<3-6 salient lowercase keywords>"],\n'
    ' "takeaway": "<one plain-English sentence, NO numbers/figures at all>",\n'
    ' "sentiment": "positive|neutral|negative",\n'
    ' "materiality": <0.0-1.0 — how likely this moves the fundamental thesis>}\n'
    "HARD RULES:\n"
    "- The takeaway MUST contain no numerals — describe magnitude in words ('raised "
    "guidance', 'a large acquisition'), never a figure. Do not invent anything not "
    "in the headline/summary.\n"
    "- materiality: routine press-wire noise / listicles ~0.1-0.3; a real earnings, "
    "M&A, guidance, litigation, or executive-change item ~0.5-0.9.\n"
    "- Output the JSON object and nothing else."
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_WORD = re.compile(r"[a-z][a-z']{2,}")
_STOP = frozenset(
    "the a an and or of to in on for with at by from as is are was be it its this that "
    "into over after amid say says said new inc corp co ltd plc group holdings".split()
)

_EVENT_CUES = [
    ("M&A", ("acquire", "acquisition", "merger", "buyout", "takeover", "to buy",
             "deal", "stake", "combine")),
    ("earnings", ("earnings", "eps", "revenue", "profit", "quarter", "results",
                  "beats", "misses", "loss")),
    ("guidance", ("guidance", "forecast", "outlook", "raises", "cuts", "warns",
                  "lowers", "hikes")),
    ("litigation", ("lawsuit", "sues", "sued", "court", "settlement", "probe",
                    "investigation", "charges", "fine", "antitrust")),
    ("exec", ("ceo", "cfo", "resign", "appoint", "names", "steps down", "chief",
              "director", "hires")),
    ("product", ("launch", "unveil", "release", "product", "recall", "approval",
                 "rollout")),
]
_POS = ("beats", "raises", "surge", "soar", "jump", "gains", "record", "wins",
        "upgrade", "approval", "tops")
_NEG = ("misses", "cuts", "falls", "plunge", "probe", "lawsuit", "warns", "recall",
        "downgrade", "loss", "slump", "fraud", "halts")
_EVENT_BASE = {"M&A": 0.8, "earnings": 0.6, "guidance": 0.6, "litigation": 0.6,
               "exec": 0.5, "product": 0.4, "other": 0.25}


def _article_numbers(article: dict) -> set[float]:
    text = f"{article.get('title', '')} {article.get('summary', '')}"
    return {round(n, 4) for n in extract_numbers(text)}


def _keywords(title: str, k: int = 5) -> list[str]:
    seen, out = set(), []
    for w in _WORD.findall(title.lower()):
        if w in _STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= k:
            break
    return out


def heuristic_extraction(article: dict) -> dict:
    """Deterministic, number-free extraction — the no-LLM fallback (always valid)."""
    from ..narrate.packet import _strip_numbers

    title = article.get("title") or ""
    low = f"{title} {article.get('summary', '')}".lower()
    event = "other"
    for ev, cues in _EVENT_CUES:
        if any(c in low for c in cues):
            event = ev
            break
    sentiment = ("positive" if any(w in low for w in _POS) else
                 "negative" if any(w in low for w in _NEG) else "neutral")
    strong = any(w in low for w in _POS + _NEG)
    materiality = min(1.0, _EVENT_BASE[event] + (0.1 if strong else 0.0))
    return {
        "event_type": event,
        "entities": [],
        "keywords": _keywords(title),
        "takeaway": _strip_numbers(title)[:160],
        "sentiment": sentiment,
        "materiality": round(materiality, 2),
        "model": "heuristic",
    }


def _parse_json(text: str) -> dict | None:
    """First brace-balanced JSON object carrying a ``takeaway`` (tolerates prose/fences)."""
    cleaned = _FENCE.sub("", (text or "").strip())
    pos = 0
    while (start := cleaned.find("{", pos)) >= 0:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(cleaned)):
            c = cleaned[j]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:j + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "takeaway" in obj:
                        return obj
                    break
        pos = start + 1
    return None


def shape_extraction(parsed: dict | None, article: dict, model: str) -> tuple[dict, list]:
    """Coerce + validate an LLM extraction. Returns (extraction, violations).

    Violations (non-empty => reject): unparseable, empty takeaway, or a numeral in
    the takeaway that does not appear in the raw article (the fabrication guard)."""
    if not isinstance(parsed, dict):
        return {}, ["unparseable-json"]
    takeaway = str(parsed.get("takeaway") or "").strip()
    if not takeaway:
        return {}, ["missing-takeaway"]
    violations: list = []
    allowed = _article_numbers(article)
    for n in extract_numbers(takeaway):
        if not any(n == a if float(a).is_integer() else abs(n - a) <= 0.02 for a in allowed):
            violations.append(f"fabricated-number:{n}")

    event = str(parsed.get("event_type") or "other")
    event = event if event in EVENT_TYPES else "other"
    sentiment = str(parsed.get("sentiment") or "neutral")
    sentiment = sentiment if sentiment in SENTIMENTS else "neutral"
    try:
        materiality = max(0.0, min(1.0, float(parsed.get("materiality"))))
    except (TypeError, ValueError):
        materiality = 0.3

    def _strlist(x, cap):
        if not isinstance(x, list):
            return []
        return [str(v).strip() for v in x if str(v).strip()][:cap]

    extraction = {
        "event_type": event,
        "entities": _strlist(parsed.get("entities"), 8),
        "keywords": _strlist(parsed.get("keywords"), 8),
        "takeaway": takeaway[:160],
        "sentiment": sentiment,
        "materiality": round(materiality, 3),
        "model": model,
    }
    return extraction, violations


def extract_article(article: dict, llm=None, model: str = "", max_retries: int = 1) -> dict:
    """Extract one article. LLM path with a numeral guard + one feedback retry, then a
    deterministic number-free fallback. ``llm`` is a callable(system, user) -> str."""
    if llm is None:
        return heuristic_extraction(article)
    payload = json.dumps({"title": article.get("title", ""),
                          "summary": article.get("summary", "")}, ensure_ascii=False)
    model = model or getattr(llm, "model", "llm")
    last: list = []
    for attempt in range(max_retries + 1):
        prompt = payload if not last else (
            payload + "\n\nYour previous reply was REJECTED for: "
            + json.dumps([str(v) for v in last]) + ". Fix exactly these and resend the JSON.")
        try:
            text = llm(EXTRACT_SYSTEM, prompt)
        except Exception as exc:  # transport/timeout: fall back, never crash ingestion
            last = [f"llm-error:{type(exc).__name__}"]
            break
        extraction, violations = shape_extraction(_parse_json(text), article, model)
        if not violations:
            return extraction
        last = violations
    fallback = heuristic_extraction(article)
    fallback["fallback_from"] = last
    return fallback
