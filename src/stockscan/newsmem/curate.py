"""'Good article' curation: keep material, credible, non-duplicate news; drop wire spam.

A pure, testable filter over recalled rows. Storage keeps EVERYTHING (the raw article
is ground truth, deduped by Intrinio id); curation decides what is worth surfacing to
the narration/TUI. Three gates:

- materiality: the extraction's 0-1 score must clear a floor.
- source credibility: reputable financial press outranks press-wire/listicle spam; a
  low-credibility source is dropped UNLESS the item is decisively material (a real 8-K
  release crossing the wire still matters).
- dedup: wire services repost identical headlines — keep one per normalized title
  (the most material, then most recent).
"""

from __future__ import annotations

import re

from ..config import NEWS_CREDIBILITY_FLOOR, NEWS_MATERIALITY_FLOOR

# Publisher-host credibility. Unknown hosts get a neutral default; press-wire and
# known listicle/aggregator spam are down-weighted. Not exhaustive — a lookup, not
# a signal (news never enters the model, so this only shapes what a human reader sees).
SOURCE_CREDIBILITY = {
    "reuters.com": 1.0, "bloomberg.com": 1.0, "wsj.com": 1.0, "ft.com": 1.0,
    "apnews.com": 0.95, "nytimes.com": 0.9, "cnbc.com": 0.9, "barrons.com": 0.9,
    "economist.com": 0.9, "washingtonpost.com": 0.85,
    "marketwatch.com": 0.75, "forbes.com": 0.7, "businessinsider.com": 0.7,
    "investors.com": 0.75, "theinformation.com": 0.85, "axios.com": 0.8,
    "finance.yahoo.com": 0.6, "yahoo.com": 0.6, "fool.com": 0.5, "seekingalpha.com": 0.55,
    # press-wire: real releases, but a firehose of promotional noise
    "businesswire.com": 0.4, "prnewswire.com": 0.4, "globenewswire.com": 0.4,
    "accesswire.com": 0.35, "newsfilecorp.com": 0.35, "prweb.com": 0.3,
    # listicle / low-signal
    "247wallst.com": 0.2, "benzinga.com": 0.35, "zacks.com": 0.4, "insidermonkey.com": 0.25,
}
DEFAULT_CREDIBILITY = 0.5
DECISIVE_MATERIALITY = 0.7   # this material => keep even a low-credibility source

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def credibility(source: str) -> float:
    return SOURCE_CREDIBILITY.get((source or "").lower().strip(), DEFAULT_CREDIBILITY)


def dedup_key(title: str) -> str:
    """Normalized headline for near-duplicate detection (wire reposts collide here)."""
    t = _PUNCT.sub(" ", (title or "").lower())
    return _WS.sub(" ", t).strip()


def is_good(row: dict, materiality_floor: float = NEWS_MATERIALITY_FLOOR,
            credibility_floor: float = NEWS_CREDIBILITY_FLOOR) -> bool:
    mat = float(row.get("materiality") or 0.0)
    if mat < materiality_floor:
        return False
    cred = credibility(row.get("source"))
    return cred >= credibility_floor or mat >= DECISIVE_MATERIALITY


def curate(rows: list[dict], materiality_floor: float = NEWS_MATERIALITY_FLOOR,
           credibility_floor: float = NEWS_CREDIBILITY_FLOOR) -> list[dict]:
    """Filter to good, de-duplicated rows (highest materiality per headline, then newest)."""
    kept: dict[str, dict] = {}
    for r in rows:
        if not is_good(r, materiality_floor, credibility_floor):
            continue
        key = dedup_key(r.get("title", ""))
        cur = kept.get(key)
        if cur is None or _rank(r) > _rank(cur):
            kept[key] = r
    return sorted(kept.values(), key=_rank, reverse=True)


def _rank(r: dict) -> tuple:
    return (float(r.get("materiality") or 0.0), str(r.get("date") or ""))
