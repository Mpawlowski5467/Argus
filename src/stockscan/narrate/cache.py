"""Materiality-gated, cached narration (DESIGN.md §7).

Narration is the expensive stage (~30-90s per name on a 27B local model), so it
only runs when the underlying facts changed enough to matter:

- UNCHANGED packet (same hash)             -> serve the cached narration.
- MINOR change (numbers wiggled, same story) -> the light 14B-class tier.
- MATERIAL change                            -> the full 27B-class tier.

Material means: a new filing (period_end changed), the model percentile moved by
``percentile_threshold`` or more, or the top model drivers changed — the things a
reader would actually want re-explained. State lives in SQLite (the project's
mutable-state store); the cache is an optimization, never a correctness layer:
whatever tier runs, the result still passes the full grounding + citation
validation in narrator.narrate_packet.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import ARTIFACTS_DIR
from .narrator import narrate_packet

CACHE_PATH = ARTIFACTS_DIR / "narration_cache.sqlite"
PERCENTILE_THRESHOLD = 10


VOLATILE_META = ("as_of",)
VOLATILE_MODEL = ("as_of", "score", "n_names", "percentile", "decile")


def packet_hash(packet: dict) -> str:
    """Hash of the packet's DURABLE content. Volatile per-query fields (as-of
    stamps, the exact score, cross-section size) are excluded — otherwise every
    daily run would look 'changed' and the cache could never hit. Percentile/driver
    drift is judged separately (and first) by :func:`materiality`."""
    p = json.loads(json.dumps(packet, sort_keys=True, default=str))
    for k in VOLATILE_META:
        p.get("meta", {}).pop(k, None)
    for k in VOLATILE_MODEL:
        p.get("model", {}).pop(k, None)
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def _top_drivers(packet: dict, k: int = 3) -> list[str]:
    return [d["id"] for d in packet.get("model", {}).get("drivers", [])[:k]]


class NarrationCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = Path(path)
        # WAL + a generous busy timeout: the nightly monitor holds this DB for
        # minutes (LLM latency) while a manual analyze/scan may write concurrently.
        self._db = sqlite3.connect(str(self.path), timeout=30.0)
        self._db.execute("pragma journal_mode=wal")
        self._db.execute(
            """create table if not exists narrations (
                cik integer primary key,
                packet_hash text not null,
                period_end text,
                percentile integer,
                drivers text,
                result_json text not null,
                model_tag text,
                updated text not null
            )"""
        )
        self._db.commit()

    def get(self, cik: int) -> dict | None:
        row = self._db.execute(
            "select packet_hash, period_end, percentile, drivers, result_json, model_tag "
            "from narrations where cik = ?", (int(cik),)
        ).fetchone()
        if row is None:
            return None
        return {
            "packet_hash": row[0], "period_end": row[1], "percentile": row[2],
            "drivers": json.loads(row[3] or "[]"), "result": json.loads(row[4]),
            "model_tag": row[5],
        }

    def put(self, cik: int, packet: dict, result: dict, model_tag: str = "",
            baseline: bool = True) -> None:
        """Store a narration. ``baseline=False`` (light-tier refresh) keeps the LAST
        FULL narration's facts as the materiality baseline — otherwise successive
        minor drifts would each be compared to the previous minor state and a slow
        large move could ratchet past the material threshold unnoticed."""
        stored = {k: v for k, v in result.items() if k != "packet"}
        prev = self.get(int(cik)) if not baseline else None
        base = prev if prev is not None else {
            "period_end": packet["meta"].get("period_end"),
            "percentile": packet.get("model", {}).get("percentile"),
            "drivers": _top_drivers(packet),
        }
        self._db.execute(
            "insert or replace into narrations values (?,?,?,?,?,?,?,?)",
            (
                int(cik),
                packet_hash(packet),
                base["period_end"],
                base["percentile"],
                json.dumps(base["drivers"]),
                json.dumps(stored, default=str),
                model_tag,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


def materiality(prev: dict | None, packet: dict,
                percentile_threshold: int = PERCENTILE_THRESHOLD) -> str:
    """Classify the change since the cached narration: unchanged | minor | material.

    Material conditions are checked FIRST — an identical durable hash must never
    short-circuit past a large percentile move (percentile is deliberately not part
    of the hash)."""
    if prev is None:
        return "material"
    if prev.get("period_end") != packet["meta"].get("period_end"):
        return "material"  # a new filing is always worth a fresh full read
    new_pct = packet.get("model", {}).get("percentile")
    old_pct = prev.get("percentile")
    if new_pct is not None and old_pct is not None \
            and abs(new_pct - old_pct) >= percentile_threshold:
        return "material"
    if _top_drivers(packet) != prev.get("drivers"):
        return "material"
    if prev["packet_hash"] == packet_hash(packet):
        return "unchanged"
    return "minor"


def narrate_smart(
    packet: dict,
    llm_full=None,
    llm_light=None,
    cache: NarrationCache | None = None,
    max_retries: int = 1,
) -> dict:
    """Cache-aware, tiered narration. Returns the narrate_packet result plus
    ``tier`` ("cache" | "light" | "full" | "template") and ``materiality``."""
    cik = packet["meta"]["cik"]
    prev = cache.get(cik) if cache is not None else None
    change = materiality(prev, packet)

    if change == "unchanged" and prev is not None:
        return {**prev["result"], "packet": packet, "tier": "cache",
                "materiality": change}

    llm = llm_light if (change == "minor" and llm_light is not None) else llm_full
    result = narrate_packet(packet, llm=llm, max_retries=max_retries)
    result["tier"] = "template" if llm is None else \
        ("light" if llm is llm_light and change == "minor" else "full")
    result["materiality"] = change
    # A template run must never enter the cache: a no-LLM invocation (scheduled
    # monitor with --no-llm, LLM endpoint down) would otherwise overwrite a cached
    # full-tier narration AND reset the materiality baseline, after which the next
    # LLM-enabled run can classify "unchanged" and serve the template forever.
    if cache is not None and result["tier"] != "template":
        # only a full-quality narration resets the materiality baseline
        cache.put(cik, packet, result, model_tag=result["tier"],
                  baseline=result["tier"] != "light")
    return result
