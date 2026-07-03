"""The news-memory store (SQLite): raw articles + versioned extractions + fetch log.

Mirrors the project's other mutable stores (ops_state, narration_cache): WAL + a busy
timeout for safe concurrent launchd/CLI/TUI access, idempotent upsert, everything
timestamped. FIREWALL: this database is read by narration + the TUI ONLY. It is never
joined into the panel, never a feature, never scored. Timestamps exist so it COULD be
made point-in-time later, not because it is used point-in-time now.

- ``articles``: one row per Intrinio article id (the dedup key). The raw title+summary
  is immutable ground truth — ``insert or ignore`` never overwrites it.
- ``extractions``: derived, regenerable reads keyed (article_id, version); a newer run
  of the same version replaces its row (``insert or replace``).
- ``fetches``: per-company last-fetch stamp so the network re-pull is quota-capped.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import NEWS_DB_PATH
from .extract import EXTRACT_VERSION

_SCHEMA = """
create table if not exists articles (
    id text primary key,
    cik integer,
    ticker text,
    publication_date text not null,     -- full ISO stamp (point-in-time honest)
    source text,
    title text not null,
    summary text not null default '',   -- RAW ground truth, never mutated
    url text,
    first_seen text not null            -- when WE ingested it
);
create table if not exists extractions (
    article_id text not null,
    version text not null,
    event_type text,
    entities text,                      -- JSON list
    keywords text,                      -- JSON list
    takeaway text,
    sentiment text,
    materiality real,
    model text,
    created text not null,
    primary key (article_id, version)
);
create table if not exists fetches (
    cik integer primary key,
    ticker text,
    last_fetched text not null,
    last_new integer not null default 0
);
create index if not exists idx_articles_cik on articles(cik, publication_date);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NewsStore:
    def __init__(self, path: Path = NEWS_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), timeout=30.0)
        self._db.execute("pragma journal_mode=wal")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "NewsStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- articles ------------------------------------------------------------------
    def upsert_articles(self, cik: int, ticker: str | None, articles: list[dict]) -> list[dict]:
        """Insert shaped articles (from news.shape_article), skipping known ids.

        Idempotent: replaying the same fetch returns []. The raw article is never
        overwritten (ground truth). Returns the NEW rows (the actual news)."""
        new: list[dict] = []
        for a in articles:
            aid = str(a.get("id") or "").strip()
            if not aid or not (a.get("title") or "").strip():
                continue
            pub = str(a.get("publication_date") or a.get("date") or "")
            cur = self._db.execute(
                "insert or ignore into articles "
                "(id, cik, ticker, publication_date, source, title, summary, url, first_seen) "
                "values (?,?,?,?,?,?,?,?,?)",
                (aid, int(cik) if cik is not None else a.get("cik"), ticker, pub,
                 a.get("source") or "", a.get("title") or "", a.get("summary") or "",
                 a.get("url") or "", _utcnow()),
            )
            if cur.rowcount:
                new.append({**a, "id": aid, "publication_date": pub})
        self._db.commit()
        return new

    def _articles_for(self, cik: int) -> list[dict]:
        rows = self._db.execute(
            "select id, cik, ticker, publication_date, source, title, summary, url "
            "from articles where cik = ?", (int(cik),)
        ).fetchall()
        cols = ("id", "cik", "ticker", "publication_date", "source", "title", "summary", "url")
        return [dict(zip(cols, r)) for r in rows]

    def articles_missing_extraction(self, cik: int, version: str = EXTRACT_VERSION,
                                    upgrade_heuristic: bool = False) -> list[dict]:
        """Articles for ``cik`` needing extraction of ``version`` — the work-list.

        Missing an extraction entirely, or (when ``upgrade_heuristic``) carrying only a
        deterministic 'heuristic' placeholder: a lazy TUI-open extracts heuristically
        (instant), and the nightly LLM job upgrades those placeholders in place. Without
        the flag, an existing extraction (heuristic or LLM) is left untouched — so
        repeated no-LLM opens never churn."""
        cond = "e.article_id is null"
        if upgrade_heuristic:
            cond = "(e.article_id is null or e.model = 'heuristic')"
        rows = self._db.execute(
            "select a.id, a.title, a.summary, a.source, a.publication_date "
            "from articles a left join extractions e "
            "  on e.article_id = a.id and e.version = ? "
            f"where a.cik = ? and {cond}", (version, int(cik))
        ).fetchall()
        cols = ("id", "title", "summary", "source", "publication_date")
        return [dict(zip(cols, r)) for r in rows]

    # -- extractions ----------------------------------------------------------------
    def put_extraction(self, article_id: str, version: str, extraction: dict) -> None:
        self._db.execute(
            "insert or replace into extractions "
            "(article_id, version, event_type, entities, keywords, takeaway, sentiment, "
            " materiality, model, created) values (?,?,?,?,?,?,?,?,?,?)",
            (str(article_id), version, extraction.get("event_type"),
             json.dumps(extraction.get("entities") or []),
             json.dumps(extraction.get("keywords") or []),
             extraction.get("takeaway"), extraction.get("sentiment"),
             float(extraction.get("materiality") or 0.0), extraction.get("model"), _utcnow()),
        )
        self._db.commit()

    def get_extraction(self, article_id: str, version: str = EXTRACT_VERSION) -> dict | None:
        row = self._db.execute(
            "select event_type, entities, keywords, takeaway, sentiment, materiality, model "
            "from extractions where article_id = ? and version = ?",
            (str(article_id), version)).fetchone()
        if row is None:
            return None
        return {"event_type": row[0], "entities": json.loads(row[1] or "[]"),
                "keywords": json.loads(row[2] or "[]"), "takeaway": row[3],
                "sentiment": row[4], "materiality": row[5], "model": row[6]}

    # -- recall (keyword + structured query) ---------------------------------------
    def recall(self, cik: int, since: str | None = None, event_types=None,
               keywords: str | None = None, limit: int = 8,
               version: str = EXTRACT_VERSION, min_materiality: float | None = None) -> list[dict]:
        """Past material articles for a company. Structured (event_type/date) + keyword.

        Joined to the ``version`` extraction; ordered by materiality then recency.
        Returns merged rows the packet/TUI can use directly."""
        q = [
            "select a.id, a.cik, a.ticker, a.publication_date, a.source, a.title, a.summary,",
            "       a.url, e.event_type, e.entities, e.keywords, e.takeaway, e.sentiment,",
            "       e.materiality, e.model",
            "from articles a left join extractions e",
            "  on e.article_id = a.id and e.version = ?",
            "where a.cik = ?",
        ]
        args: list = [version, int(cik)]
        if since is not None:
            q.append("and a.publication_date >= ?")
            args.append(str(since))
        if event_types:
            ev = list(event_types)
            q.append(f"and e.event_type in ({','.join('?' * len(ev))})")
            args += ev
        if min_materiality is not None:
            q.append("and coalesce(e.materiality, 0) >= ?")
            args.append(float(min_materiality))
        if keywords:
            q.append("and lower(a.title || ' ' || coalesce(a.summary,'') || ' ' || "
                     "coalesce(e.takeaway,'') || ' ' || coalesce(e.keywords,'')) like ?")
            args.append(f"%{str(keywords).lower()}%")
        q.append("order by coalesce(e.materiality, 0) desc, a.publication_date desc limit ?")
        args.append(int(limit))
        rows = self._db.execute("\n".join(q), args).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0], "cik": r[1], "ticker": r[2],
                "publication_date": r[3], "date": (r[3] or "")[:10], "source": r[4],
                "title": r[5], "summary": r[6], "url": r[7],
                "event_type": r[8] or "other",
                "entities": json.loads(r[9] or "[]"), "keywords": json.loads(r[10] or "[]"),
                "takeaway": r[11] or "", "sentiment": r[12],
                "materiality": r[13] if r[13] is not None else 0.0, "model": r[14],
            })
        return out

    def context_for(self, cik: int, recent: int = 3, notable: int = 3,
                    version: str = EXTRACT_VERSION, curate_rows: bool = True) -> list[dict]:
        """The packet's ``context.news``: the newest few + the most-material older few.

        This is how narration "brings up the past" — recent headlines plus notable
        prior events, curated (material, credible, deduped). Number-free takeaways are
        re-sanitized by the packet builder; here we just pick and order."""
        from .curate import curate

        pool = self.recall(cik, limit=200, version=version)
        pool = [r for r in pool if r.get("takeaway")]
        if curate_rows:
            pool = curate(pool)
        by_date = sorted(pool, key=lambda r: str(r.get("date") or ""), reverse=True)
        chosen: dict[str, dict] = {r["id"]: r for r in by_date[:recent]}
        for r in sorted(pool, key=lambda r: float(r.get("materiality") or 0.0), reverse=True):
            if len(chosen) >= recent + notable:
                break
            chosen.setdefault(r["id"], r)
        return sorted(chosen.values(), key=lambda r: str(r.get("date") or ""), reverse=True)

    # -- fetch throttle (hard quota cap) -------------------------------------------
    def last_fetch(self, cik: int) -> dict | None:
        row = self._db.execute(
            "select ticker, last_fetched, last_new from fetches where cik = ?", (int(cik),)
        ).fetchone()
        if row is None:
            return None
        return {"ticker": row[0], "last_fetched": row[1], "last_new": row[2]}

    def record_fetch(self, cik: int, ticker: str | None, n_new: int) -> None:
        self._db.execute(
            "insert into fetches (cik, ticker, last_fetched, last_new) values (?,?,?,?) "
            "on conflict(cik) do update set ticker = excluded.ticker, "
            "last_fetched = excluded.last_fetched, last_new = excluded.last_new",
            (int(cik), ticker, _utcnow(), int(n_new)),
        )
        self._db.commit()
