"""Mutable operational state (SQLite): job runs, watchlist, alerts, paper book.

One small database (config.OPS_STATE_PATH) holds everything the continuous loop
needs to remember between runs. WAL mode + a busy timeout make it safe for a
launchd job and a manual CLI invocation to touch the store at the same time; the
coarse per-job serialization lives in ops.lock, not here.

The job_runs table is the idempotency evidence trail: every scheduled job appends
one row with a JSON summary of what it actually changed ("deltas"), so a no-op
re-run is visibly a no-op.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import OPS_STATE_PATH

_SCHEMA = """
create table if not exists job_runs (
    id integer primary key autoincrement,
    job text not null,
    started text not null,
    finished text,
    status text not null default 'running',   -- running | ok | failed | noop
    deltas text                                -- JSON summary of what changed
);
create table if not exists watchlist (
    cik integer primary key,
    column text,
    note text,
    added text not null,
    active integer not null default 1
);
create table if not exists signal_state (
    cik integer primary key,
    percentile integer,
    decile integer,
    as_of text,
    updated text
);
create table if not exists known_filings (
    cik integer not null,
    form text not null,
    filed_date text not null,
    period_end text not null default '',       -- '' until FSDS delivers the numbers
    source text not null,                      -- 'fsds' | 'edgar'
    first_seen text not null,
    -- period_end is in the key: a delinquent filer catching up files several
    -- same-form docs for different periods on ONE day; each is real news.
    primary key (cik, form, period_end, filed_date, source)
);
create table if not exists alerts (
    id integer primary key autoincrement,
    created text not null,
    cik integer,
    kind text not null,
    message text not null,
    payload text,
    seen integer not null default 0
);
create table if not exists book (
    cik integer primary key,
    column text,
    entered_as_of text,
    exited_as_of text,
    active integer not null default 1
);
create table if not exists positions (
    -- PERSONAL holdings the user records to see value & P/L. Live-view display data:
    -- read back to the user only, NEVER an input to the score / paper book / backtest.
    cik integer primary key,
    shares real not null,
    cost_basis real not null,
    added_at text not null
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class OpsState:
    def __init__(self, path: Path = OPS_STATE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), timeout=30.0)
        self._db.execute("pragma journal_mode=wal")
        self._db.executescript(_SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        """Additive column migrations (create-table-if-not-exists can't add columns)."""
        cols = {r[1] for r in self._db.execute("pragma table_info(signal_state)")}
        if "distress" not in cols:   # FIREWALLED distress prob, for escalation alerts
            self._db.execute("alter table signal_state add column distress real")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "OpsState":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- job runs ----------------------------------------------------------------
    def job_start(self, job: str) -> int:
        cur = self._db.execute(
            "insert into job_runs (job, started) values (?, ?)", (job, _utcnow())
        )
        self._db.commit()
        return int(cur.lastrowid)

    def job_finish(self, run_id: int, status: str, deltas: dict | None = None) -> None:
        self._db.execute(
            "update job_runs set finished = ?, status = ?, deltas = ? where id = ?",
            (_utcnow(), status, json.dumps(deltas or {}, default=str), run_id),
        )
        self._db.commit()

    def last_run(self, job: str, status: str | None = None) -> dict | None:
        q = "select id, job, started, finished, status, deltas from job_runs where job = ?"
        args: list = [job]
        if status is not None:
            q += " and status = ?"
            args.append(status)
        row = self._db.execute(q + " order by id desc limit 1", args).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "job": row[1], "started": row[2], "finished": row[3],
            "status": row[4], "deltas": json.loads(row[5] or "{}"),
        }

    # -- watchlist -----------------------------------------------------------------
    def watch_add(self, cik: int, column: str | None = None, note: str = "") -> None:
        self._db.execute(
            "insert into watchlist (cik, column, note, added, active) values (?,?,?,?,1) "
            "on conflict(cik) do update set active = 1, column = excluded.column, "
            "note = excluded.note",
            (int(cik), column, note, _utcnow()),
        )
        self._db.commit()

    def watch_remove(self, cik: int) -> None:
        self._db.execute("update watchlist set active = 0 where cik = ?", (int(cik),))
        self._db.commit()

    def watchlist(self) -> list[dict]:
        rows = self._db.execute(
            "select cik, column, note, added from watchlist where active = 1 order by added"
        ).fetchall()
        return [{"cik": r[0], "column": r[1], "note": r[2], "added": r[3]} for r in rows]

    # -- positions (PERSONAL holdings — DISPLAY-ONLY live-view; NEVER a signal input) --
    def position_set(self, cik: int, shares: float, cost_basis: float) -> None:
        """Upsert the user's holding (add or update in one call). ``cost_basis`` is
        personal live-view data: it is stored to show value & P/L back to the user and
        is never read into the score, the paper book, or the backtest. On-conflict keeps
        the original ``added_at`` (mirrors ``watch_add`` preserving ``added``)."""
        self._db.execute(
            "insert into positions (cik, shares, cost_basis, added_at) values (?,?,?,?) "
            "on conflict(cik) do update set shares = excluded.shares, "
            "cost_basis = excluded.cost_basis",
            (int(cik), float(shares), float(cost_basis), _utcnow()),
        )
        self._db.commit()

    def position_remove(self, cik: int) -> None:
        self._db.execute("delete from positions where cik = ?", (int(cik),))
        self._db.commit()

    def positions(self) -> list[dict]:
        rows = self._db.execute(
            "select cik, shares, cost_basis, added_at from positions order by added_at"
        ).fetchall()
        return [
            {"cik": r[0], "shares": r[1], "cost_basis": r[2], "added_at": r[3]} for r in rows
        ]

    # -- signal state (percentile-move detection) -----------------------------------
    def get_signal(self, cik: int) -> dict | None:
        row = self._db.execute(
            "select percentile, decile, as_of, distress from signal_state where cik = ?",
            (int(cik),),
        ).fetchone()
        if row is None:
            return None
        return {"percentile": row[0], "decile": row[1], "as_of": row[2], "distress": row[3]}

    def record_signal(self, cik: int, percentile: int, decile: int, as_of: str,
                      distress: float | None = None) -> None:
        """Persist the latest signal. ``distress`` (FIREWALLED risk-flag prob) is stored
        only so the monitor can alert on ESCALATION; it is never a trade input."""
        self._db.execute(
            "insert into signal_state (cik, percentile, decile, as_of, updated, distress) "
            "values (?,?,?,?,?,?) on conflict(cik) do update set "
            "percentile = excluded.percentile, decile = excluded.decile, "
            "as_of = excluded.as_of, updated = excluded.updated, distress = excluded.distress",
            (int(cik), int(percentile), int(decile), str(as_of), _utcnow(),
             float(distress) if distress is not None else None),
        )
        self._db.commit()

    # -- filings ---------------------------------------------------------------------
    def add_filings(self, rows: list[dict]) -> list[dict]:
        """Insert filings, returning only the ones not seen before (the news).

        Each row: {cik, form, filed_date, period_end?, source}. Idempotent: replaying
        the same rows returns [].
        """
        new: list[dict] = []
        for r in rows:
            cur = self._db.execute(
                "insert or ignore into known_filings "
                "(cik, form, filed_date, period_end, source, first_seen) "
                "values (?,?,?,?,?,?)",
                (int(r["cik"]), r["form"], str(r["filed_date"]),
                 str(r.get("period_end") or ""), r["source"], _utcnow()),
            )
            if cur.rowcount:
                new.append(r)
        self._db.commit()
        return new

    def has_filings(self, cik: int, source: str | None = None) -> bool:
        q = "select 1 from known_filings where cik = ?"
        args: list = [int(cik)]
        if source is not None:
            q += " and source = ?"
            args.append(source)
        return self._db.execute(q + " limit 1", args).fetchone() is not None

    def latest_filing_date(self, cik: int) -> str | None:
        row = self._db.execute(
            "select max(filed_date) from known_filings where cik = ?", (int(cik),)
        ).fetchone()
        return row[0]

    # -- alerts ------------------------------------------------------------------------
    def add_alert(self, kind: str, message: str, cik: int | None = None,
                  payload: dict | None = None) -> int:
        cur = self._db.execute(
            "insert into alerts (created, cik, kind, message, payload) values (?,?,?,?,?)",
            (_utcnow(), cik, kind, message, json.dumps(payload or {}, default=str)),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def alerts(self, unseen_only: bool = True, limit: int = 100) -> list[dict]:
        q = ("select id, created, cik, kind, message, payload, seen from alerts "
             + ("where seen = 0 " if unseen_only else "")
             + "order by id desc limit ?")
        rows = self._db.execute(q, (limit,)).fetchall()
        return [
            {"id": r[0], "created": r[1], "cik": r[2], "kind": r[3], "message": r[4],
             "payload": json.loads(r[5] or "{}"), "seen": bool(r[6])}
            for r in rows
        ]

    def mark_alerts_seen(self, ids: list[int] | None = None) -> int:
        if ids is None:
            cur = self._db.execute("update alerts set seen = 1 where seen = 0")
        else:
            cur = self._db.execute(
                f"update alerts set seen = 1 where id in ({','.join('?' * len(ids))})", ids
            )
        self._db.commit()
        return cur.rowcount

    # -- hysteresis paper book -----------------------------------------------------------
    def book(self) -> dict[int, dict]:
        rows = self._db.execute(
            "select cik, column, entered_as_of from book where active = 1"
        ).fetchall()
        return {r[0]: {"column": r[1], "entered_as_of": r[2]} for r in rows}

    def book_apply(self, enters: dict[int, str], exits: set[int], as_of: str) -> None:
        """Apply one rebalance's membership changes. ``enters`` maps cik -> column."""
        for cik, column in enters.items():
            self._db.execute(
                "insert into book (cik, column, entered_as_of, active) values (?,?,?,1) "
                "on conflict(cik) do update set active = 1, column = excluded.column, "
                "entered_as_of = excluded.entered_as_of, exited_as_of = null",
                (int(cik), column, str(as_of)),
            )
        for cik in exits:
            self._db.execute(
                "update book set active = 0, exited_as_of = ? where cik = ?",
                (str(as_of), int(cik)),
            )
        self._db.commit()
