"""Nightly housekeeping: SQLite backups + log rotation.

The stores under artifacts/ are the only irreplaceable personal state in the whole
system — positions, watchlist, alerts, job history (ops_state.sqlite) — plus caches
that are expensive to rebuild against vendor quotas (news, profiles). Everything
else on disk regenerates from EDGAR/Intrinio. So the nightly copies each store with
SQLite's online backup API (WAL-safe: a consistent snapshot even mid-write) into a
dated folder and prunes old folders. Logs get copy-truncate rotation — safe with
append-mode writers (launchd's StandardOutPath, the web nightly log), which keep
writing at the new EOF after truncation.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from ..config import ARTIFACTS_DIR, BACKUP_KEEP_DAYS, BACKUPS_DIR, LOG_ROTATE_MB, LOGS_DIR


def backup_stores(stores_dir: Path = ARTIFACTS_DIR, out_dir: Path = BACKUPS_DIR,
                  keep: int = BACKUP_KEEP_DAYS, today=None) -> dict:
    """Online-backup every ``*.sqlite`` under ``stores_dir`` into ``out_dir/YYYY-MM-DD/``.

    Idempotent: a re-run the same day overwrites that day's snapshot. Prunes to the
    ``keep`` most recent dated folders. One unreadable store never blocks the rest —
    it lands in ``errors`` and the job reports degraded."""
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    day_dir = Path(out_dir) / str(t.date())
    day_dir.mkdir(parents=True, exist_ok=True)

    copied, errors = [], []
    for src in sorted(Path(stores_dir).glob("*.sqlite")):
        dst = day_dir / src.name
        try:
            con = sqlite3.connect(str(src), timeout=10.0)
            try:
                bak = sqlite3.connect(str(dst))
                try:
                    con.backup(bak)
                finally:
                    bak.close()
            finally:
                con.close()
            copied.append(src.name)
        except sqlite3.Error as exc:
            errors.append({"store": src.name, "error": str(exc)})
            dst.unlink(missing_ok=True)   # never leave a half-written snapshot

    # prune: keep the most recent ``keep`` dated folders (lexicographic == chronological)
    dated = sorted(p for p in Path(out_dir).iterdir()
                   if p.is_dir() and len(p.name) == 10 and p.name[4] == "-")
    pruned = []
    for old in dated[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
        pruned.append(old.name)

    out: dict = {"day": str(t.date()), "copied": copied, "pruned": pruned}
    if errors:
        out["errors"] = errors
        out["_status"] = "degraded"
    return out


def rotate_logs(logs_dir: Path = LOGS_DIR, max_mb: float = LOG_ROTATE_MB) -> dict:
    """Copy-truncate any ``*.log`` past ``max_mb`` to ``<name>.1`` (one generation).

    Copy-then-truncate (not rename) because launchd and the web app hold these files
    open in append mode: after ``truncate(0)`` an O_APPEND writer continues at the
    new EOF, so nothing is lost and no writer needs restarting."""
    rotated = []
    for log in sorted(Path(logs_dir).glob("*.log")):
        try:
            if log.stat().st_size <= max_mb * 1024 * 1024:
                continue
            shutil.copy2(log, log.with_suffix(log.suffix + ".1"))
            with open(log, "r+b") as fh:
                fh.truncate(0)
            rotated.append(log.name)
        except OSError:
            continue   # a busy/vanished log is tomorrow's problem, not tonight's failure
    return {"rotated": rotated, "noop": not rotated}
