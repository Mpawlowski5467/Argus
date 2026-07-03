"""Per-job file locks so a launchd firing can't overlap a still-running instance.

flock is advisory but every entry point goes through job_lock(), and the lock
is released by the OS even if the process dies — no stale-lockfile cleanup.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

from ..config import ARTIFACTS_DIR

LOCK_DIR = ARTIFACTS_DIR / "locks"


class JobAlreadyRunning(RuntimeError):
    pass


@contextmanager
def job_lock(name: str, lock_dir: Path = LOCK_DIR):
    """Hold an exclusive non-blocking lock for the duration of a job.

    Raises JobAlreadyRunning immediately if another process holds it — the
    caller logs and exits 0 (the running instance is doing the work).
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{name}.lock"
    fh = open(path, "w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise JobAlreadyRunning(f"job {name!r} is already running") from exc
        yield
    finally:
        fh.close()  # closing releases the flock
