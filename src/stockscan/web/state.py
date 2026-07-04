"""The one shared ``ArgusData`` for the web app, plus a load-status flag.

Loaded once in a background thread at startup so uvicorn binds the port instantly
— the browser's mandala loader covers the ~7s cold load, then flips to the app.
Read-mostly; the only guarded mutation is ``refresh()`` rebuilding the scored
cross-section (watchlist toggles go straight to WAL-safe SQLite).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

from ..config import REPO_ROOT

_NIGHTLY_LOG = REPO_ROOT / "data" / "logs" / "nightly.web.log"


class AppState:
    def __init__(self) -> None:
        self.adata = None                 # ArgusData | None
        self.status = "loading"           # "loading" | "ready" | "error"
        self.error: str | None = None
        self.lock = threading.Lock()
        # on-demand nightly ("update data" button): the SAME dispatcher launchd runs,
        # launched as a self-guarded subprocess so it can't collide with the schedule.
        self.nightly_proc: subprocess.Popen | None = None
        self.nightly_started: float | None = None
        self.nightly_last: dict | None = None

    def start_load(self) -> None:
        """Kick off the heavy load off the event loop (non-blocking)."""
        threading.Thread(target=self._load, name="argus-load", daemon=True).start()

    def _load(self) -> None:
        try:
            from ..tui.data import ArgusData
            adata = ArgusData.load()
        except Exception as exc:  # bind now — surface to the loader, don't crash the server
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            return
        with self.lock:
            self.adata = adata
            self.status = "ready"

    def refresh(self) -> None:
        with self.lock:
            if self.adata is not None:
                self.adata.refresh()

    def reload(self) -> None:
        """Full reload from disk — the ONLY way to pick up freshly-ingested prices/filings
        (``refresh`` just re-scores the already-loaded data). Reuses the startup loader UX:
        flip to 'loading' and rebuild ArgusData in a background thread."""
        with self.lock:
            self.status = "loading"
            self.error = None
        self.start_load()

    def start_nightly(self) -> dict:
        """Launch the nightly dispatcher (``ops.py nightly``) as a background subprocess.
        It self-guards with the repo-wide ops flock, so a button-triggered run can't collide
        with the scheduled launchd firing. No-ops if one this app started is still running."""
        with self.lock:
            if self.nightly_proc is not None and self.nightly_proc.poll() is None:
                return {"running": True, "already": True}
            _NIGHTLY_LOG.parent.mkdir(parents=True, exist_ok=True)
            logf = open(_NIGHTLY_LOG, "ab")
            try:
                self.nightly_proc = subprocess.Popen(
                    [sys.executable, "scripts/ops.py", "nightly"],
                    cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT,
                )
            finally:
                logf.close()   # the child keeps its own dup of the fd
            self.nightly_started = time.time()
            return {"running": True, "already": False}

    def nightly_status(self) -> dict:
        """Poll the on-demand nightly: running + elapsed, or the last run's exit code."""
        proc = self.nightly_proc
        if proc is None:
            return {"running": False, "last": self.nightly_last}
        rc = proc.poll()
        if rc is None:
            return {"running": True, "elapsed": round(time.time() - (self.nightly_started or 0), 1)}
        self.nightly_last = {"returncode": int(rc)}
        return {"running": False, "returncode": int(rc),
                "elapsed": round(time.time() - (self.nightly_started or 0), 1)}


STATE = AppState()
