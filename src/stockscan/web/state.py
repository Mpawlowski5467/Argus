"""The one shared ``ArgusData`` for the web app, plus a load-status flag.

Loaded once in a background thread at startup so uvicorn binds the port instantly
— the browser's mandala loader covers the ~7s cold load, then flips to the app.
Read-mostly; the only guarded mutation is ``refresh()`` rebuilding the scored
cross-section (watchlist toggles go straight to WAL-safe SQLite).
"""

from __future__ import annotations

import threading


class AppState:
    def __init__(self) -> None:
        self.adata = None                 # ArgusData | None
        self.status = "loading"           # "loading" | "ready" | "error"
        self.error: str | None = None
        self.lock = threading.Lock()

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


STATE = AppState()
