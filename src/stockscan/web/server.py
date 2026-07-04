"""argus-web — the FastAPI app.

Same core as the TUI; the only new thing is HTTP. ``ArgusData`` loads once in a
background thread at startup (so the port binds instantly and the mandala loader
covers the ~7s), and the whole app is served same-origin so there's no CORS.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router
from .state import STATE

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.start_load()          # non-blocking; the port binds immediately
    yield


app = FastAPI(title="argus-web", lifespan=lifespan)
app.include_router(router, prefix="/api")
# Static mount LAST so /api/* wins; html=True serves index.html at "/".
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
