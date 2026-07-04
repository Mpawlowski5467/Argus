"""Launch argus-web, the browser UI (needs the [web] extra: `uv sync --extra web`).

  uv run python scripts/argus_web.py
  uv run python -m stockscan.web        # equivalent

Then open http://127.0.0.1:8000/ — the mandala loader shows while the data loads,
then reveals the scanner: scan (1), ticker (2), watch (3), paper (4), markets (5).
Read-mostly over the frozen artifact; the sole writes are watchlist curation
and personal position tracking (both live-view, firewalled from the signal).
"""

from stockscan.web.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
