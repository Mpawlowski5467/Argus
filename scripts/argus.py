"""Launch argus, the terminal UI (needs the [ui] extra: `uv sync --extra ui`).

  uv run python scripts/argus.py
  uv run python -m stockscan.tui        # equivalent

A read-mostly, all-seeing view over the scanner: scan (1), ticker drill-down (2),
watchlist + alerts (3), paper-forward (4). `t` toggles light/dark, `r` refreshes,
`n` narrates the open ticker with the local model, `q` quits.
"""

from stockscan.tui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
