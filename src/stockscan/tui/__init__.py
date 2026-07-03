"""argus — the terminal UI: an all-seeing, read-mostly viewer over the scanner.

A thin Textual app on top of the existing serve/ops layer. It only READS the
frozen artifact, the price/fundamentals data, and the paper-forward record; the
sole writes are the user's own watchlist/alert curation (to the ops SQLite,
which is WAL-safe against the nightly job). It never trains, re-baselines, or
touches the model. `python -m stockscan.tui` (needs the [ui] extra: textual).
"""

APP_NAME = "argus"
