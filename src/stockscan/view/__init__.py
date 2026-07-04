"""argus view layer — the read-mostly data facade the web UI is built on.

:class:`~stockscan.view.data.ArgusData` loads the frozen artifact, the price /
fundamentals matrices and the ops DB once, and shapes them into plain rows +
per-ticker analysis for the browser front-end (``stockscan.web``). Alongside it
sit the shared pure helpers ``chart.verdict`` / ``chart.price_summary`` and
``treemap.squarify``. It never trains, re-baselines, or touches the model; the
only writes are the user's own watchlist / position curation (to the ops SQLite).
"""

APP_NAME = "argus"
