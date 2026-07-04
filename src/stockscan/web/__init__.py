"""argus-web — a browser front-end over the same serve/ops layer as the TUI.

A thin FastAPI app on top of the existing ``ArgusData`` facade: it only READS the
frozen artifact, the price/fundamentals data, and the paper-forward record; the
sole writes are the user's own watchlist curation. It never trains, re-baselines,
or touches the model. Needs the ``[web]`` extra (fastapi + uvicorn):

    uv sync --extra web
    uv run python -m stockscan.web        # or scripts/argus_web.py

The live-view firewall is preserved structurally: the signal packet endpoint
returns score/verdict with no live data; profile/news/quote/market-cap/themes are
separate endpoints the browser fetches after the packet.
"""

APP_NAME = "argus-web"
