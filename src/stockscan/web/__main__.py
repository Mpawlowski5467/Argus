"""`python -m stockscan.web` → run the server on http://127.0.0.1:<WEB_PORT>/.

Host/port come from config (STOCKSCAN_WEB_HOST / STOCKSCAN_WEB_PORT, default
127.0.0.1:8000). The default HOST must stay loopback: the API carries write
endpoints (positions, a nightly-subprocess launcher) and has no auth — never
bind 0.0.0.0. The PORT knob exists because localhost real estate is shared;
this machine already had an unrelated service squatting on 8000.
"""

from __future__ import annotations


def main() -> int:
    import uvicorn

    from ..config import WEB_HOST, WEB_PORT

    uvicorn.run("stockscan.web.server:app", host=WEB_HOST, port=WEB_PORT,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
