"""`python -m stockscan.web` → run the dev server on http://127.0.0.1:8000/."""

from __future__ import annotations


def main() -> int:
    import uvicorn
    uvicorn.run("stockscan.web.server:app", host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
