"""Capture argus-web screenshots for the docs (docs/img/*.png).

Drives a running argus-web instance with headless Chromium. Point it at a
THROWAWAY server seeded with demo data so no real portfolio is published:

    pip/uv install playwright && python -m playwright install chromium
    STOCKSCAN_OPS_STATE_PATH=/tmp/demo.sqlite \
        uv run python -m uvicorn stockscan.web.server:app --port 8024   # (seed demo first)
    uv run python scripts/capture_screenshots.py http://127.0.0.1:8024

Regenerate whenever the UI changes. Pure docs tooling — not imported anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8024"
OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)


def _tab(page, view):
    page.click(f'.tab[data-view="{view}"]')


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 860},
                                device_scale_factor=2)
        page.goto(BASE, wait_until="domcontentloaded")
        # the facade loads in a background thread; the app reveals once scan is populated
        page.wait_for_selector("#scan-table tbody tr", timeout=90_000)
        page.wait_for_timeout(400)

        # 1 · scan — the ranked universe (viewport)
        page.screenshot(path=str(OUT / "scan.png"))
        print("scan.png")

        # 2 · book — the portfolio scorecard (full page; demo holdings)
        _tab(page, "scorecard")
        page.wait_for_selector("#sc-table tbody tr", timeout=20_000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "book.png"), full_page=True)
        print("book.png")

        # 3 · ticker — one name in full, with the chart crosshair/tooltip
        _tab(page, "scan")                          # the search box lives in the scan view
        page.wait_for_selector("#search", state="visible")
        page.fill("#search", "AAPL")
        page.press("#search", "Enter")
        page.wait_for_selector("#tk-body .tk-head", timeout=20_000)
        page.wait_for_function("() => { const c = document.getElementById('tk-chart');"
                               " return c && c.__chart; }", timeout=20_000)
        page.wait_for_timeout(600)
        chart = page.query_selector("#tk-chart")
        box = chart.bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.62, box["y"] + box["height"] * 0.4)
        page.wait_for_timeout(250)
        page.screenshot(path=str(OUT / "ticker.png"), full_page=True)
        print("ticker.png")

        # 4 · markets — themes / industries + treemap picks
        _tab(page, "markets")
        try:
            page.wait_for_selector("#markets-body .mkt-group", timeout=20_000)
            page.wait_for_timeout(2500)   # let live market caps fill the bars
        except Exception:
            pass
        page.screenshot(path=str(OUT / "markets.png"))
        print("markets.png")

        # 5 · paper — the honesty gate (frozen model's live accuracy)
        _tab(page, "paper")
        page.wait_for_selector("#paper-body", timeout=20_000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "paper.png"))
        print("paper.png")

        browser.close()
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
