"""Fetch daily prices for the ingested fundamentals universe (or explicit tickers).

  uv run python scripts/ingest_prices.py AAPL MSFT NVDA --start 2024-01-01
  uv run python scripts/ingest_prices.py --limit 50 --start 2023-06-01
"""

import argparse
import glob

import duckdb

from stockscan.edgar.fsds import FUNDAMENTALS_DIR
from stockscan.edgar.tickers import cik_to_ticker
from stockscan.prices import PRICES_DIR, download_prices


def universe_tickers(limit: int | None = None) -> list[str]:
    files = sorted(glob.glob(str(FUNDAMENTALS_DIR / "*.parquet")))
    if not files:
        return []
    src = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "])"
    ciks = [r[0] for r in duckdb.query(f"select distinct cik from {src}").fetchall()]
    c2t = cik_to_ticker()
    tickers = sorted({c2t[c] for c in ciks if c in c2t})
    return tickers[:limit] if limit else tickers


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch daily prices to per-ticker Parquet.")
    ap.add_argument("tickers", nargs="*", help="explicit tickers; default = fundamentals universe")
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap the universe size")
    ap.add_argument("--provider", default=None, help="yfinance (default) or tiingo")
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers] or universe_tickers(args.limit)
    if not tickers:
        print("no tickers to fetch (ingest fundamentals first, or pass tickers explicitly)")
        return 1
    src = args.provider or "yfinance (default)"
    print(f"Fetching prices [{src}] for {len(tickers)} ticker(s) ({args.start} -> {args.end or 'today'}) ...")
    written = download_prices(tickers, start=args.start, end=args.end, provider=args.provider)
    print(f"Wrote {len(written)} ticker price file(s) to {PRICES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
