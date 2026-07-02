"""Fetch survivorship-free daily prices for the whole Intrinio universe (long job).

Reads data/parquet/intrinio_universe.parquet (build_intrinio_universe.py first),
fetches every company's candidate securities BY SECURITY ID, splices/clips, and
writes one Parquet per column. Resumable: existing files are skipped.

  uv run python scripts/fetch_intrinio_prices.py --start 2011-01-01 --pause 0.2
"""

import argparse

from stockscan.intrinio_universe import load_universe
from stockscan.prices import PRICES_DIR, download_intrinio_universe


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Intrinio prices for the universe map.")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pause", type=float, default=0.2, help="seconds between requests per worker")
    ap.add_argument("--workers", type=int, default=1, help="concurrent fetch threads")
    ap.add_argument("--limit", type=int, default=None, help="cap number of companies (smoke test)")
    args = ap.parse_args()

    uni = load_universe()
    if uni.empty:
        print("no universe map; run scripts/build_intrinio_universe.py first")
        return 1
    if args.limit:
        keep = uni.drop_duplicates("cik")["column"].head(args.limit)
        uni = uni[uni["column"].isin(set(keep))]
    print(f"fetching {uni['column'].nunique()} companies "
          f"({len(uni)} candidate securities) -> {PRICES_DIR}", flush=True)
    written = download_intrinio_universe(
        uni, start=args.start, end=args.end, pause=args.pause, workers=args.workers,
    )
    print(f"wrote {len(written)} new price files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
