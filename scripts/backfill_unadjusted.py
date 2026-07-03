"""Backfill the unadjusted close/volume (uclose/uvolume) into the price store.

The Phase-5 schema addition captures Intrinio's raw close/volume alongside the
adjusted OHLCV — the data-layer fix for liquidity floors distorted by retroactive
adjustment (a heavy dividend/split history can push a real $30 close below the
$1 floor years back). Files fetched before that change lack the columns; this
job refetches ONLY those, by security id, resumably (a refetched file has
uclose, so a re-run skips it).

  uv run python scripts/backfill_unadjusted.py --limit 50   # smoke test
  uv run python scripts/backfill_unadjusted.py              # full store (hours)

This does NOT change any threshold. Switching the liquidity floor to unadjusted
close is a deliberate re-baseline (panel rebuild -> retrain -> new artifact
vintage -> re-freeze), because it alters historical universe membership; see
RESULTS.md Phase-5. This job only makes the raw data available for that step.
"""

import argparse

from stockscan.config import INTRINIO_API_KEY
from stockscan.intrinio_universe import load_universe
from stockscan.ops.jobs import columns_missing_unadjusted, refetch_columns
from stockscan.prices import PRICES_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill uclose/uvolume into the price store.")
    ap.add_argument("--limit", type=int, default=None, help="cap columns (smoke test)")
    ap.add_argument("--pause", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", default="2011-01-01")
    args = ap.parse_args()

    if not INTRINIO_API_KEY:
        print("no Intrinio key; set STOCKSCAN_INTRINIO_KEY in .env")
        return 1
    uni = load_universe()
    if uni.empty:
        print("no universe map; run scripts/build_intrinio_universe.py first")
        return 1

    todo = columns_missing_unadjusted(uni, PRICES_DIR)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} column(s) lack unadjusted close; refetching by security id ...",
          flush=True)
    if not todo:
        return 0
    res = refetch_columns(uni, todo, INTRINIO_API_KEY, PRICES_DIR, start=args.start,
                          pause=args.pause, workers=args.workers)
    print(f"written={len(res['written'])} suspect={len(res['suspect'])} "
          f"failed={len(res['failed'])} (re-run to retry failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
