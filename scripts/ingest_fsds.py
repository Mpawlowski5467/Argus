"""Download + build FSDS fundamentals for the given quarters.

Examples:
  uv run python scripts/ingest_fsds.py 2026q1
  uv run python scripts/ingest_fsds.py 2025q1:2026q1        # inclusive range
  uv run python scripts/ingest_fsds.py 2024q4 2025q2        # explicit list
"""

import sys

from stockscan.edgar.fsds import ingest, iter_quarters


def expand(args: list[str]) -> list[str]:
    quarters: list[str] = []
    for a in args:
        if ":" in a:
            start, end = a.split(":", 1)
            quarters.extend(iter_quarters(start, end))
        else:
            quarters.append(a)
    return quarters


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: ingest_fsds.py <quarter|start:end> [...]")
        return 1
    quarters = expand(argv)
    print(f"Ingesting {len(quarters)} quarter(s): {', '.join(quarters)}")
    total = 0
    for summ in ingest(quarters):
        total += summ["rows"]
        print(f"  {summ['quarter']}: {summ['rows']:>12,} facts  ->  {summ['parquet']}")
    print(f"Done. {total:,} fact rows across {len(quarters)} quarter(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
