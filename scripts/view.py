"""View a company's ingested fundamentals from the terminal.

  uv run python scripts/view.py AAPL
  uv run python scripts/view.py --cik 320193

(Named view.py, not inspect.py, to avoid shadowing the stdlib `inspect` module.)
"""

import argparse
import glob

import duckdb

from stockscan.edgar.fsds import FUNDAMENTALS_DIR
from stockscan.edgar.tickers import cik_for

CURATED = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "NetIncomeLoss",
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
]


def fmt(v) -> str:
    if v is None:
        return "-"
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect ingested FSDS fundamentals for one company.")
    ap.add_argument("ticker", nargs="?", help="e.g. AAPL")
    ap.add_argument("--cik", type=int, help="look up by CIK instead of ticker")
    args = ap.parse_args()

    cik = args.cik or (cik_for(args.ticker) if args.ticker else None)
    if not cik:
        print("provide a ticker (e.g. AAPL) or --cik <n>")
        return 1

    files = sorted(glob.glob(str(FUNDAMENTALS_DIR / "*.parquet")))
    if not files:
        print(f"no fundamentals parquet yet in {FUNDAMENTALS_DIR}; run scripts/ingest_fsds.py first")
        return 1
    src = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "])"

    def q(s):
        return duckdb.query(s).fetchall()

    name = q(f"select any_value(name) from {src} where cik={cik}")
    if not name or name[0][0] is None:
        print(f"CIK {cik} not present in the ingested quarters")
        return 1

    adsh, form, filed, pend = q(
        f"select adsh, form, filed_date, period_end from {src} "
        f"where cik={cik} order by filed_date desc, period_end desc limit 1"
    )[0]
    print(f"\n{name[0][0]}  (CIK {cik})")
    print(f"latest filing: {form}  filed {filed}  period {pend}\n")

    tags_sql = ",".join(f"'{t}'" for t in CURATED)
    rows = q(
        f"select tag, ddate, arg_max(value, abs(value)) v from {src} "
        f"where adsh='{adsh}' and tag in ({tags_sql}) group by tag, ddate "
        f"qualify row_number() over (partition by tag order by ddate desc) = 1 order by tag"
    )
    print(f"  {'concept':<46}{'period':<12}{'value':>14}")
    print("  " + "-" * 72)
    for tag, d, v in rows:
        print(f"  {tag:<46}{str(d):<12}{fmt(v):>14}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
