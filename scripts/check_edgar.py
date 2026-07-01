"""Connectivity + client smoke check: fetch the CIK <-> ticker map from EDGAR.

Run: uv run python scripts/check_edgar.py
"""

from stockscan.edgar.client import EdgarClient


def main() -> None:
    with EdgarClient() as ec:
        data = ec.company_tickers()
    print(f"EDGAR OK: fetched {len(data)} ticker records")
    for row in list(data.values())[:3]:
        print(f"  {row.get('ticker'):<8} CIK={row.get('cik_str'):<10} {row.get('title')}")


if __name__ == "__main__":
    main()
