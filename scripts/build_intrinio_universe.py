"""Enumerate the Intrinio security master and build the survivorship-free universe map.

Paginates /companies (CIK linkage, dead companies retained) and /securities for US
common stocks (EQS) in BOTH active states, joins to our EDGAR fundamentals CIKs and
the delisting ledger, and writes data/parquet/intrinio_universe.parquet — one price
column per company, one row per candidate security (fetched by id).

  uv run python scripts/build_intrinio_universe.py
"""

import duckdb
import httpx
import pandas as pd

from stockscan.concepts import WIDE_PATH
from stockscan.config import INTRINIO_API_KEY
from stockscan.edgar.delistings import load_delistings
from stockscan.intrinio_universe import (
    UNIVERSE_PATH,
    enumerate_companies,
    enumerate_securities,
    select_universe,
)


def main() -> int:
    if not INTRINIO_API_KEY:
        print("no Intrinio key; set STOCKSCAN_INTRINIO_KEY in .env")
        return 1
    our_ciks = {
        r[0] for r in duckdb.query(
            f"select distinct cik from read_parquet('{WIDE_PATH}')"
        ).fetchall()
    }
    print(f"fundamentals universe: {len(our_ciks)} CIKs")

    with httpx.Client(base_url="https://api-v2.intrinio.com", timeout=60.0) as client:
        companies = enumerate_companies(client, INTRINIO_API_KEY)
        print(f"intrinio companies: {len(companies)} ({companies['cik'].notna().sum()} with CIK)",
              flush=True)
        securities = pd.concat(
            [enumerate_securities(client, INTRINIO_API_KEY, active=True),
             enumerate_securities(client, INTRINIO_API_KEY, active=False)],
            ignore_index=True,
        )
    print(f"intrinio US EQS securities: {len(securities)} "
          f"({int(securities['active'].sum())} active)")

    uni = select_universe(securities, companies, our_ciks, load_delistings())
    uni.to_parquet(UNIVERSE_PATH, index=False)

    per_cik = uni.drop_duplicates("cik")
    n_dead = int(per_cik["column"].str.contains("~").sum())
    print(f"universe written: {UNIVERSE_PATH}")
    print(f"  companies matched: {len(per_cik)} of {len(our_ciks)} fundamentals CIKs")
    print(f"  active: {len(per_cik) - n_dead}   dead: {n_dead}")
    print(f"  candidate securities total: {len(uni)}")
    print(f"  dead with ledger clip: {int(per_cik['clip_date'].notna().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
