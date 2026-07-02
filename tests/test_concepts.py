"""Network-free tests for concept extraction: consolidated-value pick + filters."""

import duckdb
import pandas as pd

from stockscan.concepts import build_fundamentals_wide, coverage


def _facts() -> pd.DataFrame:
    rows = []

    def add(tag, ddate, qtrs, uom, value, form="10-K", adsh="a1", period="2023-12-31"):
        rows.append(
            dict(
                cik=1, adsh=adsh, name="TESTCO", sic=3571, form=form,
                period_end=pd.Timestamp(period), fy=2023, fp="FY",
                filed_date=pd.Timestamp("2024-02-15"), tag=tag, version="us-gaap/2023",
                ddate=pd.Timestamp(ddate), qtrs=qtrs, uom=uom, value=float(value),
            )
        )

    rev = "RevenueFromContractWithCustomerExcludingAssessedTax"
    add(rev, "2023-12-31", 4, "USD", 1000)          # consolidated total (current period)
    add(rev, "2023-12-31", 4, "USD", 600)           # segment
    add(rev, "2023-12-31", 4, "USD", 400)           # segment
    add(rev, "2022-12-31", 4, "USD", 900)           # prior-year comparative -> excluded (ddate)
    add("Assets", "2023-12-31", 0, "USD", 5000)
    add("Assets", "2023-12-31", 0, "shares", 9999)  # wrong uom -> excluded
    add("NetIncomeLoss", "2023-12-31", 4, "USD", 200)
    add("Assets", "2023-12-31", 0, "USD", 7777, form="10-Q", adsh="q1")  # non-10-K -> excluded
    return pd.DataFrame(rows)


def test_extract_picks_consolidated_and_applies_filters(tmp_path):
    fp = tmp_path / "facts.parquet"
    _facts().to_parquet(fp, index=False)
    out = tmp_path / "wide.parquet"

    n = build_fundamentals_wide(out_path=out, fact_paths=[fp], forms=("10-K",))
    assert n == 1  # only the 10-K filing becomes a row

    rev, assets, ni, gp = duckdb.query(
        f"select revenue, assets, net_income, gross_profit from read_parquet('{out}')"
    ).fetchone()
    assert rev == 1000     # consolidated total, not a 600/400 segment
    assert assets == 5000  # the USD row, not the 9999 'shares' row
    assert ni == 200
    assert gp is None      # no GrossProfit tag present -> null


def test_coverage_reports_present_and_missing(tmp_path):
    fp = tmp_path / "f.parquet"
    _facts().to_parquet(fp, index=False)
    out = tmp_path / "w.parquet"
    build_fundamentals_wide(out_path=out, fact_paths=[fp])
    cov = coverage(out)
    assert cov["revenue"] == 1.0
    assert cov["assets"] == 1.0
    assert cov["gross_profit"] == 0.0
