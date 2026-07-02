"""Network-free tests for the FSDS ingest: the num->sub join, filters, date keying."""

from datetime import date

import duckdb
import pytest

from stockscan.edgar.fsds import build_fundamentals, iter_quarters, parse_quarter, quarter_url

_SUB = (
    "\t".join(["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed"]) + "\n"
    + "\t".join(["0001-24-01", "320193", "APPLE INC", "3571", "10-K", "20230930", "2023", "FY", "20231103"]) + "\n"
    + "\t".join(["0002-24-02", "111", "BADFORM CO", "1234", "8-K", "20230930", "2023", "FY", "20231103"]) + "\n"
)

_NUM = (
    "\t".join(["adsh", "tag", "version", "coreg", "segments", "ddate", "qtrs", "uom", "value", "footnote"]) + "\n"
    + "\t".join(["0001-24-01", "Revenues", "us-gaap/2023", "", "", "20230930", "4", "USD", "383285000000", ""]) + "\n"
    + "\t".join(["0001-24-01", "Assets", "us-gaap/2023", "", "", "20230930", "0", "USD", "352583000000", ""]) + "\n"
    + "\t".join(["0001-24-01", "Revenues", "us-gaap/2023", "", "Geo=US;", "20230930", "4", "USD", "300000000000", ""]) + "\n"
    + "\t".join(["0001-24-01", "SubRev", "us-gaap/2023", "SUBSID", "", "20230930", "4", "USD", "999", ""]) + "\n"
    + "\t".join(["0002-24-02", "Revenues", "us-gaap/2023", "", "", "20230930", "4", "USD", "500", ""]) + "\n"
)


def test_build_fundamentals_join_and_filters(tmp_path):
    (tmp_path / "sub.txt").write_text(_SUB)
    (tmp_path / "num.txt").write_text(_NUM)
    out = tmp_path / "out.parquet"

    n = build_fundamentals(tmp_path / "sub.txt", tmp_path / "num.txt", out)

    # Only Apple's two consolidated facts survive: the 8-K is dropped by the form filter,
    # the SUBSID (coreg) fact by the coreg filter, and the Geo=US revenue row (a segment,
    # segments != '') by the new segments filter -- so the $383B consolidated total wins.
    assert n == 2
    rows = duckdb.query(
        f"SELECT tag, value, form, sic, cik, filed_date "
        f"FROM read_parquet('{out}') ORDER BY tag"
    ).fetchall()
    by_tag = {r[0]: r for r in rows}
    assert list(by_tag) == ["Assets", "Revenues"]
    assert by_tag["Revenues"][1] == 383285000000.0
    assert by_tag["Revenues"][2] == "10-K"
    assert by_tag["Revenues"][3] == 3571
    assert by_tag["Revenues"][4] == 320193
    assert by_tag["Revenues"][5] == date(2023, 11, 3)  # keyed to filing date, not period-end


def test_iter_quarters_inclusive_across_year_boundary():
    assert iter_quarters("2023q3", "2024q2") == ["2023q3", "2023q4", "2024q1", "2024q2"]


def test_quarter_url_and_validation():
    assert quarter_url("2024q1").endswith("/2024q1.zip")
    with pytest.raises(ValueError):
        parse_quarter("2024Q1")  # capital Q rejected
    with pytest.raises(ValueError):
        parse_quarter("2024q5")  # no 5th quarter
