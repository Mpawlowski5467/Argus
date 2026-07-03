"""Ingest the SEC Financial Statement Data Sets (FSDS) into a point-in-time panel.

FSDS ships one zip per quarter (2009q2 -> present). Each zip contains tab-delimited
``sub.txt`` (one row per filing: adsh, cik, sic, form, period, filed, ...) and
``num.txt`` (one row per numeric XBRL fact: adsh, tag, version, coreg, ddate, qtrs,
uom, value).

Crucially, ``num.txt`` carries NO date -- the filing date lives only in ``sub.txt``
as ``filed``. We join num -> sub on ``adsh`` and key every fact to that filing date;
the +1 business-day availability lag is applied later at feature-build time via
:mod:`stockscan.pit`. We keep only 10-K/10-Q(/A) filings and consolidated
(empty-``coreg``) facts, and write an append-only Parquet fact ledger, one file per
quarter.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import duckdb

from ..config import PARQUET_DIR, RAW_DIR
from .client import EdgarClient

FSDS_BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
FUNDAMENTALS_DIR = PARQUET_DIR / "fundamentals"
_QUARTER_RE = re.compile(r"^(\d{4})q([1-4])$")

# read_csv options shared by both raw text files. Note: the Python string turns
# `\t` into a real tab, which DuckDB takes as the delimiter. Empty quote/escape
# disable quoting so a stray quote in a footnote can't swallow rows.
_CSV_OPTS = (
    "delim='\t', header=true, all_varchar=true, quote='', escape='', "
    "nullstr='', null_padding=true, ignore_errors=true"
)


def parse_quarter(quarter: str) -> tuple[int, int]:
    m = _QUARTER_RE.match(quarter)
    if not m:
        raise ValueError(f"bad quarter {quarter!r}; expected e.g. '2024q1'")
    return int(m.group(1)), int(m.group(2))


def quarter_url(quarter: str) -> str:
    parse_quarter(quarter)
    return f"{FSDS_BASE}/{quarter}.zip"


def iter_quarters(start: str, end: str) -> list[str]:
    """Inclusive list of quarter labels from ``start`` to ``end`` (e.g. 2023q3..2024q2)."""
    (sy, sq), (ey, eq) = parse_quarter(start), parse_quarter(end)
    out: list[str] = []
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        out.append(f"{y}q{q}")
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


def download_quarter(
    quarter: str, client: EdgarClient | None = None, dest_dir: Path = RAW_DIR / "fsds"
) -> Path:
    dest = Path(dest_dir) / f"{quarter}.zip"
    own = client is None
    client = client or EdgarClient()
    try:
        return client.download(quarter_url(quarter), dest)
    finally:
        if own:
            client.close()


def _extract(
    zip_path: Path, members: tuple[str, ...] = ("sub.txt", "num.txt"), dest_dir: Path | None = None
) -> dict[str, Path]:
    dest_dir = Path(dest_dir or zip_path.with_suffix(""))
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in members:
            target = dest_dir / name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out[name] = target
    return out


def build_fundamentals(sub_path: Path, num_path: Path, out_path: Path) -> int:
    """Join num -> sub on adsh, key to ``filed_date``, write a Parquet fact ledger.

    Keeps only 10-K/10-Q(/A) filings and consolidated (empty-coreg) facts.
    Returns the number of fact rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # COPY writes to a tmp path, promoted only on success: the quarter file's very
    # existence is the ingest checkpoint, so a crash mid-COPY must never leave a
    # truncated parquet pinned at the final path (it would read as "done" forever).
    tmp_path = out_path.with_name("." + out_path.name + ".tmp")
    sub_p = str(sub_path).replace("'", "''")
    num_p = str(num_path).replace("'", "''")
    out_p = str(tmp_path).replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute(
            f"""CREATE VIEW sub AS
                  SELECT adsh, cik, name, sic, form, period, fy, fp, filed
                  FROM read_csv('{sub_p}', {_CSV_OPTS})
                  WHERE form LIKE '10-K%' OR form LIKE '10-Q%';"""
        )
        con.execute(
            f"""CREATE VIEW num AS
                  SELECT adsh, tag, version, coreg, segments, ddate, qtrs, uom, value
                  FROM read_csv('{num_p}', {_CSV_OPTS});"""
        )
        con.execute(
            f"""COPY (
                  SELECT
                    CAST(s.cik AS BIGINT)                          AS cik,
                    s.adsh                                         AS adsh,
                    s.name                                         AS name,
                    TRY_CAST(s.sic AS INTEGER)                     AS sic,
                    s.form                                         AS form,
                    CAST(try_strptime(s.period, '%Y%m%d') AS DATE) AS period_end,
                    TRY_CAST(s.fy AS INTEGER)                      AS fy,
                    s.fp                                           AS fp,
                    CAST(try_strptime(s.filed, '%Y%m%d') AS DATE)  AS filed_date,
                    n.tag                                          AS tag,
                    n.version                                      AS version,
                    CAST(try_strptime(n.ddate, '%Y%m%d') AS DATE)  AS ddate,
                    TRY_CAST(n.qtrs AS INTEGER)                    AS qtrs,
                    n.uom                                          AS uom,
                    TRY_CAST(n.value AS DOUBLE)                    AS value
                  FROM num n JOIN sub s USING (adsh)
                  WHERE (n.coreg IS NULL OR n.coreg = '')
                    AND (n.segments IS NULL OR n.segments = '')  -- consolidated (undimensioned) only
                    AND try_strptime(s.filed, '%Y%m%d') IS NOT NULL
                ) TO '{out_p}' (FORMAT PARQUET, COMPRESSION ZSTD);"""
        )
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_p}')").fetchone()[0]
    finally:
        con.close()
    import os

    os.replace(tmp_path, out_path)
    return rows


def ingest_quarter(
    quarter: str, client: EdgarClient | None = None, keep_extracted: bool = False
) -> dict:
    zip_path = download_quarter(quarter, client=client)
    extracted = _extract(zip_path)
    out_path = FUNDAMENTALS_DIR / f"{quarter}.parquet"
    rows = build_fundamentals(extracted["sub.txt"], extracted["num.txt"], out_path)
    if not keep_extracted:
        for p in extracted.values():
            p.unlink(missing_ok=True)
        try:
            zip_path.with_suffix("").rmdir()  # remove now-empty extract dir
        except OSError:
            pass
    return {
        "quarter": quarter,
        "rows": rows,
        "parquet": str(out_path),
        "zip_bytes": zip_path.stat().st_size,
    }


def ingest(
    quarters, client: EdgarClient | None = None, keep_extracted: bool = False
) -> list[dict]:
    own = client is None
    client = client or EdgarClient()
    try:
        return [ingest_quarter(q, client=client, keep_extracted=keep_extracted) for q in quarters]
    finally:
        if own:
            client.close()
