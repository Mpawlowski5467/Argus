"""Extract standardized financial concepts from the raw FSDS fact ledger.

The plain FSDS ``num.txt`` drops XBRL dimensional qualifiers, so a single concept
(e.g. revenue) appears as many rows for one period: the consolidated total plus
geographic/product/segment members, indistinguishable by key. For additive
line items the consolidated total is the maximum-absolute value, so we select each
concept via ``arg_max(value, abs(value))`` over a priority-ordered list of candidate
us-gaap tags, restricted to the filing's own primary period (``ddate = period_end``)
and the right duration (balance-sheet stocks at ``qtrs=0``, annual flows at ``qtrs=4``).

This is a documented heuristic. A fully-correct fix would use the Financial Statement
AND Notes dataset's dimension hash to isolate undimensioned facts; noted for later.
Coverage per concept is logged so a systematically-missing tag can't hide.
"""

from __future__ import annotations

import glob
from pathlib import Path

import duckdb

from .config import PARQUET_DIR
from .edgar.fsds import FUNDAMENTALS_DIR

WIDE_PATH = PARQUET_DIR / "fundamentals_wide.parquet"
STOCK_QTRS = 0   # balance-sheet items are point-in-time
FLOW_QTRS = 4    # income/cash-flow items: annual (10-K) duration

# concept -> (priority-ordered candidate us-gaap tags, kind)
CONCEPTS: dict[str, tuple[list[str], str]] = {
    "revenue": (
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
         "RevenueFromContractWithCustomerIncludingAssessedTax"],
        "flow",
    ),
    "cost_of_revenue": (["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"], "flow"),
    "gross_profit": (["GrossProfit"], "flow"),
    "operating_income": (["OperatingIncomeLoss"], "flow"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "flow"),
    "cfo": (
        ["NetCashProvidedByUsedInOperatingActivities",
         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "flow",
    ),
    "assets": (["Assets"], "stock"),
    "assets_current": (["AssetsCurrent"], "stock"),
    "liabilities": (["Liabilities"], "stock"),
    "liabilities_current": (["LiabilitiesCurrent"], "stock"),
    # Raw equity tag is unreliable via max-abs: a component (e.g. common stock/APIC)
    # can exceed the total when retained earnings are negative (buyback-heavy firms
    # like Apple). We keep it but DERIVE `equity` from the accounting identity below.
    "equity_tag": (
        ["StockholdersEquity",
         "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "stock",
    ),
    "cash": (
        ["CashAndCashEquivalentsAtCarryingValue",
         "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
        "stock",
    ),
}


def _concept_expr(name: str, tags: list[str], kind: str) -> str:
    qtrs = STOCK_QTRS if kind == "stock" else FLOW_QTRS
    picks = [
        f"arg_max(value, abs(value)) FILTER "
        f"(WHERE tag = '{t}' AND qtrs = {qtrs} AND ddate = period_end AND uom = 'USD')"
        for t in tags
    ]
    return f"coalesce({', '.join(picks)}) AS {name}"


def extract_sql(src: str, forms: tuple[str, ...] = ("10-K",)) -> str:
    cols = ",\n    ".join(_concept_expr(n, tags, kind) for n, (tags, kind) in CONCEPTS.items())
    form_list = ",".join(f"'{f}'" for f in forms)
    inner = f"""
        SELECT
            cik,
            adsh,
            any_value(name)        AS name,
            any_value(sic)         AS sic,
            any_value(form)        AS form,
            any_value(fy)          AS fy,
            any_value(filed_date)  AS filed_date,
            any_value(period_end)  AS period_end,
            {cols}
        FROM {src}
        WHERE form IN ({form_list})
        GROUP BY cik, adsh
    """
    # Derive robust total equity from the accounting identity (assets - liabilities),
    # falling back to the raw tag only when a balance-sheet side is missing.
    return f"SELECT *, coalesce(assets - liabilities, equity_tag) AS equity FROM ({inner})"


def build_fundamentals_wide(out_path=None, fact_paths=None, forms=("10-K",)) -> int:
    """Build a one-row-per-filing wide table of standardized concepts. Returns row count."""
    fact_paths = fact_paths or sorted(glob.glob(str(FUNDAMENTALS_DIR / "*.parquet")))
    if not fact_paths:
        return 0
    import os

    # WIDE_PATH is read on EVERY serve/monitor/paper pass; write to a tmp file and
    # os.replace it in, so a crash mid-COPY can never pin a truncated parquet at the
    # live path (it would read as corrupt forever, with no self-heal since the
    # quarter files still look ingested). Mirrors edgar/fsds.build_fundamentals.
    final = Path(out_path or WIDE_PATH)
    tmp = final.with_name("." + final.name + ".tmp")
    out_p = str(tmp).replace("'", "''")
    src = "read_parquet([" + ",".join(f"'{str(p)}'" for p in fact_paths) + "])"
    con = duckdb.connect()
    try:
        con.execute(f"COPY ({extract_sql(src, forms)}) TO '{out_p}' "
                    f"(FORMAT PARQUET, COMPRESSION ZSTD)")
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_p}')").fetchone()[0]
    finally:
        con.close()
    os.replace(tmp, final)
    return rows


def coverage(wide_path=None) -> dict[str, float]:
    """Fraction of filings with a non-null value for each concept (data-quality check)."""
    path = str(wide_path or WIDE_PATH).replace("'", "''")
    con = duckdb.connect()
    try:
        total = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        if not total:
            return {}
        cols = ", ".join(f"count({c}) AS {c}" for c in CONCEPTS)
        row = con.execute(f"SELECT {cols} FROM read_parquet('{path}')").fetchdf().iloc[0]
        return {c: round(float(row[c]) / total, 3) for c in CONCEPTS}
    finally:
        con.close()
