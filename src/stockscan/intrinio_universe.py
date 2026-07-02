"""Survivorship-free price universe from the Intrinio security master.

Enumerates Intrinio's US common stocks (code ``EQS``) — active AND inactive — and
joins them to our EDGAR fundamentals CIKs via Intrinio's ``/companies`` records
(which retain delisted companies with their CIK). The result is one price *column*
per company, backed by one or more Intrinio *securities* fetched strictly BY
SECURITY ID.

Why by id: tickers are recycled aggressively (five different dead companies used
"AAC"), so a by-ticker lookup for a delisted name can silently return a different,
currently-listed company's prices. A security id is one security forever, which
closes that contamination channel structurally.

Column naming: active companies use their plain Intrinio ticker; dead companies use
``TICKER~CIK`` so a recycled ticker can never collide with the live name that now
owns it. A dead company may have several securities (exchange listing + OTC
bankruptcy afterlife, e.g. Borders' BGP then BGPIQ); all candidates are fetched and
spliced by date, exchange listing preferred where they overlap.

Clip: OFF by default. By-id fetching already bounds every series to its own
security's life, and an audit of the full fetch (2026-07) showed the ledger-based
clip (earliest Form-25/15 + 3y grace) actively destroys real history: 473 dead
columns ended exactly at the clip boundary because the ledger's earliest event is
often a BOND delisting years before the stock died (Sears lost its whole 2016-18
collapse to a 2013 notes Form-25). Zero contamination was found without the clip.
``clip_grace_days`` remains available for providers that need it. Known residual:
a company that died and RELISTED under the same CIK (rare — e.g. AMR->AAL) keeps
only its active security's era.
"""

from __future__ import annotations

import pandas as pd

from .config import PARQUET_DIR
from .prices import intrinio_get_json

UNIVERSE_PATH = PARQUET_DIR / "intrinio_universe.parquet"
CLIP_GRACE_DAYS = 1095  # grace beyond the ledger delist date, when clipping is enabled
MAX_CANDIDATES = 6      # securities fetched per company, in priority order

_SEC_COLS = ["security_id", "company_id", "ticker", "composite_ticker", "name",
             "primary_listing", "figi", "active"]


def enumerate_companies(client, api_key: str, page_size: int = 1000) -> pd.DataFrame:
    """All Intrinio companies (dead ones included): company_id, ticker, name, cik."""
    rows, next_page = [], None
    while True:
        params = {"api_key": api_key, "page_size": page_size}
        if next_page:
            params["next_page"] = next_page
        data = intrinio_get_json(client, "/companies", params)
        if data is None:
            raise RuntimeError("Intrinio /companies enumeration failed (rate limit or auth)")
        for c in data.get("companies", []):
            rows.append((c.get("id"), c.get("ticker"), c.get("name"), c.get("cik")))
        next_page = data.get("next_page")
        if not next_page:
            break
    return pd.DataFrame(rows, columns=["company_id", "ticker", "name", "cik"])


def enumerate_securities(client, api_key: str, active: bool, page_size: int = 10000) -> pd.DataFrame:
    """US common-stock securities (code EQS, USD, :US composite) for one active state."""
    rows, next_page = [], None
    while True:
        params = {"api_key": api_key, "page_size": page_size, "code": "EQS",
                  "currency": "USD", "active": "true" if active else "false"}
        if next_page:
            params["next_page"] = next_page
        data = intrinio_get_json(client, "/securities", params)
        if data is None:
            raise RuntimeError("Intrinio /securities enumeration failed (rate limit or auth)")
        for s in data.get("securities", []):
            if not (s.get("composite_ticker") or "").endswith(":US"):
                continue
            rows.append((s.get("id"), s.get("company_id"), s.get("ticker"),
                         s.get("composite_ticker"), s.get("name"),
                         bool(s.get("primary_listing")), s.get("figi"), active))
        next_page = data.get("next_page")
        if not next_page:
            break
    return pd.DataFrame(rows, columns=_SEC_COLS)


def _column_name(ticker: str, cik: int, dead: bool) -> str:
    tick = str(ticker or "").upper().replace("/", ".")
    return f"{tick}~{cik}" if dead else tick


def select_universe(
    securities: pd.DataFrame,
    companies: pd.DataFrame,
    our_ciks,
    delistings: pd.DataFrame | None = None,
    max_candidates: int = MAX_CANDIDATES,
    clip_grace_days: int | None = None,
) -> pd.DataFrame:
    """Pure join/selection: one price column per CIK, candidate securities ranked.

    Returns one row per (cik, candidate security): ``cik, column, security_id,
    ticker, name, active, priority, clip_date``. A company is *dead* when none of
    its securities is active; dead columns are ``TICKER~CIK``. ``clip_date`` is set
    only when ``clip_grace_days`` is given (see module docstring for why the ledger
    clip is off by default with by-id fetching).
    """
    comp = companies.dropna(subset=["cik", "company_id"]).copy()
    comp["cik"] = comp["cik"].astype(str).str.lstrip("0")
    comp = comp[comp["cik"].str.isdigit() & (comp["cik"] != "")]
    comp["cik"] = comp["cik"].astype(int)

    sec = securities.drop_duplicates("security_id").merge(
        comp[["company_id", "cik"]], on="company_id", how="inner"
    )
    sec = sec[sec["cik"].isin({int(c) for c in our_ciks})].copy()
    sec = sec[sec["ticker"].notna() & (sec["ticker"].astype(str).str.len() > 0)]
    if sec.empty:
        return pd.DataFrame(columns=["cik", "column", "security_id", "ticker", "name",
                                     "active", "priority", "clip_date"])

    # Rank candidates within each company: active first, then primary exchange
    # listing, then FIGI-carrying records (better-curated), deterministic tail.
    sec["_figi"] = sec["figi"].notna()
    sec = sec.sort_values(
        ["cik", "active", "primary_listing", "_figi", "security_id"],
        ascending=[True, False, False, False, True],
    )
    sec = sec.groupby("cik").head(max_candidates).copy()
    sec["priority"] = sec.groupby("cik").cumcount()

    alive = sec.groupby("cik")["active"].transform("any")
    # Living companies keep only ACTIVE securities: Intrinio already carries full
    # pre-rename history on the active record (META includes the FB era, AAL the
    # AMR era), so an inactive sibling adds nothing — except in a wipeout-relist,
    # where splicing it would fabricate a return across the bankruptcy boundary.
    sec = sec[~alive | sec["active"]].copy()
    alive = sec.groupby("cik")["active"].transform("any")
    sec["priority"] = sec.groupby("cik").cumcount()
    top_ticker = sec.groupby("cik")["ticker"].transform("first")
    sec["column"] = [
        _column_name(t, c, not a) for t, c, a in zip(top_ticker, sec["cik"], alive)
    ]

    # Two live companies must not share a column (defensive: shouldn't happen for
    # real US listings) — later claimants get the ~CIK suffix, keeping their own
    # TOP ticker for every candidate row so the company stays in ONE column.
    first_cik = sec.groupby("column")["cik"].transform("first")
    clash = alive & (sec["cik"] != first_cik)
    sec.loc[clash, "column"] = [
        _column_name(t, c, True)
        for t, c in zip(top_ticker[clash], sec.loc[clash, "cik"])
    ]

    sec["clip_date"] = pd.NaT
    if clip_grace_days is not None and delistings is not None and len(delistings):
        dmap = {
            int(c): pd.Timestamp(d) + pd.Timedelta(days=clip_grace_days)
            for c, d in zip(delistings["cik"], delistings["delist_date"])
        }
        dead = ~alive
        sec.loc[dead, "clip_date"] = sec.loc[dead, "cik"].map(dmap)

    out = sec[["cik", "column", "security_id", "ticker", "name", "active", "priority", "clip_date"]]
    return out.reset_index(drop=True)


def load_universe(path=UNIVERSE_PATH) -> pd.DataFrame:
    if not pd.io.common.file_exists(str(path)):
        return pd.DataFrame(columns=["cik", "column", "security_id", "ticker", "name",
                                     "active", "priority", "clip_date"])
    return pd.read_parquet(path)


def universe_ticker_map(path=UNIVERSE_PATH) -> dict[int, str]:
    """{cik: price-matrix column} for the panel's PIT join. Empty if no universe built."""
    uni = load_universe(path)
    if uni.empty:
        return {}
    top = uni.sort_values("priority").drop_duplicates("cik")
    return dict(zip(top["cik"].astype(int), top["column"]))
