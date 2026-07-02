"""Assemble the point-in-time, survivorship-aware fundamental panel + forward label.

For each monthly as-of date we take each company's latest 10-K whose numbers were
already public (``available_date = filed + lag <= as_of``), keep only companies that
were ALIVE then (not delisted on/before the date, per the ledger) and not stale, attach
sector + ticker, and label with the 63-day forward excess return.

Survivorship correction: companies that DIE within the forward window but have no free
price series are not dropped -- they are re-injected with an imputed delisting return
(by reason), so the failures the model should learn are present in the cross-section.
This is the best achievable free-data correction; the residual gap (delisted names' full
price history) is measured, not hidden. See DESIGN.md.
"""

from __future__ import annotations

import pandas as pd

from .config import DELISTING_RETURN, LABEL_HORIZON_DAYS, MIN_SECTOR_BUCKET
from .edgar.tickers import cik_to_ticker
from .features import FEATURES
from .panel import forward_return_to_last, month_end_dates
from .pit import assert_pit, available_date
from .sector import sic_division

# Imputed terminal return by ledger reason, sourced from the locked config decision
# (DESIGN.md §10): Form 15 deregistration ("dereg") = going-dark -> -1.00; Form 25/25-NSE
# exchange delisting ("delist") -> distress haircut when no price survives. Injectable so
# the Phase-1 haircut sweep flows through.
REASON_RETURN = {"delist": DELISTING_RETURN["distress"], "dereg": DELISTING_RETURN["going_dark"]}


def build_fundamental_panel(
    features_df: pd.DataFrame,
    close: pd.DataFrame,
    delistings: pd.DataFrame | None = None,
    ticker_map: dict | None = None,
    horizon: int = LABEL_HORIZON_DAYS,
    max_stale_days: int = 550,
    min_names: int = 30,
    reason_return: dict | None = None,
    dollar_volume: pd.DataFrame | None = None,
    min_dollar_volume: float | None = None,
    min_price: float = 1.0,
    winsorize: tuple[float, float] | None = None,
) -> pd.DataFrame:
    reason_return = reason_return or REASON_RETURN
    c2t = ticker_map if ticker_map is not None else cik_to_ticker()
    dmap = {}
    if delistings is not None and len(delistings):
        for cik, dd, reason in zip(delistings["cik"], delistings["delist_date"], delistings["reason"]):
            dmap[int(cik)] = (pd.Timestamp(dd), reason)

    feats = features_df.copy()
    feats["available_date"] = available_date(feats["filed_date"])
    feats["sector"] = feats["sic"].map(sic_division)
    feats = feats.dropna(subset=["available_date"]).sort_values("available_date")
    feats = feats[["cik", "filed_date", "available_date", "sector", *FEATURES]]

    # Terminal-aware label: a name that stops trading inside the window is labeled
    # with its real last-trade return, so (with delisted-inclusive prices) death
    # declines enter the label without any imputed haircut.
    fwd = forward_return_to_last(close, horizon)
    idx = close.index
    dv_med = dollar_volume.rolling(20, min_periods=10).median() if dollar_volume is not None else None

    rows, coverage = [], []
    for d in month_end_dates(close.index):
        if d not in fwd.index:
            continue
        avail = feats[feats["available_date"] <= d]
        latest = avail.drop_duplicates("cik", keep="last").copy()
        latest = latest[(d - latest["available_date"]).dt.days <= max_stale_days]  # not stale
        if latest.empty:
            continue
        assert_pit(latest, d, filed_col="filed_date")  # tripwire: no future filing by construction

        dl = latest["cik"].map(lambda c: dmap.get(c, (pd.NaT, None)))
        latest["delist_date"] = [x[0] for x in dl]
        latest["reason"] = [x[1] for x in dl]
        latest = latest[~(latest["delist_date"] <= d)]  # alive at d

        latest["ticker"] = latest["cik"].map(c2t)
        real = latest["ticker"].map(fwd.loc[d])  # real forward return where priced
        latest["label"] = real

        # Forward window = the actual 63rd trading day (positional), not a calendar guess.
        loc = idx.get_loc(d)
        window_end = idx[min(loc + horizon, len(idx) - 1)]
        dies = latest["delist_date"].notna() & (latest["delist_date"] <= window_end)
        # Going-dark ('dereg') names go to ~0 -> OVERRIDE any survivorship-artifact real return
        # with the haircut. Exchange delistings ('delist', often M&A) keep their real price if
        # they still traded; impute only when no price survived.
        to_impute = dies & ((latest["reason"] == "dereg") | latest["label"].isna())
        latest.loc[to_impute, "label"] = latest.loc[to_impute, "reason"].map(reason_return)
        latest["imputed"] = to_impute

        # Liquidity filter: keep imputed failures (their missing price is a data gap, not an
        # illiquidity signal) plus priced names clearing the dollar-volume and price floors.
        if dv_med is not None and min_dollar_volume:
            tk = latest["ticker"]
            liquid = (tk.map(dv_med.loc[d]) >= min_dollar_volume) & (
                tk.map(close.loc[d]) >= min_price
            )
            latest = latest[latest["imputed"] | liquid.fillna(False)]

        labeled = latest.dropna(subset=["label"])
        if len(labeled) < min_names:
            continue
        labeled = labeled.copy()
        labeled["date"] = d
        rows.append(labeled)
        coverage.append(
            {"date": d, "universe": len(latest),
             "priced": int((~latest["imputed"]).sum()),
             "imputed": int(latest["imputed"].sum()), "labeled": len(labeled)}
        )

    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    if winsorize:
        lo, hi = winsorize
        panel["label"] = panel.groupby("date")["label"].transform(
            lambda s: s.clip(s.quantile(lo), s.quantile(hi))
        )
    panel["label_excess"] = panel["label"] - panel.groupby("date")["label"].transform("mean")
    # Rank-normalize each feature within (date x sector); fall back to date-only where a
    # sector bucket is too thin to rank reliably.
    for f in FEATURES:
        sec = panel.groupby(["date", "sector"])[f]
        sec_rank = sec.rank(pct=True)
        bucket = sec.transform("count")
        date_rank = panel.groupby("date")[f].rank(pct=True)
        panel[f"{f}_rank"] = sec_rank.where(bucket >= MIN_SECTOR_BUCKET, date_rank)
    panel.attrs["coverage"] = pd.DataFrame(coverage)
    return panel
