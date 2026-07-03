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

from .config import DELISTING_RETURN, LABEL_HORIZON_DAYS, MAX_STALE_DAYS, MIN_SECTOR_BUCKET
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


# --- shared transforms (train/serve parity invariant, DESIGN.md §2.2) ----------
# The panel build below and the serve path (stockscan.serve) both go through these
# four functions. Any feature-shaping logic added elsewhere breaks the parity test.

def prepare_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Availability + sector prep shared by the TRAIN (panel) and SERVE paths."""
    feats = features_df.copy()
    feats["available_date"] = available_date(feats["filed_date"])
    feats["sector"] = feats["sic"].map(sic_division)
    feats = feats.dropna(subset=["available_date"]).sort_values("available_date")
    meta_cols = [c for c in ("cik", "name", "fy", "sic", "period_end") if c in feats.columns]
    return feats[[*meta_cols, "filed_date", "available_date", "sector", *FEATURES]]


def pit_snapshot(feats: pd.DataFrame, as_of, max_stale_days: int = MAX_STALE_DAYS) -> pd.DataFrame:
    """Latest filing per company that was PUBLIC at ``as_of`` and not stale.

    ``feats`` must come from :func:`prepare_features` (sorted by available_date).
    assert_pit is the build-failing tripwire on every snapshot, train or serve.
    """
    as_of = pd.Timestamp(as_of)
    avail = feats[feats["available_date"] <= as_of]
    latest = avail.drop_duplicates("cik", keep="last").copy()
    latest = latest[(as_of - latest["available_date"]).dt.days <= max_stale_days]
    if not latest.empty:
        assert_pit(latest, as_of, filed_col="filed_date")
    return latest


def liquidity_mask(
    latest: pd.DataFrame,
    price_date,
    close: pd.DataFrame,
    dv_med: pd.DataFrame,
    min_dollar_volume: float,
    min_price: float = 1.0,
) -> pd.Series:
    """Tradable-universe mask: 20d-median dollar volume and price floors at ``price_date``."""
    tk = latest["ticker"]
    liquid = (tk.map(dv_med.loc[price_date]) >= min_dollar_volume) & (
        tk.map(close.loc[price_date]) >= min_price
    )
    return liquid.fillna(False)


def add_sector_ranks(cross: pd.DataFrame, min_sector_bucket: int = MIN_SECTOR_BUCKET) -> pd.DataFrame:
    """Rank-normalize FEATURES within sector for ONE date's cross-section.

    Falls back to a cross-section-wide rank where a sector bucket is too thin to
    rank reliably. Ranks are computed over the full known-at-date universe --
    never conditioned on whether a name later got a label (that would let the
    future pick the rank basis).
    """
    out = cross.copy()
    for f in FEATURES:
        g = out.groupby("sector")[f]
        sec_rank = g.rank(pct=True)
        bucket = g.transform("count")
        date_rank = out[f].rank(pct=True)
        out[f"{f}_rank"] = sec_rank.where(bucket >= min_sector_bucket, date_rank)
    return out


def build_fundamental_panel(
    features_df: pd.DataFrame,
    close: pd.DataFrame,
    delistings: pd.DataFrame | None = None,
    ticker_map: dict | None = None,
    horizon: int = LABEL_HORIZON_DAYS,
    max_stale_days: int = MAX_STALE_DAYS,
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

    feats = prepare_features(features_df)

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
        latest = pit_snapshot(feats, d, max_stale_days)
        if latest.empty:
            continue

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
            liquid = liquidity_mask(latest, d, close, dv_med, min_dollar_volume, min_price)
            latest = latest[latest["imputed"] | liquid]

        # Ranks over the full known-at-date universe, BEFORE the label drop (the serve
        # path ranks the identical universe -- there are no labels at serve time).
        latest = add_sector_ranks(latest, MIN_SECTOR_BUCKET)
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
    panel.attrs["coverage"] = pd.DataFrame(coverage)
    return panel
