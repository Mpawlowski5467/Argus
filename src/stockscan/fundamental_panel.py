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
from .panel import (
    amihud,
    forward_return_to_last,
    low_vol,
    momentum_6_1,
    momentum_12_1,
    month_end_dates,
    short_term_reversal,
)
from .pit import assert_pit, available_date
from .sector import sic_division

# Price-derived features: computed as-of the rebalance date from the shared close
# (and dollar-volume) matrices, NOT carried on the filing row like FEATURES. Point-in-time
# and survivorship-free by construction (only past prices; dead names keep their column).
# This is the shared registry the head-to-head research scripts draw arms from; the
# shipped model uses NONE of them (build_fundamental_panel(price_features=False) default).
# ``amihud`` needs the dollar-volume matrix, so it is only built when ``dv`` is supplied.
PRICE_FEATURES = ["mom_12_1", "mom_6_1", "st_rev", "low_vol", "amihud"]

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


def add_sector_ranks(
    cross: pd.DataFrame,
    min_sector_bucket: int = MIN_SECTOR_BUCKET,
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """Rank-normalize ``features`` within sector for ONE date's cross-section.

    Falls back to a cross-section-wide rank where a sector bucket is too thin to
    rank reliably. Ranks are computed over the full known-at-date universe --
    never conditioned on whether a name later got a label (that would let the
    future pick the rank basis). ``features`` defaults to the fundamentals; the
    caller widens it to ``FEATURES + PRICE_FEATURES`` when price features are on,
    so momentum is ranked through the exact same code the fundamentals are.
    """
    out = cross.copy()
    for f in features:
        g = out.groupby("sector")[f]
        sec_rank = g.rank(pct=True)
        bucket = g.transform("count")
        date_rank = out[f].rank(pct=True)
        out[f"{f}_rank"] = sec_rank.where(bucket >= min_sector_bucket, date_rank)
    return out


def price_feature_matrices(
    close: pd.DataFrame, dv: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Wide [date x ticker] matrix per price feature, computed once for the whole run.

    ``dv`` (dollar volume) is optional: without it the volume-based ``amihud`` matrix
    is skipped (the price-only features still build). Callers derive the rank list from
    the returned keys, so a skipped feature is simply never attached or ranked.
    """
    mats = {
        "mom_12_1": momentum_12_1(close),
        "mom_6_1": momentum_6_1(close),
        "st_rev": short_term_reversal(close),
        "low_vol": low_vol(close),
    }
    if dv is not None:
        mats["amihud"] = amihud(close, dv)
    return mats


def attach_price_features(
    cross: pd.DataFrame, price_date, mom_mats: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Attach each price feature's as-of-``price_date`` value, keyed by price column.

    Shared by the TRAIN (panel) and SERVE paths so a name's momentum vector is
    identical both sides. ``cross['ticker']`` is the price column (dead names are
    ``TICKER~CIK``), so delisted names map through unchanged. A name with too little
    history — or no quote on ``price_date`` — gets NaN, handled downstream like any
    thin fundamental (ranked where present, else filled at score time).
    """
    out = cross.copy()
    for name, mat in mom_mats.items():
        row = mat.loc[price_date] if price_date in mat.index else pd.Series(dtype=float)
        out[name] = out["ticker"].map(row)
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
    price_features: bool = False,
) -> pd.DataFrame:
    reason_return = reason_return or REASON_RETURN
    c2t = ticker_map if ticker_map is not None else cik_to_ticker()
    # Price features attach as-of each rebalance date; matrices built once up front.
    # rank_features follows the matrices ACTUALLY built (amihud is skipped without dv),
    # so add_sector_ranks never ranks a column that was never attached.
    mom_mats = price_feature_matrices(close, dollar_volume) if price_features else None
    rank_features = FEATURES + list(mom_mats) if mom_mats else FEATURES
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

        # Price features attach as-of d (a month-end trading day in close.index),
        # then are ranked through the same add_sector_ranks the fundamentals use.
        if mom_mats is not None:
            latest = attach_price_features(latest, d, mom_mats)

        # Ranks over the full known-at-date universe, BEFORE the label drop (the serve
        # path ranks the identical universe -- there are no labels at serve time).
        latest = add_sector_ranks(latest, MIN_SECTOR_BUCKET, features=rank_features)
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
