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

import numpy as np
import pandas as pd

from .config import DELISTING_RETURN, LABEL_HORIZON_DAYS, MAX_STALE_DAYS, MIN_SECTOR_BUCKET
from .edgar.tickers import cik_to_ticker
from .features import FEATURES
from .panel import forward_return_to_last, momentum_6_1, momentum_12_1, month_end_dates
from .pit import assert_pit, available_date
from .sector import sic_division

# Price-derived features: computed as-of the rebalance date from the shared close
# matrix, NOT carried on the filing row like FEATURES. Point-in-time and
# survivorship-free by construction (only past closes; dead names keep their column).
PRICE_FEATURES = ["mom_12_1", "mom_6_1"]
EXTRA_PRICE_FEATURES = ["st_rev", "low_vol", "amihud"]

# Value features (EXPERIMENT plumbing, default-off like the price features): classic
# yields from true point-in-time market cap = UNADJUSTED close (as-of date) x PIT
# shares from the filing itself. Need the raw statement values + shares to survive
# prepare_features, and an unadjusted-close matrix passed as ``value_price`` — the
# adjusted close would scale cap by future splits (the exact trap the prior flat
# value gate fell into via proxy caps).
VALUE_FEATURES = ["ep", "bm", "sp"]
VALUE_RAWS = ("net_income", "equity", "revenue", "shares")

# Imputed terminal return by ledger reason, sourced from the locked config decision
# (DESIGN.md §10): Form 15 deregistration ("dereg") = going-dark -> -1.00; Form 25/25-NSE
# exchange delisting ("delist") -> distress haircut when no price survives. Injectable so
# the Phase-1 haircut sweep flows through.
REASON_RETURN = {"delist": DELISTING_RETURN["distress"], "dereg": DELISTING_RETURN["going_dark"]}


# --- shared transforms (train/serve parity invariant, DESIGN.md §2.2) ----------
# The panel build below and the serve path (stockscan.serve) both go through these
# four functions. Any feature-shaping logic added elsewhere breaks the parity test.

def prepare_features(features_df: pd.DataFrame, extra_cols: tuple = ()) -> pd.DataFrame:
    """Availability + sector prep shared by the TRAIN (panel) and SERVE paths.

    ``extra_cols`` lets an experiment carry additional per-filing columns (e.g. the
    raw statement values + shares behind the value features) through the pruning —
    default empty, so the production paths are byte-identical."""
    feats = features_df.copy()
    feats["available_date"] = available_date(feats["filed_date"])
    feats["sector"] = feats["sic"].map(sic_division)
    feats = feats.dropna(subset=["available_date"]).sort_values("available_date")
    meta_cols = [c for c in ("cik", "name", "fy", "sic", "period_end") if c in feats.columns]
    extras = [c for c in extra_cols if c in feats.columns and c not in FEATURES]
    return feats[[*meta_cols, "filed_date", "available_date", "sector", *FEATURES, *extras]]


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
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame | None = None,
    features: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Wide [date x ticker] matrix per price feature, computed once for the whole run."""
    features = features or PRICE_FEATURES
    ret = close.pct_change()
    mats: dict[str, pd.DataFrame] = {}
    for f in features:
        if f == "mom_12_1":
            mats[f] = momentum_12_1(close)
        elif f == "mom_6_1":
            mats[f] = momentum_6_1(close)
        elif f == "st_rev":
            mats[f] = close / close.shift(21) - 1.0
        elif f == "low_vol":
            mats[f] = -ret.rolling(126, min_periods=63).std()
        elif f == "amihud":
            if dollar_volume is None:
                raise ValueError("amihud price feature requires dollar_volume")
            mats[f] = (ret.abs() / dollar_volume.replace(0, np.nan)).rolling(21, min_periods=10).mean()
        else:
            raise ValueError(f"unknown price feature: {f}")
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
    liquidity_price: pd.DataFrame | None = None,
    min_dollar_volume: float | None = None,
    min_price: float = 1.0,
    winsorize: tuple[float, float] | None = None,
    price_features: bool = False,
    value_price: pd.DataFrame | None = None,
) -> pd.DataFrame:
    reason_return = reason_return or REASON_RETURN
    c2t = ticker_map if ticker_map is not None else cik_to_ticker()
    # Price features attach as-of each rebalance date; matrices built once up front.
    if isinstance(price_features, (list, tuple)):
        price_feature_names = list(price_features)
    elif price_features:
        price_feature_names = PRICE_FEATURES
    else:
        price_feature_names = []
    # Value features (default-off): need the raw statement values + shares on the
    # filing rows AND an unadjusted close matrix. Enabled only when both exist.
    value_feature_names = []
    if value_price is not None:
        missing = [c for c in VALUE_RAWS if c not in features_df.columns]
        if missing:
            raise ValueError(f"value_price given but features_df lacks {missing}")
        value_feature_names = list(VALUE_FEATURES)
    rank_features = FEATURES + price_feature_names + value_feature_names
    mom_mats = price_feature_matrices(close, dollar_volume, price_feature_names) \
        if price_feature_names else None
    dmap = {}
    if delistings is not None and len(delistings):
        for cik, dd, reason in zip(delistings["cik"], delistings["delist_date"], delistings["reason"]):
            dmap[int(cik)] = (pd.Timestamp(dd), reason)

    feats = prepare_features(features_df,
                             extra_cols=VALUE_RAWS if value_feature_names else ())

    # Terminal-aware label: a name that stops trading inside the window is labeled
    # with its real last-trade return, so (with delisted-inclusive prices) death
    # declines enter the label without any imputed haircut.
    fwd = forward_return_to_last(close, horizon)
    idx = close.index
    dv_med = dollar_volume.rolling(20, min_periods=10).median() if dollar_volume is not None else None
    liq_price = liquidity_price if liquidity_price is not None else close

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
            liquid = liquidity_mask(latest, d, liq_price, dv_med, min_dollar_volume, min_price)
            latest = latest[latest["imputed"] | liquid]

        # Price features attach as-of d (a month-end trading day in close.index),
        # then are ranked through the same add_sector_ranks the fundamentals use.
        if mom_mats is not None:
            latest = attach_price_features(latest, d, mom_mats)

        # Value yields as-of d: cap = unadjusted close x PIT shares from the filing
        # row itself. A name with no unadjusted print or no shares gets NaN — ranked
        # like any other missing fundamental, never imputed.
        if value_feature_names:
            px = latest["ticker"].map(value_price.loc[d]) \
                if d in value_price.index else np.nan
            cap = px * latest["shares"]
            with np.errstate(divide="ignore", invalid="ignore"):
                latest["ep"] = np.where(cap > 0, latest["net_income"] / cap, np.nan)
                latest["bm"] = np.where(cap > 0, latest["equity"] / cap, np.nan)
                latest["sp"] = np.where(cap > 0, latest["revenue"] / cap, np.nan)

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
