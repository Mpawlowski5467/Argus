"""Value retest with TRUE point-in-time market cap: do E/P, B/M, S/P add honest OOS edge?

The original value gate (free-data era) was flat — but it ran on proxy caps over
ADJUSTED closes, which scale historical caps by future splits (look-ahead in the
denominator). Both blockers are now cleared: the unadjusted uclose backfill is
complete across all price columns (2026-07-02), and the PIT shares recipe is known
(CommonStockSharesOutstanding at the balance-sheet date, falling back to
WeightedAverageNumberOfSharesOutstandingBasic — ~85% coverage, AAPL-validated).
So this is the honest re-ask: cap = uclose(as-of date) x PIT shares from the filing,
yields joined per (date, cik), same panel, same harness as the momentum/reversal
tests (walk-forward AND CPCV — walk-forward-only wins have always reversed here).

    uv run python scripts/run_value_pit_test.py [--no-cpcv]

KNOWN LIMIT (recorded, not hidden): FSDS drops XBRL dimensions, so a dual-class
name's CommonStockSharesOutstanding max-abs pick is its LARGEST class, undercounting
total cap (GOOG-style). The wavg-basic fallback is consolidated and mostly avoids
this; the bias is toward NaN/undercount, never invented cap.

This NEVER touches the frozen artifact, the defaults, or the paper-forward book —
it only reports numbers. Promotion is a separate, gated step.
"""

import argparse
import glob

import duckdb
import numpy as np
import pandas as pd

from stockscan.concepts import WIDE_PATH
from stockscan.config import MIN_DOLLAR_VOLUME
from stockscan.edgar.fsds import FUNDAMENTALS_DIR
from stockscan.features import FEATURES, compute_features
from stockscan.fundamental_panel import VALUE_FEATURES, build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import DEFAULT_PARAMS, RANK_COLS, evaluate
from stockscan.panel import load_matrices_cached
from stockscan.prices import PRICES_DIR
from stockscan.validation import cpcv_splits, ic_summary, rank_ic

WINSOR = (0.01, 0.99)
VALUE_RANKS = [f"{f}_rank" for f in VALUE_FEATURES]

ARMS = {
    "baseline (10 fundamentals)": RANK_COLS,
    "+bm": RANK_COLS + ["bm_rank"],
    "+ep+bm+sp": RANK_COLS + VALUE_RANKS,
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Value (PIT true-cap) honest OOS test.")
    ap.add_argument("--no-cpcv", action="store_true",
                    help="skip the CPCV distribution (walk-forward only, faster)")
    return ap.parse_args(argv)


def build_pit_shares() -> pd.DataFrame:
    """Per-filing PIT shares from the raw FSDS fact ledger: balance-sheet-date
    CommonStockSharesOutstanding first, wavg-basic (annual duration) fallback.
    Keyed (cik, adsh) so it joins the wide table exactly one-to-one."""
    paths = sorted(glob.glob(str(FUNDAMENTALS_DIR / "*.parquet")))
    src = "read_parquet([" + ",".join(f"'{p}'" for p in paths) + "])"
    q = f"""
        SELECT cik, adsh,
            arg_max(value, abs(value)) FILTER (
                WHERE tag = 'CommonStockSharesOutstanding' AND qtrs = 0
                  AND ddate = period_end AND uom = 'shares') AS shares_out,
            arg_max(value, abs(value)) FILTER (
                WHERE tag = 'WeightedAverageNumberOfSharesOutstandingBasic' AND qtrs = 4
                  AND ddate = period_end AND uom = 'shares') AS shares_wavg
        FROM {src}
        WHERE form = '10-K'
        GROUP BY cik, adsh
    """
    df = duckdb.query(q).df()
    df["shares"] = df["shares_out"].fillna(df["shares_wavg"])
    # shares are counts: zero/negative picks are ledger noise, not information
    df.loc[df["shares"] <= 0, "shares"] = np.nan
    return df[["cik", "adsh", "shares"]]


def load_uclose(index_like: pd.DatetimeIndex) -> pd.DataFrame:
    """UNADJUSTED close matrix from the per-column store (slow path — research use).
    Reindexed to the adjusted matrix's calendar so `d in value_price.index` holds."""
    files = sorted(glob.glob(str(PRICES_DIR / "*.parquet")))
    src = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "], union_by_name=true)"
    df = duckdb.query(f"select ticker, date, uclose from {src} where uclose is not null").df()
    df["date"] = pd.to_datetime(df["date"])
    mat = df.pivot_table(index="date", columns="ticker", values="uclose").sort_index()
    return mat.reindex(index_like)


def cpcv_ic(panel, feature_cols, n_groups=10, k_test=2):
    """CPCV OOS rank-IC distribution for one feature set (mirrors run_momentum_test)."""
    import lightgbm as lgb

    dates = sorted(panel["date"].unique())
    ics = []
    for tr_d, te_d in cpcv_splits(dates, n_groups=n_groups, k_test=k_test):
        tr = panel[panel["date"].isin(tr_d)].dropna(subset=["label_excess"])
        te = panel[panel["date"].isin(te_d)]
        if len(tr) < 50 or te.empty:
            continue
        mdl = lgb.LGBMRegressor(**DEFAULT_PARAMS)
        mdl.fit(tr[feature_cols].fillna(0.5), tr["label_excess"])
        te = te[["date", "label_excess"]].assign(pred=mdl.predict(te[feature_cols].fillna(0.5)))
        te.attrs = {}
        ics.append(rank_ic(te, feature="pred").mean())
    return np.asarray(ics)


def main(argv=None) -> int:
    args = parse_args(argv)
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()

    shares = build_pit_shares()
    wide = wide.merge(shares, on=["cik", "adsh"], how="left")
    cov = wide["shares"].notna().mean()
    print(f"PIT shares coverage: {cov:.0%} of {len(wide):,} filings "
          f"(recipe: CommonStockSharesOutstanding, wavg-basic fallback)")

    feats = compute_features(wide)
    close, dv = load_matrices_cached()
    if close.empty:
        print("no prices; run scripts/ingest_prices.py first")
        return 1
    print("building unadjusted close matrix (slow path) ...")
    uclose = load_uclose(close.index)
    print(f"uclose: {uclose.shape[1]:,} columns, "
          f"{uclose.notna().any().mean():.0%} with any unadjusted print")

    # sanity anchor before any modeling: AAPL's latest PIT cap should be O($1T)
    aapl = feats[feats["cik"] == 320193].sort_values("period_end").tail(1)
    if len(aapl) and pd.notna(aapl["shares"].iloc[0]):
        px = uclose["AAPL"].dropna()
        if len(px):
            cap_t = aapl["shares"].iloc[0] * px.iloc[-1] / 1e12
            print(f"sanity: AAPL PIT cap at last print ≈ ${cap_t:.2f}T "
                  f"(shares {aapl['shares'].iloc[0] / 1e9:.2f}B x ${px.iloc[-1]:.0f})")

    tmap = universe_ticker_map() or None
    panel = build_fundamental_panel(
        feats, close, delistings=None, dollar_volume=dv, ticker_map=tmap,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR, value_price=uclose,
    )
    if panel.empty:
        print("empty panel")
        return 1

    n_dates = panel["date"].nunique()
    print(f"\npanel: {len(panel):,} rows  dates={n_dates}  ~{len(panel) // n_dates} names/date")
    for f in VALUE_FEATURES:
        print(f"  {f} coverage: {panel[f].notna().mean():.0%} of labeled rows")

    # is value orthogonal to the fundamental block, or a re-expression of it?
    corrs = [panel["bm_rank"].corr(panel[c]) for c in RANK_COLS]
    print(f"\nbm_rank mean |corr| vs 10 fundamental ranks: "
          f"{np.nanmean(np.abs(corrs)):.3f}  (max {np.nanmax(np.abs(corrs)):.3f})")

    print("\nsingle-feature rank IC (value standalone vs the fundamentals):")
    for f in VALUE_FEATURES + FEATURES:
        s = ic_summary(rank_ic(panel, feature=f"{f}_rank"))
        tag = "  <-- VALUE" if f in VALUE_FEATURES else ""
        print(f"  {f:<20} IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}{tag}")

    print("\nLightGBM walk-forward OOS (same panel, feature set varies):")
    for name, cols in ARMS.items():
        m = evaluate(panel, feature_cols=cols)
        print(f"  {name:<26} IC={m['mean_ic']:+.4f}  t_nw={m['t_nw']:+.2f}  "
              f"decile_spread={m['decile_spread']:+.4f}  (oos dates={m['oos_dates']})")

    if not args.no_cpcv:
        print("\nCPCV IC distribution (C(10,2)=45 purged combos per arm — the real test):")
        base_ics = None
        for name, cols in ARMS.items():
            ics = cpcv_ic(panel, cols)
            if name.startswith("baseline"):
                base_ics = ics
            delta = ""
            if base_ics is not None and not name.startswith("baseline") \
                    and len(ics) == len(base_ics):
                delta = f"  Δmean vs baseline {ics.mean() - base_ics.mean():+.4f}"
            print(f"  {name:<26} mean {ics.mean():+.4f}  5th pct {np.percentile(ics, 5):+.4f}  "
                  f"frac>0 {np.mean(ics > 0):.0%}{delta}")

    print(
        "\nHONEST READ: value is promotable ONLY if +value beats baseline on BOTH the\n"
        "walk-forward OOS IC/t AND the CPCV mean, without the 5th-pct/frac>0 collapsing.\n"
        "Every factor addition to date (momentum, reversal, low-vol, amihud, the 13-config\n"
        "salvage) won somewhere and died under CPCV — that is the bar, not the best cell."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
