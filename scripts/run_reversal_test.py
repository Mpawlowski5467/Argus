"""Reversal / low-vol / illiquidity test: do these price features add honest OOS edge?

Direct follow-up to run_momentum_test.py, which concluded momentum is a real but
NON-additive signal (RESULTS.md). Same disciplined harness, three new Tier-0 price
features attached to the SAME panel (they never filter rows, so every arm scores the
identical cross-section — only the feature set varies):

    baseline  = the 10 fundamental ranks (what the frozen artifact uses)
    +st_rev   = baseline + short-term reversal (raw trailing 21d return; IC is NEGATIVE
                by design — recent losers outperform — the tree learns the flip)
    +low_vol  = baseline + NEGATIVE trailing 126d realized vol (low-vol anomaly)
    +amihud   = baseline + Amihud illiquidity (|ret| / dollar-volume; IC POSITIVE by design)
    +all3     = baseline + all three at once

For each: walk-forward OOS (IC, Newey-West t, decile spread) AND the CPCV IC
distribution — because the bar a new feature must clear is BOTH the walk-forward number
AND the CPCV mean (momentum, and the 13-config salvage before it, won walk-forward and
died under CPCV). rank_ic is SIGNED, so the standalone ICs are reported with their real
sign; the model-level arms are what actually decide promote / don't-promote.

HONEST CAVEAT: short-term reversal is a ~1-month effect, but the label is the 63-day
(3-month) forward return — the reversal edge may be muddied at this horizon. That is
part of what this test measures, not a reason to discount a clean win if one appears.

  uv run python scripts/run_reversal_test.py [--impute] [--no-cpcv]

This NEVER touches the frozen artifact, the defaults, or the paper-forward book — it
only reports numbers. Promotion is a separate, gated step (STOP and report first).
"""

import argparse

import duckdb
import numpy as np

from stockscan.concepts import WIDE_PATH
from stockscan.config import MIN_DOLLAR_VOLUME
from stockscan.edgar.delistings import load_delistings
from stockscan.features import FEATURES, compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import DEFAULT_PARAMS, RANK_COLS, evaluate
from stockscan.panel import load_matrices_cached
from stockscan.validation import cpcv_splits, ic_summary, rank_ic

WINSOR = (0.01, 0.99)
# The reversal-test arms. These name their columns explicitly rather than sweeping the
# shared PRICE_FEATURES registry (which also carries momentum) so the arms stay focused.
NEW_FEATURES = ["st_rev", "low_vol", "amihud"]
EXPECTED_SIGN = {"st_rev": "-", "low_vol": "+", "amihud": "+"}  # raw single-feature IC direction

ARMS = {
    "baseline (10 fundamentals)": RANK_COLS,
    "+st_rev": RANK_COLS + ["st_rev_rank"],
    "+low_vol": RANK_COLS + ["low_vol_rank"],
    "+amihud": RANK_COLS + ["amihud_rank"],
    "+all3": RANK_COLS + [f"{f}_rank" for f in NEW_FEATURES],
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Reversal / low-vol / illiquidity honest OOS test.")
    ap.add_argument("--impute", action="store_true",
                    help="use the delisting-return imputation (default: no-impute, the "
                         "trustworthy config — dead names enter via real terminal returns)")
    ap.add_argument("--no-cpcv", action="store_true",
                    help="skip the CPCV distribution (walk-forward only, faster)")
    return ap.parse_args(argv)


def cpcv_ic(panel, feature_cols, n_groups=10, k_test=2):
    """CPCV OOS rank-IC distribution for one feature set (mirrors run_phase3)."""
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
    feats = compute_features(wide)
    close, dv = load_matrices_cached()
    if close.empty:
        print("no prices; run scripts/ingest_prices.py first")
        return 1
    tmap = universe_ticker_map() or None
    delistings = load_delistings() if args.impute else None

    print(f"mode: {'with imputation' if args.impute else 'NO-IMPUTE (real terminal returns)'}"
          f"  |  ticker map: {'intrinio (' + str(len(tmap)) + ' ciks)' if tmap else 'EDGAR'}")
    # dollar_volume=dv is REQUIRED for amihud (it divides |ret| by dollar volume).
    panel = build_fundamental_panel(
        feats, close, delistings=delistings, dollar_volume=dv, ticker_map=tmap,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR, price_features=NEW_FEATURES,
    )
    if panel.empty:
        print("empty panel")
        return 1

    n_dates = panel["date"].nunique()
    print(f"panel: {len(panel):,} rows  dates={n_dates}  ~{len(panel) // n_dates} names/date")
    for f in NEW_FEATURES:
        cov = panel[f].notna().mean()
        print(f"  {f} coverage: {cov:.0%} of labeled rows")

    # Orthogonality: are the new features independent of the fundamental block?
    print("\northogonality to the 10 fundamental ranks (mean |corr|, max |corr|):")
    for f in NEW_FEATURES:
        corrs = [panel[f"{f}_rank"].corr(panel[c]) for c in RANK_COLS]
        m = np.nanmean(np.abs(corrs))
        print(f"  {f:<10} mean |corr| {m:.3f}  (max {np.nanmax(np.abs(corrs)):.3f}) "
              f"-> {'orthogonal' if m < 0.1 else 'some overlap'}")

    print("\nsingle-feature rank IC (SIGNED — expected sign in parens; fundamentals for scale):")
    for f in NEW_FEATURES:
        s = ic_summary(rank_ic(panel, feature=f"{f}_rank"))
        print(f"  {f:<20} IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}"
              f"   <-- NEW (expect {EXPECTED_SIGN[f]})")
    for f in FEATURES:
        s = ic_summary(rank_ic(panel, feature=f"{f}_rank"))
        print(f"  {f:<20} IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}")

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
            if base_ics is not None and not name.startswith("baseline") and len(ics) == len(base_ics):
                delta = f"  Δmean vs baseline {ics.mean() - base_ics.mean():+.4f}"
            print(f"  {name:<26} mean {ics.mean():+.4f}  5th pct {np.percentile(ics, 5):+.4f}  "
                  f"frac>0 {np.mean(ics > 0):.0%}{delta}")

    print(
        "\nHONEST READ: a feature is promotable ONLY if its arm beats baseline on BOTH the\n"
        "walk-forward OOS IC/t AND the CPCV mean, without collapsing the CPCV 5th-pct/frac>0.\n"
        "A negative standalone IC (st_rev) is fine — the tree learns the sign — but a\n"
        "walk-forward win that dies under CPCV is exactly the failure mode momentum and the\n"
        "13-config salvage hit. Do NOT promote on the walk-forward number alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
