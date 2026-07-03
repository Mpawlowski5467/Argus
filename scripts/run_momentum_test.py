"""Momentum feature test: does 12-1 / 6-1 price momentum add honest OOS edge?

The model today is fundamentals-only (10 ratios, no price features). DESIGN.md §4
lists momentum (12-1, 6-1) in the intended set, but it was never wired in. This is
the cheapest honest test of that gap: build ONE panel with the momentum ranks
attached (rows are identical to the fundamentals-only panel — momentum never filters),
then score three feature sets on the SAME panel so the comparison is apples-to-apples:

    baseline    = the 10 fundamental ranks (what the frozen artifact uses)
    +mom12      = baseline + 12-1 momentum
    +mom12+mom6 = baseline + 12-1 + 6-1 momentum

For each: walk-forward OOS (IC, Newey-West t, decile spread) AND the CPCV IC
distribution — because prior feature adds here won walk-forward and REVERSED under
CPCV (that is the bar a new feature must clear, not the walk-forward number).

  uv run python scripts/run_momentum_test.py [--impute] [--no-cpcv]

This NEVER touches the frozen artifact, the defaults, or the paper-forward book —
it only reports numbers. Promotion is a separate, gated step.
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
# Momentum-only arms. The shared PRICE_FEATURES registry now also carries reversal /
# low-vol / amihud (see scripts/run_reversal_test.py), so this test names its momentum
# columns explicitly rather than deriving them from PRICE_FEATURES.
MOM_FEATURES = ["mom_12_1", "mom_6_1"]
MOM_RANKS = [f"{f}_rank" for f in MOM_FEATURES]  # ["mom_12_1_rank", "mom_6_1_rank"]

ARMS = {
    "baseline (10 fundamentals)": RANK_COLS,
    "+mom12": RANK_COLS + ["mom_12_1_rank"],
    "+mom12+mom6": RANK_COLS + MOM_RANKS,
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Momentum feature honest OOS test.")
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
    panel = build_fundamental_panel(
        feats, close, delistings=delistings, dollar_volume=dv, ticker_map=tmap,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR, price_features=True,
    )
    if panel.empty:
        print("empty panel")
        return 1

    n_dates = panel["date"].nunique()
    print(f"panel: {len(panel):,} rows  dates={n_dates}  ~{len(panel) // n_dates} names/date")
    for f in MOM_FEATURES:
        cov = panel[f].notna().mean()
        print(f"  {f} coverage: {cov:.0%} of labeled rows")

    # Orthogonality check: is momentum actually independent of the fundamental block?
    # (my whole case for it is that it is.) Mean |corr| of the mom rank vs each fund rank.
    corrs = [panel["mom_12_1_rank"].corr(panel[c]) for c in RANK_COLS]
    print(f"\nmom_12_1_rank mean |corr| vs 10 fundamental ranks: "
          f"{np.nanmean(np.abs(corrs)):.3f}  (max {np.nanmax(np.abs(corrs)):.3f}) "
          f"-> {'orthogonal' if np.nanmean(np.abs(corrs)) < 0.1 else 'some overlap'}")

    print("\nsingle-feature rank IC (momentum standalone vs the fundamentals):")
    for f in MOM_FEATURES + FEATURES:
        s = ic_summary(rank_ic(panel, feature=f"{f}_rank"))
        tag = "  <-- MOMENTUM" if f in MOM_FEATURES else ""
        print(f"  {f:<20} IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}{tag}")

    print("\nLightGBM walk-forward OOS (same panel, feature set varies):")
    wf = {}
    for name, cols in ARMS.items():
        m = evaluate(panel, feature_cols=cols)
        wf[name] = m
        print(f"  {name:<26} IC={m['mean_ic']:+.4f}  t_nw={m['t_nw']:+.2f}  "
              f"decile_spread={m['decile_spread']:+.4f}  (oos dates={m['oos_dates']})")

    if not args.no_cpcv:
        print("\nCPCV IC distribution (C(10,2)=45 purged combos per arm — the real test):")
        base_ics = None
        for name, cols in ARMS.items():
            ics = cpcv_ic(panel, cols)
            if base_ics is None:
                base_ics = ics if name.startswith("baseline") else base_ics
            delta = ""
            if base_ics is not None and not name.startswith("baseline") and len(ics) == len(base_ics):
                delta = f"  Δmean vs baseline {ics.mean() - base_ics.mean():+.4f}"
            print(f"  {name:<26} mean {ics.mean():+.4f}  5th pct {np.percentile(ics, 5):+.4f}  "
                  f"frac>0 {np.mean(ics > 0):.0%}{delta}")

    print(
        "\nHONEST READ: momentum is promotable ONLY if +mom beats baseline on BOTH the\n"
        "walk-forward OOS IC/t AND the CPCV mean (not just walk-forward), and its CPCV\n"
        "5th-pct/frac>0 don't collapse. A walk-forward-only win that dies under CPCV is\n"
        "exactly the failure mode the last 13-config salvage hit — do NOT promote it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
