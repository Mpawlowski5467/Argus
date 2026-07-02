"""Phase 3 gate: does the NET-of-cost edge survive real trading mechanics?

Signals are the purged walk-forward OOS model scores (never the in-sample frozen
artifact). Backtests execute at next-bar open with liquidity-scaled costs, hysteresis
membership, and short-side borrow realism, on the honest no-impute panel.

Gate (DESIGN.md §8): net-of-cost edge survives turnover and the liquidity filter and
is NOT concentrated in untradeable micro-caps. Long-only and long/short net curves
are reported side-by-side (§6 mandate); if the short leg doesn't add net of borrow,
the verdict says to drop it.

  uv run python scripts/run_phase3.py [--cpcv]   # --cpcv adds the 45-fit CPCV distribution
"""

import argparse

import duckdb
import numpy as np
import pandas as pd

from stockscan.backtest import run_backtest
from stockscan.concepts import WIDE_PATH
from stockscan.config import BORROW_TIERS_BPS, MIN_DOLLAR_VOLUME
from stockscan.features import compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import DEFAULT_PARAMS, RANK_COLS, walk_forward_predict
from stockscan.panel import load_matrices
from stockscan.validation import cpcv_splits, ic_summary, pbo_cscv, rank_ic

WINSOR = (0.01, 0.99)
BUCKETS = ((1e6, 5e6, "1-5M"), (5e6, 25e6, "5-25M"), (25e6, np.inf, ">25M"))


def _fmt(s: dict) -> str:
    return (f"net CAGR {s['cagr_net']:+.2%}  gross {s['cagr_gross']:+.2%}  "
            f"Sharpe {s['sharpe_net']:+.2f}  vol {s['ann_vol']:.1%}  "
            f"maxDD {s['max_drawdown']:.1%}  turn {s['ann_turnover']:.1f}x/yr  "
            f"nL {s['avg_n_long']:.0f} nS {s['avg_n_short']:.0f}")


def liquidity_bucket_spreads(preds: pd.DataFrame, dv_med: pd.DataFrame) -> pd.DataFrame:
    """Per-liquidity-bucket decile spread of realized labels by prediction."""
    rows = []
    for d, g in preds.groupby("date"):
        dv_at = dv_med.loc[dv_med.index.asof(d)]
        g = g.assign(dv=g["ticker"].map(dv_at))
        for lo, hi, name in BUCKETS:
            b = g[(g["dv"] >= lo) & (g["dv"] < hi)].dropna(subset=["label_excess", "pred"])
            if len(b) < 20:
                continue
            r = b["pred"].rank(pct=True)
            spread = b.loc[r >= 0.9, "label_excess"].mean() - b.loc[r <= 0.1, "label_excess"].mean()
            rows.append({"bucket": name, "date": d, "spread": spread, "n": len(b)})
    df = pd.DataFrame(rows)
    return df.groupby("bucket").agg(
        mean_spread=("spread", "mean"), dates=("spread", "size"), avg_names=("n", "mean")
    ).reindex([b[2] for b in BUCKETS])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3 gate: backtest + signal mechanics.")
    ap.add_argument("--cpcv", action="store_true",
                    help="also run the CPCV distribution (45 LightGBM fits, slow)")
    args = ap.parse_args(argv)

    print("assembling honest panel + OOS walk-forward scores ...")
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, dv, opn = load_matrices(with_open=True)
    if close.empty:
        print("no prices on disk")
        return 1
    tmap = universe_ticker_map() or None
    panel = build_fundamental_panel(
        feats, close, delistings=None, ticker_map=tmap, dollar_volume=dv,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR,
    )
    preds = walk_forward_predict(panel, id_cols=("cik", "ticker", "sector"))
    if preds.empty:
        print("no OOS predictions")
        return 1
    dv_med = dv.rolling(20, min_periods=10).median()
    scores = preds[["date", "ticker", "pred"]]
    print(f"OOS dates: {preds['date'].nunique()}  "
          f"({preds['date'].min().date()} -> {preds['date'].max().date()})")

    # ---- the headline backtests (§6: LO and L/S side-by-side, net of everything)
    common = dict(close=close, opn=opn, dv_med=dv_med)
    bt_uni = run_backtest(scores, mode="universe", **common)
    bt_lo = run_backtest(scores, mode="long_only", **common)
    bt_ls = run_backtest(scores, mode="long_short", **common)
    bt_hard = run_backtest(scores, mode="long_only", enter=0.10, exit=0.10, **common)

    s_uni, s_lo, s_ls, s_hard = (b.summary() for b in (bt_uni, bt_lo, bt_ls, bt_hard))
    print(f"\n[data hygiene: {bt_uni.config['n_break_days']:,} scale-break days repaired "
          f"across {bt_uni.config['n_break_names']:,} series (mis-adjusted corporate "
          f"actions; real crashes kept); {bt_uni.config['n_masked_prints']:,} sub-penny "
          f"prints masked as quantization noise]")
    print("== backtests (monthly rebalance, next-open exec, liquidity-scaled costs) ==")
    print(f"universe EW (benchmark):  {_fmt(s_uni)}")
    print(f"long-only  (hysteresis):  {_fmt(s_lo)}")
    print(f"long-only  (hard decile): {_fmt(s_hard)}")
    print(f"long/short (hysteresis):  {_fmt(s_ls)}")
    lo_excess = s_lo["cagr_net"] - s_uni["cagr_net"]
    print(f"\nlong-only NET excess over universe EW: {lo_excess:+.2%}/yr")
    print(f"hysteresis turnover saving vs hard decile: "
          f"{s_hard['ann_turnover'] - s_lo['ann_turnover']:+.1f}x/yr "
          f"(net CAGR {s_lo['cagr_net'] - s_hard['cagr_net']:+.2%})")

    # ---- cost + borrow stress
    print("\n== cost stress (long-only net CAGR) ==")
    for scale in (0.5, 1.0, 2.0):
        s = run_backtest(scores, mode="long_only", cost_scale=scale, **common).summary()
        print(f"  {scale:.1f}x costs: {s['cagr_net']:+.2%}  (Sharpe {s['sharpe_net']:+.2f})")
    borrow2x = tuple((f, b * 2) for f, b in BORROW_TIERS_BPS)
    s = run_backtest(scores, mode="long_short", borrow_tiers=borrow2x, **common).summary()
    print(f"== borrow stress: L/S at 2x borrow: net CAGR {s['cagr_net']:+.2%} "
          f"(Sharpe {s['sharpe_net']:+.2f})")

    # ---- where does the edge live? (micro-cap concentration check)
    print("\n== decile spread by liquidity bucket (realized 63d label, OOS preds) ==")
    buckets = liquidity_bucket_spreads(preds, dv_med)
    print(buckets.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 1 else f"{x:.0f}"))

    fin = preds[preds["sector"] == "Finance"]
    nonfin = preds[preds["sector"] != "Finance"]
    for name, sub in (("financials", fin), ("non-financials", nonfin)):
        s = ic_summary(rank_ic(sub, feature="pred", label="label_excess"))
        print(f"IC {name}: {s['mean_ic']:+.4f} (t_nw {s['t_nw']:+.2f}, n={s['n']})")

    # ---- PBO over the variant grid we actually tried
    print("\n== PBO (CSCV) over the strategy-variant grid ==")
    variants = {}
    for mode in ("long_only", "long_short"):
        for enter, exit_ in ((0.10, 0.10), (0.10, 0.25), (0.20, 0.40)):
            for weighting in ("equal", "rank"):
                tag = "LO" if mode == "long_only" else "LS"
                key = f"{tag}_{int(enter*100)}_{int(exit_*100)}_{weighting[:2]}"
                bt = run_backtest(scores, mode=mode, enter=enter, exit=exit_,
                                  weighting=weighting, **common)
                variants[key] = bt.monthly_returns()
    trials = pd.DataFrame(variants).dropna(how="all")
    pbo_all = pbo_cscv(trials, n_blocks=16)
    # gate on the family we would actually select from: with the short book dropped
    # (§6), the selection universe is the 6 long-only variants — PBO over all 12 is
    # structurally flattered because the L/S half never wins in-sample.
    lo_trials = trials[[c for c in trials.columns if c.startswith("LO")]]
    pbo = pbo_cscv(lo_trials, n_blocks=16)
    print(f"all {trials.shape[1]} trials:      PBO={pbo_all['pbo']:.2f} "
          f"(lambda_mean {pbo_all['lambda_mean']:+.2f}, {pbo_all['n_combos']} combos)")
    print(f"long-only family ({lo_trials.shape[1]}): PBO={pbo['pbo']:.2f} "
          f"(lambda_mean {pbo['lambda_mean']:+.2f}, {pbo['n_combos']} combos) <- gated")

    # ---- CPCV distribution (optional, slow)
    if args.cpcv:
        import lightgbm as lgb
        print("\n== CPCV: IC distribution over 45 purged combinations ==")
        dates = sorted(panel["date"].unique())
        ics = []
        for i, (tr_d, te_d) in enumerate(cpcv_splits(dates, n_groups=10, k_test=2)):
            tr = panel[panel["date"].isin(tr_d)].dropna(subset=["label_excess"])
            te = panel[panel["date"].isin(te_d)]
            mdl = lgb.LGBMRegressor(**DEFAULT_PARAMS)
            mdl.fit(tr[RANK_COLS].fillna(0.5), tr["label_excess"])
            te = te[["date", "label_excess"]].assign(
                pred=mdl.predict(te[RANK_COLS].fillna(0.5)))
            te.attrs = {}
            ics.append(rank_ic(te, feature="pred").mean())
            if (i + 1) % 9 == 0:
                print(f"  [{i + 1}/45] running mean IC {np.mean(ics):+.4f}")
        ics = np.asarray(ics)
        print(f"CPCV mean IC {ics.mean():+.4f}  5th pct {np.percentile(ics, 5):+.4f}  "
              f"95th {np.percentile(ics, 95):+.4f}  frac>0 {np.mean(ics > 0):.0%}")

    # ---- gate verdict
    print("\n== PHASE-3 GATE ==")
    top_bucket_ok = bool(buckets.loc[">25M", "mean_spread"] > 0)
    checks = {
        "long-only net beats universe EW": lo_excess > 0,
        "long-only net positive at 2x costs": run_backtest(
            scores, mode="long_only", cost_scale=2.0, **common).summary()["cagr_net"] > 0,
        "edge present in the most-liquid bucket (>25M ADV)": top_bucket_ok,
        "edge not micro-cap-only": top_bucket_ok or bool(buckets.loc["5-25M", "mean_spread"] > 0),
        "PBO below 0.5": bool(pbo["pbo"] < 0.5),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    short_adds = s_ls["sharpe_net"] > s_lo["sharpe_net"]
    print(f"  short leg verdict: L/S net Sharpe {s_ls['sharpe_net']:+.2f} vs LO "
          f"{s_lo['sharpe_net']:+.2f} -> {'keep short book' if short_adds else 'DROP the short book (§6 rule)'}")
    print(f"\nGATE: {'PASS' if all(checks.values()) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
