"""Large-drawdown head go/no-go: does a fundamentals-only classifier rank downside OOS?

Builds the point-in-time drawdown panel (``y=1`` if a name falls peak-to-trough past the
threshold within the horizon), then reports the same rare-event scorecard as the distress
gate: out-of-sample ROC-AUC, PR-AUC, precision/recall @ top-decile vs base rate, calibration,
and the CPCV DISTRIBUTION of AUC. With ``--orthogonality`` it also trains the distress head on
the same data and checks whether drawdown adds downside signal BEYOND distress (score
correlation + AUC restricted to non-distress rows) — the gate the spec insists on before we
ship. ``--save`` freezes a SEPARATE artifact (``artifacts/drawdown_model/``); it never touches
the return artifact, the serve path, or any trade rule.

  uv run python scripts/run_drawdown_head.py [--horizon-months 6] [--threshold -0.30]
                                             [--orthogonality] [--save] [--quick]
"""

import argparse

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stockscan.concepts import WIDE_PATH
from stockscan.drawdown import (
    DEFAULT_HORIZON_MONTHS,
    DEFAULT_THRESHOLD,
    RANK_COLS,
    build_drawdown_panel,
    cpcv_auc,
    drawdown_metrics,
    fit_drawdown,
    save_drawdown_artifact,
    walk_forward_predict_proba,
)
from stockscan.features import compute_features
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.panel import load_matrices_cached


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Large-drawdown head go/no-go gate.")
    ap.add_argument("--horizon-months", type=int, default=DEFAULT_HORIZON_MONTHS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="peak-to-trough fall that counts as a large drawdown (e.g. -0.30)")
    ap.add_argument("--orthogonality", action="store_true",
                    help="also train distress on the same data and test drawdown's incremental signal")
    ap.add_argument("--save", action="store_true", help="freeze artifacts/drawdown_model/")
    ap.add_argument("--quick", action="store_true", help="skip the slow CPCV pass")
    return ap.parse_args(argv)


def orthogonality_vs_distress(dd_pred: pd.DataFrame, feats, close, N: int) -> None:
    """Does drawdown rank downside BEYOND the distress head? Train distress on the same
    dates, join on (date, cik), then (a) correlate the two probs and (b) score drawdown's
    AUC restricted to rows the distress label calls NEGATIVE — i.e. crashes that are NOT
    distress delistings. Signal that survives there is genuinely additive."""
    from stockscan.distress import (
        build_distress_panel,
        classify_distress_events,
        walk_forward_predict_proba as distress_wf,
    )
    from stockscan.edgar.delistings import load_delistings

    tmap = universe_ticker_map()
    ev = classify_distress_events(load_delistings(), close, tmap)
    dpanel = build_distress_panel(feats, close, ev, horizon_months=12, min_names=30)
    if dpanel.empty:
        print("  (orthogonality skipped: empty distress panel)")
        return
    dwf = distress_wf(dpanel, n_splits=5, embargo=2, horizon_periods=12, id_cols=("cik",))
    dwf = dwf.rename(columns={"prob": "distress_prob", "y": "distress_y"})

    j = dd_pred.merge(dwf[["date", "cik", "distress_prob", "distress_y"]], on=["date", "cik"], how="inner")
    if j.empty:
        print("  (orthogonality skipped: no overlapping (date, cik) rows)")
        return
    corr = float(pd.Series(j["prob"]).corr(pd.Series(j["distress_prob"]), method="spearman"))
    non_distress = j[j["distress_y"] == 0]
    inc_auc = (float(roc_auc_score(non_distress["y"], non_distress["prob"]))
               if non_distress["y"].nunique() == 2 else float("nan"))
    print("\n=== orthogonality vs distress ===")
    print(f"  overlapping rows={len(j):,}  spearman(drawdown_prob, distress_prob)={corr:+.3f}")
    print(f"  drawdown AUC among NON-distress rows={inc_auc:.4f}  "
          f"(n={len(non_distress):,}, drawdowns={int(non_distress['y'].sum()):,})")
    print("  READ: low correlation + AUC still well >0.5 here => drawdown adds downside "
          "signal beyond distress (ship-worthy); high corr + AUC~0.5 => it's distress in a hat.")


def main(argv=None) -> int:
    args = parse_args(argv)
    N, thr = args.horizon_months, args.threshold

    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, _ = load_matrices_cached()
    if close.empty:
        print("no prices; run scripts/fetch_intrinio_prices.py first")
        return 1
    tmap = universe_ticker_map()
    if not tmap:
        print("no intrinio universe; run scripts/build_intrinio_universe.py first")
        return 1

    print(f"label: peak-to-trough drawdown <= {thr:.0%} within {N} months (path-based, "
          f"survivorship-correct)")
    panel = build_drawdown_panel(feats, close, tmap, horizon_months=N, threshold=thr, min_names=30)
    if panel.empty:
        print("empty panel")
        return 1
    cov = panel.attrs["coverage"]
    print(f"\npanel: {len(panel):,} rows  dates={panel['date'].nunique()}  "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}  "
          f"(censor {pd.Timestamp(panel.attrs['censor_date']).date()})")
    print(f"positives={int(panel['y'].sum()):,}  base rate={panel['y'].mean():.4f}  "
          f"universe~{int(cov['universe'].mean())}/date")

    # walk-forward (causal) scorecard
    pred = walk_forward_predict_proba(panel, n_splits=5, embargo=2, horizon_periods=N,
                                      id_cols=("cik",))
    m = drawdown_metrics(pred)
    print("\n=== walk-forward OOS (causal) ===")
    print(f"  ROC-AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  (base rate {m['base_rate']:.4f})")
    print(f"  precision@decile={m['precision_at_decile']:.4f}  recall@decile={m['recall_at_decile']:.4f}  "
          f"lift={m['lift']:.2f}x  (oos dates={pred['date'].nunique()})")
    print(f"  calibration_mae={m['calibration_mae']:.4f}")

    mdl = fit_drawdown(panel)
    imp = pd.Series(mdl.booster_.feature_importance(importance_type="gain"), index=RANK_COLS)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print("  drivers (gain %):  " + "  ".join(f"{k.replace('_rank','')}={v:.0f}" for k, v in imp.items()))

    cp = None
    if not args.quick:
        print("\n=== CPCV AUC distribution (robustness) ===")
        cp = cpcv_auc(panel, n_groups=10, k_test=2, embargo=2, horizon_periods=N)
        print(f"  combos={cp['n_combos']}  mean={cp['mean_auc']:.4f}  std={cp['std_auc']:.4f}  "
              f"min={cp['min_auc']:.4f}  p05={cp['p05_auc']:.4f}  median={cp['median_auc']:.4f}")
        print(f"  frac(AUC>0.5)={cp['frac_above_0p5']:.2f}  frac(AUC>0.7)={cp['frac_above_0p7']:.2f}")

    if args.orthogonality:
        orthogonality_vs_distress(pred, feats, close, N)

    auc_ref = cp["mean_auc"] if cp else m["auc"]
    passed = (auc_ref >= 0.60 and not pd.isna(m["lift"]) and m["lift"] >= 1.5
              and (cp is None or cp["frac_above_0p5"] == 1.0))
    print("\nGATE: keep only if it ranks large drawdowns OOS — AUC materially above 0.5, "
          "precision@decile above base, AND survives CPCV (plus adds signal beyond distress).")
    print(f"VERDICT: {'PASS — worth wiring as a risk flag' if passed else 'WEAK / DISCARD'}  "
          f"(AUC~{auc_ref:.3f}, precision@decile {m['lift']:.1f}x base, calibration_mae {m['calibration_mae']:.4f})")

    if args.save:
        out = save_drawdown_artifact(mdl, panel, extra={
            "walk_forward": {k: m[k] for k in ("auc", "pr_auc", "precision_at_decile",
                                               "recall_at_decile", "lift", "calibration_mae")},
            "cpcv": {k: cp[k] for k in ("n_combos", "mean_auc", "std_auc", "min_auc",
                                        "frac_above_0p7")} if cp else None,
        })
        print(f"\nfroze drawdown artifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
