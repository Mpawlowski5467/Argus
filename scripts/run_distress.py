"""Distress-head go/no-go: does a fundamentals-only classifier rank distress risk OOS?

Builds the point-in-time distress panel (price-confirmed distress delistings as positives,
benign M&A / premium going-private as negatives, unpriced deaths dropped), then reports the
rare-event scorecard the return-head IC gate can't speak to: out-of-sample ROC-AUC, PR-AUC,
precision / recall @ top-decile vs the base rate, calibration, and — the honest read — the
CPCV DISTRIBUTION of AUC. ``--save`` freezes a SEPARATE artifact (artifacts/distress_model/);
it never touches the return artifact, the serve path, or any trade rule.

  uv run python scripts/run_distress.py [--horizon-months 12] [--save] [--naive] [--quick]

``--naive`` labels on ledger reason alone (no price confirmation) to show how much M&A
contamination costs. ``--quick`` skips the (slow) CPCV pass.
"""

import argparse
import json

import duckdb
import pandas as pd

from stockscan.concepts import WIDE_PATH
from stockscan.distress import (
    DEFAULT_HORIZON_MONTHS,
    RANK_COLS,
    build_distress_panel,
    classify_distress_events,
    cpcv_auc,
    fit_distress,
    save_distress_artifact,
    walk_forward_predict_proba,
    distress_metrics,
)
from stockscan.edgar.delistings import load_delistings
from stockscan.features import compute_features
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.panel import load_matrices_cached


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Distress-head go/no-go gate.")
    ap.add_argument("--horizon-months", type=int, default=DEFAULT_HORIZON_MONTHS,
                    help="forward window (months) over which a distress delisting counts")
    ap.add_argument("--save", action="store_true", help="freeze artifacts/distress_model/")
    ap.add_argument("--naive", action="store_true",
                    help="label on ledger reason alone (no price confirmation) — shows M&A cost")
    ap.add_argument("--balanced", action="store_true",
                    help="scale_pos_weight=balanced (better recall, wrecks calibration)")
    ap.add_argument("--quick", action="store_true", help="skip the slow CPCV pass")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    N = args.horizon_months
    spw = "balanced" if args.balanced else None

    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, _ = load_matrices_cached()
    if close.empty:
        print("no prices; run scripts/ingest_prices.py first")
        return 1
    dl = load_delistings()
    tmap = universe_ticker_map()
    if not tmap:
        print("no intrinio universe; run scripts/build_intrinio_universe.py first")
        return 1

    confirm = () if args.naive else ("delist", "dereg")
    ev = classify_distress_events(dl, close, tmap, confirm_reasons=confirm)
    n_true = int((ev["is_distress"] == True).sum())   # noqa: E712
    n_false = int((ev["is_distress"] == False).sum())  # noqa: E712
    n_amb = int(ev["is_distress"].isna().sum())
    print(f"label mode: {'NAIVE reason-only' if args.naive else 'PRICE-CONFIRMED'}  |  "
          f"horizon={N}mo  |  weighting={'balanced' if args.balanced else 'natural prior'}")
    print(f"ledger events: distress={n_true}  benign(M&A/LBO)={n_false}  ambiguous(unpriced)={n_amb}")

    panel = build_distress_panel(feats, close, ev, horizon_months=N, min_names=30)
    if panel.empty:
        print("empty panel")
        return 1
    cov = panel.attrs["coverage"]
    print(f"\npanel: {len(panel):,} rows  dates={panel['date'].nunique()}  "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}  "
          f"(censor {pd.Timestamp(panel.attrs['censor_date']).date()})")
    print(f"positives={int(panel['y'].sum()):,}  base rate={panel['y'].mean():.4f}  "
          f"universe~{int(cov['universe'].mean())}/date  ambiguous dropped~{int(cov['ambiguous_dropped'].mean())}/date")

    # walk-forward (causal) scorecard
    pred = walk_forward_predict_proba(panel, n_splits=5, embargo=2, horizon_periods=N,
                                      scale_pos_weight=spw)
    m = distress_metrics(pred)
    print("\n=== walk-forward OOS (causal) ===")
    print(f"  ROC-AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  (base rate {m['base_rate']:.4f})")
    print(f"  precision@decile={m['precision_at_decile']:.4f}  recall@decile={m['recall_at_decile']:.4f}  "
          f"lift={m['lift']:.2f}x  (oos dates={pred['date'].nunique()})")
    print(f"  calibration_mae={m['calibration_mae']:.4f}")
    print("  reliability (predicted -> realized):")
    for c in m["calibration"]:
        bar = "#" * int(round(c["realized"] / max(1e-9, m["calibration"][-1]["realized"]) * 20))
        print(f"    {c['pred']:.4f} -> {c['realized']:.4f}  {bar}")

    # feature importance (full-panel fit) — should read as a distress signature
    mdl = fit_distress(panel, scale_pos_weight=spw)
    imp = pd.Series(mdl.booster_.feature_importance(importance_type="gain"), index=RANK_COLS)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print("\n  drivers (gain %):  " + "  ".join(f"{k.replace('_rank','')}={v:.0f}" for k, v in imp.items()))

    # CPCV distribution (the non-fluke read)
    cp = None
    if not args.quick:
        print("\n=== CPCV AUC distribution (robustness) ===")
        cp = cpcv_auc(panel, n_groups=10, k_test=2, embargo=2, horizon_periods=N,
                      scale_pos_weight=spw)
        print(f"  combos={cp['n_combos']}  mean={cp['mean_auc']:.4f}  std={cp['std_auc']:.4f}  "
              f"min={cp['min_auc']:.4f}  p05={cp['p05_auc']:.4f}  median={cp['median_auc']:.4f}  p95={cp['p95_auc']:.4f}")
        print(f"  frac(AUC>0.5)={cp['frac_above_0p5']:.2f}  frac(AUC>0.7)={cp['frac_above_0p7']:.2f}  "
              f"mean PR-AUC={cp['mean_pr_auc']:.4f}")

    # gate verdict
    auc_ref = cp["mean_auc"] if cp else m["auc"]
    passed = (auc_ref >= 0.70 and not pd.isna(m["lift"]) and m["lift"] >= 2.0
              and (cp is None or cp["frac_above_0p5"] == 1.0))
    print("\nGATE: keep only if it ranks distress OOS — AUC well above 0.5 (aim ~0.75), "
          "precision@decile many× base, decent calibration, AND survives CPCV.")
    print(f"VERDICT: {'PASS — keep the distress head' if passed else 'DISCARD'}  "
          f"(AUC~{auc_ref:.3f}, precision@decile {m['lift']:.1f}x base, calibration_mae {m['calibration_mae']:.4f})")

    if args.save:
        if args.naive or args.balanced:
            print("\nrefusing --save with --naive/--balanced (would freeze a non-product config)")
            return 1
        out = save_distress_artifact(mdl, panel, extra={
            "walk_forward": {k: m[k] for k in ("auc", "pr_auc", "precision_at_decile",
                                               "recall_at_decile", "lift", "calibration_mae")},
            "cpcv": {k: cp[k] for k in ("n_combos", "mean_auc", "std_auc", "min_auc",
                                        "frac_above_0p7")} if cp else None,
            "n_distress_events": n_true, "n_benign_events": n_false, "n_ambiguous_events": n_amb,
        })
        print(f"\nfroze distress artifact -> {out}")
        print(json.dumps(json.loads((out / 'meta.json').read_text()), indent=2)[:600] + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
