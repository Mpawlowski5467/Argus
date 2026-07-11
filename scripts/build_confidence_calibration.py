"""Freeze the confidence calibration artifact: per prediction-decile OOS hit-rate.

Rebuilds the honest panel EXACTLY as ``train_model.py`` (no-impute, liquidity floors,
1/99 winsor), runs the purged walk-forward (``model.walk_forward_predict``), buckets the
pooled out-of-sample rows into per-date prediction deciles, and records how often each
decile actually beat the cross-section (``hit_rate`` = P(label_excess > per-date
cross-sectional median); the older mean-anchored P(label_excess > 0) rides along as
``hit_rate_pos``) plus a Wilson 95% CI. The result anchors ``stockscan.confidence`` — it
describes the SAME model the serve path scores with, so rebuild it whenever the return
artifact is refrozen.

  uv run python scripts/build_confidence_calibration.py [--n-splits 5]

FIREWALL: writes only ``artifacts/confidence_cal/`` — never the return artifact.
"""

from __future__ import annotations

import argparse
import json
import math

import duckdb
import numpy as np
import pandas as pd

from stockscan.concepts import WIDE_PATH
from stockscan.config import LABEL_HORIZON_DAYS, MIN_DOLLAR_VOLUME
from stockscan.confidence import CALIBRATION_DIR
from stockscan.features import compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import walk_forward_predict
from stockscan.panel import load_matrices
from stockscan.validation import newey_west_tstat

WINSOR = (0.01, 0.99)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial hit-rate (well-behaved at small n / extreme p)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def decile_calibration(pred: pd.DataFrame, n: int = 10, overlap_lag: int = 2) -> dict:
    """Per prediction-decile pooled OOS stats: hit-rate, mean excess, count, Wilson CI,
    and the Newey-West t-stat of the decile's per-date mean excess.

    ``hit_rate`` = P(label_excess > that date's cross-sectional MEDIAN), which centers
    at 0.5 by construction. The prior anchor — P(label_excess > 0), beating the sector
    mean — is kept as ``hit_rate_pos`` for reference: right-skewed excess returns hold
    it below 0.5 even in the top decile, which zeroed every BUY-side confidence under
    the directional edge rule in ``stockscan.confidence`` (found 2026-07-11).

    ``t_excess`` is the honest answer to a subtler problem the median anchor exposed:
    the top decile's edge is MAGNITUDE-carried (mean excess +, hit-rate ~0.49 — a
    minority of big winners), so ANY frequency anchor reads ~0 for BUYs. The NW-t of
    the per-date mean excess series (lag = label-overlap periods) measures how reliable
    the decile's average edge is — the same statistic RESULTS.md gates the model on —
    and is what ``stockscan.confidence`` anchors conviction to when present.
    """
    rows = []
    for _, g in pred.groupby("date"):
        if len(g) < n:
            continue
        r = g["pred"].rank(method="first")
        med = g["label_excess"].median()
        g = g.assign(
            _decile=np.ceil(r / len(g) * n).astype(int).clip(1, n),
            _beat_med=g["label_excess"] > med,
        )
        rows.append(g[["date", "_decile", "label_excess", "_beat_med"]])
    pooled = pd.concat(rows, ignore_index=True).dropna(subset=["label_excess"])
    deciles = {}
    for d in range(1, n + 1):
        sub = pooled[pooled["_decile"] == d]
        cnt = int(len(sub))
        if cnt == 0:
            deciles[str(d)] = {"hit_rate": None, "hit_rate_pos": None, "mean_excess": None,
                               "n": 0, "ci_low": None, "ci_high": None,
                               "t_excess": None, "n_dates": 0}
            continue
        hr = float(sub["_beat_med"].mean())
        lo, hi = wilson_ci(hr, cnt)
        date_means = sub.groupby("date")["label_excess"].mean()
        t_exc = newey_west_tstat(date_means.to_numpy(), lag=overlap_lag)
        deciles[str(d)] = {
            "hit_rate": round(hr, 4),
            "hit_rate_pos": round(float((sub["label_excess"] > 0).mean()), 4),
            "mean_excess": round(float(sub["label_excess"].mean()), 6),
            "n": cnt,
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "t_excess": round(float(t_exc), 2) if math.isfinite(t_exc) else None,
            "n_dates": int(len(date_means)),
        }
    return deciles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the confidence calibration artifact.")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=2)
    ap.add_argument("--horizon-periods", type=int, default=3,
                    help="monthly rebalance periods the forward label overlaps (~63d = 3)")
    args = ap.parse_args(argv)

    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, dv = load_matrices()
    if close.empty:
        print("no prices; run scripts/fetch_intrinio_prices.py first")
        return 1
    tmap = universe_ticker_map()
    if not tmap:
        print("no universe map; run scripts/build_intrinio_universe.py first")
        return 1

    print(f"building honest panel (no-impute, {len(tmap)} universe ciks) ...")
    panel = build_fundamental_panel(
        feats, close, delistings=None, ticker_map=tmap, dollar_volume=dv,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR,
    )
    if panel.empty:
        print("empty panel")
        return 1

    print("purged walk-forward (OOS predictions) ...")
    pred = walk_forward_predict(
        panel, n_splits=args.n_splits, embargo=args.embargo,
        horizon_periods=args.horizon_periods, id_cols=("cik",),
    )
    if pred.empty:
        print("no OOS predictions (too few dates for purged splits)")
        return 1

    deciles = decile_calibration(pred, overlap_lag=max(1, args.horizon_periods - 1))
    artifact = {
        "head": "confidence_calibration",
        "method": "purged walk-forward; per-date prediction deciles; "
                  "hit_rate = P(label_excess > per-date cross-sectional median) pooled OOS "
                  "(hit_rate_pos = P(label_excess > 0) kept for reference); "
                  "t_excess = Newey-West t of per-date mean excess (conviction anchor)",
        "trained_through": str(pd.Timestamp(pred["date"].max()).date()),
        "n_oos_dates": int(pred["date"].nunique()),
        "n_oos_rows": int(len(pred)),
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "walk_forward": {"n_splits": args.n_splits, "embargo": args.embargo,
                         "horizon_periods": args.horizon_periods},
        "deciles": deciles,
    }
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = CALIBRATION_DIR / "calibration.json"
    out.write_text(json.dumps(artifact, indent=2))

    print(f"frozen -> {out}")
    print(f"  OOS dates={artifact['n_oos_dates']}  rows={artifact['n_oos_rows']:,}")
    print("  decile  hit_rate   n      mean_excess   t_excess")
    for d in range(1, 11):
        s = deciles[str(d)]
        if s["hit_rate"] is None:
            print(f"  {d:>5}   —")
        else:
            t = f"{s['t_excess']:+.2f}" if s["t_excess"] is not None else "—"
            print(f"  {d:>5}   {s['hit_rate']:.3f}   {s['n']:>6,}   {s['mean_excess']:+.4f}   {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
