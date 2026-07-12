"""Freeze the product model: ONE fit on the full honest panel -> artifacts/model/.

Honest configuration only: survivorship-free universe map, real terminal returns,
NO delisting imputation (delistings=None), liquidity floors and per-date label
winsorization exactly as the Phase-1 gate. The artifact records its training-date
cutoff and feature list; the serve path loads it and never retrains.

  uv run python scripts/train_model.py [--eval]
"""

import argparse

import duckdb

from stockscan.concepts import WIDE_PATH
from stockscan.config import MIN_DOLLAR_VOLUME
from stockscan.features import compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import RANK_COLS, evaluate, fit, save_artifact
from stockscan.panel import load_matrices

WINSOR = (0.01, 0.99)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train + freeze the product model artifact.")
    ap.add_argument("--eval", action="store_true",
                    help="also run the purged walk-forward OOS evaluation (slow) and "
                         "record it in the artifact metadata")
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

    extra = {
        "mode": "no-impute (real delisted prices, terminal last-trade returns)",
        "min_dollar_volume": MIN_DOLLAR_VOLUME,
        "winsorize": WINSOR,
        "n_universe_ciks": len(tmap),
    }
    if args.eval:
        summ = evaluate(panel)
        if summ:
            print(f"walk-forward OOS: IC={summ['mean_ic']:+.4f} t_nw={summ['t_nw']:+.2f} "
                  f"decile_spread={summ['decile_spread']:+.4f}")
            extra["oos_eval"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in summ.items()}
        else:
            print("walk-forward OOS skipped: too few dates/rows for purged splits")

    model = fit(panel)
    out = save_artifact(model, panel, extra=extra)
    labeled = panel.dropna(subset=["label_excess"])
    print(f"frozen -> {out}")
    print(f"  rows={len(labeled):,}  dates={labeled['date'].nunique()}  "
          f"trained_through={labeled['date'].max().date()}")
    imp = sorted(zip(RANK_COLS, model.feature_importances_), key=lambda x: -x[1])
    print("  importance: " + ", ".join(f"{c.removesuffix('_rank')}={v}" for c, v in imp[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
