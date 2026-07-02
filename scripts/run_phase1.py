"""Phase 1 go/no-go: survivorship-corrected, liquidity-filtered, winsorized, multi-regime.

Assembles the point-in-time fundamental panel (liquid tradable names + delisting
failures, forward returns winsorized per date), then reports single-factor ICs, a
composite, a walk-forward LightGBM, and the delisting-haircut sensitivity sweep.

With survivorship-free prices (Intrinio universe), ``--no-impute`` drops the ledger
imputation entirely: dead names' declines enter through their REAL price history and
terminal (last-trade) returns — no assumed haircut, no circularity. That is the
honest headline configuration; the default imputing run remains for comparison.

  uv run python scripts/run_phase1.py [--no-impute]
"""

import argparse

import duckdb
import numpy as np

from stockscan.concepts import WIDE_PATH
from stockscan.config import DELISTING_HAIRCUT_SWEEP, MIN_DOLLAR_VOLUME
from stockscan.edgar.delistings import load_delistings
from stockscan.features import FEATURE_SIGN, FEATURES, compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import evaluate
from stockscan.panel import load_matrices
from stockscan.validation import ic_summary, rank_ic

WINSOR = (0.01, 0.99)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 1 go/no-go gate.")
    ap.add_argument(
        "--no-impute", action="store_true",
        help="drop the delisting-return imputation (delistings=None); failures enter "
             "only through real delisted price history + terminal returns",
    )
    return ap.parse_args(argv)


def _build(feats, close, dv, delistings, reason_return=None, ticker_map=None):
    return build_fundamental_panel(
        feats, close, delistings=delistings, dollar_volume=dv, ticker_map=ticker_map,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR, reason_return=reason_return,
    )


def _composite_ic(panel):
    comp = np.zeros(len(panel))
    for f in FEATURES:
        comp += FEATURE_SIGN[f] * (panel[f"{f}_rank"].fillna(0.5) - 0.5)
    tmp = panel[["date", "label_excess"]].copy()
    tmp["composite"] = comp
    return ic_summary(rank_ic(tmp, feature="composite"))


def main(argv=None) -> int:
    args = parse_args(argv)
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, dv = load_matrices()
    if close.empty:
        print("no prices; run scripts/ingest_prices.py first")
        return 1

    tmap = universe_ticker_map() or None  # survivorship-free map when built; else EDGAR
    delistings = None if args.no_impute else load_delistings()
    print(f"mode: {'NO-IMPUTE (real delisted prices only)' if args.no_impute else 'with delisting imputation'}"
          f"  |  ticker map: {'intrinio universe (' + str(len(tmap)) + ' ciks)' if tmap else 'EDGAR current'}")
    panel = _build(feats, close, dv, delistings, ticker_map=tmap)
    if panel.empty:
        print("empty panel")
        return 1

    n_dates = panel["date"].nunique()
    print(f"panel: {len(panel):,} rows  dates={n_dates}  ~{len(panel) // n_dates} names/date")
    cov = panel.attrs.get("coverage")
    if cov is not None and len(cov):
        print(
            f"liquid universe~{int(cov['universe'].mean())}/date  priced~{int(cov['priced'].mean())}"
            f"  imputed(delisting)~{int(cov['imputed'].mean())}  "
            f"(priced coverage {cov['priced'].sum() / cov['universe'].sum():.0%})"
        )

    print("\nsingle-feature rank IC:")
    for f in FEATURES:
        s = ic_summary(rank_ic(panel, feature=f"{f}_rank"))
        flag = "  <-- weak" if abs(s["t_nw"]) < 1 else ""
        print(f"  {f:<20} IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}{flag}")

    cs = _composite_ic(panel)
    print(f"\ncomposite factor score:  IC={cs['mean_ic']:+.4f}  t_nw={cs['t_nw']:+.2f}  (n={cs['n']})")
    m = evaluate(panel)
    if m:
        print(
            f"LightGBM walk-forward OOS:  IC={m['mean_ic']:+.4f}  t_nw={m['t_nw']:+.2f}  "
            f"decile_spread={m['decile_spread']:+.4f}  (oos dates={m['oos_dates']})"
        )

    if args.no_impute:
        print("\n(haircut sweep skipped: nothing is imputed in --no-impute mode)")
    else:
        print("\ndelisting-haircut sweep (edge must survive the whole range):")
        for h in DELISTING_HAIRCUT_SWEEP:
            s = _composite_ic(_build(feats, close, dv, delistings,
                                     reason_return={"delist": h, "dereg": h}, ticker_map=tmap))
            print(f"  haircut {h:+.2f}:  composite IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}")

    print(
        "\nHONEST READ: with the Intrinio survivorship-free universe, dead names carry their\n"
        "real price history and terminal returns; --no-impute is the trustworthy configuration\n"
        "(no haircut circularity). Gate: mean rank IC >= 0.03, overlap-corrected t >= 2,\n"
        "positive net-of-cost decile spread, and (imputing mode) sweep-stable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
