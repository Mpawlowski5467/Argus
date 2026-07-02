"""Phase 1 go/no-go: survivorship-corrected, liquidity-filtered, winsorized, multi-regime.

Assembles the point-in-time fundamental panel (liquid tradable names + re-injected
delisting failures, forward returns winsorized per date), then reports single-factor
ICs, a composite, a walk-forward LightGBM, and the delisting-haircut sensitivity sweep.

  uv run python scripts/run_phase1.py
"""

import duckdb
import numpy as np

from stockscan.concepts import WIDE_PATH
from stockscan.config import DELISTING_HAIRCUT_SWEEP, MIN_DOLLAR_VOLUME
from stockscan.edgar.delistings import load_delistings
from stockscan.features import FEATURE_SIGN, FEATURES, compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.model import evaluate
from stockscan.panel import load_matrices
from stockscan.validation import ic_summary, rank_ic

WINSOR = (0.01, 0.99)


def _build(feats, close, dv, delistings, reason_return=None):
    return build_fundamental_panel(
        feats, close, delistings=delistings, dollar_volume=dv,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR, reason_return=reason_return,
    )


def _composite_ic(panel):
    comp = np.zeros(len(panel))
    for f in FEATURES:
        comp += FEATURE_SIGN[f] * (panel[f"{f}_rank"].fillna(0.5) - 0.5)
    tmp = panel[["date", "label_excess"]].copy()
    tmp["composite"] = comp
    return ic_summary(rank_ic(tmp, feature="composite"))


def main() -> int:
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, dv = load_matrices()
    if close.empty:
        print("no prices; run scripts/ingest_prices.py first")
        return 1

    delistings = load_delistings()
    panel = _build(feats, close, dv, delistings)
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

    print("\ndelisting-haircut sweep (edge must survive the whole range):")
    for h in DELISTING_HAIRCUT_SWEEP:
        s = _composite_ic(_build(feats, close, dv, delistings, reason_return={"delist": h, "dereg": h}))
        print(f"  haircut {h:+.2f}:  composite IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}")

    print(
        "\nHONEST READ: liquidity filter + winsorization make the decile spread meaningful and\n"
        "cut micro-cap survivorship. If t-stats are still implausibly high (>~5) or accruals/\n"
        "asset_growth stay positive, residual survivorship bias persists (free prices still miss\n"
        "delisted names' living history) -> treat as an UPPER BOUND. Full closure needs paid,\n"
        "survivorship-free data (CRSP/Compustat)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
