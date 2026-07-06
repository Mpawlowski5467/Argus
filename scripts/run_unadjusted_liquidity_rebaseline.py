"""Controlled unadjusted-liquidity rebaseline smoke gate.

This does NOT retrain/freeze the production artifact. It compares the current adjusted
liquidity universe with an experimental universe that uses Intrinio raw ``uclose`` and
``uvolume`` only for the liquidity/price floors, while labels and model features still
use adjusted total-return closes. Treat any improvement as a model-vintage candidate:
panel rebuild -> full Phase 1/3/CPCV -> train artifact -> paper baseline freeze.

  uv run python scripts/run_unadjusted_liquidity_rebaseline.py
"""

from __future__ import annotations

import duckdb
import pandas as pd

from stockscan.concepts import WIDE_PATH
from stockscan.config import MIN_DOLLAR_VOLUME
from stockscan.features import compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import evaluate
from stockscan.panel import load_matrices
from stockscan.prices import PRICES_DIR

WINSOR = (0.01, 0.99)


def _price_source() -> str:
    files = sorted(PRICES_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no price files under {PRICES_DIR}")
    return "read_parquet([" + ",".join(f"'{f}'" for f in files) + "], union_by_name=true)"


def load_unadjusted_liquidity_matrices() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return raw price and raw dollar-volume matrices, falling back per row to adjusted.

    Older price files may lack ``uclose``/``uvolume``; those rows fall back to ``close`` /
    ``volume`` so the experiment is comparable rather than silently dropping names.
    """
    src = _price_source()
    df = duckdb.query(f"""
        select ticker, date,
               coalesce(uclose, close) as liq_price,
               coalesce(uclose, close) * coalesce(uvolume, volume) as raw_dv,
               uclose is not null and uvolume is not null as has_raw
        from {src}
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    meta = {
        "rows": int(len(df)),
        "raw_rows": int(df["has_raw"].sum()),
        "raw_coverage": float(df["has_raw"].mean()) if len(df) else 0.0,
        "columns": int(df["ticker"].nunique()),
    }
    price = df.pivot_table(index="date", columns="ticker", values="liq_price").sort_index()
    dv = df.pivot_table(index="date", columns="ticker", values="raw_dv").sort_index()
    return price, dv, meta


def _build(feats, close, dv, tmap, liquidity_price=None) -> pd.DataFrame:
    return build_fundamental_panel(
        feats, close, delistings=None, ticker_map=tmap, dollar_volume=dv,
        liquidity_price=liquidity_price, min_dollar_volume=MIN_DOLLAR_VOLUME,
        winsorize=WINSOR,
    )


def _coverage(panel: pd.DataFrame) -> str:
    cov = panel.attrs.get("coverage")
    if cov is None or cov.empty:
        return "coverage unavailable"
    return (f"rows={len(panel):,} dates={panel['date'].nunique()} "
            f"names/date~{int(cov['universe'].mean())} priced~{int(cov['priced'].mean())}")


def _metrics(panel: pd.DataFrame) -> dict:
    m = evaluate(panel) or {}
    return {
        "mean_ic": m.get("mean_ic"),
        "t_nw": m.get("t_nw"),
        "decile_spread": m.get("decile_spread"),
        "oos_dates": m.get("oos_dates"),
    }


def _fmt_metrics(m: dict) -> str:
    return (f"IC={m['mean_ic']:+.4f} t_nw={m['t_nw']:+.2f} "
            f"decile_spread={m['decile_spread']:+.4f} oos_dates={m['oos_dates']}")


def membership_delta(base: pd.DataFrame, rebased: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in sorted(set(base["date"]).intersection(set(rebased["date"]))):
        b = set(base.loc[base["date"] == d, "cik"].astype(int))
        r = set(rebased.loc[rebased["date"] == d, "cik"].astype(int))
        rows.append({"date": d, "added": len(r - b), "dropped": len(b - r), "base": len(b), "rebased": len(r)})
    return pd.DataFrame(rows)


def main() -> int:
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, adj_dv = load_matrices()
    raw_price, raw_dv, raw_meta = load_unadjusted_liquidity_matrices()
    tmap = universe_ticker_map() or None

    print("UNADJUSTED-LIQUIDITY REBASELINE — experiment only, no artifact freeze")
    print(f"raw liquidity coverage: {raw_meta['raw_rows']:,}/{raw_meta['rows']:,} rows "
          f"({raw_meta['raw_coverage']:.1%}) across {raw_meta['columns']:,} columns")

    base = _build(feats, close, adj_dv, tmap)
    rebased = _build(feats, close, raw_dv, tmap, liquidity_price=raw_price)
    if base.empty or rebased.empty:
        print("empty panel")
        return 1

    print(f"\ncurrent adjusted-liquidity panel:   {_coverage(base)}")
    print(f"experimental raw-liquidity panel:   {_coverage(rebased)}")
    md = membership_delta(base, rebased)
    print(f"membership delta/date: +{md['added'].mean():.1f} added, -{md['dropped'].mean():.1f} dropped "
          f"(max +{md['added'].max()}, max -{md['dropped'].max()})")

    bm = _metrics(base)
    rm = _metrics(rebased)
    print("\nLightGBM walk-forward OOS:")
    print(f"  current adjusted-liquidity: {_fmt_metrics(bm)}")
    print(f"  raw-liquidity experiment:   {_fmt_metrics(rm)}")
    print(f"  delta: IC={rm['mean_ic'] - bm['mean_ic']:+.4f} "
          f"spread={rm['decile_spread'] - bm['decile_spread']:+.4f}")
    print("\nPROMOTION RULE: do not switch production floors from this alone. If attractive, run "
          "full Phase 3/CPCV, retrain as a new artifact vintage, and re-freeze paper baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
