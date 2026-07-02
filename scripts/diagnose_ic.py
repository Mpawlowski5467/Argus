"""Investigate why the composite/model IC and t-stats look implausibly high.

Runs five targeted checks:
  A. real composite IC (baseline)
  B. shuffled-label IC  -> must collapse to ~0 (leak check on the CURRENT pipeline)
  C. IC autocorrelation + t-stat under widening Newey-West lags + non-overlapping rebalance
     -> is the t-stat inflated by overlap/persistence the NW(lag=2) under-corrects?
  D. no-imputation composite AND model IC -> how much does the delisting imputation inflate?
  E. composite IC by year -> real & regime-stable, or concentrated in a few years?

  uv run python scripts/diagnose_ic.py
"""

import duckdb
import numpy as np

from stockscan.concepts import WIDE_PATH
from stockscan.config import MIN_DOLLAR_VOLUME
from stockscan.edgar.delistings import load_delistings
from stockscan.features import FEATURE_SIGN, FEATURES, compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.model import evaluate
from stockscan.panel import load_matrices
from stockscan.validation import ic_summary, newey_west_tstat, rank_ic

WINSOR = (0.01, 0.99)


def _build(feats, close, dv, delistings):
    return build_fundamental_panel(
        feats, close, delistings=delistings, dollar_volume=dv,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=WINSOR,
    )


def _composite(panel):
    c = np.zeros(len(panel))
    for f in FEATURES:
        c += FEATURE_SIGN[f] * (panel[f"{f}_rank"].fillna(0.5) - 0.5)
    t = panel[["date", "label_excess"]].copy()
    t["composite"] = c
    return t


def main() -> int:
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, dv = load_matrices()
    delistings = load_delistings()

    panel = _build(feats, close, dv, delistings)
    cp = _composite(panel)
    ic = rank_ic(cp, feature="composite").sort_index()
    base = ic_summary(ic)
    print(f"A. real composite IC = {base['mean_ic']:+.4f}  t_nw(lag2) = {base['t_nw']:+.2f}  n={base['n']}")

    rng = np.random.default_rng(0)
    sh = cp.copy()
    sh["label_excess"] = sh.groupby("date")["label_excess"].transform(
        lambda s: rng.permutation(s.to_numpy())
    )
    shf = ic_summary(rank_ic(sh, feature="composite"))
    print(f"B. shuffled-label IC = {shf['mean_ic']:+.4f}  t_nw = {shf['t_nw']:+.2f}  (must be ~0 -> no leak)")

    acf = [round(ic.autocorr(lag=k), 2) for k in range(1, 7)]
    print(f"C. IC autocorrelation lags 1-6: {acf}")
    for lag in (0, 2, 6, 12):
        print(f"   t-stat NW(lag={lag:>2}) = {newey_west_tstat(ic.to_numpy(), lag):+.2f}")
    nb = ic.iloc[::3]  # non-overlapping ~quarterly
    t_nb = nb.mean() / nb.std() * np.sqrt(len(nb))
    print(f"   non-overlapping (every 3rd month): IC={nb.mean():+.4f}  t={t_nb:+.2f}  n={len(nb)}")

    panel_ni = _build(feats, close, dv, None)  # drop the delisting imputation
    ni = ic_summary(rank_ic(_composite(panel_ni), feature="composite"))
    m_full = evaluate(panel)
    m_ni = evaluate(panel_ni)
    print(f"D. composite IC  with-imputation={base['mean_ic']:+.4f}   no-imputation={ni['mean_ic']:+.4f}")
    print(f"   model  IC     with-imputation={m_full['mean_ic']:+.4f}   no-imputation={m_ni['mean_ic']:+.4f}")

    by_year = ic.groupby(ic.index.year).mean()
    print("E. composite IC by year:")
    print("   " + "  ".join(f"{y}:{v:+.3f}" for y, v in by_year.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
