"""Path-2 tightening for short-term reversal: matched-horizon + block-bootstrap.

`scripts/run_reversal_test.py` found `st_rev` marginally cleared the model+CPCV bar at the
shipped 63-day label (the first price feature to). This script decides whether that is a real
reversal edge or an in-sample artifact — the paper-forward OOS window can't, because a ~0.001
IC effect is dwarfed by ~0.10 monthly-IC noise for years. Three in-sample checks instead:

  A) MATCHED HORIZON — reversal is a ~1-month effect, so rebuild the panel with a 21-day label
     (its native horizon) and re-run baseline vs +st_rev. A real reversal is STRONGEST here; an
     artifact is absent here and only shows at 63d.
  B) CONSTRUCTION SCAN — signed cross-sectional IC of the raw trailing-window return for
     {5,10,21,42,63} trading days vs BOTH labels (reversal => NEGATIVE IC).
  C) BLOCK-BOOTSTRAP significance — seed-average the walk-forward OOS preds (removes bagging
     noise), take the per-DATE ΔIC (baseline vs +st_rev), and moving-block bootstrap it (block ~
     label overlap). This is the honest test the paired CPCV t was not — CPCV combos share
     training data, so that t is inflated.

  uv run python scripts/run_reversal_matched_horizon.py

Read-only research: never touches the frozen artifact, defaults, or paper-forward book. Panels
are cached under the system temp dir (regenerable).
"""
import tempfile
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from stockscan.concepts import WIDE_PATH
from stockscan.config import MIN_DOLLAR_VOLUME
from stockscan.features import compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import DEFAULT_PARAMS, RANK_COLS, walk_forward_predict
from stockscan.panel import load_matrices_cached
from stockscan.validation import cpcv_splits, ic_summary, newey_west_tstat, rank_ic

SEEDS = [0, 7, 123]  # seed-averaged to strip LightGBM bagging noise from the per-date IC series
COLS_BASE = RANK_COLS
COLS_SR = RANK_COLS + ["st_rev_rank"]
CACHE = Path(tempfile.gettempdir()) / "argus_reversal_panels"
# per-label config: hp = CPCV/WF purge periods (~ceil(horizon/21)); lag = NW overlap lag;
# block = bootstrap block length (~ label overlap in monthly periods).
HOR = {21: dict(hp=1, lag=0, block=2), 63: dict(hp=3, lag=2, block=3)}


def get_panel(horizon):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"panel_h{horizon}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    wide = duckdb.query(f"select * from read_parquet('{WIDE_PATH}')").df()
    feats = compute_features(wide)
    close, dv = load_matrices_cached()
    tmap = universe_ticker_map() or None
    panel = build_fundamental_panel(
        feats, close, delistings=None, dollar_volume=dv, ticker_map=tmap,
        min_dollar_volume=MIN_DOLLAR_VOLUME, winsorize=(0.01, 0.99),
        price_features=True, horizon=horizon,
    )
    panel.attrs = {}
    panel.to_parquet(cache)
    return panel


def _params(seed):
    return {**DEFAULT_PARAMS, "random_state": seed, "bagging_seed": seed, "feature_fraction_seed": seed}


def seed_avg_wf(panel, cols, hp):
    """Walk-forward pooled OOS preds, averaged over SEEDS (bagging variance removed)."""
    acc = None
    for s in SEEDS:
        out = walk_forward_predict(panel, feature_cols=cols, params=_params(s), horizon_periods=hp)
        if acc is None:
            acc = out[["date", "label_excess"]].copy()
            acc["pred"] = out["pred"].to_numpy()
        else:
            acc["pred"] += out["pred"].to_numpy()
    acc["pred"] /= len(SEEDS)
    return acc


def block_boot_mean(x, block, n_boot=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    return x.mean(), np.percentile(means, [2.5, 97.5]), float(np.mean(means > 0))


def cpcv_paired(panel, hp, seed=42):
    dates = sorted(panel["date"].unique())
    p = _params(seed)
    out = {"base": [], "sr": []}
    for tr_d, te_d in cpcv_splits(dates, n_groups=10, k_test=2, horizon_periods=hp):
        tr = panel[panel["date"].isin(tr_d)].dropna(subset=["label_excess"])
        te = panel[panel["date"].isin(te_d)]
        if len(tr) < 50 or te.empty:
            continue
        for key, cols in (("base", COLS_BASE), ("sr", COLS_SR)):
            m = lgb.LGBMRegressor(**p).fit(tr[cols].fillna(0.5), tr["label_excess"])
            t = te[["date", "label_excess"]].assign(pred=m.predict(te[cols].fillna(0.5)))
            t.attrs = {}
            out[key].append(rank_ic(t, feature="pred").mean())
    return np.asarray(out["base"]), np.asarray(out["sr"])


def rev_window_ic(panel, close, w, lag):
    """Signed cross-sectional IC of the raw trailing-w return vs this panel's label_excess."""
    rev = close / close.shift(w) - 1.0
    vals = []
    for d, g in panel.groupby("date"):
        if d not in rev.index:
            continue
        sub = pd.DataFrame({"f": g["ticker"].map(rev.loc[d]).to_numpy(),
                            "y": g["label_excess"].to_numpy()}).dropna()
        if sub["f"].nunique() < 3:
            continue
        vals.append(sub.corr(method="spearman").iloc[0, 1])
    s = pd.Series(vals, dtype=float).dropna()
    return s.mean(), newey_west_tstat(s.to_numpy(), lag=lag)


def main() -> int:
    close, _ = load_matrices_cached()
    if close.empty:
        print("no prices; run scripts/ingest_prices.py first")
        return 1
    panels = {}
    for h in (21, 63):
        panels[h] = get_panel(h)
        print(f"panel h={h}: {len(panels[h]):,} rows  {panels[h]['date'].nunique()} dates  "
              f"st_rev cov {panels[h]['st_rev'].notna().mean():.0%}", flush=True)

    print("\n=== construction scan: signed cross-sectional IC (t_nw); reversal => NEGATIVE ===", flush=True)
    for h in (21, 63):
        line = f"  label {h}d:"
        for w in (5, 10, 21, 42, 63):
            ic, t = rev_window_ic(panels[h], close, w, HOR[h]["lag"])
            line += f"  w{w}={ic:+.4f}(t{t:+.1f})"
        print(line, flush=True)

    print("\n=== st_rev standalone sector-rank IC ===", flush=True)
    for h in (21, 63):
        s = ic_summary(rank_ic(panels[h], feature="st_rev_rank"), overlap_lag=HOR[h]["lag"])
        print(f"  label {h}d: IC {s['mean_ic']:+.4f}  t_nw {s['t_nw']:+.2f}", flush=True)

    for h in (21, 63):
        cfg = HOR[h]
        print(f"\n=== MODEL-LEVEL, label {h}d (hp={cfg['hp']}) ===", flush=True)
        base, sr = seed_avg_wf(panels[h], COLS_BASE, cfg["hp"]), seed_avg_wf(panels[h], COLS_SR, cfg["hp"])
        ib = ic_summary(rank_ic(base, feature="pred"), overlap_lag=cfg["lag"])
        is_ = ic_summary(rank_ic(sr, feature="pred"), overlap_lag=cfg["lag"])
        print(f"  WF (seed-avg): baseline IC {ib['mean_ic']:+.4f} t {ib['t_nw']:+.2f}  |  "
              f"+st_rev IC {is_['mean_ic']:+.4f} t {is_['t_nw']:+.2f}  |  "
              f"ΔIC {is_['mean_ic']-ib['mean_ic']:+.4f}", flush=True)

        d = (rank_ic(sr, feature="pred") - rank_ic(base, feature="pred")).dropna()
        m, ci, pg = block_boot_mean(d.to_numpy(), block=cfg["block"])
        print(f"  BOOTSTRAP per-date ΔIC (n={len(d)}, block={cfg['block']}): mean {m:+.4f}  "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  P(ΔIC>0)={pg:.0%}", flush=True)

        cb, cs = cpcv_paired(panels[h], cfg["hp"])
        dd = cs - cb
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if dd.std() > 0 else float("nan")
        print(f"  CPCV paired (seed42): base {cb.mean():+.4f}  +st_rev {cs.mean():+.4f}  "
              f"Δmean {dd.mean():+.4f}  frac(Δ>0) {np.mean(dd>0):.0%}  (inflated paired t {t:+.2f})",
              flush=True)

    print("\nREAD: a real ~1-month reversal is STRONGEST at the 21d label. st_rev being dead at 21d "
          "(ΔIC bootstrap CI crossing 0) while only helping at 63d = horizon-specific artifact, not "
          "a stable signal. Do NOT promote.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
