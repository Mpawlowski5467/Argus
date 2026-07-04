"""Distress-overlay gate: does a learned P(distress) veto improve the net-of-cost book?

The distress head passed its own ranking gate (scripts/run_distress.py). This asks the
only question that justifies wiring it into trading: used as a HARD DISTRESS EXIT — veto
any name whose out-of-sample P(distress) clears a threshold from the long book — does the
return strategy's NET-of-cost, after-turnover curve actually get better (higher Sharpe /
shallower drawdown) than the baseline that has no such veto and the old -0.70 haircut?

Everything is out-of-sample and PIT-honest, exactly as Phase 3:
- the SAME no-impute, liquidity-filtered tradable panel the return backtest uses;
- return scores AND distress probabilities are BOTH purged-walk-forward OOS (never the
  in-sample frozen artifact — that would be look-ahead), scored on the identical rank basis;
- the veto is applied by pushing a flagged name's score below the exit band, so a held
  name is force-exited and a fresh one never enters (a real "hard distress exit").

A threshold SWEEP (not one tuned number) plus a mechanism diagnostic (do high-distress
names actually die / underperform inside the tradable universe?) keep the verdict honest.

  uv run python scripts/run_distress_overlay.py [--horizon-months 12]

This is an OFFLINE gate. It does NOT modify serve, the monitor, or the paper book.
"""

import argparse

import duckdb
import pandas as pd

from stockscan.backtest import run_backtest
from stockscan.concepts import WIDE_PATH
from stockscan.config import MIN_DOLLAR_VOLUME
from stockscan.distress import (
    DEFAULT_HORIZON_MONTHS,
    attach_distress_label,
    classify_distress_events,
    walk_forward_predict_proba,
)
from stockscan.edgar.delistings import load_delistings
from stockscan.features import compute_features
from stockscan.fundamental_panel import build_fundamental_panel
from stockscan.intrinio_universe import universe_ticker_map
from stockscan.model import walk_forward_predict
from stockscan.panel import load_matrices

WINSOR = (0.01, 0.99)
THRESHOLDS = (0.03, 0.05, 0.08, 0.12)  # absolute P(distress) veto levels (base rate ~0.02)


def _fmt(s: dict) -> str:
    return (f"net CAGR {s['cagr_net']:+.2%}  Sharpe {s['sharpe_net']:+.2f}  "
            f"vol {s['ann_vol']:.1%}  maxDD {s['max_drawdown']:.1%}  "
            f"turn {s['ann_turnover']:.1f}x/yr  nL {s['avg_n_long']:.0f}")


def veto(scores: pd.DataFrame, dprob: pd.DataFrame, thresh: float) -> tuple[pd.DataFrame, float]:
    """Push vetoed (P(distress) >= thresh) names below the exit band -> excluded / force-exited."""
    m = scores.merge(dprob, on=["date", "ticker"], how="left")
    flagged = m["dprob"].fillna(0.0) >= thresh
    out = m[["date", "ticker", "pred"]].copy()
    out.loc[flagged.values, "pred"] = out["pred"].min() - 1.0
    return out, float(flagged.mean())


def decile_diagnostic(preds: pd.DataFrame, dprob: pd.DataFrame, ylab: pd.DataFrame) -> pd.DataFrame:
    """Inside the tradable universe, does higher OOS P(distress) mean worse outcomes?

    Per date, bucket the OOS-scored names into P(distress) deciles and report the mean
    realized 63-day excess return and the realized N-month distress-death rate of each
    bucket. A monotone drop is the economic proof the veto has something to bite on.
    """
    df = preds.merge(dprob, on=["date", "cik", "ticker"], how="inner") \
              .merge(ylab, on=["date", "cik"], how="left")
    rows = []
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["dprob"])
        if len(g) < 50:
            continue
        q = (g["dprob"].rank(pct=True) * 10).clip(upper=9.999).astype(int)
        for dec in (0, 4, 9):  # bottom, middle, top distress decile
            b = g[q == dec]
            if len(b):
                rows.append({"decile": dec, "ret": b["label_excess"].mean(),
                             "death": b["y"].mean(), "n": len(b)})
    out = pd.DataFrame(rows).groupby("decile").agg(
        mean_fwd_excess=("ret", "mean"), death_rate=("death", "mean"),
        avg_names=("n", "mean")).reset_index()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Distress-overlay net-of-cost gate.")
    ap.add_argument("--horizon-months", type=int, default=DEFAULT_HORIZON_MONTHS)
    args = ap.parse_args(argv)
    N = args.horizon_months

    print("assembling honest tradable panel + OOS return scores + OOS distress probs ...")
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
    # distress label on the identical tradable rows, then OOS distress probabilities
    events = classify_distress_events(load_delistings(), close, tmap or {})
    panel = attach_distress_label(panel, events, horizon_months=N)

    preds = walk_forward_predict(panel, id_cols=("cik", "ticker", "sector"))
    dwf = walk_forward_predict_proba(panel, label="y", horizon_periods=N,
                                     id_cols=("cik", "ticker"))
    if preds.empty or dwf.empty:
        print("no OOS predictions")
        return 1
    dprob = dwf.rename(columns={"prob": "dprob"})[["date", "cik", "ticker", "dprob"]]
    scores = preds[["date", "ticker", "pred"]]
    dv_med = dv.rolling(20, min_periods=10).median()
    common = dict(close=close, opn=opn, dv_med=dv_med)

    n_dates = preds["date"].nunique()
    base_rate = float(panel.dropna(subset=["y"])["y"].mean())
    print(f"OOS dates: {n_dates} ({preds['date'].min().date()} -> {preds['date'].max().date()})  "
          f"tradable panel rows={len(panel):,}  distress base rate (tradable)={base_rate:.4f}")

    # ---- mechanism: does distress rank outcomes INSIDE the tradable book?
    ylab = panel[["date", "cik", "y"]]
    diag = decile_diagnostic(preds, dprob, ylab)
    print("\n== does P(distress) rank outcomes in the tradable universe? (OOS) ==")
    print("  distress decile |  mean fwd 63d excess |  realized 12m death rate |  ~names")
    for _, r in diag.iterrows():
        tag = {0: "bottom", 4: "middle", 9: "TOP  "}[int(r["decile"])]
        print(f"    {tag} (d{int(r['decile'])})   |   {r['mean_fwd_excess']:+.4f}          "
              f"|   {r['death_rate']:.4f}                |  {r['avg_names']:.0f}")

    # ---- baseline books (no veto)
    base_lo = run_backtest(scores, mode="long_only", **common)
    base_uni = run_backtest(scores, mode="universe", **common)
    s_lo, s_uni = base_lo.summary(), base_uni.summary()
    print("\n== baseline (no distress veto) ==")
    print(f"  universe EW : {_fmt(s_uni)}")
    print(f"  long-only   : {_fmt(s_lo)}")

    # ---- overlay sweep (hard distress exit at each threshold)
    print("\n== long-only WITH distress veto (hard exit at P(distress) >= t) ==")
    results = []
    for t in THRESHOLDS:
        vscores, frac = veto(scores, dprob, t)
        s = run_backtest(vscores, mode="long_only", **common).summary()
        results.append((t, s, frac))
        d_sharpe = s["sharpe_net"] - s_lo["sharpe_net"]
        d_dd = s["max_drawdown"] - s_lo["max_drawdown"]
        d_cagr = s["cagr_net"] - s_lo["cagr_net"]
        print(f"  t={t:.2f} (veto ~{frac*100:4.1f}% of names): {_fmt(s)}"
              f"   ΔSharpe {d_sharpe:+.2f}  ΔmaxDD {d_dd:+.1%}  ΔCAGR {d_cagr:+.2%}")

    # ---- the natural home the diagnostic points to: distress as a short/risk sleeve.
    # Drive the engine with pred = -P(distress): long tail = safest names, short tail =
    # riskiest (borrow-filtered by ADV, borrow charged per night). long/short MINUS the
    # long-only-safe leg isolates the short sleeve's marginal contribution net of borrow.
    dscore = dprob[["date", "ticker"]].copy()
    dscore["pred"] = -dprob["dprob"].to_numpy()
    s_safe = run_backtest(dscore, mode="long_only", **common).summary()
    s_ls = run_backtest(dscore, mode="long_short", **common).summary()
    top_ret = float(diag.set_index("decile")["mean_fwd_excess"].get(9, float("nan")))
    print("\n== distress as a short/risk sleeve (borrow-aware; where the signal lives) ==")
    print(f"  gross: top-decile names average {top_ret:+.2%} 63d excess before any cost")
    print(f"  long-only (safe tilt, -P(distress)):     {_fmt(s_safe)}")
    print(f"  long/short (add high-distress shorts):   {_fmt(s_ls)}")
    short_adds = s_ls["sharpe_net"] > s_safe["sharpe_net"]
    print(f"  short sleeve net of borrow: ΔSharpe {s_ls['sharpe_net'] - s_safe['sharpe_net']:+.2f}  "
          f"ΔCAGR {s_ls['cagr_net'] - s_safe['cagr_net']:+.2%}  -> "
          f"{'shorting distress adds net of borrow' if short_adds else 'borrow eats the short edge'}")

    # ---- verdict: overlay must improve risk-adjusted net performance, robustly
    improved = [(t, s) for t, s, _ in results
                if s["sharpe_net"] >= s_lo["sharpe_net"] or s["max_drawdown"] > s_lo["max_drawdown"]]
    best_t, best_s = max(results, key=lambda r: r[1]["sharpe_net"])[:2]
    robust = len(improved) >= len(THRESHOLDS) - 1  # helps (or neutral) across ~all thresholds
    mono = diag.set_index("decile")["death_rate"]
    signal_real = bool(mono.get(9, 0) > mono.get(0, 1))  # top distress decile dies more than bottom
    print("\n== DISTRESS-OVERLAY GATE ==")
    print(f"  [{'PASS' if signal_real else 'FAIL'}] P(distress) separates outcomes in the "
          f"tradable book (top-decile death {mono.get(9, float('nan')):.4f} vs bottom "
          f"{mono.get(0, float('nan')):.4f})")
    print(f"  [{'PASS' if robust else 'FAIL'}] veto improves or holds net Sharpe/maxDD across "
          f"{len(improved)}/{len(THRESHOLDS)} thresholds")
    print(f"  best: t={best_t:.2f}  Sharpe {best_s['sharpe_net']:+.2f} (base {s_lo['sharpe_net']:+.2f})  "
          f"maxDD {best_s['max_drawdown']:.1%} (base {s_lo['max_drawdown']:.1%})  "
          f"net CAGR {best_s['cagr_net']:+.2%} (base {s_lo['cagr_net']:+.2%})")
    verdict = "WIRE the distress veto (net-of-cost improvement is real and robust)" if (
        signal_real and robust and best_s["sharpe_net"] > s_lo["sharpe_net"]) else (
        "KEEP distress as a monitor/short signal — it does not improve the long book net-of-cost")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
