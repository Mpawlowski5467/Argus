"""Honest cross-sectional evaluation.

Rank IC computed per date then aggregated over time, with an overlap-aware
(Newey-West) t-stat because overlapping forward-return labels make adjacent
per-date ICs autocorrelated -- the naive t-stat overstates significance. Also a
purged + embargoed walk-forward splitter for Phase-1 model validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import LABEL_HORIZON_DAYS

# Overlap of the forward-return label in monthly rebalance periods (~21 trading days/month):
# a 63-day label overlaps 3 periods, so per-date ICs are autocorrelated at lags 1..2.
_OVERLAP_PERIODS = max(1, -(-LABEL_HORIZON_DAYS // 21))  # ceil(horizon / 21)


def rank_ic(
    panel: pd.DataFrame, feature: str = "feature", label: str = "label_excess", date: str = "date"
) -> pd.Series:
    """Per-date Spearman rank correlation between feature and label."""
    out: dict = {}
    for d, g in panel.groupby(date):
        sub = g[[feature, label]].dropna()
        if sub[feature].nunique() < 3 or sub[label].nunique() < 3:
            continue
        out[d] = spearmanr(sub[feature].to_numpy(), sub[label].to_numpy())[0]
    return pd.Series(out, dtype="float64").dropna()


def newey_west_tstat(x, lag: int) -> float:
    """t-stat of the mean of ``x`` with a Newey-West correction up to ``lag`` autocovariances."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1)
        var += 2.0 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def ic_summary(ic: pd.Series, overlap_lag: int = _OVERLAP_PERIODS - 1) -> dict:
    """Summarize an IC series: mean, std, naive t-stat, and overlap-corrected t-stat."""
    ic = ic.dropna()
    n = len(ic)
    mean = float(ic.mean()) if n else float("nan")
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    t_naive = float(mean / std * np.sqrt(n)) if std and std > 0 else float("nan")
    return {
        "n": n,
        "mean_ic": mean,
        "std_ic": std,
        "t_naive": t_naive,
        "t_nw": newey_west_tstat(ic.to_numpy(), lag=overlap_lag),
    }


def cpcv_splits(
    dates,
    n_groups: int = 10,
    k_test: int = 2,
    embargo: int = 2,
    horizon_periods: int = _OVERLAP_PERIODS,
):
    """Combinatorial Purged CV: every C(n_groups, k_test) pair of contiguous date
    groups serves as a test set, training on the rest with a purge of
    ``horizon_periods + embargo`` periods around EVERY test block (both sides —
    labels look forward, so leakage runs in both directions across a boundary).

    Returns a list of ``(train_dates, test_dates)``. Aggregating per-combination OOS
    metrics gives a DISTRIBUTION of the edge instead of one walk-forward number.
    """
    from itertools import combinations

    dates = list(pd.to_datetime(pd.Index(dates).unique()).sort_values())
    m = len(dates)
    if m < n_groups:
        return []
    bounds = np.array_split(np.arange(m), n_groups)
    gap = horizon_periods + embargo
    out = []
    for combo in combinations(range(n_groups), k_test):
        test_pos = np.concatenate([bounds[g] for g in combo])
        test_mask = np.zeros(m, dtype=bool)
        test_mask[test_pos] = True
        # dilate the test mask by `gap` positions on each side -> purged zone
        purged = test_mask.copy()
        for s in range(1, gap + 1):
            purged[s:] |= test_mask[:-s]
            purged[:-s] |= test_mask[s:]
        train = [dates[j] for j in range(m) if not purged[j]]
        test = [dates[j] for j in test_pos]
        if train and test:
            out.append((train, test))
    return out


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting via CSCV (Bailey/López de Prado).

    ``returns``: T x N matrix — one return series per strategy TRIAL (every variant
    actually tried, so selection bias is measured, not hidden). Rows are split into
    ``n_blocks`` contiguous blocks; for every half/half combination the in-sample
    winner's OUT-of-sample relative rank ``omega`` is logged. PBO = fraction of
    combinations where the IS winner lands in the OOS bottom half.
    """
    from itertools import combinations

    rets = returns.dropna(how="all").fillna(0.0)
    T, n = rets.shape
    if n < 2 or T < n_blocks:
        return {"pbo": float("nan"), "n_combos": 0}

    def perf(frame: pd.DataFrame) -> pd.Series:  # Sharpe-like, frequency-agnostic
        mu, sd = frame.mean(), frame.std(ddof=1)
        return mu / sd.replace(0.0, np.nan)

    blocks = np.array_split(np.arange(T), n_blocks)
    lambdas, skipped = [], 0
    for combo in combinations(range(n_blocks), n_blocks // 2):
        in_set = set(combo)
        is_idx = np.concatenate([blocks[b] for b in combo])
        oos_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in in_set])
        perf_is = perf(rets.iloc[is_idx])
        perf_oos = perf(rets.iloc[oos_idx])
        if perf_is.isna().all():
            skipped += 1
            continue
        best = perf_is.idxmax()
        rank = perf_oos.rank().get(best, np.nan)
        if not np.isfinite(rank):  # winner's OOS half degenerate: uninformative, not a pass
            skipped += 1
            continue
        omega = float(rank) / (n + 1)
        lambdas.append(np.log(omega / (1.0 - omega)))
    lam = np.asarray(lambdas)
    return {
        "pbo": float((lam <= 0).mean()) if len(lam) else float("nan"),
        "n_combos": int(len(lam)),
        "n_skipped": skipped,
        "lambda_mean": float(lam.mean()) if len(lam) else float("nan"),
    }


def purged_walk_forward(
    dates, n_splits: int = 5, embargo: int = 2, horizon_periods: int = _OVERLAP_PERIODS
):
    """Expanding-window splits with a purge+embargo gap between train and test.

    ``dates`` are the sorted unique rebalance dates. Train dates whose label window
    (``horizon_periods``) would overlap the test block are purged, plus ``embargo``
    extra periods. Returns a list of ``(train_dates, test_dates)``.
    """
    dates = list(pd.to_datetime(pd.Index(dates).unique()).sort_values())
    m = len(dates)
    fold = m // (n_splits + 1)
    if fold == 0:
        return []
    gap = horizon_periods + embargo
    out = []
    for i in range(1, n_splits + 1):
        test_lo = fold * i
        test_hi = fold * (i + 1) if i < n_splits else m
        train = dates[: max(0, test_lo - gap)]
        test = dates[test_lo:test_hi]
        if train and test:
            out.append((train, test))
    return out
