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
