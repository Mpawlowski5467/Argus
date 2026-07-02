"""Tests for the IC harness: it must SEE a real signal and NOT see a fake one."""

import numpy as np
import pandas as pd

from stockscan.validation import ic_summary, newey_west_tstat, purged_walk_forward, rank_ic


def _panel_with_signal(rng, n_dates=6, n_names=120, noise=0.6):
    frames = []
    for i in range(n_dates):
        d = pd.Timestamp("2024-01-31") + pd.offsets.MonthEnd(i)
        f = rng.standard_normal(n_names)
        lab = f + noise * rng.standard_normal(n_names)  # label genuinely correlated with feature
        frames.append(
            pd.DataFrame(
                {"date": d, "ticker": [f"T{j}" for j in range(n_names)],
                 "feature": f, "label_excess": lab}
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_rank_ic_detects_real_signal():
    p = _panel_with_signal(np.random.default_rng(0))
    s = ic_summary(rank_ic(p))
    assert s["mean_ic"] > 0.5
    assert s["t_nw"] > 3


def test_rank_ic_collapses_when_labels_shuffled():
    rng = np.random.default_rng(1)
    p = _panel_with_signal(rng)
    p["label_excess"] = p.groupby("date")["label_excess"].transform(
        lambda x: rng.permutation(x.to_numpy())
    )
    s = ic_summary(rank_ic(p))
    assert abs(s["t_nw"]) < 2  # no signal survives the shuffle


def test_newey_west_zero_mean_series_is_insignificant():
    x = np.random.default_rng(2).standard_normal(300)
    assert abs(newey_west_tstat(x, lag=2)) < 3


def test_purged_walk_forward_is_ordered_and_disjoint():
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    splits = purged_walk_forward(dates, n_splits=4, embargo=2, horizon_periods=3)
    assert len(splits) == 4
    for train, test in splits:
        assert max(train) < min(test)
        assert set(train).isdisjoint(set(test))
        # purge+embargo leaves a real gap between train end and test start
        assert (min(test) - max(train)).days >= 90
