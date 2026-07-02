"""The model must learn a real signal out-of-sample and find nothing in pure noise."""

import numpy as np
import pandas as pd

from stockscan.model import RANK_COLS, evaluate


def _panel(rng, signal: bool, n_dates=30, n=150):
    frames = []
    for i in range(n_dates):
        d = pd.Timestamp("2019-01-31") + pd.offsets.MonthEnd(i)
        df = pd.DataFrame({c: rng.uniform(0, 1, n) for c in RANK_COLS})
        df["date"] = d
        if signal:
            f = df[RANK_COLS[0]].to_numpy()
            df["label_excess"] = (f - 0.5) * 0.2 + rng.normal(0, 0.03, n)
        else:
            df["label_excess"] = rng.normal(0, 0.05, n)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_model_learns_signal_out_of_sample():
    panel = _panel(np.random.default_rng(0), signal=True)
    summ = evaluate(panel, n_splits=4, embargo=1, horizon_periods=1)
    assert summ is not None
    assert summ["mean_ic"] > 0.1
    assert summ["decile_spread"] > 0


def test_model_finds_nothing_in_noise():
    panel = _panel(np.random.default_rng(1), signal=False)
    summ = evaluate(panel, n_splits=4, embargo=1, horizon_periods=1)
    assert summ is not None
    assert abs(summ["t_nw"]) < 3  # no real edge -> not significant
