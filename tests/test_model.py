"""The model must learn a real signal out-of-sample and find nothing in pure noise."""

import numpy as np
import pandas as pd
import pytest

from stockscan.model import RANK_COLS, evaluate, fit, load_artifact, save_artifact


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


def test_artifact_roundtrip_scores_identically_and_cannot_retrain(tmp_path):
    panel = _panel(np.random.default_rng(2), signal=True)
    model = fit(panel, params=dict(n_estimators=25, min_child_samples=10))
    out = save_artifact(model, panel, out_dir=tmp_path, extra={"mode": "test"})
    assert (out / "model.txt").exists() and (out / "meta.json").exists()

    art = load_artifact(tmp_path)
    assert art.feature_cols == RANK_COLS
    assert art.meta["mode"] == "test"
    assert art.trained_through == panel["date"].max()  # training-date cutoff recorded
    assert art.meta["n_dates"] == panel["date"].nunique()

    X = panel[RANK_COLS].head(200)
    np.testing.assert_allclose(art.score(X), model.predict(X.fillna(0.5)))
    assert not hasattr(art, "fit")  # frozen: the serve side has no retrain path


def test_artifact_refuses_wrong_feature_columns(tmp_path):
    panel = _panel(np.random.default_rng(3), signal=True)
    save_artifact(fit(panel, params=dict(n_estimators=5)), panel, out_dir=tmp_path)
    art = load_artifact(tmp_path)
    with pytest.raises(KeyError):
        art.score(panel[RANK_COLS[:-1]])  # a missing feature must fail loudly
