"""Walk-forward LightGBM model + out-of-sample evaluation.

One small, regularized gradient-boosted model, trained under a purged + embargoed
expanding walk-forward so the reported rank IC and decile spread are genuinely
out-of-sample. Kept deliberately shallow with heavy regularization to resist the
overfitting that makes free-data backtests lie.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES
from .validation import ic_summary, purged_walk_forward, rank_ic

RANK_COLS = [f"{f}_rank" for f in FEATURES]

DEFAULT_PARAMS = dict(
    objective="regression",
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.03,
    min_child_samples=100,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=1.0,
    verbosity=-1,
    n_jobs=-1,
)


def walk_forward_predict(
    panel: pd.DataFrame,
    feature_cols=None,
    label: str = "label_excess",
    n_splits: int = 5,
    embargo: int = 2,
    horizon_periods: int = 3,
    params: dict | None = None,
) -> pd.DataFrame:
    """Train per fold on the past, predict the held-out future. Returns pooled OOS preds."""
    feature_cols = feature_cols or RANK_COLS
    params = {**DEFAULT_PARAMS, **(params or {})}
    dates = sorted(panel["date"].unique())
    splits = purged_walk_forward(
        dates, n_splits=n_splits, embargo=embargo, horizon_periods=horizon_periods
    )
    preds = []
    for train_dates, test_dates in splits:
        tr = panel[panel["date"].isin(train_dates)].dropna(subset=[label])
        te = panel[panel["date"].isin(test_dates)]
        if len(tr) < 50 or te.empty:
            continue
        model = lgb.LGBMRegressor(**params)
        model.fit(tr[feature_cols].fillna(0.5), tr[label])
        out = te[["date", label]].copy()
        out["pred"] = model.predict(te[feature_cols].fillna(0.5))
        out.attrs = {}  # drop inherited panel.attrs (a DataFrame) so pd.concat doesn't choke
        preds.append(out)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def decile_spread(pred_panel, pred="pred", label="label_excess", date="date", n=10) -> float:
    """Mean (top-decile minus bottom-decile) realized label, by prediction, per date."""
    spreads = []
    for _, g in pred_panel.groupby(date):
        if len(g) < n:
            continue
        r = g[pred].rank(method="first")
        q = np.ceil(r / len(g) * n).astype(int).clip(1, n)
        spreads.append(g.loc[q == n, label].mean() - g.loc[q == 1, label].mean())
    return float(np.nanmean(spreads)) if spreads else float("nan")


def evaluate(panel: pd.DataFrame, **kwargs) -> dict | None:
    """Walk-forward train/predict, then summarize OOS rank IC + decile spread."""
    pred = walk_forward_predict(panel, **kwargs)
    if pred.empty:
        return None
    summ = ic_summary(rank_ic(pred, feature="pred", label="label_excess"))
    summ["decile_spread"] = decile_spread(pred)
    summ["oos_dates"] = int(pred["date"].nunique())
    return summ
