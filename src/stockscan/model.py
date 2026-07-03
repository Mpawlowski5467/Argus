"""Walk-forward LightGBM model + out-of-sample evaluation + the frozen product artifact.

One small, regularized gradient-boosted model, trained under a purged + embargoed
expanding walk-forward so the reported rank IC and decile spread are genuinely
out-of-sample. Kept deliberately shallow with heavy regularization to resist the
overfitting that makes free-data backtests lie.

The PRODUCT model is a separate, one-shot fit on the full honest panel, frozen to
``artifacts/model/`` (booster + metadata). Serving loads the artifact and only ever
calls :meth:`Artifact.score` -- there is no retrain path at serve time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, LABEL_HORIZON_DAYS
from .features import FEATURES
from .validation import ic_summary, purged_walk_forward, rank_ic

RANK_COLS = [f"{f}_rank" for f in FEATURES]
MODEL_DIR = ARTIFACTS_DIR / "model"

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
    id_cols: tuple = (),
) -> pd.DataFrame:
    """Train per fold on the past, predict the held-out future. Returns pooled OOS preds.

    ``id_cols`` (e.g. ``("cik", "ticker", "sector")``) are carried through so the
    backtester can key positions off the predictions.
    """
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
        out = te[["date", *id_cols, label]].copy()
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


# --- frozen product artifact ----------------------------------------------------

def fit(
    panel: pd.DataFrame,
    feature_cols=None,
    label: str = "label_excess",
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    """One-shot fit on every labeled row of ``panel`` (the artifact's training step)."""
    feature_cols = feature_cols or RANK_COLS
    params = {**DEFAULT_PARAMS, **(params or {})}
    tr = panel.dropna(subset=[label])
    model = lgb.LGBMRegressor(**params)
    model.fit(tr[feature_cols].fillna(0.5), tr[label])
    return model


def save_artifact(
    model: lgb.LGBMRegressor,
    panel: pd.DataFrame,
    out_dir: Path = MODEL_DIR,
    feature_cols=None,
    label: str = "label_excess",
    extra: dict | None = None,
) -> Path:
    """Freeze the model + the metadata a scorer needs to refuse silent drift.

    ``meta.json`` records the exact feature columns (order matters to the booster),
    the training-date cutoff, and the panel shape; anything the caller wants on the
    honesty trail (config floors, universe size) goes in ``extra``.
    """
    feature_cols = list(feature_cols or RANK_COLS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled = panel.dropna(subset=[label])
    meta = {
        "feature_cols": feature_cols,
        "label": label,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "trained_through": str(pd.Timestamp(labeled["date"].max()).date()),
        "n_rows": int(len(labeled)),
        "n_dates": int(labeled["date"].nunique()),
        "params": {k: v for k, v in model.get_params().items() if k in DEFAULT_PARAMS},
        "lightgbm_version": lgb.__version__,
        **(extra or {}),
    }
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


@dataclass
class Artifact:
    """A frozen model. Has no fit method by design: serving cannot retrain."""

    booster: lgb.Booster
    meta: dict

    @property
    def feature_cols(self) -> list[str]:
        return list(self.meta["feature_cols"])

    @property
    def trained_through(self) -> pd.Timestamp:
        return pd.Timestamp(self.meta["trained_through"])

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Score a cross-section. Columns are selected in the artifact's stored order."""
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"artifact expects feature columns missing from input: {missing}")
        return self.booster.predict(X[self.feature_cols].fillna(0.5).to_numpy())

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature signed contributions to each row's score (TreeSHAP via
        LightGBM's native ``pred_contrib`` — no extra dependency). Columns are
        ``feature_cols + ['base']``; row sums equal :meth:`score` outputs, so the
        narration's "drivers" are an exact decomposition, not an approximation."""
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"artifact expects feature columns missing from input: {missing}")
        contrib = self.booster.predict(
            X[self.feature_cols].fillna(0.5).to_numpy(), pred_contrib=True
        )
        return pd.DataFrame(contrib, columns=[*self.feature_cols, "base"], index=X.index)


def load_artifact(path: Path = MODEL_DIR) -> Artifact:
    """Load the frozen artifact. Raises FileNotFoundError if none has been trained."""
    path = Path(path)
    model_file = path / "model.txt"
    if not model_file.exists():
        raise FileNotFoundError(
            f"no trained artifact at {path}; run scripts/train_model.py first"
        )
    booster = lgb.Booster(model_file=str(model_file))
    meta = json.loads((path / "meta.json").read_text())
    saved = meta.get("lightgbm_version")
    if saved and saved != lgb.__version__:
        import warnings

        warnings.warn(
            f"artifact was frozen with lightgbm {saved} but runtime is {lgb.__version__}; "
            f"scores may drift — retrain or pin the dependency",
            stacklevel=2,
        )
    return Artifact(booster=booster, meta=meta)
