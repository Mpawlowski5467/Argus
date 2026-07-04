"""Large-drawdown risk head — a learned P(deep peak-to-trough fall within N months).

A third model head, a near-clone of :mod:`stockscan.distress`: same point-in-time
fundamental ranks (``RANK_COLS``), same serve-parity seam (``pit_snapshot`` /
``add_sector_ranks``), same purged walk-forward + CPCV validation and rare-event
scorecard. It reuses distress's head-agnostic validators outright
(``walk_forward_predict_proba`` / ``cpcv_auc`` / ``distress_metrics`` / ``fit_distress``);
the ONLY thing that changes is the LABEL.

The label is a pure PRICE-PATH read, no ledger needed: for a holder entering at month-end
``d``, ``y = 1`` if the name suffers a peak-to-trough drawdown at or beyond ``threshold``
(default −30%) at any point over the next ``horizon_months`` — the running peak seeded at
``d``'s price, so both a run-up-then-fall and an outright slide count. A name that delists
mid-window carries its real terminal prices, so a distress collapse lands as a deep drawdown
(``y=1``) while a premium acquisition does not (``y=0``); the label never peeks past the price
data edge (dates whose window is unobserved are dropped, not guessed).

Like distress, this trains and freezes a SEPARATE artifact (``artifacts/drawdown_model/``);
it never touches the return artifact, the serve path's score, or any trade rule. It is a
display/risk flag and a confidence input only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, MAX_STALE_DAYS, MIN_SECTOR_BUCKET
# Head-agnostic event-classifier machinery — shared with the distress head so both stay
# in lockstep (same purged WF, same CPCV, same rare-event scorecard, same binary fit).
from .distress import (  # noqa: F401  (re-exported for the run script)
    RANK_COLS,
    cpcv_auc,
    distress_metrics as drawdown_metrics,
    fit_distress as fit_drawdown,
    walk_forward_predict_proba,
)
from .features import FEATURES  # noqa: F401  (kept for parity with distress' import surface)
from .fundamental_panel import add_sector_ranks, pit_snapshot, prepare_features
from .panel import month_end_dates

DRAWDOWN_MODEL_DIR = ARTIFACTS_DIR / "drawdown_model"

DEFAULT_HORIZON_MONTHS = 6      # forward window over which a deep drawdown counts
DEFAULT_THRESHOLD = -0.30       # peak-to-trough fall that counts as a "large" drawdown

# Display/alert flag levels for the FIREWALLED risk-flag layer (never a trade input).
# Placeholders pending calibration on the realized base rate — a 6-month 30% drawdown is
# far more common than a distress delisting, so these are tuned once the panel is built.
DRAWDOWN_FLAG_THRESHOLDS = (("high", 0.60), ("elevated", 0.40))


# --- label: forward peak-to-trough drawdown -------------------------------------

def forward_max_drawdown(
    close: pd.DataFrame, cols, d: pd.Timestamp, window_end: pd.Timestamp
) -> pd.Series:
    """Worst peak-to-trough drawdown per column over ``(d, window_end]``, vectorized.

    The running peak is seeded at the last price at/before ``d`` (a holder's entry), so a
    name that runs up then falls, and a name that just slides, both register their true
    max drawdown. ``NaN`` for a column with no entry price or no forward prices (unlabelable);
    a name that delists mid-window contributes its real prices through the last trade, then
    ``NaN`` — so ``min`` captures the terminal collapse (or lack of one) without inventing a
    post-death value.
    """
    cols = [c for c in cols if c in close.columns]
    if not cols:
        return pd.Series(dtype="float64")
    sub = close[cols]
    at_or_before = sub.loc[:d]
    if at_or_before.empty:
        return pd.Series(dtype="float64")
    anchor = at_or_before.iloc[-1]                                   # entry price per col
    win = sub.loc[(sub.index > d) & (sub.index <= window_end)]
    if win.empty:
        return pd.Series(dtype="float64")
    path = pd.concat([anchor.to_frame().T, win])                    # seed peak at the entry row
    dd = path / path.cummax() - 1.0
    return dd.min(axis=0)                                            # col -> min drawdown (<=0)


def build_drawdown_panel(
    features_df: pd.DataFrame,
    close: pd.DataFrame,
    ticker_map: dict,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    threshold: float = DEFAULT_THRESHOLD,
    max_stale_days: int = MAX_STALE_DAYS,
    min_names: int = 30,
    censor_date=None,
) -> pd.DataFrame:
    """Point-in-time drawdown panel: one row per (cik, rebalance date) with a binary label.

    For each month-end ``d`` we snapshot the latest 10-K public at ``d`` (:func:`pit_snapshot`),
    rank the fundamentals within sector over that cross-section (:func:`add_sector_ranks` —
    the serve-parity transform), and attach ``y = 1`` iff the name's forward peak-to-trough
    drawdown over ``(d, d + horizon_months]`` is at or beyond ``threshold``. Names without an
    entry price or forward prices are dropped (unlabelable); rebalance dates whose window
    extends past ``censor_date`` (default: the last price date) are skipped, so no label peeks
    past the observed price history. Survivorship-correct: dead names carry a 10-K and real
    prices until death, so their collapses are positives in the cross-section, not absences.
    """
    feats = prepare_features(features_df)
    last_price_date = pd.Timestamp(close.index.max())
    censor_date = pd.Timestamp(censor_date) if censor_date is not None else last_price_date

    rows, coverage = [], []
    for d in month_end_dates(close.index):
        window_end = d + pd.DateOffset(months=horizon_months)
        if window_end > censor_date:
            continue  # forward window not fully observed -> would fabricate false negatives
        latest = pit_snapshot(feats, d, max_stale_days)
        if latest.empty:
            continue
        latest = add_sector_ranks(latest.copy(), MIN_SECTOR_BUCKET)
        latest["col"] = latest["cik"].map(ticker_map)
        cols = [c for c in latest["col"].dropna().unique() if c in close.columns]
        if not cols:
            continue
        mdd = forward_max_drawdown(close, cols, d, window_end)
        latest["mdd"] = latest["col"].map(mdd)
        latest = latest[latest["mdd"].notna()]                      # keep only labelable names
        if len(latest) < min_names:
            continue
        latest["y"] = (latest["mdd"] <= threshold).astype("int8")
        latest["date"] = d
        rows.append(latest)
        coverage.append({
            "date": d, "universe": len(latest),
            "positives": int(latest["y"].sum()), "base_rate": float(latest["y"].mean()),
        })

    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel.attrs["coverage"] = pd.DataFrame(coverage)
    panel.attrs["censor_date"] = censor_date
    panel.attrs["horizon_months"] = horizon_months
    panel.attrs["threshold"] = threshold
    return panel


# --- frozen product artifact (separate dir; never touches artifacts/model/) ------

def save_drawdown_artifact(
    model: lgb.LGBMClassifier,
    panel: pd.DataFrame,
    out_dir: Path = DRAWDOWN_MODEL_DIR,
    feature_cols=None,
    label: str = "y",
    extra: dict | None = None,
) -> Path:
    """Freeze the classifier + the metadata a scorer needs to refuse silent drift."""
    feature_cols = list(feature_cols or RANK_COLS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled = panel.dropna(subset=[label])
    meta = {
        "head": "drawdown",
        "feature_cols": feature_cols,
        "label": label,
        "horizon_months": int(panel.attrs.get("horizon_months", DEFAULT_HORIZON_MONTHS)),
        "threshold": float(panel.attrs.get("threshold", DEFAULT_THRESHOLD)),
        "trained_through": str(pd.Timestamp(labeled["date"].max()).date()),
        "censor_date": str(pd.Timestamp(panel.attrs["censor_date"]).date())
        if panel.attrs.get("censor_date") is not None else None,
        "n_rows": int(len(labeled)),
        "n_positives": int(labeled[label].sum()),
        "base_rate": float(labeled[label].mean()),
        "n_dates": int(labeled["date"].nunique()),
        "params": {k: v for k, v in model.get_params().items()},
        "lightgbm_version": lgb.__version__,
        **(extra or {}),
    }
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_dir


@dataclass
class DrawdownArtifact:
    """A frozen drawdown classifier. No fit method: serving cannot retrain."""

    booster: lgb.Booster
    meta: dict

    @property
    def feature_cols(self) -> list[str]:
        return list(self.meta["feature_cols"])

    @property
    def horizon_months(self) -> int:
        return int(self.meta["horizon_months"])

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def trained_through(self) -> pd.Timestamp:
        return pd.Timestamp(self.meta["trained_through"])

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """P(large drawdown within the horizon) for a cross-section. The binary-objective
        booster's ``predict`` already returns the probability; columns are selected in the
        artifact's stored order."""
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"drawdown artifact expects feature columns missing from input: {missing}")
        return self.booster.predict(X[self.feature_cols].fillna(0.5).to_numpy())


def load_drawdown_artifact(path: Path = DRAWDOWN_MODEL_DIR) -> DrawdownArtifact:
    """Load the frozen drawdown artifact. Raises if none has been trained."""
    path = Path(path)
    model_file = path / "model.txt"
    if not model_file.exists():
        raise FileNotFoundError(
            f"no trained drawdown artifact at {path}; run scripts/run_drawdown_head.py --save first"
        )
    booster = lgb.Booster(model_file=str(model_file))
    meta = json.loads((path / "meta.json").read_text())
    saved = meta.get("lightgbm_version")
    if saved and saved != lgb.__version__:
        import warnings

        warnings.warn(
            f"drawdown artifact was frozen with lightgbm {saved} but runtime is "
            f"{lgb.__version__}; scores may drift — retrain or pin the dependency",
            stacklevel=2,
        )
    return DrawdownArtifact(booster=booster, meta=meta)


def load_drawdown_artifact_optional(path: Path = DRAWDOWN_MODEL_DIR) -> DrawdownArtifact | None:
    """The drawdown head as an OPTIONAL risk-flag layer: the artifact if frozen, else
    ``None`` — serve/monitor/TUI/web run unchanged without it (exactly like distress)."""
    try:
        return load_drawdown_artifact(path)
    except FileNotFoundError:
        return None


def drawdown_flag(prob: float, thresholds=DRAWDOWN_FLAG_THRESHOLDS) -> str:
    """Map a P(large drawdown) to a display flag: ``high`` | ``elevated`` | ``normal``."""
    if prob is None or not np.isfinite(prob):
        return "normal"
    for name, thr in thresholds:
        if prob >= thr:
            return name
    return "normal"
