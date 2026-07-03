"""Distress-risk classifier head — a learned P(distress-delist within N months).

A second, independent model that predicts the probability a company distress-delists
inside a forward window. It is ORTHOGONAL to the return head (``stockscan.model``):
same point-in-time fundamental ranks (``RANK_COLS``), same parity seam
(``prepare_features`` / ``pit_snapshot`` / ``add_sector_ranks``), same purged
walk-forward + CPCV validation — but a binary target and rare-event metrics
(ROC-AUC, PR-AUC, precision/recall @ top-decile, calibration) instead of rank IC.

The honest part is the LABEL. EDGAR's delisting ledger tags each death ``delist``
(Form 25 exchange delisting) or ``dereg`` (Form 15 going-dark / deregistration), but
NEITHER cleanly means "distress": most Form 25s are M&A (the target is acquired at or
above the market price) and a large minority of Form 15s are going-private / LBO at a
premium. Labeling on ``reason`` alone would fill the positive class with benign exits.
So we price-CONFIRM distress: a death counts as positive only when the security's own
terminal price collapsed (sub-$1 print, or a >=70% fall from its trailing-1y high, or a
>50% trailing-year loss into the delisting). Deaths with a price but no collapse are
benign NEGATIVES (real acquisitions); deaths we cannot price at all are AMBIGUOUS and
dropped rather than guessed. Everything is point-in-time: features come from the last
10-K public at the as-of date, and the label peeks forward only inside the intended
N-month window (validation purges that overlap).

This module trains and freezes a SEPARATE artifact (``artifacts/distress_model/``); it
never touches the return artifact, the serve path, or any trade rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ARTIFACTS_DIR, MAX_STALE_DAYS, MIN_SECTOR_BUCKET
from .features import FEATURES
from .fundamental_panel import add_sector_ranks, pit_snapshot, prepare_features
from .panel import month_end_dates
from .validation import cpcv_splits, purged_walk_forward

RANK_COLS = [f"{f}_rank" for f in FEATURES]
DISTRESS_MODEL_DIR = ARTIFACTS_DIR / "distress_model"

DEFAULT_HORIZON_MONTHS = 12  # forward window over which a distress delisting counts

# Display/alert flag levels for the FIREWALLED risk-flag layer (never a trade input).
# Descending; a name is flagged at the first threshold its P(distress) clears. Anchored to
# the head's calibration (top bin ~0.06 realized) and the overlay sweep: ~0.03 is ~1.5x the
# full-universe base rate (~7x the tradable base), ~0.08 is a genuine outlier.
DISTRESS_FLAG_THRESHOLDS = (("high", 0.08), ("elevated", 0.03))

# Price-based confirmation thresholds separating distress from benign (M&A / premium
# going-private). A death is distress if ANY of these fires on the security's own
# terminal price. Chosen to match the canonical listing-rule distress triggers and a
# collapse a premium acquisition never shows (see prototype in RESULTS.md).
DISTRESS_MAX_TERMINAL_PRICE = 1.0   # sub-$1 final print (Nasdaq/NYSE minimum-bid rule)
DISTRESS_MAX_DD_FROM_HIGH = -0.70   # >=70% below its own trailing-1y high at the end
DISTRESS_MAX_RET_1Y = -0.50         # >50% decline over the trailing year into the death

DEFAULT_PARAMS = dict(
    objective="binary",
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.02,
    min_child_samples=100,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=1.0,
    verbosity=-1,
    n_jobs=-1,
)


# --- label: which deaths are distress -------------------------------------------

def _terminal_stats(
    series: pd.Series, delist_date: pd.Timestamp, lookahead_days: int, window_days: int
) -> dict | None:
    """Terminal-price behaviour of one security around its ledger delist date.

    ``terminal`` is the last real print at or before ``delist_date + lookahead_days``
    (so we never read a post-death recovery, and a slightly-lagged Form 25 still lands
    on the true last trade). Drawdown and trailing return are measured over the
    ``window_days`` calendar window ending at that terminal print. Returns ``None`` when
    the security has no usable price near the event (the AMBIGUOUS case).
    """
    s = series.dropna()
    if s.empty:
        return None
    dd = pd.Timestamp(delist_date)
    pre = s[s.index <= dd + pd.Timedelta(days=lookahead_days)]
    if pre.empty:
        return None
    term_date, term_px = pre.index[-1], float(pre.iloc[-1])
    win = s[(s.index >= term_date - pd.Timedelta(days=window_days)) & (s.index <= term_date)]
    max_1y = float(win.max()) if len(win) else np.nan
    first_1y = float(win.iloc[0]) if len(win) else np.nan
    return {
        "terminal_price": term_px,
        "dd_from_high": term_px / max_1y - 1.0 if max_1y and max_1y > 0 else np.nan,
        "ret_1y": term_px / first_1y - 1.0 if first_1y and first_1y > 0 else np.nan,
    }


def classify_distress_events(
    delistings: pd.DataFrame,
    close: pd.DataFrame,
    ticker_map: dict,
    confirm_reasons: tuple[str, ...] = ("delist", "dereg"),
    max_terminal_price: float = DISTRESS_MAX_TERMINAL_PRICE,
    max_dd_from_high: float = DISTRESS_MAX_DD_FROM_HIGH,
    max_ret_1y: float = DISTRESS_MAX_RET_1Y,
    lookahead_days: int = 45,
    window_days: int = 365,
) -> pd.DataFrame:
    """Tag each ledger death ``is_distress`` in {True, False, None}.

    ``True``  — distress: reason not requiring confirmation, OR a price-confirmed collapse.
    ``False`` — benign: a price exists near the event but shows no collapse (M&A / premium).
    ``None``  — ambiguous: the reason requires confirmation but no terminal price is available.

    ``confirm_reasons`` lists reasons that must be price-confirmed; a reason left out is
    trusted as distress (e.g. ``confirm_reasons=("delist",)`` trusts every ``dereg``).
    The default confirms both, because ~37% of ``dereg`` events are premium going-private
    and ~79% of ``delist`` events are M&A — trusting reason alone floods the positives.
    Returns ``delistings`` plus columns ``is_distress, terminal_price, dd_from_high, ret_1y``.
    """
    out = delistings.copy()
    nan_stats = {"terminal_price": np.nan, "dd_from_high": np.nan, "ret_1y": np.nan}
    verdicts: list[bool | None] = []
    stat_rows = []
    for cik, dd, reason in zip(out["cik"], out["delist_date"], out["reason"]):
        if reason not in confirm_reasons:
            verdicts.append(True)
            stat_rows.append(nan_stats)
            continue
        col = ticker_map.get(int(cik))
        stats = _terminal_stats(close[col], dd, lookahead_days, window_days) \
            if col in close.columns else None
        if stats is None:
            verdicts.append(None)
            stat_rows.append(nan_stats)
            continue
        collapsed = (
            stats["terminal_price"] < max_terminal_price
            or (stats["dd_from_high"] < max_dd_from_high)
            or (stats["ret_1y"] < max_ret_1y)
        )
        verdicts.append(bool(collapsed))
        stat_rows.append(stats)
    out["is_distress"] = verdicts
    for c in ("terminal_price", "dd_from_high", "ret_1y"):
        out[c] = [s[c] for s in stat_rows]
    return out


# --- panel: PIT features + binary distress label --------------------------------

def build_distress_panel(
    features_df: pd.DataFrame,
    close: pd.DataFrame,
    events: pd.DataFrame,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    max_stale_days: int = MAX_STALE_DAYS,
    min_names: int = 30,
    censor_date=None,
) -> pd.DataFrame:
    """Assemble the point-in-time distress panel: one row per (cik, rebalance date).

    For every month-end ``d`` we snapshot the latest 10-K already public at ``d``
    (:func:`pit_snapshot`), keep only names ALIVE at ``d`` (ledger delist date after
    ``d`` or never), rank the fundamentals within sector over that full alive universe
    (:func:`add_sector_ranks` — the serve-parity transform), and attach the label:

    * ``y = 1`` if the name has a distress-confirmed death in ``(d, d + horizon_months]``
    * ``y = 0`` if it survives the window, or exits benignly (M&A / premium) in-window
    * row DROPPED if its only in-window death is ambiguous (unpriced) — never guessed

    ``events`` is the output of :func:`classify_distress_events` (the ledger's one
    earliest event per cik plus ``is_distress``). Rebalance dates whose forward window
    extends past ``censor_date`` (default: the latest ledger event) are skipped, so no
    label is right-censored into a false negative. Survivorship-correct: dead names carry
    a 10-K in ``features_df`` and are present until their delist date, so the positives
    are in the cross-section rather than silently absent.
    """
    ev = events.dropna(subset=["delist_date"]).copy()
    ev["cik"] = ev["cik"].astype(int)
    # one earliest event per cik (the ledger already guarantees this; enforce defensively)
    ev = ev.sort_values("delist_date").drop_duplicates("cik", keep="first")
    dmap = {
        c: (pd.Timestamp(dd), isd)
        for c, dd, isd in zip(ev["cik"], ev["delist_date"], ev["is_distress"])
    }
    if censor_date is None:
        censor_date = pd.Timestamp(ev["delist_date"].max())
    else:
        censor_date = pd.Timestamp(censor_date)

    feats = prepare_features(features_df)

    rows, coverage = [], []
    for d in month_end_dates(close.index):
        window_end = d + pd.DateOffset(months=horizon_months)
        if window_end > censor_date:
            continue  # forward window not fully observed -> would fabricate false negatives
        latest = pit_snapshot(feats, d, max_stale_days)
        if latest.empty:
            continue

        info = latest["cik"].map(lambda c: dmap.get(int(c), (pd.NaT, None)))
        latest = latest.copy()
        latest["delist_date"] = [x[0] for x in info]
        latest["is_distress"] = [x[1] for x in info]
        latest = latest[~(latest["delist_date"] <= d)]  # alive at d
        if latest.empty:
            continue

        # Ranks over the FULL known-at-date universe (serve path ranks the identical set).
        latest = add_sector_ranks(latest, MIN_SECTOR_BUCKET)

        dies = latest["delist_date"].notna() & (latest["delist_date"] <= window_end)
        distress = dies & (latest["is_distress"] == True)   # noqa: E712 (None-safe elementwise)
        ambiguous = dies & latest["is_distress"].isna()
        latest["y"] = distress.astype("int8")
        latest = latest[~ambiguous]                          # drop unpriced-death rows
        if len(latest) < min_names:
            continue
        latest["date"] = d
        rows.append(latest)
        coverage.append({
            "date": d,
            "universe": len(latest),
            "positives": int(latest["y"].sum()),
            "ambiguous_dropped": int(ambiguous.sum()),
            "base_rate": float(latest["y"].mean()),
        })

    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel.attrs["coverage"] = pd.DataFrame(coverage)
    panel.attrs["censor_date"] = censor_date
    panel.attrs["horizon_months"] = horizon_months
    return panel


def attach_distress_label(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    censor_date=None,
    drop_ambiguous: bool = True,
) -> pd.DataFrame:
    """Add the binary distress label ``y`` to an EXISTING panel keyed by (date, cik).

    Unlike :func:`build_distress_panel` (which builds its own PIT cross-sections and
    ranks), this labels a panel already assembled elsewhere — e.g. the return head's
    liquidity-filtered no-impute panel — so the distress overlay is scored on exactly
    the tradable rows the return backtest holds, with the identical rank basis. Same
    windowed rule: ``y=1`` for a distress-confirmed death in ``(date, date+N]``, ``y=0``
    for survivors and benign exits, ``NaN`` where the forward window runs past
    ``censor_date`` (so a possible-future death never trains as a false negative), and
    ambiguous (unpriced) in-window deaths dropped when ``drop_ambiguous``.
    """
    ev = events.dropna(subset=["delist_date"]).copy()
    ev["cik"] = ev["cik"].astype(int)
    ev = ev.sort_values("delist_date").drop_duplicates("cik", keep="first")
    dmap = {
        c: (pd.Timestamp(dd), isd)
        for c, dd, isd in zip(ev["cik"], ev["delist_date"], ev["is_distress"])
    }
    censor_date = pd.Timestamp(censor_date) if censor_date is not None \
        else pd.Timestamp(ev["delist_date"].max())

    out = panel.copy()
    info = out["cik"].astype(int).map(lambda c: dmap.get(c, (pd.NaT, None)))
    dd = pd.Series([x[0] for x in info], index=out.index)
    isd = pd.Series([x[1] for x in info], index=out.index)
    window_end = out["date"] + pd.DateOffset(months=horizon_months)
    dies = dd.notna() & (dd > out["date"]) & (dd <= window_end)
    y = (dies & (isd == True)).astype(float)      # noqa: E712 (None-safe elementwise)
    y[window_end > censor_date] = np.nan          # unobserved window -> label unknown
    out["y"] = y
    if drop_ambiguous:
        out = out[~(dies & isd.isna())]
    return out


# --- rare-event metrics ---------------------------------------------------------

def _decile_precision_recall(g: pd.DataFrame, prob: str, y: str, top_frac: float) -> tuple:
    """Precision & recall of the top-``top_frac`` predicted-risk slice of one date."""
    n_pos = int(g[y].sum())
    if len(g) < 10 or n_pos == 0:
        return (np.nan, np.nan)
    k = max(1, int(round(len(g) * top_frac)))
    top = g.nlargest(k, prob)
    tp = int(top[y].sum())
    return (tp / k, tp / n_pos)


def distress_metrics(
    pred: pd.DataFrame, prob: str = "prob", y: str = "y", date: str = "date",
    top_frac: float = 0.10, n_cal_bins: int = 10,
) -> dict:
    """Rare-event OOS scorecard from pooled predictions.

    * ``auc`` / ``pr_auc`` — pooled ROC-AUC and average precision over all OOS rows.
    * ``base_rate`` — pooled positive frequency.
    * ``precision_at_decile`` / ``recall_at_decile`` — per-date top-decile screen,
      averaged over dates, plus ``lift`` = precision / base rate.
    * ``calibration`` — pooled reliability table (mean predicted vs realized per prob bin)
      and ``calibration_mae`` (mean |predicted - realized| over populated bins).
    """
    p = pred.dropna(subset=[prob, y])
    if p.empty or p[y].nunique() < 2:
        return {"auc": float("nan"), "pr_auc": float("nan"), "base_rate": float(p[y].mean()) if len(p) else float("nan"), "n": len(p)}
    yv, pv = p[y].to_numpy(), p[prob].to_numpy()
    base = float(yv.mean())
    prec, rec = [], []
    for _, g in p.groupby(date):
        a, b = _decile_precision_recall(g, prob, y, top_frac)
        if not np.isnan(a):
            prec.append(a)
            rec.append(b)
    precision = float(np.mean(prec)) if prec else float("nan")

    # calibration: equal-count bins of predicted prob, pooled
    order = np.argsort(pv)
    bins = np.array_split(order, n_cal_bins)
    cal = [
        {"pred": float(pv[b].mean()), "realized": float(yv[b].mean()), "n": int(len(b))}
        for b in bins if len(b)
    ]
    cal_mae = float(np.mean([abs(c["pred"] - c["realized"]) for c in cal])) if cal else float("nan")

    return {
        "n": int(len(p)),
        "base_rate": base,
        "auc": float(roc_auc_score(yv, pv)),
        "pr_auc": float(average_precision_score(yv, pv)),
        "precision_at_decile": precision,
        "recall_at_decile": float(np.mean(rec)) if rec else float("nan"),
        "lift": precision / base if base > 0 and not np.isnan(precision) else float("nan"),
        "calibration": cal,
        "calibration_mae": cal_mae,
    }


# --- walk-forward + CPCV --------------------------------------------------------

def _scale_pos_weight(y: pd.Series, scale_pos_weight) -> float | None:
    if scale_pos_weight == "balanced":
        pos = int(y.sum())
        return float((len(y) - pos) / pos) if pos else None
    return scale_pos_weight


def walk_forward_predict_proba(
    panel: pd.DataFrame,
    feature_cols=None,
    label: str = "y",
    n_splits: int = 5,
    embargo: int = 2,
    horizon_periods: int = DEFAULT_HORIZON_MONTHS,
    params: dict | None = None,
    scale_pos_weight=None,
    id_cols: tuple = (),
) -> pd.DataFrame:
    """Expanding purged walk-forward; return pooled OOS distress probabilities.

    ``horizon_periods`` defaults to the label horizon in months (the distress label
    overlaps ``horizon_months`` rebalances), so train dates whose window overlaps the
    test block are purged. ``scale_pos_weight=None`` keeps the natural prior (so the
    predicted probabilities are directly calibratable); ``"balanced"`` sets neg/pos.
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
        if len(tr) < 50 or te.empty or tr[label].nunique() < 2:
            continue
        model = lgb.LGBMClassifier(**params, scale_pos_weight=_scale_pos_weight(tr[label], scale_pos_weight))
        model.fit(tr[feature_cols].fillna(0.5), tr[label])
        out = te[["date", *id_cols, label]].copy()
        out["prob"] = model.predict_proba(te[feature_cols].fillna(0.5))[:, 1]
        out.attrs = {}
        preds.append(out)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def evaluate_distress(panel: pd.DataFrame, **kwargs) -> dict | None:
    """Walk-forward predict, then the rare-event scorecard. ``None`` if no OOS preds."""
    metric_keys = ("top_frac", "n_cal_bins")
    metric_kwargs = {k: kwargs.pop(k) for k in metric_keys if k in kwargs}
    pred = walk_forward_predict_proba(panel, **kwargs)
    if pred.empty:
        return None
    summ = distress_metrics(pred, **metric_kwargs)
    summ["oos_dates"] = int(pred["date"].nunique())
    return summ


def cpcv_auc(
    panel: pd.DataFrame,
    feature_cols=None,
    label: str = "y",
    n_groups: int = 10,
    k_test: int = 2,
    embargo: int = 2,
    horizon_periods: int = DEFAULT_HORIZON_MONTHS,
    params: dict | None = None,
    scale_pos_weight=None,
) -> dict:
    """Combinatorial purged CV: an out-of-sample ROC-AUC per test combination.

    Returns the DISTRIBUTION of AUC (mean/std/min/quantiles + share above 0.5/0.7) —
    the honest read on whether the edge is a single-split fluke. Each combination
    purges ``horizon_periods + embargo`` rebalances around every test block on BOTH
    sides (the label looks forward, so leakage crosses boundaries both ways).
    """
    feature_cols = feature_cols or RANK_COLS
    params = {**DEFAULT_PARAMS, **(params or {})}
    dates = sorted(panel["date"].unique())
    splits = cpcv_splits(
        dates, n_groups=n_groups, k_test=k_test, embargo=embargo, horizon_periods=horizon_periods
    )
    aucs, prs = [], []
    for train_dates, test_dates in splits:
        tr = panel[panel["date"].isin(train_dates)].dropna(subset=[label])
        te = panel[panel["date"].isin(test_dates)].dropna(subset=[label])
        if len(tr) < 50 or te.empty or tr[label].nunique() < 2 or te[label].nunique() < 2:
            continue
        model = lgb.LGBMClassifier(**params, scale_pos_weight=_scale_pos_weight(tr[label], scale_pos_weight))
        model.fit(tr[feature_cols].fillna(0.5), tr[label])
        pr = model.predict_proba(te[feature_cols].fillna(0.5))[:, 1]
        aucs.append(float(roc_auc_score(te[label], pr)))
        prs.append(float(average_precision_score(te[label], pr)))
    a = np.asarray(aucs, dtype=float)
    if not len(a):
        return {"n_combos": 0, "mean_auc": float("nan")}
    return {
        "n_combos": int(len(a)),
        "mean_auc": float(a.mean()),
        "std_auc": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
        "min_auc": float(a.min()),
        "p05_auc": float(np.quantile(a, 0.05)),
        "median_auc": float(np.median(a)),
        "p95_auc": float(np.quantile(a, 0.95)),
        "frac_above_0p5": float((a > 0.5).mean()),
        "frac_above_0p7": float((a > 0.7).mean()),
        "mean_pr_auc": float(np.mean(prs)),
        "aucs": a.tolist(),
    }


# --- frozen product artifact (separate dir; never touches artifacts/model/) ------

def fit_distress(
    panel: pd.DataFrame,
    feature_cols=None,
    label: str = "y",
    params: dict | None = None,
    scale_pos_weight=None,
) -> lgb.LGBMClassifier:
    """One-shot fit on every labeled row (the artifact's training step)."""
    feature_cols = feature_cols or RANK_COLS
    params = {**DEFAULT_PARAMS, **(params or {})}
    tr = panel.dropna(subset=[label])
    model = lgb.LGBMClassifier(**params, scale_pos_weight=_scale_pos_weight(tr[label], scale_pos_weight))
    model.fit(tr[feature_cols].fillna(0.5), tr[label])
    return model


def save_distress_artifact(
    model: lgb.LGBMClassifier,
    panel: pd.DataFrame,
    out_dir: Path = DISTRESS_MODEL_DIR,
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
        "head": "distress",
        "feature_cols": feature_cols,
        "label": label,
        "horizon_months": int(panel.attrs.get("horizon_months", DEFAULT_HORIZON_MONTHS)),
        "trained_through": str(pd.Timestamp(labeled["date"].max()).date()),
        "censor_date": str(pd.Timestamp(panel.attrs["censor_date"]).date())
        if panel.attrs.get("censor_date") is not None else None,
        "n_rows": int(len(labeled)),
        "n_positives": int(labeled[label].sum()),
        "base_rate": float(labeled[label].mean()),
        "n_dates": int(labeled["date"].nunique()),
        "confirmation": {
            "max_terminal_price": DISTRESS_MAX_TERMINAL_PRICE,
            "max_dd_from_high": DISTRESS_MAX_DD_FROM_HIGH,
            "max_ret_1y": DISTRESS_MAX_RET_1Y,
        },
        "params": {k: v for k, v in model.get_params().items() if k in DEFAULT_PARAMS},
        "lightgbm_version": lgb.__version__,
        **(extra or {}),
    }
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


@dataclass
class DistressArtifact:
    """A frozen distress classifier. No fit method: serving cannot retrain."""

    booster: lgb.Booster
    meta: dict

    @property
    def feature_cols(self) -> list[str]:
        return list(self.meta["feature_cols"])

    @property
    def horizon_months(self) -> int:
        return int(self.meta["horizon_months"])

    @property
    def trained_through(self) -> pd.Timestamp:
        return pd.Timestamp(self.meta["trained_through"])

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """P(distress-delist within the horizon) for a cross-section. Binary-objective
        booster ``predict`` already returns the probability; columns are selected in the
        artifact's stored order."""
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"distress artifact expects feature columns missing from input: {missing}")
        return self.booster.predict(X[self.feature_cols].fillna(0.5).to_numpy())


def load_distress_artifact(path: Path = DISTRESS_MODEL_DIR) -> DistressArtifact:
    """Load the frozen distress artifact. Raises if none has been trained."""
    path = Path(path)
    model_file = path / "model.txt"
    if not model_file.exists():
        raise FileNotFoundError(
            f"no trained distress artifact at {path}; run scripts/run_distress.py --save first"
        )
    booster = lgb.Booster(model_file=str(model_file))
    meta = json.loads((path / "meta.json").read_text())
    saved = meta.get("lightgbm_version")
    if saved and saved != lgb.__version__:
        import warnings

        warnings.warn(
            f"distress artifact was frozen with lightgbm {saved} but runtime is "
            f"{lgb.__version__}; scores may drift — retrain or pin the dependency",
            stacklevel=2,
        )
    return DistressArtifact(booster=booster, meta=meta)


def load_distress_artifact_optional(path: Path = DISTRESS_MODEL_DIR) -> DistressArtifact | None:
    """The distress head as an OPTIONAL risk-flag layer: return the artifact if one has
    been frozen, else ``None`` — the serve/monitor/TUI paths run unchanged without it."""
    try:
        return load_distress_artifact(path)
    except FileNotFoundError:
        return None


def distress_flag(prob: float, thresholds=DISTRESS_FLAG_THRESHOLDS) -> str:
    """Map a P(distress) to a display flag: ``high`` | ``elevated`` | ``normal``."""
    if prob is None or not np.isfinite(prob):
        return "normal"
    for name, thr in thresholds:
        if prob >= thr:
            return name
    return "normal"
