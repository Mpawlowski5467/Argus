"""Distress head: PIT-correct windowed label, price-confirmed positives, honest OOS ranking.

The label is where the leakage risk lives, so most of these pin the label: a death is
positive at date T only if it lands in (T, T+N] AND the fundamentals were public at T; a
benign acquisition in-window is a real negative; an unpriced death is dropped, never
guessed. The rest mirror tests/test_model.py — learns a signal OOS, finds nothing in
noise, and the frozen artifact scores identically and cannot retrain.
"""

import numpy as np
import pandas as pd
import pytest

from stockscan.distress import (
    RANK_COLS,
    attach_distress_label,
    build_distress_panel,
    classify_distress_events,
    distress_flag,
    distress_metrics,
    fit_distress,
    load_distress_artifact,
    load_distress_artifact_optional,
    save_distress_artifact,
    walk_forward_predict_proba,
)
from stockscan.features import FEATURES


def _features(ciks, filed="2019-06-01", sic=3571):
    rows = []
    for i, cik in enumerate(ciks):
        r = {"cik": cik, "filed_date": pd.Timestamp(filed), "sic": sic}
        r.update({f: 0.1 * (i + 1) for f in FEATURES})
        rows.append(r)
    return pd.DataFrame(rows)


def _grid(start="2019-01-01", end="2021-06-30", cols=("AAA",)):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({c: 1.0 for c in cols}, index=idx)  # values unused by the panel builder


def _y_at(panel, date, cik):
    sub = panel[(panel["date"] == pd.Timestamp(date)) & (panel["cik"] == cik)]
    return None if sub.empty else int(sub["y"].iloc[0])


def _flag(v):
    """Normalize an is_distress cell (python bool / numpy bool / None / NaN) to {True, False, None}."""
    if v is None or (not isinstance(v, (bool, np.bool_)) and pd.isna(v)):
        return None
    return bool(v)


# --- label correctness / no-leakage --------------------------------------------

def test_label_is_windowed_pit_and_survivorship_correct():
    """A distress death is positive only inside (T, T+N]; a benign exit is a negative;
    a name already dead at T is absent (it was in the panel while alive)."""
    events = pd.DataFrame({
        "cik": [2, 3],
        "delist_date": pd.to_datetime(["2020-03-15", "2020-03-15"]),
        "reason": ["dereg", "delist"],
        "is_distress": [True, False],  # cik2 distress, cik3 benign M&A
    })
    panel = build_distress_panel(
        _features([1, 2, 3]), _grid(), events,
        horizon_months=3, min_names=1, censor_date="2021-06-30",
    )

    # 2019-12-31: the 2020-03-15 death is INSIDE the 3-month window
    assert _y_at(panel, "2019-12-31", 2) == 1   # distress death -> positive
    assert _y_at(panel, "2019-12-31", 3) == 0   # benign M&A -> real negative, still present
    assert _y_at(panel, "2019-12-31", 1) == 0   # survivor

    # 2019-11-29: window ends 2020-02-29, BEFORE the death -> no look-ahead into the future
    assert _y_at(panel, "2019-11-29", 2) == 0

    # once delisted, the name leaves the cross-section (alive-at-T filter)
    assert _y_at(panel, "2020-06-30", 2) is None


def test_ambiguous_unpriced_death_is_dropped_not_guessed():
    """A death we cannot price is neither a clean positive nor a clean negative: the row
    is dropped inside the window, but the name is a normal negative before the window."""
    events = pd.DataFrame({
        "cik": [2], "delist_date": [pd.Timestamp("2020-03-15")],
        "reason": ["delist"], "is_distress": [None],  # unpriced -> ambiguous
    })
    panel = build_distress_panel(
        _features([1, 2]), _grid(), events,
        horizon_months=3, min_names=1, censor_date="2021-06-30",
    )
    # inside the window the ambiguous name is dropped (not labeled 0 or 1)
    assert _y_at(panel, "2019-12-31", 2) is None
    assert _y_at(panel, "2019-12-31", 1) == 0
    # before the window it is a plain negative, present in the panel
    assert _y_at(panel, "2019-06-28", 2) == 0


def test_features_must_be_public_before_the_asof_date():
    """A filing not yet public at T cannot appear at T (PIT via pit_snapshot/assert_pit)."""
    events = pd.DataFrame(columns=["cik", "delist_date", "reason", "is_distress"])
    feats = _features([1], filed="2020-01-15")  # public ~2020-01-16
    panel = build_distress_panel(feats, _grid(), events, horizon_months=3, min_names=1,
                                 censor_date="2021-06-30")
    assert _y_at(panel, "2019-12-31", 1) is None  # not yet filed -> absent
    assert _y_at(panel, "2020-02-28", 1) == 0     # public by then -> present


def test_censoring_drops_dates_whose_window_is_unobserved():
    """No date may be kept whose forward window runs past the ledger observation edge —
    that would relabel a possible future death as a false negative."""
    events = pd.DataFrame({
        "cik": [2], "delist_date": [pd.Timestamp("2020-06-15")],
        "reason": ["dereg"], "is_distress": [True],
    })
    panel = build_distress_panel(
        _features([1, 2]), _grid(end="2020-12-31"), events,
        horizon_months=6, censor_date="2020-06-30", min_names=1,
    )
    # last admissible date has window_end <= 2020-06-30, i.e. d <= 2019-12-31
    assert panel["date"].max() <= pd.Timestamp("2019-12-31")


# --- attach_distress_label (overlay path): same windowed rule on an existing panel ---

def _y(out, date, cik):
    r = out[(out["date"] == pd.Timestamp(date)) & (out["cik"] == cik)]
    return None if r.empty else r["y"].iloc[0]


def test_attach_distress_label_applies_the_windowed_rule():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-31", "2019-11-29", "2019-12-31"]),
        "cik": [2, 2, 1], "ticker": ["BBB", "BBB", "AAA"],
    })
    events = pd.DataFrame({"cik": [2], "delist_date": [pd.Timestamp("2020-03-15")],
                           "reason": ["dereg"], "is_distress": [True]})
    out = attach_distress_label(panel, events, horizon_months=3, censor_date="2021-06-30")
    assert _y(out, "2019-12-31", 2) == 1.0   # distress death inside the 3-month window
    assert _y(out, "2019-11-29", 2) == 0.0   # window ends 2020-02-29, before the death
    assert _y(out, "2019-12-31", 1) == 0.0   # survivor


def test_attach_distress_label_drops_ambiguous_and_censors_unobserved():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-31", "2025-12-31"]),
        "cik": [2, 1], "ticker": ["BBB", "AAA"],
    })
    events = pd.DataFrame({"cik": [2], "delist_date": [pd.Timestamp("2020-03-15")],
                           "reason": ["delist"], "is_distress": [None]})
    out = attach_distress_label(panel, events, horizon_months=3, censor_date="2021-06-30")
    assert _y(out, "2019-12-31", 2) is None       # unpriced death in-window -> row dropped
    assert pd.isna(_y(out, "2025-12-31", 1))      # window past the ledger edge -> label unknown


# --- price confirmation: distress vs benign M&A --------------------------------

def test_price_confirmation_separates_collapse_from_premium_exit():
    idx = pd.bdate_range("2019-07-01", "2020-06-30")
    n = len(idx)
    close = pd.DataFrame(index=idx)
    close["A~1"] = np.linspace(20.0, 0.5, n)     # sub-$1 terminal -> distress
    close["E~5"] = np.concatenate([np.linspace(30, 100, n // 2),
                                   np.linspace(100, 20, n - n // 2)])  # -80% from high -> distress
    close["B~2"] = np.linspace(45.0, 50.0, n)    # acquired at a premium -> benign
    close["C~3"] = np.linspace(80.0, 64.0, n)    # mild -20% loser -> not distress

    dl = pd.DataFrame({
        "cik": [1, 5, 2, 3, 9],
        "delist_date": pd.to_datetime(["2020-06-15"] * 5),
        "reason": ["delist"] * 5,
    })
    tmap = {1: "A~1", 5: "E~5", 2: "B~2", 3: "C~3", 9: "GONE"}  # cik9 unmapped
    g = classify_distress_events(dl, close, tmap).set_index("cik")["is_distress"]

    assert _flag(g[1]) is True    # sub-$1 print
    assert _flag(g[5]) is True    # >=70% drawdown from its own 1y high
    assert _flag(g[2]) is False   # premium acquisition
    assert _flag(g[3]) is False   # mild loser, not distress
    assert _flag(g[9]) is None    # no price -> ambiguous, never guessed


def test_reason_bypass_trusts_unconfirmed_reason():
    """confirm_reasons controls which reasons need price proof; a reason left out is
    trusted as distress even with no price."""
    dl = pd.DataFrame({"cik": [7], "delist_date": [pd.Timestamp("2020-06-15")],
                       "reason": ["dereg"]})
    g = classify_distress_events(dl, pd.DataFrame(), {}, confirm_reasons=("delist",))
    assert _flag(g["is_distress"].iloc[0]) is True


# --- OOS ranking: learns a real signal, nothing in noise -----------------------

def _synth(rng, signal: bool, n_dates=40, n=300, base=0.03):
    frames = []
    for i in range(n_dates):
        d = pd.Timestamp("2015-01-31") + pd.offsets.MonthEnd(i)
        df = pd.DataFrame({c: rng.uniform(0, 1, n) for c in RANK_COLS})
        df["date"] = d
        if signal:  # high leverage + low liquidity -> higher distress hazard
            logit = -3.5 + 3.0 * (df["leverage_rank"] - 0.5) - 3.0 * (df["current_ratio_rank"] - 0.5)
            p = 1.0 / (1.0 + np.exp(-logit))
        else:
            p = np.full(n, base)
        df["y"] = (rng.uniform(0, 1, n) < p).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_classifier_ranks_distress_out_of_sample():
    panel = _synth(np.random.default_rng(0), signal=True)
    m = distress_metrics(walk_forward_predict_proba(panel, n_splits=4, embargo=1, horizon_periods=1))
    assert m["auc"] > 0.65          # ranks the future failures
    assert m["lift"] > 1.5          # top decile enriched vs base rate


def test_classifier_finds_nothing_in_noise():
    panel = _synth(np.random.default_rng(1), signal=False)
    m = distress_metrics(walk_forward_predict_proba(panel, n_splits=4, embargo=1, horizon_periods=1))
    assert abs(m["auc"] - 0.5) < 0.06   # no real edge -> coin-flip ranking


# --- frozen artifact: parity, no retrain, refuses drift ------------------------

def _artifact_panel(rng):
    panel = _synth(rng, signal=True)
    panel.attrs["censor_date"] = panel["date"].max()
    panel.attrs["horizon_months"] = 12
    return panel


def test_distress_artifact_roundtrip_scores_identically_and_cannot_retrain(tmp_path):
    panel = _artifact_panel(np.random.default_rng(2))
    model = fit_distress(panel, params=dict(n_estimators=40, min_child_samples=20))
    out = save_distress_artifact(model, panel, out_dir=tmp_path, extra={"mode": "test"})
    assert (out / "model.txt").exists() and (out / "meta.json").exists()

    art = load_distress_artifact(tmp_path)
    assert art.feature_cols == RANK_COLS
    assert art.horizon_months == 12
    assert art.meta["head"] == "distress" and art.meta["mode"] == "test"
    assert art.trained_through == panel["date"].max()
    assert 0.0 < art.meta["base_rate"] < 1.0 and art.meta["n_positives"] > 0

    X = panel[RANK_COLS].head(200)
    # binary-objective booster.predict already returns P(distress) == predict_proba[:,1]
    np.testing.assert_allclose(art.score(X), model.predict_proba(X.fillna(0.5))[:, 1],
                               rtol=1e-6, atol=1e-9)
    assert np.all((art.score(X) >= 0) & (art.score(X) <= 1))  # a probability, not a score
    assert not hasattr(art, "fit")  # frozen: no retrain path on the serve side


def test_distress_artifact_refuses_wrong_feature_columns(tmp_path):
    panel = _artifact_panel(np.random.default_rng(3))
    save_distress_artifact(fit_distress(panel, params=dict(n_estimators=5)), panel, out_dir=tmp_path)
    art = load_distress_artifact(tmp_path)
    with pytest.raises(KeyError):
        art.score(panel[RANK_COLS[:-1]])  # a missing feature must fail loudly


# --- display/alert layer: flag thresholds + optional loader --------------------

def test_distress_flag_levels():
    assert distress_flag(0.0) == "normal"
    assert distress_flag(0.029) == "normal"
    assert distress_flag(0.03) == "elevated"
    assert distress_flag(0.079) == "elevated"
    assert distress_flag(0.08) == "high"
    assert distress_flag(0.5) == "high"
    assert distress_flag(float("nan")) == "normal"   # never flags on a bad score


def test_optional_loader_returns_none_when_absent(tmp_path):
    assert load_distress_artifact_optional(tmp_path / "nope") is None
    panel = _artifact_panel(np.random.default_rng(4))
    save_distress_artifact(fit_distress(panel, params=dict(n_estimators=5)), panel, out_dir=tmp_path / "d")
    assert load_distress_artifact_optional(tmp_path / "d") is not None
