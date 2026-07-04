"""Large-drawdown head: PIT-correct price-path label, censoring, honest artifact.

The label is a forward price-path read, so these pin its correctness: a peak-to-trough
crash is a positive, a riser/flat name is a negative, a name we can't price forward is
dropped (not guessed), and a rebalance date whose window runs past the price edge is
censored. Plus the frozen-artifact contract (scores identically, cannot retrain) and the
display flag.
"""

import numpy as np
import pandas as pd

from stockscan.drawdown import (
    RANK_COLS,
    build_drawdown_panel,
    drawdown_flag,
    fit_drawdown,
    forward_max_drawdown,
    load_drawdown_artifact,
    load_drawdown_artifact_optional,
    save_drawdown_artifact,
)
from stockscan.features import FEATURES

TMAP = {1: "RISE", 2: "CRASH", 3: "FLAT", 4: "GHOST"}


def _features(ciks, filed="2019-12-01", sic=3571):
    rows = []
    for i, cik in enumerate(ciks):
        r = {"cik": cik, "filed_date": pd.Timestamp(filed), "sic": sic}
        r.update({f: 0.1 * (i + 1) for f in FEATURES})
        rows.append(r)
    return pd.DataFrame(rows)


def _prices():
    """RISE monotonic up; CRASH flat then a −55% slide Jun→Sep; FLAT constant; GHOST unpriced."""
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    close = pd.DataFrame(index=idx)
    close["RISE"] = np.linspace(100, 140, len(idx))
    close["FLAT"] = 100.0
    lo, hi = pd.Timestamp("2020-06-01"), pd.Timestamp("2020-09-01")
    span = (hi - lo).days
    p = np.full(len(idx), 100.0)
    for i, dt in enumerate(idx):
        if dt >= hi:
            p[i] = 45.0
        elif dt >= lo:
            p[i] = 100.0 - 55.0 * (dt - lo).days / span
    close["CRASH"] = p
    close["GHOST"] = np.nan
    return close


def _y(panel, cik, date):
    sub = panel[(panel["cik"] == cik) & (panel["date"] == pd.Timestamp(date))]
    return None if sub.empty else int(sub["y"].iloc[0])


# --- the label -------------------------------------------------------------------

def test_forward_max_drawdown_is_peak_to_trough():
    close = _prices()
    d = pd.Timestamp("2020-05-29")
    mdd = forward_max_drawdown(close, ["RISE", "CRASH", "FLAT"], d, d + pd.DateOffset(months=3))
    assert mdd["RISE"] > -0.02 and abs(mdd["FLAT"]) < 1e-9
    assert mdd["CRASH"] <= -0.30          # ~-53% peak-to-trough within the window


def test_label_flags_the_crash_not_the_riser():
    panel = build_drawdown_panel(_features([1, 2, 3]), _prices(), TMAP,
                                 horizon_months=3, threshold=-0.30, min_names=1)
    assert _y(panel, 2, "2020-05-29") == 1     # CRASH
    assert _y(panel, 1, "2020-05-29") == 0     # RISE
    assert _y(panel, 3, "2020-05-29") == 0     # FLAT


def test_unpriced_name_is_dropped_not_guessed():
    panel = build_drawdown_panel(_features([1, 2, 3, 4]), _prices(), TMAP,
                                 horizon_months=3, threshold=-0.30, min_names=1)
    assert _y(panel, 4, "2020-05-29") is None  # GHOST has no forward price -> unlabelable


def test_censoring_drops_dates_whose_window_is_unobserved():
    # last price date is 2020-12-31; a 3-month window past 2020-09-30 runs off the edge.
    panel = build_drawdown_panel(_features([1, 2, 3]), _prices(), TMAP,
                                 horizon_months=3, threshold=-0.30, min_names=1)
    assert panel["date"].max() <= pd.Timestamp("2020-09-30")


def test_threshold_is_respected():
    strict = build_drawdown_panel(_features([1, 2, 3]), _prices(), TMAP,
                                  horizon_months=3, threshold=-0.60, min_names=1)
    assert _y(strict, 2, "2020-05-29") == 0    # the ~-53% slide doesn't clear a -60% bar


# --- frozen artifact contract ----------------------------------------------------

def test_artifact_roundtrip_scores_identically_and_cannot_retrain(tmp_path):
    panel = build_drawdown_panel(_features([1, 2, 3]), _prices(), TMAP,
                                 horizon_months=3, threshold=-0.30, min_names=1)
    assert panel["y"].nunique() == 2
    mdl = fit_drawdown(panel)
    art = load_drawdown_artifact(save_drawdown_artifact(mdl, panel, out_dir=tmp_path / "dd"))
    np.testing.assert_allclose(
        art.score(panel),
        mdl.predict_proba(panel[RANK_COLS].fillna(0.5))[:, 1],
        rtol=1e-6, atol=1e-9,
    )
    assert not hasattr(art, "fit")
    assert art.horizon_months == 3 and art.threshold == -0.30


def test_optional_loader_returns_none_when_absent(tmp_path):
    assert load_drawdown_artifact_optional(tmp_path / "nope") is None


def test_drawdown_flag_levels():
    assert drawdown_flag(0.7) == "high"
    assert drawdown_flag(0.45) == "elevated"
    assert drawdown_flag(0.1) == "normal"
    assert drawdown_flag(float("nan")) == "normal"
