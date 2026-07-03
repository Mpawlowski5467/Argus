"""Paper-forward: vintage discipline, append-only signals, honest compare().

The freeze/vintage tests run on fabricated artifact dirs (content-hash logic
only); log_signals runs against the same synthetic world the serve tests use
(real cross-section, real artifact); compare() runs on hand-written signal
files so every realized return is controlled.
"""

import json

import numpy as np
import pandas as pd
import pytest

from stockscan.features import FEATURES
from stockscan.fundamental_panel import build_fundamental_panel, prepare_features
from stockscan.model import fit, load_artifact, save_artifact
from stockscan.ops.paper import (
    _book_transitions,
    artifact_fingerprint,
    compare,
    current_vintage,
    default_as_of,
    freeze_baseline,
    log_signals,
    record_retrain,
)
from stockscan.ops.state import OpsState
from stockscan.serve import ServeData

MIN_DV = 1_000_000
HORIZON = 21


# --- fingerprint / freeze / vintages (fabricated artifacts) -------------------------

def _fake_model_dir(tmp_path, name, content=b"booster-bytes"):
    d = tmp_path / name
    d.mkdir()
    (d / "model.txt").write_bytes(content)
    (d / "meta.json").write_text(json.dumps({
        "trained_through": "2026-03-31", "feature_cols": ["roa_rank"],
        "lightgbm_version": "4.6.0", "n_rows": 1, "label_horizon_days": 63,
    }))
    return d


def test_fingerprint_tracks_content(tmp_path):
    d1 = _fake_model_dir(tmp_path, "m1")
    d2 = _fake_model_dir(tmp_path, "m2")
    d3 = _fake_model_dir(tmp_path, "m3", content=b"different")
    assert artifact_fingerprint(d1) == artifact_fingerprint(d2)
    assert artifact_fingerprint(d1) != artifact_fingerprint(d3)


def test_freeze_is_write_once(tmp_path):
    model_dir = _fake_model_dir(tmp_path, "model")
    paper_dir = tmp_path / "paper"
    first = freeze_baseline(model_dir, paper_dir)
    assert first["status"] == "frozen"
    assert current_vintage(paper_dir)["hash"] == first["artifact"]["hash"]
    assert first["known_asymmetries"], "the live-vs-backtest asymmetries must be frozen in"
    again = freeze_baseline(model_dir, paper_dir)
    assert again["status"] == "noop"
    (model_dir / "model.txt").write_bytes(b"retrained!")  # silent retrain
    with pytest.raises(RuntimeError, match="retrain-record"):
        freeze_baseline(model_dir, paper_dir)


def test_record_retrain_flow(tmp_path):
    model_dir = _fake_model_dir(tmp_path, "model")
    paper_dir = tmp_path / "paper"
    with pytest.raises(RuntimeError, match="baseline"):
        record_retrain("too early", model_dir, paper_dir)
    freeze_baseline(model_dir, paper_dir)
    with pytest.raises(RuntimeError, match="already the registered vintage"):
        record_retrain("nothing changed", model_dir, paper_dir)
    with pytest.raises(ValueError, match="reason"):
        record_retrain("   ", model_dir, paper_dir)
    old_hash = current_vintage(paper_dir)["hash"]
    (model_dir / "model.txt").write_bytes(b"quarterly retrain")
    entry = record_retrain("quarterly retrain 2026q3", model_dir, paper_dir)
    assert entry["previous_hash"] == old_hash
    assert current_vintage(paper_dir)["hash"] == entry["hash"] != old_hash


def test_default_as_of_is_last_completed_month_end():
    idx = pd.bdate_range("2026-01-01", "2026-07-15")
    assert default_as_of(idx, today="2026-07-15").date() == pd.Timestamp("2026-06-30").date()
    assert default_as_of(idx, today="2026-07-01").date() == pd.Timestamp("2026-06-30").date()
    # mid-month run must NOT pick a mid-month date off the monthly grid
    assert default_as_of(idx, today="2026-06-20").date() == pd.Timestamp("2026-05-29").date()
    with pytest.raises(ValueError):
        default_as_of(pd.bdate_range("2026-07-01", "2026-07-10"), today="2026-07-15")


def test_book_transitions_hysteresis():
    cross = pd.DataFrame({
        "cik": [1, 2, 3, 4], "pct": [0.95, 0.70, 0.50, 0.85],
        "column": ["A", "B", "C", "D"],
    })
    # empty book: only the top-20% tail enters
    enters, exits = _book_transitions({}, cross)
    assert set(enters) == {1, 4} and exits == set()
    # holder at 0.70 stays (inside top 40%); holder at 0.50 exits; absent holder exits
    book = {1: {}, 2: {}, 3: {}, 9: {}}
    enters, exits = _book_transitions(book, cross)
    assert set(enters) == {4}
    assert exits == {3, 9}


# --- log_signals on the synthetic world ----------------------------------------------

def _raw_features():
    rng = np.random.default_rng(7)
    rows = []
    for cik in range(1, 46):
        for fy, filed in ((2022, "2023-03-01"), (2023, "2024-03-01")):
            r = {"cik": cik, "name": f"CO{cik}", "fy": fy,
                 "sic": 3571 if cik <= 30 else 6021,
                 "filed_date": pd.Timestamp(filed),
                 "period_end": pd.Timestamp(f"{fy}-12-31")}
            r.update({f: rng.uniform(0.05, 0.9) for f in FEATURES})
            rows.append(r)
    return pd.DataFrame(rows)


def _prices():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2023-06-01", "2024-12-31")
    close = pd.DataFrame(index=idx)
    for cik in range(1, 45):
        base = rng.uniform(20, 150)
        close[f"T{cik}"] = base * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    close["T45"] = 30.0
    dv = close * 1_000_000
    return close, dv


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    raw = _raw_features()
    close, dv = _prices()
    tmap = {cik: f"T{cik}" for cik in range(1, 46)}
    panel = build_fundamental_panel(
        raw, close, delistings=None, ticker_map=tmap, dollar_volume=dv,
        min_dollar_volume=MIN_DV, horizon=HORIZON, min_names=5)
    model = fit(panel, params=dict(n_estimators=20, min_child_samples=10))
    model_dir = tmp_path_factory.mktemp("artifact")
    save_artifact(model, panel, out_dir=model_dir)
    data = ServeData(feats=prepare_features(raw), close=close,
                     dv_med=dv.rolling(20, min_periods=10).median(), ticker_map=tmap)
    return {"data": data, "artifact": load_artifact(model_dir), "model_dir": model_dir}


def test_log_signals_end_to_end(world, tmp_path):
    paper_dir = tmp_path / "paper"
    freeze_baseline(world["model_dir"], paper_dir)
    with OpsState(tmp_path / "state.sqlite") as state:
        res = log_signals(state, data=world["data"], artifact=world["artifact"],
                          as_of="2024-09-30", paper_dir=paper_dir,
                          model_dir=world["model_dir"])
        assert res["status"] == "logged"
        assert res["n"] >= 30 and res["n_book"] > 0
        path = paper_dir / "signals" / "2024-09-30.jsonl"
        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        assert header["artifact_hash"] == artifact_fingerprint(world["model_dir"])
        assert "in_sample" in header and "data_vintage" in header
        assert header["thresholds"]["hysteresis_enter"] == 0.20
        rows = [json.loads(ln) for ln in lines[1:]]
        in_book = {r["cik"] for r in rows if r["in_book"]}
        assert in_book == set(state.book())
        top = [r for r in rows if r["top_decile"]]
        assert all(r["decile"] == 10 for r in top)

        # idempotent: same month is a no-op, file byte-identical
        before = path.read_bytes()
        res2 = log_signals(state, data=world["data"], artifact=world["artifact"],
                           as_of="2024-09-30", paper_dir=paper_dir,
                           model_dir=world["model_dir"])
        assert res2["status"] == "noop"
        assert path.read_bytes() == before

        # crash-heal: wipe the sqlite book; the no-op path restores it from the file
        state.book_apply({}, set(state.book()), "wipe")
        assert state.book() == {}
        log_signals(state, data=world["data"], artifact=world["artifact"],
                    as_of="2024-09-30", paper_dir=paper_dir,
                    model_dir=world["model_dir"])
        assert set(state.book()) == in_book


def test_log_signals_refuses_unregistered_artifact(world, tmp_path):
    paper_dir = tmp_path / "paper"
    freeze_baseline(world["model_dir"], paper_dir)
    (world["model_dir"] / "model.txt").write_bytes(
        (world["model_dir"] / "model.txt").read_bytes() + b"\n# drifted")
    try:
        with OpsState(tmp_path / "s.sqlite") as state:
            with pytest.raises(RuntimeError, match="registered vintage"):
                log_signals(state, data=world["data"], artifact=world["artifact"],
                            as_of="2024-09-30", paper_dir=paper_dir,
                            model_dir=world["model_dir"])
    finally:  # restore for other tests in the module
        raw = (world["model_dir"] / "model.txt").read_bytes()
        (world["model_dir"] / "model.txt").write_bytes(raw.replace(b"\n# drifted", b""))


def test_missing_paper_months_backfill_list(world, tmp_path):
    """A multi-month outage must leave no permanent hole: every completed month
    since the freeze with no file is listed, oldest-first."""
    from stockscan.ops.paper import missing_paper_months

    paper_dir = tmp_path / "paper"
    freeze_baseline(world["model_dir"], paper_dir)
    idx = world["data"].close.index  # 2023-06 .. 2024-12
    # freeze stamped "now"; override the baseline's frozen_on to the panel's start
    import json as _json
    bp = paper_dir / "baseline.json"
    b = _json.loads(bp.read_text())
    b["frozen_on"] = "2024-06-01T00:00:00+00:00"
    bp.write_text(_json.dumps(b))
    with OpsState(tmp_path / "s.sqlite") as state:
        # log only 2024-09-30, leaving 07/08/10/11 as holes
        log_signals(state, data=world["data"], artifact=world["artifact"],
                    as_of="2024-09-30", paper_dir=paper_dir, model_dir=world["model_dir"])
    missing = missing_paper_months(idx, paper_dir, today="2024-12-15")
    labels = [str(m.date()) for m in missing]
    assert "2024-09-30" not in labels           # already logged
    assert labels == sorted(labels)             # oldest first
    assert any(m.month == 7 for m in missing) and any(m.month == 11 for m in missing)
    assert all(m >= pd.Timestamp("2024-06-01") for m in missing)  # not before the freeze


def test_book_carries_across_months(world, tmp_path):
    paper_dir = tmp_path / "paper"
    freeze_baseline(world["model_dir"], paper_dir)
    with OpsState(tmp_path / "state.sqlite") as state:
        r1 = log_signals(state, data=world["data"], artifact=world["artifact"],
                         as_of="2024-08-30", paper_dir=paper_dir,
                         model_dir=world["model_dir"])
        r2 = log_signals(state, data=world["data"], artifact=world["artifact"],
                         as_of="2024-09-30", paper_dir=paper_dir,
                         model_dir=world["model_dir"])
        # month 2 must see month 1's book: with a stable synthetic cross-section
        # most holders stay, so entries in month 2 are far fewer than the book
        assert r2["n_entered"] < r1["n_entered"]
        assert r2["n_book"] > 0
        month2_book = set(state.book())

        # re-logging the OLDER month (month 1) after month 2 exists must NOT regress
        # the book to the month-1 state — the noop path reconciles to the LATEST file
        log_signals(state, data=world["data"], artifact=world["artifact"],
                    as_of="2024-08-30", paper_dir=paper_dir,
                    model_dir=world["model_dir"])
        assert set(state.book()) == month2_book


# --- compare() on hand-written files ---------------------------------------------------

def _write_baseline(paper_dir, min_months=1):
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "signals").mkdir(exist_ok=True)
    (paper_dir / "baseline.json").write_text(json.dumps({
        "frozen_on": "2026-01-01T00:00:00+00:00",
        "backtest_expectation": {"oos_rank_ic": 0.04, "decile_spread_63d": 0.02},
        "degradation_rule": {"live_ic_frac": 0.5, "min_months": min_months},
    }))


def _write_signals(paper_dir, as_of, scores, in_sample=False, hash_="h1"):
    n = len(scores)
    rows = []
    for i, s in enumerate(scores, start=1):
        pct = i / n
        decile = int(np.clip(np.ceil(pct * 10), 1, 10))
        rows.append({"cik": i, "column": f"T{i}", "sector": "Manufacturing",
                     "score": s, "pct": pct, "decile": decile,
                     "top_decile": decile == 10, "in_book": pct >= 0.8,
                     "entered": False, "exited": False,
                     "filed_date": "2025-03-01", "available_date": "2025-03-02"})
    path = paper_dir / "signals" / f"{as_of}.jsonl"
    with open(path, "w") as fh:
        fh.write(json.dumps({"as_of": as_of, "artifact_hash": hash_,
                             "in_sample": in_sample, "stats": {"n": n}}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _close_matrix(n=40, as_of="2026-01-30", horizon=5):
    """Flat at 10 before as_of; name i then drifts to 10*(1+i/100) by as_of+h.
    Score in the signals file == i, so score ranks == realized-return ranks."""
    idx = pd.bdate_range("2025-12-01", periods=60)
    as_of = pd.Timestamp(as_of)
    loc = idx.get_loc(as_of)
    close = pd.DataFrame(10.0, index=idx, columns=[f"T{i}" for i in range(1, n + 1)])
    for i in range(1, n + 1):
        target = 10.0 * (1 + i / 100)
        ramp = np.linspace(10.0, target, len(idx) - loc)
        close.iloc[loc:, i - 1] = ramp
    return close


def test_compare_recovers_perfect_ic(tmp_path):
    paper_dir = tmp_path / "paper"
    _write_baseline(paper_dir)
    _write_signals(paper_dir, "2026-01-30", scores=list(range(1, 41)))
    close = _close_matrix()
    tmap = {i: f"T{i}" for i in range(1, 41)}
    rep = compare(close=close, paper_dir=paper_dir, ticker_map=tmap, horizons=(5,))
    m = rep["months"][0]["h5"]
    assert m["n_priced"] == 40
    assert m["rank_ic"] > 0.95
    assert m["decile_spread"] > 0
    assert rep["degraded"] is False
    assert rep["live_mean_ic"] > 0.95


def test_compare_flags_degradation(tmp_path):
    paper_dir = tmp_path / "paper"
    _write_baseline(paper_dir)
    # scores REVERSED vs realized returns -> live IC ~ -1
    _write_signals(paper_dir, "2026-01-30", scores=list(range(40, 0, -1)))
    close = _close_matrix()
    tmap = {i: f"T{i}" for i in range(1, 41)}
    rep = compare(close=close, paper_dir=paper_dir, ticker_map=tmap, horizons=(5,))
    assert rep["months"][0]["h5"]["rank_ic"] < -0.95
    assert rep["degraded"] is True


def test_compare_resolves_renamed_dead_column(tmp_path):
    """The logged column T40 died and was renamed T40~40 — the CURRENT map must
    find its terminal prices; the crashed names must not silently drop out."""
    paper_dir = tmp_path / "paper"
    _write_baseline(paper_dir)
    _write_signals(paper_dir, "2026-01-30", scores=list(range(1, 41)))
    close = _close_matrix().rename(columns={"T40": "T40~40"})
    tmap = {i: f"T{i}" for i in range(1, 40)}
    tmap[40] = "T40~40"
    rep = compare(close=close, paper_dir=paper_dir, ticker_map=tmap, horizons=(5,))
    assert rep["months"][0]["h5"]["n_priced"] == 40  # nobody dropped


def _continuous_close(n=40):
    """Every day, name i sits at 10*(1+i/1000)**t — a monotonic-in-i drift, so ANY
    forward window ranks names by i (both months are genuinely scorable)."""
    idx = pd.bdate_range("2025-12-01", periods=60)
    close = pd.DataFrame(index=idx, columns=[f"T{i}" for i in range(1, n + 1)], dtype=float)
    for i in range(1, n + 1):
        close[f"T{i}"] = 10.0 * (1 + i / 1000) ** np.arange(len(idx))
    return close


def test_compare_excludes_in_sample_months_from_gate(tmp_path):
    paper_dir = tmp_path / "paper"
    _write_baseline(paper_dir)
    # the in-sample month's scores are REVERSED (live IC ~ -1); if it leaked into
    # the gate it would drag live_mean_ic down. It must be scored yet excluded.
    _write_signals(paper_dir, "2025-12-31", scores=list(range(40, 0, -1)), in_sample=True)
    _write_signals(paper_dir, "2026-01-30", scores=list(range(1, 41)))
    close = _continuous_close()
    tmap = {i: f"T{i}" for i in range(1, 41)}
    rep = compare(close=close, paper_dir=paper_dir, ticker_map=tmap, horizons=(5,))
    assert rep["months_scored_in_sample"] == 1
    assert rep["months_scored_oos"] == 1
    assert rep["live_mean_ic"] > 0.95  # the reversed in-sample month did not pollute it


def test_compare_keeps_death_crash_loss(tmp_path):
    """A name that crashes to sub-penny inside the window must show its real ~-99%
    loss, NOT a fabricated ~0% from ffill over hygiene-masked prints (the label
    is measured on RAW prices, matching the frozen baseline)."""
    paper_dir = tmp_path / "paper"
    _write_baseline(paper_dir)
    _write_signals(paper_dir, "2026-01-30", scores=list(range(1, 41)))
    close = _close_matrix()
    # name T1 (lowest score) dies: flat 10 until as_of, then collapses to sub-penny
    idx = close.index
    as_of = pd.Timestamp("2026-01-30")
    loc = idx.get_loc(as_of)
    close.iloc[loc:, 0] = 0.0001  # T1 crashes right after the signal
    tmap = {i: f"T{i}" for i in range(1, 41)}
    rep = compare(close=close, paper_dir=paper_dir, ticker_map=tmap, horizons=(5,))
    m = rep["months"][0]["h5"]
    # the death loss is kept, so the lowest-score name has by far the worst return;
    # rank IC stays strongly positive (score i tracks return rank incl. the crash)
    assert m["rank_ic"] > 0.9
    # WITH the death kept, the bottom decile (which holds T1's ~-99%) is deeply
    # negative and the spread is clearly positive. If hygiene had masked+ffilled T1
    # to ~0%, T1 would rank mid-pack, the bottom decile would be ~0, and the spread
    # would collapse toward 0 — the flattering the fix prevents.
    assert m["decile_spread"] > 0.05
    assert m["top_decile_excess"] is not None and m["top_decile_excess"] > 0


def test_compare_not_enough_elapsed_days(tmp_path):
    paper_dir = tmp_path / "paper"
    _write_baseline(paper_dir, min_months=1)
    _write_signals(paper_dir, "2026-01-30", scores=list(range(1, 41)))
    close = _close_matrix()
    idx = close.index
    close = close.loc[idx <= "2026-01-31"]  # nothing realized yet
    tmap = {i: f"T{i}" for i in range(1, 41)}
    rep = compare(close=close, paper_dir=paper_dir, ticker_map=tmap, horizons=(5,))
    assert rep["months"][0]["h5"] is None
    assert rep["degraded"] is None
