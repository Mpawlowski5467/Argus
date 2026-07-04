"""Monitor loop: filing detection, percentile alerts, cache-safe re-narration."""

import numpy as np
import pandas as pd
import pytest

from stockscan.distress import fit_distress, load_distress_artifact, save_distress_artifact
from stockscan.features import FEATURES
from stockscan.fundamental_panel import build_fundamental_panel, prepare_features
from stockscan.model import fit, load_artifact, save_artifact
from stockscan.narrate.cache import NarrationCache, narrate_smart
from stockscan.ops.monitor import detect_wide_filings, run_monitor
from stockscan.ops.state import OpsState
from stockscan.serve import ServeData

MIN_DV = 1_000_000
HORIZON = 21


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
    for cik in range(1, 46):
        base = rng.uniform(20, 150)
        close[f"T{cik}"] = base * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
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
    return {"data": data, "artifact": load_artifact(model_dir), "raw": raw}


def test_detect_wide_filings_bootstrap_then_news(world, tmp_path):
    with OpsState(tmp_path / "s.sqlite") as state:
        feats = prepare_features(world["raw"])
        # first sight seeds silently — everything predates the watch
        assert detect_wide_filings(state, feats, [1]) == []
        assert state.has_filings(1, source="fsds")
        # a genuinely new filing lands
        new_row = world["raw"][world["raw"]["cik"] == 1].iloc[[0]].copy()
        new_row["fy"] = 2024
        new_row["filed_date"] = pd.Timestamp("2025-03-01")
        new_row["period_end"] = pd.Timestamp("2024-12-31")
        feats2 = prepare_features(pd.concat([world["raw"], new_row], ignore_index=True))
        news = detect_wide_filings(state, feats2, [1])
        assert len(news) == 1 and news[0]["period_end"] == "2024-12-31"


def test_run_monitor_no_llm_template(world, tmp_path):
    with OpsState(tmp_path / "s.sqlite") as state:
        state.watch_add(1, "T1")
        state.watch_add(2, "T2")
        cache = NarrationCache(tmp_path / "narr.sqlite")
        deltas = run_monitor(state, data=world["data"], artifact=world["artifact"],
                             llm_full=None, llm_light=None, cache=cache,
                             edgar=False, as_of="2024-09-30")
        assert deltas["n_watch"] == 2
        # first pass: fundamentals_updated alerts seed nothing (bootstrap), signal
        # state records both names, template narration ran but did NOT cache
        assert deltas["narrated"].get("template", 0) == 2
        assert state.get_signal(1) is not None
        # a --no-llm run must not poison the cache
        assert cache.get(1) is None


def test_percentile_move_alert(world, tmp_path):
    from stockscan.serve import analyze

    with OpsState(tmp_path / "s.sqlite") as state:
        state.watch_add(1, "T1")
        res = analyze(1, as_of="2024-09-30", data=world["data"],
                      artifact=world["artifact"], llm=None)
        real = res["percentile"]
        # seed an artificial prior signal >=10 points from the real one
        state.record_signal(1, (real + 40) % 100, 5, "2024-08-30")
        run_monitor(state, data=world["data"], artifact=world["artifact"],
                    narrate=False, edgar=False, as_of="2024-09-30")
        moves = [a for a in state.alerts(unseen_only=False) if a["kind"] == "percentile_move"]
        assert len(moves) == 1
        assert state.get_signal(1)["percentile"] == real  # updated to the real pct


def _data_with_distress(world, tmp_path):
    """world['data'] plus a FIREWALLED distress artifact. Fabricated ~50% labels give a
    signal-free model that scores broadly high — enough to exercise the alert plumbing."""
    raw = world["raw"]
    close, dv = _prices()
    tmap = {cik: f"T{cik}" for cik in range(1, 46)}
    panel = build_fundamental_panel(
        raw, close, delistings=None, ticker_map=tmap, dollar_volume=dv,
        min_dollar_volume=MIN_DV, horizon=HORIZON, min_names=5)
    panel = panel.copy()
    panel["y"] = (np.arange(len(panel)) % 2).astype(float)
    panel.attrs["censor_date"] = panel["date"].max()
    panel.attrs["horizon_months"] = 12
    dmodel = fit_distress(panel, params=dict(n_estimators=15, min_child_samples=5))
    dart = load_distress_artifact(save_distress_artifact(dmodel, panel, out_dir=tmp_path / "d"))
    return ServeData(feats=prepare_features(raw), close=close,
                     dv_med=dv.rolling(20, min_periods=10).median(), ticker_map=tmap,
                     distress_artifact=dart)


def test_distress_escalation_alerts_once_then_stays_quiet(world, tmp_path):
    data = _data_with_distress(world, tmp_path)
    with OpsState(tmp_path / "s.sqlite") as state:
        state.watch_add(1, "T1")
        state.record_signal(1, 50, 5, "2024-08-30", distress=0.0)  # prior = normal
        # first run: distress crosses UP from normal -> a single distress_risk alert
        run_monitor(state, data=data, artifact=world["artifact"],
                    narrate=False, edgar=False, as_of="2024-09-30")
        d1 = [a for a in state.alerts(unseen_only=False) if a["kind"] == "distress_risk"]
        assert len(d1) == 1
        assert state.get_signal(1)["distress"] is not None      # level now recorded
        # a second identical run is NOT a new escalation -> no repeat alert
        run_monitor(state, data=data, artifact=world["artifact"],
                    narrate=False, edgar=False, as_of="2024-09-30")
        d2 = [a for a in state.alerts(unseen_only=False) if a["kind"] == "distress_risk"]
        assert len(d2) == 1


def test_distress_alert_suppressed_when_degraded(world, tmp_path):
    data = _data_with_distress(world, tmp_path)
    with OpsState(tmp_path / "s.sqlite") as state:
        state.watch_add(1, "T1")
        state.record_signal(1, 50, 5, "2024-08-30", distress=0.0)
        run_monitor(state, data=data, artifact=world["artifact"], narrate=False,
                    edgar=False, as_of="2024-09-30", alerts_ok=False)
        assert [a for a in state.alerts(unseen_only=False) if a["kind"] == "distress_risk"] == []


def test_alerts_suppressed_when_degraded(world, tmp_path):
    with OpsState(tmp_path / "s.sqlite") as state:
        state.watch_add(1, "T1")
        state.record_signal(1, 5, 1, "2024-08-30")  # would trigger a big move
        deltas = run_monitor(state, data=world["data"], artifact=world["artifact"],
                             narrate=False, edgar=False, as_of="2024-09-30",
                             alerts_ok=False)
        moves = [a for a in state.alerts(unseen_only=False) if a["kind"] == "percentile_move"]
        assert moves == []          # no alerting on a degraded price night
        assert deltas["alerts_suppressed"] is True


def test_template_run_does_not_evict_full_cache(world, tmp_path):
    """The critical review finding: a --no-llm monitor must not overwrite a cached
    full-tier narration nor reset the materiality baseline."""
    import json

    from stockscan.narrate.narrator import expected_directions
    from stockscan.serve import analyze

    cache = NarrationCache(tmp_path / "narr.sqlite")
    res = analyze(1, as_of="2024-09-30", data=world["data"],
                  artifact=world["artifact"], llm=None)

    def good_llm(system, user):  # llm(system, user) -> str, grounded so source='llm'
        pkt = json.loads(user)
        s = pkt["signals"][0]
        exp = expected_directions(pkt)
        return json.dumps({
            "reasoning": f"{s['label']} sits at the {s['pct_rank']}th percentile.",
            "summary": (f"{pkt['meta']['name']} shows {s['label']} of {s['value']}"
                        f"{s['unit']} at the {s['pct_rank']}th percentile."),
            "citations": [{"id": s["id"], "direction": exp[s["id"]]}],
        })

    full = narrate_smart(res["packet"], llm_full=good_llm, cache=cache)
    assert full["tier"] == "full" and full["source"] == "llm"
    cached_before = cache.get(1)["result"]["narrative"]
    # a template (no-llm) pass must not evict the cached full narration
    narrate_smart(res["packet"], llm_full=None, cache=cache)
    assert cache.get(1)["result"]["narrative"] == cached_before
