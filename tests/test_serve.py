"""The three DESIGN.md §2 invariants, enforced on the SERVE path (Phase-2 gate).

1. PIT guard      — the serve cross-section can never contain a filing that was not
                    public at as-of; assert_pit is wired into the path itself.
2. Train/serve    — the serve feature vector for (cik, date) is bit-identical to the
   parity           training panel row (same shared transform code, same universe).
3. Grounding      — every numeral in the narration traces to the signal packet; an
                    invented number is caught.

Plus: a delisted name flows through the identical code path, no special-casing.
"""

import numpy as np
import pandas as pd
import pytest

import stockscan.fundamental_panel as fp
from stockscan.features import FEATURES
from stockscan.fundamental_panel import build_fundamental_panel, prepare_features
from stockscan.model import RANK_COLS, fit, load_artifact, save_artifact
from stockscan.narrate.ground import check_grounding
from stockscan.serve import ServeData, analyze, build_cross_section, resolve_company

MIN_DV = 1_000_000
HORIZON = 21


def _raw_features():
    """45 companies x 2 fiscal years. 30 manufacturing (sector-ranked bucket) +
    15 finance (thin bucket -> date-rank fallback path also exercised)."""
    rng = np.random.default_rng(7)
    rows = []
    for cik in range(1, 46):
        for fy, filed in ((2022, "2023-03-01"), (2023, "2024-03-01")):
            r = {
                "cik": cik, "name": f"CO{cik}", "fy": fy,
                "sic": 3571 if cik <= 30 else 6021,
                "filed_date": pd.Timestamp(filed),
                "period_end": pd.Timestamp(f"{fy}-12-31"),
            }
            r.update({f: rng.uniform(0.05, 0.9) for f in FEATURES})
            rows.append(r)
    return pd.DataFrame(rows)


def _prices():
    """Random-walk closes. DEAD~45 stops trading 2024-08-15 (delisting).
    T44 trades at $0.50 -- below the price floor, so outside the tradable universe."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2023-06-01", "2024-12-31")
    close = pd.DataFrame(index=idx)
    for cik in range(1, 44):
        base = rng.uniform(20, 150)
        close[f"T{cik}"] = base * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    close["T44"] = 0.5
    dead = pd.Series(
        np.linspace(60.0, 3.0, (idx <= "2024-08-15").sum()),
        index=idx[idx <= "2024-08-15"],
    )
    close["DEAD~45"] = dead.reindex(idx)
    dv = close * 1_000_000  # constant share volume; dollar volume ~ price * 1e6
    return close, dv


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    raw = _raw_features()
    close, dv = _prices()
    tmap = {cik: f"T{cik}" for cik in range(1, 45)}
    tmap[45] = "DEAD~45"

    panel = build_fundamental_panel(
        raw, close, delistings=None, ticker_map=tmap, dollar_volume=dv,
        min_dollar_volume=MIN_DV, horizon=HORIZON, min_names=5,
    )
    model = fit(panel, params=dict(n_estimators=20, min_child_samples=10))
    art_dir = tmp_path_factory.mktemp("artifact")
    save_artifact(model, panel, out_dir=art_dir)

    data = ServeData(
        feats=prepare_features(raw),
        close=close,
        dv_med=dv.rolling(20, min_periods=10).median(),
        ticker_map=tmap,
    )
    return {"raw": raw, "panel": panel, "data": data,
            "artifact": load_artifact(art_dir), "close": close}


# --- invariant 1: PIT guard on the serve path -----------------------------------

def test_serve_cross_section_is_point_in_time(world):
    # fy2023 10-K filed Fri 2024-03-01 -> usable Mon 2024-03-04, not a day sooner
    before = build_cross_section(world["data"], "2024-03-01", min_dollar_volume=MIN_DV)
    assert (before["available_date"] <= pd.Timestamp("2024-03-01")).all()
    assert (before["fy"] == 2022).all()

    after = build_cross_section(world["data"], "2024-03-04", min_dollar_volume=MIN_DV)
    assert (after["fy"] == 2023).all()
    assert (after["available_date"] <= pd.Timestamp("2024-03-04")).all()


def test_serve_path_calls_assert_pit(world, monkeypatch):
    """The guard must be structurally ON the serve path, not just available."""
    class Tripped(Exception):
        pass

    def tripwire(*a, **k):
        raise Tripped

    monkeypatch.setattr(fp, "assert_pit", tripwire)
    with pytest.raises(Tripped):
        build_cross_section(world["data"], "2024-06-28", min_dollar_volume=MIN_DV)


def test_serve_refuses_dates_before_any_public_filing(world):
    with pytest.raises(ValueError):
        analyze(7, as_of="2023-01-15", data=world["data"], artifact=world["artifact"])


# --- invariant 2: train/serve parity ---------------------------------------------

def test_train_serve_parity_bit_identical(world):
    """The serve feature vector equals the training panel row EXACTLY, for every
    company on a shared date -- same transforms, same universe, no tolerance."""
    panel, data = world["panel"], world["data"]
    d = pd.Timestamp("2024-06-28")  # a month-end inside the panel
    panel_rows = panel[panel["date"] == d].set_index("cik")
    assert len(panel_rows) > 30

    cross = build_cross_section(data, d, min_dollar_volume=MIN_DV).set_index("cik")
    for cik in panel_rows.index:
        for c in RANK_COLS:
            assert cross.loc[cik, c] == panel_rows.loc[cik, c], (cik, c)


def test_analyze_serves_the_panel_feature_vector(world):
    """End-to-end: analyze() returns the identical vector the model trained on."""
    d = pd.Timestamp("2024-06-28")
    r = analyze(7, as_of=d, data=world["data"], artifact=world["artifact"])
    row = world["panel"][(world["panel"]["date"] == d) & (world["panel"]["cik"] == 7)].iloc[0]
    assert r["ranks"] == {c: float(row[c]) for c in RANK_COLS}
    assert r["flags"]["in_sample"]  # d is inside the training window and says so


# --- invariant 3: grounded narration ---------------------------------------------

def test_serve_narration_is_grounded_and_tampering_is_caught(world):
    r = analyze(7, as_of="2024-06-28", data=world["data"], artifact=world["artifact"])
    assert r["grounded"] and not r["grounding_violations"]
    tampered = r["narrative"] + " Hidden upside of 77.77% awaits."
    assert 77.77 in check_grounding(tampered, r["packet"])


def test_hallucinating_llm_falls_back_to_template(world):
    def bad_llm(system, user):
        return "Secret model edge of 42.4242% guaranteed."

    r = analyze(7, as_of="2024-06-28", data=world["data"], artifact=world["artifact"],
                llm=bad_llm)
    assert r["source"] == "template-fallback"
    assert r["grounded"]


# --- the delisted name: identical path, no special-casing ------------------------

def test_dead_name_flows_through_the_same_path(world):
    live = analyze("T7", as_of="2024-06-28", data=world["data"], artifact=world["artifact"])
    dead = analyze("DEAD~45", as_of="2024-06-28", data=world["data"], artifact=world["artifact"])
    assert dead.keys() == live.keys()          # same shape, same code path
    assert dead["packet"]["meta"]["name"] == "CO45"
    assert dead["flags"]["liquidity_pass"]     # alive and liquid at that date
    assert dead["grounded"]
    assert 1 <= dead["decile"] <= 10


def test_dead_name_after_death_is_flagged_not_special_cased(world):
    # 2024-09-30 is after DEAD~45's last trade (2024-08-15) but its 10-K is fresh:
    # it still analyzes through the same path, flagged as below the liquidity floor.
    r = analyze("DEAD~45", as_of="2024-09-30", data=world["data"], artifact=world["artifact"])
    assert not r["flags"]["liquidity_pass"]
    assert r["grounded"]


def test_illiquid_name_is_kept_only_for_itself(world):
    # T44 ($0.50) fails the price floor: excluded from everyone else's cross-section...
    cross = build_cross_section(world["data"], "2024-06-28", min_dollar_volume=MIN_DV)
    assert 44 not in set(cross["cik"])
    # ...but can still be analyzed directly, flagged.
    r = analyze("T44", as_of="2024-06-28", data=world["data"], artifact=world["artifact"])
    assert not r["flags"]["liquidity_pass"]


def test_resolve_company_forms(world):
    tmap = world["data"].ticker_map
    assert resolve_company("T7", tmap) == (7, "T7")
    assert resolve_company("DEAD~45", tmap) == (45, "DEAD~45")
    assert resolve_company(45, tmap) == (45, "DEAD~45")
    assert resolve_company("45", tmap) == (45, "DEAD~45")


# --- FIREWALLED distress risk-flag: display-only, never touches the signal ---------

def test_distress_head_is_display_only_and_firewalled(world, tmp_path):
    """With a distress artifact attached, analyze() adds a `distress` block but the
    return score/percentile/decile/drivers/packet are byte-identical to a run without it."""
    from stockscan.distress import fit_distress, load_distress_artifact, save_distress_artifact

    dp = world["panel"].copy()
    dp["y"] = (dp["cik"] == 45).astype(float)     # the delisting name is the positive
    dp.attrs["censor_date"] = dp["date"].max()
    dp.attrs["horizon_months"] = 12
    dmodel = fit_distress(dp, params=dict(n_estimators=15, min_child_samples=5))
    dart = load_distress_artifact(save_distress_artifact(dmodel, dp, out_dir=tmp_path / "d"))

    d = pd.Timestamp("2024-06-28")
    with_d = analyze(7, as_of=d, data=world["data"], artifact=world["artifact"],
                     distress_artifact=dart)
    without = analyze(7, as_of=d, data=world["data"], artifact=world["artifact"])

    assert without["distress"] is None                       # optional: absent when no head
    dz = with_d["distress"]
    assert set(dz) >= {"prob", "percentile", "flag", "horizon_months"}
    assert 0.0 <= dz["prob"] <= 1.0 and dz["flag"] in ("normal", "elevated", "high")
    assert 0 <= dz["percentile"] <= 100 and dz["horizon_months"] == 12

    # THE FIREWALL: the traded signal cannot move because a risk-flag was attached
    for k in ("score", "percentile", "decile", "ranks"):
        assert with_d[k] == without[k], k
    assert with_d["packet"] == without["packet"]             # nothing leaked into the packet
