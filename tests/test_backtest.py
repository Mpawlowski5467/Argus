"""Hand-computable checks of the backtest engine's mechanics.

Every test uses prices simple enough to verify the NAV arithmetic by hand:
next-open execution, buy-and-hold windows, liquidity-tiered costs, hysteresis,
borrow accrual, dead-name freezing, and turnover math.
"""

import numpy as np
import pandas as pd

from stockscan.backtest import hysteresis_members, run_backtest, tiered_bps

NO_COST = ((0, 0.0),)
NO_BORROW = ((0, 0.0),)


def _world(prices: dict, start="2024-01-01"):
    """close matrix from per-name price lists; open == close (no overnight gap),
    so an execution at t's open fills at close[t] — hand-checkable."""
    n = len(next(iter(prices.values())))
    idx = pd.bdate_range(start, periods=n)
    close = pd.DataFrame(prices, index=idx, dtype=float)
    opn = close.copy()
    dv = close.notna() * 1e9  # everything hyper-liquid unless a test overrides
    return idx, close, opn, dv


def _scores(rows):
    return pd.DataFrame(rows, columns=["date", "ticker", "pred"])


def test_single_name_nav_tracks_price_exactly():
    idx, close, opn, dv = _world({
        "A": np.linspace(100, 200, 30),  # steady doubling
        "B": [100.0] * 30,
    })
    d = idx[4]
    scores = _scores([(d, "A", 2.0), (d, "B", 1.0)])
    # enter=0.4: threshold pct >= 0.6 -> only A (pct 1.0; B pct 0.5)
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=NO_COST)
    entry = close.loc[idx[5], "A"]  # next-bar open == that day's close here
    for t in idx[5:]:
        assert abs(r.nav.loc[t] - close.loc[t, "A"] / entry) < 1e-12
    assert r.rebalances.iloc[0]["n_long"] == 1


def test_execution_at_next_open_misses_signal_day_move():
    # A pops +50% at the bar AFTER the signal; entry at that bar's open pays the
    # popped price, so the move must NOT be captured.
    a = [100.0] * 5 + [150.0] * 25
    idx, close, opn, dv = _world({"A": a, "B": [100.0] * 30})
    scores = _scores([(idx[4], "A", 2.0), (idx[4], "B", 1.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=NO_COST)
    assert np.allclose(r.nav.to_numpy(), 1.0)  # flat: the pop predates the fill


def test_costs_charged_on_traded_notional_per_side():
    idx, close, opn, dv = _world({"A": [100.0] * 20, "B": [100.0] * 20})
    scores = _scores([(idx[4], "A", 2.0), (idx[4], "B", 1.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=((0, 100.0),))  # 100 bps per side
    # one rebalance, buys 1.0 notional -> 1% cost; prices flat -> nav constant after
    assert abs(r.rebalances.iloc[0]["cost"] - 0.01) < 1e-12
    assert abs(r.nav.iloc[-1] - 0.99) < 1e-12
    assert abs(r.nav_gross.iloc[-1] - 1.0) < 1e-12


def test_hysteresis_membership_bands():
    ranks = pd.Series({f"t{i}": i / 10 for i in range(1, 11)})  # 0.1 .. 1.0
    got = hysteresis_members(ranks, prev=set(), enter=0.2, exit=0.4, side="long")
    assert got == {"t8", "t9", "t10"}                     # top 20% (>= 0.8)
    got = hysteresis_members(ranks, prev={"t7"}, enter=0.2, exit=0.4, side="long")
    assert "t7" in got                                    # 0.7 >= 0.6: holder stays
    got = hysteresis_members(ranks, prev={"t5"}, enter=0.2, exit=0.4, side="long")
    assert "t5" not in got                                # 0.5 < 0.6: holder exits
    got = hysteresis_members(ranks, prev={"t4"}, enter=0.2, exit=0.4, side="short")
    assert got == {"t1", "t2", "t4"}                      # bottom 20% enter, 0.4 stays
    got = hysteresis_members(ranks.drop("t9"), prev={"t9"}, enter=0.2, exit=0.4)
    assert "t9" not in got                                # left the universe: forced out


def test_no_trade_when_book_unchanged_and_prices_flat():
    idx, close, opn, dv = _world({"A": [100.0] * 40, "B": [100.0] * 40})
    scores = _scores([(idx[4], "A", 2.0), (idx[4], "B", 1.0),
                      (idx[24], "A", 2.0), (idx[24], "B", 1.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=((0, 100.0),))
    second = r.rebalances.iloc[1]
    assert second["turnover"] == 0.0 and second["cost"] == 0.0


def test_dead_name_freezes_at_last_print_and_releases_costlessly():
    a = [100.0] * 40
    d_price = [100.0] * 10 + [50.0] * 5 + [np.nan] * 25   # halves, then stops trading
    idx, close, opn, dv = _world({"A": a, "D": d_price})
    scores = _scores([
        (idx[4], "A", 1.0), (idx[4], "D", 2.0),   # both held (enter band wide)
        (idx[24], "A", 1.0),                       # D has left the scored universe
    ])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.6, exit=0.6,
                     cost_tiers=((0, 100.0),))
    # window 1: 50/50; D halves then freezes -> nav 1 - cost(1%) then 0.75 of that
    assert abs(r.nav.loc[idx[20]] - 0.99 * 0.75) < 1e-12
    # rebalance 2 re-centers into A alone. In current-NAV fractions the drifted book
    # is A 2/3, D 1/3: buying A up to 1.0 trades 1/3 notional (with cost); frozen D
    # (no print) releases its third costlessly
    assert abs(r.rebalances.iloc[1]["cost"] - (1.0 / 3.0) * 0.01) < 1e-12
    assert r.rebalances.iloc[1]["n_long"] == 1


def _ls_world(d_prices):
    """4 names: A,B rank at the top (long book), D at the bottom (short book)."""
    n = len(d_prices)
    idx, close, opn, dv = _world({
        "A": [100.0] * n, "B": [100.0] * n, "C": [100.0] * n, "D": d_prices,
    })
    scores = _scores([(idx[4], "A", 4.0), (idx[4], "B", 3.0),
                      (idx[4], "C", 2.0), (idx[4], "D", 1.0)])
    return idx, close, opn, dv, scores


def test_borrow_accrues_nightly_on_short_book_only():
    idx, close, opn, dv, scores = _ls_world([100.0] * 30)
    # enter=0.3: long = pct >= 0.7 -> {A, B}; short = pct <= 0.3 -> {D}
    # borrow tier chosen so daily accrual = 1% exactly: 25200 bps / 1e4 / 252 = 0.01
    r = run_backtest(scores, close, opn, dv, mode="long_short", enter=0.3, exit=0.3,
                     cost_tiers=NO_COST, borrow_tiers=((0, 25200.0),), short_min_adv=0)
    assert r.rebalances.iloc[0]["n_short"] == 1
    k = 10
    t = idx[5 + k]
    assert abs(r.nav.loc[t] - 0.99 ** k) < 1e-9       # flat prices: pure borrow decay
    assert abs(r.nav_gross.loc[t] - 1.0) < 1e-12      # gross ignores borrow


def test_short_profits_when_price_falls():
    idx, close, opn, dv, scores = _ls_world(list(np.linspace(100, 80, 30)))  # D -20%
    r = run_backtest(scores, close, opn, dv, mode="long_short", enter=0.3, exit=0.3,
                     cost_tiers=NO_COST, borrow_tiers=NO_BORROW, short_min_adv=0)
    entry = close.loc[idx[5], "D"]
    expected = 1.0 + (-1.0) * (close.iloc[-1]["D"] / entry - 1.0)  # longs are flat
    assert abs(r.nav.iloc[-1] - expected) < 1e-12


def test_contradictory_both_tail_signal_goes_to_neither_book():
    idx, close, opn, dv = _world({"A": [100.0] * 20, "B": [100.0] * 20})
    scores = _scores([(idx[4], "A", 2.0), (idx[4], "B", 1.0)])
    # 2 names, enter=0.5: B (pct 0.5) qualifies for BOTH tails -> excluded from both
    r = run_backtest(scores, close, opn, dv, mode="long_short", enter=0.5, exit=0.5,
                     cost_tiers=NO_COST, short_min_adv=0)
    first = r.rebalances.iloc[0]
    assert first["n_long"] == 1 and first["n_short"] == 0


def test_hard_to_borrow_names_are_never_shorted():
    idx, close, opn, dv, scores = _ls_world([100.0] * 30)
    dv["D"] = 1e5  # far below the short_min_adv floor
    r = run_backtest(scores, close, opn, dv, mode="long_short", enter=0.3, exit=0.3,
                     cost_tiers=NO_COST, short_min_adv=5e6)
    assert (r.rebalances["n_short"] == 0).all()
    assert (r.rebalances["n_long"] > 0).all()  # the long book is unaffected


def test_universe_mode_is_the_equal_weight_benchmark():
    idx, close, opn, dv = _world({
        "A": list(np.linspace(100, 110, 30)),
        "B": list(np.linspace(100, 90, 30)),
        "C": [100.0] * 30,
    })
    d = idx[4]
    scores = _scores([(d, "A", 3.0), (d, "B", 2.0), (d, "C", 1.0)])
    r = run_backtest(scores, close, opn, dv, mode="universe", cost_tiers=NO_COST)
    entry = close.loc[idx[5]]
    expected = np.mean([close.iloc[-1][t] / entry[t] for t in "ABC"])
    assert abs(r.nav.iloc[-1] - expected) < 1e-12


def test_trust_floor_takes_the_crash_but_never_the_quantized_bounce():
    # D collapses over several (individually possible) days to $0.02 — all real and
    # kept — then prints sub-floor quantization noise including a 50x "bounce" to
    # $0.005 that must NOT enter NAV.
    d = [100.0] * 6 + [10.0, 1.0, 0.08, 0.02] + [0.0001] * 10 + [0.005] * 16
    idx, close, opn, dv = _world({"A": [100.0] * 36, "D": d})
    scores = _scores([(idx[4], "A", 1.0), (idx[4], "D", 2.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.6, exit=0.6,
                     cost_tiers=NO_COST, trust_floor=0.01)
    # 50/50 book entered at 100/100: D's -99.98% crash is fully taken...
    expected = 0.5 * 1.0 + 0.5 * (0.02 / 100.0)
    assert abs(r.nav.loc[idx[9]] - expected) < 1e-9
    # ...and the NAV never recovers on the untrusted bounce
    assert abs(r.nav.iloc[-1] - expected) < 1e-9
    assert r.config["n_masked_prints"] == 26  # every sub-floor print was counted


def test_scale_break_is_repaired_not_erased():
    # D's adjusted series jumps 100x mid-hold (corporate-action mis-adjustment) and
    # stays on the wrong scale; it then falls 20% ON the wrong scale (a real move).
    # Repair: the fake 100x never enters NAV, but the real -20% after it DOES.
    d = [100.0] * 10 + [10_000.0] * 10 + [8_000.0] * 10
    idx, close, opn, dv = _world({"A": [100.0] * 30, "D": d})
    scores = _scores([(idx[4], "A", 1.0), (idx[4], "D", 2.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.6, exit=0.6,
                     cost_tiers=NO_COST)
    assert abs(r.nav.loc[idx[15]] - 1.0) < 1e-9            # no fake 100x gain
    assert abs(r.nav.iloc[-1] - (0.5 + 0.5 * 0.8)) < 1e-9  # the real -20% is taken
    assert r.config["n_break_days"] == 1 and r.config["n_break_names"] == 1


def test_real_catastrophic_crash_is_never_treated_as_a_break():
    # Tricida-style: -94% in one day on real volume. This is INSIDE the repair
    # thresholds and must hit the NAV in full — erasing it would be look-ahead.
    e = [100.0] * 10 + [6.0] * 20
    idx, close, opn, dv = _world({"A": [100.0] * 30, "E": e})
    scores = _scores([(idx[4], "A", 1.0), (idx[4], "E", 2.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.6, exit=0.6,
                     cost_tiers=NO_COST)
    assert abs(r.nav.iloc[-1] - (0.5 + 0.5 * 0.06)) < 1e-12
    assert r.config["n_break_days"] == 0


def test_execution_fills_at_open_when_open_gaps_from_close():
    # prev close 100 -> exec-day open 110, close 120: the fill must be at 110,
    # so the first mark is 120/110, NOT 120/100 or 120/120.
    close_a = [100.0] * 5 + [120.0] * 15
    idx, close, opn, dv = _world({"A": close_a, "B": [100.0] * 20})
    opn.loc[idx[5], "A"] = 110.0
    scores = _scores([(idx[4], "A", 2.0), (idx[4], "B", 1.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=NO_COST)
    assert abs(r.nav.loc[idx[5]] - 120.0 / 110.0) < 1e-12


def test_missing_open_with_valid_close_fills_at_that_close_with_cost():
    # H trades on the exec bar (close 40 = -60%) but its open print is missing:
    # the exit must fill at 40 (same-day) WITH cost — never a costless phantom
    # exit at the stale prior close of 100.
    h = [100.0] * 25 + [40.0] * 15
    idx, close, opn, dv = _world({"A": [100.0] * 40, "H": h})
    opn.loc[idx[25], "H"] = np.nan
    scores = _scores([
        (idx[4], "A", 1.0), (idx[4], "H", 2.0),    # both held
        (idx[24], "A", 2.0), (idx[24], "H", 1.0),  # H must be traded on idx[25]
    ])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.6, exit=0.6,
                     cost_tiers=((0, 100.0),))
    # window 1: 50/50, prices flat until the exec bar; H marks at 40 (-60% on its half)
    nav_after_mark = 0.99 * (1.0 + 0.5 * (40.0 / 100.0 - 1.0))
    reb2 = r.rebalances.iloc[1]
    assert reb2["n_long"] == 2                      # H stays holdable via its close
    assert reb2["turnover"] > 0                     # re-centering H's crashed weight
    assert reb2["cost"] > 0                         # ...and it is NOT costless
    assert r.nav.loc[idx[25]] < nav_after_mark + 1e-9  # the -60% was taken


def test_costs_use_drift_normalized_current_nav_fractions():
    # X halves during the window (window return -25%): drifted fractions are
    # 2/3 A, 1/3 X of CURRENT nav; re-centering to 50/50 trades 1/3 of nav total.
    x = [100.0] * 8 + [50.0] * 27  # crash AFTER the idx[5] entry at 100
    idx, close, opn, dv = _world({"A": [100.0] * 35, "X": x})
    scores = _scores([
        (idx[4], "A", 1.0), (idx[4], "X", 2.0),
        (idx[24], "A", 1.0), (idx[24], "X", 2.0),  # same book, re-centered
    ])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.6, exit=0.6,
                     cost_tiers=((0, 100.0),))
    reb2 = r.rebalances.iloc[1]
    assert abs(reb2["turnover"] - (1.0 / 3.0) / 2.0) < 1e-12
    assert abs(reb2["cost"] - (1.0 / 3.0) * 0.01) < 1e-12


def test_nav_series_contains_the_inception_point():
    idx, close, opn, dv = _world({"A": [100.0] * 20, "B": [100.0] * 20})
    scores = _scores([(idx[4], "A", 2.0), (idx[4], "B", 1.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=((0, 100.0),))
    assert r.nav.index[0] == idx[4] and r.nav.iloc[0] == 1.0  # pre-execution base
    # so the very first daily return already carries the first trade's cost
    assert abs(r.daily_returns().iloc[0] - (-0.01)) < 1e-12


def test_short_squeeze_past_equity_liquidates_to_zero_not_negative():
    # the short book quintuples over two (individually possible) days: window
    # return < -100% of NAV -> account liquidates at 0 and STAYS there; summary
    # reports -100% CAGR instead of NaN.
    d = [100.0] * 6 + [300.0] * 2 + [500.0] * 22
    idx, close, opn, dv = _world({"A": [100.0] * 30, "B": [100.0] * 30,
                                  "C": [100.0] * 30, "D": d})
    scores = _scores([(idx[4], t, s) for t, s in
                      [("A", 4.0), ("B", 3.0), ("C", 2.0), ("D", 1.0)]]
                     + [(idx[24], t, s) for t, s in
                        [("A", 4.0), ("B", 3.0), ("C", 2.0), ("D", 1.0)]])
    r = run_backtest(scores, close, opn, dv, mode="long_short", enter=0.3, exit=0.3,
                     cost_tiers=NO_COST, borrow_tiers=NO_BORROW, short_min_adv=0)
    assert (r.nav >= 0).all()
    assert r.nav.iloc[-1] == 0.0 and r.config["wiped_out"]
    assert r.summary()["cagr_net"] == -1.0


def test_signal_before_price_history_is_untradable_not_wrapped():
    idx, close, opn, dv = _world({"A": [100.0] * 20, "B": [100.0] * 20})
    early = idx[0] - pd.Timedelta(days=30)  # predates the whole price history
    scores = _scores([(early, "A", 2.0), (early, "B", 1.0),
                      (idx[4], "A", 2.0), (idx[4], "B", 1.0)])
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=NO_COST)
    assert len(r.rebalances) == 1                       # the early signal is skipped
    assert r.rebalances.iloc[0]["date"] == idx[4]


def test_tiered_bps_maps_floors_and_treats_missing_as_worst():
    dv = pd.Series([100e6, 20e6, 7e6, 2e6, 0.5e6, np.nan])
    got = tiered_bps(dv)
    assert got.tolist() == [10.0, 20.0, 35.0, 60.0, 100.0, 100.0]


def test_summary_reports_cost_drag_and_turnover():
    idx, close, opn, dv = _world({"A": [100.0] * 300, "B": [100.0] * 300})
    scores = _scores(
        [(idx[i], "A", 2.0) for i in range(4, 280, 21)]
        + [(idx[i], "B", 1.0) for i in range(4, 280, 21)]
    )
    r = run_backtest(scores, close, opn, dv, mode="long_only", enter=0.4, exit=0.4,
                     cost_tiers=((0, 100.0),))
    s = r.summary()
    assert s["cagr_gross"] == 0.0 and s["cagr_net"] < 0  # flat prices: pure cost drag
    assert s["cost_drag"] > 0
    assert s["ann_turnover"] >= 0
