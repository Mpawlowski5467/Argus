"""Vectorized long/short backtester with liquidity-scaled costs and borrow realism.

The custom ~200-line pandas engine DESIGN.md §6 calls for — full point-in-time
control, no framework. Mechanics:

- Signals are OOS model scores on monthly rebalance dates (walk-forward — never the
  in-sample frozen artifact). Everything known at ``d`` trades at the NEXT bar's
  open; if the open print is missing but the name traded that day, the fill falls
  back to that day's close (never the prior day's — a stale-price fill would dodge
  same-day moves and understate costs).
- Between rebalances each book is exact buy-and-hold: the window's value path is
  ``sum_j w_j * P_j(t)/P_j(entry)``, marked at daily closes and re-marked at the
  next execution. No drift bookkeeping to get wrong.
- Membership uses hysteresis (enter inside the ``enter`` tail, stay while inside the
  ``exit`` tail) to cut turnover; a name that leaves the scored universe (death,
  liquidity failure, lapsed filings) is force-exited. A name with NO trusted print
  at the execution bar is frozen at its last trusted mark and its frozen value is
  released without a trading cost (there is nothing left to trade).
- Costs: per-side bps keyed to each name's 20d-median dollar volume at trade time,
  charged on traded notional against the DRIFTED current-NAV weights. Borrow:
  annualized bps by ADV tier accrued per night on the short book; names under
  ``short_min_adv`` are hard-to-borrow -> never shorted. A short squeeze can wipe
  the account: NAV floors at zero (liquidation) and stays there.

Data hygiene (counted and reported, never silent):

- Scale-break REPAIR: some delisted names' adjusted series jump scale mid-stream by
  a mis-applied corporate action (observed: 11.09 -> 137,160.00 overnight on ~$100k
  volume). No real security moves close-to-close beyond ~6x up / ~-95% down in a
  day (worst observed real: Tricida -94.5% on trial failure — which must be KEPT),
  so thresholds sit far outside reality (20x up, -96% down). A break day's
  log-return is zeroed and the series rebuilt, preserving every surrounding real
  return at a consistent scale — repair, not erasure: freezing/erasing would also
  erase real crashes and exit positions at pre-crash prices (look-ahead).
- Trust floor: adjusted prices are stored quantized (4 decimals), so a sub-penny
  print carries 1-2 significant digits and its "returns" are quantization noise
  (0.0001 -> 0.0002 reads as +100% on a $10 day). Prints below ``trust_floor`` are
  unmarkable: a collapsing name is marked down to its last trusted print (the real
  ~-99% loss IS taken) and then frozen — fake bounces never compound into NAV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    BORROW_TIERS_BPS,
    COST_TIERS_BPS,
    HYSTERESIS_ENTER,
    HYSTERESIS_EXIT,
    SHORT_MIN_ADV,
)

TRADING_DAYS = 252


def repair_scale_breaks(
    close: pd.DataFrame, up: float = 20.0, down: float = 25.0
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Rebuild each series with impossible one-day ratios (vendor mis-adjustments)
    zeroed out, keeping all surrounding real returns at a consistent scale.

    Returns ``(repaired_close, factor, n_break_days, n_break_names)``; multiply any
    other price matrix (opens) by ``factor`` to keep it consistent.
    """
    logp = np.log(close.ffill())
    r = logp.diff()
    # only consecutive-day prints can be a mis-adjustment; a jump across a trading
    # GAP (delisting splice: last exchange print -> OTC resumption months later)
    # is a real collapse and must be kept
    consec = close.notna() & close.shift(1).notna()
    brk = ((r > np.log(up)) | (r < -np.log(down))) & consec
    n_days = int(brk.to_numpy().sum())
    n_names = int(brk.any().sum())
    corr = r.where(brk, 0.0).fillna(0.0).cumsum()
    factor = np.exp(-corr)
    return close * factor, factor, n_days, n_names


def tiered_bps(dv: pd.Series, tiers=COST_TIERS_BPS) -> pd.Series:
    """Map each name's dollar volume to bps via descending ``(floor, bps)`` tiers.

    Missing dollar volume gets the worst (lowest-floor) tier — unknown liquidity is
    treated as expensive, never as free.
    """
    worst = float(sorted(tiers, key=lambda t: t[0])[0][1])
    out = pd.Series(worst, index=dv.index)
    for floor, bps in sorted(tiers, key=lambda t: t[0]):  # ascending: higher floors win
        out[dv.fillna(-1.0) >= floor] = float(bps)
    return out


def hysteresis_members(
    rank_pct: pd.Series, prev: set, enter: float, exit: float, side: str = "long"
) -> set:
    """New book membership: enter past the ``enter`` band, keep holders inside ``exit``.

    ``rank_pct`` is the cross-sectional percentile of the score (1.0 = best). A prev
    member absent from ``rank_pct`` (left the tradable universe) is dropped.
    """
    if side == "long":
        entering = set(rank_pct.index[rank_pct >= 1.0 - enter])
        staying = set(rank_pct.index[rank_pct >= 1.0 - exit]) & prev
    else:
        entering = set(rank_pct.index[rank_pct <= enter])
        staying = set(rank_pct.index[rank_pct <= exit]) & prev
    return entering | staying


def _book_weights(members: set, rank_pct: pd.Series, weighting: str, sign: float) -> pd.Series:
    """Equal or rank-linear (conviction) weights for one book, gross = 1.0, signed."""
    names = sorted(members)
    if not names:
        return pd.Series(dtype=float)
    if weighting == "rank":
        edge = (rank_pct.reindex(names) - 0.5).abs()
        w = edge / edge.sum() if edge.sum() > 0 else pd.Series(1.0 / len(names), index=names)
    else:
        w = pd.Series(1.0 / len(names), index=names)
    return sign * w


@dataclass
class BacktestResult:
    nav: pd.Series                 # daily net NAV (costs + borrow included)
    nav_gross: pd.Series           # daily NAV before costs and borrow
    rebalances: pd.DataFrame       # per-rebalance: turnover, cost, books, borrow rate
    config: dict = field(default_factory=dict)

    def daily_returns(self, gross: bool = False) -> pd.Series:
        nav = self.nav_gross if gross else self.nav
        return nav.pct_change().dropna()

    def monthly_returns(self, gross: bool = False) -> pd.Series:
        nav = self.nav_gross if gross else self.nav
        return nav.resample("ME").last().pct_change().dropna()

    def summary(self) -> dict:
        r = self.daily_returns()
        years = max(len(r) / TRADING_DAYS, 1e-9)
        final, final_g = float(self.nav.iloc[-1]), float(self.nav_gross.iloc[-1])
        cagr = final ** (1 / years) - 1 if final > 0 else -1.0  # wiped out = -100%
        cagr_gross = final_g ** (1 / years) - 1 if final_g > 0 else -1.0
        vol = float(r.std() * np.sqrt(TRADING_DAYS))
        dd = float((self.nav / self.nav.cummax() - 1).min())
        return {
            "cagr_net": float(cagr),
            "cagr_gross": float(cagr_gross),
            "cost_drag": float(cagr_gross - cagr),
            "ann_vol": vol,
            "sharpe_net": float(cagr / vol) if vol > 0 else float("nan"),
            "max_drawdown": dd,
            "ann_turnover": float(self.rebalances["turnover"].mean() * 12),
            "avg_n_long": float(self.rebalances["n_long"].mean()),
            "avg_n_short": float(self.rebalances["n_short"].mean()),
            "years": years,
        }


def run_backtest(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    opn: pd.DataFrame,
    dv_med: pd.DataFrame,
    mode: str = "long_only",
    enter: float = HYSTERESIS_ENTER,
    exit: float = HYSTERESIS_EXIT,
    weighting: str = "equal",
    cost_tiers=COST_TIERS_BPS,
    borrow_tiers=BORROW_TIERS_BPS,
    short_min_adv: float = SHORT_MIN_ADV,
    cost_scale: float = 1.0,
    trust_floor: float = 0.01,
    repair_breaks: bool = True,
) -> BacktestResult:
    """Run the engine. ``scores``: long frame with (date, ticker, pred) on rebalance dates.

    ``mode``: "long_only" | "long_short" | "universe" (equal-weight ALL scored names —
    the benchmark the long book must beat). Gross exposure: 1.0 for the long book,
    plus 1.0 short book in long_short (dollar-neutral gross 2.0).
    """
    scores = scores.dropna(subset=["pred"])
    idx = close.index

    # data hygiene (module doc): mask sub-penny quantization noise FIRST so it can't
    # register as scale breaks, then repair vendor mis-adjustments on trusted prints
    n_masked = 0
    if trust_floor:
        trusted = close >= trust_floor
        n_masked = int((close.notna() & ~trusted).to_numpy().sum())
        close = close.where(trusted)
        opn = opn.where(opn >= trust_floor)
    n_break_days = n_break_names = 0
    if repair_breaks:
        close, factor, n_break_days, n_break_names = repair_scale_breaks(close)
        opn = opn * factor
    valued = close.ffill()  # a name frozen at its last (trusted) print holds that value

    # (signal_date, exec_position): trade at the first bar strictly AFTER the signal.
    # pos >= 1 also guarantees idx[pos-1] (the "as known" bar) exists — a signal
    # predating the whole price history is untradable, not wrapped to the last bar.
    events = []
    for d in sorted(scores["date"].unique()):
        pos = idx.searchsorted(pd.Timestamp(d), side="right")
        if 1 <= pos < len(idx):
            events.append((pd.Timestamp(d), pos))
    if not events:
        raise ValueError("no executable rebalance dates inside the price history")

    long_book: set = set()
    short_book: set = set()
    w = pd.Series(dtype=float)          # current weights (NAV-relative at entry)
    entry_px = pd.Series(dtype=float)   # execution prices backing ``w``
    nav, nav_gross = 1.0, 1.0
    borrow_daily, wiped = 0.0, False
    navs: dict[pd.Timestamp, float] = {idx[events[0][1] - 1]: 1.0}   # inception point
    navs_gross: dict[pd.Timestamp, float] = {idx[events[0][1] - 1]: 1.0}
    rebal_rows = []

    for i, (d, pos) in enumerate(events):
        t_exec = idx[pos]
        day = scores[scores["date"] == d].set_index("ticker")
        rank_pct = day["pred"].rank(pct=True)
        # fill at the open; a missing open with a same-day close fills at the close
        # (a real print from AFTER the signal — never the stale prior close)
        exec_px = opn.loc[t_exec].fillna(close.loc[t_exec])
        dv_at = dv_med.loc[idx[pos - 1]]  # liquidity as known at the signal date

        # ---- close the old window: mark the old book at this execution's fill
        if len(w):
            mark = exec_px.reindex(w.index)
            mark = mark.fillna(valued.loc[idx[pos - 1]].reindex(w.index))  # dead: last print
            growth = (mark / entry_px).fillna(1.0)
            drifted = w * growth
            window_ret = float(drifted.sum() - w.sum())
            nights = pos - prev_pos
            nav *= (1.0 + window_ret) * (1.0 - borrow_daily) ** nights
            nav_gross *= 1.0 + window_ret
            if nav <= 0.0 or nav_gross <= 0.0:  # short squeeze past the equity: liquidated
                nav, nav_gross, wiped = max(nav, 0.0), max(nav_gross, 0.0), True
                break
            # drifted is in previous-NAV units; new targets are current-NAV fractions
            drifted = drifted / (1.0 + window_ret)
        else:
            drifted = pd.Series(dtype=float)

        # ---- select the new books
        if mode == "universe":
            new_long, new_short = set(rank_pct.index), set()
        elif mode == "long_short":
            shortable = rank_pct.index[dv_at.reindex(rank_pct.index).fillna(0.0) >= short_min_adv]
            new_long = hysteresis_members(rank_pct, long_book, enter, exit, "long")
            new_short = hysteresis_members(rank_pct.loc[shortable], short_book, enter, exit, "short")
        else:
            new_long = hysteresis_members(rank_pct, long_book, enter, exit, "long")
            new_short = set()

        # a name qualifying for BOTH tails (tiny cross-sections) is a contradictory
        # signal -> neither book
        overlap = new_long & new_short
        new_long, new_short = new_long - overlap, new_short - overlap

        # entries/holds need a same-day fill; a frozen (printless) name cannot stay
        new_long = {t for t in new_long if np.isfinite(exec_px.get(t, np.nan))}
        new_short = {t for t in new_short if np.isfinite(exec_px.get(t, np.nan))}

        w_new = pd.concat([
            _book_weights(new_long, rank_pct, weighting, +1.0),
            _book_weights(new_short, rank_pct, weighting, -1.0),
        ])

        # ---- trade: turnover + liquidity-scaled costs on traded notional
        both = w_new.index.union(drifted.index)
        delta = w_new.reindex(both, fill_value=0.0) - drifted.reindex(both, fill_value=0.0)
        tradable = pd.Series(
            [np.isfinite(exec_px.get(t, np.nan)) for t in both], index=both
        )
        traded = delta[tradable].abs()  # printless names release frozen value costlessly
        side_bps = tiered_bps(dv_at.reindex(traded.index), cost_tiers) * cost_scale
        cost = float((traded * side_bps).sum() / 1e4)
        turnover = float(delta.abs().sum() / 2.0)
        nav *= 1.0 - cost

        # ---- borrow accrual rate (per night) for the new window's short book
        if len(new_short):
            bps = tiered_bps(dv_at.reindex(sorted(new_short)), borrow_tiers)
            borrow_daily = float(
                (w_new.reindex(sorted(new_short)).abs() * bps).sum() / 1e4 / TRADING_DAYS
            )
        else:
            borrow_daily = 0.0

        w, entry_px, prev_pos = w_new, exec_px.reindex(w_new.index), pos
        long_book, short_book = new_long, new_short
        rebal_rows.append({
            "date": d, "exec_date": t_exec, "turnover": turnover, "cost": cost,
            "n_long": len(new_long), "n_short": len(new_short),
            "borrow_daily": borrow_daily,
        })

        # ---- daily marks inside the window (closes up to the next execution)
        end_pos = events[i + 1][1] if i + 1 < len(events) else len(idx)
        if len(w):
            px = valued.iloc[pos:end_pos].reindex(columns=w.index)
            rel = px.div(entry_px, axis=1).fillna(1.0)
            path = rel.mul(w, axis=1).sum(axis=1) - w.sum()
        else:
            path = pd.Series(0.0, index=idx[pos:end_pos])
        for k, (t, r_cum) in enumerate(path.items()):
            navs[t] = max(nav * (1.0 + r_cum) * (1.0 - borrow_daily) ** k, 0.0)
            navs_gross[t] = max(nav_gross * (1.0 + r_cum), 0.0)

    if wiped:  # liquidation: the account stays at zero for the rest of the sample
        for t in idx[idx > max(navs)]:
            navs[t] = nav
            navs_gross[t] = nav_gross

    return BacktestResult(
        nav=pd.Series(navs).sort_index(),
        nav_gross=pd.Series(navs_gross).sort_index(),
        rebalances=pd.DataFrame(rebal_rows),
        config=dict(mode=mode, enter=enter, exit=exit, weighting=weighting,
                    cost_scale=cost_scale, short_min_adv=short_min_adv,
                    trust_floor=trust_floor, n_masked_prints=n_masked,
                    n_break_days=n_break_days, n_break_names=n_break_names,
                    wiped_out=wiped),
    )
