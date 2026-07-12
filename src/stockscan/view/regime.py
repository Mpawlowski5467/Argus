"""Market-level context from the price matrix — DISPLAY-ONLY, three plain facts.

Breadth (share of liquid names above their own 200-day mean), the cross-sectional
median of trailing 21-day volatility with its percentile against the last ~5 years,
and the equal-weight universe's drawdown from its running high. Deliberately NOT a
"regime" classifier: naming regimes invites acting on them, and there is no OOS
evidence the model's edge varies by regime (that per-regime IC split is the study
that would have to come FIRST — see the roadmap). Never a model feature, never a
confidence modifier — a header line for the human, computed from prices the system
already holds.
"""

from __future__ import annotations

import math

import pandas as pd

MA_WINDOW = 200          # the classic breadth anchor
VOL_WINDOW = 21          # ~1 trading month
HISTORY_DAYS = 1260      # ~5y of context for the vol percentile


def compute_regime(close: pd.DataFrame, tickers=None, as_of=None) -> dict | None:
    """Three market-context reads as of the last bar <= ``as_of``.

    ``tickers`` restricts to the liquid universe the scan actually shows (the full
    matrix carries thousands of dead/illiquid columns whose breadth is meaningless).
    Returns None when there isn't enough history to say anything honest."""
    if close is None or close.empty:
        return None
    px = close
    if tickers is not None:
        cols = [t for t in tickers if t in px.columns]
        if not cols:
            return None
        px = px[cols]
    if as_of is not None:
        px = px.loc[:pd.Timestamp(as_of)]
    px = px.tail(HISTORY_DAYS)
    if len(px) < MA_WINDOW + VOL_WINDOW:
        return None

    as_of_date = px.index[-1]
    last = px.iloc[-1]

    # breadth: names above their own 200d mean, among names with a full window
    ma = px.rolling(MA_WINDOW).mean().iloc[-1]
    have = last.notna() & ma.notna()
    breadth = float((last[have] > ma[have]).mean()) if have.any() else None

    rets = px.pct_change(fill_method=None)
    # per-date cross-sectional median of trailing vol, annualized — a series, so
    # the latest value gets an honest percentile against its own history
    med_vol = (rets.rolling(VOL_WINDOW).std()
               .median(axis=1) * math.sqrt(252)).dropna()
    vol_now = float(med_vol.iloc[-1]) if len(med_vol) else None
    vol_pctile = (int(round(float((med_vol <= med_vol.iloc[-1]).mean()) * 100))
                  if len(med_vol) >= MA_WINDOW else None)

    # equal-weight universe drawdown from its running high inside the window
    eq = (1.0 + rets.mean(axis=1).fillna(0.0)).cumprod()
    dd = float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1.0)

    return {
        "as_of": str(pd.Timestamp(as_of_date).date()),
        "n_names": int(have.sum()),
        "breadth_above_200d": round(breadth, 4) if breadth is not None else None,
        "median_vol_ann": round(vol_now, 4) if vol_now is not None else None,
        "vol_pctile_5y": vol_pctile,
        "ew_drawdown": round(dd, 4),
        "note": ("market context for the human reader — computed from trailing "
                 "prices only; not a model input, not a forecast, and deliberately "
                 "not a named 'regime'"),
    }


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def regime_line(r: dict | None) -> str:
    """The one-line rendering the UI shows (kept here so tests pin the wording)."""
    if not r:
        return ""
    parts = []
    if r.get("breadth_above_200d") is not None:
        parts.append(f"{r['breadth_above_200d'] * 100:.0f}% of names above their 200d")
    if r.get("median_vol_ann") is not None:
        v = f"median vol {r['median_vol_ann'] * 100:.0f}%/yr"
        if r.get("vol_pctile_5y") is not None:
            v += f" ({_ordinal(r['vol_pctile_5y'])} pctile, 5y)"
        parts.append(v)
    if r.get("ew_drawdown") is not None:
        dd = r["ew_drawdown"]
        parts.append("equal-weight universe at its high" if dd > -0.005
                     else f"equal-weight universe {abs(dd) * 100:.0f}% off its high")
    return " · ".join(parts)
