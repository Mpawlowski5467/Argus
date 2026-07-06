"""Single source of truth for the trailing price-change horizons (1w / 1m / 3m / 1y).

The ticker page's trailing-change chips are read by three surfaces that must agree on
this set or they silently drift apart:

  - ``view.chart.price_summary`` computes each ``chg_<key>`` from a TRADING-day
    lookback (~5 / 21 / 63 / 252 sessions back);
  - ``assist.move`` windows the COINCIDING news / filings for "explain this move" by a
    CALENDAR-day span (trading days plus weekend / holiday slack — generous, never
    tight);
  - ``static/app.js`` renders the chips, in this order.

This table is the one place all three read from, so a horizon added, dropped, or
retuned here moves every surface together. ``tests/test_horizons.py`` guards the two
Python consumers structurally and pins ``app.js`` (which cannot import this) to the
same keys, so the third copy can't drift unnoticed either.

Kept dependency-free on purpose (only the stdlib) so both ``view`` and ``assist`` can
import it without any risk of an import cycle.
"""

from __future__ import annotations

from typing import NamedTuple


class Horizon(NamedTuple):
    key: str            # chip id + price_summary field suffix ("1w" -> chg_1w)
    label: str          # plain-English name ("one-week")
    window: str         # prose window handed to the LLM ("the last week of trading")
    trading_days: int   # price_summary lookback, in trading sessions
    calendar_days: int  # move.py coincidence window, in calendar days


HORIZONS: tuple[Horizon, ...] = (
    Horizon("1w", "one-week", "the last week of trading", 5, 10),
    Horizon("1m", "one-month", "the last month of trading", 21, 38),
    Horizon("3m", "three-month", "the last three months of trading", 63, 100),
    Horizon("1y", "one-year", "the last year of trading", 252, 372),
)

KEYS: tuple[str, ...] = tuple(h.key for h in HORIZONS)
BY_KEY: dict[str, Horizon] = {h.key: h for h in HORIZONS}
