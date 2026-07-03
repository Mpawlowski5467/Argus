"""Our own terminal price chart — a dependency-free braille line plot.

Braille gives 2x4 sub-cell resolution (each character is a 2-wide by 4-tall dot
grid), so a compact block of cells draws a surprisingly high-resolution line with
nothing but Unicode — no matplotlib, no image, pure terminal. Everything here is a
pure function over a list/Series of prices so it unit-tests without Textual.

Also here: ``price_summary`` (last / change / 52-week range) and ``verdict`` (the
deterministic BUY/HOLD call), both computed in code — the LLM never sets these.
"""

from __future__ import annotations

import math

BRAILLE_BASE = 0x2800
# dot bitmask for a sub-pixel at (x in {0,1}, y in {0,1,2,3}); y=0 is the TOP row.
#   standard braille dot numbering →   1 4 / 2 5 / 3 6 / 7 8
_DOTS = (
    (0x01, 0x08),   # y=0  (dot1, dot4)
    (0x02, 0x10),   # y=1  (dot2, dot5)
    (0x04, 0x20),   # y=2  (dot3, dot6)
    (0x40, 0x80),   # y=3  (dot7, dot8)
)


def _finite(values) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _resample(values: list[float], n: int) -> list[float]:
    """Bucket-average ``values`` down (or step up) to exactly ``n`` points."""
    m = len(values)
    if m == n:
        return list(values)
    if m < n:  # step-interpolate up
        return [values[int(i * m / n)] for i in range(n)]
    out = []  # average each bucket down
    for i in range(n):
        a, b = int(i * m / n), int((i + 1) * m / n)
        b = max(b, a + 1)
        chunk = values[a:b]
        out.append(sum(chunk) / len(chunk))
    return out


def braille_chart(values, width: int = 60, height: int = 12,
                  lo: float | None = None, hi: float | None = None) -> list[str]:
    """Render ``values`` as ``height`` lines of ``width`` braille characters.

    A continuous line: consecutive samples are joined vertically so there are no
    gaps. ``lo``/``hi`` pin the y-range (default = data min/max). Returns a list of
    text rows (no color — the caller adds Rich markup)."""
    vals = _finite(values)
    if not vals:
        return [" " * width for _ in range(height)]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = (hi - lo) or 1.0
    W, H = width * 2, height * 4
    samples = _resample(vals, W)

    grid = bytearray(width * height)

    def ypix(v: float) -> int:
        y = int(round((hi - v) / span * (H - 1)))
        return 0 if y < 0 else H - 1 if y > H - 1 else y

    prev = None
    for x in range(W):
        y = ypix(samples[x])
        run = (y,) if prev is None else range(min(prev, y), max(prev, y) + 1)
        for yy in run:
            cx, cy = x // 2, yy // 4
            grid[cy * width + cx] |= _DOTS[yy % 4][x % 2]
        prev = y

    return ["".join(chr(BRAILLE_BASE + grid[cy * width + cx]) for cx in range(width))
            for cy in range(height)]


def _fmt(v: float) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:,.2f}"
    return f"{v:.4f}"


def price_chart(values, width: int = 60, height: int = 12, color: str = "cyan") -> str:
    """Braille chart with a left price axis (hi/mid/lo) — Rich markup string."""
    vals = _finite(values)
    if len(vals) < 2:
        return "[dim]— not enough price history to chart —[/dim]"
    hi, lo = max(vals), min(vals)
    mid = (hi + lo) / 2
    rows = braille_chart(vals, width=width, height=height, lo=lo, hi=hi)
    labels = {0: _fmt(hi), height // 2: _fmt(mid), height - 1: _fmt(lo)}
    gutter = max((len(s) for s in labels.values()), default=6)
    out = []
    for i, row in enumerate(rows):
        tag = labels.get(i, "")
        out.append(f"[dim]{tag:>{gutter}}[/dim] │[{color}]{row}[/{color}]")
    axis = f"{'':>{gutter}} └" + "─" * width
    out.append(axis)
    return "\n".join(out)


_VBLOCKS = "▁▂▃▄▅▆▇█"   # 1/8 .. 8/8, for the volume histogram


def _bucket_ohlcv(o, h, l, c, v, width):
    """Aggregate parallel OHLCV lists into <=width candles (open=first, high=max, …)."""
    n = len(c)
    if n == 0:
        return [], [], [], [], []
    if n <= width:
        spans = [(i, i + 1) for i in range(n)]
    else:
        cuts = [int(i * n / width) for i in range(width + 1)]
        spans = [(cuts[i], cuts[i + 1]) for i in range(width) if cuts[i] < cuts[i + 1]]
    O = [o[a] for a, b in spans]
    H = [max(h[a:b]) for a, b in spans]
    L = [min(l[a:b]) for a, b in spans]
    C = [c[b - 1] for a, b in spans]
    V = [sum(v[a:b]) for a, b in spans]
    return O, H, L, C, V


def candle_panel(o, h, l, c, v, width: int = 60, height: int = 12, vheight: int = 3) -> str:
    """Our own terminal candlestick chart + a volume histogram, as a Rich-markup string.

    Each column is one candle: a thin wick spanning high→low and a block body spanning
    open→close, green when the bucket closed up and red when it closed down. Below sits a
    block-glyph volume histogram on the same buckets. Pure/testable (plain number lists in)."""
    O, H, L, C, V = _bucket_ohlcv(list(o), list(h), list(l), list(c), list(v), width)
    if not C:
        return "[dim]— no OHLC to chart —[/dim]"
    hi, lo = max(H), min(L)
    span = (hi - lo) or 1.0
    w = len(C)

    def row(p):
        r = int(round((hi - p) / span * (height - 1)))
        return 0 if r < 0 else height - 1 if r > height - 1 else r

    color = ["green" if C[x] >= O[x] else "red" for x in range(w)]
    grid = [[" "] * w for _ in range(height)]
    for x in range(w):
        hr, lr = row(H[x]), row(L[x])
        for r in range(hr, lr + 1):
            grid[r][x] = "│"                     # wick
        bt, bb = sorted((row(O[x]), row(C[x])))
        for r in range(bt, bb + 1):
            grid[r][x] = "█"                     # body (overrides wick)

    labels = {0: _fmt(hi), height // 2: _fmt((hi + lo) / 2), height - 1: _fmt(lo)}
    gutter = max(len(s) for s in labels.values())

    def paint(cells, cols):
        return "".join(f"[{cols[x]}]{cells[x]}[/{cols[x]}]" if cells[x] != " " else " "
                       for x in range(len(cells)))

    out = []
    for r in range(height):
        out.append(f"[dim]{labels.get(r, ''):>{gutter}}[/dim] │{paint(grid[r], color)}")

    maxv = max(V) or 1.0
    vgrid = [[" "] * w for _ in range(vheight)]
    for x in range(w):
        level = int(round(V[x] / maxv * (vheight * 8)))
        for rr in range(vheight):
            take = min(8, level)
            level -= take
            if take:
                vgrid[vheight - 1 - rr][x] = _VBLOCKS[take - 1]
    for r in range(vheight):
        out.append(f"[dim]{'vol' if r == 0 else '':>{gutter}}[/dim] │{paint(vgrid[r], color)}")
    out.append(f"{'':>{gutter}} └" + "─" * w)
    return "\n".join(out)


def _pct(new: float, old: float) -> float | None:
    if old in (None, 0) or new is None or not math.isfinite(old) or not math.isfinite(new):
        return None
    return (new / old - 1.0) * 100.0


def price_summary(series, adv: float | None = None) -> dict:
    """Last price, trailing % changes, and the 52-week range from a close Series.

    ``series`` is a pandas Series (date-indexed) or any ordered sequence; NaNs are
    dropped. ~252 trading days ≈ 1 year, 63 ≈ 3 months, 21 ≈ 1 month.
    """
    try:
        vals = _finite(series.tolist())  # pandas Series
    except AttributeError:
        vals = _finite(list(series))
    if not vals:
        return {"last": None, "chg_1m": None, "chg_3m": None, "chg_1y": None,
                "hi_52w": None, "lo_52w": None, "adv": adv, "n": 0}
    last = vals[-1]
    yr = vals[-252:] if len(vals) >= 2 else vals

    def back(n):
        return vals[-n - 1] if len(vals) > n else vals[0]

    return {
        "last": last,
        "chg_1m": _pct(last, back(21)),
        "chg_3m": _pct(last, back(63)),
        "chg_1y": _pct(last, back(252)),
        "hi_52w": max(yr),
        "lo_52w": min(yr),
        "adv": adv,
        "n": len(vals),
    }


# --- the deterministic call -----------------------------------------------------
# Long-only per the Phase-3 verdict (the short book was dropped: it died to borrow).
# Hysteresis from the backtest: enter the book in the top quintile, hold to the 40th
# percentile. We expose a single-snapshot read of that rule (there is no prior-state
# here), plus the raw percentile so nothing is hidden.

def verdict(percentile: float, decile: int | None = None) -> dict:
    """BUY / HOLD / AVOID from the cross-sectional percentile (0..1). Deterministic."""
    if percentile is None or not math.isfinite(percentile):
        return {"call": "N/A", "color": "dim", "reason": "no score"}
    pct = float(percentile)
    if pct >= 0.80:
        return {"call": "BUY", "color": "green",
                "reason": f"top-quintile signal ({round(pct * 100)}th pct) — enters the long book"}
    if pct >= 0.40:
        return {"call": "HOLD", "color": "yellow",
                "reason": f"mid signal ({round(pct * 100)}th pct) — hold if owned, no new buy"}
    return {"call": "AVOID", "color": "red",
            "reason": f"bottom signal ({round(pct * 100)}th pct) — long-only, so no position"}
