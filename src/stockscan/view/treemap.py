"""Squarified treemap layout (Bruls, Huizing & van Wijk, 2000): rectangles with
area proportional to value, packed to keep tiles near-square.

The web markets map calls ``squarify`` for its tile geometry (the browser paints
them). Pure + deterministic, unit-testable with plain lists. ``fmt_cap`` is the
shared human market-cap formatter.
"""

from __future__ import annotations

# -- squarified layout -------------------------------------------------------------

def _layout_row(sizes, x, y, dx, dy, horizontal):
    """Place one row/column of tiles; return their (x, y, dx, dy) rects."""
    covered = sum(sizes)
    rects = []
    if horizontal:                       # a column of tiles filling height dy
        width = covered / dy if dy else 0
        for s in sizes:
            h = s / width if width else 0
            rects.append((x, y, width, h))
            y += h
    else:                                # a row of tiles filling width dx
        height = covered / dx if dx else 0
        for s in sizes:
            w = s / height if height else 0
            rects.append((x, y, w, height))
            x += w
    return rects


def _worst(sizes, x, y, dx, dy, horizontal):
    rects = _layout_row(sizes, x, y, dx, dy, horizontal)
    ratios = [max(w / h, h / w) for (_, _, w, h) in rects if w > 0 and h > 0]
    return max(ratios) if ratios else float("inf")


def _leftover(sizes, x, y, dx, dy, horizontal):
    covered = sum(sizes)
    if horizontal:
        width = covered / dy if dy else 0
        return (x + width, y, dx - width, dy)
    height = covered / dx if dx else 0
    return (x, y + height, dx, dy - height)


def squarify(values, x=0.0, y=0.0, width=1.0, height=1.0):
    """Rectangles (x, y, w, h) for ``values`` (descending), area ∝ value.

    Rectangles tile the ``width × height`` box exactly and never overlap. Values
    must be positive and pre-sorted largest-first for the best aspect ratios.
    """
    vals = [float(v) for v in values if v and v > 0]
    if not vals:
        return []
    total = sum(vals)
    scale = (width * height) / total
    sizes = [v * scale for v in vals]
    return _squarify(sizes, x, y, width, height)


def _squarify(sizes, x, y, dx, dy):
    if not sizes:
        return []
    horizontal = dx >= dy
    length = dy if horizontal else dx
    if len(sizes) == 1:
        return _layout_row(sizes, x, y, dx, dy, horizontal)
    i = 1
    while (i < len(sizes)
           and _worst(sizes[:i], x, y, dx, dy, horizontal)
           >= _worst(sizes[: i + 1], x, y, dx, dy, horizontal)):
        i += 1
    row, rest = sizes[:i], sizes[i:]
    lx, ly, ldx, ldy = _leftover(row, x, y, dx, dy, horizontal)
    return _layout_row(row, x, y, dx, dy, horizontal) + _squarify(rest, lx, ly, ldx, ldy)


def fmt_cap(x) -> str:
    if x is None:
        return ""
    if x >= 1e12:
        return f"${x / 1e12:.1f}T"
    if x >= 1e9:
        return f"${x / 1e9:.0f}B"
    if x >= 1e6:
        return f"${x / 1e6:.0f}M"
    return f"${x:,.0f}"
