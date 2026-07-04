"""A terminal treemap: rectangles with area proportional to value, packed to keep
tiles near-square (the squarified algorithm of Bruls, Huizing & van Wijk, 2000).

Used by the markets map to draw each market's names sized by live market cap and
colored by the model signal. Pure + deterministic: ``squarify`` computes float
rectangles, ``render_treemap`` rasterizes them to Rich markup for a Static. No
Textual, no I/O — unit-testable with plain lists.
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


# -- rasterize to Rich markup ------------------------------------------------------

def _heat(decile) -> str:
    """Model-signal background: bright green (top decile) → dim → red (bottom)."""
    d = decile or 0
    if d >= 9:
        return "#1a7f37"
    if d >= 7:
        return "#238636"
    if d >= 5:
        return "#57606a"
    if d >= 3:
        return "#b62324"
    return "#8b1a1a"


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


def tile_boxes(caps, width: int = 76, height: int = 20) -> list:
    """Integer tile boxes ``(x0, y0, xend, yend)`` aligned with ``caps`` (largest first).

    The box is the filled (gutter-inset) region — the same cells ``render_treemap``
    paints — so a mouse cell inside a box maps to that tile. ``None`` for a tile too
    small to draw. Pair this with the SAME cap list the renderer used.
    """
    rects = squarify(caps, 0, 0, width, height)
    boxes = []
    for (rx, ry, rw, rh) in rects:
        x0, y0 = max(0, int(round(rx))), max(0, int(round(ry)))
        x1, y1 = min(width, int(round(rx + rw))), min(height, int(round(ry + rh)))
        if x1 <= x0 or y1 <= y0:
            boxes.append(None)
            continue
        xe = x1 - 1 if x1 - x0 > 1 else x1   # one-cell gutter on right/bottom
        ye = y1 - 1 if y1 - y0 > 1 else y1
        boxes.append((x0, y0, xe, ye))
    return boxes


def tile_at(boxes, x: int, y: int):
    """Index of the tile whose box contains cell ``(x, y)``, or None."""
    for i, b in enumerate(boxes):
        if b and b[0] <= x < b[2] and b[1] <= y < b[3]:
            return i
    return None


def render_treemap(items, width: int = 76, height: int = 20) -> str:
    """Rich-markup treemap of ``items`` = dicts with ``cap``/``ticker``/``decile``.

    Tiles are sized by ``cap`` (largest first), colored by ``decile``, labelled with
    the ticker (and cap when the tile is tall enough). Returns one markup string of
    ``height`` lines, each ``width`` cells wide.
    """
    items = [it for it in items if it.get("cap")]
    if not items:
        return "[dim]no market-cap data for this market[/dim]"
    boxes = tile_boxes([it["cap"] for it in items], width, height)
    bg = [[None] * width for _ in range(height)]
    txt = [[" "] * width for _ in range(height)]

    def place(row, x0, xend, s):
        if not s or row < 0 or row >= height:
            return
        s = s[: max(0, xend - x0)]
        start = x0 + max(0, ((xend - x0) - len(s)) // 2)
        for k, ch in enumerate(s):
            if 0 <= start + k < xend:
                txt[row][start + k] = ch

    for box, it in zip(boxes, items):
        if box is None:
            continue
        x0, y0, xe, ye = box
        color = _heat(it.get("decile"))
        for yy in range(y0, ye):
            for xx in range(x0, xe):
                bg[yy][xx] = color
        # only label a tile wide enough to hold a ticker — tiny tail tiles stay
        # colored-only rather than showing confusing 1-2 char fragments
        if xe - x0 >= 3:
            mid = y0 + (ye - y0) // 2
            place(mid, x0, xe, str(it.get("ticker") or ""))
            if ye - y0 >= 3 and xe - x0 >= 4:
                place(mid + 1, x0, xe, fmt_cap(it.get("cap")))

    lines = []
    for y in range(height):
        out, run_bg, run = [], "\0", ""
        for xcell in range(width):
            cell_bg = bg[y][xcell]
            if cell_bg != run_bg:
                if run:
                    out.append(run if run_bg is None else f"[#ffffff on {run_bg}]{run}[/]")
                run_bg, run = cell_bg, txt[y][xcell]
            else:
                run += txt[y][xcell]
        if run:
            out.append(run if run_bg is None else f"[#ffffff on {run_bg}]{run}[/]")
        lines.append("".join(out))
    return "\n".join(lines)
