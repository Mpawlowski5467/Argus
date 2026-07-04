"""argus loading splash — the all-seeing eye, looking around and blinking.

Purely cosmetic and firewalled: this shows the wordmark + a big animated eye
while ``ArgusData.load()`` runs on its background thread. It touches no data and
no model — just characters on a timer. The eye is drawn on a fixed-width grid so
every frame lines up in a monospace terminal (asserted in the tests).
"""

from __future__ import annotations

# -- the eye canvas ------------------------------------------------------------
# A clean almond drawn from a hand-tuned outline (so it reads unmistakably as an
# eye at terminal resolution), with pointed ‹ › corners that echo the mandala
# logo. Every frame is the same _W × _H box so swapping frames never shifts the
# layout. The iris is a small sprite stamped inside the outline at a gaze offset.

# Fully-open eye, hollow (the iris is stamped in later). Kept symmetric so the
# gaze offsets stay balanced.
_OPEN = r"""
    .-‾‾‾‾‾‾‾‾‾‾‾‾‾-.
  .'                 '.
 /                     \
‹                       ›
 \                     /
  '.                 .'
    '-..._______...-'
"""

# Half-shut — the blink midpoint (upper and lower lids drawn together).
_HALF = r"""
   .-‾‾‾‾‾‾‾‾‾‾‾‾‾-.
 ‹                   ›
   '-..._______...-'
"""

# Shut — a single lash line with a few lashes, echoing the logo's dashes.
_SHUT = r"""
   '   '     '   '
 ‹━━━━━━━━━━━━━━━━━━━›
"""


def _block(art: str) -> list[str]:
    lines = [ln for ln in art.splitlines() if ln.strip()]
    w = max(len(ln) for ln in lines)
    return [ln.ljust(w) for ln in lines]


_OPEN_L, _HALF_L, _SHUT_L = _block(_OPEN), _block(_HALF), _block(_SHUT)
_W = max(len(_OPEN_L[0]), len(_HALF_L[0]), len(_SHUT_L[0])) + 2
_H = 7           # rows in a rendered frame (open eye height)
_MIDX = _W // 2  # horizontal centre column
_MIDY = _H // 2  # the iris' resting row

# how far the pupil may wander from centre (columns, rows)
_GAZE = {
    "C": (0, 0),
    "L": (-6, 0), "R": (6, 0), "U": (0, -2), "D": (0, 2),
    "UL": (-5, -2), "UR": (5, -2), "DL": (-5, 2), "DR": (5, 2),
}

_IRIS = ["_", "(◉)", "‾"]   # a small round iris; centre char is the pupil


def _canvas(block: list[str]) -> list[list[str]]:
    """Centre an outline block on a blank _W × _H grid of characters."""
    grid = [[" "] * _W for _ in range(_H)]
    top = (_H - len(block)) // 2
    for r, line in enumerate(block):
        left = (_W - len(line)) // 2
        for c, ch in enumerate(line):
            if ch != " ":
                grid[top + r][left + c] = ch
    return grid


def _interior(grid: list[list[str]]) -> list[tuple[int, int]]:
    """Per-row (left, right) columns of the outline, so the iris can be clipped
    to stay inside the eye. (-1, -1) for rows with no outline."""
    spans = []
    for row in grid:
        cols = [i for i, ch in enumerate(row) if ch != " "]
        spans.append((cols[0], cols[-1]) if cols else (-1, -1))
    return spans


def render_eye(gaze: str = "C", openness: float = 1.0) -> list[str]:
    """One eye frame as a list of ``_H`` strings, each ``_W`` wide.

    ``openness`` 1.0 = wide open, ~0.5 = mid-blink, 0.0 = shut. ``gaze`` indexes
    ``_GAZE`` and only applies while the eye is open enough to show the iris.
    """
    if openness < 0.2:
        return ["".join(r) for r in _canvas(_SHUT_L)]
    if openness < 0.8:
        return ["".join(r) for r in _canvas(_HALF_L)]

    grid = _canvas(_OPEN_L)
    spans = _interior(grid)
    gx, gy = _GAZE.get(gaze, (0, 0))
    cy = _MIDY + gy
    for dr, chunk in enumerate(_IRIS):        # rows: -1, 0, +1 around the pupil
        y = cy + dr - 1
        if not 0 <= y < _H:
            continue
        lo, hi = spans[y]
        if lo < 0:
            continue
        start = _MIDX + gx - (len(chunk) // 2)
        for c, ch in enumerate(chunk):
            x = start + c
            if lo < x < hi and grid[y][x] == " ":   # inside, and don't munge the lid
                grid[y][x] = ch
    return ["".join(r) for r in grid]


# -- choreography --------------------------------------------------------------
# A scripted, deterministic "looking around" loop with the odd blink. Each entry
# is (gaze, openness); the widget steps through these on a timer and loops.

def _hold(gaze: str, ticks: int) -> list[tuple[str, float]]:
    return [(gaze, 1.0)] * ticks


def _blink() -> list[tuple[str, float]]:
    # half → shut → shut → half → open: a quick, natural blink
    return [("C", 0.5), ("C", 0.0), ("C", 0.0), ("C", 0.5), ("C", 1.0)]


def _sequence() -> list[tuple[str, float]]:
    seq: list[tuple[str, float]] = []
    seq += _hold("C", 7)
    seq += _hold("L", 6) + _hold("C", 3) + _hold("R", 6) + _hold("C", 4)
    seq += _blink()
    seq += _hold("U", 5) + _hold("C", 3) + _hold("D", 5) + _hold("C", 4)
    seq += _hold("UR", 4) + _hold("DL", 4) + _hold("C", 3)
    seq += _blink()
    seq += _hold("UL", 4) + _hold("DR", 4) + _hold("C", 5)
    seq += _hold("R", 4) + _hold("L", 4) + _hold("C", 6)
    seq += _blink()
    return seq


FRAMES: list[tuple[str, float]] = _sequence()

# a little spinner + animated ellipsis for the "scanning" line
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def scan_line(tick: int) -> str:
    spin = _SPIN[tick % len(_SPIN)]
    dots = "." * (1 + (tick // 3) % 3)
    return f"{spin}  scanning the market{dots}"
