"""Pure tests for the squarified treemap layout + the market-cap formatter."""

from stockscan.view.treemap import fmt_cap, squarify


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


def test_squarify_covers_box_and_is_area_proportional():
    vals = [50, 25, 15, 10]
    rects = squarify(vals, 0, 0, 100, 100)
    assert len(rects) == 4
    for (x, y, w, h) in rects:                       # every tile inside the box
        assert x >= -1e-6 and y >= -1e-6
        assert x + w <= 100 + 1e-6 and y + h <= 100 + 1e-6
        assert w > 0 and h > 0
    area = sum(w * h for _, _, w, h in rects)
    assert abs(area - 100 * 100) < 1e-3              # tiles fill the box
    for (_, _, w, h), v in zip(rects, vals):         # area proportional to value
        assert abs((w * h) / 10000 - v / sum(vals)) < 1e-6


def test_squarify_tiles_do_not_overlap():
    rects = squarify([40, 30, 20, 10, 5], 0, 0, 80, 50)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert _overlap(rects[i], rects[j]) < 1e-6


def test_squarify_ignores_nonpositive_and_empty():
    assert squarify([]) == []
    assert squarify([0, None, -5]) == []
    assert len(squarify([10, 0, 5], 0, 0, 10, 10)) == 2   # only the positive values


def test_fmt_cap_tiers():
    assert fmt_cap(2.94e12) == "$2.9T"
    assert fmt_cap(2.1e11) == "$210B"
    assert fmt_cap(None) == ""
