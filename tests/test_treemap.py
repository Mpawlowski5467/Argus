"""Pure tests for the squarified treemap layout + markup rasterizer."""

from stockscan.tui.treemap import fmt_cap, render_treemap, squarify, tile_at, tile_boxes


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


def test_render_treemap_smoke():
    items = [{"ticker": "NVDA", "cap": 3e12, "decile": 10},
             {"ticker": "AVGO", "cap": 1e12, "decile": 8},
             {"ticker": "MU", "cap": 1.2e11, "decile": 4}]
    out = render_treemap(items, width=40, height=12)
    assert out.count("\n") == 11                     # 12 rows
    assert "NVDA" in out and "AVGO" in out
    assert "on #" in out                             # background color markup present


def test_render_treemap_no_caps():
    assert "no market-cap" in render_treemap([{"ticker": "X", "cap": None, "decile": 5}], 40, 10)


def test_tile_boxes_and_hit_testing():
    caps = [50, 30, 20]
    boxes = tile_boxes(caps, 40, 20)
    assert len(boxes) == 3
    for b in boxes:                                  # boxes inside the grid
        if b:
            x0, y0, xe, ye = b
            assert 0 <= x0 < xe <= 40 and 0 <= y0 < ye <= 20
    b0 = boxes[0]                                     # center of tile 0 maps back to 0
    assert tile_at(boxes, (b0[0] + b0[2]) // 2, (b0[1] + b0[3]) // 2) == 0
    assert tile_at(boxes, 999, 999) is None          # off-grid -> nothing
    assert tile_at([None], 0, 0) is None


def test_fmt_cap_tiers():
    assert fmt_cap(2.94e12) == "$2.9T"
    assert fmt_cap(2.1e11) == "$210B"
    assert fmt_cap(None) == ""
