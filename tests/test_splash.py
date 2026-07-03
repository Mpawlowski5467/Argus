"""argus loading splash — the all-seeing eye that looks around + blinks.

Pure-frame tests (no textual) guarantee monospace alignment and that the gaze
actually moves the pupil / the blink shuts it; a headless app test proves the
overlay mounts, animates, and gets out of the way the moment data is ready.
"""

import asyncio

import pytest

from stockscan.tui.splash import (
    FRAMES, _GAZE, _H, _W, render_eye, scan_line,
)


def _pupil(frame: list[str]) -> tuple[int, int] | None:
    """(row, col) of the pupil glyph, or None if the eye is shut."""
    for r, line in enumerate(frame):
        c = line.find("◉")
        if c != -1:
            return r, c
    return None


# --- pure frame geometry --------------------------------------------------------

def test_every_frame_is_a_fixed_size_box():
    """Swapping any frame for any other must never shift the layout."""
    for gaze in _GAZE:
        for openness in (1.0, 0.5, 0.0):
            frame = render_eye(gaze, openness)
            assert len(frame) == _H
            assert {len(line) for line in frame} == {_W}


def test_choreography_frames_all_render_and_align():
    for gaze, openness in FRAMES:
        frame = render_eye(gaze, openness)
        assert len(frame) == _H and {len(line) for line in frame} == {_W}


def test_gaze_moves_the_pupil_left_right_and_up_down():
    _, cx = _pupil(render_eye("C"))
    _, lx = _pupil(render_eye("L"))
    _, rx = _pupil(render_eye("R"))
    assert lx < cx < rx                      # looks left / centre / right

    uy, _ = _pupil(render_eye("U"))
    cy, _ = _pupil(render_eye("C"))
    dy, _ = _pupil(render_eye("D"))
    assert uy < cy < dy                      # looks up / centre / down


def test_blink_shuts_the_eye_then_reopens():
    assert _pupil(render_eye("C", 1.0)) is not None   # open — pupil visible
    assert _pupil(render_eye("C", 0.5)) is None        # mid-blink — hidden
    assert _pupil(render_eye("C", 0.0)) is None        # shut — hidden


def test_choreography_actually_looks_around_and_blinks():
    gazes = {g for g, o in FRAMES if o >= 0.8}
    assert {"L", "R", "U", "D"} <= gazes               # it looks in every direction
    assert any(o < 0.2 for _, o in FRAMES)             # and it blinks fully shut


def test_scan_line_spins_and_stays_on_message():
    assert "scanning the market" in scan_line(0)
    assert scan_line(0) != scan_line(1)                # the spinner turns
    assert scan_line(0)[0] != " "                      # leads with a spinner glyph


# --- headless overlay lifecycle -------------------------------------------------

def _fake():
    pytest.importorskip("textual")
    from test_tui import FakeData
    return FakeData()


def test_splash_mounts_animates_then_dismisses():
    pytest.importorskip("textual")
    from stockscan.tui.app import ArgusApp, Splash

    async def scenario():
        app = ArgusApp(adata=_fake())
        app._min_splash = 5.0                          # hold the eye open to observe it
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            splash = app.query_one(Splash)
            assert splash.display is True              # covering the screen at boot
            assert splash.region.size == app.screen.region.size   # full-screen overlay
            assert "◉" in str(app.query_one("#splash-eye").render())

            t0 = splash._tick
            await pilot.pause(0.3)                     # the frame timer keeps ticking
            assert splash._tick > t0                    # it's alive (looking around)

            app._dismiss_splash(force=True)
            await pilot.pause()
            assert splash.display is False             # got out of the way
            assert splash._timer is None               # and stopped its timer

    asyncio.run(scenario())


def test_fast_load_still_holds_the_eye_for_the_minimum():
    """A load that finishes instantly must not flash the splash — it lingers for
    ``_min_splash`` (the real boot path, which dismisses via a timer)."""
    pytest.importorskip("textual")
    from stockscan.tui.app import ArgusApp, Splash

    async def scenario():
        app = ArgusApp(adata=_fake())
        app._min_splash = 0.25                         # data is "ready" at once
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            assert app.query_one(Splash).display is True    # still on-screen inside the window
            await pilot.pause(0.4)                          # wait past the minimum
            assert app.query_one(Splash).display is False   # then it bows out

    asyncio.run(scenario())


def test_boot_reveals_the_app_and_hides_the_splash():
    pytest.importorskip("textual")
    from textual.widgets import DataTable

    from stockscan.tui.app import ArgusApp, Splash

    async def scenario():
        app = ArgusApp(adata=_fake())                  # min_splash 0 → reveal at once
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            assert app.query_one(Splash).display is False           # dismissed
            assert app.query_one("#scan-table", DataTable).row_count > 0   # app usable

    asyncio.run(scenario())
