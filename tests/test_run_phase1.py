"""The --no-impute gate path: flag parsing and its delistings=None contract."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_phase1", Path(__file__).resolve().parents[1] / "scripts" / "run_phase1.py"
)
run_phase1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_phase1)


def test_no_impute_flag_parses():
    assert run_phase1.parse_args(["--no-impute"]).no_impute is True
    assert run_phase1.parse_args([]).no_impute is False


def test_build_passes_delistings_through(monkeypatch):
    """--no-impute must reach build_fundamental_panel as delistings=None."""
    seen = {}

    def fake_build(feats, close, **kwargs):
        seen.update(kwargs)
        import pandas as pd
        return pd.DataFrame()

    monkeypatch.setattr(run_phase1, "build_fundamental_panel", fake_build)
    run_phase1._build("feats", "close", None, delistings=None, ticker_map={1: "A"})
    assert seen["delistings"] is None
    assert seen["ticker_map"] == {1: "A"}
