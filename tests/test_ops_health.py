"""Health check: critical failures exit non-zero, benign ones only warn."""


import pandas as pd

from stockscan.ops.health import Check, _quarter_end, report


def test_quarter_end():
    assert _quarter_end("2026q1") == pd.Timestamp("2026-03-31")
    assert _quarter_end("2026q2") == pd.Timestamp("2026-06-30")
    assert _quarter_end("2026q4") == pd.Timestamp("2026-12-31")


def test_report_exit_code_on_critical():
    ok = [Check("critical", "prices", True, ""), Check("warn", "llm", False, "")]
    text, code = report(ok)
    assert code == 0  # a failing WARN check does not fail the command
    bad = ok + [Check("critical", "artifact", False, "vintage drift")]
    text, code = report(bad)
    assert code == 1
    assert "FAIL" in text and "artifact" in text


def test_report_formats_all_levels():
    checks = [
        Check("critical", "prices", True, "fresh"),
        Check("warn", "matrix_cache", False, "stale"),
        Check("info", "llm", False, "down"),
    ]
    text, code = report(checks)
    assert code == 0
    assert "prices" in text and "matrix_cache" in text and "llm" in text
