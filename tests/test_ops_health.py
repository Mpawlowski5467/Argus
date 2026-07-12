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


def test_health_record_alerts_only_on_newly_failing_criticals():
    from stockscan.ops.health import health_record

    checks = [Check("critical", "prices", False, "latest bar 9d ago"),
              Check("critical", "artifact", False, "vintage drift"),
              Check("warn", "llm", False, "down"),
              Check("critical", "ops_state", True, "fine")]
    alerts = []

    def add(kind, msg):
        alerts.append((kind, msg))

    # first screen: both criticals are new -> two alerts; warn never alerts
    rec = health_record(checks, prev_failing=set(), add_alert=add)
    assert rec["critical_failing"] == ["artifact", "prices"]
    assert rec["_status"] == "degraded"
    assert [k for k, _ in alerts] == ["health_critical", "health_critical"]
    assert any("vintage drift" in m for _, m in alerts)

    # same failures next night: already known -> silence
    alerts.clear()
    health_record(checks, prev_failing={"artifact", "prices"}, add_alert=add)
    assert alerts == []

    # recovery then re-failure alerts again
    alerts.clear()
    health_record(checks, prev_failing={"artifact"}, add_alert=add)
    assert [m for _, m in alerts] == ["health: prices critical — latest bar 9d ago"]


def test_health_record_all_ok_is_clean():
    from stockscan.ops.health import health_record

    rec = health_record([Check("critical", "prices", True, "fresh")],
                        prev_failing={"prices"}, add_alert=lambda *a: 1 / 0)
    assert rec["critical_failing"] == [] and "_status" not in rec or rec.get("_status") == "ok"
    assert rec["checks"][0]["ok"] is True
