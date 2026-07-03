"""Health check: is the unattended machinery actually healthy right now?

Every check returns (level, name, ok, detail). ``critical`` failures exit
non-zero (data stale, artifact drift, corrupt stores); ``warn`` failures are
reported but don't fail the command (LLM down is fine — narration degrades to
template by design). The command is cheap enough to run ad hoc or from cron.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import (
    ARTIFACTS_DIR,
    HEALTH_FSDS_GRACE_DAYS,
    HEALTH_PRICE_STALE_DAYS,
    LLM_BASE_URL,
    OPS_STATE_PATH,
    PAPER_DIR,
)
from ..prices import PRICES_DIR


@dataclass
class Check:
    level: str          # 'critical' | 'warn' | 'info'
    name: str
    ok: bool
    detail: str


def _quarter_end(quarter: str) -> pd.Timestamp:
    y, q = int(quarter[:4]), int(quarter[-1])
    return pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)


def run_checks(today=None, prices_dir: Path = PRICES_DIR) -> list[Check]:
    from ..model import MODEL_DIR
    from ..panel import matrix_cache_fresh, matrix_cache_paths
    from .jobs import latest_elapsed_quarter, quarters_present
    from .paper import artifact_fingerprint, current_vintage

    t = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    checks: list[Check] = []

    # prices freshness — via the matrix-cache meta when fresh, else a direct file
    _, _, meta_p = matrix_cache_paths()
    max_date = None
    if meta_p.exists():
        try:
            max_date = pd.Timestamp(json.loads(meta_p.read_text())["max_date"])
        except Exception:
            max_date = None
    if max_date is None:
        ref = sorted(Path(prices_dir).glob("A*.parquet"))
        if ref:
            max_date = pd.read_parquet(ref[0], columns=["date"])["date"].max()
    if max_date is None:
        checks.append(Check("critical", "prices", False, "no price data found"))
    else:
        age = (t.normalize() - pd.Timestamp(max_date).normalize()).days
        checks.append(Check(
            "critical", "prices", age <= HEALTH_PRICE_STALE_DAYS,
            f"latest bar {pd.Timestamp(max_date).date()} ({age}d ago; "
            f"stale after {HEALTH_PRICE_STALE_DAYS}d)"))

    checks.append(Check(
        "warn", "matrix_cache", matrix_cache_fresh(prices_dir=prices_dir),
        "wide-matrix cache in sync with the per-column store"
        if matrix_cache_fresh(prices_dir=prices_dir)
        else "stale/missing — loaders fall back to the ~2min slow path"))

    # fundamentals recency
    quarters = quarters_present()
    latest_have = quarters[-1] if quarters else None
    expected = latest_elapsed_quarter(t)
    if latest_have == expected:
        checks.append(Check("critical", "fundamentals", True, f"{latest_have} ingested"))
    else:
        overdue = t > _quarter_end(expected) + pd.Timedelta(days=HEALTH_FSDS_GRACE_DAYS)
        checks.append(Check(
            "critical" if overdue else "info", "fundamentals", not overdue,
            f"latest ingested {latest_have}, latest elapsed {expected}"
            + ("" if overdue else " (inside the FSDS publication window)")))

    # artifact + vintage discipline
    try:
        fp = artifact_fingerprint(MODEL_DIR)
        vintage = current_vintage()
        if vintage is None:
            checks.append(Check("warn", "artifact", True,
                                f"artifact {fp} present; no paper baseline frozen yet"))
        else:
            ok = vintage["hash"] == fp
            checks.append(Check(
                "critical", "artifact", ok,
                f"artifact {fp} == registered vintage" if ok else
                f"artifact {fp} != registered vintage {vintage['hash']} — "
                f"unregistered retrain or corrupted artifact"))
    except FileNotFoundError:
        checks.append(Check("critical", "artifact", False, "no frozen artifact on disk"))

    baseline = Path(PAPER_DIR) / "baseline.json"
    checks.append(Check("warn", "paper_baseline", baseline.exists(),
                        "frozen" if baseline.exists() else
                        "not frozen — run 'ops.py paper freeze'"))

    # paper cadence: every completed month since the freeze should have a file
    if baseline.exists():
        signals = sorted((Path(PAPER_DIR) / "signals").glob("*.jsonl"))
        frozen_on = pd.Timestamp(json.loads(baseline.read_text())["frozen_on"][:10])
        prev_month_end = (t.normalize().replace(day=1) - pd.Timedelta(days=1))
        due = prev_month_end >= frozen_on.normalize()
        have_prev = any(pd.Timestamp(p.stem).to_period("M") == prev_month_end.to_period("M")
                        for p in signals)
        checks.append(Check(
            "warn", "paper_signals", (not due) or have_prev,
            f"{len(signals)} month(s) logged"
            + ("" if (not due) or have_prev else
               f"; previous month ({prev_month_end.to_period('M')}) missing — "
               f"nightly will backfill")))

    # job recency (only meaningful once the scheduler has run at least once)
    try:
        from .state import OpsState

        with OpsState(OPS_STATE_PATH) as st:
            last = st.last_run("nightly")
        if last is None:
            checks.append(Check("info", "nightly_job", True, "never run yet"))
        else:
            age_h = (pd.Timestamp.utcnow().tz_localize(None)
                     - pd.Timestamp(last["started"]).tz_localize(None)).total_seconds() / 3600
            ok = age_h <= 48 and last["status"] in ("ok", "noop", "degraded")
            checks.append(Check("warn", "nightly_job", ok,
                                f"last {last['status']} {age_h:.0f}h ago"))
    except sqlite3.Error as exc:
        checks.append(Check("critical", "ops_state", False, f"state DB unreadable: {exc}"))

    # narration cache openable
    try:
        con = sqlite3.connect(str(ARTIFACTS_DIR / "narration_cache.sqlite"), timeout=5.0)
        con.execute("select count(*) from sqlite_master")
        con.close()
        checks.append(Check("warn", "narration_cache", True, "openable"))
    except sqlite3.Error as exc:
        checks.append(Check("warn", "narration_cache", False, str(exc)))

    # LLM endpoint (informational — template fallback is by design)
    try:
        import httpx

        r = httpx.get(LLM_BASE_URL.rstrip("/v1") + "/api/tags", timeout=3.0)
        checks.append(Check("info", "llm", r.status_code == 200,
                            f"{LLM_BASE_URL} reachable" if r.status_code == 200
                            else f"status {r.status_code}"))
    except Exception:
        checks.append(Check("info", "llm", False,
                            f"{LLM_BASE_URL} unreachable — narration falls back to template"))

    free_gb = shutil.disk_usage(str(ARTIFACTS_DIR)).free / 1e9
    checks.append(Check("warn", "disk", free_gb > 5.0, f"{free_gb:.1f} GB free"))
    return checks


def report(checks: list[Check]) -> tuple[str, int]:
    """Human-readable table + exit code (1 if any critical check failed)."""
    lines = []
    worst = 0
    for c in checks:
        mark = "OK " if c.ok else ("FAIL" if c.level == "critical" else "warn")
        lines.append(f"  [{mark:>4}] {c.level:<8} {c.name:<16} {c.detail}")
        if not c.ok and c.level == "critical":
            worst = 1
    return "\n".join(lines), worst
