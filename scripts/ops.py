"""Continuous-operation CLI: every scheduled job and its manual override.

  uv run python scripts/ops.py nightly            # the launchd entry (see below)
  uv run python scripts/ops.py prices             # nightly price refetch only
  uv run python scripts/ops.py fsds               # ingest any missing FSDS quarter
  uv run python scripts/ops.py universe           # security-master refresh
  uv run python scripts/ops.py monitor [--no-llm] [--no-edgar]
  uv run python scripts/ops.py news [--no-llm]     # watchlist headline memory (live-view)
  uv run python scripts/ops.py watch add AAPL --note "core holding"
  uv run python scripts/ops.py watch ls | rm AAPL
  uv run python scripts/ops.py alerts [--all]
  uv run python scripts/ops.py paper freeze | log | compare
  uv run python scripts/ops.py paper retrain-record --reason "quarterly retrain 2026q3"
  uv run python scripts/ops.py health
  uv run python scripts/ops.py install-launchd [--dry-run] [--uninstall]

``nightly`` is the one entry the scheduler needs: prices -> FSDS (when a new
quarter is due) -> universe (when a month has passed) -> paper log (when a
completed month is unlogged) -> monitor -> news (watchlist headline memory,
firewalled from the signal). Every step is idempotent, so a run
missed while the machine slept simply catches up on the next firing; a single
repo-wide lock makes wake-coalesced double-fires and manual overlap harmless.
Deltas of every run land in ops_state.job_runs.
"""

import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from stockscan.config import LOGS_DIR, REPO_ROOT
from stockscan.ops.lock import JobAlreadyRunning, job_lock
from stockscan.ops.state import OpsState

LAUNCHD_LABEL = "com.stockscan.nightly"
NIGHTLY_HOUR, NIGHTLY_MINUTE = 22, 45
UNIVERSE_DUE_DAYS = 28


def _run_logged(state: OpsState, job: str, fn, *args, **kwargs) -> dict:
    """Run one job under the shared lock discipline, logging deltas + status."""
    run_id = state.job_start(job)
    try:
        deltas = fn(*args, **kwargs) or {}
    except Exception as exc:
        state.job_finish(run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    # honor an explicit _status, a {"status": "noop"} return (paper log's idempotent
    # no-op), or a {"noop": True} flag; default ok
    status = (deltas.pop("_status", None)
              or (deltas.get("status") if deltas.get("status") in ("noop", "degraded") else None)
              or ("noop" if deltas.get("noop") else "ok"))
    state.job_finish(run_id, status, deltas)
    print(f"[{job}] {status}: {json.dumps(deltas, default=str)[:600]}")
    return deltas


# --- individual jobs ---------------------------------------------------------------

def job_prices(state: OpsState) -> dict:
    from stockscan.ops.jobs import refresh_active_prices

    return _run_logged(state, "prices", refresh_active_prices)


def job_fsds(state: OpsState) -> dict:
    from stockscan.ops.jobs import ingest_new_fsds, missing_quarters

    if not missing_quarters():
        return _run_logged(state, "fsds", lambda: {"noop": True, "note": "up to date"})
    return _run_logged(state, "fsds", ingest_new_fsds)


def _universe_due(state: OpsState) -> bool:
    from stockscan.intrinio_universe import UNIVERSE_PATH

    last = state.last_run("universe", status="ok")
    if last is not None:
        anchor = pd.Timestamp(last["started"]).tz_localize(None)
    elif Path(UNIVERSE_PATH).exists():
        anchor = pd.Timestamp(os.stat(UNIVERSE_PATH).st_mtime, unit="s")
    else:
        return True
    return (pd.Timestamp.utcnow().tz_localize(None) - anchor).days >= UNIVERSE_DUE_DAYS


def job_universe(state: OpsState) -> dict:
    from stockscan.ops.jobs import refresh_universe

    return _run_logged(state, "universe", refresh_universe, state=state)


def _paper_due(state: OpsState) -> bool:
    from stockscan.config import PAPER_DIR
    from stockscan.panel import load_matrices_cached
    from stockscan.ops.paper import missing_paper_months

    if not (Path(PAPER_DIR) / "baseline.json").exists():
        return False
    close, _ = load_matrices_cached()
    if close.empty:
        return False
    return bool(missing_paper_months(close.index))


def job_paper_log(state: OpsState, as_of=None) -> dict:
    from stockscan.ops.paper import log_signals

    return _run_logged(state, "paper_log", log_signals, state, as_of=as_of)


def _backfill_missing_paper_months(state: OpsState) -> None:
    """Log EVERY completed month-end since the freeze that lacks a file (oldest
    first), so a multi-month outage leaves no permanent hole in the record."""
    from stockscan.ops.paper import missing_paper_months
    from stockscan.panel import load_matrices_cached

    close, _ = load_matrices_cached()
    months = missing_paper_months(close.index)
    if not months:
        print("[paper_log] up to date")
        return
    print(f"[paper_log] backfilling {len(months)} month(s): "
          f"{[str(m.date()) for m in months]}")
    for m in months:  # oldest first: the hysteresis book must be built in order
        job_paper_log(state, as_of=m)


def job_news(state: OpsState, no_llm: bool = False) -> dict:
    """Nightly watchlist news ingest (LIVE-VIEW ONLY — never scoring/backtest/panel).

    Fetch + dedup + extract headline/summary for every watched name into news.sqlite.
    Idempotent and quota-capped: the Intrinio pull is skipped for a name fetched inside
    NEWS_REFETCH_HOURS, so a re-run does no network work. Extraction (light tier, or the
    deterministic heuristic under --no-llm) backfills only articles missing the current
    version. Runs independent of price freshness — news is firewalled from the signal."""
    from stockscan.config import LLM_LIGHT_MODEL
    from stockscan.intrinio_universe import load_universe
    from stockscan.narrate.llm import LocalLLM
    from stockscan.newsmem import NewsStore, ingest_watchlist

    wl = state.watchlist()
    if not wl:
        return _run_logged(state, "news", lambda: {"noop": True, "note": "empty watchlist"})
    uni = load_universe()
    tmap = ((uni.sort_values("priority").drop_duplicates("cik")
             .set_index("cik")["ticker"].to_dict()) if len(uni) else {})
    ciks_tickers = [(int(w["cik"]), tmap.get(int(w["cik"]))) for w in wl]
    llm = None if no_llm else LocalLLM(model=LLM_LIGHT_MODEL)

    def _ingest() -> dict:
        with NewsStore() as store:
            return ingest_watchlist(store, ciks_tickers, llm=llm)

    return _run_logged(state, "news", _ingest)


def job_monitor(state: OpsState, no_llm: bool = False, edgar: bool = True,
                alerts_ok: bool = True) -> dict:
    from stockscan.narrate.llm import LocalLLM
    from stockscan.config import LLM_LIGHT_MODEL
    from stockscan.ops.monitor import run_monitor

    # A degraded price night (alerts_ok=False) also disables narration: a
    # cross-section built over a half-updated store has jittered percentiles, and
    # a full-tier narration would cache against that wrong percentile and reset the
    # materiality baseline. Template runs never cache, so no_llm here is safe.
    use_llm = not no_llm and alerts_ok
    llm_full = LocalLLM() if use_llm else None
    llm_light = LocalLLM(model=LLM_LIGHT_MODEL) if use_llm else None
    return _run_logged(state, "monitor", run_monitor, state,
                       llm_full=llm_full, llm_light=llm_light,
                       narrate=True, edgar=edgar, alerts_ok=alerts_ok)


def job_nightly(state: OpsState, no_llm: bool = False) -> int:
    """The scheduler entry: each stage self-checks whether it is due."""
    deltas = job_prices(state)
    checked = max(1, deltas.get("active_columns", 1))
    # degraded if too many columns failed OR the heartbeat column itself failed
    # (then freshness is unverifiable and the whole store may be a day behind)
    degraded = (deltas.get("failed", 0) / checked > 0.02
                or not deltas.get("reference_ok", True))

    job_fsds(state)

    if _universe_due(state):
        job_universe(state)
    else:
        print("[universe] not due")

    if _paper_due(state):
        _backfill_missing_paper_months(state)
    else:
        print("[paper_log] not due")

    # a degraded price night suppresses percentile alerts AND narration: ranks over
    # a half-updated store fire false alerts, then fire them in reverse on recovery
    job_monitor(state, no_llm=no_llm, alerts_ok=not degraded)
    # news ingest is firewalled from the signal, so a degraded price night doesn't
    # gate it; it just refreshes the watchlist's headline memory (quota-capped)
    job_news(state, no_llm=no_llm)
    run_id = state.job_start("nightly")
    state.job_finish(run_id, "degraded" if degraded else "ok",
                     {"prices_failed_frac": round(deltas.get("failed", 0) / checked, 4),
                      "reference_ok": deltas.get("reference_ok", True)})
    return 0


# --- launchd ------------------------------------------------------------------------

def _plist() -> dict:
    uv = shutil.which("uv") or "/usr/local/bin/uv"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [uv, "run", "python", "scripts/ops.py", "nightly"],
        "WorkingDirectory": str(REPO_ROOT),
        "StartCalendarInterval": {"Hour": NIGHTLY_HOUR, "Minute": NIGHTLY_MINUTE},
        "StandardOutPath": str(LOGS_DIR / "nightly.log"),
        "StandardErrorPath": str(LOGS_DIR / "nightly.err.log"),
        "ProcessType": "Background",
    }


def install_launchd(dry_run: bool = False, uninstall: bool = False) -> int:
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    if uninstall:
        if dry_run:
            print(f"would: launchctl bootout {domain}/{LAUNCHD_LABEL}; rm {plist_path}")
            return 0
        subprocess.run(["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"],
                       capture_output=True)
        plist_path.unlink(missing_ok=True)
        print(f"uninstalled {LAUNCHD_LABEL}")
        return 0
    payload = _plist()
    if dry_run:
        print(f"would write {plist_path}:\n{plistlib.dumps(payload).decode()}")
        print(f"would: launchctl bootstrap {domain} {plist_path}")
        return 0
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(plistlib.dumps(payload))
    subprocess.run(["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"],
                   capture_output=True)  # replace an older registration quietly
    res = subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(f"launchctl bootstrap failed ({res.returncode}): {res.stderr.strip()}\n"
              f"plist written to {plist_path}; load manually with:\n"
              f"  launchctl bootstrap {domain} {plist_path}")
        return 1
    print(f"installed + loaded {LAUNCHD_LABEL} "
          f"(daily {NIGHTLY_HOUR:02d}:{NIGHTLY_MINUTE:02d}, logs in {LOGS_DIR})")
    return 0


# --- CLI ------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prices")
    sub.add_parser("fsds")
    sub.add_parser("universe")
    p = sub.add_parser("monitor")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-edgar", action="store_true")
    p = sub.add_parser("news")
    p.add_argument("--no-llm", action="store_true", help="heuristic extraction only")
    p = sub.add_parser("nightly")
    p.add_argument("--no-llm", action="store_true")

    p = sub.add_parser("watch")
    p.add_argument("action", choices=["add", "rm", "ls"])
    p.add_argument("company", nargs="?", help="ticker, TICKER~CIK, or CIK")
    p.add_argument("--note", default="")

    p = sub.add_parser("alerts")
    p.add_argument("--all", action="store_true", help="include already-seen alerts")
    p.add_argument("--keep-unseen", action="store_true",
                   help="don't mark the listed alerts as seen")

    p = sub.add_parser("paper")
    p.add_argument("action", choices=["freeze", "log", "compare", "retrain-record"])
    p.add_argument("--as-of", default=None, help="(log) month-end override")
    p.add_argument("--reason", default="", help="(retrain-record) why the retrain")

    sub.add_parser("health")
    p = sub.add_parser("install-launchd")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--uninstall", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "health":
        from stockscan.ops.health import report, run_checks

        text, code = report(run_checks())
        print("stockscan health:")
        print(text)
        return code

    if args.cmd == "install-launchd":
        return install_launchd(dry_run=args.dry_run, uninstall=args.uninstall)

    # everything below touches shared state — one repo-wide lock
    try:
        with job_lock("ops"):
            with OpsState() as state:
                if args.cmd == "prices":
                    job_prices(state)
                elif args.cmd == "fsds":
                    job_fsds(state)
                elif args.cmd == "universe":
                    job_universe(state)
                elif args.cmd == "monitor":
                    job_monitor(state, no_llm=args.no_llm, edgar=not args.no_edgar)
                elif args.cmd == "news":
                    job_news(state, no_llm=args.no_llm)
                elif args.cmd == "nightly":
                    return job_nightly(state, no_llm=args.no_llm)
                elif args.cmd == "watch":
                    return cmd_watch(state, args)
                elif args.cmd == "alerts":
                    return cmd_alerts(state, args)
                elif args.cmd == "paper":
                    return cmd_paper(state, args)
        return 0
    except JobAlreadyRunning:
        print("another ops run is active; nothing to do (the lock holder is doing it)")
        return 0


def cmd_watch(state: OpsState, args) -> int:
    if args.action == "ls":
        rows = state.watchlist()
        if not rows:
            print("watchlist empty — add with: ops.py watch add TICKER")
        for w in rows:
            print(f"  cik {w['cik']:>8}  {w['column'] or '?':<14} "
                  f"added {w['added'][:10]}  {w['note']}")
        return 0
    if not args.company:
        print("watch add/rm needs a company")
        return 1
    from stockscan.intrinio_universe import universe_ticker_map
    from stockscan.serve import resolve_company

    cik, column = resolve_company(
        int(args.company) if args.company.isdigit() else args.company,
        universe_ticker_map())
    if args.action == "add":
        state.watch_add(cik, column, args.note)
        print(f"watching cik {cik} ({column or 'no price column'})")
    else:
        state.watch_remove(cik)
        print(f"removed cik {cik}")
    return 0


def cmd_alerts(state: OpsState, args) -> int:
    rows = state.alerts(unseen_only=not args.all)
    if not rows:
        print("no " + ("" if args.all else "unseen ") + "alerts")
        return 0
    for a in rows:
        mark = " " if a["seen"] else "*"
        print(f" {mark} [{a['id']:>4}] {a['created'][:16]}  {a['kind']:<22} {a['message']}")
    if not args.all and not args.keep_unseen:
        state.mark_alerts_seen([a["id"] for a in rows])
    return 0


def cmd_paper(state: OpsState, args) -> int:
    from stockscan.ops import paper

    if args.action == "freeze":
        res = paper.freeze_baseline()
        print(f"baseline {res['status']}: artifact {res['artifact']['hash']} "
              f"(trained through {res['artifact']['trained_through']}), "
              f"frozen {res['frozen_on']}")
        return 0
    if args.action == "log":
        if args.as_of:  # explicit month
            res = job_paper_log(state, as_of=args.as_of)
            return 0 if res.get("status") in ("logged", "noop") else 1
        _backfill_missing_paper_months(state)  # all completed months since the freeze
        return 0
    if args.action == "compare":
        rep = paper.compare()
        print(json.dumps(rep, indent=2, default=str))
        return 0
    if args.action == "retrain-record":
        entry = paper.record_retrain(args.reason)
        print(f"registered vintage {entry['hash']} "
              f"(trained through {entry['trained_through']}): {entry['reason']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
