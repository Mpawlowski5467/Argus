"""Paper-forward: freeze the product config, log live signals, compare to backtest.

DESIGN.md §6 calls this "the un-overfittable test": freeze model + thresholds
TODAY, log the live monthly signals to an append-only store, and after 3-6+
months compare live behavior with the backtest. Nothing in this module can
train — it loads the frozen artifact, scores, and writes.

Vintage discipline: every artifact the paper run is allowed to use must be
registered in ``vintages.jsonl`` (append-only). ``log_signals`` refuses an
artifact whose content hash isn't the latest registered vintage, so a silent
retrain (or an accidental artifact overwrite) halts the paper trail loudly
instead of contaminating it. Registering a new vintage (``record_retrain``) is
a manual CLI act — the quarterly retrain cadence in DESIGN.md §10 stays a
human decision, never an automated loop step.

Store layout (config.PAPER_DIR):
    baseline.json        — write-once freeze record (artifact, thresholds, the
                           backtest expectations the live run will be judged by)
    vintages.jsonl       — append-only artifact-vintage registry
    signals/<date>.jsonl — one immutable file per monthly run: a header line
                           (run metadata + cross-section stats), then one line
                           per name. Existing files are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    HYSTERESIS_ENTER,
    HYSTERESIS_EXIT,
    LABEL_HORIZON_DAYS,
    MAX_STALE_DAYS,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE,
    MIN_SECTOR_BUCKET,
    PAPER_DIR,
    REPO_ROOT,
)
from ..model import MODEL_DIR, Artifact, load_artifact
from ..panel import forward_return_to_last, month_end_dates

# What the live run is judged against — the honest backtest numbers this project
# actually produced (RESULTS.md: Phase-2 gate re-run for IC/spread, Phase-3 for
# the long-only net edge). Frozen into baseline.json so later readers see the
# expectation as it stood on freeze day, not as later edited.
BACKTEST_EXPECTATION = {
    "oos_rank_ic": 0.0391,
    "oos_t_nw": 5.92,
    "decile_spread_63d": 0.0229,
    "long_only_net_excess_per_yr": 0.0148,
    "source": "RESULTS.md Phase-1/2 gate (walk-forward OOS) + Phase-3 backtest",
}

# DESIGN.md §10: unscheduled retrain if live IC < ~1/2 the frozen backtest IC —
# threshold tuned here in Phase 5. Compare flags (never acts) after min_months.
DEGRADATION = {"live_ic_frac": 0.5, "min_months": 3}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def artifact_fingerprint(model_dir: Path = MODEL_DIR) -> str:
    """Content hash of the frozen artifact (booster bytes + metadata)."""
    model_dir = Path(model_dir)
    h = hashlib.sha256()
    h.update((model_dir / "model.txt").read_bytes())
    h.update((model_dir / "meta.json").read_bytes())
    return h.hexdigest()[:16]


def frozen_thresholds() -> dict:
    return {
        "min_dollar_volume": MIN_DOLLAR_VOLUME,
        "min_price": MIN_PRICE,
        "price_floor_basis": "adjusted_close",  # raw-close switch = a retrain event
        "max_stale_days": MAX_STALE_DAYS,
        "min_sector_bucket": MIN_SECTOR_BUCKET,
        "hysteresis_enter": HYSTERESIS_ENTER,   # enter inside the top 20% by score
        "hysteresis_exit": HYSTERESIS_EXIT,     # hold while inside the top 40%
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "book": "long-only (short book dropped per DESIGN §6 rule, RESULTS Phase-3)",
    }


# Live-vs-backtest asymmetries that are STRUCTURAL, known on freeze day, and must
# not be re-discovered as "signal decay" months from now (frozen into baseline.json).
KNOWN_ASYMMETRIES = [
    "FSDS publishes filings in quarterly batches: each live month-end misses "
    "filings filed since the last published quarter, which every backtest "
    "cross-section had (available_date <= d). Freshest-filing gap ~0-3 months.",
    "compare() is GROSS, close-to-close, cost-free — gate on the like-for-like "
    "IC/decile-spread pair only; the NAV numbers (net excess +1.48%/yr) are "
    "context, measured under next-bar-open execution with tiered costs.",
    "Live cross-sections use the frozen artifact; the backtest IC used purged "
    "walk-forward scores. Post-freeze months are genuinely OOS for the frozen "
    "artifact, so the comparison is fair only for months after trained_through "
    "+ 63 trading days (in_sample months are excluded from the gate metric).",
]


def _paths(paper_dir: Path):
    d = Path(paper_dir)
    return d / "baseline.json", d / "vintages.jsonl", d / "signals"


def current_vintage(paper_dir: Path = PAPER_DIR) -> dict | None:
    _, vintages_path, _ = _paths(paper_dir)
    if not vintages_path.exists():
        return None
    lines = [ln for ln in vintages_path.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def freeze_baseline(model_dir: Path = MODEL_DIR, paper_dir: Path = PAPER_DIR) -> dict:
    """Write-once freeze of the paper-forward configuration (idempotent).

    Re-running with the same artifact is a no-op; with a DIFFERENT artifact it
    raises — the baseline is the fixed reference point of the whole exercise,
    and replacing the artifact under it belongs to record_retrain instead.
    """
    baseline_path, vintages_path, signals_dir = _paths(paper_dir)
    fp = artifact_fingerprint(model_dir)
    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    if baseline_path.exists():
        existing = json.loads(baseline_path.read_text())
        if existing["artifact"]["hash"] == fp:
            return {**existing, "status": "noop"}
        raise RuntimeError(
            "baseline already frozen with a different artifact "
            f"({existing['artifact']['hash']} != {fp}); a retrain must go through "
            "'ops.py paper retrain-record', never through re-freezing"
        )
    baseline = {
        "frozen_on": _utcnow(),
        "artifact": {
            "hash": fp,
            "trained_through": meta["trained_through"],
            "feature_cols": meta["feature_cols"],
            "lightgbm_version": meta.get("lightgbm_version"),
            "n_rows": meta.get("n_rows"),
        },
        "thresholds": frozen_thresholds(),
        "backtest_expectation": BACKTEST_EXPECTATION,
        "degradation_rule": DEGRADATION,
        "known_asymmetries": KNOWN_ASYMMETRIES,
        "git_sha": _git_sha(),
    }
    signals_dir.mkdir(parents=True, exist_ok=True)
    tmp = baseline_path.with_name("." + baseline_path.name + ".tmp")
    tmp.write_text(json.dumps(baseline, indent=2))
    os.replace(tmp, baseline_path)
    with open(vintages_path, "a") as fh:
        fh.write(json.dumps({
            "registered": _utcnow(), "hash": fp,
            "trained_through": meta["trained_through"],
            "reason": "phase-5 freeze (paper-forward baseline)",
            "git_sha": baseline["git_sha"],
        }) + "\n")
    return {**baseline, "status": "frozen"}


def record_retrain(reason: str, model_dir: Path = MODEL_DIR,
                   paper_dir: Path = PAPER_DIR) -> dict:
    """Register a NEW artifact vintage — the manual, logged retrain event.

    Run scripts/train_model.py first, then this. Refuses to register the
    already-current vintage (nothing changed) and refuses to run before a
    baseline exists (nothing to version against).
    """
    baseline_path, vintages_path, _ = _paths(paper_dir)
    if not baseline_path.exists():
        raise RuntimeError("no baseline frozen yet; run 'ops.py paper freeze' first")
    if not reason.strip():
        raise ValueError("a retrain must state its reason (the honesty trail)")
    fp = artifact_fingerprint(model_dir)
    cur = current_vintage(paper_dir)
    if cur is not None and cur["hash"] == fp:
        raise RuntimeError(f"artifact {fp} is already the registered vintage")
    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    entry = {
        "registered": _utcnow(), "hash": fp,
        "trained_through": meta["trained_through"],
        "reason": reason, "git_sha": _git_sha(),
        "previous_hash": cur["hash"] if cur else None,
    }
    with open(vintages_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def _book_from_files(paper_dir: Path) -> dict[int, dict] | None:
    """Book membership per the LATEST signals file (None if none logged yet)."""
    _, _, signals_dir = _paths(paper_dir)
    files = sorted(signals_dir.glob("*.jsonl")) if signals_dir.exists() else []
    if not files:
        return None
    lines = files[-1].read_text().splitlines()
    book: dict[int, dict] = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("in_book"):
            book[int(r["cik"])] = {"column": r["column"],
                                   "entered_as_of": json.loads(lines[0])["as_of"]}
    return book


def _reconcile_book(state, paper_dir: Path) -> None:
    """Force the SQLite book cache to match the latest signals file exactly."""
    target = _book_from_files(paper_dir) or {}
    current = set(state.book())
    enters = {c: target[c]["column"] for c in set(target) - current}
    exits = current - set(target)
    if enters or exits:
        state.book_apply(enters, exits, "reconcile")


def completed_month_ends(index: pd.DatetimeIndex, today=None) -> list[pd.Timestamp]:
    """All month-end trading days for months that have fully ENDED (backtest grid)."""
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    return [d for d in month_end_dates(index) if (d.year, d.month) < (t.year, t.month)]


def default_as_of(index: pd.DatetimeIndex, today=None) -> pd.Timestamp:
    """Last trading day of the most recent COMPLETED calendar month.

    Matches the backtest's monthly sampling grid. Signals for a month are only
    loggable once that month has ended — as-of dates inside the running month
    would give the paper trail a cadence the backtest never had.
    """
    ends = completed_month_ends(index, today)
    if not ends:
        raise ValueError("price history has no completed month before today")
    return ends[-1]


def missing_paper_months(index: pd.DatetimeIndex, paper_dir: Path = PAPER_DIR,
                         today=None) -> list[pd.Timestamp]:
    """Completed month-ends since the freeze that have NO signals file (a gap).

    A missed launchd firing (Mac asleep/off for weeks) must not leave permanent
    holes in the append-only record: log_signals is PIT at as_of, so a late run
    scores the same cross-section a timely one would have. Oldest-first so the
    hysteresis book is rebuilt in order.
    """
    baseline_path, _, signals_dir = _paths(paper_dir)
    if not baseline_path.exists():
        return []
    frozen_on = pd.Timestamp(json.loads(baseline_path.read_text())["frozen_on"][:10])
    have = {p.stem for p in signals_dir.glob("*.jsonl")} if signals_dir.exists() else set()
    out = []
    for d in completed_month_ends(index, today):
        if d.normalize() < frozen_on.normalize():
            continue  # months before the freeze are out of scope
        if str(d.date()) not in have:
            out.append(d)
    return out


def _book_transitions(book: dict[int, dict], cross: pd.DataFrame) -> tuple[dict, set]:
    """Hysteresis membership (DESIGN §6): enter top 20%, hold until below top 40%.

    A held name that leaves the cross-section entirely (delisted, stale filer,
    liquidity fail) is exited — it can no longer be held on live data.
    """
    pct = dict(zip(cross["cik"].astype(int), cross["pct"]))
    col = dict(zip(cross["cik"].astype(int), cross["column"]))
    enters = {
        int(c): col[int(c)]
        for c in cross.loc[cross["pct"] >= 1.0 - HYSTERESIS_ENTER, "cik"]
        if int(c) not in book
    }
    exits = {
        c for c in book
        if c not in pct or pct[c] < 1.0 - HYSTERESIS_EXIT
    }
    return enters, exits


def log_signals(
    state,
    data=None,
    artifact: Artifact | None = None,
    as_of=None,
    paper_dir: Path = PAPER_DIR,
    model_dir: Path = MODEL_DIR,
    today=None,
) -> dict:
    """Monthly job: score the live cross-section and append it to the paper trail.

    Idempotent: a month already logged verifies its header and no-ops (the book
    transitions recorded in the file are re-applied to state, which is itself
    idempotent — so a crash between file write and book update self-heals).
    Never overwrites an existing signals file.
    """
    from ..serve import build_cross_section, load_serve_data

    baseline_path, _, signals_dir = _paths(paper_dir)
    if not baseline_path.exists():
        raise RuntimeError("no baseline frozen; run 'ops.py paper freeze' first")
    fp = artifact_fingerprint(model_dir)
    vintage = current_vintage(paper_dir)
    if vintage is None or vintage["hash"] != fp:
        raise RuntimeError(
            f"artifact {fp} is not the registered vintage "
            f"({vintage['hash'] if vintage else 'none'}); if this was a deliberate "
            "retrain, register it with 'ops.py paper retrain-record' first"
        )

    data = data or load_serve_data()
    artifact = artifact or load_artifact(model_dir)
    as_of = pd.Timestamp(as_of) if as_of is not None else default_as_of(data.close.index, today)
    path = signals_dir / f"{as_of.date()}.jsonl"

    if path.exists():
        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        if header["artifact_hash"] != fp:
            raise RuntimeError(
                f"signals for {as_of.date()} were logged under artifact "
                f"{header['artifact_hash']}, current is {fp} — paper trail mismatch"
            )
        # crash-heal: reconcile the SQLite book cache to the LATEST signals file's
        # membership (the authoritative record) — NOT this month's transitions,
        # which would regress the book if an older month is re-logged after a newer.
        _reconcile_book(state, paper_dir)
        return {"status": "noop", "as_of": str(as_of.date()), "path": str(path),
                "n": header["stats"]["n"]}

    cross = build_cross_section(data, as_of).reset_index(drop=True)
    scores = artifact.score(cross)
    cross["score"] = scores
    cross["pct"] = cross["score"].rank(pct=True)
    cross["decile"] = np.clip(np.ceil(cross["pct"] * 10), 1, 10).astype(int)
    cross["column"] = cross["ticker"]

    # The signals FILES are the authoritative book record; SQLite is a cache that
    # a crash can leave stale. Reconstructing membership from the latest file
    # (when one exists) makes the two impossible to diverge across crashes.
    prev_book = _book_from_files(paper_dir) or state.book()
    enters, exits = _book_transitions(prev_book, cross)
    in_book = (set(prev_book) - exits) | set(enters)

    # in-sample flag mirrors serve.analyze: the training information window
    # extends label_horizon trading days past trained_through.
    idx = data.close.index
    horizon = int(artifact.meta.get("label_horizon_days", LABEL_HORIZON_DAYS))
    loc = idx.searchsorted(artifact.trained_through, side="right") - 1
    info_through = idx[min(loc + horizon, len(idx) - 1)] if loc >= 0 else artifact.trained_through

    from .jobs import quarters_present

    q = cross["score"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(5)
    logged_at = _utcnow()
    quarters = quarters_present()
    header = {
        "as_of": str(as_of.date()),
        "logged_at": logged_at,
        # A late run sees richer fundamentals than an on-time run would have
        # (FSDS batches) — same as_of, different information set. Flagged, not
        # hidden, so compare() readers can weigh backfilled months accordingly.
        "late": bool((pd.Timestamp(logged_at[:10]) - as_of).days > 7),
        "artifact_hash": fp,
        "trained_through": artifact.meta["trained_through"],
        "in_sample": bool(as_of <= info_through),
        "data_vintage": {
            "prices_max_date": str(data.close.index[-1].date()),
            "fsds_latest_quarter": quarters[-1] if quarters else None,
            "max_filed_date_ingested": str(pd.Timestamp(data.feats["filed_date"].max()).date()),
        },
        "thresholds": frozen_thresholds(),
        "git_sha": _git_sha(),
        "stats": {
            "n": int(len(cross)),
            "n_top_decile": int((cross["decile"] == 10).sum()),
            "n_book": len(in_book),
            "n_entered": len(enters),
            "n_exited": len(exits),
            "score_quantiles": {str(k): float(v) for k, v in q.items()},
            "n_sectors": int(cross["sector"].nunique()),
        },
    }

    signals_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(header) + "\n")
        for _, r in cross.iterrows():
            cik = int(r["cik"])
            fh.write(json.dumps({
                "cik": cik,
                "column": r["column"],
                "sector": r["sector"] if pd.notna(r["sector"]) else None,
                "score": round(float(r["score"]), 6),
                "pct": round(float(r["pct"]), 4),
                "decile": int(r["decile"]),
                "top_decile": bool(r["decile"] == 10),
                "in_book": cik in in_book,
                "entered": cik in enters,
                "exited": cik in exits,
                "filed_date": str(pd.Timestamp(r["filed_date"]).date()),
                "available_date": str(pd.Timestamp(r["available_date"]).date()),
            }) + "\n")
        # exited names may already be absent from the cross-section — record them
        for cik in sorted(exits - set(cross["cik"].astype(int))):
            fh.write(json.dumps({
                "cik": int(cik), "column": prev_book[cik]["column"], "sector": None,
                "score": None, "pct": None, "decile": None, "top_decile": False,
                "in_book": False, "entered": False, "exited": True,
                "filed_date": None, "available_date": None,
            }) + "\n")
    os.replace(tmp, path)
    state.book_apply(enters, exits, str(as_of.date()))
    return {"status": "logged", "as_of": str(as_of.date()), "path": str(path),
            **header["stats"], "in_sample": header["in_sample"]}


def _read_signals(paper_dir: Path) -> list[tuple[dict, pd.DataFrame]]:
    _, _, signals_dir = _paths(paper_dir)
    out = []
    for path in sorted(signals_dir.glob("*.jsonl")):
        lines = path.read_text().splitlines()
        if not lines:
            continue
        header = json.loads(lines[0])
        rows = pd.DataFrame([json.loads(ln) for ln in lines[1:] if ln.strip()])
        out.append((header, rows))
    return out


WINSOR = (0.01, 0.99)   # same per-date label clipping as the training panel / gate


def compare(
    close: pd.DataFrame | None = None,
    paper_dir: Path = PAPER_DIR,
    ticker_map: dict[int, str] | None = None,
    horizons: tuple[int, ...] = (21, 63),
) -> dict:
    """Backtest-vs-live: realized outcomes of every logged month old enough to score.

    Methodology mirrors EXACTLY how the baseline numbers were measured (like-for-
    like): the label is the forward return over RAW, UNREPAIRED prices with the
    terminal last-trade convention (build_fundamental_panel / the Phase-1/2 gate
    use load_matrices output directly), then per-month 1/99 winsorization before
    demeaning — the winsorization, not scale-break repair, is what tames the
    artifacts on the label side. Applying the backtest's NAV-side hygiene here
    would MASK sub-penny death prints to NaN and let forward_return_to_last's
    ffill carry the pre-crash price forward, fabricating a ~0% return for a name
    that actually died to zero — flattering the live track exactly where honesty
    matters most (found in code review).

    Names are re-keyed by CIK through the CURRENT universe map, so a column
    renamed by death (XYZ -> XYZ~CIK) still resolves — the crashed names must
    not be the ones that silently drop out. Months flagged in_sample are
    excluded from the gate metric and reported separately. Flags (never acts
    on) the DESIGN §10 degradation rule.
    """
    from ..intrinio_universe import universe_ticker_map
    from ..panel import load_matrices_cached

    baseline_path, _, _ = _paths(paper_dir)
    if not baseline_path.exists():
        raise RuntimeError("no baseline frozen; nothing to compare against")
    baseline = json.loads(baseline_path.read_text())
    if close is None:
        close, _ = load_matrices_cached()
    tmap = ticker_map if ticker_map is not None else universe_ticker_map()

    idx = close.index
    months = []
    fwd_cache: dict[int, pd.DataFrame] = {}
    for header, rows in _read_signals(paper_dir):
        as_of = pd.Timestamp(header["as_of"])
        if as_of not in idx:
            loc = idx.searchsorted(as_of, side="right") - 1
            if loc < 0:
                continue
            as_of = idx[loc]
        loc = idx.get_loc(as_of)
        scored = rows[rows["score"].notna()].copy()
        scored["column_now"] = [
            tmap.get(int(c), col) for c, col in zip(scored["cik"], scored["column"])
        ]
        entry: dict = {"as_of": header["as_of"], "n": len(scored),
                       "in_sample": header.get("in_sample"),
                       "late": header.get("late", False)}
        for h in horizons:
            if loc + h > len(idx) - 1:
                entry[f"h{h}"] = None  # not enough trading days elapsed yet
                continue
            if h not in fwd_cache:
                fwd_cache[h] = forward_return_to_last(close, h)
            r = scored["column_now"].map(fwd_cache[h].loc[as_of])
            ok = r.notna()
            if ok.sum() < 30:
                entry[f"h{h}"] = None
                continue
            rr = r[ok].clip(r[ok].quantile(WINSOR[0]), r[ok].quantile(WINSOR[1]))
            excess = rr - rr.mean()
            score_r = scored.loc[ok, "score"].rank()
            excess_r = excess.rank()
            # a degenerate flat cross-section has no rank variance -> IC undefined
            if score_r.std() == 0 or excess_r.std() == 0:
                entry[f"h{h}"] = None
                continue
            ic = float(score_r.corr(excess_r))
            top = scored.loc[ok, "decile"] == 10
            bot = scored.loc[ok, "decile"] == 1
            book = scored.loc[ok, "in_book"]
            entry[f"h{h}"] = {
                "n_priced": int(ok.sum()),
                "rank_ic": round(ic, 4),
                "decile_spread": round(float(excess[top].mean() - excess[bot].mean()), 5)
                if top.any() and bot.any() else None,
                "top_decile_excess": round(float(excess[top].mean()), 5) if top.any() else None,
                "book_excess": round(float(excess[book].mean()), 5) if book.any() else None,
            }
        months.append(entry)

    h_main = max(horizons)
    gate = [m for m in months if m.get(f"h{h_main}") and not m.get("in_sample")]
    in_sample = [m for m in months if m.get(f"h{h_main}") and m.get("in_sample")]
    gate_ics = [m[f"h{h_main}"]["rank_ic"] for m in gate]
    report = {
        "baseline": {
            "expected_ic": baseline["backtest_expectation"]["oos_rank_ic"],
            "expected_spread_63d": baseline["backtest_expectation"]["decile_spread_63d"],
            "frozen_on": baseline["frozen_on"],
        },
        "label_basis": "raw unrepaired prices + 1/99 per-month winsorization "
                       "(matches the frozen baseline IC methodology)",
        "months_logged": len(months),
        "months_scored_oos": len(gate_ics),
        "months_scored_in_sample": len(in_sample),
        "live_mean_ic": round(float(np.mean(gate_ics)), 4) if gate_ics else None,
        "live_mean_spread": round(float(np.mean(
            [m[f"h{h_main}"]["decile_spread"] for m in gate
             if m[f"h{h_main}"]["decile_spread"] is not None])), 5) if gate_ics else None,
        "months": months,
    }
    rule = baseline.get("degradation_rule", DEGRADATION)
    if len(gate_ics) >= rule["min_months"]:
        floor = rule["live_ic_frac"] * baseline["backtest_expectation"]["oos_rank_ic"]
        report["degraded"] = bool(np.mean(gate_ics) < floor)
        report["degradation_floor"] = round(floor, 4)
    else:
        report["degraded"] = None
        report["note"] = (f"needs {rule['min_months']} out-of-sample scored months "
                          f"before the degradation rule applies")
    return report


def month_close_report(month: dict, rep: dict, h_main: int = 63) -> str:
    """One scoreable month rendered as a markdown scorecard — every number verbatim
    from :func:`compare` (nothing recomputed here), caveats carried inline so the
    report can't read stronger than the record it summarizes."""
    as_of = str(month.get("as_of", ""))[:10]
    base = rep.get("baseline") or {}
    lines = [
        f"# Paper-forward month close — {as_of[:7]}",
        "",
        f"- signal date: {as_of} · {month.get('n')} names logged"
        + (" · **IN-SAMPLE** (inside the frozen training window — shown for "
           "completeness, EXCLUDED from the gate)" if month.get("in_sample") else "")
        + (" · logged late" if month.get("late") else ""),
    ]
    for h, tag in ((21, "h21 early read"), (h_main, f"h{h_main} (gate horizon)")):
        s = month.get(f"h{h}")
        if not s:
            lines.append(f"- {tag}: not yet scoreable")
            continue
        parts = [f"rank IC {s['rank_ic']}", f"n_priced {s['n_priced']}"]
        if s.get("decile_spread") is not None:
            parts.append(f"decile spread {s['decile_spread']}")
        if s.get("book_excess") is not None:
            parts.append(f"book excess {s['book_excess']}")
        lines.append(f"- {tag}: " + " · ".join(parts))
    lines += [
        "",
        f"## Running gate ({rep.get('months_scored_oos')} OOS month(s) scored)",
        "",
        f"- live mean IC: {rep.get('live_mean_ic')} vs backtest expectation "
        f"{base.get('expected_ic')} (baseline frozen {str(base.get('frozen_on', ''))[:10]})",
        (f"- degradation: floor {rep.get('degradation_floor')} — "
         + ("**DEGRADED**" if rep.get("degraded") else "on track"))
        if rep.get("degraded") is not None else f"- degradation: {rep.get('note')}",
        "",
        "## Caveats (fixed, not tuned per month)",
        "",
        "- One month of IC is mostly noise — the gate requires "
        "3 out-of-sample months for a reason; no conclusion is drawn here.",
        "- The h21 read uses a shorter horizon than the model was gated on; "
        "it is an early peek, never the verdict.",
        f"- Label basis: {rep.get('label_basis')}.",
        "- Returns are gross close-to-close — no cost or borrow modeling on "
        "the live side.",
    ]
    return "\n".join(lines) + "\n"


def write_month_reports(rep: dict, paper_dir: Path = PAPER_DIR,
                        h_main: int = 63) -> dict:
    """One ``reports/YYYY-MM.md`` per scoreable month, refreshed whenever the running
    gate numbers change (each report embeds them). Idempotent: unchanged text is not
    rewritten; months not yet scoreable are counted, never stubbed."""
    out_dir = Path(paper_dir) / "reports"
    written, unchanged, pending = [], 0, 0
    for month in rep.get("months") or []:
        # a report exists from the first scoreable horizon (the h21 early peek) and
        # is refreshed in place when h63 grades — the caveats already frame h21
        if not (month.get("h21") or month.get(f"h{h_main}")):
            pending += 1
            continue
        name = f"{str(month.get('as_of', ''))[:7]}.md"
        text = month_close_report(month, rep, h_main=h_main)
        path = out_dir / name
        if path.exists() and path.read_text() == text:
            unchanged += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
        written.append(name)
    return {"written": written, "unchanged": unchanged, "pending": pending}


def paper_progress_alerts(state, rep: dict) -> dict:
    """Turn a :func:`compare` report into ALERTS on progress, so the whole
    paper-forward exercise reaches the user instead of waiting to be looked at.

    Two events matter: a NEW out-of-sample month graded (the honest test gained a
    data point — the first one is the moment the freeze starts earning trust), and
    the degradation verdict FLIPPING either way. Previous counts live in the job
    log (the last ``paper_check`` run's deltas), so this is idempotent across
    re-runs and wake-coalesced double-fires. First run seeds silently, matching
    the filing-detection bootstrap convention."""
    prev = state.last_run("paper_check")
    prev_d = (prev or {}).get("deltas") or {}
    n_prev = int(prev_d.get("months_scored_oos") or 0)
    n_now = int(rep.get("months_scored_oos") or 0)
    deltas: dict = {
        "months_scored_oos": n_now,
        "degraded": rep.get("degraded"),
        "live_mean_ic": rep.get("live_mean_ic"),
        "alerts": 0,
    }
    if prev is None:                      # bootstrap: record state, alert on changes
        deltas["note"] = "seeded"
        return deltas

    if n_now > n_prev:
        exp = (rep.get("baseline") or {}).get("expected_ic")
        state.add_alert(
            "paper_graded",
            f"paper-forward: {n_now - n_prev} new out-of-sample month(s) graded "
            f"({n_now} total) — live mean IC {rep.get('live_mean_ic')} vs backtest "
            f"expectation {exp}",
            payload={"months_scored_oos": n_now,
                     "live_mean_ic": rep.get("live_mean_ic"), "expected_ic": exp})
        deltas["alerts"] += 1

    deg, prev_deg = rep.get("degraded"), prev_d.get("degraded")
    if deg is not None and deg != prev_deg:
        floor = rep.get("degradation_floor")
        if deg:
            msg = (f"paper-forward DEGRADED: live mean IC {rep.get('live_mean_ic')} "
                   f"fell below the degradation floor {floor} — the frozen model is "
                   f"underperforming its backtest expectation out of sample")
        else:
            msg = (f"paper-forward recovered: live mean IC {rep.get('live_mean_ic')} "
                   f"back above the degradation floor {floor}")
        state.add_alert("paper_degraded" if deg else "paper_recovered", msg,
                        payload={"degraded": deg, "live_mean_ic": rep.get("live_mean_ic"),
                                 "floor": floor})
        deltas["alerts"] += 1
    return deltas
