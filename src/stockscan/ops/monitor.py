"""The nightly monitoring loop: watchlist signals, filings, alerts, re-narration.

One pass over the watchlist against the latest point-in-time cross-section:

- percentile-move alerts when a name's frozen-model percentile moved by
  MONITOR_PCTILE_ALERT or more since the last recorded state;
- new-filing detection from two sources: the wide fundamentals table (numbers
  landed — triggers re-narration via the materiality gate, which treats a new
  period_end as material) and EDGAR's live submissions feed (filed, numbers
  arrive with the next FSDS batch — early warning only);
- re-narration through the EXISTING materiality-gated cache (scan.py's exact
  pattern: analyze(llm=None) builds the packet, narrate_smart decides cache /
  light / full — a template run never touches the cache);
- 8-K filing flags over the watchlist AND held names: item codes (4.02 non-reliance,
  3.01 delisting notice, 4.01 auditor change) from the same submissions JSON, plus
  EDGAR full-text phrase searches ("going concern", "material weakness", …) —
  display/alert only, firewalled from the signal like the risk heads.

The loop only observes: load_artifact + score, never fit. When price ingestion
was degraded (transient vendor failures above a small threshold), percentile
alerts and signal-state updates are suppressed for the night — cross-sectional
ranks over a half-updated store would fire false alerts and then fire them
again in reverse on the recovery night.
"""

from __future__ import annotations

import pandas as pd

from ..config import MONITOR_PCTILE_ALERT
from ..edgar.client import EdgarClient
from .state import OpsState

# forms whose arrival matters to a fundamentals-driven watchlist
_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A")


def detect_wide_filings(state: OpsState, feats: pd.DataFrame, ciks: list[int]) -> list[dict]:
    """New (cik, period_end, filed_date) rows in the wide table for watched names.

    First sight of a cik seeds silently (bootstrap): everything already on disk
    predates the watch, alerting on it would be noise. Idempotent via the
    known_filings primary key.
    """
    news = []
    for cik in ciks:
        sub = feats[feats["cik"] == cik]
        if sub.empty:
            continue
        rows = [
            {"cik": int(cik), "form": "10-K", "filed_date": str(pd.Timestamp(fd).date()),
             "period_end": str(pd.Timestamp(pe).date()) if pd.notna(pe) else "",
             "source": "fsds"}
            for fd, pe in zip(sub["filed_date"], sub["period_end"])
        ]
        bootstrap = not state.has_filings(cik, source="fsds")
        fresh = state.add_filings(rows)
        if not bootstrap:
            news.extend(fresh)
    return news


def fetch_submissions(ciks, client: EdgarClient) -> dict[int, dict]:
    """One submissions JSON per name through the throttled client, fetched ONCE a night
    and shared by every EDGAR detector. An unreachable company is skipped, never fatal."""
    out: dict[int, dict] = {}
    for cik in ciks:
        try:
            out[int(cik)] = client.get_json(
                f"{client.DATA_HOST}/submissions/CIK{int(cik):010d}.json")
        except Exception:
            continue  # one unreachable company must not kill the loop
    return out


def detect_edgar_filings(state: OpsState, ciks: list[int],
                         client: EdgarClient | None = None,
                         submissions: dict[int, dict] | None = None) -> list[dict]:
    """Filings visible on EDGAR that FSDS hasn't delivered yet (early warning).

    Uses the throttled submissions API (recent filings only) — or the night's
    already-fetched ``submissions``. Bootstrap seeding is silent, same as the
    wide-table path.
    """
    if not ciks:
        return []
    if submissions is None:
        own = client is None
        client = client or EdgarClient()
        try:
            submissions = fetch_submissions(ciks, client)
        finally:
            if own:
                client.close()
    news = []
    for cik in ciks:
        data = submissions.get(int(cik))
        if data is None:
            continue
        recent = data.get("filings", {}).get("recent", {})
        rows = [
            {"cik": int(cik), "form": form, "filed_date": filed,
             "period_end": period or "", "source": "edgar"}
            for form, filed, period in zip(
                recent.get("form", []), recent.get("filingDate", []),
                recent.get("reportDate", []))
            if form in _FORMS
        ]
        bootstrap = not state.has_filings(cik, source="edgar")
        fresh = state.add_filings(rows)
        if not bootstrap:
            news.extend(fresh)
    return news


# --- 8-K filing flags (display/alert only — FIREWALLED from the signal) ------------
# Accounting/listing trouble in a watched or HELD name's 8-K. Writes the filing_flags
# table and alerts — never signal_state, the packet, narration context or any score.

_FLAG_CURSOR = "filing_flags_cursor"
_EFTS_PAGE = 100            # the backend's fixed page size


def alert_universe(state: OpsState, watch: list[dict] | None = None) -> list[int]:
    """Names the filing checks cover: the watchlist plus every HELD name (shares > 0).
    Positions decide membership only — they never feed the score."""
    watch = state.watchlist() if watch is None else watch
    ciks = [int(w["cik"]) for w in watch]
    try:
        ciks += [int(p["cik"]) for p in state.positions() if float(p.get("shares") or 0) > 0]
    except Exception:
        pass
    return list(dict.fromkeys(ciks))


def detect_item_flags(cik: int, submission: dict | None, rules) -> list[dict]:
    """8-K / 8-K/A filings whose item codes hit an ``item`` rule — read from the
    submissions JSON the monitor already fetched (``items`` is comma-joined per filing)."""
    recent = ((submission or {}).get("filings") or {}).get("recent") or {}
    item_rules = [r for r in rules if r.get("item")]
    if not item_rules or not recent.get("items"):
        return []
    rows = []
    for form, filed, items, acc in zip(recent.get("form", []), recent.get("filingDate", []),
                                       recent.get("items", []),
                                       recent.get("accessionNumber", [])):
        if not str(form).startswith("8-K"):
            continue
        codes = sorted({c.strip() for c in str(items or "").split(",") if c.strip()})
        for r in item_rules:
            if r["item"] in codes:
                rows.append({"cik": int(cik), "accession": acc, "rule": r["id"],
                             "source": "item", "form": form, "file_date": filed,
                             "items": codes, "docs": [form]})
    return rows


def search_text_flags(client: EdgarClient, ciks, rule: dict, startdt: str, enddt: str,
                      chunk: int = 100, max_pages: int = 100) -> tuple[list[dict], int]:
    """One ``q`` rule's 8-K full-text matches in the window, collapsed from DOCUMENTS to
    filings (an exhibit can match on its own). Results are relevance-ordered, so every
    page is read. Returns (rows, requests made); raises on an unexpected response shape."""
    universe = {int(c) for c in ciks}
    found: dict[tuple[int, str], dict] = {}
    requests = 0
    ordered = sorted(universe)
    for i in range(0, len(ordered), chunk):
        part, offset = ordered[i:i + chunk], 0
        for _ in range(max_pages):
            res = client.full_text_search(rule["q"], forms=("8-K",), ciks=part,
                                          startdt=startdt, enddt=enddt, from_=offset)
            requests += 1
            block = res.get("hits") if isinstance(res, dict) else None
            if not isinstance(block, dict) or not isinstance(block.get("hits"), list):
                raise ValueError("unexpected full-text search response shape")
            hits = block["hits"]
            for h in hits:
                s = h.get("_source") or {}
                acc = s.get("adsh") or str(h.get("_id", "")).split(":")[0]
                for ck in s.get("ciks") or []:
                    if int(ck) not in universe:
                        continue      # a co-registrant outside the watch/held set
                    row = found.setdefault((int(ck), acc), {
                        "cik": int(ck), "accession": acc, "rule": rule["id"],
                        "source": "text", "form": s.get("form") or "8-K",
                        "file_date": s.get("file_date") or "",
                        "items": sorted(set(s.get("items") or [])), "docs": []})
                    doc = s.get("file_type") or s.get("file_description")
                    if doc and doc not in row["docs"]:
                        row["docs"].append(doc)
            offset += _EFTS_PAGE
            total = int(((block.get("total") or {}).get("value")) or 0)
            if not hits or offset >= total:
                break
    return list(found.values()), requests


def filing_flag_window(today, cursor: dict | None = None) -> tuple[str, str]:
    """The night's [startdt, enddt]: at least the last FRESH_DAYS, reaching back to the
    last successful run minus OVERLAP after an outage, never more than MAX_DAYS."""
    from ..config import FILING_FLAG_FRESH_DAYS, FILING_FLAG_MAX_DAYS, FILING_FLAG_OVERLAP_DAYS

    t = pd.Timestamp(today).normalize()
    start = t - pd.Timedelta(days=FILING_FLAG_FRESH_DAYS)
    last = (cursor or {}).get("enddt")
    if last:
        start = min(start, pd.Timestamp(last) - pd.Timedelta(days=FILING_FLAG_OVERLAP_DAYS))
    start = max(start, t - pd.Timedelta(days=FILING_FLAG_MAX_DAYS))
    return str(start.date()), str(t.date())


def _flag_label(cik: int, labels: dict | None, submissions: dict) -> str:
    tickers = (submissions.get(cik) or {}).get("tickers") or []
    name = (labels or {}).get(cik) or (tickers[0] if tickers else None)
    return f"cik {cik} ({name})" if name else f"cik {cik}"


def _flag_alert(cik: int, grp: list[dict], rules, label: str, held_note: str):
    """(kind, message, payload) for ONE filing — every rule it matched, one alert."""
    meta = {r["id"]: r for r in rules}
    order = [r["id"] for r in rules]
    rids = sorted({g["rule"] for g in grp}, key=order.index)
    critical = any(meta[r]["tier"] == "critical" for r in rids)
    first = grp[0]
    acc = first["accession"]
    items = sorted({i for g in grp for i in (g.get("items") or [])})
    docs = sorted({d for g in grp if g["source"] == "text" for d in (g.get("docs") or [])})
    detail = [", ".join(meta[r]["label"] for r in rids)]
    if items:
        detail.append("items " + ", ".join(items))
    if docs:
        detail.append("text matched in " + ", ".join(docs))
    message = (f"{label}: {first['form']} filed {first['file_date']} — " + "; ".join(detail)
               + " [filing flag — not a trade signal]" + held_note)
    payload = {"accession": acc, "form": first["form"], "file_date": first["file_date"],
               "rules": rids, "items": items, "docs": docs,
               "tier": "critical" if critical else "flag",
               "url": (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                       f"{acc.replace('-', '')}/{acc}-index.htm")}
    return ("eightk_critical" if critical else "eightk_flag"), message, payload


def run_filing_flags(state: OpsState, ciks, client: EdgarClient, submissions=None,
                     today=None, rules=None, labels: dict | None = None,
                     weights: dict | None = None, dry_run: bool = False,
                     startdt: str | None = None) -> dict:
    """One filing-flag pass over ``ciks`` (watch ∪ held). Returns a deltas block.

    Alerts once per FILING at first sight, listing every rule it matched; rules found
    later for an already-recorded filing are stored silently. A name new to the set
    alerts only on matches filed within FRESH_DAYS. Fail-open per rule: an error is
    recorded and the cursor is held, so the next night re-covers the window.
    ``dry_run`` writes nothing (no flags, alerts, cursor) and returns ``preview``;
    ``startdt`` overrides the cursor window (preview use)."""
    from ..config import FILING_FLAG_FRESH_DAYS, FILING_FLAG_RULES

    rules = FILING_FLAG_RULES if rules is None else rules
    t = pd.Timestamp(today if today is not None else pd.Timestamp.now()).normalize()
    ciks = [int(c) for c in dict.fromkeys(ciks)]
    out: dict = {"names": len(ciks), "matches": 0, "alerts": 0, "requests": 0, "errors": []}
    if not ciks:
        return out
    if startdt is None:
        startdt, enddt = filing_flag_window(t, state.kv_get(_FLAG_CURSOR))
    else:
        enddt = str(t.date())
    out["window"] = [startdt, enddt]
    if submissions is None:
        submissions = fetch_submissions(ciks, client)
        out["requests"] += len(ciks)

    rows: list[dict] = []
    for cik in ciks:
        rows += detect_item_flags(cik, submissions.get(cik), rules)
    text_ok = True
    for rule in (r for r in rules if r.get("q")):
        try:
            found, n = search_text_flags(client, ciks, rule, startdt, enddt)
            rows += found
            out["requests"] += n
        except Exception as exc:  # one rule failing must not hide the others
            text_ok = False
            out["errors"].append({"rule": rule["id"],
                                  "error": f"{type(exc).__name__}: {exc}"[:200]})

    groups: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["cik"], r["accession"]), []).append(r)
    seeded = set() if dry_run else state.filing_flag_ciks()
    fresh_from = str((t - pd.Timedelta(days=FILING_FLAG_FRESH_DAYS)).date())
    preview = []
    for (cik, acc), grp in sorted(groups.items(), key=lambda kv: kv[1][0]["file_date"]):
        file_date = grp[0]["file_date"]
        in_window = file_date >= startdt
        out["matches"] += int(in_window)
        if dry_run:
            if in_window:
                kind, message, payload = _flag_alert(
                    cik, grp, rules, _flag_label(cik, labels, submissions), "")
                preview.append({"kind": kind, "message": message, **payload})
            continue
        seen_before = state.filing_flag_seen(cik, acc)
        state.add_filing_flags(grp)
        if seen_before or not in_window or (cik not in seeded and file_date < fresh_from):
            continue   # already alerted / outside tonight's window / bootstrap: old news
        kind, message, payload = _flag_alert(cik, grp, rules,
                                             _flag_label(cik, labels, submissions),
                                             _held_note(weights or {}, cik))
        state.add_alert(kind, message, cik=cik, payload=payload)
        out["alerts"] += 1

    if dry_run:
        out["preview"] = preview
        return out
    cursor = state.kv_get(_FLAG_CURSOR) or {}
    if text_ok:
        state.mark_filing_flag_ciks(ciks)
        state.kv_set(_FLAG_CURSOR, {"enddt": enddt, "errors_in_a_row": 0})
    else:   # hold the cursor: the next run re-covers this window
        state.kv_set(_FLAG_CURSOR, {"enddt": cursor.get("enddt"),
                                    "errors_in_a_row": int(cursor.get("errors_in_a_row") or 0) + 1})
    return out


def book_weights(state: OpsState, data) -> dict[int, float]:
    """Value weight of each HELD cik in the user's book, from the last close.

    DISPLAY-ONLY: used solely to annotate alert text so a watch alert on a name
    you actually hold reads with the urgency it deserves. Positions never feed
    the score, the paper book, or any signal computation — this function's output
    goes into alert MESSAGES and nowhere else."""
    try:
        pos = state.positions()
    except Exception:
        return {}
    vals: dict[int, float] = {}
    for p in pos:
        cik, shares = int(p["cik"]), float(p.get("shares") or 0)
        col = data.ticker_map.get(cik)
        if shares <= 0 or not col or col not in data.close.columns:
            continue
        s = data.close[col].dropna()
        if len(s):
            vals[cik] = shares * float(s.iloc[-1])
    total = sum(vals.values())
    return {c: v / total for c, v in vals.items()} if total > 0 else {}


def _held_note(weights: dict[int, float], cik: int) -> str:
    w = weights.get(int(cik))
    if w is None:
        return ""
    return f" — you hold ≈{w * 100:.0f}% of your book"


def run_monitor(
    state: OpsState,
    data=None,
    artifact=None,
    llm_full=None,
    llm_light=None,
    cache=None,
    narrate: bool = True,
    edgar: bool = True,
    alerts_ok: bool = True,
    as_of=None,
    pctile_threshold: int = MONITOR_PCTILE_ALERT,
    paper_dir=None,
    model_dir=None,
    filing_flags: bool | None = None,
    client: EdgarClient | None = None,
    today=None,
) -> dict:
    """One monitoring pass. Returns the deltas dict for the job log.

    ``filing_flags`` (default config FILING_FLAGS) adds the 8-K flag pass over the
    watch ∪ held set; it needs EDGAR, so ``edgar=False`` skips it too. ``today`` dates
    that pass (real filing dates, not the price ``as_of``)."""
    from ..config import FILING_FLAGS, PAPER_DIR
    from ..distress import distress_flag
    from ..drawdown import drawdown_flag
    from ..model import MODEL_DIR, load_artifact
    from ..narrate.cache import NarrationCache, narrate_smart
    from ..serve import analyze, load_serve_data
    from .paper import artifact_fingerprint, current_vintage

    _LEVEL = {"normal": 0, "elevated": 1, "high": 2}

    paper_dir = paper_dir if paper_dir is not None else PAPER_DIR
    model_dir = model_dir if model_dir is not None else MODEL_DIR

    watch = state.watchlist()
    deltas: dict = {"n_watch": len(watch), "alerts": 0, "filings_new": 0,
                    "narrated": {}, "errors": []}
    do_flags = edgar and (FILING_FLAGS if filing_flags is None else filing_flags)
    universe = alert_universe(state, watch) if do_flags else []
    if not watch and not universe:
        return {**deltas, "note": "watchlist empty"}

    data = data or load_serve_data()
    artifact = artifact or load_artifact()
    as_of = pd.Timestamp(as_of) if as_of is not None else data.close.index[-1]
    deltas["as_of"] = str(as_of.date())

    # A monitor quietly running an artifact the paper trail doesn't know about
    # would diverge from the logged record for weeks — same check log_signals has.
    vintage = current_vintage(paper_dir)
    if vintage is not None:
        fp = artifact_fingerprint(model_dir)
        if fp != vintage["hash"]:
            state.add_alert(
                "unregistered_artifact",
                f"artifact {fp} differs from the registered vintage {vintage['hash']}; "
                f"register the retrain or restore the artifact",
            )
            deltas["alerts"] += 1
            deltas["unregistered_artifact"] = True

    # filings: numbers landed (wide table) + filed-on-EDGAR early warning
    ciks = [w["cik"] for w in watch]
    landed = detect_wide_filings(state, data.feats, ciks)
    for f in landed:
        state.add_alert("fundamentals_updated",
                        f"cik {f['cik']}: new 10-K numbers on disk "
                        f"(period {f['period_end']}, filed {f['filed_date']})",
                        cik=f["cik"], payload=f)
    # position-aware annotation (display-only): a signal alert on a HELD name says so
    weights = book_weights(state, data)

    filed: list[dict] = []
    if edgar:   # ONE submissions fetch per name, shared by both EDGAR detectors
        own = client is None
        client = client or EdgarClient()
        try:
            subs = fetch_submissions(list(dict.fromkeys(ciks + universe)), client)
            filed = detect_edgar_filings(state, ciks, submissions=subs)
            if do_flags:
                ff = run_filing_flags(state, universe, client, submissions=subs, today=today,
                                      labels=dict(data.ticker_map), weights=weights)
                deltas["filing_flags"] = ff
                deltas["alerts"] += ff["alerts"]
        finally:
            if own:
                client.close()
    for f in filed:
        state.add_alert("filing_detected",
                        f"cik {f['cik']}: {f['form']} filed {f['filed_date']} on EDGAR "
                        f"(numbers arrive with the next FSDS batch)",
                        cik=f["cik"], payload=f)
    deltas["filings_new"] = len(landed) + len(filed)
    deltas["alerts"] += len(landed) + len(filed)

    if not watch:   # a held-only book: the filing flags were the whole pass
        return {**deltas, "note": "watchlist empty — filing flags only"}

    if cache is None and narrate:
        cache = NarrationCache()

    tiers: dict[str, int] = {}
    for w in watch:
        cik = w["cik"]
        try:
            res = analyze(cik, as_of=as_of, data=data, artifact=artifact, llm=None)
        except ValueError as exc:  # no PIT filing / too stale — surface, don't crash
            deltas["errors"].append({"cik": cik, "error": str(exc)})
            continue

        pct = int(res["percentile"])
        dz = res.get("distress")   # FIREWALLED risk-flag block (or None)
        prev = state.get_signal(cik)
        if alerts_ok:
            if prev is not None and abs(pct - prev["percentile"]) >= pctile_threshold:
                state.add_alert(
                    "percentile_move",
                    f"cik {cik} ({res['column'] or 'no column'}): model percentile "
                    f"{prev['percentile']} -> {pct} "
                    f"({prev['as_of']} -> {as_of.date()})" + _held_note(weights, cik),
                    cik=cik,
                    payload={"from": prev["percentile"], "to": pct,
                             "decile": res["decile"]},
                )
                deltas["alerts"] += 1
            # distress escalation: alert only when the risk-flag crosses UP a level between
            # runs (never every night, never on de-escalation, and — like the percentile
            # alert — never on first sight; the first pass records the level silently). A
            # watchlist convenience: the flag is display/monitoring only and drives no trade
            # action (RESULTS.md overlay gate). The TUI already shows a newly-watched name's
            # current level, so the alert is reserved for genuine changes over time.
            if dz is not None and prev is not None:
                prev_prob = prev.get("distress")
                prev_lvl = distress_flag(prev_prob) if prev_prob is not None else "normal"
                if _LEVEL[dz["flag"]] > _LEVEL[prev_lvl] and dz["flag"] != "normal":
                    state.add_alert(
                        "distress_risk",
                        f"cik {cik} ({res['column'] or 'no column'}): distress-risk flag "
                        f"{prev_lvl} -> {dz['flag']} (P≈{dz['prob'] * 100:.1f}% within "
                        f"{dz['horizon_months']}mo) [risk-flag only, not a trade signal]"
                        + _held_note(weights, cik),
                        cik=cik,
                        payload={"from": prev_lvl, "to": dz["flag"], "prob": dz["prob"]},
                    )
                    deltas["alerts"] += 1
            # drawdown escalation mirrors distress, with one extra gate: alert ONLY on
            # entering "high" (the 39.5% base rate makes "elevated" too common to
            # interrupt for — it stays a chip in the UI). Same first-sight silence,
            # same never-on-de-escalation, same display-only framing.
            wz = res.get("drawdown")   # FIREWALLED risk-flag block (or None)
            if wz is not None and prev is not None:
                prev_wprob = prev.get("drawdown")
                prev_wlvl = drawdown_flag(prev_wprob) if prev_wprob is not None else "normal"
                if wz["flag"] == "high" and _LEVEL[prev_wlvl] < _LEVEL["high"]:
                    state.add_alert(
                        "drawdown_risk",
                        f"cik {cik} ({res['column'] or 'no column'}): drawdown-risk flag "
                        f"{prev_wlvl} -> high (P≈{wz['prob'] * 100:.0f}% of a "
                        f"{wz['threshold'] * 100:.0f}% fall within {wz['horizon_months']}mo) "
                        f"[risk-flag only, not a trade signal]" + _held_note(weights, cik),
                        cik=cik,
                        payload={"from": prev_wlvl, "to": "high", "prob": wz["prob"]},
                    )
                    deltas["alerts"] += 1
            state.record_signal(cik, pct, res["decile"], str(as_of.date()),
                                distress=(dz["prob"] if dz else None),
                                drawdown=(wz["prob"] if wz else None))

        if narrate:
            # scan.py's pattern: the packet from analyze(llm=None) carries the
            # model block (percentile, drivers) the materiality gate keys on.
            nar = narrate_smart(res["packet"], llm_full=llm_full,
                                llm_light=llm_light, cache=cache)
            tiers[nar.get("tier", "?")] = tiers.get(nar.get("tier", "?"), 0) + 1

    deltas["narrated"] = tiers
    deltas["alerts_suppressed"] = not alerts_ok
    return deltas
