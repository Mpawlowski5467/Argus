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
  light / full — a template run never touches the cache).

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


def detect_edgar_filings(state: OpsState, ciks: list[int],
                         client: EdgarClient | None = None) -> list[dict]:
    """Filings visible on EDGAR that FSDS hasn't delivered yet (early warning).

    Uses the throttled submissions API (recent filings only). Bootstrap seeding
    is silent, same as the wide-table path.
    """
    if not ciks:
        return []
    own = client is None
    client = client or EdgarClient()
    news = []
    try:
        for cik in ciks:
            try:
                data = client.get_json(
                    f"{client.DATA_HOST}/submissions/CIK{int(cik):010d}.json")
            except Exception:
                continue  # one unreachable company must not kill the loop
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
    finally:
        if own:
            client.close()
    return news


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
) -> dict:
    """One monitoring pass. Returns the deltas dict for the job log."""
    from ..config import PAPER_DIR
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
    if not watch:
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
    filed = detect_edgar_filings(state, ciks) if edgar else []
    for f in filed:
        state.add_alert("filing_detected",
                        f"cik {f['cik']}: {f['form']} filed {f['filed_date']} on EDGAR "
                        f"(numbers arrive with the next FSDS batch)",
                        cik=f["cik"], payload=f)
    deltas["filings_new"] = len(landed) + len(filed)
    deltas["alerts"] += len(landed) + len(filed)

    if cache is None and narrate:
        cache = NarrationCache()

    # position-aware annotation (display-only): a signal alert on a HELD name says so
    weights = book_weights(state, data)

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
