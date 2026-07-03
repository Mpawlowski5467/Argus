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
    from ..model import MODEL_DIR, load_artifact
    from ..narrate.cache import NarrationCache, narrate_smart
    from ..serve import analyze, load_serve_data
    from .paper import artifact_fingerprint, current_vintage

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

    tiers: dict[str, int] = {}
    for w in watch:
        cik = w["cik"]
        try:
            res = analyze(cik, as_of=as_of, data=data, artifact=artifact, llm=None)
        except ValueError as exc:  # no PIT filing / too stale — surface, don't crash
            deltas["errors"].append({"cik": cik, "error": str(exc)})
            continue

        pct = int(res["percentile"])
        prev = state.get_signal(cik)
        if alerts_ok:
            if prev is not None and abs(pct - prev["percentile"]) >= pctile_threshold:
                state.add_alert(
                    "percentile_move",
                    f"cik {cik} ({res['column'] or 'no column'}): model percentile "
                    f"{prev['percentile']} -> {pct} "
                    f"({prev['as_of']} -> {as_of.date()})",
                    cik=cik,
                    payload={"from": prev["percentile"], "to": pct,
                             "decile": res["decile"]},
                )
                deltas["alerts"] += 1
            state.record_signal(cik, pct, res["decile"], str(as_of.date()))

        if narrate:
            # scan.py's pattern: the packet from analyze(llm=None) carries the
            # model block (percentile, drivers) the materiality gate keys on.
            nar = narrate_smart(res["packet"], llm_full=llm_full,
                                llm_light=llm_light, cache=cache)
            tiers[nar.get("tier", "?")] = tiers.get(nar.get("tier", "?"), 0) + 1

    deltas["narrated"] = tiers
    deltas["alerts_suppressed"] = not alerts_ok
    return deltas
