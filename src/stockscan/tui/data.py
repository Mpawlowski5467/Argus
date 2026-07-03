"""The view-data facade: turn the serve/ops layer into plain rows for the TUI.

Heavy inputs (the ~11k-column price matrices, the frozen artifact, the ops DB)
load ONCE into an :class:`ArgusData`; the scored cross-section is built once and
reused across the scan and watch views. The row-shaping helpers are pure
functions so they unit-test without Textual or a real data store.

Nothing here trains or re-baselines. The only writes are watchlist/alert
curation, delegated to :class:`~stockscan.ops.state.OpsState`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --- pure row shapers (testable with tiny DataFrames) ---------------------------

def _decile(pct: float) -> int:
    return int(np.clip(np.ceil(pct * 10), 1, 10))


def scan_rows(cross: pd.DataFrame, sector: str | None = None) -> list[dict]:
    """Ranked rows for the scan table from a SCORED cross-section (has score/pct)."""
    view = cross if not sector or sector.lower() == "all" else cross[cross["sector"] == sector]
    view = view.sort_values("score", ascending=False)
    rows = []
    for i, (_, r) in enumerate(view.iterrows(), 1):
        rows.append({
            "rank": i,
            "cik": int(r["cik"]),
            "ticker": str(r.get("ticker") or "—"),
            "name": str(r.get("name") or "")[:38],
            "sector": str(r.get("sector") or "—"),
            "pct": int(round(float(r["pct"]) * 100)),
            "decile": _decile(float(r["pct"])),
            "fy": int(r["fy"]) if pd.notna(r.get("fy")) else None,
        })
    return rows


def search_rows(cross: pd.DataFrame, query: str, limit: int = 40) -> list[dict]:
    """Scan rows filtered to names whose ticker OR company-name contains ``query``.

    Case-insensitive substring, ranked by score like the scan (best first). Empty
    query returns the full ranked scan (capped)."""
    q = str(query or "").strip()
    if not q:
        return scan_rows(cross)[:limit]
    qu = q.upper()
    tick = cross["ticker"].astype(str).str.upper().str.contains(qu, regex=False, na=False)
    name = cross["name"].astype(str).str.upper().str.contains(qu, regex=False, na=False)
    return scan_rows(cross[tick | name])[:limit]


def sectors_in(cross: pd.DataFrame) -> list[str]:
    return ["all", *sorted(cross["sector"].dropna().unique())]


def watch_rows(watchlist: list[dict], cross: pd.DataFrame,
               prev_signals: dict[int, dict], feats: pd.DataFrame,
               as_of=None) -> list[dict]:
    """Join watched CIKs to the current scored cross-section + last-seen signal.

    A watched name absent from the cross-section (dead, stale filer, illiquid)
    still appears — flagged — so the watchlist never silently drops the failures.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else None
    by_cik = cross.set_index(cross["cik"].astype(int))
    rows = []
    for w in watchlist:
        cik = int(w["cik"])
        prev = prev_signals.get(cik) or {}
        last_filing = None
        fsub = feats[feats["cik"] == cik] if "cik" in feats.columns else feats.iloc[0:0]
        if as_of is not None and "available_date" in fsub.columns:
            fsub = fsub[fsub["available_date"] <= as_of]
        if len(fsub):
            last_filing = str(pd.Timestamp(fsub["available_date"].max()).date())
        if cik in by_cik.index:
            r = by_cik.loc[cik]
            pct = int(round(float(r["pct"]) * 100))
            prev_pct = prev.get("percentile")
            rows.append({
                "cik": cik, "ticker": str(w.get("column") or r.get("ticker") or "—"),
                "pct": pct, "decile": _decile(float(r["pct"])),
                "delta": (pct - prev_pct) if prev_pct is not None else None,
                "last_filing": last_filing, "flag": None,
            })
        else:
            rows.append({
                "cik": cik, "ticker": str(w.get("column") or "—"),
                "pct": None, "decile": None, "delta": None,
                "last_filing": last_filing,
                "flag": "not in liquid universe / lapsed filer",
            })
    return rows


# --- the loaded facade ----------------------------------------------------------

@dataclass
class ArgusData:
    """Loaded-once heavy inputs + the reusable scored cross-section.

    Holds NO long-lived SQLite connections: the load runs in a background thread
    but the views query from the main thread, and a sqlite3 connection is bound
    to its creating thread. Each DB-touching method opens its own short-lived
    connection instead (cheap; WAL makes it safe alongside the nightly job).
    """

    data: object          # stockscan.serve.ServeData
    artifact: object      # stockscan.model.Artifact
    as_of: pd.Timestamp = None
    _cross: pd.DataFrame = None

    @classmethod
    def load(cls, as_of=None) -> "ArgusData":
        from ..model import load_artifact
        from ..serve import load_serve_data

        self = cls(data=load_serve_data(), artifact=load_artifact())
        self.refresh(as_of)
        return self

    def refresh(self, as_of=None) -> None:
        """(Re)build and score the cross-section — call after data updates."""
        from ..serve import build_cross_section

        self.as_of = (pd.Timestamp(as_of) if as_of is not None
                      else self.data.close.index[-1])
        cross = build_cross_section(self.data, self.as_of).reset_index(drop=True)
        cross["score"] = self.artifact.score(cross)
        cross["pct"] = cross["score"].rank(pct=True)
        self._cross = cross

    # -- view data ---------------------------------------------------------------
    def sectors(self) -> list[str]:
        return sectors_in(self._cross)

    def scan(self, sector: str | None = None) -> list[dict]:
        return scan_rows(self._cross, sector)

    def search(self, query: str, limit: int = 40) -> list[dict]:
        return search_rows(self._cross, query, limit)

    def resolve(self, query) -> dict | None:
        """Resolve any ticker / TICKER~CIK / CIK to a cik (survivorship-safe). None if unknown."""
        from ..serve import resolve_company

        try:
            cik, column = resolve_company(query, self.data.ticker_map)
        except Exception:
            return None
        return {"cik": int(cik), "column": column}

    def price(self, cik: int) -> dict | None:
        """Close series + summary (last/changes/52wk/ADV) for a name, from the loaded matrices."""
        from .chart import price_summary

        column = self.data.ticker_map.get(int(cik))
        if not column or column not in self.data.close.columns:
            return None
        series = self.data.close[column].dropna()
        adv = None
        if column in self.data.dv_med.columns:
            dvs = self.data.dv_med[column].dropna()
            adv = float(dvs.iloc[-1]) if len(dvs) else None
        return {"column": column, "series": series, "summary": price_summary(series, adv)}

    def events(self, cik: int, limit: int = 8) -> list[dict]:
        """Recent newsworthy EDGAR filings (network; call from a worker)."""
        from ..news import recent_filings

        return recent_filings(int(cik), limit=limit)

    def ohlc(self, cik: int, tail: int = 252):
        """Adjusted OHLCV rows for a name from the per-column price store (local, no quota)."""
        import pandas as pd

        from ..prices import PRICES_DIR

        column = self.data.ticker_map.get(int(cik))
        if not column:
            return None
        p = PRICES_DIR / f"{column}.parquet"
        if not p.exists():
            return None
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = pd.read_parquet(p, columns=cols).dropna(subset=["close"]).sort_values("date")
        return df.tail(tail).reset_index(drop=True)

    def _universe(self):
        uni = self.__dict__.get("_uni")
        if uni is None:
            from ..intrinio_universe import load_universe
            uni = self.__dict__["_uni"] = load_universe()
        return uni

    def _pick(self, cik: int, field: str):
        u = self._universe()
        r = u[u["cik"] == int(cik)]
        if r.empty:
            return None
        return str(r.sort_values("priority").iloc[0][field])

    def news(self, cik: int, limit: int = 6) -> list[dict]:
        """Watchlist headline MEMORY for a name (LIVE-VIEW ONLY — never the signal).

        Lazily ingest into news.sqlite (heuristic extraction — instant; the nightly job
        upgrades watchlist names to the LLM tier) then recall recent + notable-past
        events. Quota-capped by the store's fetch throttle and session-cached per cik."""
        cache = self.__dict__.setdefault("_news_cache", {})
        if cik in cache:
            return cache[cik]
        from ..newsmem import NewsStore
        from ..newsmem.ingest import ingest_company_news

        tk = self._pick(cik, "ticker")
        rows: list[dict] = []
        try:
            with NewsStore() as store:
                ingest_company_news(int(cik), tk, store, llm=None, limit=limit)
                rows = store.context_for(int(cik)) or store.recall(int(cik), limit=limit)
        except Exception:
            rows = []
        cache[cik] = rows
        return rows

    def _news_context(self, cik: int) -> list[dict]:
        """Recalled news memory (recent + notable past) for the narration packet."""
        from ..newsmem import NewsStore

        try:
            with NewsStore() as store:
                return store.context_for(int(cik))
        except Exception:
            return []

    def live_quote(self, cik: int, refresh: bool = False) -> dict | None:
        """Latest live/intraday quote (Intrinio). Session-cached unless refresh=True."""
        cache = self.__dict__.setdefault("_quote_cache", {})
        if not refresh and cik in cache:
            return cache[cik]
        from ..quote import realtime_price

        sid = self._pick(cik, "security_id")
        res = realtime_price(sid) if sid else None
        cache[cik] = res
        return res

    # -- watchlist (the only writes — sanctioned per the read-mostly design) ---------
    def is_watched(self, cik: int) -> bool:
        from ..ops.state import OpsState

        with OpsState() as st:
            return any(int(w["cik"]) == int(cik) for w in st.watchlist())

    def toggle_watch(self, cik: int) -> bool:
        """Add/remove the name from the watchlist. Returns the new watched state."""
        from ..ops.state import OpsState

        with OpsState() as st:
            watched = any(int(w["cik"]) == int(cik) for w in st.watchlist())
            if watched:
                st.watch_remove(int(cik))
                return False
            st.watch_add(int(cik), self.data.ticker_map.get(int(cik)))
            return True

    def ticker(self, query, as_of=None) -> dict:
        """Full per-name analysis (deterministic; template narration included)."""
        from ..serve import analyze

        return analyze(query, as_of=as_of or self.as_of, data=self.data,
                       artifact=self.artifact, llm=None)

    def narrate(self, packet, llm_full=None, llm_light=None) -> dict:
        """On-demand ('n' key) narration with the local model, bringing up recalled news.

        Runs in the narrate worker thread. Attaches CURRENT news context (LIVE-VIEW;
        excluded from any cache key) and narrates FRESH at the full tier — the user
        asked for a current read, so this deliberately does NOT serve the monitor's
        cached, fundamental-only narration (nor overwrite it). A dead LLM endpoint
        degrades to the grounded template inside narrate_packet (never crashes)."""
        from ..narrate.llm import LocalLLM
        from ..narrate.narrator import narrate_packet
        from ..narrate.packet import news_context

        ctx = news_context(self._news_context(int(packet["meta"]["cik"])))
        if ctx:
            packet.setdefault("context", {})["news"] = ctx
        res = narrate_packet(packet, llm=llm_full or LocalLLM())
        res["tier"] = ("full+news" if ctx else "full") if res.get("source") == "llm" else "template"
        return res

    def watch(self) -> dict:
        from ..ops.state import OpsState

        with OpsState() as st:
            wl = st.watchlist()
            prev = {w["cik"]: (st.get_signal(w["cik"]) or {}) for w in wl}
            alerts = st.alerts(unseen_only=False, limit=40)
        rows = watch_rows(wl, self._cross, prev, self.data.feats, self.as_of)
        return {"rows": rows, "alerts": alerts}

    def status(self) -> dict:
        from ..ops.state import OpsState

        with OpsState() as st:
            return status_dict(st, self.artifact, self.as_of)

    def paper(self) -> dict | None:
        from ..config import PAPER_DIR
        from ..ops.paper import compare

        if not (PAPER_DIR / "baseline.json").exists():
            return None
        return compare(close=self.data.close, ticker_map=self.data.ticker_map)

    def close(self) -> None:
        pass  # no long-lived connections to release


def status_dict(state, artifact, as_of) -> dict:
    """Cheap status line — no LLM ping (that stays off the hot path)."""
    from ..ops.jobs import quarters_present
    from ..ops.paper import artifact_fingerprint, current_vintage

    quarters = quarters_present()
    try:
        fp = artifact_fingerprint()
    except FileNotFoundError:
        fp = None
    vintage = current_vintage()
    last = state.last_run("nightly")
    return {
        "as_of": str(pd.Timestamp(as_of).date()) if as_of is not None else None,
        "fund_quarter": quarters[-1] if quarters else None,
        "vintage": (vintage or {}).get("hash"),
        "artifact_registered": bool(vintage and fp and vintage["hash"] == fp),
        "nightly_status": (last or {}).get("status"),
        "unseen_alerts": len(state.alerts(unseen_only=True, limit=999)),
    }
