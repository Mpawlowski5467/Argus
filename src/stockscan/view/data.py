"""The view-data facade: turn the serve/ops layer into plain rows for the web UI.

Heavy inputs (the ~11k-column price matrices, the frozen artifact, the ops DB)
load ONCE into an :class:`ArgusData`; the scored cross-section is built once and
reused across the scan and watch views. The row-shaping helpers are pure
functions so they unit-test without a UI framework or a real data store.

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
            # FIREWALLED distress flag (display only): surface elevated/high risk here so a
            # watched name drifting toward failure is visible; it drives no trade action.
            dflag = r.get("dflag") if "dflag" in cross.columns else None
            flag = None
            if dflag in ("elevated", "high"):
                dp = float(r["dprob"]) if pd.notna(r.get("dprob")) else None
                flag = f"⚠ distress {dflag}" + (f" (P≈{dp * 100:.0f}%)" if dp is not None else "")
            rows.append({
                "cik": cik, "ticker": str(w.get("column") or r.get("ticker") or "—"),
                "pct": pct, "decile": _decile(float(r["pct"])),
                "delta": (pct - prev_pct) if prev_pct is not None else None,
                "last_filing": last_filing, "flag": flag,
            })
        else:
            rows.append({
                "cik": cik, "ticker": str(w.get("column") or "—"),
                "pct": None, "decile": None, "delta": None,
                "last_filing": last_filing,
                "flag": "not in liquid universe / lapsed filer",
            })
    return rows


def _pick_row(r) -> dict:
    """One markets-page pick row from a scored cross-section row."""
    return {
        "cik": int(r["cik"]),
        "ticker": str(r.get("ticker") or "—"),
        "name": str(r.get("name") or "")[:34],
        "pct": int(round(float(r["pct"]) * 100)),
        "decile": _decile(float(r["pct"])),
    }


def market_rows(cross: pd.DataFrame, top_k: int = 6, min_names: int = 10) -> list[dict]:
    """Per-industry ML top picks for the markets overview (sized later by market cap).

    Groups by the fine ``sic_industry`` label (Semiconductors, Oil & Gas E&P,
    Software, Banks, …) — NOT the coarse model sector — ordered by how many names
    each holds. Industries thinner than ``min_names`` and the catch-all 'Unknown'
    are dropped. Each market's ``picks`` are its highest-scoring names, best first;
    the page annotates each with a live market cap fetched separately.
    """
    from ..sector import sic_industry

    df = cross.copy()
    df["_industry"] = df["sic"].map(sic_industry)
    counts = df["_industry"].value_counts()
    out = []
    for industry in counts.index:
        if not industry or str(industry) == "Unknown" or counts[industry] < min_names:
            continue
        sub = (df[df["_industry"] == industry]
               .sort_values("score", ascending=False).head(top_k))
        picks = [_pick_row(r) for _, r in sub.iterrows()]
        if picks:
            out.append({"market": str(industry), "count": int(counts[industry]),
                        "picks": picks})
    return out


def theme_market_rows(cross: pd.DataFrame, tags: dict, top_k: int = 6,
                      min_names: int = 3) -> list[dict]:
    """Thematic 'markets' (AI/SaaS/EV…) from precomputed {cik: [themes]} tags.

    Same shape as ``market_rows`` but membership comes from the auto-tagged theme
    store rather than SIC. Only names present in the current cross-section count;
    themes with fewer than ``min_names`` are dropped; ordered by tagged count.
    """
    if not tags:
        return []
    present = set(cross["cik"].astype(int))
    by_theme: dict[str, set] = {}
    for cik, themes in tags.items():
        if int(cik) not in present:
            continue
        for t in themes:
            by_theme.setdefault(t, set()).add(int(cik))

    ciks_col = cross["cik"].astype(int)
    out = []
    for theme, ciks in by_theme.items():
        if len(ciks) < min_names:
            continue
        sub = (cross[ciks_col.isin(ciks)]
               .sort_values("score", ascending=False).head(top_k))
        picks = [_pick_row(r) for _, r in sub.iterrows()]
        if picks:
            out.append({"market": str(theme), "count": len(ciks), "picks": picks})
    out.sort(key=lambda m: m["count"], reverse=True)
    return out


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
        # FIREWALLED distress risk-flag: a second read on the SAME ranks, for display only
        # (never touches score/pct/decile or any trade path). Absent artifact -> no column.
        dart = getattr(self.data, "distress_artifact", None)
        if dart is not None:
            from ..distress import distress_flag

            cross["dprob"] = dart.score(cross)
            cross["dflag"] = [distress_flag(p) for p in cross["dprob"]]
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

    def profile(self, cik: int) -> dict | None:
        """Company profile — what it does + HQ (Intrinio, LIVE-VIEW ONLY, never the signal).

        Looked up by CIK (recycle-proof) and served from the persistent profiles.sqlite
        cache; the network fetch only fires on a cache miss or a >30-day-stale row.
        Session-cached on top of that so re-opening a name is instant."""
        cache = self.__dict__.setdefault("_profile_cache", {})
        if cik in cache:
            return cache[cik]
        from ..profile import get_profile

        res = get_profile(int(cik))
        cache[cik] = res
        return res

    def markets(self, top_k: int = 6) -> list[dict]:
        """Per-industry ML top picks for the markets overview (caps fetched separately)."""
        return market_rows(self._cross, top_k)

    def theme_markets(self, top_k: int = 6) -> list[dict]:
        """Thematic markets (AI/SaaS/EV…) from the precomputed tag store — [] if unbuilt.

        Tags are auto-generated by `ops.py themes` (keyword-matching Intrinio company
        descriptions, LIVE-VIEW ONLY). Loaded once per session."""
        tags = self.__dict__.get("_theme_tags")
        if tags is None:
            from ..themes import load_theme_tags
            try:
                tags = load_theme_tags()
            except Exception:
                tags = {}
            self.__dict__["_theme_tags"] = tags
        return theme_market_rows(self._cross, tags, top_k)

    def _theme_tag_map(self) -> dict:
        tags = self.__dict__.get("_theme_tags")
        if tags is None:
            from ..themes import load_theme_tags
            try:
                tags = load_theme_tags()
            except Exception:
                tags = {}
            self.__dict__["_theme_tags"] = tags
        return tags

    def market_constituents(self, kind: str, name: str, cand: int = 60) -> list[dict]:
        """Names in one market (``kind`` = 'ind' | 'theme'), for the treemap.

        Capped to the top ``cand`` by recent dollar volume — a cheap in-memory size
        proxy so the caller live-fetches market cap for only the biggest-liquidity
        candidates (the true megacaps are always among them), not the whole market.
        """
        df = self._cross
        if kind == "theme":
            ciks = {int(c) for c, ts in self._theme_tag_map().items() if name in ts}
            sub = df[df["cik"].astype(int).isin(ciks)]
        else:
            from ..sector import sic_industry
            sub = df[df["sic"].map(sic_industry) == name]
        if sub.empty:
            return []
        dv = {}
        for cik in sub["cik"].astype(int):
            col = self.data.ticker_map.get(int(cik))
            s = (self.data.dv_med[col].dropna()
                 if col and col in self.data.dv_med.columns else None)
            dv[int(cik)] = float(s.iloc[-1]) if s is not None and len(s) else 0.0
        sub = (sub.assign(_dv=sub["cik"].astype(int).map(dv))
               .sort_values("_dv", ascending=False).head(cand))
        return [{
            "cik": int(r["cik"]),
            "ticker": str(r.get("ticker") or "—"),
            "name": str(r.get("name") or ""),
            "pct": int(round(float(r["pct"]) * 100)),
            "decile": _decile(float(r["pct"])),
        } for _, r in sub.iterrows()]

    def fundamentals(self, cik: int) -> dict | None:
        """Recent financials for one name, straight from the in-memory cross row.

        Key ratios (revenue growth, margins, returns, leverage) formatted like the
        ticker view's signals, each with its within-sector percentile rank. No I/O —
        this is what makes the treemap hover instant."""
        from ..narrate.packet import LABELS, _PCT

        sub = self._cross[self._cross["cik"].astype(int) == int(cik)]
        if sub.empty:
            return None
        r = sub.iloc[0]
        metrics = []
        for f in ("revenue_growth", "op_margin", "roa", "roe", "leverage"):
            v = r.get(f)
            if pd.isna(v):
                continue
            rank = r.get(f"{f}_rank")
            metrics.append({
                "label": LABELS[f].split(" (")[0],
                "value": f"{v * 100:+.1f}%" if f in _PCT else f"{v:.2f}x",
                "rank": int(round(float(rank) * 100)) if pd.notna(rank) else None,
            })
        return {
            "name": str(r.get("name") or ""),
            "ticker": str(r.get("ticker") or "—"),
            "fy": int(r["fy"]) if pd.notna(r.get("fy")) else None,
            "filed": (str(pd.Timestamp(r["available_date"]).date())
                      if pd.notna(r.get("available_date")) else None),
            "pct": int(round(float(r["pct"]) * 100)),
            "decile": _decile(float(r["pct"])),
            "metrics": metrics,
        }

    def market_cap(self, cik: int) -> float | None:
        """Current market cap (Intrinio, LIVE-VIEW ONLY — sizing, never the signal).

        Persistent-cached (hours TTL) then session-cached, so the markets page fetches
        each name's cap at most once a day and reopening is instant."""
        cache = self.__dict__.setdefault("_mktcap_cache", {})
        if cik in cache:
            return cache[cik]
        from ..marketcap import get_market_cap

        res = get_market_cap(int(cik))
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

    # -- positions (PERSONAL holdings — DISPLAY-ONLY; NEVER an input to the signal) -----
    def positions(self) -> list[dict]:
        from ..ops.state import OpsState

        with OpsState() as st:
            return st.positions()

    def set_position(self, cik: int, shares: float, cost: float) -> dict | None:
        """Save the user's holding (personal live-view; firewalled from the score).
        Returns the stored row so the browser can echo the persisted values."""
        from ..ops.state import OpsState

        with OpsState() as st:
            st.position_set(int(cik), float(shares), float(cost))
            return next((p for p in st.positions() if int(p["cik"]) == int(cik)), None)

    def remove_position(self, cik: int) -> None:
        from ..ops.state import OpsState

        with OpsState() as st:
            st.position_remove(int(cik))

    def scorecard(self) -> dict:
        """Book-level scorecard over the names the user tracks (DISPLAY-ONLY; firewalled).

        Covers HELD positions (shares + cost) AND WATCHLIST-only names (followed, no
        shares) — the latter get model standing / distress but no value / P&L. Joins
        them onto the already-scored cross-section; value / P&L use the last close from
        the loaded matrix (no live quota). Never touches the score, the paper book, or
        any trade path."""
        from ..ops.state import OpsState
        from ..portfolio import scorecard as build_scorecard

        with OpsState() as st:
            positions = st.positions()
            watchlist = st.watchlist()
        held = {int(p["cik"]) for p in positions}
        entries = list(positions)
        for w in watchlist:                       # watched-but-not-held → no shares
            if int(w["cik"]) not in held:
                entries.append({"cik": int(w["cik"]), "shares": None,
                                "cost_basis": None, "added_at": w.get("added")})
        prices: dict[int, float] = {}
        for e in entries:
            cik = int(e["cik"])
            col = self.data.ticker_map.get(cik)
            if col and col in self.data.close.columns:
                s = self.data.close[col].dropna()
                if len(s):
                    prices[cik] = float(s.iloc[-1])
        return build_scorecard(entries, self._cross, prices, as_of=self.as_of)

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

    def ask(self, cik: int, question: str, history: list | None = None, llm=None) -> dict:
        """Grounded chat about ONE name — the narration made interactive (web 'ask' box).

        Reuses the narrate plumbing: the SAME deterministic packet plus current
        recalled news, WIDENED with the firewalled display reads (verdict /
        confidence / distress / drawdown / price / flags) so the chat can explain
        every number the ticker page shows. Every numeral in the answer must trace
        to that context or the assistant refuses (assist.core.grounded_answer) —
        never a guess, never advice. ``history`` rides in from the browser (the
        server stays stateless) and never expands the grounding domain."""
        from ..assist.qa import answer_about_company
        from ..narrate.llm import LocalLLM
        from ..narrate.packet import news_context
        from .chart import verdict as call_verdict

        res = dict(self.ticker(cik))
        packet = dict(res["packet"])
        ctx = news_context(self._news_context(int(packet["meta"]["cik"])))
        if ctx:
            packet["context"] = {**(packet.get("context") or {}), "news": ctx}
        res["packet"] = packet
        pr = self.price(int(cik))
        # a chat turn should feel conversational, not narration-length: shorter timeout
        r = answer_about_company(
            res, question, llm or LocalLLM(timeout=180.0), history=history,
            price_summary=(pr or {}).get("summary"),
            verdict=call_verdict((res.get("percentile") or 0) / 100.0))
        m = packet.get("meta") or {}
        return {**r, "cik": int(cik), "ticker": m.get("ticker"), "name": m.get("name"),
                "n_news": len(ctx or [])}

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
