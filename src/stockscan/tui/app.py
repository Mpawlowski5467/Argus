"""argus — the Textual app: status bar + four views over the scanner.

Read-mostly (the only writes are watchlist/alert curation). Heavy data loads
once in a background worker so the app opens instantly; narration (the slow LLM
path) runs in its own worker and never blocks the UI. Pass ``adata=`` to inject
a fake facade for headless tests.
"""

from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    ContentSwitcher, DataTable, Footer, Input, Select, Sparkline, Static,
)

from .chart import candle_panel, price_chart, verdict
from .logo import GLYPH, TAGLINE


class StatusBar(Static):
    """The tmux-style status line: freshness, vintage, nightly, alerts."""

    def show_loading(self) -> None:
        self.update(f"[b]{GLYPH} argus[/b]   [dim]loading data …[/dim]")

    def show_status(self, s: dict) -> None:
        reg = "green" if s.get("artifact_registered") else "red"
        al = s.get("unseen_alerts") or 0
        ns = s.get("nightly_status") or "never"
        ns_col = {"ok": "green", "noop": "green", "degraded": "yellow",
                  "failed": "red"}.get(ns, "dim")
        self.update(
            f"[b]{GLYPH} argus[/b]   "
            f"as-of {s.get('as_of') or '—'}   "
            f"fund {s.get('fund_quarter') or '—'}   "
            f"vintage [{reg}]{s.get('vintage') or '—'}[/{reg}]   "
            f"nightly [{ns_col}]{ns}[/{ns_col}]   "
            f"alerts [{'yellow' if al else 'dim'}]{al}[/]"
        )


class ScanView(Vertical):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="search ticker or name — Enter opens  ( / to focus )", id="search")
        yield Select([("all", "all")], id="sector", value="all", allow_blank=False)
        yield DataTable(id="scan-table", cursor_type="row", zebra_stripes=True)

    def populate(self, adata) -> None:
        self.adata = adata
        self.query_one("#sector", Select).set_options([(s, s) for s in adata.sectors()])
        dt = self.query_one(DataTable)
        if not dt.columns:
            dt.add_columns("#", "ticker", "name", "sector", "model", "dec", "fy")
        self._fill(adata.scan("all"))

    def _fill(self, rows: list[dict]) -> None:
        dt = self.query_one(DataTable)
        dt.clear()
        for r in rows:
            pct = f"{r['pct']}%"
            dt.add_row(str(r["rank"]), r["ticker"], r["name"], r["sector"],
                       pct, str(r["decile"]), str(r["fy"] or "—"), key=str(r["cik"]))

    def _sector(self) -> str:
        return str(self.query_one("#sector", Select).value)

    def _requery(self) -> None:
        """Re-fill the table from the current search text (falling back to sector)."""
        q = self.query_one("#search", Input).value.strip()
        self._fill(self.adata.search(q) if q else self.adata.scan(self._sector()))

    @on(Select.Changed, "#sector")
    def _sector_changed(self, e: Select.Changed) -> None:
        if getattr(self, "adata", None) is not None:
            self._requery()

    @on(Input.Changed, "#search")
    def _search_changed(self, e: Input.Changed) -> None:
        if getattr(self, "adata", None) is not None:
            self._requery()

    @on(Input.Submitted, "#search")
    def _search_submit(self, e: Input.Submitted) -> None:
        if getattr(self, "adata", None) is None:
            return
        hit = self.adata.resolve(e.value.strip())
        if hit is not None:
            self.app.show_ticker(hit["cik"])
        else:
            self.app.notify(f"no match for {e.value!r}", severity="warning")

    @on(DataTable.RowSelected, "#scan-table")
    def _row(self, e: DataTable.RowSelected) -> None:
        if e.row_key is not None and e.row_key.value is not None:
            self.app.show_ticker(int(e.row_key.value))


class TickerView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("[dim]select a name in scan (Enter) or search ( / ) to drill in[/dim]",
                     id="tk-body")

    def show(self, adata, cik: int) -> None:
        self.adata = adata
        self._cik = int(cik)
        body = self.query_one("#tk-body", Static)
        try:
            res = adata.ticker(cik)
        except Exception as exc:  # dead / stale / unresolvable — surface, don't crash
            body.update(f"[red]{exc}[/red]")
            return
        self._packet = res["packet"]
        self._res = res
        try:
            self._price = adata.price(cik)
        except Exception:
            self._price = None
        try:
            self._ohlc = adata.ohlc(cik)
        except Exception:
            self._ohlc = None
        try:
            self._watched = adata.is_watched(cik)
        except Exception:
            self._watched = False
        self._events = None          # None = still loading; [] = none found
        self._news = None
        self._quote = None
        self._narr_tier = None
        if not hasattr(self, "_chart_style"):
            self._chart_style = "candle"
        body.update(self._build(res))
        self.app.load_enrichment(self._cik)  # EDGAR filings + Intrinio news + live quote

    # -- section renderers -------------------------------------------------------
    @staticmethod
    def _chg(x):
        if x is None:
            return "[dim]—[/dim]"
        c = "green" if x >= 0 else "red"
        return f"[{c}]{x:+.1f}%[/{c}]"

    def _price_block(self) -> list[str]:
        pr = getattr(self, "_price", None)
        if not pr or pr.get("series") is None or len(pr["series"]) < 2:
            return ["", "[dim]— no price history —[/dim]"]
        s = pr["summary"]
        adv = f" · ADV ${s['adv'] / 1e6:,.1f}M" if s.get("adv") else ""
        out = ["",
               f"[b]{s['last']:,.2f}[/b]   1m {self._chg(s['chg_1m'])}   3m {self._chg(s['chg_3m'])}   "
               f"1y {self._chg(s['chg_1y'])}   [dim]close · 52wk {s['lo_52w']:,.0f}–"
               f"{s['hi_52w']:,.0f}{adv}[/dim]"]
        q = getattr(self, "_quote", None)
        if q and q.get("last") is not None:
            t = (q.get("time") or "")[11:16]
            auto = "[green]● auto[/green] · " if getattr(self.app, "_auto_timer", None) else ""
            out.append(f"[b]live [b]{q['last']:,.2f}[/b][/b] {self._chg(q.get('chg_pct'))}   "
                       f"[dim]{auto}bid {q.get('bid') or '—'} · ask {q.get('ask') or '—'} · {t}Z · "
                       f"l refresh · a auto[/dim]")
        out += self._chart_block()
        return out

    def _chart_block(self) -> list[str]:
        style = getattr(self, "_chart_style", "candle")
        ohlc = getattr(self, "_ohlc", None)
        pr = getattr(self, "_price", None)
        if style == "candle" and ohlc is not None and len(ohlc) > 1:
            chart = candle_panel(ohlc["open"].tolist(), ohlc["high"].tolist(),
                                 ohlc["low"].tolist(), ohlc["close"].tolist(),
                                 ohlc["volume"].tolist(), width=64, height=12, vheight=3)
        elif pr and pr.get("series") is not None and len(pr["series"]) > 1:
            col = "green" if (pr["summary"].get("chg_1y") or 0) >= 0 else "red"
            chart = price_chart(pr["series"].iloc[-252:], width=64, height=12, color=col)
        else:
            chart = "[dim]— no price history —[/dim]"
        return ["", chart, f"[dim]{style} chart · press c to switch[/dim]"]

    def _news_block(self) -> list[str]:
        out = ["", "[dim]news (Intrinio headlines)[/dim]"]
        n = getattr(self, "_news", None)
        if n is None:
            out.append("  [dim]loading …[/dim]")
        elif not n:
            out.append("  [dim]no recent headlines[/dim]")
        else:
            for a in n:
                src = f"  [dim]· {a['source']}[/dim]" if a.get("source") else ""
                out.append(f"  [cyan]{a['date']}[/cyan]  {a['title'][:78]}{src}")
        return out

    def _events_block(self) -> list[str]:
        out = ["", "[dim]recent filings & events (SEC EDGAR)[/dim]"]
        ev = getattr(self, "_events", None)
        if ev is None:
            out.append("  [dim]loading …[/dim]")
        elif not ev:
            out.append("  [dim]no recent newsworthy filings[/dim]")
        else:
            for e in ev:
                out.append(f"  [cyan]{e['filed_date']}[/cyan]  {e['form']:<9}[dim]{e['label']}[/dim]")
        return out

    def _build(self, res: dict) -> str:
        p = res["packet"]
        m, mm, f = p["meta"], p["model"], res["flags"]
        insample = "  [yellow][in-sample][/yellow]" if f["in_sample"] else ""
        liq = "" if f["liquidity_pass"] else "  [red][below liquidity floor][/red]"
        tk = m.get("ticker") or res.get("column") or ""
        v = verdict((mm.get("percentile") or 0) / 100.0)
        watch = ("  [yellow]★ watching[/yellow]" if getattr(self, "_watched", False)
                 else "  [dim]☆ w to watch[/dim]")
        out = [
            f"[b]{m['name']}[/b]   [dim]·[/dim]   {tk}   [dim]·[/dim]   {m['sector']}{watch}",
            f"[reverse {v['color']}] {v['call']} [/reverse {v['color']}]  [dim]{v['reason']}[/dim]",
        ]
        out += self._price_block()
        out += [
            "",
            f"[dim]FY{m['fiscal_year']} 10-K filed {f['filed_date']} "
            f"(usable {f['available_date']}, {f['staleness_days']}d old)[/dim]",
            "",
            f"model signal  [b]{mm['percentile']}th pct[/b] · decile {mm['decile']}/10 "
            f"· score {mm['score']:+.4f} · trained through {mm['trained_through']}{insample}{liq}",
            "",
            "[dim]drivers (SHAP — exact decomposition)[/dim]",
        ]
        for d in mm.get("drivers", []):
            col = "green" if d["direction"] == "supports" else "red"
            out.append(f"  {d['label']:<24} [{col}]{d['contribution']:+.4f}  {d['direction']}[/{col}]")
        out += ["", "[dim]signals[/dim]"]
        for s in p.get("signals", []):
            read = s.get("read", "")
            col = {"supports": "green", "detracts": "red"}.get(read, "dim")
            val = f"{s['value']}{s.get('unit', '')}"
            pr = f"{s['pct_rank']}th" if s.get("pct_rank") is not None else "—"
            out.append(f"  {s['label']:<24} {val:<11} {pr:<7} [{col}]{read}[/{col}]")
        out += self._news_block()
        out += self._events_block()
        tier = getattr(self, "_narr_tier", None)
        hint = (f"narrated with the local model ({tier})" if tier
                else "press n to (re)narrate with the local model")
        out += ["", "[dim]narration[/dim]", res.get("narrative", ""), f"\n[dim]{hint}[/dim]"]
        return "\n".join(out)

    def _rerender(self) -> None:
        if getattr(self, "_res", None) is not None:
            self.query_one("#tk-body", Static).update(self._build(self._res))

    def set_enrichment(self, cik: int, events, news, quote) -> None:
        if getattr(self, "_cik", None) != cik or getattr(self, "_res", None) is None:
            return
        self._events, self._news, self._quote = events, news, quote
        self._rerender()

    def set_quote(self, cik: int, quote) -> None:
        if getattr(self, "_cik", None) != cik:
            return
        self._quote = quote
        self._rerender()

    def toggle_chart(self) -> None:
        self._chart_style = "line" if getattr(self, "_chart_style", "candle") == "candle" else "candle"
        self._rerender()

    def refresh_quote(self) -> None:
        if getattr(self, "_cik", None) is not None:
            self.app.load_quote(self._cik)

    def toggle_watch(self) -> None:
        if getattr(self, "_cik", None) is None:
            return
        try:
            self._watched = self.adata.toggle_watch(self._cik)
        except Exception as exc:
            self.app.notify(f"watchlist error: {exc}", severity="error")
            return
        self._rerender()
        self.app.notify(("★ added to" if self._watched else "removed from") + " watchlist")
        self.app.refresh_watch()

    def narrate(self) -> None:
        if getattr(self, "_packet", None) is not None:
            self.app.narrate_ticker(self._packet)

    def set_narration(self, narr: dict) -> None:
        """Re-render the open ticker with an upgraded (LLM) narration."""
        if getattr(self, "_res", None) is None:
            return
        self._res = {**self._res, "narrative": narr.get("narrative", "")}
        self._narr_tier = narr.get("tier", "?")
        self.query_one("#tk-body", Static).update(self._build(self._res))


class WatchView(Vertical):
    def compose(self) -> ComposeResult:
        yield DataTable(id="watch-table", cursor_type="row")
        yield Static("alerts", classes="section")
        yield DataTable(id="alert-table")

    def populate(self, adata) -> None:
        w = adata.watch()
        wt = self.query_one("#watch-table", DataTable)
        wt.clear(columns=True)
        wt.add_columns("ticker", "model", "Δ 30d", "last 10-K", "flag")
        for r in w["rows"]:
            pct = f"{r['pct']}%" if r["pct"] is not None else "—"
            delta = (f"{r['delta']:+d}" if r["delta"] is not None else "—")
            wt.add_row(r["ticker"], pct, delta, r["last_filing"] or "—",
                       r["flag"] or "—", key=str(r["cik"]))
        if not w["rows"]:
            wt.add_row("[dim]watchlist empty — use `ops.py watch add`[/dim]", "", "", "", "")
        at = self.query_one("#alert-table", DataTable)
        at.clear(columns=True)
        at.add_columns("", "when", "kind", "message")
        for a in w["alerts"]:
            mark = "[yellow]*[/yellow]" if not a["seen"] else " "
            at.add_row(mark, a["created"][:16], a["kind"], a["message"])

    @on(DataTable.RowSelected, "#watch-table")
    def _row(self, e: DataTable.RowSelected) -> None:
        if e.row_key is not None and str(e.row_key.value).isdigit():
            self.app.show_ticker(int(e.row_key.value))


class PaperView(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(id="paper-body")
        yield Sparkline([0.0], id="paper-spark")

    def populate(self, adata) -> None:
        body = self.query_one("#paper-body", Static)
        rep = adata.paper()
        if rep is None:
            body.update("[dim]no baseline frozen — run `ops.py paper freeze`[/dim]")
            return
        b = rep["baseline"]
        live = rep.get("live_mean_ic")
        deg = rep.get("degraded")
        deg_txt = ("[green]within tolerance[/green]" if deg is False
                   else "[red]DEGRADED[/red]" if deg else "[dim]pending[/dim]")
        body.update(
            f"[dim]frozen[/dim]   vintage, expected IC [cyan]{b['expected_ic']:+.4f}[/cyan] "
            f"· spread [cyan]{b['expected_spread_63d']:+.4f}[/cyan]/63d\n"
            f"[dim]live[/dim]     {rep['months_scored_oos']} OOS month(s) scored "
            f"· mean IC {('%+.4f' % live) if live is not None else '—'} "
            f"· {rep['months_scored_in_sample']} in-sample (excluded)\n"
            f"[dim]gate[/dim]     {deg_txt}   {rep.get('note', '')}"
        )
        ics = [m["h63"]["rank_ic"] for m in rep["months"]
               if m.get("h63") and not m.get("in_sample")]
        if ics:
            self.query_one(Sparkline).data = ics


HELP_TEXT = """[b]argus[/b] — the all-seeing scanner

[b]views[/b]     1 scan    2 ticker    3 watch    4 paper
[b]find[/b]      /  search  [dim](type a ticker or name, Enter opens)[/dim]

[b]ticker page[/b]
  c   candle ↔ line chart
  l   refresh live quote        a   toggle ~12s auto-refresh
  w   add / remove from watchlist
  n   (re)narrate with the local model

[b]app[/b]       t theme    r refresh data    ? help    q quit

[dim]esc / ? / q to close[/dim]"""


class HelpScreen(ModalScreen):
    """A dismissable key-bindings cheatsheet."""

    BINDINGS = [Binding("escape,question_mark,q", "close", "close")]

    def compose(self) -> ComposeResult:
        yield Static(HELP_TEXT, id="help-box")

    def action_close(self) -> None:
        self.dismiss()

    def on_click(self) -> None:
        self.dismiss()


class ArgusApp(App):
    CSS_PATH = "argus.tcss"
    TITLE = "argus"
    # focus the scan table (not the search box) at boot, so single-key bindings
    # (1-4, t, n, r, q) work immediately; "/" grabs the search box when wanted.
    AUTO_FOCUS = "#scan-table"
    BINDINGS = [
        Binding("1", "view('scan')", "scan"),
        Binding("2", "view('ticker')", "ticker"),
        Binding("3", "view('watch')", "watch"),
        Binding("4", "view('paper')", "paper"),
        Binding("/", "search", "search"),
        Binding("c", "chart", "chart"),
        Binding("w", "watch_toggle", "±watch"),
        Binding("?", "help", "help"),
        Binding("q", "quit", "quit"),
        # secondary — still reachable, hidden from the footer to keep it readable
        Binding("l", "live", "live", show=False),
        Binding("a", "autolive", "auto-live", show=False),
        Binding("n", "narrate", "narrate", show=False),
        Binding("r", "refresh", "refresh", show=False),
        Binding("t", "toggle_theme", "theme", show=False),
    ]

    def __init__(self, adata=None):
        super().__init__()
        self._injected = adata
        self.adata = None
        self._auto_timer = None

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status")
        with ContentSwitcher(initial="scan", id="switcher"):
            yield ScanView(id="scan")
            yield TickerView(id="ticker")
            yield WatchView(id="watch")
            yield PaperView(id="paper")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        self.query_one(StatusBar).show_loading()
        self.sub_title = TAGLINE
        if self._injected is not None:
            self._loaded(self._injected)
        else:
            self._load()

    @work(thread=True)
    def _load(self) -> None:
        from .data import ArgusData

        try:
            adata = ArgusData.load()
        except Exception as exc:
            msg = f"load failed: {exc}"  # bind now — `exc` is cleared after this block
            self.call_from_thread(self.notify, msg, severity="error")
            self.call_from_thread(
                lambda: self.query_one(StatusBar).update(f"[red]{msg}[/red]"))
            return
        self.call_from_thread(self._loaded, adata)

    def _loaded(self, adata) -> None:
        self.adata = adata
        self.query_one(ScanView).populate(adata)
        self.query_one(WatchView).populate(adata)
        self.query_one(PaperView).populate(adata)
        self.query_one(StatusBar).show_status(adata.status())

    def show_ticker(self, cik: int) -> None:
        if self.adata is None:
            return
        self.query_one(TickerView).show(self.adata, cik)
        self.query_one(ContentSwitcher).current = "ticker"
        self.set_focus(None)   # don't let focus drift to the hidden search box

    @work(thread=True)
    def narrate_ticker(self, packet: dict) -> None:
        try:
            res = self.adata.narrate(packet)
        except Exception as exc:
            self.call_from_thread(self.notify, f"narration failed: {exc}", severity="warning")
            return
        self.call_from_thread(self._show_narration, res)

    def _show_narration(self, res: dict) -> None:
        self.query_one(TickerView).set_narration(res)

    @work(thread=True)
    def load_enrichment(self, cik: int) -> None:
        def _try(fn, default):
            try:
                return fn()
            except Exception:
                return default
        ev = _try(lambda: self.adata.events(cik), [])
        news = _try(lambda: self.adata.news(cik), [])
        quote = _try(lambda: self.adata.live_quote(cik), None)
        self.call_from_thread(self._show_enrichment, cik, ev, news, quote)

    def _show_enrichment(self, cik: int, ev, news, quote) -> None:
        self.query_one(TickerView).set_enrichment(cik, ev, news, quote)

    @work(thread=True, group="quote", exclusive=True)
    def load_quote(self, cik: int) -> None:
        try:
            q = self.adata.live_quote(cik, refresh=True)
        except Exception:
            q = None
        self.call_from_thread(self._show_quote, cik, q)

    def _show_quote(self, cik: int, q) -> None:
        self.query_one(TickerView).set_quote(cik, q)

    # -- actions -----------------------------------------------------------------
    _VIEW_FOCUS = {"scan": "#scan-table", "watch": "#watch-table"}

    def _focus_view(self, name: str) -> None:
        """Focus the shown view's table (or nothing) so single-key bindings keep
        working — otherwise Textual drifts focus back to the hidden search box,
        which then swallows every keypress."""
        sel = self._VIEW_FOCUS.get(name)
        if sel is not None:
            try:
                self.query_one(sel).focus()
                return
            except Exception:
                pass
        self.set_focus(None)

    def action_view(self, name: str) -> None:
        self.query_one(ContentSwitcher).current = name
        self._focus_view(name)

    def action_search(self) -> None:
        self.query_one(ContentSwitcher).current = "scan"
        self.query_one("#search", Input).focus()

    def action_toggle_theme(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    def action_narrate(self) -> None:
        if self.query_one(ContentSwitcher).current == "ticker":
            self.query_one(TickerView).narrate()

    def action_chart(self) -> None:
        if self.query_one(ContentSwitcher).current == "ticker":
            self.query_one(TickerView).toggle_chart()

    def action_live(self) -> None:
        if self.query_one(ContentSwitcher).current == "ticker":
            self.query_one(TickerView).refresh_quote()

    def action_watch_toggle(self) -> None:
        if self.query_one(ContentSwitcher).current == "ticker":
            self.query_one(TickerView).toggle_watch()

    def refresh_watch(self) -> None:
        if self.adata is not None:
            self.query_one(WatchView).populate(self.adata)

    def action_autolive(self) -> None:
        """Toggle ~12s live-quote streaming on the ticker page (quota-aware, off by default)."""
        if self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer = None
            self.notify("live auto-refresh off")
        else:
            self._auto_timer = self.set_interval(12.0, self._auto_tick)
            self.notify("live auto-refresh on (~12s)")
            self._auto_tick()

    def _auto_tick(self) -> None:
        if self.adata is None or self.query_one(ContentSwitcher).current != "ticker":
            return
        tv = self.query_one(TickerView)
        if getattr(tv, "_cik", None) is not None:
            self.load_quote(tv._cik)

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    @work(thread=True)
    def _do_refresh(self) -> None:
        self.adata.refresh()
        self.call_from_thread(self._loaded, self.adata)
        self.call_from_thread(self.notify, "refreshed")

    def action_refresh(self) -> None:
        if self.adata is not None:
            self._do_refresh()

    def action_quit(self) -> None:
        self.exit()

    def on_unmount(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.stop()
        if self.adata is not None and self._injected is None:
            self.adata.close()


def main() -> int:
    ArgusApp().run()
    return 0
