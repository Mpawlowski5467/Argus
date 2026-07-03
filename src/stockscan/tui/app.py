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
from textual.widgets import ContentSwitcher, DataTable, Footer, Select, Sparkline, Static

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

    @on(Select.Changed, "#sector")
    def _sector_changed(self, e: Select.Changed) -> None:
        if getattr(self, "adata", None) is not None:
            self._fill(self.adata.scan(str(e.value)))

    @on(DataTable.RowSelected, "#scan-table")
    def _row(self, e: DataTable.RowSelected) -> None:
        if e.row_key is not None and e.row_key.value is not None:
            self.app.show_ticker(int(e.row_key.value))


class TickerView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("[dim]select a name in scan (Enter) to drill in[/dim]", id="tk-body")

    def show(self, adata, cik: int) -> None:
        self.adata = adata
        body = self.query_one("#tk-body", Static)
        try:
            res = adata.ticker(cik)
        except Exception as exc:  # dead / stale / unresolvable — surface, don't crash
            body.update(f"[red]{exc}[/red]")
            return
        self._packet = res["packet"]
        self._res = res
        body.update(self._build(res))

    def _build(self, res: dict) -> str:
        m, mm, f = res["packet"]["meta"], res["packet"]["model"], res["flags"]
        insample = "  [yellow][in-sample][/yellow]" if f["in_sample"] else ""
        liq = "" if f["liquidity_pass"] else "  [red][below liquidity floor][/red]"
        out = [
            f"[b]{m['name']}[/b]  ·  {m.get('ticker') or res.get('column') or ''}  ·  {m['sector']}",
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
        for s in res["packet"].get("signals", []):
            read = s.get("read", "")
            col = {"supports": "green", "detracts": "red"}.get(read, "dim")
            val = f"{s['value']}{s.get('unit', '')}"
            pr = f"{s['pct_rank']}th" if s.get("pct_rank") is not None else "—"
            out.append(f"  {s['label']:<24} {val:<11} {pr:<7} [{col}]{read}[/{col}]")
        out += ["", "[dim]narration[/dim]", res.get("narrative", "")]
        out.append("\n[dim]press n to (re)narrate with the local model[/dim]")
        return "\n".join(out)

    def narrate(self) -> None:
        if getattr(self, "_packet", None) is not None:
            self.app.narrate_ticker(self._packet)

    def set_narration(self, narr: dict) -> None:
        """Re-render the open ticker with an upgraded (LLM) narration."""
        if getattr(self, "_res", None) is None:
            return
        self._res = {**self._res, "narrative": narr.get("narrative", "")}
        text = self._build(self._res).replace(
            "press n to (re)narrate with the local model",
            f"narrated with the local model ({narr.get('tier', '?')})")
        self.query_one("#tk-body", Static).update(text)


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


class ArgusApp(App):
    CSS_PATH = "argus.tcss"
    TITLE = "argus"
    BINDINGS = [
        Binding("1", "view('scan')", "scan"),
        Binding("2", "view('ticker')", "ticker"),
        Binding("3", "view('watch')", "watch"),
        Binding("4", "view('paper')", "paper"),
        Binding("t", "toggle_theme", "theme"),
        Binding("n", "narrate", "narrate"),
        Binding("r", "refresh", "refresh"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, adata=None):
        super().__init__()
        self._injected = adata
        self.adata = None

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

    # -- actions -----------------------------------------------------------------
    def action_view(self, name: str) -> None:
        self.query_one(ContentSwitcher).current = name

    def action_toggle_theme(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    def action_narrate(self) -> None:
        if self.query_one(ContentSwitcher).current == "ticker":
            self.query_one(TickerView).narrate()

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
        if self.adata is not None and self._injected is None:
            self.adata.close()


def main() -> int:
    ArgusApp().run()
    return 0
