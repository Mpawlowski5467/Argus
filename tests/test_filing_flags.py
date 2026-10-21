"""8-K filing flags: EDGAR full-text + item-code detectors, dedupe, bootstrap, fail-open,
and the firewall — all against RECORDED SEC responses (tests/fixtures/edgar_fts, captured
2026-10-20 through EdgarClient). No live SEC call can happen here: httpx is disarmed and
the real client (real URL building) is fed from the fixtures."""

import copy
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest
from test_ops_monitor import world  # noqa: F401  (module-scoped synthetic world)

from stockscan.config import FILING_FLAG_RULES
from stockscan.edgar.client import EdgarClient
from stockscan.ops.health import filing_flags_freshness
from stockscan.ops.monitor import (
    alert_universe,
    detect_item_flags,
    filing_flag_window,
    run_filing_flags,
    run_monitor,
    search_text_flags,
)
from stockscan.ops.notify import HIGH_SEVERITY
from stockscan.ops.state import OpsState

FIX = Path(__file__).parent / "fixtures" / "edgar_fts"
HCWC = 1948864          # the recorded going-concern filer
OTHER = 1407583         # the second CIK in the recorded OR query
R3_ACC = "0001493152-26-043135"   # its 2026-09-17 8-K: main doc + 4 exhibits matched
RULE = {r["id"]: r for r in FILING_FLAG_RULES}
EMPTY = {"took": 1, "timed_out": False, "hits": {"total": {"value": 0, "relation": "eq"},
                                                  "hits": []}}


def fx(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def qs(url: str) -> dict:
    return dict(parse_qsl(urlsplit(url).query))


class RecordedEdgar(EdgarClient):
    """The REAL client — real URL building, throttle bypassed — with ``get_json`` served
    from recorded fixtures. ``routes``: [(predicate(url), payload | callable | Exception)].
    Records every URL; an unrouted request fails the test."""

    def __init__(self, routes):
        super().__init__(user_agent="argus tests test@example.com", max_rps=0)
        self.routes, self.urls = routes, []

    def get_json(self, url):
        self.urls.append(url)
        for match, payload in self.routes:
            if match(url):
                if isinstance(payload, Exception):
                    raise payload
                return copy.deepcopy(payload(url) if callable(payload) else payload)
        raise AssertionError(f"unrouted request {url}")


def efts_q(fragment):
    return lambda url: "efts.sec.gov" in url and fragment in qs(url).get("q", "")


def submissions(url):
    return "data.sec.gov/submissions/" in url


def subs_with_items(*pairs) -> dict:
    """The recorded submissions JSON with chosen 8-K item strings: pairs of (index, items)."""
    s = fx("submissions_recent.json")
    for i, items in pairs:
        s["filings"]["recent"]["items"][i] = items
    return s


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("live HTTP attempted in a fixture-only test")

    monkeypatch.setattr(httpx.Client, "send", boom)


# --- the request we build matches the recorded backend contract --------------------

def test_full_text_search_builds_the_documented_query():
    c = RecordedEdgar([(lambda u: True, EMPTY)])
    c.full_text_search('"going concern"', forms=("8-K",), ciks=[HCWC, OTHER],
                       startdt="2026-09-01", enddt="2026-10-17", from_=100)
    url = c.urls[-1]
    assert url.startswith("https://efts.sec.gov/LATEST/search-index?")
    assert qs(url) == {"q": '"going concern"', "dateRange": "custom", "startdt": "2026-09-01",
                       "enddt": "2026-10-17", "category": "custom", "forms": "8-K",
                       "ciks": "0001948864,0001407583", "from": "100"}
    c.full_text_search('"going concern"', ciks=[HCWC])
    assert "from" not in qs(c.urls[-1])      # first page: no offset


def test_non_reliance_rule_searches_the_item_402_caption():
    """Bare "non-reliance" matched legal-opinion boilerplate (an EX-5.1) in the 2026-10-20
    dry-run; the rule searches the Item 4.02 caption instead, and stays critical."""
    rule = RULE["non_reliance"]
    assert rule["tier"] == "critical"
    c = RecordedEdgar([(lambda u: True, EMPTY)])
    search_text_flags(c, [HCWC], rule, "2026-10-14", "2026-10-21")
    assert qs(c.urls[-1])["q"] == '"non-reliance on previously issued"'


def test_exhibit_hits_collapse_to_one_filing():
    c = RecordedEdgar([(efts_q("going concern"), fx("efts_going_concern_one_cik.json"))])
    rows, n = search_text_flags(c, [HCWC], RULE["going_concern"], "2026-09-01", "2026-10-17")
    assert n == 1 and len(rows) == 1                     # 5 documents, ONE 8-K
    r = rows[0]
    assert (r["cik"], r["accession"], r["form"], r["file_date"]) == \
        (HCWC, R3_ACC, "8-K", "2026-09-17")
    assert "8-K" in r["docs"] and any(d.startswith("EX-99") for d in r["docs"])


def test_or_query_over_two_ciks_includes_8k_amendments():
    rule = {"id": "or_rule", "q": '"going concern" OR "material weakness"', "tier": "flag",
            "label": "or rule"}
    c = RecordedEdgar([(efts_q("going concern"), fx("efts_or_phrases_two_ciks.json"))])
    rows, _ = search_text_flags(c, [HCWC, OTHER], rule, "2026-09-01", "2026-10-17")
    assert {r["cik"] for r in rows} == {HCWC, OTHER}
    assert len(rows) == 5                                # 14 documents -> 5 filings
    assert "8-K/A" in {r["form"] for r in rows}         # root_forms=8-K includes amendments


def test_pages_to_total_and_drops_names_outside_the_set():
    template = fx("efts_going_concern_one_cik.json")["hits"]["hits"][0]
    hits = []
    for i in range(230):
        h = copy.deepcopy(template)
        h["_source"]["adsh"] = f"0000000000-26-{i:06d}"
        h["_source"]["ciks"] = ["0001948864"] if i % 2 == 0 else ["0009999999"]
        hits.append(h)

    def page(url):
        off = int(qs(url).get("from", 0))
        return {"hits": {"total": {"value": 230, "relation": "eq"},
                         "hits": hits[off:off + 100]}}

    c = RecordedEdgar([(efts_q("going concern"), page)])
    rows, n = search_text_flags(c, [HCWC], RULE["going_concern"], "2026-09-01", "2026-10-17")
    assert n == 3                                        # relevance-ordered: read every page
    assert [qs(u).get("from") for u in c.urls] == [None, "100", "200"]
    assert len(rows) == 115 and {r["cik"] for r in rows} == {HCWC}


def test_item_codes_from_the_submissions_json():
    # recorded rows: [0] 8-K 2026-10-20, [2] S-3, [3] 8-K 2026-10-07
    s = subs_with_items((0, "3.01,9.01"), (2, "4.02"), (3, "1.01,4.02,9.01"))
    rows = detect_item_flags(HCWC, s, FILING_FLAG_RULES)
    got = {(r["rule"], r["file_date"]) for r in rows}
    assert got == {("delisting_301", "2026-10-20"), ("nonreliance_402", "2026-10-07")}
    assert all(r["source"] == "item" for r in rows)      # the S-3's "4.02" is ignored
    assert detect_item_flags(HCWC, fx("submissions_recent.json"), FILING_FLAG_RULES) == []


# --- the nightly pass: window, bootstrap, dedupe, alerts ---------------------------

def test_window_is_fresh_days_plus_outage_catch_up_capped():
    assert filing_flag_window("2026-10-21") == ("2026-10-14", "2026-10-21")
    assert filing_flag_window("2026-10-21", {"enddt": "2026-10-20"}) == ("2026-10-14", "2026-10-21")
    assert filing_flag_window("2026-10-21", {"enddt": "2026-09-21"}) == ("2026-09-18", "2026-10-21")
    assert filing_flag_window("2026-10-21", {"enddt": "2026-01-01"}) == ("2026-07-23", "2026-10-21")


def _routes(subs, going=None, extra=()):
    return list(extra) + [
        (submissions, subs),
        (efts_q("going concern"), going if going is not None else EMPTY),
        (lambda u: "efts.sec.gov" in u, EMPTY),
    ]


def test_new_name_alerts_only_fresh_matches_then_dedupes(tmp_path):
    subs = subs_with_items((1, "4.02,9.01"))           # the 2026-10-19 8-K: non-reliance
    going = fx("efts_going_concern_one_cik.json")       # a 2026-09-17 match: OLD news
    with OpsState(tmp_path / "s.sqlite") as st:
        c = RecordedEdgar(_routes(subs, going))
        out = run_filing_flags(st, [HCWC], c, today="2026-10-21",
                               submissions={HCWC: subs})
        # window 10-14..10-21 for a NEW name: the fresh 4.02 alerts; the 09-17 going-concern
        # match is outside the window (recorded, never alerted)
        alerts = [a for a in st.alerts(unseen_only=False) if a["kind"].startswith("eightk")]
        assert out["alerts"] == 1 and len(alerts) == 1
        a = alerts[0]
        assert a["kind"] == "eightk_critical" and a["cik"] == HCWC
        assert "item 4.02 non-reliance" in a["message"]
        assert "[filing flag — not a trade signal]" in a["message"]
        acc = a["payload"]["accession"]
        assert a["payload"]["url"] == (f"https://www.sec.gov/Archives/edgar/data/{HCWC}/"
                                       f"{acc.replace('-', '')}/{acc}-index.htm")
        assert st.filing_flag_seen(HCWC, R3_ACC)          # the old match is stored
        assert HCWC in st.filing_flag_ciks()
        assert st.kv_get("filing_flags_cursor")["enddt"] == "2026-10-21"

        # the same night again: nothing new
        out2 = run_filing_flags(st, [HCWC], RecordedEdgar(_routes(subs, going)),
                                today="2026-10-21", submissions={HCWC: subs})
        assert out2["alerts"] == 0


def test_one_filing_matching_several_rules_is_one_alert(tmp_path):
    hit = copy.deepcopy(fx("efts_going_concern_one_cik.json"))
    for h in hit["hits"]["hits"]:
        h["_source"]["file_date"] = "2026-10-20"
    subs = fx("submissions_recent.json")
    with OpsState(tmp_path / "s.sqlite") as st:
        c = RecordedEdgar(_routes(subs, hit, extra=[(efts_q("material weakness"), hit)]))
        out = run_filing_flags(st, [HCWC], c, today="2026-10-21", submissions={HCWC: subs})
        assert out["alerts"] == 1
        a = st.alerts(unseen_only=False)[0]
        assert a["payload"]["rules"] == ["going_concern", "material_weakness"]
        assert a["kind"] == "eightk_critical"            # any critical rule escalates
        assert "going concern, material weakness" in a["message"]


def test_flag_tier_only_stays_in_app():
    assert "eightk_critical" in HIGH_SEVERITY and "eightk_flag" not in HIGH_SEVERITY


def test_flag_tier_alert_kind(tmp_path):
    hit = copy.deepcopy(fx("efts_going_concern_one_cik.json"))
    for h in hit["hits"]["hits"]:
        h["_source"]["file_date"] = "2026-10-20"
    subs = fx("submissions_recent.json")
    with OpsState(tmp_path / "s.sqlite") as st:
        c = RecordedEdgar(_routes(subs, None, extra=[(efts_q("restatement"), hit)]))
        run_filing_flags(st, [HCWC], c, today="2026-10-21", submissions={HCWC: subs})
        (a,) = st.alerts(unseen_only=False)
        assert a["kind"] == "eightk_flag" and a["payload"]["tier"] == "flag"


def test_seeded_name_alerts_on_any_new_filing_in_the_window(tmp_path):
    hit = copy.deepcopy(fx("efts_going_concern_one_cik.json"))
    for h in hit["hits"]["hits"]:
        h["_source"]["file_date"] = "2026-10-09"       # 12 days old: in an outage window
    subs = fx("submissions_recent.json")
    with OpsState(tmp_path / "s.sqlite") as st:
        st.mark_filing_flag_ciks([HCWC])                 # an existing name...
        st.kv_set("filing_flags_cursor", {"enddt": "2026-10-08", "errors_in_a_row": 0})
        out = run_filing_flags(st, [HCWC], RecordedEdgar(_routes(subs, hit)),
                               today="2026-10-21", submissions={HCWC: subs})
        assert out["window"] == ["2026-10-05", "2026-10-21"]   # caught up past the outage
        assert out["alerts"] == 1                        # ...is not limited to FRESH_DAYS


def test_a_failing_rule_fails_open_and_holds_the_cursor(tmp_path):
    hit = copy.deepcopy(fx("efts_going_concern_one_cik.json"))
    for h in hit["hits"]["hits"]:
        h["_source"]["file_date"] = "2026-10-20"
    subs = fx("submissions_recent.json")
    with OpsState(tmp_path / "s.sqlite") as st:
        st.kv_set("filing_flags_cursor", {"enddt": "2026-10-19", "errors_in_a_row": 0})
        c = RecordedEdgar(_routes(subs, hit, extra=[
            (efts_q("material weakness"), RuntimeError("EDGAR request failed")),
            (efts_q("restatement"), {"unexpected": "shape"}),
        ]))
        out = run_filing_flags(st, [HCWC], c, today="2026-10-21", submissions={HCWC: subs})
        assert {e["rule"] for e in out["errors"]} == {"material_weakness", "restatement"}
        assert out["alerts"] == 1                         # the healthy rule still alerts
        cur = st.kv_get("filing_flags_cursor")
        assert cur["enddt"] == "2026-10-19" and cur["errors_in_a_row"] == 1
        assert HCWC not in st.filing_flag_ciks()          # not seeded on a partial pass


def test_dry_run_writes_nothing(tmp_path):
    subs = subs_with_items((1, "4.02,9.01"))
    with OpsState(tmp_path / "s.sqlite") as st:
        c = RecordedEdgar(_routes(subs, fx("efts_going_concern_one_cik.json")))
        out = run_filing_flags(st, [HCWC], c, today="2026-10-21", dry_run=True,
                               startdt="2026-09-01")
        assert {p["accession"] for p in out["preview"]} >= {R3_ACC}
        assert out["requests"] == 1 + sum(1 for r in FILING_FLAG_RULES if r.get("q"))
        assert st.alerts(unseen_only=False) == []
        assert not st.filing_flag_seen(HCWC, R3_ACC)
        assert st.kv_get("filing_flags_cursor") is None and st.filing_flag_ciks() == set()


def test_alert_universe_is_watchlist_plus_held(tmp_path):
    with OpsState(tmp_path / "s.sqlite") as st:
        st.watch_add(1, "T1")
        st.position_set(2, 10, 5.0)
        st.position_set(3, 0, 5.0)                        # closed out: not held
        st.position_set(1, 4, 5.0)                        # watched AND held: once
        assert alert_universe(st) == [1, 2]


def test_health_check_reads_the_cursor():
    assert filing_flags_freshness("2026-10-21", None).ok
    assert filing_flags_freshness("2026-10-21", None, enabled=False).level == "info"
    assert filing_flags_freshness("2026-10-21", {"enddt": "2026-10-19"}).ok
    stale = filing_flags_freshness("2026-10-21", {"enddt": "2026-10-10"})
    assert not stale.ok and stale.level == "warn"
    assert not filing_flags_freshness("2026-10-21", {"enddt": "2026-10-20",
                                                     "errors_in_a_row": 3}).ok


# --- monitor wiring + THE FIREWALL ------------------------------------------------

def _monitor_routes():
    hit = copy.deepcopy(fx("efts_going_concern_one_cik.json"))
    for h in hit["hits"]["hits"]:
        h["_source"]["file_date"] = "2026-10-20"
    empty_subs = {"filings": {"recent": {}}}
    return [(lambda u: f"CIK{HCWC:010d}" in u, fx("submissions_recent.json")),
            (submissions, empty_subs),
            (efts_q("going concern"), hit),
            (lambda u: "efts.sec.gov" in u, EMPTY)]


def test_monitor_flags_held_names_and_never_touches_the_signal(world, tmp_path):  # noqa: F811
    def run(flags: bool):
        with OpsState(tmp_path / f"s_{flags}.sqlite") as st:
            st.watch_add(1, "T1")
            st.position_set(HCWC, 100, 2.0)               # held, NOT watched
            deltas = run_monitor(st, data=world["data"], artifact=world["artifact"],
                                 narrate=False, edgar=True, as_of="2024-09-30",
                                 filing_flags=flags, client=RecordedEdgar(_monitor_routes()),
                                 today="2026-10-21")
            return deltas, st.get_signal(1), st.alerts(unseen_only=False)

    d_on, sig_on, alerts_on = run(True)
    d_off, sig_off, alerts_off = run(False)
    assert d_on["filing_flags"]["names"] == 2             # watch ∪ held
    assert any(a["kind"] == "eightk_critical" and a["cik"] == HCWC for a in alerts_on)
    assert not any(a["kind"].startswith("eightk") for a in alerts_off)
    # THE FIREWALL: flags add their own alerts and nothing else — the recorded signal and
    # every other alert (percentile moves, risk escalations, filings) are identical
    assert sig_on == sig_off

    def others(alerts):
        return sorted((a["kind"], a["cik"], a["message"]) for a in alerts
                      if not a["kind"].startswith("eightk"))

    assert others(alerts_on) == others(alerts_off)


def test_held_only_book_still_gets_filing_flags(world, tmp_path):  # noqa: F811
    with OpsState(tmp_path / "s.sqlite") as st:
        st.position_set(HCWC, 100, 2.0)
        deltas = run_monitor(st, data=world["data"], artifact=world["artifact"],
                             narrate=False, edgar=True, as_of="2024-09-30",
                             filing_flags=True, client=RecordedEdgar(_monitor_routes()),
                             today="2026-10-21")
        assert deltas["note"] == "watchlist empty — filing flags only"
        assert deltas["filing_flags"]["alerts"] == 1
        assert st.get_signal(HCWC) is None                # no signal pass for held-only


def test_edgar_off_skips_filing_flags_without_any_request(world, tmp_path):  # noqa: F811
    with OpsState(tmp_path / "s.sqlite") as st:
        st.watch_add(1, "T1")
        c = RecordedEdgar([])
        deltas = run_monitor(st, data=world["data"], artifact=world["artifact"],
                             narrate=False, edgar=False, as_of="2024-09-30", client=c)
        assert "filing_flags" not in deltas and c.urls == []
