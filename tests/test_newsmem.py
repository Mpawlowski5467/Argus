"""News-memory tests: extraction shaping/guard, curation, store upsert/dedup/recall,
ingest quota cache. All offline — the LLM and the Intrinio fetch are mocked."""

import json

from stockscan.newsmem.curate import credibility, curate, dedup_key, is_good
from stockscan.newsmem.extract import (
    extract_article,
    heuristic_extraction,
    shape_extraction,
)
from stockscan.newsmem.store import NewsStore


def _article(aid, title, summary="", source="reuters.com", date="2026-06-01T00:00:00Z"):
    return {"id": aid, "title": title, "summary": summary, "source": source,
            "date": date[:10], "publication_date": date, "url": f"https://x/{aid}"}


# --- extraction guard ------------------------------------------------------------

def test_shape_extraction_rejects_number_not_in_article():
    art = {"title": "Acme acquires a rival", "summary": "No figures disclosed."}
    parsed = {"takeaway": "Acme buys rival for 500 million dollars",
              "event_type": "M&A", "materiality": 0.8}
    ext, viol = shape_extraction(parsed, art, model="m")
    assert any(str(v).startswith("fabricated-number") for v in viol)


def test_shape_extraction_coerces_and_accepts_clean():
    art = {"title": "Acme names new CEO", "summary": ""}
    parsed = {"takeaway": "Acme appointed a new chief executive",
              "event_type": "not-a-type", "sentiment": "meh", "materiality": 5.0,
              "entities": ["Acme"], "keywords": ["ceo", "leadership"]}
    ext, viol = shape_extraction(parsed, art, model="m")
    assert viol == []
    assert ext["event_type"] == "other"          # unknown type coerced
    assert ext["sentiment"] == "neutral"         # unknown sentiment coerced
    assert ext["materiality"] == 1.0             # clamped to [0,1]
    assert ext["model"] == "m"


def test_shape_extraction_allows_real_article_number():
    art = {"title": "Acme Q3 revenue rises", "summary": ""}
    parsed = {"takeaway": "Acme reported Q3 results", "materiality": 0.6}
    _, viol = shape_extraction(parsed, art, model="m")
    assert viol == []                            # the 3 in Q3 is in the article


def test_heuristic_extraction_is_number_free_and_typed():
    ext = heuristic_extraction(_article("a1", "MegaCorp to acquire Rival Inc for $9 billion"))
    assert ext["event_type"] == "M&A"
    assert ext["model"] == "heuristic"
    assert not any(ch.isdigit() for ch in ext["takeaway"])   # every numeral stripped


def test_extract_article_retries_then_falls_back_number_free():
    calls = {"n": 0}

    def always_fabricates(system, user):
        calls["n"] += 1
        return json.dumps({"takeaway": "guidance cut to 42 dollars", "materiality": 0.7})

    art = {"title": "Acme cuts guidance", "summary": ""}
    ext = extract_article(art, llm=always_fabricates, model="m", max_retries=1)
    assert calls["n"] == 2                        # tried once, retried once
    assert ext["model"] == "heuristic"            # then deterministic fallback
    assert "fallback_from" in ext
    assert not any(ch.isdigit() for ch in ext["takeaway"])


def test_extract_article_accepts_clean_llm_reply():
    def good(system, user):
        return "```json\n" + json.dumps(
            {"takeaway": "Acme acquired a competitor", "event_type": "M&A",
             "materiality": 0.8}) + "\n```"

    ext = extract_article({"title": "Acme buys rival", "summary": ""}, llm=good, model="m")
    assert ext["event_type"] == "M&A" and ext["model"] == "m"


def test_extract_article_without_llm_is_heuristic():
    ext = extract_article(_article("a1", "Acme sues supplier over contract"))
    assert ext["model"] == "heuristic" and ext["event_type"] == "litigation"


# --- curation --------------------------------------------------------------------

def test_credibility_ranks_press_and_wire():
    assert credibility("reuters.com") > credibility("businesswire.com")
    assert credibility("unknown-blog.example") == 0.5


def test_curate_drops_low_materiality_and_wire_spam_keeps_material():
    rows = [
        {"id": "1", "title": "Reuters: real earnings beat", "source": "reuters.com",
         "materiality": 0.7, "date": "2026-06-01"},
        {"id": "2", "title": "Sponsored listicle 5 stocks", "source": "247wallst.com",
         "materiality": 0.2, "date": "2026-06-02"},                       # low materiality -> drop
        {"id": "3", "title": "Wire promo blurb", "source": "prnewswire.com",
         "materiality": 0.4, "date": "2026-06-03"},                       # wire + modest -> drop
        {"id": "4", "title": "Major merger announced on the wire", "source": "prnewswire.com",
         "materiality": 0.85, "date": "2026-06-04"},                      # decisive -> keep
    ]
    kept = {r["id"] for r in curate(rows)}
    assert kept == {"1", "4"}


def test_curate_dedups_by_normalized_title():
    rows = [
        {"id": "a", "title": "Acme wins big contract!", "source": "reuters.com",
         "materiality": 0.6, "date": "2026-06-01"},
        {"id": "b", "title": "ACME wins big contract", "source": "bloomberg.com",
         "materiality": 0.8, "date": "2026-06-02"},
    ]
    kept = curate(rows)
    assert len(kept) == 1 and kept[0]["id"] == "b"    # higher materiality survives
    assert dedup_key("Acme wins big contract!") == dedup_key("ACME wins big contract")


# --- store: upsert / recall / context ---------------------------------------------

def test_store_upsert_is_idempotent(tmp_path):
    with NewsStore(tmp_path / "news.sqlite") as st:
        arts = [_article("n1", "Acme earnings beat"), _article("n2", "Acme new CEO")]
        assert len(st.upsert_articles(320193, "AAPL", arts)) == 2
        assert st.upsert_articles(320193, "AAPL", arts) == []      # replay -> no new rows


def test_store_recall_structured_and_keyword(tmp_path):
    with NewsStore(tmp_path / "news.sqlite") as st:
        arts = [
            _article("e1", "Acme quarterly earnings beat estimates", date="2026-05-01T00:00:00Z"),
            _article("m1", "Acme to acquire Rival", date="2026-06-01T00:00:00Z"),
        ]
        st.upsert_articles(1, "ACME", arts)
        st.put_extraction("e1", "v1", {"event_type": "earnings", "takeaway": "beat",
                                       "materiality": 0.6, "keywords": ["earnings"]})
        st.put_extraction("m1", "v1", {"event_type": "M&A", "takeaway": "acquisition",
                                       "materiality": 0.9, "keywords": ["merger"]})
        # structured: by event_type
        assert [r["id"] for r in st.recall(1, event_types=["M&A"])] == ["m1"]
        # ordered by materiality desc
        assert [r["id"] for r in st.recall(1)] == ["m1", "e1"]
        # keyword substring
        assert [r["id"] for r in st.recall(1, keywords="acquire")] == ["m1"]
        # since filter
        assert [r["id"] for r in st.recall(1, since="2026-05-15")] == ["m1"]


def test_store_context_for_mixes_recent_and_notable(tmp_path):
    with NewsStore(tmp_path / "news.sqlite") as st:
        # one very-material OLD event + several recent minor ones
        st.upsert_articles(1, "ACME", [
            _article("old", "Acme landmark acquisition", date="2026-01-01T00:00:00Z"),
            _article("r1", "Acme minor update one", date="2026-06-01T00:00:00Z"),
            _article("r2", "Acme minor update two", date="2026-06-02T00:00:00Z"),
        ])
        st.put_extraction("old", "v1", {"event_type": "M&A", "takeaway": "landmark deal",
                                        "materiality": 0.95})
        st.put_extraction("r1", "v1", {"event_type": "other", "takeaway": "minor one",
                                       "materiality": 0.5})
        st.put_extraction("r2", "v1", {"event_type": "other", "takeaway": "minor two",
                                       "materiality": 0.5})
        ctx = st.context_for(1, recent=2, notable=1)
        ids = {r["id"] for r in ctx}
        assert "old" in ids                        # the notable past event is brought up
        assert {"r1", "r2"} & ids                   # plus recent headlines


# --- ingest: quota cache ----------------------------------------------------------

def test_ingest_quota_cache_skips_refetch(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(ticker, limit=12, client=None):
        calls["n"] += 1
        return [_article("z1", "Acme acquires a rival")]

    monkeypatch.setattr("stockscan.newsmem.ingest.company_news", fake_fetch)
    from stockscan.newsmem.ingest import ingest_company_news

    with NewsStore(tmp_path / "news.sqlite") as st:
        d1 = ingest_company_news(1, "ACME", st, llm=None)     # no LLM -> heuristic extraction
        assert d1["fetched"] == 1 and d1["new"] == 1 and d1["extracted"] == 1
        d2 = ingest_company_news(1, "ACME", st, llm=None)     # within refetch window
        assert d2["skipped_fetch"] and d2["fetched"] == 0
        assert calls["n"] == 1                                 # network hit exactly once
        # forcing re-fetch bypasses the cache but stays idempotent (no new rows)
        d3 = ingest_company_news(1, "ACME", st, llm=None, force=True)
        assert calls["n"] == 2 and d3["new"] == 0


def test_ingest_upgrades_heuristic_placeholder_to_llm(tmp_path, monkeypatch):
    """A lazy no-LLM open leaves a heuristic placeholder; the nightly LLM run upgrades
    it in place (a plain no-LLM re-run does not, so opens never churn)."""
    monkeypatch.setattr("stockscan.newsmem.ingest.company_news",
                        lambda ticker, limit=12, client=None: [_article("u1", "Acme to acquire Rival")])
    from stockscan.newsmem.ingest import ingest_company_news

    def mock_llm(system, user):
        return json.dumps({"takeaway": "a large acquisition", "event_type": "M&A",
                           "materiality": 0.9})

    with NewsStore(tmp_path / "news.sqlite") as st:
        ingest_company_news(1, "ACME", st, llm=None)
        assert st.get_extraction("u1")["model"] == "heuristic"
        # a no-LLM re-run must NOT re-extract the existing placeholder
        assert ingest_company_news(1, "ACME", st, llm=None, force=True)["extracted"] == 0
        # the LLM run upgrades the heuristic placeholder in place
        d = ingest_company_news(1, "ACME", st, llm=mock_llm, force=True)
        assert d["extracted"] == 1
        up = st.get_extraction("u1")
        assert up["model"] != "heuristic" and up["takeaway"] == "a large acquisition"


def test_recall_feeds_grounded_number_free_packet(tmp_path):
    """End-to-end firewall seam: store recall -> packet news context adds NO free
    numeral except a date year, so a fabricated figure is still caught."""
    from stockscan.narrate.ground import check_grounding
    from stockscan.narrate.packet import news_context

    with NewsStore(tmp_path / "news.sqlite") as st:
        st.upsert_articles(1, "ACME", [_article("p1", "Acme raises guidance by 25%",
                                                date="2026-06-01T00:00:00Z")])
        st.put_extraction("p1", "v1", {"event_type": "guidance", "materiality": 0.8,
                                       "takeaway": "Acme raised its full-year guidance"})
        ctx = news_context(st.context_for(1))
    assert ctx and not any(ch.isdigit() for ch in ctx[0]["takeaway"])
    packet = {"meta": {"cik": 1}, "signals": [], "context": {"news": ctx}}
    assert check_grounding("Reported in June 2026.", packet) == []   # date year grounds
    assert check_grounding("A fabricated 25% move.", packet) == [25.0]  # article's number is NOT blessed
