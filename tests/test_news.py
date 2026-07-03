"""Pure-function tests for the news/events shapers (no network)."""

from stockscan.news import shape_article, shape_filings


def test_shape_filings_labels_sorts_and_filters():
    recent = {
        "form": ["8-K", "4", "10-Q", "10-K"],
        "filingDate": ["2026-06-01", "2026-06-15", "2026-05-01", "2025-10-31"],
        "reportDate": ["", "", "2026-03-31", "2025-09-30"],
    }
    rows = shape_filings(recent, limit=8)
    forms = [r["form"] for r in rows]
    assert "4" not in forms                       # Form 4 is filtered as noise
    assert forms == ["8-K", "10-Q", "10-K"]       # newest filingDate first
    assert rows[0]["label"] == "material event"   # 8-K label
    assert rows[1]["period_end"] == "2026-03-31"


def test_shape_article_extracts_source_and_date():
    a = shape_article({"title": "  Apple   beats  ", "url": "https://www.reuters.com/x/y",
                       "publication_date": "2026-07-03T12:18:30.000Z"})
    assert a["title"] == "Apple beats"            # whitespace collapsed
    assert a["date"] == "2026-07-03"              # date trimmed to day
    assert a["source"] == "reuters.com"           # www. stripped from host
