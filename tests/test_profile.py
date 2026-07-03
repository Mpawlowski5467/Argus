"""Network-free tests for the company-profile normalize + cache logic (live-view)."""

from datetime import timedelta

import pytest

from stockscan import profile as P


_RAW = {  # Intrinio /companies shape (trimmed to the fields we consume)
    "cik": "0000320193",
    "name": "Apple Inc.",
    "legal_name": "Apple Inc.",
    "short_description": "Apple Inc. designs, manufactures, and markets smartphones.",
    "long_description": "A much longer paragraph about Apple that we prefer NOT to show.",
    "sector": "Manufacturing",
    "industry_category": "Computer Hardware",
    "industry_group": "Computer & Office Equipment",
    "hq_address_city": "Cupertino",
    "hq_state": "California",
    "hq_country": "United States of America",
    "employees": "164000",
    "ceo": "Timothy D. Cook",
    "company_url": "apple.com",
}


def test_normalize_maps_and_prefers_short_description():
    p = P.normalize_profile(_RAW)
    assert p["name"] == "Apple Inc."
    assert p["description"].startswith("Apple Inc. designs")   # short, not long
    assert p["industry"] == "Computer Hardware"               # category over group
    assert (p["city"], p["state"], p["country"]) == (
        "Cupertino", "California", "United States of America")
    assert p["employees"] == 164000 and isinstance(p["employees"], int)
    assert p["url"] == "apple.com"


def test_normalize_falls_back_and_blanks_to_none():
    raw = {"long_description": "only the long one", "employees": "",
           "hq_address_city": "  ", "industry_group": "Retail"}
    p = P.normalize_profile(raw)
    assert p["description"] == "only the long one"   # falls back to long
    assert p["industry"] == "Retail"                 # falls back to group
    assert p["employees"] is None                    # unparseable -> None
    assert p["city"] is None                         # whitespace -> None
    assert p["name"] is None                         # absent -> None


def test_normalize_titlecases_all_caps_city():
    assert P.normalize_profile({"hq_address_city": "AUSTIN"})["city"] == "Austin"
    assert P.normalize_profile({"hq_address_city": "New York"})["city"] == "New York"


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PROFILE_DB_PATH", tmp_path / "profiles.sqlite")
    prof = P.normalize_profile(_RAW)
    assert P._cache_get(320193) is None              # empty store
    P._cache_put(320193, prof)
    got, fetched_at = P._cache_get(320193)
    assert got == prof and fetched_at is not None


def test_get_profile_serves_fresh_cache_without_fetching(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PROFILE_DB_PATH", tmp_path / "profiles.sqlite")
    P._cache_put(320193, P.normalize_profile(_RAW))

    def _boom(*a, **k):
        raise AssertionError("fetch_profile must not be called on a fresh cache hit")

    monkeypatch.setattr(P, "fetch_profile", _boom)
    assert P.get_profile(320193)["name"] == "Apple Inc."


def test_get_profile_serves_stale_copy_when_fetch_fails(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(P, "PROFILE_DB_PATH", tmp_path / "profiles.sqlite")
    # write a row stamped well past the refetch window (reads as stale now)
    old = (P._utcnow() - timedelta(days=P.PROFILE_REFETCH_DAYS + 5)).isoformat(timespec="seconds")
    prof = P.normalize_profile(_RAW)
    db = P._connect()
    db.execute("insert or replace into profiles (cik, fetched_at, name, data) values (?,?,?,?)",
               (320193, old, prof["name"], json.dumps(prof)))
    db.commit()
    db.close()
    monkeypatch.setattr(P, "fetch_profile", lambda *a, **k: None)  # network down
    assert P.get_profile(320193)["name"] == "Apple Inc."           # stale beats nothing
