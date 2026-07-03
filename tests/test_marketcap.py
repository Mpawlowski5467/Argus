"""Network-free tests for the live market-cap cache (markets view, live-view only)."""

from datetime import timedelta

from stockscan import marketcap as M


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MARKETCAP_DB_PATH", tmp_path / "marketcap.sqlite")
    assert M._cache_get(320193) is None
    M._cache_put(320193, 3.1e12)
    cap, fetched_at = M._cache_get(320193)
    assert cap == 3.1e12 and fetched_at is not None


def test_get_serves_fresh_cache_without_fetching(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MARKETCAP_DB_PATH", tmp_path / "marketcap.sqlite")
    M._cache_put(320193, 2.9e12)

    def _boom(*a, **k):
        raise AssertionError("fetch_market_cap must not run on a fresh cache hit")

    monkeypatch.setattr(M, "fetch_market_cap", _boom)
    assert M.get_market_cap(320193) == 2.9e12


def test_get_serves_stale_value_when_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MARKETCAP_DB_PATH", tmp_path / "marketcap.sqlite")
    old = (M._utcnow() - timedelta(hours=M.MARKETCAP_REFETCH_HOURS + 2)).isoformat(timespec="seconds")
    db = M._connect()
    db.execute("insert or replace into market_caps (cik, fetched_at, mktcap) values (?,?,?)",
               (320193, old, 1.5e12))
    db.commit()
    db.close()
    monkeypatch.setattr(M, "fetch_market_cap", lambda *a, **k: None)  # network down
    assert M.get_market_cap(320193) == 1.5e12                          # stale beats nothing


def test_get_caches_the_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MARKETCAP_DB_PATH", tmp_path / "marketcap.sqlite")
    monkeypatch.setattr(M, "fetch_market_cap", lambda *a, **k: None)   # no data anywhere
    assert M.get_market_cap(999) is None
    row = M._cache_get(999)                                            # miss was remembered
    assert row is not None and row[0] is None
