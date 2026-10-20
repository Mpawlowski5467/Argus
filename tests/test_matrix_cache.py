"""Matrix cache: cached == slow-path exactly, and freshness catches renames.

The cache exists so a daily monitor loop doesn't pay ~2 minutes pivoting 11k
per-column parquets. Correctness rests on two invariants: the cached frames are
byte-for-value identical to the slow path, and staleness is detected on ANY
change to the column set — including os.replace renames, which preserve mtime.
"""

import json
import os

import pandas as pd
import pytest

from stockscan.panel import (
    load_matrices,
    load_matrices_cached,
    matrix_cache_fresh,
    save_matrix_cache,
)


def _write_prices(prices_dir, ticker, dates, closes, with_uclose=True):
    df = pd.DataFrame({
        "ticker": ticker, "date": pd.to_datetime(list(dates)),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1000.0] * len(closes),
    })
    if with_uclose:
        df["uclose"] = closes
        df["uvolume"] = [1000.0] * len(closes)
    df.to_parquet(prices_dir / f"{ticker}.parquet", index=False)


def _setup(tmp_path):
    prices = tmp_path / "prices"
    prices.mkdir()
    dates = pd.bdate_range("2026-01-01", periods=10)
    _write_prices(prices, "AAA", dates, [float(i) for i in range(10, 20)])
    _write_prices(prices, "BBB", dates, [float(i) for i in range(50, 60)])
    # a delisted column lacking the newer uclose schema — union_by_name must cope
    _write_prices(prices, "DEAD~9", dates[:5], [3.0, 2.5, 2.0, 1.5, 1.0],
                  with_uclose=False)
    return prices, tmp_path / "cache"


def test_cached_equals_slow_path(tmp_path):
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    assert matrix_cache_fresh(cache, prices)
    c2, d2 = load_matrices_cached(prices_dir=prices, cache_dir=cache)
    pd.testing.assert_frame_equal(close, c2)
    pd.testing.assert_frame_equal(dv, d2)


def test_cache_stale_on_new_file(tmp_path):
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    _write_prices(prices, "CCC", pd.bdate_range("2026-01-01", periods=10),
                  [float(i) for i in range(1, 11)])
    assert not matrix_cache_fresh(cache, prices)
    # loader transparently falls back to the slow path (includes CCC)
    c2, _ = load_matrices_cached(prices_dir=prices, cache_dir=cache)
    assert "CCC" in c2.columns


def test_cache_stale_on_rename_preserving_mtime(tmp_path):
    """A universe-refresh death renames AAA -> AAA~9 via os.replace, which keeps
    mtime. An mtime-only check would call the cache fresh; the manifest hash
    (filenames included) must catch it."""
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    assert matrix_cache_fresh(cache, prices)
    os.replace(prices / "AAA.parquet", prices / "AAA~9.parquet")
    assert not matrix_cache_fresh(cache, prices), "rename must invalidate the cache"


def test_cache_missing_meta_falls_back(tmp_path):
    prices, cache = _setup(tmp_path)
    close, _ = load_matrices(prices_dir=prices)
    # no cache written yet
    assert not matrix_cache_fresh(cache, prices)
    c2, _ = load_matrices_cached(prices_dir=prices, cache_dir=cache)
    pd.testing.assert_frame_equal(close, c2)


# --- the 2026-09-19 torn cache ------------------------------------------------------
# The live cache held EMPTY close/dv frames under a meta.json whose manifest still
# matched the store, so it read as fresh and the app served "no prices on disk" for a
# month. Three guards: never write an empty matrix, never leave an old meta standing
# over swapped data, never serve an empty cached frame.

def test_save_refuses_an_empty_matrix_and_keeps_the_good_cache(tmp_path):
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    with pytest.raises(ValueError):
        save_matrix_cache(pd.DataFrame(), pd.DataFrame(), cache_dir=cache, prices_dir=prices)
    assert matrix_cache_fresh(cache, prices)
    c2, _ = load_matrices_cached(prices_dir=prices, cache_dir=cache)
    pd.testing.assert_frame_equal(close, c2)


def test_torn_legacy_cache_is_never_served(tmp_path):
    """The exact live state: empty data files under a pre-fix meta (no recorded file
    sizes) whose manifest matches the store. The loader must fall back to the store."""
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    meta_p = cache / "meta.json"
    meta = json.loads(meta_p.read_text())
    meta.pop("files")
    meta_p.write_text(json.dumps(meta))
    pd.DataFrame().to_parquet(cache / "close.parquet")
    pd.DataFrame().to_parquet(cache / "dv.parquet")

    c2, d2 = load_matrices_cached(prices_dir=prices, cache_dir=cache)
    pd.testing.assert_frame_equal(close, c2)
    pd.testing.assert_frame_equal(dv, d2)


def test_data_swapped_behind_the_meta_reads_stale(tmp_path):
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    close.iloc[:3].to_parquet(cache / "close.parquet")   # not the file meta was written for
    assert not matrix_cache_fresh(cache, prices)


def test_crash_before_the_meta_lands_leaves_the_cache_stale(tmp_path, monkeypatch):
    """Rebuilding over a good cache from a bad read (the store itself unchanged, so the
    OLD meta's manifest still matches) and dying on the meta rename — EPERM, as on
    2026-09-19 — must leave no meta vouching for the swapped-in data files."""
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    close2, dv2 = close.iloc[:, :1], dv.iloc[:, :1]   # a partial read of the same store

    real_replace = os.replace

    def failing_replace(src, dst):
        if os.path.basename(dst) == "meta.json":
            raise PermissionError(1, "Operation not permitted", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(PermissionError):
        save_matrix_cache(close2, dv2, cache_dir=cache, prices_dir=prices)
    monkeypatch.setattr(os, "replace", real_replace)

    assert not matrix_cache_fresh(cache, prices)
    c3, _ = load_matrices_cached(prices_dir=prices, cache_dir=cache)
    pd.testing.assert_frame_equal(close, c3)   # the store, not the half-written cache


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unreadable_store_raises_instead_of_reading_empty(tmp_path):
    """glob reports an unreadable directory as zero files; that is how a denied
    listing became an empty matrix. The loaders must raise, and freshness must be
    unverifiable (False), never 'fresh'."""
    prices, cache = _setup(tmp_path)
    close, dv = load_matrices(prices_dir=prices)
    save_matrix_cache(close, dv, cache_dir=cache, prices_dir=prices)
    os.chmod(prices, 0)
    try:
        with pytest.raises(PermissionError):
            load_matrices(prices_dir=prices)
        assert not matrix_cache_fresh(cache, prices)
    finally:
        os.chmod(prices, 0o755)
