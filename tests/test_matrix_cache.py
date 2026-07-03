"""Matrix cache: cached == slow-path exactly, and freshness catches renames.

The cache exists so a daily monitor loop doesn't pay ~2 minutes pivoting 11k
per-column parquets. Correctness rests on two invariants: the cached frames are
byte-for-value identical to the slow path, and staleness is detected on ANY
change to the column set — including os.replace renames, which preserve mtime.
"""

import os

import pandas as pd

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
