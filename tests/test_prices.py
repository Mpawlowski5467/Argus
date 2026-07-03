"""Network-free tests for the price long-format transform."""

import httpx
import numpy as np
import pandas as pd

import stockscan.prices as prices_mod
from stockscan.prices import (
    _intrinio_to_tidy,
    _iter_long,
    _tidy,
    _tiingo_to_tidy,
    download_intrinio_universe,
    intrinio_get_json,
)


def _fake_yf_frame():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    tuples = []
    for tk in ("AAPL", "MSFT"):
        for field in ("Open", "High", "Low", "Close", "Volume"):
            tuples.append((tk, field))
    cols = pd.MultiIndex.from_tuples(tuples)
    data = np.arange(len(idx) * len(cols), dtype=float).reshape(len(idx), len(cols))
    df = pd.DataFrame(data, index=idx, columns=cols)
    # Punch a NaN into MSFT's last close so we can check the dropna.
    df.loc[idx[-1], ("MSFT", "Close")] = np.nan
    return df


def test_iter_long_splits_tickers_and_columns():
    df = _fake_yf_frame()
    out = dict(_iter_long(df, ["AAPL", "MSFT"]))
    assert set(out) == {"AAPL", "MSFT"}
    aapl = out["AAPL"]
    assert list(aapl.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(aapl) == 3
    # MSFT's NaN-close row is dropped.
    assert len(out["MSFT"]) == 2
    assert str(aapl["date"].dtype).startswith("datetime64")


def test_tidy_single_ticker_flat_columns():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1, 2], "Low": [1, 2], "Close": [1.5, 2.5], "Volume": [10, 20]},
        index=idx,
    )
    tidy = _tidy(df)
    assert list(tidy.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert tidy["close"].tolist() == [1.5, 2.5]


def test_tiingo_to_tidy_uses_adjusted_and_matches_schema():
    rows = [
        {"date": "2024-01-02T00:00:00.000Z", "close": 1.4, "adjOpen": 1.0, "adjHigh": 2.0,
         "adjLow": 0.5, "adjClose": 1.5, "adjVolume": 100},
        {"date": "2024-01-03T00:00:00.000Z", "adjOpen": 1.1, "adjHigh": 2.1,
         "adjLow": 0.6, "adjClose": 1.6, "adjVolume": 110},
    ]
    df = _tiingo_to_tidy(rows, "aapl")
    # same schema as the yfinance adapter -> downstream is provider-agnostic
    assert list(df.columns) == ["ticker", "date", "open", "high", "low", "close", "volume"]
    assert (df["ticker"] == "AAPL").all()
    assert df["close"].tolist() == [1.5, 1.6]  # adjusted close, not raw
    assert str(df["date"].dtype).startswith("datetime64")
    assert _tiingo_to_tidy([], "x").empty


def test_intrinio_to_tidy_uses_adjusted_and_matches_schema():
    rows = [
        {"date": "2024-01-02", "close": 1.4, "volume": 90, "adj_open": 1.0,
         "adj_high": 2.0, "adj_low": 0.5, "adj_close": 1.5, "adj_volume": 100},
        {"date": "2024-01-03", "close": 1.55, "volume": 95, "adj_open": 1.1,
         "adj_high": 2.1, "adj_low": 0.6, "adj_close": 1.6, "adj_volume": 110},
    ]
    df = _intrinio_to_tidy(rows, "aapl")
    # OHLCV stay ADJUSTED (downstream unchanged); uclose/uvolume carry the raw
    # print for the liquidity-floor fix (Phase-5 data-layer schema addition).
    assert list(df.columns) == ["ticker", "date", "open", "high", "low", "close",
                                "volume", "uclose", "uvolume"]
    assert (df["ticker"] == "AAPL").all()
    assert df["close"].tolist() == [1.5, 1.6]        # adjusted close
    assert df["uclose"].tolist() == [1.4, 1.55]      # unadjusted close
    assert _intrinio_to_tidy([], "x").empty


def test_intrinio_get_json_retries_429_then_succeeds(monkeypatch):
    naps = []
    monkeypatch.setattr(prices_mod.time, "sleep", naps.append)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler)) as c:
        assert intrinio_get_json(c, "/anything", {}) == {"ok": True}
    assert calls["n"] == 2 and naps == [3.0]  # honored Retry-After


def test_intrinio_get_json_permanent_error_returns_none():
    def handler(request):
        return httpx.Response(403, json={"error": "not entitled"})

    with httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler)) as c:
        assert intrinio_get_json(c, "/anything", {}) is None


def test_intrinio_get_json_transient_exhaustion_raises(monkeypatch):
    monkeypatch.setattr(prices_mod.time, "sleep", lambda s: None)

    def handler(request):
        return httpx.Response(503)

    with httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler)) as c:
        try:
            intrinio_get_json(c, "/anything", {}, max_tries=3)
            raise AssertionError("expected IntrinioTransientError")
        except prices_mod.IntrinioTransientError:
            pass


def test_intrinio_get_json_survives_http_date_retry_after(monkeypatch):
    monkeypatch.setattr(prices_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:  # nonstandard-but-legal HTTP-date Retry-After must not crash
            return httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        return httpx.Response(200, json={"ok": 1})

    with httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler)) as c:
        assert intrinio_get_json(c, "/anything", {}) == {"ok": 1}


def _price_rows(dates, closes):
    return [{"date": d, "adj_open": c, "adj_high": c, "adj_low": c,
             "adj_close": c, "adj_volume": 1000} for d, c in zip(dates, closes)]


def test_download_intrinio_universe_splices_clips_and_skips(tmp_path):
    # DEAD~2: exchange candidate (Jan 2-6) + OTC candidate (Jan 5-13, lower priority)
    # overlapping Jan 5-6 -> exchange wins; ledger clip drops rows after Jan 9.
    by_sec = {
        "sec_exch": _price_rows(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], [10.0, 9.0, 8.0, 7.0]),
        "sec_otc": _price_rows(
            ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"],
            [6.5, 6.0, 5.0, 4.0, 3.0]),
        "sec_live": _price_rows(["2024-01-02", "2024-01-03"], [100.0, 101.0]),
    }

    def handler(request):
        sec = request.url.path.split("/")[2]
        if sec == "sec_403":
            return httpx.Response(403, json={"error": "no"})
        return httpx.Response(200, json={"stock_prices": by_sec[sec], "next_page": None})

    universe = pd.DataFrame(
        {
            "cik": [2, 2, 1, 9],
            "column": ["DEAD~2", "DEAD~2", "LIVE", "GONE~9"],
            "security_id": ["sec_exch", "sec_otc", "sec_live", "sec_403"],
            "priority": [0, 1, 0, 0],
            "clip_date": [pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-09"), pd.NaT, pd.NaT],
        }
    )
    written = download_intrinio_universe(
        universe, start="2024-01-01", api_key="k", out_dir=tmp_path,
        transport=httpx.MockTransport(handler), log_every=0,
    )
    assert sorted(written) == ["DEAD~2", "LIVE"]     # 403 candidate -> no file
    dead = pd.read_parquet(tmp_path / "DEAD~2.parquet")
    assert dead["date"].max() == pd.Timestamp("2024-01-09")          # clipped
    jan5 = dead.loc[dead["date"] == pd.Timestamp("2024-01-05"), "close"].iloc[0]
    assert jan5 == 7.0                               # exchange candidate wins the overlap
    assert (dead["ticker"] == "DEAD~2").all()
    assert dead["date"].is_monotonic_increasing

    # resumability: second run skips existing files
    again = download_intrinio_universe(
        universe, start="2024-01-01", api_key="k", out_dir=tmp_path,
        transport=httpx.MockTransport(handler), log_every=0,
    )
    assert again == []


def test_download_intrinio_universe_transient_failure_writes_nothing(tmp_path, monkeypatch):
    """A transient error on ONE candidate must abort the whole column (no partial

    splice cached as done) — the missing file makes the resume checkpoint retry it."""
    monkeypatch.setattr(prices_mod.time, "sleep", lambda s: None)

    def handler(request):
        sec = request.url.path.split("/")[2]
        if sec == "sec_flaky":
            return httpx.Response(503)  # transient outage on the exchange-era candidate
        return httpx.Response(200, json={
            "stock_prices": _price_rows(["2024-01-02"], [5.0]), "next_page": None})

    universe = pd.DataFrame(
        {
            "cik": [2, 2],
            "column": ["DEAD~2", "DEAD~2"],
            "security_id": ["sec_flaky", "sec_otc"],
            "priority": [0, 1],
            "clip_date": [pd.NaT, pd.NaT],
        }
    )
    written = download_intrinio_universe(
        universe, start="2024-01-01", api_key="k", out_dir=tmp_path,
        transport=httpx.MockTransport(handler), log_every=0,
    )
    assert written == []
    assert not (tmp_path / "DEAD~2.parquet").exists()  # not pinned as complete
