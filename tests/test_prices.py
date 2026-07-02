"""Network-free tests for the price long-format transform."""

import numpy as np
import pandas as pd

from stockscan.prices import _iter_long, _tidy, _tiingo_to_tidy


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
