"""Unit tests for the web layer's pandas/numpy → JSON converters.

Tiny frames, no server — same discipline as the pure row/chart-helper tests.
These guard the highest-risk serialization surfaces: price series, OHLC frames,
and stray numpy/NaN/Timestamp values leaking out of the ticker packet.
"""

import numpy as np
import pandas as pd

from stockscan.web import convert


def test_series_to_points_drops_nan_and_formats_dates():
    s = pd.Series(
        [10.0, float("nan"), 12.5],
        index=pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06"]),
    )
    pts = convert.series_to_points(s)
    assert pts == [
        {"date": "2026-01-02", "close": 10.0},
        {"date": "2026-01-06", "close": 12.5},
    ]
    assert convert.series_to_points(None) == []


def test_ohlc_to_arrays_is_columnar_with_null_nans():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
        "open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
        "close": [1.2, float("nan")], "volume": [100, 200],
    })
    out = convert.ohlc_to_arrays(df)
    assert out["date"] == ["2026-01-02", "2026-01-03"]
    assert out["close"] == [1.2, None]           # NaN → null
    assert out["volume"] == [100.0, 200.0]
    assert convert.ohlc_to_arrays(pd.DataFrame()) is None


def test_jsonable_coerces_numpy_nan_timestamp_and_nesting():
    obj = {
        "i": np.int64(7),
        "f": np.float64(1.5),
        "nan": float("nan"),
        "b": np.bool_(True),
        "ts": pd.Timestamp("2026-07-02"),
        "arr": np.array([1, 2, 3]),
        "nested": [{"x": np.float32(0.25)}, None, "ok"],
    }
    out = convert.jsonable(obj)
    assert out["i"] == 7 and isinstance(out["i"], int)
    assert out["f"] == 1.5 and isinstance(out["f"], float)
    assert out["nan"] is None                    # NaN → None (valid JSON)
    assert out["b"] is True
    assert out["ts"] == "2026-07-02"
    assert out["arr"] == [1, 2, 3]
    assert out["nested"] == [{"x": 0.25}, None, "ok"]


def test_jsonable_passes_through_json_natives():
    assert convert.jsonable({"a": 1, "b": "s", "c": True, "d": None, "e": 3.14}) == {
        "a": 1, "b": "s", "c": True, "d": None, "e": 3.14,
    }
