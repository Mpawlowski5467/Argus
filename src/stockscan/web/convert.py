"""pandas / numpy → JSON-native helpers for the web API.

Pure and unit-testable (same discipline as the row-shapers in tui/data.py). The
facade's row-shapers already cast to int/float/str, so scan/watch/markets are
clean; the risk surfaces are ``price().series`` (a Series), ``ohlc()`` (a
DataFrame), and stray ``pd.Timestamp`` / numpy scalars in the ticker packet —
``jsonable`` is the catch-all net over those.
"""

from __future__ import annotations

import datetime as dt
import math


def _num(x):
    """float, or None for None / NaN / inf / non-numeric."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _date_str(d) -> str:
    """A date-index label → 'YYYY-MM-DD' (handles Timestamp / date / str)."""
    if hasattr(d, "date") and not isinstance(d, dt.date):
        try:
            return d.date().isoformat()
        except Exception:
            pass
    if isinstance(d, (dt.date, dt.datetime)):
        return d.isoformat()[:10]
    return str(d)[:10]


def series_to_points(s) -> list[dict]:
    """pd.Series(date → close) → [{"date","close"}], dropping NaN closes."""
    out: list[dict] = []
    if s is None:
        return out
    for idx, val in s.items():
        v = _num(val)
        if v is not None:
            out.append({"date": _date_str(idx), "close": v})
    return out


def ohlc_to_arrays(df) -> dict | None:
    """pd.DataFrame[date,open,high,low,close,volume] → columnar arrays (compact
    for canvas). NaN → null. Returns None for an empty/missing frame."""
    if df is None or len(df) == 0:
        return None
    cols = ("date", "open", "high", "low", "close", "volume")
    out: dict = {}
    for name in cols:
        if name not in df.columns:
            continue
        if name == "date":
            out[name] = [_date_str(d) for d in df[name]]
        else:
            out[name] = [_num(x) for x in df[name]]
    return out


def jsonable(obj):
    """Recursively coerce pandas/numpy scalars, timestamps, NaN, arrays and
    Series into JSON-native types so FastAPI's encoder never chokes."""
    import numpy as np
    import pandas as pd

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _num(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):     # a Timestamp is-a datetime — check it first
        return _date_str(obj)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if isinstance(obj, dt.date):
        return _date_str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [jsonable(v) for v in obj.tolist()]
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj
