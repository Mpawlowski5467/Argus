"""Tests for the point-in-time guard — the invariant everything else leans on."""

import pandas as pd
import pytest

from stockscan.pit import LookAheadError, assert_pit, available_date


def test_available_date_adds_one_business_day():
    # Fri 2024-01-05 + 1 business day -> Mon 2024-01-08
    assert available_date("2024-01-05") == pd.Timestamp("2024-01-08")


def test_available_date_series_shape_and_values():
    s = pd.Series(pd.to_datetime(["2024-01-05", "2024-02-01"]))
    out = available_date(s)
    assert list(out) == [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-02-02")]


def test_assert_pit_passes_when_all_public():
    df = pd.DataFrame({"filed_date": pd.to_datetime(["2024-01-02", "2024-02-01"])})
    assert_pit(df, "2024-03-01")  # both filings + lag are well before as_of


def test_assert_pit_blocks_future_filing():
    df = pd.DataFrame({"filed_date": pd.to_datetime(["2024-01-02", "2024-06-30"])})
    with pytest.raises(LookAheadError):
        assert_pit(df, "2024-03-01")


def test_assert_pit_same_day_not_yet_available():
    # A filing accepted ON the as_of date is not usable until the next business day.
    df = pd.DataFrame({"filed_date": pd.to_datetime(["2024-03-01"])})
    with pytest.raises(LookAheadError):
        assert_pit(df, "2024-03-01")


def test_assert_pit_requires_filed_column():
    with pytest.raises(KeyError):
        assert_pit(pd.DataFrame({"x": [1]}), "2024-01-01")
