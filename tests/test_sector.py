"""Tests for the SIC -> sector division mapping."""

from stockscan.sector import sic_division


def test_sic_division_buckets():
    assert sic_division(3571) == "Manufacturing"
    assert sic_division(6021) == "Finance"
    assert sic_division(7372) == "Services"
    assert sic_division(1311) == "Mining"
    assert sic_division(5411) == "Retail"


def test_sic_division_handles_missing():
    assert sic_division(None) == "Unknown"
    assert sic_division(float("nan")) == "Unknown"
