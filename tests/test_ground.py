"""Tests for the grounding guard — it must catch invented numbers, pass real ones."""

from stockscan.narrate.ground import check_grounding, extract_numbers, is_grounded


def test_extract_numbers_and_strips_form_types():
    assert extract_numbers("ROA 31% ranks 97th; rev 391") == [31.0, 97.0, 391.0]
    assert extract_numbers("filed a 10-K and a 10-Q") == []  # form types are not numbers
    assert extract_numbers("total 1,234.5") == [1234.5]


def test_grounding_accepts_packet_numbers():
    packet = {"signals": [{"value": 31.0, "pct_rank": 97}], "meta": {"fiscal_year": 2025}}
    assert check_grounding("ROA 31% is 97th percentile in fiscal 2025", packet) == []


def test_grounding_flags_invented_numbers():
    packet = {"signals": [{"value": 31.0, "pct_rank": 97}]}
    violations = check_grounding("ROA 31% but margin jumped to 45%", packet)
    assert 45.0 in violations
    assert 31.0 not in violations


def test_is_grounded():
    assert is_grounded("value 12.5 and 97", {"a": 12.5, "b": 97})
    assert not is_grounded("a suspicious 99", {"a": 12.5})
