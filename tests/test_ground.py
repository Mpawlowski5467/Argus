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


# --- adversarial cases from the Phase-2 review workflow ---------------------------

def test_no_relative_tolerance_around_large_packet_numbers():
    """cik/fiscal_year must not bless nearby fabrications (old 0.5% window did)."""
    packet = {"meta": {"cik": 886158, "fiscal_year": 2024}}
    assert 884000.0 in check_grounding("revenue of 884,000 thousand", packet)
    assert 2018.0 in check_grounding("margins have expanded since 2018", packet)
    assert check_grounding("the fiscal 2024 10-K", packet) == []  # exact year still fine


def test_integer_packet_values_require_exact_match():
    packet = {"model": {"percentile": 96, "n_names": 1500}}
    assert 96.5 in check_grounding("scores at the 96.5th percentile", packet)
    assert 1505.0 in check_grounding("one of 1,505 names scored", packet)
    assert check_grounding("96th percentile of 1,500 names", packet) == []


def test_dates_decompose_positively_not_as_signed_fragments():
    packet = {"meta": {"as_of": "2026-03-31"}}
    # a reformatted date must trace back to the packet's ISO date...
    assert check_grounding("as of March 31, 2026", packet) == []
    # ...but the date must NOT whitelist fabricated negatives like -3% or -31%
    assert -3.0 in check_grounding("revenue fell -3% this year", packet)
    assert -31.0 in check_grounding("a -31% collapse", packet)


def test_plural_form_types_are_stripped():
    assert extract_numbers("across its last two 10-Ks") == []
