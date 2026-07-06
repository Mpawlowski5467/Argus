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
    assert check_grounding("in March 2026 the filing", packet) == []
    # ...but the date must NOT whitelist fabricated negatives like -3% or -31%
    assert -3.0 in check_grounding("revenue fell -3% this year", packet)
    assert -31.0 in check_grounding("a -31% collapse", packet)


def test_date_components_do_not_bless_fabricated_figures():
    """Month/day integers from packet dates must not whitelist invented numbers
    (review finding: '12%' and '31%' passed for any Dec-31 filer)."""
    packet = {"meta": {"period_end": "2025-09-30", "as_of": "2026-06-30"},
              "model": {"trained_through": "2025-12-31"}}
    assert 30.0 in check_grounding("margins expanded 30% this year", packet)
    assert 31.0 in check_grounding("a 31% market share", packet)
    assert 12.0 in check_grounding("revenue up 9% over the past 12 months", packet) \
        or 9.0 in check_grounding("revenue up 9% over the past 12 months", packet)
    assert 6.0 in check_grounding("a 6% dividend yield", packet)
    # years still trace
    assert check_grounding("the fiscal 2025 period", packet) == []


def test_plural_form_types_are_stripped():
    assert extract_numbers("across its last two 10-Ks") == []


# --- ordinal / slash date forms (the reformats that leaked a bare day-number) -----

def test_ordinal_dates_strip_the_day_and_keep_only_the_year():
    # "July 1st, 2026" broke the old regex on the "st" and leaked a bare "1"
    assert extract_numbers("filed July 1st, 2026") == [2026.0]
    assert extract_numbers("as of March 31st, 2025") == [2025.0]
    assert extract_numbers("dated 1st of July 2026") == [2026.0]
    assert extract_numbers("on the 31st of March 2025") == [2025.0]
    # ordinal with NO year: the day is stripped, nothing spurious survives
    assert extract_numbers("around July 1st") == []
    assert extract_numbers("by the 31st of March") == []


def test_slash_dates_strip_month_and_day_keep_the_year():
    assert extract_numbers("on 07/01/2026") == [2026.0]        # US MM/DD/YYYY
    assert extract_numbers("on 2026/07/01") == [2026.0]        # ISO-ish YYYY/MM/DD
    assert extract_numbers("between 6/30/2025 and 12/31/2025") == [2025.0, 2025.0]
    # a bare ratio / fraction is NOT a date (no 4-digit year anchor) — it must ground
    assert extract_numbers("a 1/2 split") == [1.0, 2.0]


def test_reformatted_dates_ground_when_the_iso_date_is_in_context():
    """The bug: a model that rewrote the packet's ISO date leaked a day-number and
    was wrongly refused. All of these must now trace back to the one ISO date."""
    packet = {"meta": {"as_of": "2026-07-01"}}
    for phrasing in ("as of 2026-07-01", "as of July 1st, 2026", "as of 07/01/2026",
                     "as of 2026/07/01", "in July 2026"):
        assert check_grounding(phrasing, packet) == [], phrasing


def test_bare_month_day_and_ordinals_are_not_blessed():
    """The deliberate boundary: stripping needs a month PLUS a year or an explicit
    ordinal. A bare "June 30", a lone percentile "97th", or a month next to a
    fabricated figure must still be caught — the guard never blesses a bare number."""
    packet = {"meta": {"as_of": "2026-07-01", "period_end": "2026-06-30"}}
    assert 30.0 in check_grounding("the quarter ended June 30", packet)  # no ordinal/yr
    assert 30.0 in check_grounding("June 30% growth", packet)            # month + figure
    assert 97.0 in check_grounding("ranked 97th of its peers", {"a": 1})  # percentile
    # extract_numbers keeps the ambiguous bare forms rather than swallow a figure
    assert extract_numbers("June 30 revenue rose") == [30.0]


def test_out_of_range_date_slots_are_not_swallowed_as_dates():
    """A day/month slot only strips as a date when it is a REAL calendar value
    (day 1-31, month 1-12). A fabricated out-of-range figure dressed as a date
    component ("March 99", "the 45th", "95/5/2025") is NOT a date — it must survive
    to be grounded, or the guard would silently bless the hallucination it exists
    to catch."""
    # natural-language: fabricated day > 31 next to a month/year survives (only the
    # year is a real date part)
    assert 45.0 in extract_numbers("revenue rose March 45, 2026")
    assert 99.0 in extract_numbers("up March 99, 2026")
    assert 88.0 in extract_numbers("March 88th")          # ordinal, out of range
    assert 45.0 in extract_numbers("gains of 45th of July 2026")
    # slash: US month-first, so an out-of-range month (25, 95, 99) or day is not a date
    assert extract_numbers("the stock jumped 25/12/2025") == [25.0, 12.0, 2025.0]
    assert extract_numbers("a 95/5/2025 revenue split") == [95.0, 5.0, 2025.0]
    assert extract_numbers("a 30/70/2025 debt-equity ratio") == [30.0, 70.0, 2025.0]
    # end-to-end: none of these trace to a packet holding only the year → all flagged
    pkt = {"meta": {"period_end": "2025-12-31"}, "model": {"percentile": 72}}
    for s in ("revenue rose March 45, 2026", "up March 99, 2026",
              "the stock jumped 25/12/2025", "a 95/5/2025 split", "gains of 45th of July"):
        assert check_grounding(s, pkt), s     # non-empty violations = correctly caught


def test_in_range_valid_dates_still_strip_after_the_bound():
    """The bound must not cause false refusals: every genuine date (day ≤ 31,
    month ≤ 12, any order the guard supports) still collapses to its year."""
    assert extract_numbers("on 07/01/2026") == [2026.0]
    assert extract_numbers("between 6/30/2025 and 12/31/2025") == [2025.0, 2025.0]
    assert extract_numbers("filed March 31st, 2025") == [2025.0]
    assert extract_numbers("dated 1st of July 2026") == [2026.0]
