"""Network-free test for master.idx delisting-form parsing."""

from stockscan.edgar.delistings import _parse_master

_MASTER = """Description:           Master Index
CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
320193|APPLE INC|10-K|2020-10-30|edgar/data/320193/x.txt
100378|TWIN DISC INC|15-12B|2020-01-21|edgar/data/100378/y.txt
1009672|CARBO CERAMICS INC|25-NSE|2020-01-10|edgar/data/1009672/z.txt
1004530|MOUNTAIN PROVINCE|25|2020-02-10|edgar/data/1004530/w.txt
999|SOME CO|8-K|2020-01-01|edgar/data/999/v.txt
"""


def test_parse_master_keeps_only_delist_forms():
    events = _parse_master(_MASTER)
    ciks = {cik for cik, *_ in events}
    forms = {form for _, _, form, _ in events}
    assert ciks == {100378, 1009672, 1004530}  # the delist/dereg filers only
    assert forms == {"15-12B", "25-NSE", "25"}
    assert 320193 not in ciks  # a 10-K is not a delisting
