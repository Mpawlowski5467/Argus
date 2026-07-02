"""Network-free tests for the survivorship-free universe join/selection logic."""

import pandas as pd

from stockscan.intrinio_universe import CLIP_GRACE_DAYS, select_universe, universe_ticker_map


def _companies():
    return pd.DataFrame(
        {
            "company_id": ["com_live", "com_dead", "com_reuse", "com_nocik", "com_multi"],
            "ticker": ["LIVE", None, None, "X", None],
            "name": ["Live Corp", "Dead Corp", "Old Live Corp", "No Cik", "Multi Dead"],
            "cik": ["0000000001", "0000000002", "0000000003", None, "0000000004"],
        }
    )


def _securities():
    rows = [
        # (security_id, company_id, ticker, composite, name, primary, figi, active)
        ("sec_live", "com_live", "LIVE", "LIVE:US", "Live Corp", True, "BBG1", True),
        # a living company's dead sibling security (old listing) must be dropped
        ("sec_live_old", "com_live", "LIVEOLD", "LIVEOLD:US", "Live Corp", True, None, False),
        # dead company, single inactive security
        ("sec_dead", "com_dead", "DEAD", "DEAD:US", "Dead Corp", True, None, False),
        # ticker reuse: a DEAD company that used the ticker LIVE years ago
        ("sec_old", "com_reuse", "LIVE", "LIVE:US", "Old Live Corp", True, None, False),
        # dead company with exchange listing + OTC afterlife (splice candidates)
        ("sec_exch", "com_multi", "MULT", "MULT:US", "Multi Dead", True, "BBG2", False),
        ("sec_otc", "com_multi", "MULTQ", "MULTQ:US", "Multi Dead", False, None, False),
    ]
    return pd.DataFrame(
        rows,
        columns=["security_id", "company_id", "ticker", "composite_ticker", "name",
                 "primary_listing", "figi", "active"],
    )


def _delistings():
    return pd.DataFrame(
        {"cik": [2], "delist_date": [pd.Timestamp("2015-06-01")], "reason": ["delist"]}
    )


def test_select_universe_names_columns_and_handles_reuse():
    uni = select_universe(_securities(), _companies(), {1, 2, 3, 4}, _delistings())

    live = uni[uni["cik"] == 1]
    assert list(live["column"]) == ["LIVE"]          # active -> plain ticker
    assert list(live["security_id"]) == ["sec_live"]  # dead sibling dropped, not spliced
    assert live["clip_date"].isna().all()            # live names never clipped

    dead = uni[uni["cik"] == 2].iloc[0]
    assert dead["column"] == "DEAD~2"                # dead -> TICKER~CIK
    assert pd.isna(dead["clip_date"])                # ledger clip OFF by default (by-id bounds)

    # the recycled ticker's OLD owner must not collide with the live name
    reuse = uni[uni["cik"] == 3].iloc[0]
    assert reuse["column"] == "LIVE~3"
    assert pd.isna(reuse["clip_date"])


def test_select_universe_optional_ledger_clip():
    uni = select_universe(
        _securities(), _companies(), {1, 2}, _delistings(), clip_grace_days=CLIP_GRACE_DAYS
    )
    dead = uni[uni["cik"] == 2].iloc[0]
    assert dead["clip_date"] == pd.Timestamp("2015-06-01") + pd.Timedelta(days=CLIP_GRACE_DAYS)
    assert uni[uni["cik"] == 1]["clip_date"].isna().all()


def test_select_universe_splice_candidates_share_column_ranked():
    uni = select_universe(_securities(), _companies(), {4}, None)
    multi = uni[uni["cik"] == 4].sort_values("priority")
    assert len(multi) == 2
    assert set(multi["column"]) == {"MULT~4"}        # one column, two candidates
    # exchange listing (primary + figi) outranks the OTC afterlife
    assert list(multi["security_id"]) == ["sec_exch", "sec_otc"]
    assert list(multi["priority"]) == [0, 1]


def test_select_universe_restricts_to_our_ciks():
    uni = select_universe(_securities(), _companies(), {1}, None)
    assert set(uni["cik"]) == {1}


def test_universe_ticker_map_roundtrip(tmp_path):
    uni = select_universe(_securities(), _companies(), {1, 2, 4}, _delistings())
    path = tmp_path / "uni.parquet"
    uni.to_parquet(path, index=False)
    m = universe_ticker_map(path)
    assert m == {1: "LIVE", 2: "DEAD~2", 4: "MULT~4"}
    assert universe_ticker_map(tmp_path / "missing.parquet") == {}
