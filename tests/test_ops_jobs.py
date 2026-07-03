"""Ingestion jobs: idempotency, crash-safety, and the adjustment-rebase story.

The nightly job full-refetches active columns precisely because adjusted series
rebase retroactively — these tests pin that behavior (no merge seams), the
sanity guards (a shrunken vendor response never destroys good history), and the
FSDS/universe logic that must be safe to replay.
"""


import httpx
import pandas as pd
import pytest

from stockscan.ops.jobs import (
    _replace_verdict,
    _revised,
    apply_renames,
    ingest_new_fsds,
    latest_elapsed_quarter,
    missing_quarters,
    quarters_present,
    recent_dead_columns,
    refresh_active_prices,
    universe_diff,
)


# --- pure verdict / revision logic -------------------------------------------------

def _frame(dates, closes, ticker="X", with_uclose=True):
    df = pd.DataFrame({
        "ticker": ticker, "date": pd.to_datetime(list(dates)),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1000.0] * len(closes),
    })
    if with_uclose:
        df["uclose"] = closes
        df["uvolume"] = [1000.0] * len(closes)
    return df


def test_replace_verdict_guards_history():
    old = _frame(pd.bdate_range("2026-01-01", periods=10), list(range(10, 20)))
    shorter = _frame(pd.bdate_range("2026-01-01", periods=5), list(range(10, 15)))
    assert _replace_verdict(old, shorter) == "suspect"          # lost the tail
    good = _frame(pd.bdate_range("2026-01-01", periods=11), list(range(10, 21)))
    assert _replace_verdict(old, good) == "replace"
    assert _replace_verdict(None, good) == "replace"
    assert _replace_verdict(None, None) == "empty"
    assert _replace_verdict(old, None) == "suspect"             # had data, got nothing


def test_revised_detects_rebase():
    dates = pd.bdate_range("2026-01-01", periods=10)
    old = _frame(dates, [float(i) for i in range(10, 20)])
    same = _frame(dates, [float(i) for i in range(10, 20)])
    assert not _revised(old, same)
    rebased = _frame(dates, [float(i) * 10 for i in range(10, 20)])  # 1:10 reverse split
    assert _revised(old, rebased)


# --- nightly refetch through a mock Intrinio ---------------------------------------

def _intrinio_transport(book: dict[str, list[dict]], calls: list[str]):
    """MockTransport serving /securities/{id}/prices from a dict of rows."""
    def handler(request: httpx.Request) -> httpx.Response:
        sec_id = request.url.path.split("/")[2]
        calls.append(sec_id)
        return httpx.Response(200, json={"stock_prices": book.get(sec_id, [])})
    return httpx.MockTransport(handler)


def _rows(dates, closes):
    return [
        {"date": str(pd.Timestamp(d).date()),
         "adj_open": c, "adj_high": c, "adj_low": c, "adj_close": c,
         "adj_volume": 1000.0, "close": c * 2, "volume": 500.0}
        for d, c in zip(dates, closes)
    ]


@pytest.fixture
def universe():
    return pd.DataFrame({
        "cik": [1, 2, 3],
        "column": ["AREF", "BCO", "DEAD~3"],
        "security_id": ["sa", "sb", "sd"],
        "ticker": ["AREF", "BCO", "DEAD"],
        "name": ["A", "B", "D"],
        "active": [True, True, False],
        "priority": [0, 0, 0],
        "clip_date": [pd.NaT] * 3,
    })


def test_refresh_active_prices_writes_and_skips_dead(tmp_path, universe):
    dates = pd.bdate_range("2026-06-01", periods=5)
    calls: list[str] = []
    transport = _intrinio_transport(
        {"sa": _rows(dates, [10, 11, 12, 13, 14]),
         "sb": _rows(dates, [20, 21, 22, 23, 24]),
         "sd": _rows(dates, [1, 1, 1, 1, 1])}, calls)
    deltas = refresh_active_prices(
        universe, api_key="k", out_dir=tmp_path, reference_column="AREF",
        transport=transport, pause=0, workers=2, rebuild_cache=False)
    assert deltas["written"] == 2
    assert "sd" not in calls, "dead columns must never be fetched nightly"
    a = pd.read_parquet(tmp_path / "AREF.parquet")
    assert "uclose" in a.columns and a["uclose"].iloc[0] == 20  # raw = 2x adjusted here
    assert str(deltas["target_date"]) == str(dates[-1].date())
    assert not (tmp_path / "DEAD~3.parquet").exists()


def test_refresh_second_run_is_noop(tmp_path, universe):
    dates = pd.bdate_range("2026-06-01", periods=5)
    calls: list[str] = []
    transport = _intrinio_transport(
        {"sa": _rows(dates, [10, 11, 12, 13, 14]),
         "sb": _rows(dates, [20, 21, 22, 23, 24])}, calls)
    for _ in range(2):
        deltas = refresh_active_prices(
            universe, api_key="k", out_dir=tmp_path, reference_column="AREF",
            transport=transport, pause=0, rebuild_cache=False)
    # run 2: the reference column refetches (identical -> fresh, no rewrite);
    # the other column skips on the target-date check without any request
    assert deltas["written"] == 0
    assert deltas["fresh"] == 2
    assert calls.count("sb") == 1, "fresh column must not be re-requested"


def test_refresh_heals_rebase_without_seam(tmp_path, universe):
    """A 1:10 reverse split rebases the whole vendor history; the full refetch
    replaces the file wholesale — no scale seam — and counts the revision."""
    d1 = pd.bdate_range("2026-06-01", periods=5)
    _frame(d1, [10.0, 11, 12, 13, 14], ticker="BCO").to_parquet(
        tmp_path / "BCO.parquet", index=False)
    d2 = pd.bdate_range("2026-06-01", periods=6)  # one new day, all history x10
    calls: list[str] = []
    # the reference column also advances to d2 so the session target moves past
    # BCO's stale max date, forcing a refetch (otherwise it's correctly 'fresh')
    transport = _intrinio_transport(
        {"sa": _rows(d2, [1, 1, 1, 1, 1, 1]),
         "sb": _rows(d2, [100, 110, 120, 130, 140, 150])}, calls)
    deltas = refresh_active_prices(
        universe, api_key="k", out_dir=tmp_path, reference_column="AREF",
        transport=transport, pause=0, rebuild_cache=False)
    b = pd.read_parquet(tmp_path / "BCO.parquet")
    ratios = b["close"].pct_change().abs().dropna()
    assert (ratios < 0.2).all(), f"seam in healed series: {b['close'].tolist()}"
    assert deltas["revised_columns"] >= 1


def test_refresh_reference_failure_does_not_strand_universe(tmp_path, universe):
    """If the heartbeat column fails transiently, the run must NOT adopt its stale
    date as the session target (which would short-circuit every other column to
    'fresh') — it fetches the rest normally and flags itself degraded."""
    dates = pd.bdate_range("2026-06-01", periods=5)
    # BCO already has yesterday's data; AREF (reference) has a stale file too
    _frame(dates[:4], [20.0, 21, 22, 23], ticker="AREF").to_parquet(
        tmp_path / "AREF.parquet", index=False)
    _frame(dates[:4], [30.0, 31, 32, 33], ticker="BCO").to_parquet(
        tmp_path / "BCO.parquet", index=False)
    # AREF fetch returns nothing (transient-ish 'unavailable' -> suspect since old exists),
    # BCO has a genuinely newer bar available
    transport = _intrinio_transport(
        {"sa": [], "sb": _rows(dates, [30, 31, 32, 33, 34])}, [])
    deltas = refresh_active_prices(
        universe, api_key="k", out_dir=tmp_path, reference_column="AREF",
        transport=transport, pause=0, rebuild_cache=False)
    assert deltas["reference_ok"] is False
    assert deltas["_status"] == "degraded"
    assert deltas["target_date"] is None
    # BCO was still fetched to its newest bar despite the reference failing
    b = pd.read_parquet(tmp_path / "BCO.parquet")
    assert str(b["date"].max().date()) == str(dates[-1].date())


def test_refresh_never_replaces_with_shrunken_history(tmp_path, universe):
    d_long = pd.bdate_range("2026-05-01", periods=20)
    _frame(d_long, [float(i) for i in range(20)], ticker="BCO").to_parquet(
        tmp_path / "BCO.parquet", index=False)
    d_short = pd.bdate_range("2026-06-20", periods=3)
    transport = _intrinio_transport(
        {"sa": _rows(d_short, [1, 2, 3]), "sb": _rows(d_short, [5, 6, 7])}, [])
    deltas = refresh_active_prices(
        universe, api_key="k", out_dir=tmp_path, reference_column="AREF",
        transport=transport, pause=0, rebuild_cache=False)
    assert deltas["suspect"] == 1
    b = pd.read_parquet(tmp_path / "BCO.parquet")
    assert len(b) == 20, "shrunken vendor response must not clobber good history"


# --- FSDS quarter logic --------------------------------------------------------------

def test_latest_elapsed_quarter():
    assert latest_elapsed_quarter("2026-07-02") == "2026q2"
    assert latest_elapsed_quarter("2026-01-15") == "2025q4"
    assert latest_elapsed_quarter("2026-04-01") == "2026q1"
    assert latest_elapsed_quarter("2026-12-31") == "2026q3"


def _write_quarter(dirpath, quarter):
    pd.DataFrame({"x": [1]}).to_parquet(dirpath / f"{quarter}.parquet")


def test_missing_quarters_and_unreadable_selfheal(tmp_path):
    _write_quarter(tmp_path, "2026q1")
    (tmp_path / "2025q4.parquet").write_bytes(b"not a parquet")  # crash damage
    assert quarters_present(tmp_path) == ["2026q1"]
    missing = missing_quarters("2026-07-02", tmp_path, first="2025q3")
    assert missing == ["2025q3", "2025q4", "2026q2"]


def test_ingest_new_fsds_waiting_vs_failed(tmp_path):
    _write_quarter(tmp_path, "2026q1")

    def fake_ingest(quarter):
        if quarter == "2026q2":  # newest: not published yet (opaque retry error)
            raise RuntimeError("EDGAR request failed after 5 retries")
        if quarter == "2025q4":
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", "http://x"),
                response=httpx.Response(404))
        return {"quarter": quarter, "rows": 100}

    deltas = ingest_new_fsds("2026-07-02", tmp_path, ingest_fn=fake_ingest,
                             rebuild_wide=False, first="2025q3")
    assert deltas["ingested"] == ["2025q3"]
    assert deltas["waiting"] == ["2025q4", "2026q2"]  # explicit 404 + newest-opaque
    assert deltas["failed"] == []
    assert deltas["fact_rows"] == 100


def test_ingest_new_fsds_old_gap_is_failure(tmp_path):
    def fake_ingest(quarter):
        raise RuntimeError("boom")

    deltas = ingest_new_fsds("2026-07-02", tmp_path, ingest_fn=fake_ingest,
                             rebuild_wide=False, first="2026q1")
    assert deltas["failed"] == ["2026q1"]   # an OLD missing quarter is a real gap
    assert deltas["waiting"] == ["2026q2"]  # only the newest gets the benefit of doubt


# --- universe diff + renames -----------------------------------------------------------

def _uni(rows):
    return pd.DataFrame(rows, columns=["cik", "column", "security_id", "ticker",
                                       "name", "active", "priority", "clip_date"])


def test_universe_diff_classification():
    old = _uni([(1, "AAA", "s1", "AAA", "A", True, 0, pd.NaT),
                (2, "BBB", "s2", "BBB", "B", True, 0, pd.NaT),
                (3, "CCC~3", "s3", "CCC", "C", False, 0, pd.NaT),
                (4, "DDD", "s4", "DDD", "D", True, 0, pd.NaT)])
    new = _uni([(1, "AAA", "s1", "AAA", "A", True, 0, pd.NaT),      # unchanged
                (2, "BBBQ~2", "s2b", "BBBQ", "B", False, 0, pd.NaT),  # died (new ticker!)
                (3, "CCC", "s3b", "CCC", "C", True, 0, pd.NaT),     # revived
                (4, "EEE", "s4", "EEE", "D", True, 0, pd.NaT),      # ticker rename
                (5, "FFF", "s5", "FFF", "F", True, 0, pd.NaT)])     # added
    diff = universe_diff(old, new)
    assert diff["added"] == [5]
    assert diff["died"] == [2]
    assert diff["revived"] == [3]
    assert diff["renamed"] == [4]
    assert ("BBB", "BBBQ~2") in diff["renames"]
    assert ("DDD", "EEE") in diff["renames"]


def test_apply_renames_rewrites_internal_ticker(tmp_path):
    _frame(pd.bdate_range("2026-01-01", periods=5), [1., 2, 3, 4, 5],
           ticker="FOO").to_parquet(tmp_path / "FOO.parquet", index=False)
    res = apply_renames([("FOO", "FOO~9")], tmp_path)
    assert res["moved"] == [("FOO", "FOO~9")]
    assert not (tmp_path / "FOO.parquet").exists()
    moved = pd.read_parquet(tmp_path / "FOO~9.parquet")
    # the matrix pivot keys on the INTERNAL ticker column — it must be renamed too,
    # or the old name lingers in the matrix and a recycled ticker would collide
    assert (moved["ticker"] == "FOO~9").all()
    # replay after crash: source gone -> skip, target intact
    res2 = apply_renames([("FOO", "FOO~9")], tmp_path)
    assert res2["moved"] == [] and res2["missing"] == ["FOO"]


def test_apply_renames_keeps_fuller_target(tmp_path):
    """A death refetch already wrote the new column with MORE history (the OTC
    afterlife); the rename must keep it and drop the shorter source."""
    _frame(pd.bdate_range("2026-01-01", periods=5), [5.] * 5,
           ticker="BAR").to_parquet(tmp_path / "BAR.parquet", index=False)
    _frame(pd.bdate_range("2026-01-01", periods=9), [5.] * 9,
           ticker="BARQ~7").to_parquet(tmp_path / "BARQ~7.parquet", index=False)
    res = apply_renames([("BAR", "BARQ~7")], tmp_path)
    assert res["kept_target"] == [("BAR", "BARQ~7")]
    assert not (tmp_path / "BAR.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "BARQ~7.parquet")) == 9


def test_recent_dead_columns(tmp_path):
    uni = _uni([(1, "AAA", "s1", "AAA", "A", True, 0, pd.NaT),
                (2, "OLD~2", "s2", "OLD", "O", False, 0, pd.NaT),
                (3, "NEW~3", "s3", "NEW", "N", False, 0, pd.NaT)])
    _frame(pd.bdate_range("2020-01-01", periods=3), [1., 1, 1],
           ticker="OLD~2").to_parquet(tmp_path / "OLD~2.parquet", index=False)
    _frame(pd.bdate_range("2026-06-01", periods=3), [1., 1, 1],
           ticker="NEW~3").to_parquet(tmp_path / "NEW~3.parquet", index=False)
    got = recent_dead_columns(uni, tmp_path, today="2026-07-02", grace_days=120)
    assert got == ["NEW~3"]  # long-dead frozen; recent death still in grace window


# --- the no-retrain honesty guard ------------------------------------------------------

def test_ops_package_cannot_train():
    """The continuous loop only observes: nothing under stockscan.ops may reach
    a training entry point. A source-level guard is crude but unambiguous."""
    import pathlib

    import stockscan.ops as ops_pkg

    forbidden = ("save_artifact", "LGBMRegressor", "lgb.train", "from ..model import fit",
                 "model.fit(", " fit(")
    for path in pathlib.Path(ops_pkg.__path__[0]).glob("*.py"):
        src = path.read_text()
        for token in forbidden:
            assert token not in src, f"{path.name} references {token!r}"
