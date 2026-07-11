"""Nightly housekeeping: WAL-safe store backups, pruning, copy-truncate log rotation,
and the frozen-head staleness check."""

import json
import sqlite3


from stockscan.ops.health import head_staleness
from stockscan.ops.housekeeping import backup_artifacts, backup_stores, rotate_logs


def _make_store(path, rows: int) -> None:
    con = sqlite3.connect(str(path))
    con.execute("create table t (x integer)")
    con.executemany("insert into t values (?)", [(i,) for i in range(rows)])
    con.commit()
    con.close()


def test_backup_copies_queryable_snapshots_and_prunes(tmp_path):
    stores = tmp_path / "artifacts"
    out = tmp_path / "backups"
    stores.mkdir()
    _make_store(stores / "ops_state.sqlite", 3)
    _make_store(stores / "news.sqlite", 5)
    # two stale dated folders beyond keep=2 (with today's) — oldest must go
    for day in ("2026-06-01", "2026-06-02"):
        (out / day).mkdir(parents=True)

    d = backup_stores(stores_dir=stores, out_dir=out, keep=2, today="2026-07-05")
    assert d["copied"] == ["news.sqlite", "ops_state.sqlite"]
    assert d["pruned"] == ["2026-06-01"]
    con = sqlite3.connect(str(out / "2026-07-05" / "ops_state.sqlite"))
    assert con.execute("select count(*) from t").fetchone()[0] == 3
    con.close()
    # idempotent: the same day re-runs into the same folder without error
    d2 = backup_stores(stores_dir=stores, out_dir=out, keep=2, today="2026-07-05")
    assert d2["copied"] == ["news.sqlite", "ops_state.sqlite"]


def test_backup_degrades_on_a_corrupt_store_and_keeps_going(tmp_path):
    stores = tmp_path / "artifacts"
    out = tmp_path / "backups"
    stores.mkdir()
    (stores / "corrupt.sqlite").write_bytes(b"this is not a database")
    _make_store(stores / "good.sqlite", 1)

    d = backup_stores(stores_dir=stores, out_dir=out, keep=5, today="2026-07-05")
    assert d["copied"] == ["good.sqlite"]
    assert d["_status"] == "degraded" and d["errors"][0]["store"] == "corrupt.sqlite"
    assert not (out / "2026-07-05" / "corrupt.sqlite").exists()  # no half snapshot


def test_backup_artifacts_copies_frozen_heads_and_paper_trail(tmp_path):
    stores = tmp_path / "artifacts"
    out = tmp_path / "backups"
    (stores / "model").mkdir(parents=True)
    (stores / "model" / "booster.txt").write_text("frozen bytes")
    (stores / "model" / "meta.json").write_text("{}")
    (stores / "paper_forward" / "signals").mkdir(parents=True)
    (stores / "paper_forward" / "baseline.json").write_text("{}")
    (stores / "paper_forward" / "signals" / "2026-06-30.jsonl").write_text("{}\n")

    d = backup_artifacts(stores_dir=stores, out_dir=out, today="2026-07-11")
    assert d["copied"] == ["model", "paper_forward"]
    # unfrozen heads are reported missing, never an error
    assert set(d["missing"]) == {"distress_model", "drawdown_model", "confidence_cal"}
    assert "_status" not in d
    day = out / "2026-07-11"
    assert (day / "model" / "booster.txt").read_text() == "frozen bytes"
    assert (day / "paper_forward" / "signals" / "2026-06-30.jsonl").exists()
    # idempotent: same-day re-run overwrites cleanly
    d2 = backup_artifacts(stores_dir=stores, out_dir=out, today="2026-07-11")
    assert d2["copied"] == ["model", "paper_forward"]


def test_backup_artifacts_shares_the_dated_snapshot_with_stores(tmp_path):
    stores = tmp_path / "artifacts"
    out = tmp_path / "backups"
    stores.mkdir()
    _make_store(stores / "ops_state.sqlite", 2)
    (stores / "confidence_cal").mkdir()
    (stores / "confidence_cal" / "calibration.json").write_text("{}")

    backup_artifacts(stores_dir=stores, out_dir=out, today="2026-07-11")
    backup_stores(stores_dir=stores, out_dir=out, keep=2, today="2026-07-11")
    day = out / "2026-07-11"
    assert (day / "ops_state.sqlite").exists()
    assert (day / "confidence_cal" / "calibration.json").exists()


def test_rotate_logs_copy_truncates_only_fat_logs(tmp_path):
    fat = tmp_path / "nightly.log"
    slim = tmp_path / "nightly.err.log"
    fat.write_bytes(b"x" * 2048)
    slim.write_bytes(b"ok")

    d = rotate_logs(logs_dir=tmp_path, max_mb=0.001)   # ~1KB threshold
    assert d["rotated"] == ["nightly.log"]
    assert fat.stat().st_size == 0                      # truncated in place
    assert (tmp_path / "nightly.log.1").read_bytes() == b"x" * 2048
    assert slim.read_bytes() == b"ok"                   # small log untouched
    assert rotate_logs(logs_dir=tmp_path, max_mb=0.001)["noop"] is True


def _write_meta(base, rel, trained_through) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"trained_through": trained_through}))


def test_head_staleness_warns_only_past_the_window(tmp_path):
    _write_meta(tmp_path, "model/meta.json", "2026-03-31")
    _write_meta(tmp_path, "drawdown_model/meta.json", "2024-12-31")   # ancient

    c = head_staleness("2026-07-05", artifacts_dir=tmp_path, stale_days=400)
    assert c is not None and c.level == "warn" and c.ok is False
    assert "drawdown" in c.detail and "2024-12-31" in c.detail

    fresh = head_staleness("2026-07-05", artifacts_dir=tmp_path, stale_days=10_000)
    assert fresh.ok is True and "model" in fresh.detail


def test_head_staleness_absent_heads_are_silent(tmp_path):
    assert head_staleness("2026-07-05", artifacts_dir=tmp_path) is None
    _write_meta(tmp_path, "model/meta.json", "2026-06-30")
    c = head_staleness("2026-07-05", artifacts_dir=tmp_path)
    assert c.ok is True and "model 5d" in c.detail
