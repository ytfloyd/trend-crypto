"""Tests for the volbook 1-minute futures lake."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from volbook.bundle import Bar
from volbook.datalake import (
    CONTINUOUS_EXPIRY,
    DEFAULT_BARS_TABLE,
    DEFAULT_STATE_TABLE,
    MinuteLake,
    plan_backfill_chunks,
)


def _bar(ts: datetime, c: float = 100.0) -> Bar:
    return Bar(t=ts.isoformat(), o=c, h=c + 1, l=c - 1, c=c, v=10.0)


def test_minute_lake_creates_schema_and_upserts(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        conn = lake.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        assert DEFAULT_BARS_TABLE in tables
        assert DEFAULT_STATE_TABLE in tables

        bars_columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({DEFAULT_BARS_TABLE})").fetchall()
        }
        assert {"symbol", "expiry", "ts", "o", "h", "l", "c", "v"} <= bars_columns

        ts = datetime(2026, 4, 30, 14, 30, tzinfo=timezone.utc)
        bars = [_bar(ts), _bar(ts + timedelta(minutes=1), c=101.0)]
        written = lake.upsert_bars("ES", bars)
        assert written == 2
        assert lake.row_count("ES") == 2

        again = lake.upsert_bars(
            "ES", [_bar(ts + timedelta(minutes=1), c=999.0)]
        )
        assert again == 1
        assert lake.row_count("ES") == 2

        last_close = conn.execute(
            f"""
            SELECT c FROM {DEFAULT_BARS_TABLE}
            WHERE symbol = ? AND expiry = ? AND ts = ?
            """,
            ["ES", CONTINUOUS_EXPIRY, ts + timedelta(minutes=1)],
        ).fetchone()[0]
        assert last_close == 999.0


def test_minute_lake_separates_continuous_and_dated_bars(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        ts = datetime(2026, 4, 30, 14, 30, tzinfo=timezone.utc)
        lake.upsert_bars("ES", [_bar(ts, c=100.0)])
        lake.upsert_bars("ES", [_bar(ts, c=200.0)], expiry="202606")

        assert lake.row_count("ES") == 2
        assert lake.row_count("ES", expiry=CONTINUOUS_EXPIRY) == 1
        assert lake.row_count("ES", expiry="202606") == 1

        cont_state = lake.get_state("ES", expiry=CONTINUOUS_EXPIRY)
        dated_state = lake.get_state("ES", expiry="202606")
        assert cont_state is not None and cont_state.expiry == CONTINUOUS_EXPIRY
        assert dated_state is not None and dated_state.expiry == "202606"

        states = lake.list_states()
        assert {(s.symbol, s.expiry) for s in states} == {
            ("ES", CONTINUOUS_EXPIRY),
            ("ES", "202606"),
        }


def test_minute_lake_tracks_ingest_state(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        ts = datetime(2026, 4, 30, 14, 30, tzinfo=timezone.utc)
        head = datetime(2025, 11, 1, tzinfo=timezone.utc)
        lake.set_head_timestamp("ES", head)

        lake.upsert_bars(
            "ES",
            [_bar(ts), _bar(ts + timedelta(minutes=1), c=101.0)],
        )
        state = lake.get_state("ES")
        assert state is not None
        assert state.symbol == "ES"
        assert state.expiry == CONTINUOUS_EXPIRY
        assert state.head_ts == head
        assert state.earliest_ts == ts
        assert state.latest_ts == ts + timedelta(minutes=1)
        assert state.last_run_at is not None

        lake.record_notes("ES", "manual override")
        assert lake.get_state("ES").notes == "manual override"


def test_minute_lake_handles_unknown_symbol(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        assert lake.get_state("XYZ") is None
        assert lake.row_count("XYZ") == 0
        assert lake.list_states() == []


def test_plan_backfill_chunks_walks_forward_when_lake_empty() -> None:
    head = datetime(2025, 1, 1, tzinfo=timezone.utc)
    now = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
    chunks = plan_backfill_chunks(
        head_ts=head,
        earliest_ts=None,
        latest_ts=None,
        now=now,
        chunk_days=30,
    )
    assert chunks, "expected at least one forward chunk"
    last_end, _delta = chunks[-1]
    assert last_end == now
    assert all(c[0] <= now for c in chunks)


def test_plan_backfill_chunks_extends_backwards_to_head() -> None:
    head = datetime(2025, 1, 1, tzinfo=timezone.utc)
    earliest = datetime(2025, 4, 1, tzinfo=timezone.utc)
    latest = datetime(2026, 4, 1, tzinfo=timezone.utc)
    now = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)

    chunks = plan_backfill_chunks(
        head_ts=head,
        earliest_ts=earliest,
        latest_ts=latest,
        now=now,
        chunk_days=30,
    )
    ends = [end for end, _ in chunks]
    assert ends[0] >= latest, "first forward chunk should extend latest_ts"
    assert any(end <= earliest for end in ends), (
        "expected at least one chunk reaching back to head"
    )
    assert all(end >= head for end in ends)


@pytest.mark.parametrize("chunk_days", [1, 5, 30])
def test_plan_backfill_chunks_chunk_size(chunk_days: int) -> None:
    head = datetime(2026, 4, 1, tzinfo=timezone.utc)
    now = datetime(2026, 4, 30, tzinfo=timezone.utc)
    chunks = plan_backfill_chunks(
        head_ts=head,
        earliest_ts=None,
        latest_ts=None,
        now=now,
        chunk_days=chunk_days,
    )
    assert chunks
    expected_seconds = chunk_days * 24 * 3600
    for _end, delta in chunks:
        assert delta.total_seconds() == expected_seconds


def test_stitch_continuous_series_picks_front_month(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        # Two contracts: M (June '26) and U (Sep '26). With roll_days=0 the
        # June contract is active for ts < 2026-06-01, then the September
        # contract takes over.
        ts_may = datetime(2026, 5, 15, 14, 0, tzinfo=timezone.utc)
        ts_jun = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)
        ts_aug = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
        lake.upsert_bars("ES", [_bar(ts_may, c=100.0)], expiry="202606")
        lake.upsert_bars("ES", [_bar(ts_jun, c=110.0)], expiry="202606")  # post-roll
        lake.upsert_bars("ES", [_bar(ts_may, c=200.0)], expiry="202609")  # pre-roll for U
        lake.upsert_bars("ES", [_bar(ts_jun, c=210.0)], expiry="202609")
        lake.upsert_bars("ES", [_bar(ts_aug, c=220.0)], expiry="202609")
        lake.upsert_bars("ES", [_bar(ts_may, c=999.0)])  # continuous, ignored

        df = lake.stitch_continuous_series("ES", roll_days_before_expiry=0)
        assert df.height == 3
        rows = sorted(zip(df["ts"].to_list(), df["expiry"].to_list(), df["c"].to_list()))
        assert rows[0][1] == "202606" and rows[0][2] == 100.0  # May -> June front
        assert rows[1][1] == "202609" and rows[1][2] == 210.0  # June after roll -> September
        assert rows[2][1] == "202609" and rows[2][2] == 220.0  # August -> September


def test_stitch_continuous_series_roll_days(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        ts_may_25 = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)
        lake.upsert_bars("ES", [_bar(ts_may_25, c=100.0)], expiry="202606")
        lake.upsert_bars("ES", [_bar(ts_may_25, c=200.0)], expiry="202609")

        # roll_days=8 → June contract retires on 2026-05-24, so on May 25 the
        # September contract is now front.
        df_early = lake.stitch_continuous_series("ES", roll_days_before_expiry=8)
        assert df_early.height == 1
        assert df_early["expiry"].to_list() == ["202609"]

        # roll_days=0 → June contract still active on May 25.
        df_late = lake.stitch_continuous_series("ES", roll_days_before_expiry=0)
        assert df_late["expiry"].to_list() == ["202606"]


def test_stitch_continuous_series_empty_returns_empty_frame(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        df = lake.stitch_continuous_series("ES")
        assert df.height == 0
        assert set(df.columns) == {"ts", "expiry", "o", "h", "l", "c", "v"}


def test_stitch_continuous_series_rejects_far_dated_only_cl(tmp_path: Path) -> None:
    db = tmp_path / "lake.duckdb"
    with MinuteLake(db) as lake:
        ts = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
        lake.upsert_bars("CL", [_bar(ts, c=70.0)], expiry="202702")

        with pytest.raises(ValueError, match="eligible front range.*202510"):
            lake.stitch_continuous_series("CL")
