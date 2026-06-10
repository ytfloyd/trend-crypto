"""Tests for the futures lake → Parquet exporter.

Covers the two pieces of behavior outside callers will care about:

1. ``snapshot_lake`` makes a usable, point-in-time copy of the lake
   (including any WAL companion file) without touching the source.
2. ``export_lake`` produces a hive-partitioned ``bars_1m/`` tree that
   round-trips back into DuckDB with the expected rows, honors the
   ``--symbols`` / ``--include-continuous`` / ``--include-state`` flags,
   and cleans up the snapshot unless ``keep_snapshot`` is set.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from scripts.volbook import export_lake as exporter
from volbook.bundle import Bar
from volbook.datalake import CONTINUOUS_EXPIRY, MinuteLake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bar(ts: datetime, c: float = 100.0) -> Bar:
    return Bar(t=ts.isoformat(), o=c, h=c + 1, l=c - 1, c=c, v=10.0)


@pytest.fixture()
def populated_lake(tmp_path: Path) -> Path:
    """Lake with a deterministic, mixed-symbol/expiry payload.

    ES has continuous + two dated expiries; NQ has one dated expiry.
    Closing the lake guarantees DuckDB releases its file lock so the
    exporter can open it read-only.
    """
    db_path = tmp_path / "lake.duckdb"
    base = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)

    with MinuteLake(db_path) as lake:
        lake.upsert_bars(
            "ES",
            [_bar(base + timedelta(minutes=i)) for i in range(5)],
            expiry=CONTINUOUS_EXPIRY,
        )
        lake.upsert_bars(
            "ES",
            [_bar(base + timedelta(minutes=i), c=200.0) for i in range(3)],
            expiry="202603",
        )
        lake.upsert_bars(
            "ES",
            [_bar(base + timedelta(minutes=i), c=300.0) for i in range(2)],
            expiry="202606",
        )
        lake.upsert_bars(
            "NQ",
            [_bar(base + timedelta(minutes=i), c=400.0) for i in range(4)],
            expiry="202603",
        )
    return db_path


def _bars_row_count(parquet_root: Path) -> int:
    """Read every parquet file under ``bars_1m/`` and count rows."""
    pattern = (parquet_root / "bars_1m" / "**" / "*.parquet").as_posix()
    return duckdb.connect().execute(
        f"SELECT COUNT(*) FROM read_parquet('{pattern}', hive_partitioning=1)"
    ).fetchone()[0]


def _bars_distinct(parquet_root: Path) -> set[tuple[str, str]]:
    """Distinct (symbol, expiry) pairs from the exported dataset.

    Hive partitioning auto-casts numeric-looking values, so we ``CAST``
    expiry back to VARCHAR to match the lake's string representation.
    """
    pattern = (parquet_root / "bars_1m" / "**" / "*.parquet").as_posix()
    rows = duckdb.connect().execute(
        f"""
        SELECT DISTINCT symbol, CAST(expiry AS VARCHAR)
        FROM read_parquet('{pattern}', hive_partitioning=1)
        """
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


# ---------------------------------------------------------------------------
# snapshot_lake
# ---------------------------------------------------------------------------


def test_snapshot_lake_copies_main_file(tmp_path: Path, populated_lake: Path) -> None:
    snap = tmp_path / "snap.duckdb"
    exporter.snapshot_lake(populated_lake, snap)
    assert snap.exists()
    assert snap.stat().st_size > 0
    # Snapshot is openable read-only and still has the data.
    con = duckdb.connect(str(snap), read_only=True)
    try:
        rows = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    finally:
        con.close()
    assert rows == 5 + 3 + 2 + 4


def test_snapshot_lake_copies_wal_companion(tmp_path: Path) -> None:
    """If a WAL exists next to the lake we copy it; otherwise no error."""
    src = tmp_path / "lake.duckdb"
    src.write_bytes(b"fake-duckdb-bytes")
    wal = src.with_suffix(src.suffix + ".wal")
    wal.write_bytes(b"fake-wal-bytes")

    dst = tmp_path / "snap.duckdb"
    exporter.snapshot_lake(src, dst)

    assert dst.read_bytes() == b"fake-duckdb-bytes"
    assert dst.with_suffix(dst.suffix + ".wal").read_bytes() == b"fake-wal-bytes"


def test_snapshot_lake_clears_stale_wal_on_destination(tmp_path: Path) -> None:
    """A stale WAL at the destination gets removed when source has none."""
    src = tmp_path / "lake.duckdb"
    src.write_bytes(b"fresh-bytes")
    dst = tmp_path / "snap.duckdb"
    stale_wal = dst.with_suffix(dst.suffix + ".wal")
    stale_wal.parent.mkdir(parents=True, exist_ok=True)
    stale_wal.write_bytes(b"stale-wal")

    exporter.snapshot_lake(src, dst)

    assert dst.exists()
    assert not stale_wal.exists()


def test_snapshot_lake_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        exporter.snapshot_lake(tmp_path / "missing.duckdb", tmp_path / "out.duckdb")


# ---------------------------------------------------------------------------
# export_lake — partitioning + filters
# ---------------------------------------------------------------------------


def test_export_lake_partitions_by_symbol_and_expiry_dated_only(
    tmp_path: Path, populated_lake: Path
) -> None:
    out = tmp_path / "parquet_out"
    snap = tmp_path / "snap.duckdb"

    summary = exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=snap,
        use_snapshot=True,
        symbols=None,
        include_continuous=False,
        include_state=False,
    )

    # Default excludes continuous → 3+2+4 dated bars only.
    assert summary.bars_rows == 3 + 2 + 4
    assert summary.state_rows == 0
    assert _bars_row_count(out) == 9
    assert _bars_distinct(out) == {("ES", "202603"), ("ES", "202606"), ("NQ", "202603")}

    # Hive layout: symbol=…/expiry=… directories.
    parts = sorted(p.relative_to(out / "bars_1m").as_posix()
                   for p in (out / "bars_1m").glob("symbol=*/expiry=*"))
    assert parts == sorted([
        "symbol=ES/expiry=202603",
        "symbol=ES/expiry=202606",
        "symbol=NQ/expiry=202603",
    ])

    # Snapshot cleaned up by default.
    assert not snap.exists()


def test_export_lake_includes_continuous_when_flag_set(
    tmp_path: Path, populated_lake: Path
) -> None:
    out = tmp_path / "parquet_out"
    summary = exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=tmp_path / "snap.duckdb",
        use_snapshot=True,
        symbols=None,
        include_continuous=True,
        include_state=False,
    )
    assert summary.bars_rows == 5 + 3 + 2 + 4
    assert ("ES", CONTINUOUS_EXPIRY) in _bars_distinct(out)


def test_export_lake_filters_by_symbol(
    tmp_path: Path, populated_lake: Path
) -> None:
    out = tmp_path / "parquet_out"
    summary = exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=tmp_path / "snap.duckdb",
        use_snapshot=True,
        symbols=["NQ"],
        include_continuous=False,
        include_state=False,
    )
    assert summary.bars_rows == 4
    assert _bars_distinct(out) == {("NQ", "202603")}


def test_export_lake_lowercase_symbol_is_uppercased(
    tmp_path: Path, populated_lake: Path
) -> None:
    """Symbol filter is case-insensitive (lake stores upper-case)."""
    out = tmp_path / "parquet_out"
    summary = exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=tmp_path / "snap.duckdb",
        use_snapshot=True,
        symbols=["nq"],
        include_continuous=False,
        include_state=False,
    )
    assert summary.bars_rows == 4


def test_export_lake_writes_state_parquet_when_requested(
    tmp_path: Path, populated_lake: Path
) -> None:
    out = tmp_path / "parquet_out"
    summary = exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=tmp_path / "snap.duckdb",
        use_snapshot=True,
        symbols=None,
        include_continuous=True,
        include_state=True,
    )
    state_path = out / "ingest_state.parquet"
    assert state_path.exists()
    rows = duckdb.connect().execute(
        f"SELECT COUNT(*) FROM read_parquet('{state_path.as_posix()}')"
    ).fetchone()[0]
    # One ingest_state row per (symbol, expiry) populated above.
    assert rows == 4
    assert summary.state_rows == 4


def test_export_lake_keep_snapshot_preserves_file(
    tmp_path: Path, populated_lake: Path
) -> None:
    out = tmp_path / "parquet_out"
    snap = tmp_path / "snap.duckdb"
    exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=snap,
        use_snapshot=True,
        symbols=None,
        include_continuous=False,
        include_state=False,
        keep_snapshot=True,
    )
    assert snap.exists()


def test_export_lake_no_snapshot_reads_lake_directly(
    tmp_path: Path, populated_lake: Path
) -> None:
    """With ``use_snapshot=False`` we read the lake file in place."""
    out = tmp_path / "parquet_out"
    summary = exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=None,
        use_snapshot=False,
        symbols=None,
        include_continuous=True,
        include_state=False,
    )
    assert summary.bars_rows == 5 + 3 + 2 + 4


def test_export_lake_clears_old_partitions(
    tmp_path: Path, populated_lake: Path
) -> None:
    """Old partition directories should not linger between exports."""
    out = tmp_path / "parquet_out"
    exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=tmp_path / "snap.duckdb",
        use_snapshot=True,
        symbols=None,
        include_continuous=True,
        include_state=False,
    )
    assert (out / "bars_1m" / "symbol=ES" / f"expiry={CONTINUOUS_EXPIRY}").exists()

    exporter.export_lake(
        lake_path=populated_lake,
        output_dir=out,
        snapshot_path=tmp_path / "snap2.duckdb",
        use_snapshot=True,
        symbols=["NQ"],
        include_continuous=False,
        include_state=False,
    )

    remaining = sorted(
        p.relative_to(out / "bars_1m").as_posix()
        for p in (out / "bars_1m").glob("symbol=*/expiry=*")
    )
    assert remaining == ["symbol=NQ/expiry=202603"]
