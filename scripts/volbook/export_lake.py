#!/usr/bin/env python3
"""Export the futures minute lake to partitioned Parquet for outside readers.

DuckDB enforces an exclusive file lock while a writer is connected
(walker / refresh CLIs), which means external consumers can't open the
lake for ad-hoc queries while data is being collected. This exporter
sidesteps the contention by:

1. Snapshotting the lake to a temp file (default: ``/tmp/futures_lake_snapshot.duckdb``)
   at the OS level — DuckDB's single-file format is robust to ``cp``
   between transactions, and we copy the WAL alongside if present.
2. Streaming ``bars_1m`` out of the snapshot into a hive-partitioned
   Parquet dataset (``symbol=<sym>/expiry=<exp>/`` directories) that
   any tool with a Parquet reader (Polars, pandas, Spark, DuckDB,
   AWS Athena, etc.) can read with no Python required.
3. Optionally also exporting ``ingest_state`` as a single Parquet file
   so downstream consumers can introspect lake freshness.

Run nightly (or after the daily refresh) to keep the Parquet view in
sync. Reads against the Parquet output have no lock conflicts and run
in parallel; the price is one extra dump per day.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.logging import get_logger
from volbook.datalake import (
    CONTINUOUS_EXPIRY,
    DEFAULT_BARS_TABLE,
    DEFAULT_LAKE_PATH,
    DEFAULT_STATE_TABLE,
)

logger = get_logger("volbook.export_lake")

DEFAULT_OUTPUT_DIR = Path("../data/futures_market_parquet")
DEFAULT_SNAPSHOT_PATH = Path("/tmp/futures_lake_snapshot.duckdb")


@dataclass(frozen=True)
class ExportSummary:
    bars_rows: int
    bars_partitions: int
    state_rows: int
    output_dir: Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volbook.export_lake",
        description=(
            "Snapshot the futures minute lake and export bars_1m to a "
            "hive-partitioned Parquet dataset for outside consumers."
        ),
    )
    p.add_argument(
        "--lake-path",
        default=str(DEFAULT_LAKE_PATH),
        help=f"Source DuckDB path (default: {DEFAULT_LAKE_PATH}).",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            f"Parquet destination root (default: {DEFAULT_OUTPUT_DIR}). "
            "A `bars_1m/symbol=…/expiry=…/` tree is created inside; "
            "`ingest_state.parquet` is dropped at the same level when "
            "--include-state is set."
        ),
    )
    p.add_argument(
        "--snapshot-path",
        default=str(DEFAULT_SNAPSHOT_PATH),
        help=(
            "Path used when snapshotting the lake before reading "
            f"(default: {DEFAULT_SNAPSHOT_PATH}). Ignored with --no-snapshot."
        ),
    )
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help=(
            "Read the lake directly without snapshotting first. Only safe "
            "when no walker / refresh process is active; otherwise DuckDB's "
            "exclusive write lock will block the connection."
        ),
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Restrict export to the given symbols (default: all).",
    )
    p.add_argument(
        "--include-continuous",
        action="store_true",
        help="Include rows where expiry='continuous' (default: dated only).",
    )
    p.add_argument(
        "--include-state",
        action="store_true",
        help="Also export `ingest_state` to a single Parquet file.",
    )
    p.add_argument(
        "--bars-table",
        default=DEFAULT_BARS_TABLE,
        help="Source bars table name (default: bars_1m).",
    )
    p.add_argument(
        "--state-table",
        default=DEFAULT_STATE_TABLE,
        help="Source state table name (default: ingest_state).",
    )
    p.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="Leave the snapshot file in place after export (default: deleted).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def snapshot_lake(src: Path, dst: Path) -> None:
    """File-system snapshot of a DuckDB lake (data file + WAL if present).

    DuckDB's single-file format is point-in-time consistent under a
    plain ``cp``; we copy the optional ``.duckdb.wal`` alongside so any
    in-flight transactions are fully captured. The snapshot can then be
    opened read-only without contending with the live writer.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"lake not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    wal_src = src.with_suffix(src.suffix + ".wal")
    wal_dst = dst.with_suffix(dst.suffix + ".wal")
    if wal_src.exists():
        shutil.copy2(wal_src, wal_dst)
    elif wal_dst.exists():
        # Stale WAL from a prior snapshot would replay against the new copy.
        wal_dst.unlink()


def _connect_readonly(path: Path):
    import duckdb

    return duckdb.connect(str(path), read_only=True)


def export_lake(
    *,
    lake_path: Path,
    output_dir: Path,
    snapshot_path: Path | None,
    use_snapshot: bool,
    symbols: Sequence[str] | None,
    include_continuous: bool,
    include_state: bool,
    bars_table: str = DEFAULT_BARS_TABLE,
    state_table: str = DEFAULT_STATE_TABLE,
    keep_snapshot: bool = False,
) -> ExportSummary:
    """Snapshot (optionally) and export the lake. Idempotent."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bars_dir = output_dir / "bars_1m"
    state_path = output_dir / "ingest_state.parquet"

    if use_snapshot:
        if snapshot_path is None:
            raise ValueError("snapshot_path is required when use_snapshot=True")
        logger.info("Snapshotting %s -> %s", lake_path, snapshot_path)
        snapshot_lake(Path(lake_path), Path(snapshot_path))
        read_path = Path(snapshot_path)
    else:
        read_path = Path(lake_path)

    try:
        con = _connect_readonly(read_path)
    except Exception:
        logger.exception("Could not open lake for reading at %s", read_path)
        raise

    try:
        # Build a single SELECT with optional filters; DuckDB partitions
        # whatever the source query returns.
        where_clauses: list[str] = []
        if not include_continuous:
            where_clauses.append(f"expiry != '{CONTINUOUS_EXPIRY}'")
        if symbols:
            placeholders = ", ".join(f"'{s.upper()}'" for s in symbols)
            where_clauses.append(f"symbol IN ({placeholders})")
        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Wipe the bars partition directory before re-export so partitions
        # for symbols that no longer exist don't linger.
        if bars_dir.exists():
            shutil.rmtree(bars_dir)
        bars_dir.mkdir(parents=True, exist_ok=True)

        bars_rows = con.execute(
            f"SELECT COUNT(*) FROM {bars_table}{where_sql}"
        ).fetchone()[0]
        if bars_rows > 0:
            logger.info(
                "Exporting %d rows -> %s (partitioned by symbol, expiry)",
                bars_rows,
                bars_dir,
            )
            con.execute(
                f"""
                COPY (SELECT * FROM {bars_table}{where_sql})
                TO '{bars_dir.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY (symbol, expiry), OVERWRITE_OR_IGNORE)
                """
            )
        else:
            logger.warning("No rows matched the filter; bars dataset is empty")

        bars_partitions = sum(
            1 for _ in bars_dir.glob("symbol=*/expiry=*/*.parquet")
        )
        logger.info(
            "Wrote %d Parquet files across (symbol, expiry) partitions",
            bars_partitions,
        )

        state_rows = 0
        if include_state:
            if state_path.exists():
                state_path.unlink()
            state_rows = con.execute(
                f"SELECT COUNT(*) FROM {state_table}"
            ).fetchone()[0]
            logger.info(
                "Exporting %d ingest_state rows -> %s", state_rows, state_path
            )
            con.execute(
                f"""
                COPY (SELECT * FROM {state_table})
                TO '{state_path.as_posix()}'
                (FORMAT PARQUET)
                """
            )
    finally:
        con.close()
        if use_snapshot and not keep_snapshot:
            for p in (
                Path(snapshot_path),
                Path(snapshot_path).with_suffix(
                    Path(snapshot_path).suffix + ".wal"
                ),
            ):
                if p.exists():
                    p.unlink()

    return ExportSummary(
        bars_rows=bars_rows,
        bars_partitions=bars_partitions,
        state_rows=state_rows,
        output_dir=output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    summary = export_lake(
        lake_path=Path(args.lake_path),
        output_dir=Path(args.output_dir),
        snapshot_path=Path(args.snapshot_path),
        use_snapshot=not args.no_snapshot,
        symbols=args.symbols,
        include_continuous=args.include_continuous,
        include_state=args.include_state,
        bars_table=args.bars_table,
        state_table=args.state_table,
        keep_snapshot=args.keep_snapshot,
    )
    logger.info(
        "Export complete: %d bars across %d partitions; %d state rows; output=%s",
        summary.bars_rows,
        summary.bars_partitions,
        summary.state_rows,
        summary.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
