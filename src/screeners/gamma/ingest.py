"""Bulk IB snapshot ingestion for the gamma screener universe.

Loops a list of tickers, calls IBVolSurfaceCollector.snapshot() on each,
and persists each ticker's vol surface to the shared ``vol_surface_snaps``
table. Uses one persistent IB connection across the whole batch to avoid
handshake overhead.

Designed to be resilient: one ticker failing does not abort the batch.
Errors are logged and reported in the summary.

Checkpoint / resume: by default, tickers that already have a
``vol_surface_snaps`` row for today (UTC) are skipped and counted as
succeeded. This makes a re-run of the same command (after a TWS restart
or a crash) free — it only snaps the names that haven't been captured
yet. Pass ``force_resnap=True`` to override and re-snap everything.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from common.logging import get_logger
from data.options.snapshot import IBVolSurfaceCollector

from .config import GammaScreenerConfig

logger = get_logger("gamma_screener_ingest")


@dataclass
class IngestResult:
    succeeded: list[str]
    failed: dict[str, str]  # symbol -> error message
    elapsed_secs: float

    @property
    def n_total(self) -> int:
        return len(self.succeeded) + len(self.failed)

    @property
    def success_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return len(self.succeeded) / self.n_total


def _already_snapped_today(
    conn: Any,
    symbol: str,
    as_of_date: date,
) -> bool:
    """Return True if ``vol_surface_snaps`` has a row for this symbol today (UTC).

    Uses a range predicate on ``snap_ts`` so the existing unique index
    ``(underlying, expiry, strike, right, snap_ts)`` narrows the scan to
    a single symbol; stops at the first hit with LIMIT 1.
    """
    start_ts = datetime.combine(as_of_date, datetime.min.time(), tzinfo=timezone.utc)
    end_ts = start_ts + timedelta(days=1)
    row = conn.execute(
        """
        SELECT 1 FROM vol_surface_snaps
        WHERE underlying = ?
          AND snap_ts >= ?
          AND snap_ts < ?
        LIMIT 1
        """,
        [symbol, start_ts, end_ts],
    ).fetchone()
    return row is not None


def snapshot_universe(
    tickers: list[str],
    cfg: GammaScreenerConfig,
    db_path: Optional[str] = None,
    progress_every: int = 5,
    force_resnap: bool = False,
    as_of_date: Optional[date] = None,
) -> IngestResult:
    """Snapshot option surfaces for a universe of equities via IB.

    Parameters
    ----------
    tickers : list of US equity symbols (e.g. ["AAPL", "MSFT", ...]).
    cfg : screener configuration (IB connection + snapshot params).
    db_path : override for the DuckDB path. Defaults to cfg.stocks_db_path.
    progress_every : log a progress line every N tickers.
    force_resnap : if True, snap every ticker even if one is already
        present for ``as_of_date`` in ``vol_surface_snaps``. Defaults to
        False, which is the resume-friendly behaviour.
    as_of_date : UTC date used for the "already snapped today" check.
        Defaults to today (UTC).

    Returns
    -------
    IngestResult summarising successes, skips (already-snapped, counted
    as succeeded), and failures.
    """
    db_path = db_path or cfg.stocks_db_path
    as_of_date = as_of_date or datetime.now(timezone.utc).date()
    collector = IBVolSurfaceCollector(
        db_path=db_path,
        host=cfg.ib_host,
        port=cfg.ib_port,
        client_id=cfg.ib_client_id,
    )

    start = time.time()
    succeeded: list[str] = []
    failed: dict[str, str] = {}
    skipped: list[str] = []

    # Partition tickers into (need-to-snap, already-snapped) before we
    # bother connecting to IB. No IB session is opened if every ticker
    # is already done for today.
    if force_resnap:
        todo = list(tickers)
    else:
        todo = []
        for symbol in tickers:
            if _already_snapped_today(collector._conn, symbol, as_of_date):
                skipped.append(symbol)
                succeeded.append(symbol)
            else:
                todo.append(symbol)
        if skipped:
            logger.info(
                "Resume: %d/%d tickers already snapped for %s, skipping",
                len(skipped), len(tickers), as_of_date,
            )

    if not todo:
        collector._conn.close()
        elapsed = time.time() - start
        logger.info(
            "Nothing to snap: all %d tickers already present for %s",
            len(tickers), as_of_date,
        )
        return IngestResult(
            succeeded=succeeded, failed=failed, elapsed_secs=elapsed,
        )

    try:
        collector.connect()
        logger.info(
            "Starting snapshot of %d tickers (IB %s:%d)",
            len(todo), cfg.ib_host, cfg.ib_port,
        )

        for i, symbol in enumerate(todo, start=1):
            try:
                surface = collector.snapshot(
                    symbol=symbol,
                    exchange="SMART",
                    sec_type="STK",
                    currency="USD",
                    max_expiries=cfg.max_expiries,
                    strike_range_pct=cfg.strike_range_pct,
                    min_tte_days=cfg.min_tte_days,
                    max_tte_days=cfg.max_tte_days,
                )
                n_slices = len(surface.slices)
                if n_slices < cfg.min_slices_required:
                    failed[symbol] = f"only {n_slices} valid slices"
                    continue
                succeeded.append(symbol)
            except Exception as exc:  # noqa: BLE001 - resilience is the point
                failed[symbol] = str(exc)
                logger.warning("%s snapshot failed: %s", symbol, exc)

            if i % progress_every == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                remaining = (len(todo) - i) / rate if rate > 0 else 0.0
                logger.info(
                    "Progress: %d/%d (%.1f tick/min, ~%.0fs remaining)",
                    i, len(todo), rate * 60.0, remaining,
                )

    finally:
        try:
            collector.close()
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.time() - start
    logger.info(
        "Snapshot batch done: %d succeeded (incl. %d resumed), %d failed in %.1fs",
        len(succeeded), len(skipped), len(failed), elapsed,
    )
    return IngestResult(succeeded=succeeded, failed=failed, elapsed_secs=elapsed)
