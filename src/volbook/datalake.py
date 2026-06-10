"""DuckDB-backed minute-bar lake for the volbook futures universe.

Persists 1-minute OHLCV bars for each product into a single DuckDB file
alongside per-(symbol, expiry) ingest state so backfills can resume cleanly.
The ``expiry`` dimension allows the lake to hold *both* IBKR continuous
front-month bars (``expiry='continuous'``) and dated contract bars
(``expiry='YYYYMM'``) — the latter is what powers the deep-history dated
walk added on top of the continuous trailing-window store.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .bundle import Bar

DEFAULT_LAKE_PATH = Path("../data/futures_market.duckdb")
DEFAULT_BARS_TABLE = "bars_1m"
DEFAULT_STATE_TABLE = "ingest_state"
CONTINUOUS_EXPIRY = "continuous"


@dataclass(frozen=True)
class IngestState:
    """Tracking record describing how much data we have for a (symbol, expiry)."""

    symbol: str
    expiry: str
    earliest_ts: datetime | None
    latest_ts: datetime | None
    head_ts: datetime | None
    last_run_at: datetime | None
    notes: str = ""


class MinuteLake:
    """Thin wrapper around the futures minute DuckDB.

    Owns the schema and exposes the small set of operations the IBKR
    backfill / refresh CLIs need. The connection is opened lazily so
    tests can construct a ``MinuteLake`` against a temp path without
    side effects on the real lake.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_LAKE_PATH,
        *,
        bars_table: str = DEFAULT_BARS_TABLE,
        state_table: str = DEFAULT_STATE_TABLE,
    ) -> None:
        self.path = Path(path)
        self.bars_table = bars_table
        self.state_table = state_table
        self._conn: Any = None

    def connect(self) -> Any:
        if self._conn is None:
            import duckdb

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.path))
            self._ensure_schema(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "MinuteLake":
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _ensure_schema(self, conn: Any) -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.bars_table} (
                symbol VARCHAR NOT NULL,
                expiry VARCHAR NOT NULL,
                ts TIMESTAMP WITH TIME ZONE NOT NULL,
                o DOUBLE,
                h DOUBLE,
                l DOUBLE,
                c DOUBLE,
                v DOUBLE,
                fetched_at TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (symbol, expiry, ts)
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.state_table} (
                symbol VARCHAR NOT NULL,
                expiry VARCHAR NOT NULL,
                earliest_ts TIMESTAMP WITH TIME ZONE,
                latest_ts TIMESTAMP WITH TIME ZONE,
                head_ts TIMESTAMP WITH TIME ZONE,
                last_run_at TIMESTAMP WITH TIME ZONE,
                notes VARCHAR,
                PRIMARY KEY (symbol, expiry)
            )
            """
        )

    @contextmanager
    def _txn(self) -> Iterator[Any]:
        conn = self.connect()
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def upsert_bars(
        self,
        symbol: str,
        bars: Iterable[Bar],
        *,
        expiry: str = CONTINUOUS_EXPIRY,
    ) -> int:
        """Insert/replace minute bars for ``(symbol, expiry)``. Returns rows written."""
        rows = [
            (
                symbol,
                expiry,
                _parse_ts(b.t),
                float(b.o),
                float(b.h),
                float(b.l),
                float(b.c),
                float(b.v),
            )
            for b in bars
        ]
        if not rows:
            return 0
        now = datetime.now(timezone.utc)
        with self._txn() as conn:
            conn.executemany(
                f"""
                INSERT INTO {self.bars_table}
                    (symbol, expiry, ts, o, h, l, c, v, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, expiry, ts) DO UPDATE SET
                    o = excluded.o,
                    h = excluded.h,
                    l = excluded.l,
                    c = excluded.c,
                    v = excluded.v,
                    fetched_at = excluded.fetched_at
                """,
                [(*r, now) for r in rows],
            )
            self._refresh_state(conn, symbol, expiry, last_run_at=now)
        return len(rows)

    def _refresh_state(
        self,
        conn: Any,
        symbol: str,
        expiry: str,
        *,
        head_ts: datetime | None = None,
        notes: str | None = None,
        last_run_at: datetime | None = None,
    ) -> None:
        bounds = conn.execute(
            f"""
            SELECT MIN(ts), MAX(ts)
            FROM {self.bars_table}
            WHERE symbol = ? AND expiry = ?
            """,
            [symbol, expiry],
        ).fetchone()
        earliest_ts, latest_ts = bounds if bounds else (None, None)
        existing = conn.execute(
            f"""
            SELECT head_ts, last_run_at, notes
            FROM {self.state_table}
            WHERE symbol = ? AND expiry = ?
            """,
            [symbol, expiry],
        ).fetchone()
        prior_head, prior_last, prior_notes = existing if existing else (None, None, None)

        new_head = head_ts if head_ts is not None else prior_head
        new_last = last_run_at if last_run_at is not None else prior_last
        new_notes = notes if notes is not None else (prior_notes or "")

        conn.execute(
            f"""
            INSERT INTO {self.state_table}
                (symbol, expiry, earliest_ts, latest_ts, head_ts, last_run_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, expiry) DO UPDATE SET
                earliest_ts = excluded.earliest_ts,
                latest_ts = excluded.latest_ts,
                head_ts = excluded.head_ts,
                last_run_at = excluded.last_run_at,
                notes = excluded.notes
            """,
            [symbol, expiry, earliest_ts, latest_ts, new_head, new_last, new_notes],
        )

    def set_head_timestamp(
        self,
        symbol: str,
        head_ts: datetime | None,
        *,
        expiry: str = CONTINUOUS_EXPIRY,
    ) -> None:
        with self._txn() as conn:
            self._refresh_state(conn, symbol, expiry, head_ts=head_ts)

    def record_notes(
        self,
        symbol: str,
        notes: str,
        *,
        expiry: str = CONTINUOUS_EXPIRY,
    ) -> None:
        with self._txn() as conn:
            self._refresh_state(conn, symbol, expiry, notes=notes)

    def get_state(
        self,
        symbol: str,
        *,
        expiry: str = CONTINUOUS_EXPIRY,
    ) -> IngestState | None:
        conn = self.connect()
        row = conn.execute(
            f"""
            SELECT symbol, expiry, earliest_ts, latest_ts, head_ts, last_run_at, notes
            FROM {self.state_table}
            WHERE symbol = ? AND expiry = ?
            """,
            [symbol, expiry],
        ).fetchone()
        if row is None:
            return None
        return IngestState(
            symbol=row[0],
            expiry=row[1],
            earliest_ts=row[2],
            latest_ts=row[3],
            head_ts=row[4],
            last_run_at=row[5],
            notes=row[6] or "",
        )

    def list_states(
        self,
        *,
        expiry: str | None = None,
    ) -> Sequence[IngestState]:
        conn = self.connect()
        if expiry is None:
            rows = conn.execute(
                f"""
                SELECT symbol, expiry, earliest_ts, latest_ts, head_ts, last_run_at, notes
                FROM {self.state_table}
                ORDER BY symbol, expiry
                """,
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT symbol, expiry, earliest_ts, latest_ts, head_ts, last_run_at, notes
                FROM {self.state_table}
                WHERE expiry = ?
                ORDER BY symbol
                """,
                [expiry],
            ).fetchall()
        return [
            IngestState(
                symbol=r[0],
                expiry=r[1],
                earliest_ts=r[2],
                latest_ts=r[3],
                head_ts=r[4],
                last_run_at=r[5],
                notes=r[6] or "",
            )
            for r in rows
        ]

    def stitch_continuous_series(
        self,
        symbol: str,
        *,
        roll_days_before_expiry: int = 0,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        front_month_guard: Any | None = None,
        calendar: Any | None = None,
    ) -> Any:
        """Return a continuous front-month series for ``symbol`` from dated bars.

        For each timestamp, the row from the contract with the smallest
        ``expiry`` whose roll point has not yet been reached is chosen.
        The roll point is the first calendar day of the expiry month
        minus ``roll_days_before_expiry`` days; products that trade past
        first-of-month can shift this value (e.g. ``8`` for ES which
        rolls ~third Friday).

        Continuous-source rows (``expiry='continuous'``) are excluded —
        this view is intended for the dated-walk lake. Returns a
        ``polars.DataFrame`` ordered by ``ts`` with columns
        ``(ts, expiry, o, h, l, c, v)``.
        """
        import polars as pl

        from .continuous import FrontMonthGuard

        conn = self.connect()
        guard = front_month_guard or FrontMonthGuard()
        coverage = self.front_month_coverage(
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            front_month_guard=guard,
            calendar=calendar,
        )
        invalid = coverage[coverage.get("front_month_guard_status") == "invalid"] if not coverage.empty else coverage
        if not invalid.empty and guard.on_missing != "mark":
            first = invalid.iloc[0]
            raise ValueError(
                "front-month coverage guard failed for "
                f"{symbol.upper()} on {first.get('session_date')}: {first.get('reason')}. "
                f"Available expiries: {first.get('available_expiries')}; "
                f"eligible front range: {first.get('eligible_expiries')}. "
                "Backfill missing near/front dated contracts before using dated_front."
            )
        clauses = ["symbol = ?", "expiry != ?"]
        params: list[Any] = [symbol, CONTINUOUS_EXPIRY]
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts < ?")
            params.append(end_ts)
        where_clause = " AND ".join(clauses)
        roll_days = max(0, int(roll_days_before_expiry))
        sql = f"""
            WITH active AS (
                SELECT
                    ts,
                    expiry,
                    o, h, l, c, v,
                    strptime(expiry || '01', '%Y%m%d')
                        - INTERVAL '{roll_days} days' AS roll_point
                FROM {self.bars_table}
                WHERE {where_clause}
            ),
            ranked AS (
                SELECT ts, expiry, o, h, l, c, v,
                       row_number() OVER (
                           PARTITION BY ts ORDER BY expiry ASC
                       ) AS rn
                FROM active
                WHERE ts < roll_point
            )
            SELECT ts, expiry, o, h, l, c, v
            FROM ranked
            WHERE rn = 1
            ORDER BY ts
        """
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return pl.DataFrame(
                schema={
                    "ts": pl.Datetime("us", "UTC"),
                    "expiry": pl.Utf8,
                    "o": pl.Float64,
                    "h": pl.Float64,
                    "l": pl.Float64,
                    "c": pl.Float64,
                    "v": pl.Float64,
                }
            )
        out = pl.DataFrame(
            rows,
            schema=["ts", "expiry", "o", "h", "l", "c", "v"],
            orient="row",
        )
        if not invalid.empty and guard.on_missing == "mark":
            invalid_dates = {str(value) for value in invalid["session_date"].to_list()}
            out = out.with_columns(
                pl.col("ts").dt.date().cast(pl.Utf8).is_in(invalid_dates).not_().alias("front_month_valid")
            )
        return out

    def front_month_coverage(
        self,
        symbol: str,
        *,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        front_month_guard: Any | None = None,
        calendar: Any | None = None,
    ) -> Any:
        """Return front-month coverage diagnostics for dated bars in the lake."""
        import pandas as pd

        from .continuous import FrontMonthGuard, evaluate_front_month_coverage

        guard = front_month_guard or FrontMonthGuard()
        clauses = ["symbol = ?", "expiry != ?"]
        params: list[Any] = [symbol.upper(), CONTINUOUS_EXPIRY]
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts < ?")
            params.append(end_ts)
        rows = self.connect().execute(
            f"""
            SELECT ts, symbol, expiry, o, h, l, c, v
            FROM {self.bars_table}
            WHERE {" AND ".join(clauses)}
            ORDER BY ts, expiry
            """,
            params,
        ).fetchall()
        frame = pd.DataFrame(rows, columns=["ts", "symbol", "expiry", "o", "h", "l", "c", "v"])
        return evaluate_front_month_coverage(frame, symbol=symbol.upper(), guard=guard, calendar=calendar)

    def institutional_continuous_series(
        self,
        symbol: str,
        *,
        adjustment: str = "additive",
        roll_policy: Any | None = None,
        calendar: Any | None = None,
        front_month_guard: Any | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ) -> Any:
        """Return an institutional CL continuous series from dated bars.

        This path excludes opaque vendor continuous rows and delegates CL roll
        scheduling/back-adjustment to ``volbook.continuous``.
        """
        import pandas as pd

        from .continuous import construct_continuous_series

        conn = self.connect()
        clauses = ["symbol = ?", "expiry != ?"]
        params: list[Any] = [symbol.upper(), CONTINUOUS_EXPIRY]
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts < ?")
            params.append(end_ts)
        rows = conn.execute(
            f"""
            SELECT ts, symbol, expiry, o, h, l, c, v
            FROM {self.bars_table}
            WHERE {" AND ".join(clauses)}
            ORDER BY ts, expiry
            """,
            params,
        ).fetchall()
        frame = pd.DataFrame(rows, columns=["ts", "symbol", "expiry", "o", "h", "l", "c", "v"])
        return construct_continuous_series(
            frame,
            symbol=symbol.upper(),
            policy=roll_policy,
            adjustment=adjustment,
            calendar=calendar,
            front_month_guard=front_month_guard,
        )

    def row_count(
        self,
        symbol: str | None = None,
        *,
        expiry: str | None = None,
    ) -> int:
        conn = self.connect()
        clauses: list[str] = []
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if expiry is not None:
            clauses.append("expiry = ?")
            params.append(expiry)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        row = conn.execute(
            f"SELECT COUNT(*) FROM {self.bars_table}{where}",
            params,
        ).fetchone()
        return int(row[0]) if row else 0


def _parse_ts(value: str | datetime) -> datetime:
    """Normalise string ISO timestamps to tz-aware ``datetime``."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"unrecognised timestamp {value!r}") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def plan_backfill_chunks(
    *,
    head_ts: datetime,
    earliest_ts: datetime | None,
    latest_ts: datetime | None,
    now: datetime,
    chunk_days: int = 30,
) -> list[tuple[datetime, datetime]]:
    """Return ``(end, duration)`` chunks needed to walk back to ``head_ts``.

    The IBKR ``reqHistoricalData`` API for 1-minute bars on **dated**
    futures contracts accepts ``durationStr`` up to roughly ``"30 D"`` per
    request and walks backward from ``endDateTime``. We backfill missing
    history on both sides of any existing data: forward from ``latest_ts``
    to ``now`` and backward from ``earliest_ts`` to ``head_ts``.

    Note: This planner is intended for the dated-walk path. IBKR rejects
    ``endDateTime`` on continuous (``ContFuture``) historical requests, so
    the continuous-trailing-window path does *not* use this function.
    """
    chunks: list[tuple[datetime, datetime]] = []
    chunk_seconds = chunk_days * 24 * 3600

    forward_start = latest_ts if latest_ts is not None else None
    if forward_start is None:
        forward_start = max(head_ts, now - _seconds(chunk_seconds))
    cursor = forward_start
    while cursor < now:
        end = min(cursor + _seconds(chunk_seconds), now)
        chunks.append((end, _seconds(chunk_seconds)))
        cursor = end

    if earliest_ts is not None and earliest_ts > head_ts:
        cursor = earliest_ts
        while cursor > head_ts:
            end = cursor
            cursor = max(end - _seconds(chunk_seconds), head_ts)
            chunks.append((end, _seconds(chunk_seconds)))

    return chunks


def _seconds(secs: int) -> Any:
    """Helper to keep the planner readable; returns a ``timedelta``."""
    from datetime import timedelta

    return timedelta(seconds=secs)
