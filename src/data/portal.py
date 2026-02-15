from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import duckdb
import polars as pl

from common.config import DataConfig
from common.timeframe import timeframe_to_seconds
from utils.duckdb_inspect import (
    describe_table,
    infer_native_timeframe,
    resolve_bars_table,
    resolve_ts_column,
    validate_required_columns,
)


class DataPortal:
    def __init__(self, cfg: DataConfig, *, strict_validation: bool = True):
        self.cfg = cfg
        self.strict_validation = strict_validation
        self.last_provenance: dict[str, Any] = {}

    def _infer_native_timeframe_from_column(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str,
        symbol: str,
    ) -> Optional[str]:
        query = f"SELECT DISTINCT timeframe FROM {table} WHERE symbol = ?"
        rows = conn.execute(query, [symbol]).fetchall()
        if not rows:
            return None
        values = [r[0] for r in rows if r[0] is not None]
        if not values:
            return None
        # Choose finest (smallest seconds)
        secs = []
        for v in values:
            try:
                secs.append((timeframe_to_seconds(str(v)), str(v)))
            except Exception:
                continue
        if not secs:
            return None
        secs.sort(key=lambda x: x[0])
        return secs[0][1]

    def _validate_ohlc_sanity(self, bars: pl.DataFrame) -> None:
        sample = bars.head(100)
        if sample.is_empty():
            return
        highs = sample["high"]
        lows = sample["low"]
        opens = sample["open"]
        closes = sample["close"]
        if (highs < opens).any() or (highs < closes).any():
            raise ValueError("OHLC sanity check failed: high < open/close")
        if (lows > opens).any() or (lows > closes).any():
            raise ValueError("OHLC sanity check failed: low > open/close")

    def load_bars(self) -> pl.DataFrame:
        db_path = Path(self.cfg.db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"DuckDB database not found at {db_path}. Provide a valid db_path."
            )
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            resolved_table = resolve_bars_table(str(db_path), self.cfg.table, self.cfg.timeframe)
            columns = describe_table(str(db_path), resolved_table)
            validate_required_columns(columns)
            ts_col = resolve_ts_column(columns)
            has_timeframe_col = "timeframe" in columns

            # Determine native timeframe (explicit or inferred)
            native_tf = self.cfg.native_timeframe
            if native_tf is None:
                if has_timeframe_col:
                    native_tf = self._infer_native_timeframe_from_column(conn, resolved_table, self.cfg.symbol)
                if native_tf is None:
                    native_tf = infer_native_timeframe(
                        str(db_path),
                        resolved_table,
                        self.cfg.symbol,
                        ts_col=ts_col,
                    )

            requested_seconds = timeframe_to_seconds(self.cfg.timeframe)
            native_seconds = timeframe_to_seconds(native_tf)
            if native_seconds > requested_seconds:
                raise ValueError(
                    f"Requested timeframe {self.cfg.timeframe} is finer than native {native_tf}. "
                    f"Provide higher-frequency data."
                )
            if requested_seconds % native_seconds != 0:
                raise ValueError(
                    f"Requested timeframe {self.cfg.timeframe} is not an integer multiple of native {native_tf}."
                )

            # Funding and extra numeric columns
            base_cols = {ts_col, "symbol", "open", "high", "low", "close", "volume"}
            extra_cols = columns - base_cols - {"timeframe"}
            supported_extra = {"funding_rate", "funding_cost"}
            unknown_extra = extra_cols - supported_extra
            if unknown_extra and self.strict_validation:
                raise ValueError(
                    f"Unsupported extra columns in {resolved_table}: {sorted(unknown_extra)}. "
                    f"Supported extra columns: {sorted(supported_extra)}"
                )

            # Normalize timestamps to UTC for bucketing (explicit UTC)
            ts_expr = f"timezone('UTC', {ts_col})"

            # Filters
            filters = ["symbol = ?"]
            params = [self.cfg.symbol]
            if has_timeframe_col and self.cfg.native_timeframe:
                filters.append("timeframe = ?")
                params.append(self.cfg.native_timeframe)

            # Count native rows after filters (pre-resample)
            count_query = f"""
                SELECT COUNT(*) AS n
                FROM {resolved_table}
                WHERE {" AND ".join(filters)}
                  AND {ts_col} >= ?
                  AND {ts_col} <= ?
            """
            count_params = params + [self.cfg.start, self.cfg.end]
            native_rows = int(conn.execute(count_query, count_params).fetchone()[0])

            if requested_seconds == native_seconds:
                select_cols = [f"{ts_expr} AS ts", "symbol", "open", "high", "low", "close", "volume"]
                if "funding_rate" in columns:
                    select_cols.append("funding_rate")
                if "funding_cost" in columns:
                    select_cols.append("funding_cost")
                query = f"""
                    SELECT {", ".join(select_cols)}
                    FROM {resolved_table}
                    WHERE {" AND ".join(filters)}
                      AND {ts_col} >= ?
                      AND {ts_col} <= ?
                    ORDER BY {ts_col} ASC
                """
                params.extend([self.cfg.start, self.cfg.end])
                result = conn.execute(query, params).pl()
                resample_rule = "none"
            else:
                bucket_seconds = requested_seconds
                # Build bucketed resample query (UTC-aligned)
                bucket_expr = f"to_timestamp(floor(epoch({ts_expr}) / {bucket_seconds}) * {bucket_seconds})"

                agg_cols = [
                    f"{bucket_expr} AS ts",
                    "symbol",
                    f"arg_min(open, {ts_expr}) AS open",
                    "max(high) AS high",
                    "min(low) AS low",
                    f"arg_max(close, {ts_expr}) AS close",
                    "sum(volume) AS volume",
                    "count(*) AS bucket_rows",
                ]
                if "funding_rate" in columns:
                    # funding_rate is a per-minute rate; average approximates bucket rate
                    agg_cols.append("avg(funding_rate) AS funding_rate")
                if "funding_cost" in columns:
                    # funding_cost is additive; sum within bucket
                    agg_cols.append("sum(funding_cost) AS funding_cost")
                query = f"""
                    SELECT {", ".join(agg_cols)}
                    FROM {resolved_table}
                    WHERE {" AND ".join(filters)}
                      AND {ts_col} >= ?
                      AND {ts_col} <= ?
                    GROUP BY 1, 2
                    ORDER BY ts ASC
                """
                params.extend([self.cfg.start, self.cfg.end])
                result = conn.execute(query, params).pl()
                resample_rule = "duckdb_ohlcv_aggregate"

            if result.is_empty():
                raise ValueError(
                    f"No bars returned for {self.cfg.symbol} between {self.cfg.start} and {self.cfg.end}. "
                    f"Available data range may be outside requested bounds."
                )

            # Timezone normalization / validation
            ts_dtype = result.schema.get("ts")
            if hasattr(ts_dtype, "time_zone"):
                if ts_dtype.time_zone is None:
                    # Treat tz-naive as UTC explicitly
                    result = result.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
                elif ts_dtype.time_zone != "UTC":
                    result = result.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
            ts_dtype = result.schema.get("ts")
            if self.strict_validation and hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone != "UTC":
                raise ValueError("Timestamp column is not UTC after normalization")

            # Integrity checks (strict mode only)
            if self.strict_validation:
                if result["ts"].is_duplicated().any():
                    raise ValueError("Duplicate timestamps detected after resampling")
                if result.select(pl.all().is_null().any()).row(0)[0]:
                    raise ValueError("Null values detected in resampled bars")
                self._validate_ohlc_sanity(result)

            resample_time_range_before_drop = {
                "start": result[0, "ts"].isoformat(),
                "end": result[result.height - 1, "ts"].isoformat(),
            }

            # Incomplete bar policy (only for resampled data)
            expected_rows_per_bucket = None
            first_bucket_rows = None
            last_bucket_rows = None
            first_bucket_coverage = None
            last_bucket_coverage = None
            dropped_first = False
            dropped_last = False

            if requested_seconds > native_seconds:
                expected_rows_per_bucket = int(requested_seconds / native_seconds)
                if "bucket_rows" in result.columns:
                    first_bucket_rows = int(result[0, "bucket_rows"])
                    last_bucket_rows = int(result[result.height - 1, "bucket_rows"])
                    first_bucket_coverage = first_bucket_rows / expected_rows_per_bucket
                    last_bucket_coverage = last_bucket_rows / expected_rows_per_bucket

                    if self.cfg.drop_incomplete_bars:
                        if first_bucket_coverage is not None and first_bucket_coverage < self.cfg.min_bucket_coverage_frac:
                            result = result.slice(1, result.height - 1)
                            dropped_first = True
                        if result.height > 0 and last_bucket_coverage is not None and last_bucket_coverage < self.cfg.min_bucket_coverage_frac:
                            result = result.slice(0, result.height - 1)
                            dropped_last = True
                        if not self.strict_validation:
                            if dropped_first or dropped_last:
                                print("Warning: dropped incomplete resampled buckets")

            if result.is_empty():
                raise ValueError(
                    "Resampling produced no complete buckets; widen start/end or lower min_bucket_coverage_frac."
                )

            # Remove internal columns
            if "bucket_rows" in result.columns:
                result = result.drop("bucket_rows")

            bars_start = result[0, "ts"]
            bars_end = result[result.height - 1, "ts"]
            resampled_rows = result.height

            # Median delta hours on returned bars
            median_delta_hours = None
            if result.height >= 2:
                diffs = result.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
                if diffs.len() > 0:
                    median_delta_hours = float(diffs.median()) / 3600.0

            # Strict validation: downsample must reduce rows and match expected delta
            if self.strict_validation and requested_seconds != native_seconds:
                if resampled_rows >= native_rows:
                    raise ValueError("Resampling did not reduce row count; check native/requested timeframes.")
                expected_hours = requested_seconds / 3600.0
                if median_delta_hours is not None and abs(median_delta_hours - expected_hours) > 1e-6:
                    raise ValueError(
                        f"Median delta hours {median_delta_hours} does not match expected {expected_hours}."
                    )

            self.last_provenance = {
                "requested_timeframe": self.cfg.timeframe,
                "native_timeframe": native_tf,
                "resampling_rule": resample_rule,
                "bucket_alignment": "utc",
                "table": resolved_table,
                "filters": {"symbol": self.cfg.symbol},
                "aggregation_rules": {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "funding_rate": "mean" if "funding_rate" in result.columns else None,
                    "funding_cost": "sum" if "funding_cost" in result.columns else None,
                },
                "funding_rate_agg": "mean" if "funding_rate" in result.columns else None,
                "funding_cost_agg": "sum" if "funding_cost" in result.columns else None,
                "native_rows": native_rows,
                "resampled_rows": resampled_rows,
                "median_delta_hours": median_delta_hours,
                "expected_rows_per_bucket": expected_rows_per_bucket,
                "first_bucket_rows": first_bucket_rows,
                "last_bucket_rows": last_bucket_rows,
                "first_bucket_coverage": first_bucket_coverage,
                "last_bucket_coverage": last_bucket_coverage,
                "dropped_first_bucket": dropped_first,
                "dropped_last_bucket": dropped_last,
                "resample_time_range_before_drop": resample_time_range_before_drop,
                "resample_time_range_after_drop": {"start": bars_start.isoformat(), "end": bars_end.isoformat()},
                "bars_start": bars_start.isoformat(),
                "bars_end": bars_end.isoformat(),
                "columns": sorted(result.columns),
            }
            return result
        finally:
            conn.close()

