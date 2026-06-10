"""Research adapter over the existing ``volbook`` futures minute lake."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from volbook.continuous import (
    FrontMonthCoverageError,
    FrontMonthGuard,
    RollPolicy,
    calendar_from_name,
    construct_continuous_series,
    evaluate_front_month_coverage,
)
from volbook.datalake import CONTINUOUS_EXPIRY, MinuteLake

_OHLCV_COLUMNS = ["ts", "symbol", "expiry", "o", "h", "l", "c", "v"]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _PROJECT_ROOT.parent / "data"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DEFAULT_PARQUET_ROOT = _path_from_env(
    "VOLBOOK_PARQUET_ROOT",
    _DATA_ROOT / "futures_market_parquet",
)
DEFAULT_LAKE_PATH = _path_from_env(
    "VOLBOOK_LAKE_PATH",
    _DATA_ROOT / "futures_market.duckdb",
)


@dataclass(frozen=True)
class VolbookLoadSpec:
    """Describes one extraction from the volbook minute store."""

    symbol: str = "CL"
    source: str = "auto"
    contract_source: str = "dated_front"
    expiry: str | None = None
    roll_days_before_expiry: int = 5
    continuous_adjustment: str = "additive"
    continuous_roll_policy: str = "volume_crossover_with_calendar_guard"
    continuous_business_calendar: str = "nymex_observed"
    continuous_roll_window_business_days: int = 10
    continuous_forced_roll_business_days_before_last_trade: int = 3
    continuous_volume_crossover_sessions: int = 2
    front_month_guard: bool = True
    front_month_guard_max_curve_position: int = 2
    front_month_guard_on_missing: str = "raise"
    start_ts: str | datetime | None = None
    end_ts: str | datetime | None = None
    parquet_root: Path = DEFAULT_PARQUET_ROOT
    lake_path: Path = DEFAULT_LAKE_PATH


class VolbookResearchDataAdapter:
    """Load volbook 1-minute bars and build session-aware research bars."""

    def __init__(self, spec: VolbookLoadSpec) -> None:
        self.spec = spec
        self.last_load_metadata: dict[str, object] = {}
        self.last_roll_schedule: pd.DataFrame = pd.DataFrame()

    def load_minute_bars(self) -> pd.DataFrame:
        """Load normalized 1-minute bars for the configured symbol.

        ``contract_source='dated_front'`` preserves the old unadjusted
        front-month stitch. ``contract_source='institutional_continuous'`` uses
        dated contracts with CL exchange roll dates, volume crossover, and
        explicit adjustment metadata.
        """
        self.last_load_metadata = {}
        self.last_roll_schedule = pd.DataFrame()
        source = self._resolve_source()
        if source == "parquet":
            bars = self._load_from_parquet()
        elif source == "duckdb":
            bars = self._load_from_duckdb()
        else:
            raise ValueError("source must be 'auto', 'parquet', or 'duckdb'")
        return _standardize_minute_frame(bars, symbol=self.spec.symbol)

    def _resolve_source(self) -> str:
        source = self.spec.source.lower()
        if source != "auto":
            return source

        if self._parquet_paths():
            return "parquet"
        if self.spec.lake_path.exists():
            return "duckdb"
        raise FileNotFoundError(
            "no volbook data source found. Looked for Parquet bars under "
            f"{self.spec.parquet_root / 'bars_1m'} and DuckDB lake at "
            f"{self.spec.lake_path}. Set source/lake_path/parquet_root explicitly "
            "or export/populate the volbook futures lake."
        )

    def resample(
        self,
        minute_bars: pd.DataFrame,
        timeframe: str,
        *,
        session_tz: str = "America/New_York",
        session_start_hour: int = 18,
    ) -> pd.DataFrame:
        """Resample minute bars to OHLCV bars anchored to the futures session."""
        return resample_ohlcv(
            minute_bars,
            timeframe,
            session_tz=session_tz,
            session_start_hour=session_start_hour,
        )

    def _load_from_duckdb(self) -> pd.DataFrame:
        if not self.spec.lake_path.exists():
            raise FileNotFoundError(
                f"volbook DuckDB lake not found at {self.spec.lake_path}. "
                "Run the volbook minute refresh/backfill workflow or point lake_path "
                "at an existing futures_market.duckdb file."
            )
        contract_source = self.spec.contract_source.lower()
        with MinuteLake(self.spec.lake_path) as lake:
            if contract_source == "institutional_continuous":
                result = lake.institutional_continuous_series(
                    self.spec.symbol.upper(),
                    adjustment=self.spec.continuous_adjustment,
                    roll_policy=_roll_policy_from_spec(self.spec),
                    calendar=_calendar_from_spec(self.spec),
                    front_month_guard=_front_month_guard_from_spec(self.spec),
                    start_ts=_parse_bound(self.spec.start_ts),
                    end_ts=_parse_bound(self.spec.end_ts),
                )
                self.last_load_metadata = result.metadata
                self.last_roll_schedule = result.schedule
                return result.bars[_OHLCV_COLUMNS]

            if contract_source == "dated_front":
                coverage = lake.front_month_coverage(
                    self.spec.symbol.upper(),
                    front_month_guard=_front_month_guard_from_spec(self.spec),
                    calendar=_calendar_from_spec(self.spec),
                    start_ts=_parse_bound(self.spec.start_ts),
                    end_ts=_parse_bound(self.spec.end_ts),
                )
                self.last_load_metadata = _front_month_metadata(self.spec, coverage, "dated_front")
                _enforce_front_month_coverage(self.spec, coverage)
                frame = lake.stitch_continuous_series(
                    self.spec.symbol.upper(),
                    roll_days_before_expiry=self.spec.roll_days_before_expiry,
                    front_month_guard=_front_month_guard_from_spec(self.spec),
                    start_ts=_parse_bound(self.spec.start_ts),
                    end_ts=_parse_bound(self.spec.end_ts),
                ).to_pandas()
                frame["symbol"] = self.spec.symbol.upper()
                return frame[_OHLCV_COLUMNS]

            expiry = _resolve_expiry(contract_source, self.spec.expiry)
            if contract_source == "continuous":
                self.last_load_metadata = {
                    "construction": "vendor_continuous",
                    "front_month_coverage": {
                        "status": "opaque",
                        "reason": "vendor continuous source is not reconstructable from dated-contract coverage",
                    },
                }
            conn = lake.connect()
            clauses = ["symbol = ?", "expiry = ?"]
            params: list[object] = [self.spec.symbol.upper(), expiry]
            start_ts = _parse_bound(self.spec.start_ts)
            end_ts = _parse_bound(self.spec.end_ts)
            if start_ts is not None:
                clauses.append("ts >= ?")
                params.append(start_ts)
            if end_ts is not None:
                clauses.append("ts < ?")
                params.append(end_ts)
            rows = conn.execute(
                f"""
                SELECT ts, symbol, expiry, o, h, l, c, v
                FROM {lake.bars_table}
                WHERE {" AND ".join(clauses)}
                ORDER BY ts
                """,
                params,
            ).fetchall()
        return pd.DataFrame(rows, columns=_OHLCV_COLUMNS)

    def _load_from_parquet(self) -> pd.DataFrame:
        root = self.spec.parquet_root / "bars_1m"
        if not root.exists():
            raise FileNotFoundError(
                f"volbook Parquet export not found at {root}. Run "
                "`python scripts/volbook/export_lake.py` or use source='duckdb'."
            )
        paths = self._parquet_paths()
        if not paths:
            raise FileNotFoundError(
                f"no volbook Parquet bars found for {self.spec.symbol.upper()} under {root}"
            )

        frames = [_read_partitioned_parquet(path) for path in paths]
        bars = pd.concat(frames, ignore_index=True)
        contract_source = self.spec.contract_source.lower()
        if contract_source == "institutional_continuous":
            bars = bars[bars["expiry"].astype(str) != CONTINUOUS_EXPIRY]
            bars = _filter_bounds(bars, self.spec.start_ts, self.spec.end_ts)
            result = construct_continuous_series(
                bars,
                symbol=self.spec.symbol.upper(),
                policy=_roll_policy_from_spec(self.spec),
                adjustment=self.spec.continuous_adjustment,
                calendar=_calendar_from_spec(self.spec),
                front_month_guard=_front_month_guard_from_spec(self.spec),
            )
            self.last_load_metadata = result.metadata
            self.last_roll_schedule = result.schedule
            return result.bars[_OHLCV_COLUMNS]
        if contract_source == "dated_front":
            bars = bars[bars["expiry"].astype(str) != CONTINUOUS_EXPIRY]
            bars = _filter_bounds(bars, self.spec.start_ts, self.spec.end_ts)
            coverage = evaluate_front_month_coverage(
                bars,
                symbol=self.spec.symbol.upper(),
                guard=_front_month_guard_from_spec(self.spec),
                calendar=_calendar_from_spec(self.spec),
            )
            self.last_load_metadata = _front_month_metadata(self.spec, coverage, "dated_front")
            _enforce_front_month_coverage(self.spec, coverage)
            return _stitch_dated_front(
                bars,
                self.spec.roll_days_before_expiry,
                front_month_guard=_front_month_guard_from_spec(self.spec),
            )
        expiry = _resolve_expiry(contract_source, self.spec.expiry)
        if contract_source == "continuous":
            self.last_load_metadata = {
                "construction": "vendor_continuous",
                "front_month_coverage": {
                    "status": "opaque",
                    "reason": "vendor continuous source is not reconstructable from dated-contract coverage",
                },
            }
        bars = _filter_bounds(bars, self.spec.start_ts, self.spec.end_ts)
        return bars[bars["expiry"].astype(str) == expiry]

    def _parquet_paths(self) -> list[Path]:
        root = self.spec.parquet_root / "bars_1m"
        return sorted((root / f"symbol={self.spec.symbol.upper()}").glob("expiry=*/*.parquet"))


def _read_partitioned_parquet(path: Path) -> pd.DataFrame:
    import polars as pl

    frame = pl.read_parquet(path).to_pandas()
    if "symbol" not in frame:
        frame["symbol"] = path.parent.parent.name.split("=", 1)[1]
    if "expiry" not in frame:
        frame["expiry"] = path.parent.name.split("=", 1)[1]
    return frame


def _parse_bound(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _filter_bounds(
    frame: pd.DataFrame,
    start_ts: str | datetime | None,
    end_ts: str | datetime | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    start = _parse_bound(start_ts)
    end = _parse_bound(end_ts)
    if start is not None:
        out = out[out["ts"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["ts"] < pd.Timestamp(end)]
    return out


def resample_ohlcv(
    minute_bars: pd.DataFrame,
    timeframe: str,
    *,
    session_tz: str = "America/New_York",
    session_start_hour: int = 18,
) -> pd.DataFrame:
    """Build session-anchored OHLCV bars from normalized minute bars."""
    if minute_bars.empty:
        return _empty_bars_frame()
    rule = _timeframe_to_pandas_rule(timeframe)
    local_tz = ZoneInfo(session_tz)
    df = _standardize_minute_frame(minute_bars)
    local_index = df["ts"].dt.tz_convert(local_tz)
    working = df.set_index(local_index).sort_index()
    offset = pd.Timedelta(hours=session_start_hour)
    grouped = working.resample(rule, origin="start_day", offset=offset, closed="left", label="right")
    out = grouped.agg(
        o=("o", "first"),
        h=("h", "max"),
        l=("l", "min"),
        c=("c", "last"),
        v=("v", "sum"),
        bar_count=("c", "count"),
        symbol=("symbol", "last"),
        expiry=("expiry", "last"),
    ).dropna(subset=["o", "c"])
    out = out[out["bar_count"] > 0].copy()
    out["ts"] = out.index.tz_convert(timezone.utc)
    out["timeframe"] = timeframe
    out["source_kind"] = "volbook"
    out["session_tz"] = session_tz
    starts = out.index - pd.tseries.frequencies.to_offset(rule)
    out["session_date"] = [_session_date(ts, session_start_hour) for ts in starts]
    out["session_start_ts"] = [
        _session_start_ts(session_date, local_tz, session_start_hour)
        for session_date in out["session_date"]
    ]
    return out.reset_index(drop=True)[
        [
            "ts",
            "symbol",
            "expiry",
            "timeframe",
            "o",
            "h",
            "l",
            "c",
            "v",
            "bar_count",
            "session_date",
            "session_start_ts",
            "session_tz",
            "source_kind",
        ]
    ]


def _standardize_minute_frame(frame: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)
    missing = {"ts", "o", "h", "l", "c", "v"} - set(frame.columns)
    if missing:
        raise ValueError(f"minute bars missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out["symbol"] = out.get("symbol", symbol or "").astype(str).str.upper()
    out["expiry"] = out.get("expiry", "").astype(str)
    for col in ["o", "h", "l", "c", "v"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "o", "h", "l", "c"]).sort_values(["ts", "expiry"])
    return out[_OHLCV_COLUMNS].reset_index(drop=True)


def _stitch_dated_front(
    bars: pd.DataFrame,
    roll_days_before_expiry: int,
    *,
    front_month_guard: FrontMonthGuard | None = None,
) -> pd.DataFrame:
    bars = _standardize_minute_frame(bars)
    if bars.empty:
        return bars
    dated = bars[bars["expiry"].str.fullmatch(r"\d{6}")].copy()
    if dated.empty:
        return dated
    coverage = evaluate_front_month_coverage(
        dated,
        symbol=str(dated["symbol"].dropna().iloc[0]) if "symbol" in dated and not dated.empty else "CL",
        guard=front_month_guard or FrontMonthGuard(),
    )
    guard = front_month_guard or FrontMonthGuard()
    _enforce_front_month_coverage_for_guard(guard, coverage)
    roll_days = max(0, int(roll_days_before_expiry))
    expiry_month = pd.to_datetime(dated["expiry"] + "01", format="%Y%m%d", utc=True)
    dated["roll_point"] = expiry_month - pd.to_timedelta(roll_days, unit="D")
    active = dated[dated["ts"] < dated["roll_point"]]
    active = active.sort_values(["ts", "expiry"])
    return active.groupby("ts", as_index=False).first()[_OHLCV_COLUMNS]


def _resolve_expiry(contract_source: str, expiry: str | None) -> str:
    if contract_source == "continuous":
        return CONTINUOUS_EXPIRY
    if contract_source == "expiry" and expiry:
        return expiry
    raise ValueError(
        "contract_source must be 'dated_front', 'institutional_continuous', "
        "'continuous', or 'expiry' with expiry set"
    )


def _roll_policy_from_spec(spec: VolbookLoadSpec) -> RollPolicy:
    return RollPolicy(
        name=spec.continuous_roll_policy,  # type: ignore[arg-type]
        roll_window_business_days=spec.continuous_roll_window_business_days,
        forced_roll_business_days_before_last_trade=(
            spec.continuous_forced_roll_business_days_before_last_trade
        ),
        volume_crossover_sessions=spec.continuous_volume_crossover_sessions,
    )


def _front_month_guard_from_spec(spec: VolbookLoadSpec) -> FrontMonthGuard:
    return FrontMonthGuard(
        enabled=spec.front_month_guard,
        max_curve_position=spec.front_month_guard_max_curve_position,
        on_missing=spec.front_month_guard_on_missing,  # type: ignore[arg-type]
    )


def _calendar_from_spec(spec: VolbookLoadSpec):
    return calendar_from_name(spec.continuous_business_calendar)


def _front_month_metadata(
    spec: VolbookLoadSpec,
    coverage: pd.DataFrame,
    construction: str,
) -> dict[str, object]:
    invalid = coverage[coverage["front_month_guard_status"] == "invalid"] if "front_month_guard_status" in coverage else pd.DataFrame()
    return {
        "construction": construction,
        "front_month_guard": {
            "enabled": spec.front_month_guard,
            "max_curve_position": spec.front_month_guard_max_curve_position,
            "on_missing": spec.front_month_guard_on_missing,
        },
        "front_month_coverage": {
            "status": "valid" if invalid.empty else "invalid",
            "sessions_checked": int(coverage.shape[0]),
            "invalid_session_count": int(invalid.shape[0]),
            "first_invalid": _jsonable_record(invalid.iloc[0].to_dict()) if not invalid.empty else {},
        },
    }


def _enforce_front_month_coverage(spec: VolbookLoadSpec, coverage: pd.DataFrame) -> None:
    _enforce_front_month_coverage_for_guard(_front_month_guard_from_spec(spec), coverage)


def _enforce_front_month_coverage_for_guard(guard: FrontMonthGuard, coverage: pd.DataFrame) -> None:
    if not guard.enabled or guard.on_missing == "mark" or coverage.empty:
        return
    invalid = coverage[coverage["front_month_guard_status"] == "invalid"] if "front_month_guard_status" in coverage else pd.DataFrame()
    if invalid.empty:
        return
    first = invalid.iloc[0]
    raise FrontMonthCoverageError(
        f"front-month coverage guard failed for {first.get('symbol')} on {first.get('session_date')}: "
        f"{first.get('reason')}. Available expiries: {first.get('available_expiries')}; "
        f"eligible front range: {first.get('eligible_expiries')}. Backfill missing near/front contracts "
        "or use contract_source='continuous'/'expiry' explicitly."
    )


def _jsonable_record(record: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in record.items():
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def _timeframe_to_pandas_rule(timeframe: str) -> str:
    mapping = {"1h": "1h", "4h": "4h", "1d": "1D", "daily": "1D"}
    key = timeframe.lower()
    if key not in mapping:
        raise ValueError(f"unsupported timeframe {timeframe!r}; expected one of {sorted(mapping)}")
    return mapping[key]


def _session_date(ts: pd.Timestamp, session_start_hour: int) -> str:
    if ts.hour >= session_start_hour:
        return (ts.date() + pd.Timedelta(days=1)).isoformat()
    return ts.date().isoformat()


def _session_start_ts(session_date: str, tz: ZoneInfo, session_start_hour: int) -> pd.Timestamp:
    trade_date = pd.Timestamp(session_date).date()
    start_date = trade_date - pd.Timedelta(days=1)
    local = pd.Timestamp(datetime.combine(start_date, datetime.min.time())).replace(
        hour=session_start_hour,
        tzinfo=tz,
    )
    return local.tz_convert(timezone.utc)


def _empty_bars_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ts",
            "symbol",
            "expiry",
            "timeframe",
            "o",
            "h",
            "l",
            "c",
            "v",
            "bar_count",
            "session_date",
            "session_start_ts",
            "session_tz",
            "source_kind",
        ]
    )
