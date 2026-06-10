"""Default CL research configuration for the K2 first slice."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
DEFAULT_RESEARCH_ROOT = _path_from_env(
    "K2_RESEARCH_ROOT",
    _DATA_ROOT / "k2_systematic_macro",
)


@dataclass(frozen=True)
class CLResearchConfig:
    """Runtime configuration for the canonical CL research dataset."""

    symbol: str = "CL"
    primary_timeframe: str = "4h"
    secondary_timeframes: tuple[str, ...] = ("1h", "1d")
    source: str = "auto"  # "auto", "parquet", or "duckdb"
    contract_source: str = "dated_front"  # "dated_front", "institutional_continuous", "continuous", or "expiry"
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
    start_ts: str | None = "2025-09-01T00:00:00+00:00"
    end_ts: str | None = None
    parquet_root: Path = DEFAULT_PARQUET_ROOT
    lake_path: Path = DEFAULT_LAKE_PATH
    output_root: Path = DEFAULT_RESEARCH_ROOT
    session_tz: str = "America/New_York"
    session_start_hour: int = 18
    feature_timeframes: tuple[str, ...] = field(default_factory=lambda: ("1h", "4h", "1d"))

    @property
    def timeframes(self) -> tuple[str, ...]:
        """Return primary plus secondary timeframes without duplicates."""
        ordered = (self.primary_timeframe, *self.secondary_timeframes)
        return tuple(dict.fromkeys(ordered))
