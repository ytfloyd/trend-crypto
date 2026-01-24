from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hash import hash_config
from .timeframe import hours_per_bar


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    db_path: str
    table: Optional[str] = None
    symbol: str = "BTC-USD"
    start: datetime
    end: datetime
    timeframe: str = "1h"
    native_timeframe: Optional[str] = None
    drop_incomplete_bars: bool = True
    min_bucket_coverage_frac: float = 0.8


class EngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    strict_validation: bool = True
    lookback: Optional[int] = None
    initial_cash: float = 100000.0


class StrategyConfigRaw(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Optional[str] = None
    fast: Optional[int] = None
    slow: Optional[int] = None
    vol_window: Optional[int] = None
    k: Optional[float] = None
    min_band: Optional[float] = None
    weight_on: float = 1.0
    window_units: str = "hours"
    target_vol_annual: Optional[float] = None
    vol_lookback: Optional[int] = None
    max_weight: float = 1.0
    adx_window: int = 14
    adx_threshold: float = 20.0
    enable_adx_filter: bool = False
    adx_entry_only: bool = False

    @model_validator(mode="after")
    def check_windows(self) -> "StrategyConfigRaw":
        if self.mode == "buy_and_hold":
            return self
        if self.mode == "ma_crossover_long_only":
            if self.fast is None or self.slow is None:
                raise ValueError("MA crossover requires fast and slow")
            if self.fast <= 0 or self.slow <= 0:
                raise ValueError("fast and slow windows must be positive")
            if self.fast >= self.slow:
                raise ValueError("fast must be shorter than slow")
            if self.window_units not in ("hours", "bars"):
                raise ValueError("window_units must be 'hours' or 'bars'")
            return self
        if self.fast is None or self.slow is None or self.vol_window is None or self.k is None or self.min_band is None:
            raise ValueError("MA strategy requires fast, slow, vol_window, k, min_band")
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("fast and slow windows must be positive")
        if self.fast >= self.slow:
            raise ValueError("fast must be shorter than slow")
        if self.window_units not in ("hours", "bars"):
            raise ValueError("window_units must be 'hours' or 'bars'")
        return self


class StrategyConfigResolved(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Optional[str] = None
    fast: Optional[int] = None
    slow: Optional[int] = None
    vol_window: Optional[int] = None
    k: Optional[float] = None
    min_band: Optional[float] = None
    weight_on: float = 1.0
    window_units: str = "bars"
    target_vol_annual: Optional[float] = None
    vol_lookback: Optional[int] = None
    max_weight: float = 1.0
    adx_window: int = 14
    adx_threshold: float = 20.0
    enable_adx_filter: bool = False
    adx_entry_only: bool = False

    @model_validator(mode="after")
    def check_windows(self) -> "StrategyConfigResolved":
        if self.window_units != "bars":
            raise ValueError("Resolved config must use window_units='bars'")
        if self.mode == "buy_and_hold":
            return self
        if self.mode == "ma_crossover_long_only":
            if self.fast is None or self.slow is None:
                raise ValueError("MA crossover requires fast and slow")
            if self.fast <= 0 or self.slow <= 0:
                raise ValueError("fast and slow windows must be positive")
            if self.fast >= self.slow:
                raise ValueError("fast must be shorter than slow")
            return self
        if self.fast is None or self.slow is None or self.vol_window is None or self.k is None or self.min_band is None:
            raise ValueError("MA strategy requires fast, slow, vol_window, k, min_band")
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("fast and slow windows must be positive")
        if self.fast >= self.slow:
            raise ValueError("fast must be shorter than slow")
        return self


class RiskConfigRaw(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vol_window: int
    target_vol_annual: Optional[float]
    max_weight: float = 1.0
    min_vol_floor: float = 1e-8
    window_units: str = "hours"

    @model_validator(mode="after")
    def check_window_units(self) -> "RiskConfigRaw":
        if self.window_units not in ("hours", "bars"):
            raise ValueError("window_units must be 'hours' or 'bars'")
        return self


class RiskConfigResolved(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vol_window: int
    target_vol_annual: Optional[float]
    max_weight: float = 1.0
    min_vol_floor: float = 1e-8
    window_units: str = "bars"

    @model_validator(mode="after")
    def check_window_units(self) -> "RiskConfigResolved":
        if self.window_units != "bars":
            raise ValueError("Resolved config must use window_units='bars'")
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    min_trade_notional: float = 0.0
    weight_deadband: float = 0.0
    min_rebalance_notional: float = 0.0
    min_rebalance_notional_frac: float = 0.0
    cooldown_bars: int = 0
    cooldown_override: float = 0.0
    execution_lag_bars: int = 1
    rebalance_deadband: float = 0.05
    max_weight_step: Optional[float] = None
    enable_dd_throttle: bool = False
    max_allowed_drawdown: float = 0.35
    dd_throttle_floor: float = 0.30
    cash_yield_annual: float = 0.0

    @model_validator(mode="after")
    def check_lag(self) -> "ExecutionConfig":
        if self.execution_lag_bars < 1:
            raise ValueError("execution_lag_bars must be >= 1")
        return self


class RunConfigRaw(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_name: str = Field(default_factory=lambda: "btc_hourly")
    data: DataConfig
    engine: EngineConfig
    strategy: StrategyConfigRaw
    risk: RiskConfigRaw
    execution: ExecutionConfig

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())


class RunConfigResolved(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_name: str
    data: DataConfig
    engine: EngineConfig
    strategy: StrategyConfigResolved
    risk: RiskConfigResolved
    execution: ExecutionConfig
    bar_hours: float
    annualization_factor: float
    raw: RunConfigRaw

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def to_resolved_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json(exclude={"raw"}))

    def compute_hash(self) -> str:
        return hash_config(self.to_resolved_dict())


def _hours_to_bars(hours: Optional[int], bar_hours: float, min_bars: int) -> Optional[int]:
    if hours is None:
        return None
    # Explicit rounding policy: round to nearest bar, then enforce minimum bars
    bars = int(round(hours / bar_hours))
    bars = max(min_bars, bars)
    return bars


def compile_config(raw: RunConfigRaw) -> RunConfigResolved:
    """
    Compile raw config into resolved config (bars-based windows).
    """
    bar_hours = hours_per_bar(raw.data.timeframe)
    annualization_factor = (24.0 * 365.0) / bar_hours

    # Strategy window conversion
    if raw.strategy.window_units == "hours":
        fast_bars = _hours_to_bars(raw.strategy.fast, bar_hours, min_bars=1)
        slow_bars = _hours_to_bars(raw.strategy.slow, bar_hours, min_bars=1)
        vol_window_bars = _hours_to_bars(raw.strategy.vol_window, bar_hours, min_bars=2)
        window_units = "bars"
    else:
        fast_bars = raw.strategy.fast
        slow_bars = raw.strategy.slow
        vol_window_bars = raw.strategy.vol_window
        window_units = raw.strategy.window_units

    # Enforce slow > fast invariant for all non-buy-and-hold strategies
    if raw.strategy.mode != "buy_and_hold" and fast_bars is not None and slow_bars is not None:
        slow_bars = max(slow_bars, fast_bars + 1)

    strategy_resolved = StrategyConfigResolved(
        mode=raw.strategy.mode,
        fast=fast_bars,
        slow=slow_bars,
        vol_window=vol_window_bars,
        k=raw.strategy.k,
        min_band=raw.strategy.min_band,
        weight_on=raw.strategy.weight_on,
        window_units=window_units,
        target_vol_annual=raw.strategy.target_vol_annual,
        vol_lookback=raw.strategy.vol_lookback,
        max_weight=raw.strategy.max_weight,
        adx_window=raw.strategy.adx_window,
        adx_threshold=raw.strategy.adx_threshold,
        enable_adx_filter=raw.strategy.enable_adx_filter,
        adx_entry_only=raw.strategy.adx_entry_only,
    )

    # Risk window conversion
    if raw.risk.window_units == "hours":
        risk_vol_window = _hours_to_bars(raw.risk.vol_window, bar_hours, min_bars=2)
    else:
        risk_vol_window = raw.risk.vol_window

    risk_resolved = RiskConfigResolved(
        vol_window=risk_vol_window,
        target_vol_annual=raw.risk.target_vol_annual,
        max_weight=raw.risk.max_weight,
        min_vol_floor=raw.risk.min_vol_floor,
        window_units="bars",
    )

    # Safety: enforce minimum effective risk vol window in strict mode
    if raw.engine.strict_validation and risk_resolved.vol_window < 5:
        raise ValueError(
            "risk.vol_window resolves to <5 bars; increase risk.vol_window (hours) for this timeframe."
        )
    if not raw.engine.strict_validation and risk_resolved.vol_window < 5:
        print(
            "Warning: risk.vol_window resolves to <5 bars; increase risk.vol_window (hours) for this timeframe."
        )

    return RunConfigResolved(
        run_name=raw.run_name,
        data=raw.data,
        engine=raw.engine,
        strategy=strategy_resolved,
        risk=risk_resolved,
        execution=raw.execution,
        bar_hours=bar_hours,
        annualization_factor=annualization_factor,
        raw=raw,
    )


# Backward-compatible aliases for resolved config usage
StrategyConfig = StrategyConfigResolved
RiskConfig = RiskConfigResolved
RunConfig = RunConfigResolved

def load_config_from_yaml(path: str | Path) -> RunConfigRaw:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfigRaw.model_validate(raw)


def make_run_id(run_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_name}_{ts}"


def get_git_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".", stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def write_manifest(
    run_dir: Path,
    cfg: RunConfigResolved,
    *,
    bars_start: datetime,
    bars_end: datetime,
    data_provenance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = cfg.to_resolved_dict()
    cfg_hash = cfg.compute_hash()
    manifest = {
        "config_hash": cfg_hash,
        "git_hash": get_git_hash(),
        "params": cfg_dict,
        "raw": cfg.raw.to_dict(),
        "data_provenance": data_provenance or {},
        "time_range": {"start": bars_start.isoformat(), "end": bars_end.isoformat()},
        "symbol": cfg.data.symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]

