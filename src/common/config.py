from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .hash import hash_config


class DataConfig(BaseModel):
    db_path: str
    table: str = "bars"
    symbol: str = "BTC-USD"
    start: datetime
    end: datetime
    timeframe: str = "1h"


class EngineConfig(BaseModel):
    strict_validation: bool = True
    lookback: Optional[int] = None
    initial_cash: float = 100000.0


class StrategyConfig(BaseModel):
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
    def check_windows(self) -> "StrategyConfig":
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
            if self.vol_lookback is None:
                self.vol_lookback = 20
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


class RiskConfig(BaseModel):
    vol_window: int
    target_vol_annual: Optional[float]
    max_weight: float = 1.0
    min_vol_floor: float = 1e-8
    window_units: str = "hours"

    @model_validator(mode="after")
    def check_window_units(self) -> "RiskConfig":
        if self.window_units not in ("hours", "bars"):
            raise ValueError("window_units must be 'hours' or 'bars'")
        return self


class ExecutionConfig(BaseModel):
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


class RunConfig(BaseModel):
    run_name: str = Field(default_factory=lambda: "btc_hourly")
    data: DataConfig
    engine: EngineConfig
    strategy: StrategyConfig
    risk: RiskConfig
    execution: ExecutionConfig

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())


def load_config_from_yaml(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)


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
    cfg: RunConfig,
    *,
    bars_start: datetime,
    bars_end: datetime,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = cfg.to_dict()
    cfg_hash = hash_config(cfg_dict)
    manifest = {
        "config_hash": cfg_hash,
        "git_hash": get_git_hash(),
        "params": cfg_dict,
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

