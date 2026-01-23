"""Tests for config compiler invariants and annualization."""
from datetime import datetime, timezone

from common.config import (
    DataConfig,
    EngineConfig,
    ExecutionConfig,
    RiskConfigRaw,
    RunConfigRaw,
    StrategyConfigRaw,
    compile_config,
)


def _base_raw_config(timeframe: str, fast_hours: int, slow_hours: int) -> RunConfigRaw:
    """Helper to build a minimal raw config for compiler tests."""
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end = datetime(2021, 12, 31, tzinfo=timezone.utc)

    return RunConfigRaw(
        run_name="test",
        data=DataConfig(
            db_path=":memory:",
            table="bars",
            symbol="BTC-USD",
            start=start,
            end=end,
            timeframe=timeframe,
        ),
        engine=EngineConfig(strict_validation=True, lookback=10, initial_cash=1000.0),
        strategy=StrategyConfigRaw(
            mode="ma_crossover_long_only",
            fast=fast_hours,
            slow=slow_hours,
            window_units="hours",
        ),
        risk=RiskConfigRaw(vol_window=20, target_vol_annual=None, max_weight=1.0, window_units="hours"),
        execution=ExecutionConfig(),
    )


def test_compiler_invariants_and_hash_divergence():
    """Validate bars conversion and annualization by timeframe; hashes must diverge."""
    raw_1h = _base_raw_config(timeframe="1h", fast_hours=20, slow_hours=240)
    cfg_1h = compile_config(raw_1h)

    assert cfg_1h.strategy.slow == 240
    assert cfg_1h.strategy.slow > cfg_1h.strategy.fast
    assert cfg_1h.annualization_factor == 8760.0

    raw_4h = _base_raw_config(timeframe="4h", fast_hours=20, slow_hours=240)
    cfg_4h = compile_config(raw_4h)

    assert cfg_4h.strategy.slow == 60
    assert cfg_4h.strategy.slow > cfg_4h.strategy.fast
    assert cfg_4h.annualization_factor == 2190.0

    assert cfg_1h.compute_hash() != cfg_4h.compute_hash()


def test_compiler_rounding_safety_daily():
    """Daily timeframe should not round fast window down to zero."""
    raw_1d = _base_raw_config(timeframe="1d", fast_hours=12, slow_hours=48)
    cfg_1d = compile_config(raw_1d)

    assert cfg_1d.strategy.fast >= 1
    assert cfg_1d.strategy.slow > cfg_1d.strategy.fast
    assert cfg_1d.annualization_factor == 365.0
