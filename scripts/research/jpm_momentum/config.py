"""
Strategy configuration dataclasses for JPM Momentum research.

Centralises all tunable parameters so that grid sweeps and runner scripts
can build configs declaratively.  Supports both crypto and ETF markets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# ---------------------------------------------------------------------------
# Signal types recognised by the signals module
# ---------------------------------------------------------------------------
SIGNAL_TYPES = ("RET", "MAC", "EMAC", "BRK", "LREG", "RADJ")

# Lookback grid (in trading days — same for both markets):
#   5d (1wk), 10d (2wk), 21d (1mo), 42d (2mo), 63d (3mo), 126d (6mo), 252d (12mo)
DEFAULT_LOOKBACKS = (5, 10, 21, 42, 63, 126, 252)

# Market-specific defaults
MARKET_DEFAULTS: dict[str, dict] = {
    "crypto": {
        "ann_factor": 365.0,
        "cost_bps": 20.0,
        "min_adv_usd": 1_000_000,
        "min_history_days": 90,
        "start": "2017-01-01",
        "benchmark": "BTC-USD",
    },
    "etf": {
        "ann_factor": 252.0,
        "cost_bps": 10.0,
        "min_adv_usd": 5_000_000,
        "min_history_days": 252,
        "start": "2006-01-01",
        "benchmark": "SPY",
    },
}


# ---------------------------------------------------------------------------
# Signal config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalConfig:
    """Parameters for a single momentum signal."""

    signal_type: str = "RET"
    lookback: int = 21
    fast_span: int | None = None   # for MAC / EMAC: fast window
    slow_span: int | None = None   # for MAC / EMAC: slow window

    def __post_init__(self) -> None:
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(
                f"Unknown signal_type {self.signal_type!r}. "
                f"Choose from {SIGNAL_TYPES}"
            )


# ---------------------------------------------------------------------------
# Position-sizing config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightConfig:
    """Position sizing parameters."""

    method: Literal["equal", "inv_vol", "risk_parity"] = "equal"
    vol_lookback: int = 42
    vol_floor: float = 0.10


# ---------------------------------------------------------------------------
# Risk-management config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    """Risk management overlay parameters."""

    vol_target: float | None = 0.20
    vol_lookback: int = 42
    max_leverage: float = 2.0

    trailing_stop_pct: float | None = None        # e.g. 0.15 for 15%
    stop_reentry_bars: int = 0                     # time-based re-entry wait

    mean_revert_window: int | None = None          # e.g. 5 (short-term return window)
    mean_revert_threshold: float = 2.0             # scale down if return > N sigma

    vol_filter_multiplier: float | None = None     # e.g. 2.0 → reduce when vol > 2x median
    vol_filter_scale: float = 0.5                  # exposure fraction in high-vol regime


# ---------------------------------------------------------------------------
# Backtest config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestConfig:
    """Full backtest configuration."""

    signal: SignalConfig = field(default_factory=SignalConfig)
    weight: WeightConfig = field(default_factory=WeightConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Market
    market: Literal["crypto", "etf"] = "crypto"

    # Absolute vs. relative momentum
    mode: Literal["absolute", "relative"] = "absolute"
    top_k: int | None = None       # for relative mode: top-K assets
    top_quantile: float = 0.20     # for relative mode: top quintile

    # Execution
    cost_bps: float = 20.0         # 10 fee + 10 slippage (crypto), 5+5 (ETF)
    cash_yield: float = 0.04       # annualised risk-free rate
    rebalance_freq: int = 1        # rebalance every N bars

    # Universe
    min_adv_usd: float = 1_000_000
    min_history_days: int = 90
    adv_window: int = 20

    # Data
    start: str = "2017-01-01"
    end: str = "2026-12-31"

    @property
    def ann_factor(self) -> float:
        """Annualisation factor for this market."""
        return MARKET_DEFAULTS[self.market]["ann_factor"]

    @property
    def benchmark_ticker(self) -> str:
        """Default benchmark for this market."""
        return MARKET_DEFAULTS[self.market]["benchmark"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ann_factor"] = self.ann_factor
        d["benchmark_ticker"] = self.benchmark_ticker
        return d

    @classmethod
    def for_etf(cls, **overrides) -> BacktestConfig:
        """Create a BacktestConfig with ETF-appropriate defaults."""
        etf_defaults = {
            "market": "etf",
            "cost_bps": 10.0,
            "min_adv_usd": 5_000_000,
            "min_history_days": 252,
            "start": "2006-01-01",
        }
        etf_defaults.update(overrides)
        return cls(**etf_defaults)
