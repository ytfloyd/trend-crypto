from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import polars as pl

from .context import StrategyContext, make_strategy_context


@dataclass(frozen=True)
class StrategySignals:
    """Typed container for diagnostic signals emitted by a strategy."""

    target_weight: float
    vol_scalar: Optional[float] = None
    adx: Optional[float] = None
    ma_signal: bool = False
    adx_pass: bool = False
    in_pos: bool = False


class TargetWeightStrategy(ABC):
    @abstractmethod
    def on_bar_close(self, ctx: StrategyContext) -> float:
        """Return target portfolio weight decided at bar close."""

    def get_last_signals(self) -> StrategySignals:
        """Return diagnostic signals from the last on_bar_close call.

        Override in subclasses to expose strategy-specific diagnostics.
        """
        return StrategySignals(target_weight=0.0)

    def compute_signals_vectorized(
        self, bars: pl.DataFrame, lookback: Optional[int]
    ) -> pl.DataFrame:
        """Compute target weights and diagnostic signals for all bars at once.

        Default implementation falls back to the per-bar on_bar_close loop.
        Override for O(n) performance using Polars expressions.

        Returns a DataFrame with at least a ``target_weight`` column, and
        optionally: vol_scalar, adx, ma_signal, adx_pass, in_pos.
        """
        n = bars.height
        weights: list[float] = []
        vol_scalars: list[Optional[float]] = []
        adx_vals: list[Optional[float]] = []
        ma_signals: list[bool] = []
        adx_passes: list[bool] = []
        in_pos_flags: list[bool] = []

        for i in range(n):
            ctx = make_strategy_context(bars, i, lookback)
            w = self.on_bar_close(ctx)
            signals = self.get_last_signals()
            weights.append(w)
            vol_scalars.append(signals.vol_scalar)
            adx_vals.append(signals.adx)
            ma_signals.append(signals.ma_signal)
            adx_passes.append(signals.adx_pass)
            in_pos_flags.append(signals.in_pos)

        return pl.DataFrame(
            {
                "target_weight": weights,
                "vol_scalar": vol_scalars,
                "adx": adx_vals,
                "ma_signal": ma_signals,
                "adx_pass": adx_passes,
                "in_pos": in_pos_flags,
            }
        )


# ---------------------------------------------------------------------------
# Multi-asset portfolio strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PortfolioStrategy(Protocol):
    """Protocol for strategies that produce joint multi-asset target weights.

    Implementations receive per-symbol StrategyContexts and return a dict
    mapping symbol → target weight.  This allows cross-asset logic such as
    risk-parity, mean-variance, or pair-trading strategies.
    """

    def on_bar_close_portfolio(
        self, contexts: dict[str, StrategyContext]
    ) -> dict[str, float]:
        """Return target weights per symbol decided at bar close."""
        ...


class SingleAssetAdapter:
    """Wraps a per-symbol dict of TargetWeightStrategy instances into a PortfolioStrategy.

    Each underlying strategy sees only its own symbol's bars and produces an
    independent target weight.  This is the simplest way to run multiple
    single-asset strategies inside the PortfolioEngine.
    """

    def __init__(self, strategies: dict[str, TargetWeightStrategy]) -> None:
        self.strategies = strategies

    def on_bar_close_portfolio(
        self, contexts: dict[str, StrategyContext]
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for symbol, ctx in contexts.items():
            strategy = self.strategies.get(symbol)
            if strategy is not None:
                weights[symbol] = strategy.on_bar_close(ctx)
            else:
                weights[symbol] = 0.0
        return weights

