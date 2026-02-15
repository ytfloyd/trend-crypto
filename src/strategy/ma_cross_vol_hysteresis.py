from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from common.config import StrategyConfig
from .base import StrategySignals, TargetWeightStrategy
from .context import StrategyContext


@dataclass
class MACrossVolHysteresis(TargetWeightStrategy):
    """
    MA cross with volatility-adjusted hysteresis band.
    """

    cfg: StrategyConfig
    long: bool = False

    def on_bar_close(self, ctx: StrategyContext) -> float:
        history = ctx.history
        slow = self.cfg.slow or 0
        fast = self.cfg.fast or 0
        vol_window = self.cfg.vol_window or 0
        if history.height < slow:
            return 0.0

        closes = history.select(pl.col("close"))
        fast_ma = float(closes.tail(fast).select(pl.col("close").mean()).item())
        slow_ma = float(closes.tail(slow).select(pl.col("close").mean()).item())
        if slow_ma == 0:
            return 0.0
        spread = (fast_ma - slow_ma) / slow_ma

        log_rets = history.select(pl.col("close").log().diff().alias("lr")).drop_nulls()
        sigma = (
            float(log_rets.tail(vol_window).select(pl.col("lr").std(ddof=1)).item())
            if log_rets.height >= vol_window
            else None
        )
        k = self.cfg.k if self.cfg.k is not None else 0.0
        min_band = self.cfg.min_band if self.cfg.min_band is not None else 0.0
        band = max(min_band, (k * sigma) if sigma is not None else min_band)

        if not self.long and spread > band:
            self.long = True
        elif self.long and spread < -band:
            self.long = False

        return self.cfg.weight_on if self.long else 0.0

    def get_last_signals(self) -> StrategySignals:
        return StrategySignals(
            target_weight=self.cfg.weight_on if self.long else 0.0,
            in_pos=self.long,
        )

