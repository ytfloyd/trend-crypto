from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from common.config import StrategyConfig
from .base import TargetWeightStrategy
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
        if history.height < self.cfg.slow:
            return 0.0

        closes = history.select(pl.col("close"))
        fast_ma = closes.tail(self.cfg.fast).select(pl.col("close").mean()).item()
        slow_ma = closes.tail(self.cfg.slow).select(pl.col("close").mean()).item()
        if slow_ma == 0:
            return 0.0
        spread = (fast_ma - slow_ma) / slow_ma

        log_rets = history.select(pl.col("close").log().diff().alias("lr")).drop_nulls()
        sigma = (
            log_rets.tail(self.cfg.vol_window).select(pl.col("lr").std(ddof=1)).item()
            if log_rets.height >= self.cfg.vol_window
            else None
        )
        band = max(self.cfg.min_band, (self.cfg.k * sigma) if sigma is not None else self.cfg.min_band)

        if not self.long and spread > band:
            self.long = True
        elif self.long and spread < -band:
            self.long = False

        return self.cfg.weight_on if self.long else 0.0

