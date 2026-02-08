from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import polars as pl

from common.config import StrategyConfig
from .base import StrategySignals, TargetWeightStrategy
from .context import StrategyContext


@dataclass
class BuyAndHoldStrategy(TargetWeightStrategy):
    cfg: StrategyConfig

    def on_bar_close(self, ctx: StrategyContext) -> float:
        return self.cfg.weight_on

    def get_last_signals(self) -> StrategySignals:
        return StrategySignals(target_weight=self.cfg.weight_on, in_pos=True)

    def compute_signals_vectorized(
        self, bars: pl.DataFrame, lookback: Optional[int]
    ) -> pl.DataFrame:
        n = bars.height
        w = self.cfg.weight_on
        return pl.DataFrame(
            {
                "target_weight": [w] * n,
                "vol_scalar": [None] * n,
                "adx": [None] * n,
                "ma_signal": [False] * n,
                "adx_pass": [False] * n,
                "in_pos": [True] * n,
            }
        )

