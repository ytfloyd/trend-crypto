from __future__ import annotations

from dataclasses import dataclass

from common.config import StrategyConfig
from .base import TargetWeightStrategy
from .context import StrategyContext


@dataclass
class BuyAndHoldStrategy(TargetWeightStrategy):
    cfg: StrategyConfig

    def on_bar_close(self, ctx: StrategyContext) -> float:  # type: ignore[override]
        return self.cfg.weight_on

