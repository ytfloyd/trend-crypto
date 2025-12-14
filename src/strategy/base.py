from __future__ import annotations

from abc import ABC, abstractmethod

from .context import StrategyContext


class TargetWeightStrategy(ABC):
    @abstractmethod
    def on_bar_close(self, ctx: StrategyContext) -> float:
        """Return target portfolio weight decided at bar close."""

