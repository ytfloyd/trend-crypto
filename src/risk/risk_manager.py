from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import polars as pl

from common.config import PortfolioConfig, RiskConfig
from .vol_target import VolTargeting


@dataclass
class RiskManager:
    cfg: RiskConfig
    periods_per_year: float

    def __post_init__(self) -> None:
        self.vol_target = VolTargeting(self.cfg, self.periods_per_year)

    def apply(self, base_weight: float, history: pl.DataFrame) -> float:
        scaled = self.vol_target.scale(base_weight, history)
        return max(0.0, min(scaled, self.cfg.max_weight))


@dataclass
class PortfolioRiskManager:
    """Multi-asset risk manager with portfolio-level constraints.

    Applies risk controls in order:
    1. Per-asset vol targeting (delegated to RiskManager)
    2. Single-name concentration limit
    3. Gross leverage limit
    4. Net leverage limit
    """

    per_asset_rm: RiskManager
    portfolio_cfg: PortfolioConfig

    def apply(
        self,
        raw_weights: dict[str, float],
        histories: dict[str, pl.DataFrame],
    ) -> dict[str, float]:
        """Apply all portfolio risk constraints to raw target weights.

        Args:
            raw_weights: Symbol → raw target weight from strategy.
            histories: Symbol → historical bar data for vol targeting.

        Returns:
            Symbol → risk-adjusted target weight.
        """
        # Step 1: Per-asset vol targeting
        scaled: dict[str, float] = {}
        for sym, w in raw_weights.items():
            history = histories.get(sym)
            if history is not None:
                scaled[sym] = self.per_asset_rm.apply(w, history)
            else:
                scaled[sym] = 0.0

        # Step 2: Single-name concentration limit
        max_single = self.portfolio_cfg.max_single_name_weight
        for sym in scaled:
            if abs(scaled[sym]) > max_single:
                sign = 1.0 if scaled[sym] >= 0 else -1.0
                scaled[sym] = sign * max_single

        # Step 3: Gross leverage limit
        gross = sum(abs(v) for v in scaled.values())
        max_gross = self.portfolio_cfg.max_gross_leverage
        if gross > max_gross and gross > 0:
            factor = max_gross / gross
            scaled = {s: v * factor for s, v in scaled.items()}

        # Step 4: Net leverage limit
        net = sum(scaled.values())
        max_net = self.portfolio_cfg.max_net_leverage
        if abs(net) > max_net and abs(net) > 0:
            factor = max_net / abs(net)
            scaled = {s: v * factor for s, v in scaled.items()}

        return scaled

