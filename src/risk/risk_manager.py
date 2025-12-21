from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from common.config import RiskConfig
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

