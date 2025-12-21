from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import polars as pl

from common.config import RiskConfig
import math


def _sigma_hourly(history: pl.DataFrame, window: int) -> Optional[float]:
    if history.height <= window:
        return None
    log_rets = history.select(pl.col("close").log().diff().alias("lr")).drop_nulls()
    if log_rets.height < window:
        return None
    return log_rets.tail(window).select(pl.col("lr").std(ddof=1)).item()


@dataclass
class VolTargeting:
    cfg: RiskConfig
    periods_per_year: float

    def scale(self, base_weight: float, history: pl.DataFrame) -> float:
        if self.cfg.target_vol_annual is None:
            return min(base_weight, self.cfg.max_weight)
        sigma = _sigma_hourly(history, self.cfg.vol_window)
        target_sigma = self.cfg.target_vol_annual
        if sigma is None or sigma <= self.cfg.min_vol_floor:
            return min(base_weight, self.cfg.max_weight)
        sigma_annual = sigma * math.sqrt(self.periods_per_year)
        scaled = base_weight * (target_sigma / sigma_annual)
        return max(0.0, min(scaled, self.cfg.max_weight))

