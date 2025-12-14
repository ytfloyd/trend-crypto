from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import polars as pl

from common.config import RiskConfig


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

    def scale(self, base_weight: float, history: pl.DataFrame) -> float:
        sigma = _sigma_hourly(history, self.cfg.vol_window)
        target_sigma_hourly = self.cfg.target_vol_annual / math.sqrt(8760)
        if sigma is None or sigma <= self.cfg.min_vol_floor:
            return min(base_weight, self.cfg.max_weight)
        scaled = base_weight * (target_sigma_hourly / sigma)
        return max(0.0, min(scaled, self.cfg.max_weight))

