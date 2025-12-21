from datetime import datetime, timedelta, timezone

import polars as pl

from common.config import RiskConfig
from risk.risk_manager import RiskManager


def _history(closes: list[float]) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i, c in enumerate(closes):
        ts = start + timedelta(hours=i)
        rows.append(
            {"ts": ts, "symbol": "BTC-USD", "open": c, "high": c, "low": c, "close": c, "volume": 1.0}
        )
    return pl.DataFrame(rows)


def test_vol_target_scales_down_when_vol_high():
    cfg = RiskConfig(vol_window=3, target_vol_annual=0.6, max_weight=1.0, min_vol_floor=1e-8)
    rm = RiskManager(cfg, periods_per_year=8760)

    low_vol = _history([100, 100.2, 100.4, 100.5, 100.6])
    high_vol = _history([100, 110, 90, 120, 80])

    w_low = rm.apply(1.0, low_vol)
    w_high = rm.apply(1.0, high_vol)

    assert w_high <= w_low
    assert w_high <= cfg.max_weight

