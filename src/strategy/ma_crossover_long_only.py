from __future__ import annotations

import math
from typing import Optional

import polars as pl

from strategy.base import TargetWeightStrategy
from strategy.context import StrategyContext


class MACrossoverLongOnlyStrategy(TargetWeightStrategy):
    def __init__(
        self,
        fast: int,
        slow: int,
        weight_on: float = 1.0,
        target_vol_annual: Optional[float] = None,
        vol_lookback: int = 20,
        max_weight: float = 1.0,
        enable_adx_filter: bool = True,
        adx_window: int = 14,
        adx_threshold: float = 20.0,
        adx_entry_only: bool = False,
    ) -> None:
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("MA Crossover requires 0 < fast < slow")
        self.fast = fast
        self.slow = slow
        self.weight_on = weight_on
        self.target_vol_annual = target_vol_annual
        self.vol_lookback = vol_lookback
        self.max_weight = max_weight
        self.enable_adx_filter = enable_adx_filter
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold
        self.last_vol_scalar: float = 0.0
        self.last_adx: float = 0.0
        self.last_ma_signal: bool = False
        self.last_adx_pass: bool = False
        self.adx_entry_only = adx_entry_only
        self._in_pos: bool = False

    def _vol_scalar(self, closes: pl.Series) -> float:
        if self.target_vol_annual is None or self.target_vol_annual <= 0:
            return 1.0
        n = closes.len()
        if n <= self.vol_lookback:
            return 0.0
        window = closes.slice(n - self.vol_lookback - 1, self.vol_lookback + 1)
        rets = window.pct_change().drop_nulls()
        if rets.is_empty():
            return 0.0
        sigma = rets.std()
        if sigma is None or sigma <= 0:
            return 0.0
        sigma_ann = sigma * math.sqrt(365)
        if sigma_ann <= 0:
            return 0.0
        return max(0.0, min(self.target_vol_annual / sigma_ann, 1.0, self.max_weight))

    def _adx(self, high: pl.Series, low: pl.Series, close: pl.Series) -> float:
        w = self.adx_window
        n = close.len()
        if n <= w + 1:
            return 0.0
        h = high.slice(n - (w + 1), w + 1).to_list()
        low_list = low.slice(n - (w + 1), w + 1).to_list()
        c = close.slice(n - (w + 1), w + 1).to_list()
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        for i in range(1, len(h)):
            up_move = h[i] - h[i - 1]
            down_move = low_list[i - 1] - low_list[i]
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
            tr = max(h[i] - low_list[i], abs(h[i] - c[i - 1]), abs(low_list[i] - c[i - 1]))
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
            tr_list.append(tr)
        if len(tr_list) < w:
            return 0.0

        def rma(values, window):
            prev = sum(values[:window])
            rma_vals = [prev]
            for v in values[window:]:
                prev = prev - (prev / window) + v
                rma_vals.append(prev)
            return rma_vals[-1] / window

        tr_rma = rma(tr_list, w)
        plus_rma = rma(plus_dm_list, w)
        minus_rma = rma(minus_dm_list, w)
        if tr_rma <= 0:
            return 0.0
        plus_di = 100 * (plus_rma / tr_rma)
        minus_di = 100 * (minus_rma / tr_rma)
        denom = plus_di + minus_di
        if denom == 0:
            return 0.0
        dx = 100 * abs(plus_di - minus_di) / denom
        adx = rma([dx] * w, w)  # seed with constant dx for the window
        return float(adx)

    def on_bar_close(self, ctx: StrategyContext) -> float:
        hist = ctx.history
        closes = hist["close"]
        n = closes.len()
        if n < self.slow:
            return 0.0
        fast_ma = closes.slice(n - self.fast, self.fast).mean()
        slow_ma = closes.slice(n - self.slow, self.slow).mean()
        if fast_ma is None or slow_ma is None:
            return 0.0
        base_signal = fast_ma > slow_ma
        vol_scalar = self._vol_scalar(closes)
        self.last_vol_scalar = vol_scalar
        self.last_ma_signal = base_signal
        adx_pass = True
        adx_val = 0.0
        if self.enable_adx_filter:
            adx_val = self._adx(hist["high"], hist["low"], closes)
            adx_pass = adx_val >= self.adx_threshold
        self.last_adx = adx_val
        self.last_adx_pass = adx_pass

        # entry-only gating
        if not base_signal:
            self._in_pos = False
            return 0.0

        entry_allowed = True
        if self.adx_entry_only and not self._in_pos:
            entry_allowed = adx_pass
        elif self.enable_adx_filter and not self.adx_entry_only:
            entry_allowed = adx_pass

        if base_signal and entry_allowed:
            self._in_pos = True

        if not self._in_pos:
            return 0.0

        base = self.max_weight if base_signal else 0.0
        weight = base * vol_scalar
        weight = min(weight, self.max_weight)
        return max(0.0, weight)

