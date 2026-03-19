"""Medallion Lite V3 — PortfolioStrategy implementation.

Implements the V3 Deep Signal tiered (core/satellite) strategy conforming
to the PortfolioStrategy protocol so it runs through the hardened
PortfolioEngine.

Architecture
------------
Signal generation (regime scores, factor composites) is precomputed
externally and injected at construction time — the standard quant pattern
of separating alpha research from execution.  The strategy owns:

  * Entry / exit logic (thresholds, trailing stops, max hold, decay)
  * Tiered portfolio construction (core + satellite slots)
  * Inverse-vol weighting and regime-scaled gross exposure
  * All stateful position tracking

The PortfolioEngine handles:

  * Model B execution timing (decide close[t], execute open[t+1])
  * Transaction cost + slippage accounting
  * Deadband, execution lag, drawdown throttle
  * Gross / single-name leverage limits
  * Correct w_held drift (no phantom turnover)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from strategy.context import StrategyContext


@dataclass
class MedallionLiteV3Config:
    core_slots: int = 10
    satellite_slots: int = 25
    core_capital_share: float = 0.60
    entry_threshold: float = 0.65
    satellite_entry_threshold: float = 0.55
    exit_score_threshold: float = 0.40
    regime_entry_min: float = 0.45
    regime_exit_min: float = 0.15
    regime_floor: float = 0.20
    core_max_weight: float = 0.10
    sat_max_weight: float = 0.05
    core_trailing_stop: float = 0.15
    sat_trailing_stop: float = 0.12
    core_max_hold_hours: int = 336
    sat_max_hold_hours: int = 168
    rebalance_every_hours: int = 24
    score_decay_per_day: float = 0.02
    vol_lookback: int = 168


@dataclass
class _Holding:
    tier: str
    entry_ts: datetime
    hours_held: int = 0
    cum_ret: float = 0.0
    peak_cum: float = 0.0
    entry_score: float = 0.0


class MedallionLiteV3Strategy:
    """PortfolioStrategy for Medallion Lite V3 — runs through PortfolioEngine.

    Parameters
    ----------
    regime_scores : pd.Series
        Precomputed ensemble regime score ∈ [0, 1], indexed by timestamp.
    composite_scores : pd.DataFrame
        Precomputed EMA-smoothed composite factor scores (timestamp × symbol).
    cfg : MedallionLiteV3Config
        Strategy configuration.
    """

    def __init__(
        self,
        regime_scores: pd.Series,
        composite_scores: pd.DataFrame,
        cfg: Optional[MedallionLiteV3Config] = None,
    ) -> None:
        self.cfg = cfg or MedallionLiteV3Config()
        self._regime = regime_scores
        self._composite = composite_scores
        self.holdings: dict[str, _Holding] = {}
        self._hour_counter: int = 0

    def on_bar_close_portfolio(
        self, contexts: dict[str, StrategyContext]
    ) -> dict[str, float]:
        """Return target weights per symbol for this bar."""
        self._hour_counter += 1
        cfg = self.cfg
        symbols = sorted(contexts.keys())

        # Use the decision timestamp from any context (they all share the bar)
        dt = next(iter(contexts.values())).decision_ts

        # ── Look up precomputed regime score ─────────────────────
        regime_score = self._get_regime(dt)

        # ── Look up precomputed smoothed composite scores ────────
        scores: dict[str, float] = {}
        for sym in symbols:
            scores[sym] = self._get_score(dt, sym)

        # ── Extract per-bar returns for stop tracking ────────────
        current_ret: dict[str, float] = {}
        for sym, ctx in contexts.items():
            h = ctx.history
            if h.height < 2:
                current_ret[sym] = 0.0
                continue
            c1 = float(h[-1, "close"])
            c0 = float(h[-2, "close"])
            current_ret[sym] = (c1 / c0 - 1) if c0 > 0 else 0.0

        # ── Update existing holdings ─────────────────────────────
        for sym in list(self.holdings.keys()):
            hld = self.holdings[sym]
            hld.hours_held += 1
            ret = current_ret.get(sym, 0.0)
            hld.cum_ret = (1 + hld.cum_ret) * (1 + ret) - 1
            hld.peak_cum = max(hld.peak_cum, hld.cum_ret)

            t_stop = cfg.core_trailing_stop if hld.tier == "core" else cfg.sat_trailing_stop
            t_max_hold = cfg.core_max_hold_hours if hld.tier == "core" else cfg.sat_max_hold_hours

            if regime_score < cfg.regime_exit_min:
                del self.holdings[sym]
                continue

            dd = hld.peak_cum - hld.cum_ret
            if hld.peak_cum > 0.02 and dd > t_stop:
                del self.holdings[sym]
                continue

            if hld.hours_held >= t_max_hold:
                del self.holdings[sym]
                continue

            if self._hour_counter % cfg.rebalance_every_hours == 0:
                sc = scores.get(sym, 0.5)
                decay = (hld.hours_held / 24) * cfg.score_decay_per_day
                if sc - decay < cfg.exit_score_threshold:
                    del self.holdings[sym]
                    continue

        # ── Entry logic (at rebalance points) ────────────────────
        if (
            self._hour_counter % cfg.rebalance_every_hours == 0
            and regime_score >= cfg.regime_entry_min
        ):
            existing = set(self.holdings.keys())
            ranked = sorted(
                [(sym, scores.get(sym, 0)) for sym in symbols if sym not in existing],
                key=lambda x: -x[1],
            )

            n_core = sum(1 for h in self.holdings.values() if h.tier == "core")
            n_sat = sum(1 for h in self.holdings.values() if h.tier == "satellite")

            for sym, sc in ranked:
                if n_core >= cfg.core_slots:
                    break
                if sc > cfg.entry_threshold:
                    self.holdings[sym] = _Holding(
                        tier="core", entry_ts=dt, entry_score=sc,
                    )
                    n_core += 1

            existing = set(self.holdings.keys())
            for sym, sc in ranked:
                if sym in existing:
                    continue
                if n_sat >= cfg.satellite_slots:
                    break
                if sc > cfg.satellite_entry_threshold:
                    self.holdings[sym] = _Holding(
                        tier="satellite", entry_ts=dt, entry_score=sc,
                    )
                    n_sat += 1

        # ── Compute target weights ───────────────────────────────
        if not self.holdings:
            return {sym: 0.0 for sym in symbols}

        regime_scale = (
            max(regime_score, cfg.regime_floor)
            if regime_score >= cfg.regime_exit_min
            else 0.0
        )

        core_syms = [s for s, h in self.holdings.items() if h.tier == "core"]
        sat_syms = [s for s, h in self.holdings.items() if h.tier == "satellite"]

        core_share = cfg.core_capital_share if core_syms else 0.0
        sat_share = 1.0 - core_share if core_syms else 1.0

        w_core = self._tier_weights(contexts, core_syms, core_share, cfg.core_max_weight)
        w_sat = self._tier_weights(contexts, sat_syms, sat_share, cfg.sat_max_weight)

        combined = {**w_core, **w_sat}
        total = sum(combined.values())
        if total > 0:
            combined = {s: v / total for s, v in combined.items()}
        combined = {s: v * regime_scale for s, v in combined.items()}

        return {sym: combined.get(sym, 0.0) for sym in symbols}

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_regime(self, dt: datetime) -> float:
        """Look up regime score, handling timezone and nearest match."""
        ts = pd.Timestamp(dt).tz_localize(None)
        if ts in self._regime.index:
            return float(self._regime.loc[ts])
        idx = self._regime.index.get_indexer([ts], method="ffill")
        if idx[0] >= 0:
            return float(self._regime.iloc[idx[0]])
        return 0.0

    def _get_score(self, dt: datetime, sym: str) -> float:
        """Look up composite score for a symbol at a timestamp."""
        if sym not in self._composite.columns:
            return 0.5
        ts = pd.Timestamp(dt).tz_localize(None)
        if ts in self._composite.index:
            val = self._composite.loc[ts, sym]
            return float(val) if not pd.isna(val) else 0.5
        idx = self._composite.index.get_indexer([ts], method="ffill")
        if idx[0] >= 0:
            val = self._composite.iloc[idx[0]][sym]
            return float(val) if not pd.isna(val) else 0.5
        return 0.5

    def _tier_weights(
        self,
        contexts: dict[str, StrategyContext],
        syms: list[str],
        target_share: float,
        max_w: float,
    ) -> dict[str, float]:
        """Inverse-vol weighting within a tier, capped at max_w."""
        if not syms:
            return {}

        vols: dict[str, float] = {}
        lb = self.cfg.vol_lookback
        for sym in syms:
            ctx = contexts.get(sym)
            if ctx is not None and ctx.history.height >= lb // 4:
                closes = ctx.history["close"].to_numpy().astype(float)
                rets = np.diff(np.log(np.clip(closes, 1e-10, None)))
                recent = rets[-min(len(rets), lb):]
                vol = float(np.std(recent)) * np.sqrt(8760) if len(recent) > 10 else 1.0
                vols[sym] = max(vol, 0.1)
            else:
                vols[sym] = 1.0

        raw = {s: 1.0 / vols[s] for s in syms}
        total = sum(raw.values())
        if total > 0:
            raw = {s: v / total * target_share for s, v in raw.items()}
        for s in syms:
            raw[s] = min(raw[s], max_w)
        return raw
