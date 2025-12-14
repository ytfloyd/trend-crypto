from __future__ import annotations

from typing import Optional

from common.config import ExecutionConfig
from execution.types import Order, Side


def _within_deadband(current_weight: float, target_weight: float, deadband: float) -> bool:
    return abs(target_weight - current_weight) < deadband


def rebalance_to_target_weight(
    *,
    target_weight: float,
    cash: float,
    units: float,
    price: float,
    nav: float,
    cfg: ExecutionConfig,
    bars_since_last_trade: Optional[int],
) -> Optional[Order]:
    """
    Convert desired target weight into a single market order, respecting invariants.
    """
    if nav <= 0 or price <= 0:
        return None

    # long-only clamp
    target_weight = max(0.0, target_weight)

    current_notional = units * price
    current_weight = current_notional / nav if nav > 0 else 0.0

    if _within_deadband(current_weight, target_weight, cfg.weight_deadband):
        return None

    target_notional = target_weight * nav
    min_rebalance_notional = max(
        cfg.min_rebalance_notional, cfg.min_rebalance_notional_frac * nav
    )
    if abs(target_notional - current_notional) < min_rebalance_notional:
        return None

    target_units = target_notional / price
    delta_units = target_units - units
    if abs(delta_units) * price < cfg.min_trade_notional:
        return None

    # cooldown handling
    if cfg.cooldown_bars > 0 and bars_since_last_trade is not None:
        if bars_since_last_trade < cfg.cooldown_bars:
            if _within_deadband(current_weight, target_weight, cfg.cooldown_override):
                return None

    if delta_units > 0:
        side = Side.BUY
        fee_rate = cfg.fee_bps / 10_000
        slip = cfg.slippage_bps / 10_000
        worst_fill_price = price * (1 + slip)
        effective_cost_per_unit = worst_fill_price * (1 + fee_rate)
        max_affordable_units = (
            cash / effective_cost_per_unit if effective_cost_per_unit > 0 else 0.0
        )
        qty = min(delta_units, max_affordable_units)
    else:
        side = Side.SELL
        qty = min(abs(delta_units), units)

    if qty <= 0 or qty * price < cfg.min_trade_notional:
        return None

    return Order(side=side, qty=qty, ref_price=price)

