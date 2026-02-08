from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List

import polars as pl

from execution.types import Fill, Side


@dataclass
class Portfolio:
    """
    Simple single-asset cash and units portfolio with mark-to-market history.
    """

    cash: float
    position_units: float = 0.0
    equity_history: List[dict[str, Any]] = field(default_factory=list)
    position_history: List[dict[str, Any]] = field(default_factory=list)
    trade_history: List[dict[str, Any]] = field(default_factory=list)

    def nav(self, price: float) -> float:
        return self.cash + self.position_units * price

    def apply_fill(self, fill: Fill, reason: str = "rebalance") -> None:
        """
        Apply a simulated fill to portfolio state and record the trade.
        """
        if fill.side == Side.BUY:
            total_cost = fill.notional + fill.fee
            self.cash -= total_cost
            self.position_units += fill.qty
        else:
            proceeds = fill.notional - fill.fee
            self.cash += proceeds
            self.position_units -= fill.qty

        self.trade_history.append(
            {
                "ts": fill.ts,
                "side": fill.side.value,
                "qty": fill.qty,
                "ref_price": fill.ref_price,
                "fill_price": fill.fill_price,
                "notional": fill.notional,
                "fee": fill.fee,
                "cash_after": self.cash,
                "pos_after": self.position_units,
                "reason": reason,
            }
        )

    def mark_to_market(self, ts: datetime, price: float) -> None:
        nav = self.nav(price)
        self.equity_history.append({"ts": ts, "nav": nav})
        self.position_history.append({"ts": ts, "position_units": self.position_units})

    def to_frames(self) -> dict[str, pl.DataFrame]:
        return {
            "equity": pl.DataFrame(self.equity_history) if self.equity_history else pl.DataFrame(),
            "positions": pl.DataFrame(self.position_history)
            if self.position_history
            else pl.DataFrame(),
            "trades": pl.DataFrame(self.trade_history) if self.trade_history else pl.DataFrame(),
        }

