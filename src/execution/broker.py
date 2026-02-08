"""Abstract broker interface for live trading.

Defines the contract that all broker implementations must follow,
plus order/fill dataclasses for the live trading path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class BrokerOrder:
    """Order submitted to a broker."""

    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: Optional[float] = None  # None = market order


@dataclass(frozen=True)
class BrokerFill:
    """Fill received from a broker."""

    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    fill_price: float
    fee: float
    ts: datetime


@dataclass
class Position:
    """Current position in a single asset."""

    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0


class BrokerInterface(ABC):
    """Abstract broker interface.

    All broker implementations (paper, Coinbase, etc.) must implement
    these methods to be used with the OMS and LiveRunner.
    """

    @abstractmethod
    def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order. Returns order_id."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Query the status of an order."""
        ...

    @abstractmethod
    def get_fills(self, order_id: str) -> list[BrokerFill]:
        """Get fills for an order."""
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Get all current positions."""
        ...

    @abstractmethod
    def get_balances(self) -> dict[str, float]:
        """Get account balances (e.g. {"USD": 50000.0, "BTC": 1.5})."""
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        ...
