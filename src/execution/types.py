from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    side: Side
    qty: float
    ref_price: float


@dataclass
class Fill:
    ts: datetime
    side: Side
    qty: float
    ref_price: float
    fill_price: float
    notional: float
    fee: float

