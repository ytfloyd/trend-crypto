"""Persistence format for the volatility book OHLCV tool.

A *bundle* is an ordered collection of ``OhlcvSeries`` keyed by
``(symbol, expiry, bar_size)``. It is what the CLI writes to
``data/volbook/bundle.json`` and what the canvas writer inlines into
the generated ``.canvas.tsx`` so the React app has everything it needs
without making any network calls.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar. Timestamp is ISO-8601 UTC."""

    t: str
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class OhlcvSeries:
    """One (contract, bar_size) slice of history.

    ``key`` is the upsert primary key inside a bundle. ``fetched_at`` is
    stamped by the client when a series is produced so the canvas can
    show staleness.
    """

    symbol: str
    expiry: str
    exchange: str
    bar_size: str
    duration: str
    what_to_show: str
    bars: list[Bar] = field(default_factory=list)
    fetched_at: str = ""
    currency: str = "USD"
    # Optional TA-Lib indicator snapshot keyed by category → list of
    # indicator records. Populated by ``volbook.indicators`` and
    # consumed by the canvas; plain dicts so it serialises as JSON.
    indicators: dict[str, list[dict]] = field(default_factory=dict)
    # Optional ranked trade-setup list produced by ``volbook.signals``
    # — each entry is a dict with entry/stop/target/rr/confidence/etc.
    setups: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.symbol}-{self.expiry}-{self.bar_size}"

    @property
    def contract_key(self) -> str:
        return f"{self.symbol}-{self.expiry}"

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "OhlcvSeries":
        bars = [Bar(**b) for b in d.get("bars", [])]
        return cls(
            symbol=d["symbol"],
            expiry=d["expiry"],
            exchange=d["exchange"],
            bar_size=d["bar_size"],
            duration=d["duration"],
            what_to_show=d.get("what_to_show", "TRADES"),
            bars=bars,
            fetched_at=d.get("fetched_at", ""),
            currency=d.get("currency", "USD"),
            indicators=d.get("indicators", {}) or {},
            setups=d.get("setups", []) or [],
        )


@dataclass
class OhlcvBundle:
    """Aggregate of all series available to the canvas.

    The canvas writer treats this as the single source of truth: every
    contract / timeframe the dropdown can select is in ``series``.
    """

    series: list[OhlcvSeries] = field(default_factory=list)
    generated_at: str = ""

    def upsert(self, series: OhlcvSeries) -> None:
        """Replace an existing series with matching ``key`` or append."""
        for i, existing in enumerate(self.series):
            if existing.key == series.key:
                self.series[i] = series
                return
        self.series.append(series)

    def extend(self, many: Iterable[OhlcvSeries]) -> None:
        for s in many:
            self.upsert(s)

    def keys(self) -> Sequence[str]:
        return [s.key for s in self.series]

    def to_json_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "series": [s.to_json_dict() for s in self.series],
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "OhlcvBundle":
        return cls(
            generated_at=d.get("generated_at", ""),
            series=[OhlcvSeries.from_json_dict(s) for s in d.get("series", [])],
        )

    def save(self, path: str | Path) -> Path:
        """Serialize to JSON. Stamps ``generated_at`` to now (UTC)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        p.write_text(json.dumps(self.to_json_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "OhlcvBundle":
        """Load from JSON, returning an empty bundle if the file is missing."""
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_json_dict(json.loads(p.read_text()))
