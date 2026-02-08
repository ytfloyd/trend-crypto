"""In-process metrics collection to JSONL files.

Lightweight metrics infrastructure for tracking gauges and counters
during live trading or batch processing. No external dependencies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from common.logging import get_logger

logger = get_logger("metrics")


@dataclass
class MetricPoint:
    """A single metric observation."""

    name: str
    value: float
    metric_type: str  # "gauge" or "counter"
    ts: str
    tags: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects gauge and counter metrics, flushes to JSONL.

    Args:
        output_path: Path to the JSONL output file.
        auto_flush_interval: Flush after this many points (0 = manual only).
    """

    def __init__(
        self,
        output_path: str | Path = "metrics.jsonl",
        auto_flush_interval: int = 100,
    ) -> None:
        self.output_path = Path(output_path)
        self.auto_flush_interval = auto_flush_interval
        self._buffer: list[MetricPoint] = []
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}

    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a gauge metric (last-value-wins).

        Args:
            name: Metric name.
            value: Current value.
            tags: Optional key-value tags.
        """
        self._gauges[name] = value
        point = MetricPoint(
            name=name,
            value=value,
            metric_type="gauge",
            ts=datetime.now(timezone.utc).isoformat(),
            tags=tags or {},
        )
        self._buffer.append(point)
        self._maybe_flush()

    def counter(
        self,
        name: str,
        increment: float = 1.0,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name.
            increment: Amount to increment.
            tags: Optional key-value tags.
        """
        self._counters[name] = self._counters.get(name, 0.0) + increment
        point = MetricPoint(
            name=name,
            value=self._counters[name],
            metric_type="counter",
            ts=datetime.now(timezone.utc).isoformat(),
            tags=tags or {},
        )
        self._buffer.append(point)
        self._maybe_flush()

    def get_gauge(self, name: str) -> Optional[float]:
        """Get the current value of a gauge."""
        return self._gauges.get(name)

    def get_counter(self, name: str) -> float:
        """Get the current value of a counter."""
        return self._counters.get(name, 0.0)

    def flush(self) -> int:
        """Write buffered metrics to JSONL file.

        Returns:
            Number of points flushed.
        """
        if not self._buffer:
            return 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        count = len(self._buffer)
        with open(self.output_path, "a", encoding="utf-8") as f:
            for point in self._buffer:
                line = json.dumps({
                    "name": point.name,
                    "value": point.value,
                    "type": point.metric_type,
                    "ts": point.ts,
                    "tags": point.tags,
                })
                f.write(line + "\n")
        self._buffer.clear()
        return count

    def _maybe_flush(self) -> None:
        if self.auto_flush_interval > 0 and len(self._buffer) >= self.auto_flush_interval:
            self.flush()

    def read_all(self) -> list[dict[str, Any]]:
        """Read all metrics from the output file.

        Returns:
            List of metric dicts.
        """
        if not self.output_path.exists():
            return []
        results: list[dict[str, Any]] = []
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
