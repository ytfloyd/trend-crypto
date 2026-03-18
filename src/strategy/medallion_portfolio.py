"""
Medallion Lite — PortfolioStrategy adapter.

Wraps MedallionSignalService into the PortfolioStrategy protocol
so it can be used directly with the existing LiveRunner + OMS.

Two modes of operation:

  1. Embedded: SignalService runs inline, computing signals each cycle.
     Good for paper trading and single-process deployment.

  2. External: Reads pre-computed target_weights from signal_output.json.
     Good when the signal service runs as a separate process/cron.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from strategy.base import PortfolioStrategy  # noqa: F401 (protocol reference)
from strategy.context import StrategyContext

logger = logging.getLogger("medallion_portfolio")


class MedallionPortfolioAdapter:
    """Reads target weights from a signal file and exposes them
    as a PortfolioStrategy for the LiveRunner.

    The signal file is produced by MedallionSignalService.run_cycle()
    and contains the full target_weights dict.

    This is the recommended integration path when the signal service
    runs as a separate process (cron, scheduler, etc.).
    """

    def __init__(self, signal_file: str = "signal_output.json") -> None:
        self.signal_file = Path(signal_file)
        self._last_weights: dict[str, float] = {}
        self._last_ts: str = ""

    def on_bar_close_portfolio(
        self, contexts: dict[str, StrategyContext],
    ) -> dict[str, float]:
        """Read latest weights from the signal file.

        If the file hasn't changed since last read, returns cached weights.
        If the file is missing or stale, returns zero weights (flat).
        """
        weights = self._read_signal_file()

        result: dict[str, float] = {}
        for symbol in contexts:
            result[symbol] = weights.get(symbol, 0.0)
        return result

    def _read_signal_file(self) -> dict[str, float]:
        if not self.signal_file.exists():
            logger.warning("Signal file not found: %s", self.signal_file)
            return {}

        try:
            data = json.loads(self.signal_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read signal file: %s", e)
            return self._last_weights

        ts = data.get("ts", "")
        if ts == self._last_ts:
            return self._last_weights

        if data.get("stale", False):
            logger.warning("Signal file marked as stale (data freshness issue)")

        weights = data.get("target_weights", {})
        self._last_weights = weights
        self._last_ts = ts

        logger.info(
            "Loaded weights: %d symbols, %.1f%% exposure, regime=%.2f",
            sum(1 for v in weights.values() if v > 0),
            sum(weights.values()) * 100,
            data.get("regime_score", 0),
        )
        return weights


class MedallionEmbeddedAdapter:
    """Runs the signal service inline within LiveRunner.

    Suitable for paper trading where everything runs in one process.
    Requires the MedallionSignalService to be available.
    """

    def __init__(self, signal_service) -> None:
        """
        Parameters
        ----------
        signal_service : MedallionSignalService
            An initialised signal service instance.
        """
        self._service = signal_service
        self._last_weights: dict[str, float] = {}

    def on_bar_close_portfolio(
        self, contexts: dict[str, StrategyContext],
    ) -> dict[str, float]:
        output = self._service.run_cycle()
        weights = output.target_weights

        result: dict[str, float] = {}
        for symbol in contexts:
            result[symbol] = weights.get(symbol, 0.0)
        return result
