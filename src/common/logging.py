"""Structured logging for trend-crypto-backtest."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_dict.update(record.extra_data)
        if record.exc_info and record.exc_info[0] is not None:
            log_dict["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_dict)


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure root logger for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, emit JSON lines; otherwise human-readable.
    """
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under ``trend_crypto.<name>``."""
    return logging.getLogger(f"trend_crypto.{name}")
