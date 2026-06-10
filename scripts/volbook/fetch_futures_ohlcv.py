#!/usr/bin/env python3
"""Thin CLI shim for the volbook IBKR OHLCV fetcher.

See ``src/volbook/README.md`` for flags and IB prerequisites.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from volbook.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
