"""Unified core stack for trend_crypto.

Single home for the load -> signal -> risk -> backtest -> report chain shared by
production (`src/`) and research (`scripts/research/`). See
docs/RESEARCH_PIPELINE_REORGANIZATION.md for the consolidation plan.

First migrated module: `core.metrics` (the canonical equity-curve metrics, moved
here from `scripts/research/common/metrics.py`, which now re-exports from here).
"""
