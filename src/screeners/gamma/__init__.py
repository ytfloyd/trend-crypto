"""Underpriced-gamma equity screener.

Daily batch: snapshot a universe of equities' option surfaces via IB,
compute IV (constant-maturity 7/30/60/90d) and realized vol (Yang-Zhang
and close-to-close), combine into a cross-sectional score, and rank.

Components:
    config    — GammaScreenerConfig: universe, horizons, weights, filters
    universe  — curated ticker lists (S&P 100 default)
    schema    — gamma_screener_daily DuckDB table
    ingest    — loop IB snapshotting into vol_surface_snaps
    signals   — per-symbol IV/RV/skew/term features from DuckDB
    score     — cross-sectional ranking
    earnings  — Finnhub calendar (optional; no-op if no API key)
"""
from __future__ import annotations

from .config import GammaScreenerConfig
from .schema import GammaScreenerSchema
from .signals import compute_features, FeatureRow
from .score import rank_universe, ScoredRow

__all__ = [
    "GammaScreenerConfig",
    "GammaScreenerSchema",
    "compute_features",
    "FeatureRow",
    "rank_universe",
    "ScoredRow",
]
