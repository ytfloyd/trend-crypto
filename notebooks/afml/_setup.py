"""
Notebook bootstrap for AFML (Advances in Financial Machine Learning) notebooks.

Usage (first cell):
    from _setup import *

Provides:
    Data:       load_daily_bars, load_bars, filter_universe,
                compute_btc_benchmark, ANN_FACTOR
    AFML:       (all src/afml modules as they are built)
    Paths:      PROJECT_ROOT, DATA_DIR, ARTIFACTS_DIR
    Libs:       np, pd, plt  (numpy, pandas, matplotlib.pyplot)

NOTE: Ad hoc backtesting helpers (simple_backtest, compute_metrics)
      have been removed.  All backtesting must use src/backtest/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # notebooks/afml/
PROJECT_ROOT = _THIS_DIR.parents[1]                  # trend_crypto/
DATA_DIR = PROJECT_ROOT.parent / "data"              # ../data/
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "research" / "afml"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Wire up imports
# ---------------------------------------------------------------------------
_SRC_DIR = str(PROJECT_ROOT / "src")
_RESEARCH_DIR = str(PROJECT_ROOT / "scripts" / "research")
_PROJECT_ROOT_STR = str(PROJECT_ROOT)

for _p in (_SRC_DIR, _RESEARCH_DIR, _PROJECT_ROOT_STR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import matplotlib                                                   # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402

# Data loading
from common.data import (                                           # noqa: E402
    load_daily_bars,
    load_bars,
    filter_universe,
    compute_btc_benchmark,
    ANN_FACTOR,
)

# AFML library
import afml.bars                                                    # noqa: E402
import afml.labeling                                                # noqa: E402
import afml.sample_weights                                          # noqa: E402
import afml.fracdiff                                                # noqa: E402
import afml.cross_validation                                        # noqa: E402
import afml.feature_importance                                      # noqa: E402
import afml.bet_sizing                                              # noqa: E402
import afml.features                                                # noqa: E402
import afml.microstructure                                          # noqa: E402
import afml.backtest_stats                                          # noqa: E402
import afml.portfolio                                               # noqa: E402

# ---------------------------------------------------------------------------
# Matplotlib defaults (matches notebooks/alpha style)
# ---------------------------------------------------------------------------
NAVY = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GREEN = "#336633"; GRAY = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
COLORS = [NAVY, TEAL, RED, GOLD, GREEN, GRAY]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.facecolor": BG,
    "axes.edgecolor": LGRAY,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": GRAY,
    "figure.facecolor": "white",
    "figure.figsize": (12, 5),
})

# ---------------------------------------------------------------------------
print(f"[setup] Project: {PROJECT_ROOT.name}  |  DuckDB: {DATA_DIR / 'market.duckdb'}")
print(f"[setup] AFML notebooks ready — np, pd, plt, afml.bars, ...")
