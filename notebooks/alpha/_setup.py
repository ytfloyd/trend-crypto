"""
Notebook bootstrap — import this once at the top of every alpha notebook.

Usage (first cell):
    from _setup import *

Provides:
    Data:       load_daily_bars, load_bars, filter_universe,
                compute_btc_benchmark, ANN_FACTOR
    Bayesian:   compute_bayesian_metrics, posterior_sharpe,
                sharpe_credible_interval, p_a_beats_b, beta_hit_rate,
                bayes_factor_positive_sharpe, parameter_robustness
    Paths:      PROJECT_ROOT, DATA_DIR, ARTIFACTS_DIR
    Libs:       np, pd, plt  (numpy, pandas, matplotlib.pyplot)

NOTE: Ad hoc backtesting helpers (simple_backtest, quick_backtest,
      compute_metrics, risk overlays) have been removed.
      All backtesting must use src/backtest/engine.py or
      src/backtest/portfolio_engine.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # notebooks/alpha/
PROJECT_ROOT = _THIS_DIR.parents[1]                  # trend_crypto/
DATA_DIR = PROJECT_ROOT.parent / "data"              # ../data/
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Wire up imports so research utilities are available
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

# Bayesian evaluation
from common.bayesian import (                                       # noqa: E402
    compute_bayesian_metrics,
    posterior_sharpe,
    sharpe_credible_interval,
    p_a_beats_b,
    beta_hit_rate,
    bayes_factor_positive_sharpe,
    parameter_robustness,
)

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
NAVY = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GREEN = "#336633"; GRAY = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"

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
print(f"[setup] Ready — np, pd, plt, load_daily_bars, ANN_FACTOR, ...")
