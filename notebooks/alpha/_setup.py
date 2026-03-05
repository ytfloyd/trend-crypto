"""
Notebook bootstrap — import this once at the top of every alpha notebook.

Usage (first cell):
    from _setup import *

Provides:
    Data:       load_daily_bars, load_bars, filter_universe,
                compute_btc_benchmark, ANN_FACTOR
    Backtest:   simple_backtest, DEFAULT_COST_BPS
    Metrics:    compute_metrics
    Overlays:   apply_vol_targeting, apply_dd_control, apply_position_limit_wide
    Paths:      PROJECT_ROOT, DATA_DIR, ARTIFACTS_DIR
    Libs:       np, pd, plt  (numpy, pandas, matplotlib.pyplot)
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

for _p in (_SRC_DIR, _RESEARCH_DIR):
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

# Backtesting
from common.backtest import simple_backtest, DEFAULT_COST_BPS       # noqa: E402

# Metrics
from common.metrics import compute_metrics                          # noqa: E402

# Risk overlays
from common.risk_overlays import (                                  # noqa: E402
    apply_vol_targeting,
    apply_dd_control,
    apply_position_limit_wide,
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
# Convenience helpers
# ---------------------------------------------------------------------------

def quick_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    label: str = "strategy",
) -> dict:
    """Run simple_backtest and return {label, metrics, equity}."""
    bt = simple_backtest(weights, returns, cost_bps=cost_bps)
    if bt.empty or len(bt) < 30:
        return {"label": label, "metrics": {}, "equity": pd.Series(dtype=float)}
    eq = bt.set_index("ts")["portfolio_equity"]
    metrics = compute_metrics(eq)
    metrics["avg_turnover"] = float(bt["turnover"].mean())
    metrics["avg_gross"] = float(bt["gross_exposure"].mean())
    return {"label": label, "metrics": metrics, "equity": eq}


def plot_equity(results: list[dict], title: str = "Equity Curves", log: bool = True):
    """Plot equity curves from a list of quick_backtest results."""
    colors = [NAVY, TEAL, RED, GOLD, GREEN, GRAY]
    fig, ax = plt.subplots()
    for i, r in enumerate(results):
        if r["equity"].empty:
            continue
        ax.plot(r["equity"].index, r["equity"].values,
                label=r["label"], color=colors[i % len(colors)],
                lw=1.8 if i == 0 else 1.2, alpha=1.0 if i == 0 else 0.7)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel("Equity (log)" if log else "Equity")
    ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    return fig, ax


def plot_drawdowns(results: list[dict], title: str = "Drawdowns"):
    """Plot drawdown curves from a list of quick_backtest results."""
    colors = [NAVY, TEAL, RED, GOLD, GREEN, GRAY]
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, r in enumerate(results):
        if r["equity"].empty:
            continue
        dd = r["equity"] / r["equity"].cummax() - 1.0
        ax.fill_between(dd.index, dd.values, 0, alpha=0.15, color=colors[i % len(colors)])
        ax.plot(dd.index, dd.values, label=r["label"],
                color=colors[i % len(colors)], lw=0.8)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9, loc="lower left", frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    return fig, ax


def metrics_table(results: list[dict]) -> pd.DataFrame:
    """Return a tidy DataFrame of metrics from quick_backtest results."""
    rows = []
    for r in results:
        m = r.get("metrics", {})
        if not m:
            continue
        row = {"strategy": r["label"]}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    fmt_cols = ["cagr", "vol", "max_dd", "hit_rate", "avg_gross"]
    for c in fmt_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    num_cols = ["sharpe", "sortino", "calmar", "skewness", "avg_turnover"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return df.set_index("strategy")


# ---------------------------------------------------------------------------
print(f"[setup] Project: {PROJECT_ROOT.name}  |  DuckDB: {DATA_DIR / 'market.duckdb'}")
print(f"[setup] Ready — np, pd, plt, load_daily_bars, simple_backtest, compute_metrics, ...")
