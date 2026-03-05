"""
Forward Simulation Engine
=========================

Block-bootstrap Monte Carlo projections for strategy equity curves,
post-drawdown conditional simulation, historical analogue extraction,
and conditional entry analysis.

All simulations use the strategy's own historical daily returns —
no distributional assumptions.  Block resampling preserves the
autocorrelation and volatility clustering present in the actual track
record.

Usage (as library)::

    from forward_simulation import (
        block_bootstrap_paths,
        post_drawdown_bootstrap_paths,
        conditional_entry_returns,
        historical_analogues,
        fan_chart_summary,
        terminal_wealth_table,
    )

Usage (standalone — writes CSVs to artifacts/)::

    python -m scripts.research.alpha_lab.forward_simulation
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from common.data import load_daily_bars, ANN_FACTOR
from common.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    n_paths: int = 10_000
    horizon_days: int = 365 * 3       # default 3-year projection
    block_size: int = 21              # 21-day blocks preserve monthly autocorrelation
    seed: int = 42
    initial_capital: float = 1_000_000.0
    percentiles: tuple = (5, 25, 50, 75, 95)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_paths(
    daily_returns: pd.Series,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """Generate *n_paths* simulated equity paths via circular block bootstrap.

    Parameters
    ----------
    daily_returns : pd.Series
        Historical daily arithmetic returns of the strategy.
    cfg : SimConfig
        Simulation parameters.

    Returns
    -------
    np.ndarray, shape (n_paths, horizon_days)
        Each row is a simulated equity path (cumulative returns starting at 1.0).
    """
    if cfg is None:
        cfg = SimConfig()

    rng = np.random.RandomState(cfg.seed)
    rets = daily_returns.dropna().values
    n = len(rets)
    if n < cfg.block_size:
        raise ValueError(f"Need at least {cfg.block_size} returns, got {n}")

    paths = np.empty((cfg.n_paths, cfg.horizon_days))

    n_blocks = int(np.ceil(cfg.horizon_days / cfg.block_size))

    for i in range(cfg.n_paths):
        starts = rng.randint(0, n, size=n_blocks)
        sampled = np.concatenate([
            np.take(rets, range(s, s + cfg.block_size), mode="wrap")
            for s in starts
        ])[:cfg.horizon_days]
        paths[i] = np.cumprod(1.0 + sampled)

    return paths


# ---------------------------------------------------------------------------
# Post-drawdown conditional bootstrap
# ---------------------------------------------------------------------------

def post_drawdown_bootstrap_paths(
    daily_returns: pd.Series,
    strategy_equity: pd.Series,
    current_dd: float = -0.29,
    dd_band: float = 0.10,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """Block bootstrap conditioned on entering at a comparable drawdown level.

    Finds all historical dates where the strategy drawdown was within
    *dd_band* of *current_dd*, and samples blocks exclusively from those
    dates.  This directly models: "historically, when we entered at a
    similar drawdown to today, here is the distribution of returns that
    followed."

    Parameters
    ----------
    daily_returns : pd.Series
        Historical daily arithmetic returns of the strategy.
    strategy_equity : pd.Series
        Equity curve (used to compute drawdown at each date).
    current_dd : float
        Current drawdown from peak (e.g., -0.29 for -29%).
    dd_band : float
        Half-width of the drawdown window. Dates where drawdown is in
        [current_dd - dd_band, current_dd + dd_band] are eligible.
    cfg : SimConfig

    Returns
    -------
    np.ndarray, shape (n_paths, horizon_days)
    """
    if cfg is None:
        cfg = SimConfig()

    eq = strategy_equity.copy()
    eq.index = pd.to_datetime(eq.index)
    dd = eq / eq.cummax() - 1.0

    dd_lo = current_dd - dd_band
    dd_hi = current_dd + dd_band
    eligible_mask = (dd >= dd_lo) & (dd <= dd_hi)
    eligible_dates = dd[eligible_mask].index

    common = daily_returns.index.intersection(eligible_dates)
    rets_full = daily_returns.dropna()

    eligible_indices = np.array([
        rets_full.index.get_loc(d)
        for d in common
        if d in rets_full.index
    ])

    if len(eligible_indices) < cfg.block_size:
        return block_bootstrap_paths(daily_returns, cfg)

    rng = np.random.RandomState(cfg.seed + 1)
    vals = rets_full.values
    n = len(vals)
    n_blocks = int(np.ceil(cfg.horizon_days / cfg.block_size))
    paths = np.empty((cfg.n_paths, cfg.horizon_days))

    for i in range(cfg.n_paths):
        starts = eligible_indices[rng.randint(0, len(eligible_indices), size=n_blocks)]
        sampled = np.concatenate([
            np.take(vals, range(s, s + cfg.block_size), mode="wrap")
            for s in starts
        ])[:cfg.horizon_days]
        paths[i] = np.cumprod(1.0 + sampled)

    return paths


# ---------------------------------------------------------------------------
# Historical analogues
# ---------------------------------------------------------------------------

ANALOGUE_ENTRIES = {
    "Post-2018 Bear (Jan 2019)": "2019-01-01",
    "Post-COVID Crash (Apr 2020)": "2020-04-01",
    "Post-May 2021 Crash (Aug 2021)": "2021-08-01",
    "Post-FTX / 2022 Bear (Jan 2023)": "2023-01-01",
}


def historical_analogues(
    strategy_equity: pd.Series,
    horizon_days: int = 365 * 3,
) -> dict[str, pd.Series]:
    """Extract actual forward equity paths from historical analogue entry points.

    Returns dict mapping label -> normalised equity path (starts at 1.0)
    trimmed to *horizon_days*.
    """
    results = {}
    eq = strategy_equity.copy()
    eq.index = pd.to_datetime(eq.index)

    for label, entry_date in ANALOGUE_ENTRIES.items():
        entry = pd.Timestamp(entry_date)
        mask = eq.index >= entry
        if mask.sum() < 30:
            continue
        path = eq[mask].iloc[:horizon_days]
        path = path / path.iloc[0]
        results[label] = path

    return results


# ---------------------------------------------------------------------------
# Conditional entry analysis
# ---------------------------------------------------------------------------

def conditional_entry_returns(
    strategy_equity: pd.Series,
    horizons_days: tuple[int, ...] = (180, 365, 365 * 2, 365 * 3),
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute forward returns conditioned on entry drawdown quintile.

    For every day in the backtest, compute the current drawdown from
    peak, bucket into *n_quantiles* bins, and measure the median forward
    cumulative return and annualised Sharpe at each horizon.

    pd.qcut assigns Q1 to the *lowest* (most negative) drawdown values
    and Q5 to the *highest* (shallowest, near peak).

    Returns
    -------
    pd.DataFrame
        Indexed by drawdown quantile label, columns are horizon labels
        (return and Sharpe), values are medians across days in that quintile.
    """
    eq = strategy_equity.copy()
    eq.index = pd.to_datetime(eq.index)
    daily_ret = eq.pct_change()

    dd = eq / eq.cummax() - 1.0

    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    try:
        dd_q = pd.qcut(dd, n_quantiles, labels=labels, duplicates="drop")
    except ValueError:
        dd_q = pd.cut(dd, n_quantiles, labels=labels[:n_quantiles], duplicates="drop")

    results = {}
    horizon_labels = []
    sharpe_labels = []
    for h in horizons_days:
        if h < 365:
            lbl = f"{h // 30}m"
        else:
            lbl = f"{h // 365}y"
        horizon_labels.append(lbl)
        sharpe_lbl = f"{lbl} Sharpe"
        sharpe_labels.append(sharpe_lbl)

        fwd = eq.shift(-h) / eq - 1.0
        results[lbl] = fwd

        fwd_sharpe = pd.Series(np.nan, index=eq.index)
        for idx_pos in range(len(eq) - h):
            window = daily_ret.iloc[idx_pos + 1 : idx_pos + 1 + h]
            window = window.dropna()
            if len(window) > 20:
                mu = window.mean()
                sigma = window.std()
                if sigma > 1e-12:
                    fwd_sharpe.iloc[idx_pos] = (mu / sigma) * np.sqrt(ANN_FACTOR)
        results[sharpe_lbl] = fwd_sharpe

    fwd_df = pd.DataFrame(results)
    fwd_df["dd_quantile"] = dd_q

    all_cols = horizon_labels + sharpe_labels
    summary = fwd_df.groupby("dd_quantile", observed=True)[all_cols].median()
    summary.index.name = "Entry Drawdown Quintile"

    dd_range = dd.groupby(dd_q, observed=True).agg(["min", "max"])
    summary.insert(0, "DD Range", [
        f"{dd_range.loc[q, 'min']:.1%} to {dd_range.loc[q, 'max']:.1%}"
        if q in dd_range.index else "—"
        for q in summary.index
    ])

    return summary


# ---------------------------------------------------------------------------
# Fan chart summary statistics
# ---------------------------------------------------------------------------

def fan_chart_summary(
    paths: np.ndarray,
    cfg: SimConfig | None = None,
) -> pd.DataFrame:
    """Compute percentile bands across simulated paths for plotting.

    Returns DataFrame with index = day (0..horizon-1), columns = percentile
    labels (e.g. 'p5', 'p25', ..., 'p95', 'mean').
    """
    if cfg is None:
        cfg = SimConfig()

    records = {}
    for p in cfg.percentiles:
        records[f"p{p}"] = np.percentile(paths, p, axis=0)
    records["mean"] = paths.mean(axis=0)

    return pd.DataFrame(records)


def terminal_wealth_table(
    paths: np.ndarray,
    cfg: SimConfig | None = None,
) -> dict:
    """Summary statistics on terminal wealth from simulated paths.

    Returns dict with terminal wealth stats for a *cfg.initial_capital*
    starting allocation.
    """
    if cfg is None:
        cfg = SimConfig()

    terminal = paths[:, -1] * cfg.initial_capital
    return {
        "initial_capital": cfg.initial_capital,
        "horizon_years": cfg.horizon_days / 365,
        "n_simulations": cfg.n_paths,
        "mean": float(np.mean(terminal)),
        "median": float(np.median(terminal)),
        "p5": float(np.percentile(terminal, 5)),
        "p25": float(np.percentile(terminal, 25)),
        "p75": float(np.percentile(terminal, 75)),
        "p95": float(np.percentile(terminal, 95)),
        "min": float(np.min(terminal)),
        "max": float(np.max(terminal)),
        "prob_profit": float((terminal > cfg.initial_capital).mean()),
        "prob_double": float((terminal > 2 * cfg.initial_capital).mean()),
        "prob_triple": float((terminal > 3 * cfg.initial_capital).mean()),
    }


def simulated_drawdown_stats(
    paths: np.ndarray,
    cfg: SimConfig | None = None,
) -> dict:
    """Compute max-drawdown distribution across simulated paths."""
    if cfg is None:
        cfg = SimConfig()

    max_dds = np.empty(paths.shape[0])
    for i in range(paths.shape[0]):
        running_max = np.maximum.accumulate(paths[i])
        dd = paths[i] / running_max - 1.0
        max_dds[i] = dd.min()

    return {
        "median_max_dd": float(np.median(max_dds)),
        "p5_max_dd": float(np.percentile(max_dds, 5)),
        "p25_max_dd": float(np.percentile(max_dds, 25)),
        "p75_max_dd": float(np.percentile(max_dds, 75)),
        "p95_max_dd": float(np.percentile(max_dds, 95)),
        "worst_max_dd": float(np.min(max_dds)),
        "best_max_dd": float(np.max(max_dds)),
        "prob_dd_lt_20pct": float((max_dds > -0.20).mean()),
        "prob_dd_lt_30pct": float((max_dds > -0.30).mean()),
        "prob_dd_lt_50pct": float((max_dds > -0.50).mean()),
    }


# ---------------------------------------------------------------------------
# BTC buy-and-hold forward simulation (for comparison)
# ---------------------------------------------------------------------------

def btc_bootstrap_paths(
    btc_daily_returns: pd.Series,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """Block bootstrap for BTC buy-and-hold (same method, different input)."""
    return block_bootstrap_paths(btc_daily_returns, cfg)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_turtle_backtest():
    """Run the DD+BTC+Top10 turtle variant and return equity + daily returns."""
    from scripts.research.alpha_lab.turtle_portfolio_v2 import (
        prepare_data, run_simulation, OverlayConfig,
    )
    data = prepare_data()
    cfg = OverlayConfig(
        name="DD + BTC + Top10",
        dd_control=True,
        btc_filter=True,
        concentrated=True,
        top_n=10,
    )
    sim = run_simulation(data, cfg)
    equity = sim["equity_norm"]
    daily_ret = equity.pct_change().dropna()

    btc = data["close"]["BTC-USD"].dropna()
    btc_eq = btc / btc.iloc[0]
    btc_ret = btc_eq.pct_change().dropna()

    returns_wide = data["close"].pct_change().dropna()

    return equity, daily_ret, btc_eq, btc_ret, returns_wide, data


def main():
    """Run forward simulations and save summary outputs."""
    out_dir = ROOT / "artifacts" / "research" / "alpha_lab"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Forward Simulation Engine")
    print("=" * 70)

    equity, strat_ret, btc_eq, btc_ret, returns_wide, data = _run_turtle_backtest()

    cfg = SimConfig(n_paths=10_000, horizon_days=365 * 3, block_size=21)

    print(f"\n[sim] Block bootstrap: {cfg.n_paths:,} paths × {cfg.horizon_days} days ...")
    paths = block_bootstrap_paths(strat_ret, cfg)
    fan = fan_chart_summary(paths, cfg)
    fan.to_csv(out_dir / "fwd_sim_fan_chart.csv", index=True)
    print(f"  Fan chart saved ({len(fan)} rows)")

    tw = terminal_wealth_table(paths, cfg)
    print(f"  Terminal wealth: median ${tw['median']:,.0f}, "
          f"P(profit)={tw['prob_profit']:.0%}, P(2×)={tw['prob_double']:.0%}")

    current_dd = float(equity.iloc[-1] / equity.max() - 1.0)
    print(f"\n[sim] Post-drawdown bootstrap (current DD = {current_dd:.1%}) ...")
    dd_paths = post_drawdown_bootstrap_paths(strat_ret, equity, current_dd, cfg=cfg)
    dd_fan = fan_chart_summary(dd_paths, cfg)
    dd_fan.to_csv(out_dir / "fwd_sim_post_dd_fan.csv", index=True)

    dd_tw = terminal_wealth_table(dd_paths, cfg)
    print(f"  Post-DD-entry terminal: median ${dd_tw['median']:,.0f}, "
          f"P(profit)={dd_tw['prob_profit']:.0%}")

    print("\n[sim] BTC buy-and-hold bootstrap ...")
    btc_paths = btc_bootstrap_paths(btc_ret, cfg)
    btc_fan = fan_chart_summary(btc_paths, cfg)
    btc_fan.to_csv(out_dir / "fwd_sim_btc_bh_fan.csv", index=True)

    print("\n[sim] Historical analogues ...")
    analogues = historical_analogues(equity, cfg.horizon_days)
    for label, path in analogues.items():
        print(f"  {label}: {len(path)} days, terminal = {path.iloc[-1]:.2f}×")

    print("\n[sim] Conditional entry analysis ...")
    cond = conditional_entry_returns(equity)
    cond.to_csv(out_dir / "fwd_sim_conditional_entry.csv")
    print(cond.to_string())

    print("\n[sim] Drawdown distribution ...")
    dd_stats = simulated_drawdown_stats(paths, cfg)
    print(f"  Median max DD: {dd_stats['median_max_dd']:.1%}")
    print(f"  95th pctl max DD: {dd_stats['p5_max_dd']:.1%}")
    print(f"  P(DD < 20%): {dd_stats['prob_dd_lt_20pct']:.0%}")
    print(f"  P(DD < 50%): {dd_stats['prob_dd_lt_50pct']:.0%}")

    print("\n[sim] Done. Artifacts in:", out_dir)


if __name__ == "__main__":
    main()
