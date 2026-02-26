#!/usr/bin/env python3
"""
Combined Alpha Strategy — Production Prototype v2

Concentrated trend-timing system on the top liquid crypto assets.

Core logic:
  - Universe: top N assets by rolling ADV (concentrated, not broad)
  - Per-asset trend score: dual-speed MA crossover + breakout confirmation
  - Position sizing: score × inverse-vol (risk-parity style)
  - Portfolio-level vol targeting + BTC danger overlay
  - Walk-forward: 2017-2021 train, 2022-2026 test

Key design decisions based on research:
  - Concentrated (5-15 assets) not diversified (50+ altcoins)
  - Trend timing is the alpha, not cross-sectional selection
  - Majors (BTC, ETH) should dominate weight naturally via lower vol
  - Danger flags provide crash protection

Usage:
    python -m scripts.research.alpha_lab.run_combined_alpha
    python -m scripts.research.alpha_lab.run_combined_alpha --vol-target 0.15 --n-assets 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, filter_universe, ANN_FACTOR
from common.backtest import simple_backtest
from common.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "research" / "combined_alpha"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GREEN = "#336633"; GRAY = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRAY,
    "figure.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation — dual-speed trend score per asset
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_score(close: pd.Series) -> pd.Series:
    """Dual-speed trend score in [0, 1] for a single asset.

    Fast channel: MA(5, 40) + breakout(10) confirmation
    Slow channel: MA(20, 200) + breakout(50) confirmation

    Each channel outputs 0 or 1. Final score = 0.3 × fast + 0.7 × slow.
    Uses lagged prices (shift(1)) for all indicators to prevent lookahead.
    """
    c = close.copy()
    c_lag = c.shift(1)

    ma5 = c_lag.rolling(5, min_periods=5).mean()
    ma40 = c_lag.rolling(40, min_periods=40).mean()
    fast_ma = (ma5 > ma40).astype(float)

    brk10_high = c_lag.rolling(10, min_periods=10).max()
    fast_brk = (c > brk10_high).astype(float)

    fast_signal = fast_ma * fast_brk

    ma20 = c_lag.rolling(20, min_periods=20).mean()
    ma200 = c_lag.rolling(200, min_periods=200).mean()
    slow_ma = (ma20 > ma200).astype(float)

    brk50_high = c_lag.rolling(50, min_periods=50).max()
    slow_brk = (c > brk50_high).astype(float)

    slow_signal = slow_ma * slow_brk

    score = 0.3 * fast_signal + 0.7 * slow_signal
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Danger flags (from transtrend v2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_danger_flags(
    btc_close: pd.Series,
    vol_threshold: float = 0.80,
    dd_threshold: float = -0.20,
    ret5_threshold: float = -0.10,
) -> pd.Series:
    """BTC-based danger flags: high vol, drawdown, or crash."""
    ret = btc_close.pct_change()
    vol_ann = ret.shift(1).rolling(20, min_periods=20).std() * np.sqrt(365)
    peak20 = btc_close.shift(1).rolling(20, min_periods=20).max()
    dd20 = btc_close / peak20 - 1.0
    ret5 = btc_close / btc_close.shift(5) - 1.0
    danger = (vol_ann > vol_threshold) | (dd20 < dd_threshold) | (ret5 < ret5_threshold)
    return danger.fillna(False).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic top-N universe
# ─────────────────────────────────────────────────────────────────────────────

def build_topn_universe(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    n: int = 10,
    adv_window: int = 20,
    min_history: int = 200,
) -> pd.DataFrame:
    """Select top-N assets by rolling ADV at each date. Boolean mask."""
    dollar_vol = close_wide * volume_wide
    adv = dollar_vol.rolling(adv_window, min_periods=adv_window).mean()

    history_mask = close_wide.notna().cumsum() >= min_history

    ranked = adv.where(history_mask).rank(axis=1, ascending=False, method="first")
    topn = ranked <= n
    return topn.fillna(False).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ─────────────────────────────────────────────────────────────────────────────

def build_trend_weights(
    close_wide: pd.DataFrame,
    universe_mask: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 20,
    vol_floor: float = 0.10,
    target_vol: float = 0.20,
    max_gross: float = 1.0,
    danger_gross: float = 0.25,
    cash_buffer: float = 0.05,
    danger: pd.Series | None = None,
) -> pd.DataFrame:
    """Build portfolio weights: trend score × inverse-vol, vol-targeted.

    This mirrors the transtrend v2 architecture:
    1. Compute per-asset trend score (0 to 1)
    2. Weight = score / annualized_vol
    3. Normalize to target gross exposure
    4. Vol-target the portfolio
    5. Apply danger flags (scale down to danger_gross)
    """
    symbols = close_wide.columns.tolist()
    dates = close_wide.index

    scores = {}
    for sym in symbols:
        close = close_wide[sym].dropna()
        if len(close) < 200:
            continue
        scores[sym] = compute_trend_score(close)

    score_wide = pd.DataFrame(scores).reindex(dates).fillna(0.0)
    score_wide = score_wide.where(universe_mask.reindex(columns=score_wide.columns, index=dates).fillna(False), 0.0)

    vol_ann = (
        returns_wide.shift(1)
        .rolling(vol_lookback, min_periods=max(10, vol_lookback // 2))
        .std() * np.sqrt(ANN_FACTOR)
    )
    vol_ann = vol_ann.clip(lower=vol_floor)

    inv_vol = score_wide.reindex(columns=vol_ann.columns, index=dates).fillna(0.0) / \
              vol_ann.reindex(columns=score_wide.columns, index=dates).fillna(1.0)
    inv_vol = inv_vol.where(np.isfinite(inv_vol), 0.0)

    gross_target = min(max_gross, 1.0 - cash_buffer)
    gross_raw = inv_vol.sum(axis=1).replace(0, np.nan)
    weights = inv_vol.div(gross_raw, axis=0).fillna(0.0) * gross_target

    port_vol_est = np.sqrt(((weights * vol_ann.reindex(columns=weights.columns, index=dates).fillna(vol_floor)) ** 2).sum(axis=1))
    vol_scalar = (target_vol / port_vol_est).clip(lower=0.0, upper=1.5).fillna(1.0)
    weights = weights.mul(vol_scalar, axis=0)

    final_gross = weights.sum(axis=1)
    over = final_gross > max_gross
    if over.any():
        scale_down = (max_gross / final_gross).clip(upper=1.0)
        weights = weights.mul(scale_down, axis=0)

    if danger is not None:
        danger_aligned = danger.reindex(dates).fillna(False).astype(bool)
        for dt in dates[danger_aligned]:
            g = weights.loc[dt].sum()
            if g > danger_gross:
                weights.loc[dt] *= danger_gross / g

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Backtest & analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cost_bps: float = 20.0,
    label: str = "strategy",
    cash_yield: float = 0.04,
) -> dict:
    """Run backtest and return metrics + equity curve."""
    bt = simple_backtest(weights, returns_wide, cost_bps=cost_bps)
    if bt.empty or len(bt) < 30:
        return {"label": label, "metrics": {}, "equity": pd.Series(dtype=float)}

    # Add cash yield on uninvested portion
    rf_bar = (1 + cash_yield) ** (1 / ANN_FACTOR) - 1.0
    cash_wt = 1.0 - bt["gross_exposure"]
    bt["portfolio_ret"] = bt["portfolio_ret"] + cash_wt.clip(lower=0) * rf_bar
    bt["portfolio_equity"] = (1 + bt["portfolio_ret"]).cumprod()

    eq = bt.set_index("ts")["portfolio_equity"]
    metrics = compute_metrics(eq)
    metrics["avg_turnover"] = float(bt["turnover"].mean())
    metrics["avg_gross"] = float(bt["gross_exposure"].mean())
    return {"label": label, "metrics": metrics, "equity": eq}


def build_ew_topn_weights(universe_mask: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight B&H of the top-N universe."""
    n = universe_mask.sum(axis=1).replace(0, np.nan)
    w = universe_mask.astype(float).div(n, axis=0).fillna(0.0)
    return w


def build_btc_bh_weights(close_wide: pd.DataFrame) -> pd.DataFrame:
    """100% BTC buy-and-hold."""
    w = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    if "BTC-USD" in w.columns:
        w["BTC-USD"] = 1.0
    return w


def build_btc_eth_bh_weights(close_wide: pd.DataFrame) -> pd.DataFrame:
    """50/50 BTC + ETH buy-and-hold."""
    w = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    n = 0
    for sym in ["BTC-USD", "ETH-USD"]:
        if sym in w.columns:
            w[sym] = 0.5
            n += 1
    if n > 0:
        w = w / (w.sum(axis=1).replace(0, 1).values[:, None]) * 1.0
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Charting
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curves(results: list[dict], title: str, filename: str, split_date=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [NAVY, TEAL, RED, GOLD, GREEN, GRAY]
    for i, r in enumerate(results):
        if r["equity"].empty:
            continue
        color = colors[i % len(colors)]
        ax.plot(r["equity"].index, r["equity"].values, label=r["label"],
                color=color, lw=1.5 if i == 0 else 1.0, alpha=1.0 if i == 0 else 0.7)
    if split_date:
        ax.axvline(pd.Timestamp(split_date), color=RED, ls="--", lw=0.8, alpha=0.5, label="OOS Start")
    ax.set_yscale("log")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel("Equity (log scale)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(results: list[dict], title: str, filename: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [NAVY, TEAL, RED, GOLD, GREEN, GRAY]
    for i, r in enumerate(results):
        if r["equity"].empty:
            continue
        dd = r["equity"] / r["equity"].cummax() - 1.0
        color = colors[i % len(colors)]
        ax.fill_between(dd.index, dd.values, 0, alpha=0.2 if i > 0 else 0.3, color=color)
        ax.plot(dd.index, dd.values, label=r["label"], color=color, lw=0.8)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel("Drawdown", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_annual_returns(results: list[dict], filename: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    years_all = set()
    for r in results:
        if r["equity"].empty:
            continue
        rets = r["equity"].resample("YE").last().pct_change().dropna()
        years_all.update(rets.index.year)
    years = sorted(years_all)
    if not years:
        plt.close(fig)
        return

    x = np.arange(len(years))
    width = 0.8 / max(len(results), 1)
    colors = [NAVY, TEAL, RED, GOLD, GREEN, GRAY]

    for i, r in enumerate(results):
        if r["equity"].empty:
            continue
        rets = r["equity"].resample("YE").last().pct_change().dropna()
        vals = [rets.get(pd.Timestamp(f"{y}-12-31"), 0) for y in years]
        ax.bar(x + i * width, vals, width, label=r["label"], color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x + width * len(results) / 2)
    ax.set_xticklabels(years, fontsize=8)
    ax.set_ylabel("Annual Return", fontsize=10)
    ax.set_title("Annual Returns by Strategy", fontweight="bold", fontsize=12)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_metrics_table(results: list[dict], period: str = ""):
    header = f"\n  {'Strategy':<35s} {'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>7s} {'Sortino':>8s} {'Calmar':>7s} {'Vol':>6s} {'Turn':>6s} {'Gross':>6s}"
    print(f"\n  {'─' * 100}")
    if period:
        print(f"  {period}")
    print(header)
    print(f"  {'─' * 100}")
    for r in results:
        m = r["metrics"]
        if not m:
            continue
        print(f"  {r['label']:<35s} "
              f"{m.get('sharpe', 0):>7.2f} "
              f"{m.get('cagr', 0):>6.1%} "
              f"{m.get('max_dd', 0):>6.1%} "
              f"{m.get('sortino', 0):>8.2f} "
              f"{m.get('calmar', 0):>7.2f} "
              f"{m.get('vol', 0):>5.1%} "
              f"{m.get('avg_turnover', 0):>6.3f} "
              f"{m.get('avg_gross', 0):>5.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compute_simple_ma_score(close: pd.Series, fast: int = 5, slow: int = 40) -> pd.Series:
    """Simple MA crossover score: 1 when fast > slow, else 0."""
    c_lag = close.shift(1)
    ma_f = c_lag.rolling(fast, min_periods=fast).mean()
    ma_s = c_lag.rolling(slow, min_periods=slow).mean()
    return (ma_f > ma_s).astype(float)


def build_simple_ma_weights(
    close_wide: pd.DataFrame,
    universe_mask: pd.DataFrame,
    returns_wide: pd.DataFrame,
    fast: int = 5,
    slow: int = 40,
    vol_lookback: int = 20,
    vol_floor: float = 0.10,
    target_vol: float = 0.20,
    max_gross: float = 1.0,
    danger_gross: float = 0.25,
    cash_buffer: float = 0.05,
    danger: pd.Series | None = None,
) -> pd.DataFrame:
    """Build weights using simple SMA crossover + inverse-vol sizing."""
    symbols = close_wide.columns.tolist()
    dates = close_wide.index

    scores = {}
    for sym in symbols:
        close = close_wide[sym].dropna()
        if len(close) < slow + 10:
            continue
        scores[sym] = compute_simple_ma_score(close, fast, slow)

    score_wide = pd.DataFrame(scores).reindex(dates).fillna(0.0)
    score_wide = score_wide.where(
        universe_mask.reindex(columns=score_wide.columns, index=dates).fillna(False), 0.0
    )

    vol_ann = (
        returns_wide.shift(1)
        .rolling(vol_lookback, min_periods=max(10, vol_lookback // 2))
        .std() * np.sqrt(ANN_FACTOR)
    )
    vol_ann = vol_ann.clip(lower=vol_floor)

    inv_vol = score_wide / vol_ann.reindex(columns=score_wide.columns, index=dates).fillna(1.0)
    inv_vol = inv_vol.where(np.isfinite(inv_vol), 0.0)

    gross_target = min(max_gross, 1.0 - cash_buffer)
    gross_raw = inv_vol.sum(axis=1).replace(0, np.nan)
    weights = inv_vol.div(gross_raw, axis=0).fillna(0.0) * gross_target

    port_vol_est = np.sqrt(
        ((weights * vol_ann.reindex(columns=weights.columns, index=dates).fillna(vol_floor)) ** 2).sum(axis=1)
    )
    vol_scalar = (target_vol / port_vol_est).clip(lower=0.0, upper=1.5).fillna(1.0)
    weights = weights.mul(vol_scalar, axis=0)

    final_gross = weights.sum(axis=1)
    over = final_gross > max_gross
    if over.any():
        scale_down = (max_gross / final_gross).clip(upper=1.0)
        weights = weights.mul(scale_down, axis=0)

    if danger is not None:
        danger_aligned = danger.reindex(dates).fillna(False).astype(bool)
        for dt in dates[danger_aligned]:
            g = weights.loc[dt].sum()
            if g > danger_gross:
                weights.loc[dt] *= danger_gross / g

    return weights


def main():
    parser = argparse.ArgumentParser(description="Combined Alpha Strategy v2")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-02-22")
    parser.add_argument("--split", default="2021-12-31", help="Walk-forward split date")
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--vol-target", type=float, default=0.20)
    parser.add_argument("--n-assets", type=int, default=10, help="Top-N assets by ADV")
    parser.add_argument("--max-gross", type=float, default=1.0)
    parser.add_argument("--danger-gross", type=float, default=0.25)
    args = parser.parse_args()

    print("=" * 70)
    print("  COMBINED ALPHA STRATEGY v2 — CONCENTRATED TREND TIMING")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  [1/6] Loading universe...")
    t0 = time.time()
    panel = load_daily_bars(start=args.start, end=args.end)
    print(f"  Loaded {len(panel):,} rows in {time.time()-t0:.0f}s")

    close_wide = panel.pivot(index="ts", columns="symbol", values="close")
    volume_wide = panel.pivot(index="ts", columns="symbol", values="volume")
    returns_wide = close_wide.pct_change(fill_method=None)

    # Top-N dynamic universe
    topn_mask = build_topn_universe(close_wide, volume_wide, n=args.n_assets, min_history=200)
    n_in = topn_mask.sum(axis=1).median()
    print(f"  Top-{args.n_assets} universe: {len(close_wide)} days, ~{n_in:.0f} median assets")

    # ── Danger flags ─────────────────────────────────────────────────────
    danger = None
    if "BTC-USD" in close_wide.columns:
        danger = compute_danger_flags(close_wide["BTC-USD"])

    # ── Build signal weights ─────────────────────────────────────────────
    print("\n  [2/6] Computing dual-speed trend scores + inverse-vol weights...")
    t1 = time.time()
    weights_trend = build_trend_weights(
        close_wide, topn_mask, returns_wide,
        target_vol=args.vol_target,
        max_gross=args.max_gross,
        danger_gross=args.danger_gross,
        danger=danger,
    )
    print(f"  Weights computed in {time.time()-t1:.0f}s")
    avg_gross = weights_trend.sum(axis=1).mean()
    avg_n = (weights_trend > 0.001).sum(axis=1).mean()
    print(f"  Avg gross exposure: {avg_gross:.1%}, avg assets held: {avg_n:.1f}")

    # ── Simple MA alternatives ───────────────────────────────────────────
    print("\n  [3/6] Building simple MA variants + benchmarks...")

    w_ma_5_40 = build_simple_ma_weights(
        close_wide, topn_mask, returns_wide,
        fast=5, slow=40,
        target_vol=args.vol_target, max_gross=args.max_gross,
        danger_gross=args.danger_gross, danger=danger,
    )
    w_ma_10_50 = build_simple_ma_weights(
        close_wide, topn_mask, returns_wide,
        fast=10, slow=50,
        target_vol=args.vol_target, max_gross=args.max_gross,
        danger_gross=args.danger_gross, danger=danger,
    )
    w_ma_20_100 = build_simple_ma_weights(
        close_wide, topn_mask, returns_wide,
        fast=20, slow=100,
        target_vol=args.vol_target, max_gross=args.max_gross,
        danger_gross=args.danger_gross, danger=danger,
    )

    ew_topn_weights = build_ew_topn_weights(topn_mask)
    btc_weights = build_btc_bh_weights(close_wide)
    btc_eth_weights = build_btc_eth_bh_weights(close_wide)

    # ── Run backtests ────────────────────────────────────────────────────
    print("\n  [4/6] Running backtests...")

    results_full = [
        run_backtest(weights_trend, returns_wide, args.cost_bps, "Dual-Speed Trend (MA+BRK)"),
        run_backtest(w_ma_5_40, returns_wide, args.cost_bps, "SMA(5,40) + InvVol"),
        run_backtest(w_ma_10_50, returns_wide, args.cost_bps, "SMA(10,50) + InvVol"),
        run_backtest(w_ma_20_100, returns_wide, args.cost_bps, "SMA(20,100) + InvVol"),
        run_backtest(ew_topn_weights, returns_wide, args.cost_bps, f"EW Top-{args.n_assets} B&H"),
        run_backtest(btc_eth_weights, returns_wide, args.cost_bps, "BTC+ETH 50/50 B&H"),
        run_backtest(btc_weights, returns_wide, args.cost_bps, "BTC B&H"),
    ]

    print_metrics_table(results_full, "FULL PERIOD")

    # ── Walk-forward split ───────────────────────────────────────────────
    print("\n  [5/6] Walk-forward analysis...")
    split = pd.Timestamp(args.split)

    in_mask = weights_trend.index <= split
    out_mask = weights_trend.index > split

    if out_mask.any():
        results_is = []
        results_oos = []
        for w, label in [
            (weights_trend, "Dual-Speed Trend"),
            (w_ma_5_40, "SMA(5,40)"),
            (w_ma_10_50, "SMA(10,50)"),
            (w_ma_20_100, "SMA(20,100)"),
            (ew_topn_weights, f"EW Top-{args.n_assets}"),
            (btc_eth_weights, "BTC+ETH 50/50"),
            (btc_weights, "BTC B&H"),
        ]:
            results_is.append(run_backtest(w.loc[in_mask], returns_wide.loc[in_mask], args.cost_bps, label))
            results_oos.append(run_backtest(w.loc[out_mask], returns_wide.loc[out_mask], args.cost_bps, label))

        print_metrics_table(results_is, f"IN-SAMPLE ({args.start} to {args.split})")
        print_metrics_table(results_oos, f"OUT-OF-SAMPLE ({args.split} to {args.end})")

        if results_is[0]["metrics"] and results_oos[0]["metrics"]:
            is_sharpe = results_is[0]["metrics"].get("sharpe", 0)
            oos_sharpe = results_oos[0]["metrics"].get("sharpe", 0)
            decay = 1 - oos_sharpe / is_sharpe if is_sharpe != 0 else float("nan")
            print(f"\n  Sharpe decay IS→OOS: {is_sharpe:.2f} → {oos_sharpe:.2f} ({decay:.0%} decay)")

            # Calmar comparison
            is_calmar = results_is[0]["metrics"].get("calmar", 0)
            oos_calmar = results_oos[0]["metrics"].get("calmar", 0)
            print(f"  Calmar IS→OOS: {is_calmar:.2f} → {oos_calmar:.2f}")

            # vs BTC OOS
            btc_oos = results_oos[-1]["metrics"].get("sharpe", 0)
            print(f"  OOS: Strategy {oos_sharpe:.2f} vs BTC {btc_oos:.2f} ({oos_sharpe - btc_oos:+.2f} Sharpe difference)")

    # ── Charts ───────────────────────────────────────────────────────────
    print("\n  [6/6] Generating charts...")
    plot_equity_curves(results_full, "Concentrated Trend Alpha — Equity Curves", "equity_curves.png", args.split)
    plot_drawdowns(results_full, "Concentrated Trend Alpha — Drawdowns", "drawdowns.png")
    plot_annual_returns(results_full, "annual_returns.png")

    # Exposure and holdings
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    gross = weights_trend.sum(axis=1)
    axes[0].fill_between(gross.index, gross.values, alpha=0.3, color=TEAL)
    axes[0].plot(gross.index, gross.values, color=TEAL, lw=0.5)
    axes[0].set_ylabel("Gross Exposure", fontsize=10)
    axes[0].set_title("Portfolio Exposure Over Time", fontweight="bold", fontsize=12)
    if danger is not None:
        danger_dates = danger[danger].index
        for dt in danger_dates:
            if dt in gross.index:
                axes[0].axvline(dt, color=RED, alpha=0.02, lw=0.5)

    n_held = (weights_trend > 0.001).sum(axis=1)
    axes[1].plot(n_held.index, n_held.values, color=NAVY, lw=0.5)
    axes[1].set_ylabel("# Assets Held", fontsize=10)
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(OUT / "exposure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Top holdings snapshot
    last_date = weights_trend.index[-1]
    last_weights = weights_trend.loc[last_date].sort_values(ascending=False)
    top_holdings = last_weights[last_weights > 0.001]
    if not top_holdings.empty:
        print(f"\n  Current Holdings ({last_date.strftime('%Y-%m-%d')}):")
        for sym, wt in top_holdings.items():
            print(f"    {sym:<12s} {wt:>6.1%}")
        print(f"    {'CASH':<12s} {1 - top_holdings.sum():>6.1%}")

    # ── Save results ─────────────────────────────────────────────────────
    rows = []
    for r in results_full:
        row = {"strategy": r["label"]}
        row.update(r["metrics"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "metrics_full.csv", index=False)

    if out_mask.any():
        oos_rows = []
        for r in results_oos:
            row = {"strategy": r["label"]}
            row.update(r["metrics"])
            oos_rows.append(row)
        pd.DataFrame(oos_rows).to_csv(OUT / "metrics_oos.csv", index=False)

    weights_trend.to_parquet(OUT / "weights_final.parquet")

    for r in results_full:
        if not r["equity"].empty:
            fname = r["label"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "").replace("+", "").replace("/", "")
            r["equity"].to_csv(OUT / f"equity_{fname}.csv")

    print(f"\n  Outputs saved to {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
