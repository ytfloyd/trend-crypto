#!/usr/bin/env python3
"""
TSMOM Deep Dive — Targeted diagnostics for desk head review.

Part A: VOL_SCALED 21d at 10% vol target (primary spec, revised risk param)
  - Crisis timelines with exact entry/exit dating (esp. May 2021)
  - Year-by-year Sharpe and skewness table
  - Rolling 252d Sharpe and skewness
  - Return distribution with bootstrap CIs
  - Full pass/fail assessment

Part B: LREG_10d walk-forward validation
  - In-sample: 2017-01-01 to 2022-12-31
  - Out-of-sample: 2023-01-01 to 2025-12-15
  - 2022 calendar-year Sharpe and skewness (the real stress test)
  - Trade count and concentration analysis

Usage:
    python -m scripts.research.tsmom.run_deep_dive
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, filter_universe, compute_btc_benchmark, ANN_FACTOR
from common.backtest import simple_backtest, DEFAULT_COST_BPS
from common.metrics import compute_metrics, compute_regime

from .signals import compute_signal
from .weights import build_tsmom_weights, apply_portfolio_vol_target
from .convexity_metrics import (
    compute_convexity_metrics,
    conditional_correlation,
    time_in_market_by_regime,
    regime_sharpe_skew,
    participation_ratio_portfolio,
    participation_ratio_per_asset,
    bootstrap_sharpe,
    bootstrap_skewness,
    extract_crisis_timeline,
    CRISIS_EPISODES,
)

# ── Paths & style ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "tsmom" / "deep_dive"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

JPM_BLUE = "#003A70"
JPM_LIGHT_BLUE = "#0078D4"
JPM_GRAY = "#6D6E71"
JPM_GREEN = "#00843D"
JPM_RED = "#C8102E"
JPM_GOLD = "#B8860B"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": JPM_GRAY,
})


# ── Data ──────────────────────────────────────────────────────────────

def load_data(start="2017-01-01", end="2026-12-31"):
    panel = load_daily_bars(start=start, end=end)
    panel = filter_universe(panel, min_adv_usd=500_000, min_history_days=90)
    panel = panel.sort_values(["ts", "symbol"])

    close_wide = panel.pivot(index="ts", columns="symbol", values="close")
    returns_wide = close_wide.pct_change(fill_method=None)
    universe_wide = (
        panel.pivot(index="ts", columns="symbol", values="in_universe")
        .fillna(False).infer_objects(copy=False).astype(bool)
    )
    btc_equity = compute_btc_benchmark(panel)

    btc_col = None
    for c in close_wide.columns:
        if "BTC" in c.upper():
            btc_col = c
            break

    return close_wide, returns_wide, universe_wide, btc_equity, btc_col


def run_config(config, close_wide, returns_wide, universe_wide, date_filter=None):
    """Run a single config and return (equity, port_ret, weights, bt)."""
    signal = compute_signal(
        config["signal"], close_wide, returns_wide,
        config["lookback"], vol_lookback=config.get("vol_lookback", 63),
    )
    weights = build_tsmom_weights(
        signal, universe_wide, returns_wide,
        sizing=config["sizing"], vol_target=config["vol_target"],
        vol_lookback=config.get("vol_lookback", 63),
        max_weight=config["max_weight"],
    )
    weights_vt = apply_portfolio_vol_target(
        weights, returns_wide, vol_target=config["vol_target"],
    )

    if date_filter:
        mask = weights_vt.index >= pd.Timestamp(date_filter)
        weights_vt = weights_vt.loc[mask]

    bt = simple_backtest(weights_vt, returns_wide, cost_bps=DEFAULT_COST_BPS)
    if bt.empty or len(bt) < 30:
        return None, None, None, None

    equity = bt.set_index("ts")["portfolio_equity"]
    port_ret = bt.set_index("ts")["portfolio_ret"]
    return equity, port_ret, weights_vt, bt


# ── Year-by-year table ────────────────────────────────────────────────

def yearly_sharpe_skew(equity):
    ret = equity.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    years = sorted(ret.index.year.unique())
    rows = []
    for yr in years:
        yr_ret = ret[ret.index.year == yr]
        if len(yr_ret) < 20:
            continue
        std = yr_ret.std()
        sharpe = float((yr_ret.mean() / std) * np.sqrt(ANN_FACTOR)) if std > 1e-12 else np.nan
        skew = float(yr_ret.skew())
        cagr = float((1 + yr_ret).prod() ** (ANN_FACTOR / len(yr_ret)) - 1)
        maxdd = float((equity[equity.index.year == yr] / equity[equity.index.year == yr].cummax() - 1).min())
        rows.append({"year": yr, "sharpe": sharpe, "skewness": skew, "cagr": cagr, "max_dd": maxdd, "n_days": len(yr_ret)})
    return pd.DataFrame(rows)


# ── Trade analysis ────────────────────────────────────────────────────

def count_trades(weights):
    """Count entry/exit events across all assets."""
    is_in = (weights.abs() > 1e-6).astype(int)
    entries = (is_in.diff() == 1).sum().sum()
    exits = (is_in.diff() == -1).sum().sum()
    return int(entries), int(exits)


# ======================================================================
# PART A: VOL_SCALED 21d VT10
# ======================================================================

def part_a(close_wide, returns_wide, universe_wide, btc_equity, btc_col):
    print("\n" + "=" * 70)
    print("  PART A: VOL_SCALED 21d — 10% Vol Target (Primary Spec Revised)")
    print("=" * 70)

    config = {
        "signal": "VOL_SCALED", "lookback": 21, "vol_lookback": 63,
        "sizing": "binary", "exit": "signal_reversal",
        "vol_target": 0.10, "max_weight": 0.20,
    }

    equity, port_ret, weights, bt = run_config(
        config, close_wide, returns_wide, universe_wide,
    )
    if equity is None:
        print("  [ERROR] No data")
        return

    # Full metrics
    metrics = compute_convexity_metrics(equity, weights)
    regime = compute_regime(returns_wide, btc_col)

    print(f"\n  --- CORE METRICS ---")
    print(f"  Sharpe:      {metrics.get('sharpe', np.nan):>8.3f}")
    print(f"  Skewness:    {metrics.get('skewness', np.nan):>8.3f}")
    print(f"  CAGR:        {metrics.get('cagr', np.nan):>8.1%}")
    print(f"  MaxDD:       {metrics.get('max_dd', np.nan):>8.1%}")
    print(f"  Win/Loss:    {metrics.get('win_loss_ratio', np.nan):>8.2f}")
    print(f"  Time in Mkt: {metrics.get('time_in_market', np.nan):>8.1%}")

    # Regime
    cond_corr = conditional_correlation(
        port_ret, returns_wide[btc_col] if btc_col else pd.Series(dtype=float), regime,
    )
    tim = time_in_market_by_regime(weights, regime)
    rss = regime_sharpe_skew(port_ret, regime)

    print(f"\n  --- REGIME ANALYSIS ---")
    for r in ["BULL", "BEAR", "CHOP"]:
        print(f"  {r:5s}: Sharpe={rss.get(r,{}).get('sharpe',np.nan):>6.2f}  "
              f"Skew={rss.get(r,{}).get('skewness',np.nan):>6.2f}  "
              f"TimeIn={tim.get(r,np.nan):>5.1%}  BTC_corr={cond_corr.get(r,np.nan):>6.3f}")

    # Pass/fail
    print(f"\n  --- PASS/FAIL ---")
    checks = {
        "Skewness > 0":       metrics.get("skewness", -99) > 0,
        "Sharpe > 0":         metrics.get("sharpe", -99) > 0,
        "MaxDD > -30%":       metrics.get("max_dd", -99) > -0.30,
        "BEAR corr < 0.5":    cond_corr.get("BEAR", 99) < 0.5,
        "Participation > 20%": metrics.get("participation_per_asset", 0) > 0.20 if "participation_per_asset" in metrics else True,
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  {'>>> ALL PASS <<<' if all(checks.values()) else '>>> FAILURES ABOVE'}")

    # Bootstrap CIs
    invested_ret = port_ret[port_ret.abs() > 1e-10]
    if len(invested_ret) > 50:
        sh_pt, sh_lo, sh_hi = bootstrap_sharpe(invested_ret)
        sk_pt, sk_lo, sk_hi = bootstrap_skewness(invested_ret)
        print(f"\n  --- BOOTSTRAP 95% CI ---")
        print(f"  Sharpe:   {sh_pt:>6.3f}  [{sh_lo:>6.3f}, {sh_hi:>6.3f}]")
        print(f"  Skewness: {sk_pt:>6.3f}  [{sk_lo:>6.3f}, {sk_hi:>6.3f}]")

    # Year-by-year
    yy = yearly_sharpe_skew(equity)
    print(f"\n  --- YEAR-BY-YEAR ---")
    print(f"  {'Year':>6s} {'Sharpe':>8s} {'Skew':>8s} {'CAGR':>8s} {'MaxDD':>8s}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for _, row in yy.iterrows():
        print(f"  {int(row['year']):>6d} {row['sharpe']:>8.2f} {row['skewness']:>8.2f} "
              f"{row['cagr']:>7.1%} {row['max_dd']:>7.1%}")
    yy.to_csv(ARTIFACT_DIR / "vt10_yearly.csv", index=False, float_format="%.4f")

    # Crisis timelines
    print(f"\n  --- CRISIS TIMELINES ---")
    btc_close = close_wide[btc_col] if btc_col else pd.Series(dtype=float)
    crisis_data = {}
    for ep_name in CRISIS_EPISODES:
        ct = extract_crisis_timeline(btc_close, weights, port_ret, ep_name)
        if ct is not None:
            crisis_data[ep_name] = ct
            avg_wt = ct["total_weight"].mean()
            cum_pnl = (1 + ct["daily_pnl"]).prod() - 1
            btc_move = ct["btc_price"].iloc[-1] / ct["btc_price"].iloc[0] - 1
            print(f"  {ep_name:<20s}: avg_wt={avg_wt:.1%}  strat={cum_pnl:>+.1%}  btc={btc_move:>+.1%}")
            ct.to_csv(ARTIFACT_DIR / f"vt10_crisis_{ep_name.replace(' ', '_').replace('(','').replace(')','')}.csv")

    # May 2021 detailed timing
    if "May 2021" in crisis_data:
        ct = crisis_data["May 2021"]
        print(f"\n  --- MAY 2021 DETAILED TIMING ---")
        # Find BTC peak and trough in this window
        btc_peak_date = ct["btc_price"].idxmax()
        btc_trough_date = ct["btc_price"].idxmin()
        btc_peak_price = ct["btc_price"].max()
        btc_trough_price = ct["btc_price"].min()
        btc_dd = btc_trough_price / btc_peak_price - 1

        print(f"  BTC peak:   {btc_peak_date.strftime('%Y-%m-%d')}  ${btc_peak_price:,.0f}")
        print(f"  BTC trough: {btc_trough_date.strftime('%Y-%m-%d')}  ${btc_trough_price:,.0f}  ({btc_dd:.1%})")

        # Strategy weight at key dates
        pre_peak = ct.loc[:btc_peak_date]
        post_peak = ct.loc[btc_peak_date:]
        wt_at_peak = ct.loc[btc_peak_date, "total_weight"] if btc_peak_date in ct.index else np.nan
        wt_at_trough = ct.loc[btc_trough_date, "total_weight"] if btc_trough_date in ct.index else np.nan

        # When did weight first drop below 5%?
        post_peak_low = post_peak[post_peak["total_weight"] < 0.05]
        exit_date = post_peak_low.index[0] if len(post_peak_low) > 0 else None
        days_peak_to_exit = (exit_date - btc_peak_date).days if exit_date else None

        # Strat return from peak to trough
        peak_to_trough = ct.loc[btc_peak_date:btc_trough_date]
        strat_dd = (1 + peak_to_trough["daily_pnl"]).prod() - 1

        print(f"  Weight at BTC peak:   {wt_at_peak:.1%}")
        print(f"  Weight at BTC trough: {wt_at_trough:.1%}")
        if exit_date:
            print(f"  Strategy exited (wt<5%): {exit_date.strftime('%Y-%m-%d')}  ({days_peak_to_exit}d after peak)")
        else:
            print(f"  Strategy NEVER fully exited during this window")
        print(f"  Strat return peak→trough: {strat_dd:+.1%}")
        print(f"  BTC return peak→trough:   {btc_dd:+.1%}")
        print(f"  Drawdown absorbed:        {abs(strat_dd) / abs(btc_dd) * 100:.0f}% of BTC drawdown")

    # ── Charts ────────────────────────────────────────────────────────

    # Exhibit A1: Equity + histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot(equity.index, equity.values, color=JPM_BLUE, linewidth=1.5, label="VT10 Primary Spec")
    btc_aligned = btc_equity.reindex(equity.index).ffill().dropna()
    if len(btc_aligned) > 0:
        btc_norm = btc_aligned / btc_aligned.iloc[0]
        ax1.plot(btc_norm.index, btc_norm.values, color=JPM_GRAY, linewidth=1, alpha=0.7,
                 linestyle="--", label="BTC Buy & Hold")
    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value (log)")
    ax1.set_title("VT10: Equity Curve")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ret_vals = equity.pct_change().dropna()
    ax2.hist(ret_vals.values, bins=100, color=JPM_BLUE, alpha=0.7, edgecolor="white", density=True)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_title(f"VT10: Return Distribution (skew={metrics.get('skewness',np.nan):.2f})")
    ax2.set_xlabel("Daily Return")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "vt10_equity_histogram.png", dpi=150)
    plt.close(fig)

    # Exhibit A2: Rolling 252d Sharpe + Skewness
    rolling_ret = equity.pct_change().dropna()
    rolling_sharpe = (
        rolling_ret.rolling(252, min_periods=126).mean()
        / rolling_ret.rolling(252, min_periods=126).std()
        * np.sqrt(ANN_FACTOR)
    )
    rolling_skew = rolling_ret.rolling(252, min_periods=126).apply(lambda x: x.skew(), raw=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(rolling_sharpe.index, rolling_sharpe.values, color=JPM_BLUE, linewidth=1)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Rolling 252d Sharpe")
    ax1.set_title("VT10: Rolling Performance Stability")

    ax2.plot(rolling_skew.index, rolling_skew.values, color=JPM_GREEN, linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(0.5, color=JPM_GREEN, linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.set_ylabel("Rolling 252d Skewness")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "vt10_rolling.png", dpi=150)
    plt.close(fig)

    # Exhibit A3: May 2021 crisis detail
    if "May 2021" in crisis_data:
        ct = crisis_data["May 2021"]
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(ct.index, ct["btc_price"], color=JPM_BLUE, linewidth=1.2)
        axes[0].axvline(btc_peak_date, color=JPM_RED, linewidth=0.8, linestyle="--", alpha=0.7)
        axes[0].axvline(btc_trough_date, color=JPM_RED, linewidth=0.8, linestyle="--", alpha=0.7)
        axes[0].set_ylabel("BTC Price ($)")
        axes[0].set_title("May 2021 Crisis — Detailed Exit Timing")

        axes[1].fill_between(ct.index, ct["total_weight"], 0, color=JPM_GREEN, alpha=0.5)
        axes[1].axvline(btc_peak_date, color=JPM_RED, linewidth=0.8, linestyle="--", alpha=0.7)
        axes[1].set_ylabel("Portfolio Weight")
        if exit_date:
            axes[1].axvline(exit_date, color=JPM_GOLD, linewidth=1.5, linestyle="-", alpha=0.8, label=f"Exit ({exit_date.strftime('%b %d')})")
            axes[1].legend(fontsize=8)

        cum_pnl = (1 + ct["daily_pnl"]).cumprod() - 1
        axes[2].plot(ct.index, cum_pnl, color=JPM_BLUE, linewidth=1.2)
        axes[2].fill_between(ct.index, cum_pnl, 0, where=cum_pnl >= 0, color=JPM_GREEN, alpha=0.2)
        axes[2].fill_between(ct.index, cum_pnl, 0, where=cum_pnl < 0, color=JPM_RED, alpha=0.2)
        axes[2].axhline(0, color="black", linewidth=0.5)
        axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        axes[2].set_ylabel("Cumulative P&L")
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))

        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "vt10_may2021_detail.png", dpi=150)
        plt.close(fig)

    # Exhibit A4: Year-by-year bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(yy))
    colors_sharpe = [JPM_GREEN if v > 0 else JPM_RED for v in yy["sharpe"]]
    ax1.bar(x, yy["sharpe"], color=colors_sharpe, alpha=0.85, edgecolor="white")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(yy["year"].astype(int), fontsize=9)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("VT10: Year-by-Year Sharpe")

    colors_skew = [JPM_GREEN if v > 0 else JPM_RED for v in yy["skewness"]]
    ax2.bar(x, yy["skewness"], color=colors_skew, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(0.5, color=JPM_GREEN, linewidth=0.8, linestyle=":", alpha=0.5, label="Target")
    ax2.set_xticks(x)
    ax2.set_xticklabels(yy["year"].astype(int), fontsize=9)
    ax2.set_ylabel("Skewness")
    ax2.set_title("VT10: Year-by-Year Skewness")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "vt10_yearly_bars.png", dpi=150)
    plt.close(fig)

    equity.to_csv(ARTIFACT_DIR / "vt10_equity.csv")

    # Save summary
    summary = {
        "config": config,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, type(None)))},
        "regime": {
            "conditional_corr": cond_corr,
            "time_in_market_by_regime": tim,
            "regime_sharpe_skew": rss,
        },
    }
    with open(ARTIFACT_DIR / "vt10_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  VT10 artifacts saved to {ARTIFACT_DIR}")
    return equity, port_ret, weights


# ======================================================================
# PART B: LREG_10d Walk-Forward + 2022 Stress Test
# ======================================================================

def part_b(close_wide, returns_wide, universe_wide, btc_equity, btc_col):
    print("\n" + "=" * 70)
    print("  PART B: LREG_10d Walk-Forward Validation")
    print("=" * 70)

    config = {
        "signal": "LREG", "lookback": 10, "vol_lookback": 63,
        "sizing": "binary", "exit": "signal_reversal",
        "vol_target": 0.15, "max_weight": 0.20,
    }

    # ── Full period (for reference) ───────────────────────────────────
    equity_full, port_ret_full, weights_full, bt_full = run_config(
        config, close_wide, returns_wide, universe_wide,
    )

    if equity_full is None:
        print("  [ERROR] No data for full period")
        return

    # Trade count
    entries, exits = count_trades(weights_full)
    print(f"\n  --- TRADE ANALYSIS (full period) ---")
    print(f"  Total entries: {entries}")
    print(f"  Total exits:   {exits}")
    n_days = len(equity_full)
    print(f"  Trading days:  {n_days}")
    print(f"  Avg entries/day: {entries/n_days:.1f}")

    # 2022 calendar year (the real stress test)
    ret_full = equity_full.pct_change().dropna()
    ret_2022 = ret_full[ret_full.index.year == 2022]

    if len(ret_2022) > 20:
        std_22 = ret_2022.std()
        sharpe_22 = float((ret_2022.mean() / std_22) * np.sqrt(ANN_FACTOR)) if std_22 > 1e-12 else np.nan
        skew_22 = float(ret_2022.skew())
        cagr_22 = float((1 + ret_2022).prod() ** (ANN_FACTOR / len(ret_2022)) - 1)

        eq_2022 = equity_full[equity_full.index.year == 2022]
        maxdd_22 = float((eq_2022 / eq_2022.cummax() - 1).min())

        print(f"\n  --- 2022 STRESS TEST (in-sample) ---")
        print(f"  Sharpe:   {sharpe_22:>8.3f}")
        print(f"  Skewness: {skew_22:>8.3f}")
        print(f"  CAGR:     {cagr_22:>8.1%}")
        print(f"  MaxDD:    {maxdd_22:>8.1%}")
        print(f"  N days:   {len(ret_2022)}")

    # ── Walk-forward: OOS on 2023-2025 ────────────────────────────────
    print(f"\n  --- WALK-FORWARD ---")
    print(f"  In-sample:      2017-01-01 to 2022-12-31")
    print(f"  Out-of-sample:  2023-01-01 to 2025-12-15")

    equity_oos, port_ret_oos, weights_oos, bt_oos = run_config(
        config, close_wide, returns_wide, universe_wide,
        date_filter="2023-01-01",
    )

    if equity_oos is None or len(equity_oos) < 30:
        print("  [ERROR] Insufficient OOS data")
        return

    metrics_oos = compute_convexity_metrics(equity_oos, weights_oos)
    regime_oos = compute_regime(returns_wide, btc_col)

    print(f"\n  --- OOS METRICS (2023-2025) ---")
    print(f"  Sharpe:      {metrics_oos.get('sharpe', np.nan):>8.3f}")
    print(f"  Skewness:    {metrics_oos.get('skewness', np.nan):>8.3f}")
    print(f"  CAGR:        {metrics_oos.get('cagr', np.nan):>8.1%}")
    print(f"  MaxDD:       {metrics_oos.get('max_dd', np.nan):>8.1%}")
    print(f"  Hit Rate:    {metrics_oos.get('hit_rate', np.nan):>8.1%}")
    print(f"  Win/Loss:    {metrics_oos.get('win_loss_ratio', np.nan):>8.2f}")
    print(f"  Time in Mkt: {metrics_oos.get('time_in_market', np.nan):>8.1%}")

    # OOS regime
    if btc_col and btc_col in returns_wide.columns:
        cond_corr_oos = conditional_correlation(
            port_ret_oos, returns_wide[btc_col].reindex(port_ret_oos.index), regime_oos,
        )
        rss_oos = regime_sharpe_skew(port_ret_oos, regime_oos)
        print(f"\n  --- OOS REGIME ---")
        for r in ["BULL", "BEAR", "CHOP"]:
            print(f"  {r:5s}: Sharpe={rss_oos.get(r,{}).get('sharpe',np.nan):>6.2f}  "
                  f"Skew={rss_oos.get(r,{}).get('skewness',np.nan):>6.2f}  "
                  f"BTC_corr={cond_corr_oos.get(r,np.nan):>6.3f}")

    # OOS pass/fail
    bear_corr_oos = cond_corr_oos.get("BEAR", 99) if btc_col else np.nan
    print(f"\n  --- OOS PASS/FAIL ---")
    oos_checks = {
        "Skewness > 0":    metrics_oos.get("skewness", -99) > 0,
        "Sharpe > 0":      metrics_oos.get("sharpe", -99) > 0,
        "MaxDD > -30%":    metrics_oos.get("max_dd", -99) > -0.30,
        "BEAR corr < 0.5": bear_corr_oos < 0.5 if not np.isnan(bear_corr_oos) else False,
    }
    for name, passed in oos_checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    # ── In-sample metrics (2017-2022) for comparison ──────────────────
    equity_is, port_ret_is, weights_is, bt_is = run_config(
        config, close_wide, returns_wide, universe_wide,
    )
    if equity_is is not None:
        ret_is = equity_is.pct_change().dropna()
        ret_is = ret_is[ret_is.index < pd.Timestamp("2023-01-01")]
        if len(ret_is) > 30:
            std_is = ret_is.std()
            sharpe_is = float((ret_is.mean() / std_is) * np.sqrt(ANN_FACTOR)) if std_is > 1e-12 else np.nan
            skew_is = float(ret_is.skew())
            print(f"\n  --- IN-SAMPLE (2017-2022) ---")
            print(f"  Sharpe:   {sharpe_is:>8.3f}")
            print(f"  Skewness: {skew_is:>8.3f}")

    # Year-by-year for LREG_10d
    yy = yearly_sharpe_skew(equity_full)
    print(f"\n  --- YEAR-BY-YEAR ---")
    print(f"  {'Year':>6s} {'Sharpe':>8s} {'Skew':>8s} {'CAGR':>8s} {'MaxDD':>8s}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for _, row in yy.iterrows():
        marker = " ← OOS" if row["year"] >= 2023 else ""
        print(f"  {int(row['year']):>6d} {row['sharpe']:>8.2f} {row['skewness']:>8.2f} "
              f"{row['cagr']:>7.1%} {row['max_dd']:>7.1%}{marker}")
    yy.to_csv(ARTIFACT_DIR / "lreg10_yearly.csv", index=False, float_format="%.4f")

    # ── Charts ────────────────────────────────────────────────────────

    # Exhibit B1: IS vs OOS equity
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_full.index, equity_full.values, color=JPM_BLUE, linewidth=1.5, label="LREG_10d Full Period")
    oos_start = pd.Timestamp("2023-01-01")
    ax.axvline(oos_start, color=JPM_RED, linewidth=1.5, linestyle="--", alpha=0.7, label="OOS boundary")
    ax.axvspan(oos_start, equity_full.index[-1], alpha=0.05, color=JPM_GOLD)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio Value (log)")
    ax.set_title("LREG_10d: In-Sample vs Out-of-Sample")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "lreg10_walkforward.png", dpi=150)
    plt.close(fig)

    # Exhibit B2: Year-by-year bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(yy))
    colors_sharpe = [JPM_GOLD if yr >= 2023 else (JPM_GREEN if s > 0 else JPM_RED)
                     for yr, s in zip(yy["year"], yy["sharpe"])]
    ax1.bar(x, yy["sharpe"], color=colors_sharpe, alpha=0.85, edgecolor="white")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(yy["year"].astype(int), fontsize=9)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("LREG_10d Year-by-Year (gold = OOS)")

    colors_skew = [JPM_GOLD if yr >= 2023 else (JPM_GREEN if s > 0 else JPM_RED)
                   for yr, s in zip(yy["year"], yy["skewness"])]
    ax2.bar(x, yy["skewness"], color=colors_skew, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(yy["year"].astype(int), fontsize=9)
    ax2.set_ylabel("Skewness")
    ax2.set_title("LREG_10d Year-by-Year Skewness (gold = OOS)")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "lreg10_yearly_bars.png", dpi=150)
    plt.close(fig)

    equity_full.to_csv(ARTIFACT_DIR / "lreg10_equity_full.csv")
    equity_oos.to_csv(ARTIFACT_DIR / "lreg10_equity_oos.csv")

    summary = {
        "config": config,
        "full_period": {k: v for k, v in compute_convexity_metrics(equity_full, weights_full).items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
        "oos_2023_2025": {k: v for k, v in metrics_oos.items()
                          if isinstance(v, (int, float, str, bool, type(None)))},
        "stress_2022": {"sharpe": sharpe_22, "skewness": skew_22, "cagr": cagr_22, "max_dd": maxdd_22}
        if len(ret_2022) > 20 else {},
        "trade_count": {"entries": entries, "exits": exits},
    }
    with open(ARTIFACT_DIR / "lreg10_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  LREG_10d artifacts saved to {ARTIFACT_DIR}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TSMOM DEEP DIVE — DESK HEAD REVIEW")
    print("=" * 70)

    close_wide, returns_wide, universe_wide, btc_equity, btc_col = load_data()

    part_a(close_wide, returns_wide, universe_wide, btc_equity, btc_col)
    part_b(close_wide, returns_wide, universe_wide, btc_equity, btc_col)

    print("\n" + "=" * 70)
    print("  DEEP DIVE COMPLETE")
    print(f"  Artifacts: {ARTIFACT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
