#!/usr/bin/env python3
"""
Generate JPM-style TSMOM research report with charts and PDF.

Produces:
  - artifacts/research/tsmom/report_charts/*.png
  - docs/research/tsmom_report.md
  - artifacts/research/tsmom/tsmom_report.pdf

Usage:
    python -m scripts.research.tsmom.generate_report
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
TSMOM_DIR = ROOT / "artifacts" / "research" / "tsmom"
CHART_DIR = TSMOM_DIR / "report_charts"
DOCS_DIR = ROOT / "docs" / "research"

CHART_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── JPM-style palette ─────────────────────────────────────────────────
JPM_BLUE = "#003A70"
JPM_LIGHT_BLUE = "#0078D4"
JPM_GRAY = "#6D6E71"
JPM_GREEN = "#00843D"
JPM_RED = "#C8102E"
JPM_GOLD = "#B8860B"
JPM_ORANGE = "#E87722"
PALETTE = [JPM_BLUE, JPM_RED, JPM_GREEN, JPM_GOLD, JPM_LIGHT_BLUE, JPM_ORANGE, JPM_GRAY]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": JPM_GRAY,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": JPM_GRAY,
})


def sf(v):
    """Safe float."""
    if v is None:
        return float("nan")
    if isinstance(v, complex):
        return float(v.real)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ── Load artifacts ────────────────────────────────────────────────────

def load_primary():
    path = TSMOM_DIR / "primary_spec_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_primary_equity():
    path = TSMOM_DIR / "primary_equity.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0] if len(df.columns) > 0 else df


def load_grid():
    path = TSMOM_DIR / "sensitivity_grid.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_crisis(name: str):
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    path = TSMOM_DIR / f"crisis_{safe}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ── Chart generators ──────────────────────────────────────────────────

def exhibit_1_equity_and_histogram(equity, primary):
    """Lead exhibit: equity curve + return distribution side by side."""
    if equity is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: equity curve
    ax1.plot(equity.index, equity.values, color=JPM_BLUE, linewidth=1.5,
             label="TSMOM Primary Spec")

    # Overlay BTC if available
    btc_path = TSMOM_DIR / "primary_equity.csv"
    if btc_path.exists():
        try:
            from scripts.research.common.data import load_daily_bars, compute_btc_benchmark
            panel = load_daily_bars()
            btc_eq = compute_btc_benchmark(panel)
            btc_aligned = btc_eq.reindex(equity.index).ffill().dropna()
            if len(btc_aligned) > 0:
                btc_aligned = btc_aligned / btc_aligned.iloc[0]
                ax1.plot(btc_aligned.index, btc_aligned.values, color=JPM_GRAY,
                         linewidth=1, alpha=0.7, linestyle="--", label="BTC Buy & Hold")
        except Exception:
            pass

    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value (log scale)")
    ax1.set_title("Exhibit 1a: TSMOM Equity Curve vs BTC")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Right: return distribution histogram
    ret = equity.pct_change().dropna()
    ax2.hist(ret.values, bins=100, color=JPM_BLUE, alpha=0.7, edgecolor="white",
             density=True)
    skew_val = sf(primary["metrics"].get("skewness")) if primary else sf(ret.skew())
    kurt_val = sf(primary["metrics"].get("kurtosis")) if primary else sf(ret.kurtosis())
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Density")
    ax2.set_title("Exhibit 1b: Return Distribution")
    ax2.text(0.95, 0.95, f"Skewness: {skew_val:.2f}\nKurtosis: {kurt_val:.1f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_1_equity_histogram.png", dpi=150)
    plt.close(fig)
    print("  [1/9] Equity curve + histogram")


def exhibit_2_skewness_heatmap(grid):
    """Skewness heatmap: signal type x lookback."""
    if grid.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (ax1, "skewness", "Exhibit 2a: Skewness by Signal x Lookback"),
        (ax2, "sharpe", "Exhibit 2b: Sharpe by Signal x Lookback"),
    ]:
        # Parse signal and lookback from label
        rows = []
        for _, r in grid.iterrows():
            label = str(r.get("label", ""))
            parts = label.split("_")
            if len(parts) >= 2:
                sig = parts[0]
                lb_str = parts[1].replace("d", "")
                try:
                    lb = int(lb_str)
                except ValueError:
                    continue
                rows.append({"signal": sig, "lookback": lb, metric: sf(r.get(metric, np.nan))})

        if not rows:
            continue
        df = pd.DataFrame(rows)
        pivot = df.pivot_table(index="signal", columns="lookback", values=metric, aggfunc="first")

        signals = sorted(pivot.index)
        lookbacks = sorted(pivot.columns)
        data = pivot.reindex(index=signals, columns=lookbacks).values

        vabs = max(0.3, np.nanmax(np.abs(data)))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(len(lookbacks)))
        ax.set_xticklabels(lookbacks, fontsize=8)
        ax.set_yticks(range(len(signals)))
        ax.set_yticklabels(signals, fontsize=9)
        for i in range(len(signals)):
            for j in range(len(lookbacks)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("Lookback (days)")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_2_heatmaps.png", dpi=150)
    plt.close(fig)
    print("  [2/9] Skewness + Sharpe heatmaps")


def exhibit_3_regime_analysis(primary):
    """Regime decomposition: Sharpe, skewness, time-in-market, conditional corr."""
    if primary is None:
        return
    m = primary["metrics"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    regimes = ["BULL", "BEAR", "CHOP"]
    colors_map = {"BULL": JPM_GREEN, "BEAR": JPM_RED, "CHOP": JPM_GOLD}
    x = np.arange(len(regimes))

    # Sharpe by regime
    vals = [sf(m.get("regime_sharpe_skew", {}).get(r, {}).get("sharpe", np.nan)) for r in regimes]
    axes[0].bar(x, vals, color=[colors_map[r] for r in regimes], alpha=0.85, edgecolor="white")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regimes)
    axes[0].set_title("Sharpe by Regime")

    # Skewness by regime
    vals = [sf(m.get("regime_sharpe_skew", {}).get(r, {}).get("skewness", np.nan)) for r in regimes]
    axes[1].bar(x, vals, color=[colors_map[r] for r in regimes], alpha=0.85, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regimes)
    axes[1].set_title("Skewness by Regime")

    # Time in market by regime
    vals = [sf(m.get("time_in_market_by_regime", {}).get(r, np.nan)) for r in regimes]
    axes[2].bar(x, vals, color=[colors_map[r] for r in regimes], alpha=0.85, edgecolor="white")
    axes[2].set_ylim(0, 1)
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(regimes)
    axes[2].set_title("Time in Market by Regime")

    # Conditional correlation to BTC
    vals = [sf(m.get("conditional_corr", {}).get(r, np.nan)) for r in regimes]
    axes[3].bar(x, vals, color=[colors_map[r] for r in regimes], alpha=0.85, edgecolor="white")
    axes[3].axhline(0.5, color=JPM_RED, linewidth=1, linestyle="--", alpha=0.7, label="Failure threshold")
    axes[3].set_ylim(-0.2, 1.0)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(regimes)
    axes[3].set_title("BTC Correlation by Regime")
    axes[3].legend(fontsize=7)

    fig.suptitle("Exhibit 3: Regime-Conditional Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_3_regime_analysis.png", dpi=150)
    plt.close(fig)
    print("  [3/9] Regime analysis")


def exhibit_4_drawdown(equity):
    """Drawdown comparison: TSMOM vs BTC."""
    if equity is None:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    dd = equity / equity.cummax() - 1.0
    ax.fill_between(dd.index, dd.values, 0, color=JPM_BLUE, alpha=0.4, label="TSMOM")

    try:
        from scripts.research.common.data import load_daily_bars, compute_btc_benchmark
        panel = load_daily_bars()
        btc_eq = compute_btc_benchmark(panel)
        btc_aligned = btc_eq.reindex(equity.index).ffill().dropna()
        if len(btc_aligned) > 0:
            btc_aligned = btc_aligned / btc_aligned.iloc[0]
            btc_dd = btc_aligned / btc_aligned.cummax() - 1.0
            ax.fill_between(btc_dd.index, btc_dd.values, 0, color=JPM_RED, alpha=0.25, label="BTC")
    except Exception:
        pass

    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Exhibit 4: Drawdown — TSMOM vs BTC Buy & Hold")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_4_drawdown.png", dpi=150)
    plt.close(fig)
    print("  [4/9] Drawdown comparison")


def exhibit_5_win_loss_profile(equity):
    """Distribution of trade returns showing right-tail asymmetry."""
    if equity is None:
        return

    ret = equity.pct_change().dropna()
    invested = ret[ret.abs() > 1e-8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of invested-day returns
    ax1.hist(invested.values, bins=80, color=JPM_BLUE, alpha=0.7, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=0.5)
    wins = invested[invested > 0]
    losses = invested[invested < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_rate = len(wins) / len(invested) if len(invested) > 0 else 0
    ax1.set_title("Exhibit 5a: Invested-Day Returns")
    ax1.set_xlabel("Daily Return")
    ax1.text(0.95, 0.95,
             f"Win rate: {win_rate:.1%}\nAvg win: {avg_win:.3%}\nAvg loss: {avg_loss:.3%}\n"
             f"Ratio: {abs(avg_win/avg_loss):.1f}x" if abs(avg_loss) > 1e-10 else "",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Monthly returns bar chart
    monthly = equity.resample("ME").last().pct_change().dropna()
    colors = [JPM_GREEN if v > 0 else JPM_RED for v in monthly.values]
    ax2.bar(monthly.index, monthly.values, width=25, color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_title("Exhibit 5b: Monthly Returns")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_5_win_loss.png", dpi=150)
    plt.close(fig)
    print("  [5/9] Win/loss profile")


def exhibit_6_crisis_timelines():
    """Crisis timeline: BTC price, portfolio weight, daily P&L per episode."""
    episodes = {
        "2018 Bear": "2018_Bear",
        "Mar 2020": "Mar_2020",
        "May 2021": "May_2021",
        "Nov 2022 (FTX)": "Nov_2022_FTX",
    }

    loaded = {}
    for display_name, file_key in episodes.items():
        ct = load_crisis(display_name)
        if ct is not None:
            loaded[display_name] = ct

    if not loaded:
        print("  [6/9] Crisis timelines — no data")
        return

    n = len(loaded)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n), squeeze=False)

    for row, (ep_name, ct) in enumerate(loaded.items()):
        # BTC price
        ax_btc = axes[row, 0]
        ax_btc.plot(ct.index, ct["btc_price"], color=JPM_BLUE, linewidth=1.2)
        ax_btc.set_title(f"{ep_name}: BTC Price", fontsize=9)
        ax_btc.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

        # Portfolio weight
        ax_wt = axes[row, 1]
        ax_wt.fill_between(ct.index, ct["total_weight"], 0, color=JPM_GREEN, alpha=0.5)
        ax_wt.set_ylim(0, max(0.1, ct["total_weight"].max() * 1.2))
        ax_wt.set_title(f"{ep_name}: Portfolio Weight", fontsize=9)
        ax_wt.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

        # Cumulative P&L
        ax_pnl = axes[row, 2]
        cum_pnl = (1 + ct["daily_pnl"]).cumprod() - 1
        ax_pnl.plot(ct.index, cum_pnl, color=JPM_BLUE, linewidth=1.2)
        ax_pnl.fill_between(ct.index, cum_pnl, 0,
                            where=cum_pnl >= 0, color=JPM_GREEN, alpha=0.2)
        ax_pnl.fill_between(ct.index, cum_pnl, 0,
                            where=cum_pnl < 0, color=JPM_RED, alpha=0.2)
        ax_pnl.axhline(0, color="black", linewidth=0.5)
        ax_pnl.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_pnl.set_title(f"{ep_name}: Cumulative P&L", fontsize=9)
        ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.suptitle("Exhibit 6: Crisis Timelines — BTC Price, Portfolio Weight, P&L",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CHART_DIR / "exhibit_6_crisis_timelines.png", dpi=150)
    plt.close(fig)
    print("  [6/9] Crisis timelines")


def exhibit_7_exit_comparison(grid):
    """Exit method comparison bar chart."""
    if grid.empty:
        return

    exit_rows = grid[grid["label"].str.contains("VOL_SCALED_21d_binary_", na=False)]
    if exit_rows.empty:
        print("  [7/9] Exit comparison — no data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = []
    sharpes = []
    skews = []
    for _, r in exit_rows.iterrows():
        lbl = str(r.get("label", ""))
        short_label = lbl.replace("VOL_SCALED_21d_binary_", "").replace("_vt15", "")
        labels.append(short_label)
        sharpes.append(sf(r.get("sharpe")))
        skews.append(sf(r.get("skewness")))

    x = np.arange(len(labels))
    ax1.barh(x, sharpes, color=JPM_BLUE, alpha=0.85, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_yticks(x)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_title("Exhibit 7a: Sharpe by Exit Method")

    ax2.barh(x, skews, color=JPM_GREEN, alpha=0.85, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Skewness")
    ax2.set_title("Exhibit 7b: Skewness by Exit Method")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_7_exit_comparison.png", dpi=150)
    plt.close(fig)
    print("  [7/9] Exit method comparison")


def exhibit_8_sharpe_skew_pareto(grid):
    """Sharpe x Skewness scatter with Pareto frontier."""
    if grid.empty:
        return

    valid = grid.dropna(subset=["sharpe", "skewness"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    sharpes = valid["sharpe"].values
    skews = valid["skewness"].values

    ax.scatter(sharpes, skews, color=JPM_LIGHT_BLUE, alpha=0.5, s=30, edgecolor="white")

    # Pareto frontier (non-dominated in Sharpe AND skewness)
    pareto_mask = np.zeros(len(sharpes), dtype=bool)
    for i in range(len(sharpes)):
        dominated = False
        for j in range(len(sharpes)):
            if i != j and sharpes[j] >= sharpes[i] and skews[j] >= skews[i]:
                if sharpes[j] > sharpes[i] or skews[j] > skews[i]:
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True

    if pareto_mask.any():
        pareto_pts = valid[pareto_mask].sort_values("sharpe")
        ax.scatter(pareto_pts["sharpe"], pareto_pts["skewness"],
                   color=JPM_RED, s=60, zorder=5, edgecolor="white", linewidth=1.5,
                   label="Pareto frontier")
        ax.plot(pareto_pts["sharpe"], pareto_pts["skewness"],
                color=JPM_RED, linewidth=1, alpha=0.5, linestyle="--")

    # Highlight primary spec
    primary_row = valid[valid["label"].str.contains("VOL_SCALED_21d_binary_signal_reversal_vt15", na=False)]
    if not primary_row.empty:
        ax.scatter(primary_row["sharpe"].values[0], primary_row["skewness"].values[0],
                   color=JPM_GREEN, s=120, zorder=6, marker="*", edgecolor="black",
                   linewidth=1, label="Primary spec")

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axhline(0.5, color=JPM_GREEN, linewidth=0.8, linestyle=":", alpha=0.5, label="Skew target (0.5)")

    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Skewness")
    ax.set_title("Exhibit 8: Sharpe x Skewness — All Configurations")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_8_sharpe_skew_pareto.png", dpi=150)
    plt.close(fig)
    print("  [8/9] Sharpe x Skewness Pareto")


def exhibit_9_vol_target_frontier(grid):
    """Vol target frontier: Sharpe vs skewness at different vol targets."""
    if grid.empty:
        return

    vt_rows = grid[grid["label"].str.contains("VOL_SCALED_21d_binary_signal_reversal_vt", na=False)]
    if len(vt_rows) < 2:
        print("  [9/9] Vol target frontier — insufficient data")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, r in vt_rows.iterrows():
        lbl = str(r.get("label", ""))
        vt_str = lbl.split("vt")[-1] if "vt" in lbl else "?"
        sharpe = sf(r.get("sharpe"))
        skew = sf(r.get("skewness"))
        maxdd = sf(r.get("max_dd"))

        ax.scatter(sharpe, skew, s=150, color=JPM_BLUE, zorder=5, edgecolor="white")
        ax.annotate(f"VT={vt_str}%\nDD={maxdd:.0%}" if not np.isnan(maxdd) else f"VT={vt_str}%",
                    (sharpe, skew), textcoords="offset points", xytext=(10, 5),
                    fontsize=8, ha="left")

    ax.axhline(0.5, color=JPM_GREEN, linewidth=0.8, linestyle=":", alpha=0.5, label="Skew target")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Skewness")
    ax.set_title("Exhibit 9: Vol Target Sensitivity — Sharpe vs Skewness Frontier")
    ax.legend()

    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_9_vol_target_frontier.png", dpi=150)
    plt.close(fig)
    print("  [9/9] Vol target frontier")


# ── Markdown report ───────────────────────────────────────────────────

def build_markdown(primary, grid) -> str:
    m = primary["metrics"] if primary else {}
    config = primary["config"] if primary else {}

    md = f"""---
# Time-Series Momentum as a Long Convexity Engine for Crypto

### Pre-Registered TSMOM Experiment

**NRT Research** | {datetime.now().strftime('%B %d, %Y')}

---

> *We test whether time-series momentum (TSMOM) can serve as the entry/exit timing layer
> for a long convexity compounding engine in cryptocurrency markets.  Unlike our prior
> cross-sectional factor study (1,121 signals, median Sharpe -0.97), TSMOM evaluates
> each asset against its own history — "is BTC trending?" not "is BTC trending more than
> ETH?"  The default position is 100% cash; we enter long only when a trend signal fires
> and exit when it reverses.  The target payoff profile is a portfolio of synthetic call
> options: bounded left tail, fat right tail, positive skewness.*

---

## 1. Pre-Registered Primary Specification

| Parameter | Value |
|---|---|
| Signal | {config.get('signal', 'VOL_SCALED')} |
| Lookback | {config.get('lookback', 21)} days |
| Sizing | {config.get('sizing', 'binary')} (equal risk per position) |
| Exit | {config.get('exit', 'signal_reversal')} |
| Vol target | {config.get('vol_target', 0.15):.0%} annualised |
| Max weight | {config.get('max_weight', 0.20):.0%} per asset (excess → cash) |
| Costs | 20 bps round-trip |
| Execution lag | 1 day |

## 2. Primary Specification Results

| Metric | Value |
|---|---|
| **Skewness** | **{sf(m.get('skewness')):.3f}** |
| Sharpe | {sf(m.get('sharpe')):.3f} |
| CAGR | {sf(m.get('cagr')):.1%} |
| Max Drawdown | {sf(m.get('max_dd')):.1%} |
| Sortino | {sf(m.get('sortino')):.3f} |
| Calmar | {sf(m.get('calmar')):.3f} |
| Hit Rate | {sf(m.get('hit_rate')):.1%} |
| Win/Loss Ratio | {sf(m.get('win_loss_ratio')):.2f} |
| Time in Market | {sf(m.get('time_in_market')):.1%} |
| Avg Turnover | {sf(m.get('avg_turnover')):.4f} |
| Participation (portfolio) | {sf(m.get('participation_portfolio')):.1%} |
| Participation (per-asset) | {sf(m.get('participation_per_asset')):.1%} |

### 2.1 Regime-Conditional Analysis

| Regime | Sharpe | Skewness | Time in Market | BTC Correlation |
|---|---|---|---|---|"""

    for r in ["BULL", "BEAR", "CHOP"]:
        rs = m.get("regime_sharpe_skew", {}).get(r, {})
        tim = m.get("time_in_market_by_regime", {}).get(r, np.nan)
        cc = m.get("conditional_corr", {}).get(r, np.nan)
        md += f"\n| {r} | {sf(rs.get('sharpe')):.2f} | {sf(rs.get('skewness')):.2f} | {sf(tim):.1%} | {sf(cc):.3f} |"

    cond_corr = m.get("conditional_corr", {})
    bear_corr = cond_corr.get("BEAR", 99) if isinstance(cond_corr, dict) else 99

    md += f"""

### 2.2 Pass/Fail Assessment

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| Skewness > 0 | > 0 | {sf(m.get('skewness')):.3f} | {'PASS' if sf(m.get('skewness', -1)) > 0 else 'FAIL'} |
| Sharpe > 0 | > 0 | {sf(m.get('sharpe')):.3f} | {'PASS' if sf(m.get('sharpe', -1)) > 0 else 'FAIL'} |
| Max DD > -30% | > -30% | {sf(m.get('max_dd')):.1%} | {'PASS' if sf(m.get('max_dd', -1)) > -0.30 else 'FAIL'} |
| BEAR BTC corr < 0.5 | < 0.5 | {sf(bear_corr):.3f} | {'PASS' if sf(bear_corr) < 0.5 else 'FAIL'} |
| Participation > 20% | > 20% | {sf(m.get('participation_per_asset', 0)):.1%} | {'PASS' if sf(m.get('participation_per_asset', 0)) > 0.20 else 'FAIL'} |

## 3. Exhibits

![Exhibit 1](../../artifacts/research/tsmom/report_charts/exhibit_1_equity_histogram.png)

![Exhibit 2](../../artifacts/research/tsmom/report_charts/exhibit_2_heatmaps.png)

![Exhibit 3](../../artifacts/research/tsmom/report_charts/exhibit_3_regime_analysis.png)

![Exhibit 4](../../artifacts/research/tsmom/report_charts/exhibit_4_drawdown.png)

![Exhibit 5](../../artifacts/research/tsmom/report_charts/exhibit_5_win_loss.png)

![Exhibit 6](../../artifacts/research/tsmom/report_charts/exhibit_6_crisis_timelines.png)

![Exhibit 7](../../artifacts/research/tsmom/report_charts/exhibit_7_exit_comparison.png)

![Exhibit 8](../../artifacts/research/tsmom/report_charts/exhibit_8_sharpe_skew_pareto.png)

![Exhibit 9](../../artifacts/research/tsmom/report_charts/exhibit_9_vol_target_frontier.png)

## 4. Sensitivity Grid Summary
"""

    if not grid.empty:
        valid = grid.dropna(subset=["sharpe", "skewness"])
        if len(valid) > 0:
            md += f"\nTotal configurations tested: **{len(valid)}**\n"
            md += f"\nMedian Sharpe: {valid['sharpe'].median():.3f}\n"
            md += f"\nMedian Skewness: {valid['skewness'].median():.3f}\n"

            md += "\n### Top 5 by Skewness\n\n"
            md += "| Config | Skewness | Sharpe | CAGR | Max DD |\n"
            md += "|---|---|---|---|---|\n"
            for _, r in valid.nlargest(5, "skewness").iterrows():
                md += (f"| {r.get('label', '?')} | {sf(r.get('skewness')):.2f} | "
                       f"{sf(r.get('sharpe')):.2f} | {sf(r.get('cagr')):.1%} | "
                       f"{sf(r.get('max_dd')):.1%} |\n")

            md += "\n### Top 5 by Sharpe\n\n"
            md += "| Config | Sharpe | Skewness | CAGR | Max DD |\n"
            md += "|---|---|---|---|---|\n"
            for _, r in valid.nlargest(5, "sharpe").iterrows():
                md += (f"| {r.get('label', '?')} | {sf(r.get('sharpe')):.2f} | "
                       f"{sf(r.get('skewness')):.2f} | {sf(r.get('cagr')):.1%} | "
                       f"{sf(r.get('max_dd')):.1%} |\n")

    md += f"""
---

*Data: Coinbase Advanced spot OHLCV. Universe: point-in-time, $500K ADV filter,
90-day minimum history. Costs: 20 bps round-trip. Execution: 1-day lag.
Annualisation: 365 days. Period: 2017–2025.*
"""
    return md


# ── PDF Generation ────────────────────────────────────────────────────

def generate_pdf(charts_dir: Path, output_path: Path, primary: dict | None):
    """Generate PDF report using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
        )
        from reportlab.lib import colors
    except ImportError:
        print("[report] reportlab not installed, skipping PDF generation")
        return

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )

    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("ReportTitle", parent=ss["Title"], fontSize=20,
                           spaceAfter=6, textColor=HexColor(JPM_BLUE), fontName="Times-Bold"))
    ss.add(ParagraphStyle("ReportSubtitle", parent=ss["Normal"], fontSize=12,
                           spaceAfter=12, textColor=HexColor(JPM_GRAY), fontName="Times-Italic"))
    ss.add(ParagraphStyle("SectionHead", parent=ss["Heading1"], fontSize=14,
                           spaceBefore=18, spaceAfter=8, textColor=HexColor(JPM_BLUE),
                           fontName="Times-Bold"))
    ss.add(ParagraphStyle("BodyText2", parent=ss["Normal"], fontSize=9,
                           leading=13, fontName="Times-Roman", spaceAfter=6))
    ss.add(ParagraphStyle("Abstract", parent=ss["Normal"], fontSize=9,
                           leading=13, fontName="Times-Italic", leftIndent=20,
                           rightIndent=20, spaceAfter=12, textColor=HexColor("#333333")))
    ss.add(ParagraphStyle("Caption", parent=ss["Normal"], fontSize=8,
                           fontName="Times-Italic", textColor=HexColor(JPM_GRAY),
                           spaceBefore=4, spaceAfter=12, alignment=1))

    elements = []

    # Title
    elements.append(Spacer(1, 1.5 * inch))
    elements.append(Paragraph(
        "Time-Series Momentum as a<br/>Long Convexity Engine for Crypto",
        ss["ReportTitle"],
    ))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Pre-Registered TSMOM Experiment", ss["ReportSubtitle"]))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        f"NRT Research &nbsp;|&nbsp; {datetime.now().strftime('%B %d, %Y')}",
        ss["BodyText2"],
    ))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        "We test whether time-series momentum can serve as the entry/exit timing layer "
        "for a long convexity compounding engine in cryptocurrency markets.  The default "
        "position is 100% cash; we enter long only when a trend signal fires and exit "
        "when it reverses.  The target payoff is bounded downside, unbounded upside, "
        "positive skewness.",
        ss["Abstract"],
    ))
    elements.append(PageBreak())

    def add_img(name, caption=None, w=6.0 * inch):
        p = charts_dir / name
        if p.exists():
            elements.append(Image(str(p), width=w, height=w * 0.55))
            if caption:
                elements.append(Paragraph(caption, ss["Caption"]))
            elements.append(Spacer(1, 6))

    # Primary results summary
    if primary:
        m = primary["metrics"]
        elements.append(Paragraph("Primary Specification Results", ss["SectionHead"]))
        elements.append(Paragraph(
            f"Skewness: <b>{sf(m.get('skewness')):.3f}</b> &nbsp;|&nbsp; "
            f"Sharpe: {sf(m.get('sharpe')):.3f} &nbsp;|&nbsp; "
            f"CAGR: {sf(m.get('cagr')):.1%} &nbsp;|&nbsp; "
            f"Max DD: {sf(m.get('max_dd')):.1%} &nbsp;|&nbsp; "
            f"Win/Loss: {sf(m.get('win_loss_ratio')):.1f}x &nbsp;|&nbsp; "
            f"Time in Market: {sf(m.get('time_in_market')):.0%}",
            ss["BodyText2"],
        ))
        elements.append(Spacer(1, 12))

    # Exhibits
    add_img("exhibit_1_equity_histogram.png",
            "Exhibit 1: Equity curve (left) and return distribution (right). "
            "Positive skewness = fat right tail = convex payoff.")
    add_img("exhibit_2_heatmaps.png",
            "Exhibit 2: Skewness and Sharpe heatmaps across signal types and lookbacks.")
    elements.append(PageBreak())
    add_img("exhibit_3_regime_analysis.png",
            "Exhibit 3: Regime-conditional analysis — Sharpe, skewness, time in market, "
            "and BTC correlation by regime.")
    add_img("exhibit_4_drawdown.png",
            "Exhibit 4: Drawdown comparison — TSMOM vs BTC buy-and-hold.")
    elements.append(PageBreak())
    add_img("exhibit_5_win_loss.png",
            "Exhibit 5: Win/loss profile — daily return distribution and monthly returns.")
    add_img("exhibit_6_crisis_timelines.png",
            "Exhibit 6: Crisis timelines — BTC price, portfolio weight, and P&L "
            "during each bear episode.")
    elements.append(PageBreak())
    add_img("exhibit_7_exit_comparison.png",
            "Exhibit 7: Exit method comparison — signal reversal vs trailing stop variants.")
    add_img("exhibit_8_sharpe_skew_pareto.png",
            "Exhibit 8: Sharpe x Skewness scatter — all configurations with Pareto frontier.")
    add_img("exhibit_9_vol_target_frontier.png",
            "Exhibit 9: Vol target sensitivity — Sharpe vs skewness frontier.")

    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        "<i>Data: Coinbase Advanced spot OHLCV. Universe: point-in-time, $500K ADV, "
        "90-day minimum history. Costs: 20 bps round-trip. Lag: 1 day. ANN=365. "
        "Period: 2017–2025.</i>",
        ss["Caption"],
    ))

    doc.build(elements)
    print(f"[report] PDF generated: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("[report] Loading artifacts ...")
    primary = load_primary()
    equity = load_primary_equity()
    grid = load_grid()

    if primary is None:
        print("[report] No primary_spec_results.json found — run run_tsmom.py first")
        return

    print(f"[report] Primary spec loaded.  Grid: {len(grid)} rows")
    print("[report] Generating charts ...")

    exhibit_1_equity_and_histogram(equity, primary)
    exhibit_2_skewness_heatmap(grid)
    exhibit_3_regime_analysis(primary)
    exhibit_4_drawdown(equity)
    exhibit_5_win_loss_profile(equity)
    exhibit_6_crisis_timelines()
    exhibit_7_exit_comparison(grid)
    exhibit_8_sharpe_skew_pareto(grid)
    exhibit_9_vol_target_frontier(grid)

    print("[report] Building markdown report ...")
    md = build_markdown(primary, grid)
    md_path = DOCS_DIR / "tsmom_report.md"
    md_path.write_text(md)
    print(f"[report] Markdown saved to {md_path}")

    print("[report] Generating PDF ...")
    pdf_path = TSMOM_DIR / "tsmom_report.pdf"
    generate_pdf(CHART_DIR, pdf_path, primary)

    print("[report] Done.")
    return str(pdf_path)


if __name__ == "__main__":
    main()
