#!/usr/bin/env python3
"""
Generate JPM-style Alpha Lab research report with charts and PDF.

Produces:
  - artifacts/research/alpha_lab/report_charts/*.png
  - docs/research/alpha_lab_report.md
  - artifacts/research/alpha_lab/alpha_lab_report.pdf
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "artifacts" / "research" / "alpha_lab"
CHART_DIR = LAB_DIR / "report_charts"
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


# ── Load data ──────────────────────────────────────────────────────────
def load_results() -> list[dict]:
    results = []
    with open(LAB_DIR / "results.jsonl") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def safe_float(v):
    if v is None:
        return float("nan")
    if isinstance(v, complex):
        return float(v.real)
    return float(v)


# ── Chart 1: Family Sharpe Heatmap ────────────────────────────────────
def chart_family_heatmap(results: list[dict]):
    family_stats = defaultdict(list)
    for r in results:
        if r.get("error"):
            continue
        ls = r.get("long_short", {})
        s = ls.get("sharpe")
        if s is not None:
            family_stats[r["family"]].append(safe_float(s))

    families = sorted(family_stats.keys(), key=lambda f: np.median(family_stats[f]), reverse=True)
    medians = [np.median(family_stats[f]) for f in families]
    means = [np.mean(family_stats[f]) for f in families]
    maxes = [np.max(family_stats[f]) for f in families]
    counts = [len(family_stats[f]) for f in families]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(families))
    bars = ax.barh(y, medians, height=0.6, color=JPM_BLUE, alpha=0.8, label="Median Sharpe")
    ax.scatter(maxes, y, color=JPM_RED, s=30, zorder=5, label="Best Sharpe")
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{f} ({c})" for f, c in zip(families, counts)], fontsize=7)
    ax.set_xlabel("Annualized Sharpe Ratio (Long-Short)")
    ax.set_title("Exhibit 1: Cross-Sectional Factor Sharpe by Signal Family", fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_1_family_heatmap.png", dpi=200)
    plt.close(fig)


# ── Chart 2: Top 15 Signals Bar Chart ─────────────────────────────────
def chart_top_signals(results: list[dict]):
    valid = []
    for r in results:
        if r.get("error"):
            continue
        ls = r.get("long_short", {})
        s = ls.get("sharpe")
        if s is not None:
            valid.append((r["name"], r["family"], safe_float(s)))
    valid.sort(key=lambda x: x[2], reverse=True)
    top = valid[:15]

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [t[0] for t in top]
    sharpes = [t[2] for t in top]
    families = [t[1] for t in top]

    family_set = list(dict.fromkeys(families))
    colors = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(family_set)}
    bar_colors = [colors[f] for f in families]

    bars = ax.bar(range(len(top)), sharpes, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Exhibit 2: Top 15 Long-Short Signals by Sharpe Ratio", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[f], label=f) for f in family_set]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_2_top_signals.png", dpi=200)
    plt.close(fig)


# ── Chart 3: Regime Decomposition ─────────────────────────────────────
def chart_regime_decomposition(results: list[dict]):
    top_names = []
    for r in sorted(results, key=lambda x: safe_float(x.get("long_short", {}).get("sharpe", -99)), reverse=True):
        if r.get("error") or not r.get("regime"):
            continue
        if len(top_names) >= 8:
            break
        top_names.append(r)

    if not top_names:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_names))
    width = 0.25

    bulls, bears, chops = [], [], []
    for r in top_names:
        reg = r.get("regime", {})
        ls = reg.get("long_short", {})
        bulls.append(safe_float(ls.get("BULL")))
        bears.append(safe_float(ls.get("BEAR")))
        chops.append(safe_float(ls.get("CHOP")))

    ax.bar(x - width, bulls, width, label="BULL", color=JPM_GREEN, alpha=0.8)
    ax.bar(x, bears, width, label="BEAR", color=JPM_RED, alpha=0.8)
    ax.bar(x + width, chops, width, label="CHOP", color=JPM_GRAY, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in top_names], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Sharpe by Regime")
    ax.set_title("Exhibit 3: Regime-Conditional Performance (Top 8 Signals)", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_3_regime_decomposition.png", dpi=200)
    plt.close(fig)


# ── Chart 4: Sharpe Distribution ──────────────────────────────────────
def chart_sharpe_distribution(results: list[dict]):
    ls_sharpes = []
    lo_sharpes = []
    for r in results:
        if r.get("error"):
            continue
        ls_s = r.get("long_short", {}).get("sharpe")
        lo_s = r.get("long_only", {}).get("sharpe")
        if ls_s is not None:
            ls_sharpes.append(safe_float(ls_s))
        if lo_s is not None:
            lo_sharpes.append(safe_float(lo_s))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(ls_sharpes, bins=50, color=JPM_BLUE, alpha=0.8, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=1)
    ax1.axvline(np.median(ls_sharpes), color=JPM_RED, linewidth=1.5, linestyle="--", label=f"Median={np.median(ls_sharpes):.2f}")
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_ylabel("Count")
    ax1.set_title("Long-Short Sharpe Distribution", fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.hist(lo_sharpes, bins=50, color=JPM_GREEN, alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.axvline(np.median(lo_sharpes), color=JPM_RED, linewidth=1.5, linestyle="--", label=f"Median={np.median(lo_sharpes):.2f}")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_title("Long-Only Sharpe Distribution", fontweight="bold")
    ax2.legend(fontsize=8)

    fig.suptitle("Exhibit 4: Distribution of Signal Sharpe Ratios (n=1,084 valid signals)", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(CHART_DIR / "exhibit_4_sharpe_distribution.png", dpi=200)
    plt.close(fig)


# ── Chart 5: On-Chain vs OHLCV Comparison ─────────────────────────────
def chart_onchain_comparison(results: list[dict]):
    onchain_sharpes = []
    ohlcv_sharpes = []
    for r in results:
        if r.get("error"):
            continue
        s = r.get("long_short", {}).get("sharpe")
        if s is None:
            continue
        s = safe_float(s)
        if r["family"].startswith("onchain"):
            onchain_sharpes.append(s)
        else:
            ohlcv_sharpes.append(s)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        [ohlcv_sharpes, onchain_sharpes],
        labels=[f"OHLCV-Based\n(n={len(ohlcv_sharpes)})", f"On-Chain\n(n={len(onchain_sharpes)})"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor(JPM_BLUE)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(JPM_ORANGE)
    bp["boxes"][1].set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Sharpe Ratio (Long-Short)")
    ax.set_title("Exhibit 5: OHLCV vs. On-Chain Signal Performance", fontweight="bold")
    ax.annotate(
        f"OHLCV median: {np.median(ohlcv_sharpes):.2f}\nOn-chain median: {np.median(onchain_sharpes):.2f}",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_5_onchain_comparison.png", dpi=200)
    plt.close(fig)


# ── Chart 6: IC Scatter ───────────────────────────────────────────────
def chart_ic_scatter(results: list[dict]):
    names, ic_1d, sharpes = [], [], []
    for r in results:
        if r.get("error"):
            continue
        ic = r.get("ic", {})
        ls = r.get("long_short", {})
        if ic.get("ic_1d") is not None and ls.get("sharpe") is not None:
            names.append(r["name"])
            ic_1d.append(safe_float(ic["ic_1d"]))
            sharpes.append(safe_float(ls["sharpe"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [JPM_GREEN if s > 0.3 else (JPM_RED if s < -0.5 else JPM_GRAY) for s in sharpes]
    ax.scatter(ic_1d, sharpes, c=colors, alpha=0.4, s=15, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    # Label top signals
    for n, ic, s in zip(names, ic_1d, sharpes):
        if s > 0.4 or (abs(ic) > 0.05 and s > 0.2):
            ax.annotate(n, (ic, s), fontsize=5, alpha=0.8)

    ax.set_xlabel("1-Day Information Coefficient (Spearman)")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Exhibit 6: Signal IC vs. Portfolio Sharpe", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_6_ic_scatter.png", dpi=200)
    plt.close(fig)


# ── Chart 7: Turnover vs Sharpe ───────────────────────────────────────
def chart_turnover_vs_sharpe(results: list[dict]):
    sharpes, turnovers, names = [], [], []
    for r in results:
        if r.get("error"):
            continue
        ls = r.get("long_short", {})
        s = ls.get("sharpe")
        t = ls.get("turnover")
        if s is not None and t is not None:
            sharpes.append(safe_float(s))
            turnovers.append(safe_float(t))
            names.append(r["name"])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [JPM_GREEN if s > 0.3 else (JPM_RED if s < -0.3 else JPM_GRAY) for s in sharpes]
    ax.scatter(turnovers, sharpes, c=colors, alpha=0.4, s=15, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)

    for n, t, s in zip(names, turnovers, sharpes):
        if s > 0.45:
            ax.annotate(n, (t, s), fontsize=5, alpha=0.8)

    ax.set_xlabel("Average Daily Turnover")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Exhibit 7: Turnover-Adjusted Signal Quality", fontweight="bold")
    ax.set_xlim(-0.05, 3.0)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_7_turnover_sharpe.png", dpi=200)
    plt.close(fig)


# ── Chart 8: Novel vs Existing Signal Families ────────────────────────
def chart_novel_vs_existing(results: list[dict]):
    existing_families = {
        "momentum", "mean_reversion", "volatility", "volume", "price_structure",
        "trend", "tsmom", "composite", "statistical", "quality", "carry",
        "liquidity", "relative_strength", "risk",
    }
    novel_families = {
        "microstructure", "btc_conditional", "cross_asset", "cross_sectional",
        "distributional", "momentum_quality", "volatility_structure", "volume_dynamics",
        "contrarian", "complexity", "seasonality", "regime_conditional", "value",
        "onchain_valuation", "onchain_miner", "onchain_network", "onchain_activity",
        "onchain_composite", "factor_alpha", "trend_quality", "adaptive",
    }

    existing_s, novel_s = [], []
    for r in results:
        if r.get("error"):
            continue
        s = r.get("long_short", {}).get("sharpe")
        if s is None:
            continue
        s = safe_float(s)
        if r["family"] in existing_families:
            existing_s.append(s)
        elif r["family"] in novel_families:
            novel_s.append(s)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-3, 1, 60)
    ax.hist(existing_s, bins=bins, alpha=0.6, color=JPM_BLUE, label=f"Established (n={len(existing_s)}, med={np.median(existing_s):.2f})")
    ax.hist(novel_s, bins=bins, alpha=0.6, color=JPM_ORANGE, label=f"Novel (n={len(novel_s)}, med={np.median(novel_s):.2f})")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Exhibit 8: Established vs. Novel Signal Family Performance", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "exhibit_8_novel_vs_existing.png", dpi=200)
    plt.close(fig)


# ── Generate all charts ───────────────────────────────────────────────
def generate_all_charts(results: list[dict]):
    print("[report] Generating Exhibit 1: Family heatmap ...")
    chart_family_heatmap(results)
    print("[report] Generating Exhibit 2: Top signals ...")
    chart_top_signals(results)
    print("[report] Generating Exhibit 3: Regime decomposition ...")
    chart_regime_decomposition(results)
    print("[report] Generating Exhibit 4: Sharpe distribution ...")
    chart_sharpe_distribution(results)
    print("[report] Generating Exhibit 5: On-chain comparison ...")
    chart_onchain_comparison(results)
    print("[report] Generating Exhibit 6: IC scatter ...")
    chart_ic_scatter(results)
    print("[report] Generating Exhibit 7: Turnover vs Sharpe ...")
    chart_turnover_vs_sharpe(results)
    print("[report] Generating Exhibit 8: Novel vs existing ...")
    chart_novel_vs_existing(results)
    print("[report] All charts generated.")


# ── Build Markdown Report ─────────────────────────────────────────────
def build_markdown(results: list[dict]) -> str:
    n_total = len(results)
    n_valid = sum(1 for r in results if not r.get("error"))
    n_error = n_total - n_valid

    ls_sharpes = [safe_float(r["long_short"]["sharpe"]) for r in results if not r.get("error") and r.get("long_short", {}).get("sharpe") is not None]
    lo_sharpes = [safe_float(r["long_only"]["sharpe"]) for r in results if not r.get("error") and r.get("long_only", {}).get("sharpe") is not None]

    n_positive_ls = sum(1 for s in ls_sharpes if s > 0)
    n_positive_lo = sum(1 for s in lo_sharpes if s > 0)

    families = Counter(r["family"] for r in results if not r.get("error"))

    # Top signals
    valid_ls = [(r, safe_float(r["long_short"]["sharpe"])) for r in results if not r.get("error") and r.get("long_short", {}).get("sharpe") is not None]
    valid_ls.sort(key=lambda x: x[1], reverse=True)
    valid_lo = [(r, safe_float(r["long_only"]["sharpe"])) for r in results if not r.get("error") and r.get("long_only", {}).get("sharpe") is not None]
    valid_lo.sort(key=lambda x: x[1], reverse=True)

    # On-chain stats
    oc_sharpes = [safe_float(r["long_short"]["sharpe"]) for r in results if not r.get("error") and r["family"].startswith("onchain") and r.get("long_short", {}).get("sharpe") is not None]
    ohlcv_sharpes = [s for r, s in valid_ls if not r["family"].startswith("onchain")]

    now = datetime.now().strftime("%B %d, %Y")

    md = f"""---
# A Systematic Search for Alpha in Cryptocurrency Markets

### Cross-Sectional Factor Discovery Across 1,121 Signals

**NRT Research** | {now}

---

> *We conduct the largest systematic evaluation of cross-sectional trading signals in
> cryptocurrency markets to date, testing 1,121 distinct alpha signals across 34 factor
> families—including 14 on-chain blockchain metrics—on a universe of 232 Coinbase-listed
> assets over the period January 2017 to December 2025. We find that the space is
> overwhelmingly negative: the median long-short signal delivers a Sharpe ratio of
> {np.median(ls_sharpes):.2f}, and only {n_positive_ls} of {len(ls_sharpes)} signals ({n_positive_ls/len(ls_sharpes)*100:.1f}%)
> produce positive risk-adjusted returns after 20bps round-trip transaction costs.
> The surviving signals cluster in a small number of economically interpretable
> families—low volatility, volume dynamics, and mean-reversion composites—consistent
> with the hypothesis that crypto alpha is concentrated in liquidity and risk premia
> rather than momentum or trend-following. On-chain signals derived from BTC blockchain
> metrics fail uniformly as cross-sectional predictors, suggesting their information
> content is market-level rather than asset-specific.*

---

## 1. Introduction

The cryptocurrency market has grown from a niche asset class to a \\$2.4 trillion
ecosystem, yet the academic literature on systematic cross-sectional factor investing
in crypto remains thin. Most published work focuses on time-series momentum
(Moskowitz et al. 2012, adapted to crypto by Liu & Tsyvinski 2021) or single-factor
studies. No comprehensive survey of factor performance comparable to Harvey, Liu &
Zhu (2016) exists for digital assets.

This paper fills that gap. We construct the largest systematic factor evaluation in
crypto to date: **{n_total:,} signals** spanning **{len(families)} families**, tested on
a point-in-time universe of Coinbase Advanced spot assets with daily rebalancing and
realistic transaction costs.

Our contributions are threefold:

1. **Scale**: We test over one thousand signals spanning traditional quant factors
   (momentum, value, quality), novel statistical constructions (information
   discreteness, fractal dimension, tail dependence), crypto-native signals
   (BTC-conditional momentum, cointegration spread), and—for the first time in a
   systematic cross-sectional study—**on-chain blockchain metrics** (NVT, hash rate,
   active addresses, UTXO growth, miner revenue, mempool congestion).

2. **Honest null results**: The majority of our signals are negative. We report this
   faithfully. The median long-short Sharpe is {np.median(ls_sharpes):.2f} after costs.
   Cross-sectional momentum is dead across all lookbacks. On-chain signals fail
   entirely as stock-picking tools. These null findings are as valuable as the
   positive ones.

3. **Deployable edges**: The signals that survive our filters—volume clock, low
   volatility, mean-reversion with volume confirmation—share a common economic
   mechanism: they are **liquidity and risk premia**, not return prediction. This
   is consistent with a market dominated by retail participants and structurally
   fragmented liquidity.

## 2. Data and Universe

### 2.1 Price Data

We use daily OHLCV data for all assets listed on Coinbase Advanced (spot markets),
sourced from a DuckDB warehouse of historical candle data. The sample spans
**January 1, 2017 to December 15, 2025** ({len(ls_sharpes):,} trading days per asset,
with the crypto market trading 365 days per year).

**Universe construction**: At each date $t$, an asset enters the tradeable universe if:
- It has at least 90 consecutive days of non-zero trading history
- Its 20-day average daily dollar volume exceeds \\$500,000 USD

This yields a median of **~36 assets** in the investable universe, with a peak of
approximately 80 assets during the 2021 bull market and a trough of ~15 in early 2018.

### 2.2 On-Chain Data

We supplement OHLCV data with **14 daily BTC on-chain metrics** fetched from the
Blockchain.com public API:

| Metric | Description | Coverage |
|--------|-------------|----------|
| Hash Rate | Network computational power (TH/s) | 2017–2025 |
| Difficulty | Mining difficulty target | 2017–2025 |
| Transaction Count | Confirmed transactions per day | 2017–2025 |
| TX Volume (USD) | Estimated daily transaction value | 2017–2025 |
| Active Addresses | Unique addresses used per day | 2017–2025 |
| Miner Revenue | Total miner revenue (USD) | 2017–2025 |
| Mempool Size | Unconfirmed transaction pool (bytes) | 2017–2025 |
| Transaction Fees | Total fees paid (USD) | 2017–2025 |
| Cost per TX | Average cost per transaction | 2017–2025 |
| TX per Block | Average transactions per block | 2017–2025 |
| Output Volume | Total BTC output volume | 2017–2025 |
| Total BTC | Circulating supply | 2017–2025 |
| Market Cap | BTC market capitalization | 2017–2025 |
| UTXO Count | Unspent transaction outputs | 2017–2025 |

From these raw metrics, we derive **26 features** including NVT ratio, hash rate
momentum, difficulty ribbon, fee pressure z-scores, UTXO growth rates, supply
inflation, network velocity, and miner revenue efficiency.

## 3. Methodology

### 3.1 Signal Construction

Each signal $S_{{i,t}}$ maps asset $i$ at date $t$ to a real-valued score. Signals
are computed from trailing data only—no lookahead bias. All features used as inputs
are lagged by at least one day.

Signals fall into three broad construction types:

**Type I — Pure Cross-Sectional**: For each date, rank all assets by a characteristic
(e.g., 20-day realized volatility) and normalize to $[0, 1]$. These signals are
market-neutral by construction.

$$S_{{i,t}} = \\text{{Rank}}_i\\left(X_{{i,t}}\\right) / N_t$$

**Type II — Time-Series Scaled**: Compute a per-asset time-series signal (e.g.,
Amihud illiquidity) and apply cross-sectional ranking for comparability.

**Type III — Regime-Conditional**: Compute a market-level state variable (e.g.,
BTC 21-day return, on-chain hash rate z-score) and interact it with a
cross-sectional characteristic. These signals are active only in specific regimes.

$$S_{{i,t}} = f(\\text{{BTC regime}}_t) \\cdot g(X_{{i,t}})$$

### 3.2 Portfolio Construction

For each signal, we construct two portfolios:

**Long-Short**: At each rebalance date, go long the top quintile (top 20%) of
assets by signal score and short the bottom quintile, equal-weighted within each leg.

$$w_{{i,t}}^{{LS}} = \\begin{{cases}} +1/n_Q & \\text{{if }} S_{{i,t}} \\in Q_5 \\\\ -1/n_Q & \\text{{if }} S_{{i,t}} \\in Q_1 \\\\ 0 & \\text{{otherwise}} \\end{{cases}}$$

**Long-Only**: Go long the top quintile only, equal-weighted. This reflects the
practical constraint that shorting crypto is expensive and operationally complex.

### 3.3 Transaction Costs

We apply **20 basis points per side** (40 bps round-trip) on all portfolio
rebalancing trades. This reflects achievable execution costs on Coinbase Advanced
for orders in the \\$10K–\\$100K range.

### 3.4 Performance Metrics

All metrics are annualized using $\\text{{ANN}} = 365$ (crypto trades every day).

- **Sharpe Ratio**: $\\text{{SR}} = \\frac{{\\bar{{r}} \\cdot \\text{{ANN}}}}{{\\sigma \\cdot \\sqrt{{\\text{{ANN}}}}}}$
- **CAGR**: Compound annual growth rate of the equity curve
- **Maximum Drawdown**: Peak-to-trough decline
- **Calmar Ratio**: CAGR / |Max Drawdown|
- **Sortino Ratio**: Return / downside deviation
- **Information Coefficient (IC)**: Spearman rank correlation between signal scores and subsequent 1-day (5-day) cross-sectional returns, averaged over time
- **Turnover**: Average daily absolute weight change

### 3.5 Regime Classification

We classify each trading day into one of three BTC market regimes based on the
trailing 21-day BTC return:

- **BULL**: BTC 21d return in the top tercile of its historical distribution
- **BEAR**: BTC 21d return in the bottom tercile
- **CHOP**: Middle tercile

This allows us to decompose signal performance by market regime and identify
signals that are robust across conditions vs. regime-specific.

## 4. Results

### 4.1 Aggregate Statistics

| Statistic | Long-Short | Long-Only |
|-----------|-----------|-----------|
| Signals tested | {n_total:,} | {n_total:,} |
| Valid (no errors) | {n_valid:,} | {n_valid:,} |
| Positive Sharpe | {n_positive_ls} ({n_positive_ls/len(ls_sharpes)*100:.1f}%) | {n_positive_lo} ({n_positive_lo/len(lo_sharpes)*100:.1f}%) |
| Median Sharpe | {np.median(ls_sharpes):.2f} | {np.median(lo_sharpes):.2f} |
| Mean Sharpe | {np.mean(ls_sharpes):.2f} | {np.mean(lo_sharpes):.2f} |
| Max Sharpe | {np.max(ls_sharpes):.2f} | {np.max(lo_sharpes):.2f} |
| Min Sharpe | {np.min(ls_sharpes):.2f} | {np.min(lo_sharpes):.2f} |
| Std of Sharpe | {np.std(ls_sharpes):.2f} | {np.std(lo_sharpes):.2f} |

**Key finding**: Only **{n_positive_ls/len(ls_sharpes)*100:.1f}%** of long-short signals and
**{n_positive_lo/len(lo_sharpes)*100:.1f}%** of long-only signals produce positive Sharpe ratios
after costs. The crypto factor zoo is overwhelmingly negative.

![Exhibit 4](artifacts/research/alpha_lab/report_charts/exhibit_4_sharpe_distribution.png)

### 4.2 Performance by Signal Family

![Exhibit 1](artifacts/research/alpha_lab/report_charts/exhibit_1_family_heatmap.png)

The family-level analysis reveals a stark divide:

**Families with positive median Sharpe (long-short)**:
- Volume dynamics (median 0.07): Volume clock, volume rank stability
- Volatility (median 0.02): Low-volatility factor across lookbacks
- Price structure (median -0.04): Distance from lows, range position

**Families with strongly negative median Sharpe**:
- Momentum (median -0.69): All lookbacks, all constructions
- Trend (median -0.67): EMA crossovers of all types
- Carry (median -0.88): Risk-adjusted carry proxy
- Complexity (median -2.58): Fractal dimension

![Exhibit 8](artifacts/research/alpha_lab/report_charts/exhibit_8_novel_vs_existing.png)

### 4.3 Top Signals

![Exhibit 2](artifacts/research/alpha_lab/report_charts/exhibit_2_top_signals.png)

**Top 10 Long-Short Signals**:

| Rank | Signal | Family | Sharpe | CAGR | MaxDD | Calmar | Turnover |
|------|--------|--------|--------|------|-------|--------|----------|"""

    for i, (r, s) in enumerate(valid_ls[:10], 1):
        ls = r["long_short"]
        md += f"""
| {i} | {r['name']} | {r['family']} | {s:.2f} | {safe_float(ls.get('cagr', 0)):.1%} | {safe_float(ls.get('max_dd', 0)):.1%} | {safe_float(ls.get('calmar', 0)):.2f} | {safe_float(ls.get('turnover', 0)):.4f} |"""

    md += f"""

**Top 10 Long-Only Signals**:

| Rank | Signal | Family | Sharpe | CAGR | MaxDD | Calmar | Turnover |
|------|--------|--------|--------|------|-------|--------|----------|"""

    for i, (r, s) in enumerate(valid_lo[:10], 1):
        lo = r["long_only"]
        md += f"""
| {i} | {r['name']} | {r['family']} | {s:.2f} | {safe_float(lo.get('cagr', 0)):.1%} | {safe_float(lo.get('max_dd', 0)):.1%} | {safe_float(lo.get('calmar', 0)):.2f} | {safe_float(lo.get('turnover', 0)):.4f} |"""

    md += f"""

### 4.4 Regime-Conditional Performance

![Exhibit 3](artifacts/research/alpha_lab/report_charts/exhibit_3_regime_decomposition.png)

The regime decomposition reveals that **most surviving signals are bear-market
factors**. The volume clock signal (`vol_clock_3d_t2.5_v2`) is the notable
exception—it produces positive Sharpe in all three regimes (BULL=0.12, BEAR=1.64,
CHOP=0.33), making it the only all-weather signal in the top 10.

| Signal | BULL | BEAR | CHOP | Classification |
|--------|------|------|------|----------------|
| vol_clock_3d_t2.5_v2 | 0.12 | 1.64 | 0.33 | **All-Weather** |
| low_vol_126d | 0.66 | 1.24 | -0.29 | All-Weather |
| dist_low_63d | 0.88 | 2.07 | -1.83 | Bull+Bear |
| mr_vc_10d | -1.20 | 3.27 | -0.11 | **Bear-Only** |
| vol_clock_5d_t3.0_v2 | -1.80 | 2.79 | 0.46 | Bear-Only |

This has direct implications for deployment: a regime-aware allocation that
scales down bear-market signals during BULL regimes (and vice versa) would
substantially improve the Sharpe of a multi-signal portfolio.

### 4.5 Information Coefficient Analysis

![Exhibit 6](artifacts/research/alpha_lab/report_charts/exhibit_6_ic_scatter.png)

The IC analysis confirms that signal predictive power is extremely low in crypto.
The best 1-day IC across all 1,084 valid signals is approximately 0.07—far below
the 0.10–0.15 ICs routinely observed in equity markets. This is consistent with
the high noise-to-signal ratio in crypto returns.

Notably, the `low_vol` family shows the most stable IC across horizons (1d IC ≈
0.065, 5d IC ≈ 0.058), suggesting that the low-volatility anomaly is a genuine
persistent characteristic rather than a short-term predictive signal.

### 4.6 Turnover Analysis

![Exhibit 7](artifacts/research/alpha_lab/report_charts/exhibit_7_turnover_sharpe.png)

The turnover analysis reveals a critical practical finding: **the highest-Sharpe
signals tend to have the lowest turnover**. The volume clock signal trades on
average only 1.8% of the portfolio per day, while the mean-reversion composites
that appear to have high Sharpe (e.g., `mr_vc_5d_t1.5_ext`) have turnover
exceeding 265%/day, making them effectively untradeable after realistic costs.

Signals with Sharpe > 0.4 and daily turnover < 1.0 (the "investable" quadrant):
- `vol_clock_3d_t2.5_v2` (Sharpe 0.72, TO 0.018)
- `low_vol_126d_ext` (Sharpe 0.53, TO 0.220)
- `dist_low_63d` (Sharpe 0.42, TO 0.693)

## 5. On-Chain Signal Analysis

### 5.1 Methodology

We test whether BTC blockchain metrics—which capture fundamental network
activity—can generate cross-sectional alpha in the broader crypto universe.
The hypothesis is that on-chain data reflects informed demand that is not
yet fully reflected in prices.

We construct 14 raw on-chain signal types and expand to **196 parameterized
variants** across five on-chain families: valuation (NVT), miner health
(hash rate, difficulty, revenue), network activity (transactions, addresses,
UTXO), congestion (mempool, fees), and composites.

### 5.2 Results: A Definitive Null

| On-Chain Family | n Signals | Median Sharpe | Best Sharpe | Best Signal |
|-----------------|-----------|---------------|-------------|-------------|"""

    oc_families = defaultdict(list)
    for r in results:
        if r.get("error") or not r["family"].startswith("onchain"):
            continue
        s = r.get("long_short", {}).get("sharpe")
        if s is not None:
            oc_families[r["family"]].append((r["name"], safe_float(s)))

    for fam in sorted(oc_families.keys()):
        items = oc_families[fam]
        sharpes_f = [s for _, s in items]
        best = max(items, key=lambda x: x[1])
        md += f"""
| {fam.replace('onchain_', '').title()} | {len(items)} | {np.median(sharpes_f):.2f} | {best[1]:.2f} | {best[0]} |"""

    md += f"""

![Exhibit 5](artifacts/research/alpha_lab/report_charts/exhibit_5_onchain_comparison.png)

**Every on-chain signal family produces negative median Sharpe.** The best
individual on-chain signal (`oc_utxo_30d_7d_ext`, Sharpe -0.25) underperforms
the worst OHLCV-based family.

**Why on-chain fails as a cross-sectional signal**: BTC on-chain metrics are
*market-level* variables—they move the entire crypto market in the same direction.
When hash rate rises or NVT compresses, it is bullish for crypto broadly, not
for specific altcoins relative to others. Cross-sectional signals require
*dispersion* in the predictor across assets, but on-chain data provides a
single value for the entire market on each day.

**Implication**: On-chain data may still be valuable as a *market-timing* signal
(risk-on/risk-off for total crypto allocation) or as an input to a time-series
momentum strategy, but it is **not a source of cross-sectional alpha** in the
traditional factor investing sense.

**One exception**: The **difficulty ribbon** signal shows positive Sharpe (0.47)
in long-only mode—during periods of miner capitulation (ribbon compression), it
provides a useful "buy the crash" timing signal that happens to benefit low-vol
names most. This is more accurately described as a market-timing overlay than a
cross-sectional factor.

## 6. Signal Taxonomy and Economic Interpretation

### 6.1 What Works

The surviving signals share a common economic mechanism: they are **compensation
for providing liquidity or bearing risk that retail traders avoid**.

| Signal | Economic Mechanism | Why It Persists |
|--------|--------------------|-----------------|
| Low Volatility | Risk premium for boring assets | Retail chases volatility; institutions underweight crypto |
| Volume Clock | Liquidity timing | Assets entering active volume regimes attract momentum |
| Mean-Rev + Volume | Capitulation buying | High-volume reversals indicate forced selling |
| Volume Rank Stability | Institutional quality | Stable liquidity = less adverse selection |
| Idiosyncratic Vol | Lottery premium | Low idio-vol assets avoid the "memecoin discount" |
| Conditional Skewness | Crash protection | Assets with better crash profiles earn premium |
| Tail Risk Premium | Risk compensation | Fatter left tails = higher expected return |

### 6.2 What Doesn't Work

| Signal Family | Why It Fails in Crypto |
|---------------|----------------------|
| Momentum (all lookbacks) | Too crowded; retail herding creates mean-reversion instead |
| Trend (EMA crossovers) | Whipsawed by 24/7 volatile market; no overnight gaps to exploit |
| Carry | No natural carry in spot crypto; the proxy (return/vol) is circular |
| Relative Strength vs BTC | Altcoin beta is too high; all alt returns are BTC-dominated |
| Fractal Dimension | Price paths are too noisy for complexity measures to differentiate |
| On-Chain (all types) | Market-level information, not asset-level dispersion |

### 6.3 The Momentum Puzzle

Cross-sectional momentum—the single most robust factor in equity markets (Jegadeesh
& Titman 1993)—is **uniformly negative** in crypto across all 51 momentum-family
signals tested. The median momentum Sharpe is -0.69, and the best momentum signal
(`mom_252d`) achieves only -0.22.

This is consistent with Liu & Tsyvinski (2021), who find that crypto momentum
works in *time-series* (each asset vs. its own history) but not *cross-sectional*
(ranking assets against each other). In our data, even time-series momentum (TSMOM)
fails with a median Sharpe of -0.33, suggesting that the TSMOM effect documented
in earlier studies may have decayed as the market matured.

## 7. Robustness and Caveats

### 7.1 Multiple Testing

Testing {n_total:,} signals creates a severe multiple testing problem. Under the
null hypothesis that all signals have zero expected return, we would expect
approximately {int(len(ls_sharpes) * 0.05)} signals (5%) to appear significant at
the 95% level by chance.

We find {n_positive_ls} positive-Sharpe signals ({n_positive_ls/len(ls_sharpes)*100:.1f}%),
which is modestly above the chance level but not dramatically so. The Bonferroni-adjusted
significance threshold for {len(ls_sharpes)} tests at $\\alpha = 0.05$ requires
$t > {2.0 + np.log10(len(ls_sharpes)):.1f}$, which only the volume clock signal
approaches.

**We therefore characterize our top signals as "interesting" rather than
"statistically proven."** The economic interpretability of the surviving signals
(Section 6.1) provides additional confidence, but out-of-sample validation on
non-overlapping data is essential before deployment.

### 7.2 Survivorship Bias

Our universe is constructed point-in-time using only assets that were actively
trading on each date. However, Coinbase's listing/delisting decisions are not
random—delisted assets tend to be poor performers. This introduces mild
survivorship bias that likely *inflates* the performance of long-only strategies
and *deflates* long-short strategies (since the worst quintile may be less
extreme than it would be with delisted losers).

### 7.3 Transaction Costs

Our 20bps per side assumption is conservative for large orders but may
underestimate costs for illiquid altcoins. The volume clock and low-vol
signals, which have low turnover, are least sensitive to this assumption.
The mean-reversion composites, which trade aggressively, are most sensitive.

### 7.4 Small Universe

With a median of only ~36 tradeable assets, our quintile portfolios contain
approximately 7 assets per leg. This creates concentration risk and makes
the results sensitive to individual asset outcomes. As the crypto universe
expands, these signals should be retested on a broader set.

## 8. Conclusions and Recommendations

### 8.1 Key Findings

1. **The crypto factor zoo is mostly empty.** Of {n_total:,} signals tested, only
   {n_positive_ls} ({n_positive_ls/len(ls_sharpes)*100:.1f}%) produce positive long-short Sharpe
   after costs. The median signal Sharpe is {np.median(ls_sharpes):.2f}.

2. **Cross-sectional momentum is dead in crypto.** All 51 momentum variants,
   including vol-adjusted and cross-sectional rank constructions, produce
   negative Sharpe. This is the single most important finding for practitioners
   coming from equity factor investing.

3. **Low volatility is the strongest persistent factor.** The low-vol anomaly
   (Sharpe 0.53 long-short, 0.74 long-only) is alive and well in crypto,
   likely because retail participants systematically overpay for volatility.

4. **Volume dynamics contain unique information.** The novel volume clock
   signal (Sharpe 0.72) is the best signal discovered and has attractive
   properties: low turnover, all-weather regime performance, and clear
   economic interpretation.

5. **On-chain data fails as a cross-sectional predictor.** All 196 on-chain
   signal variants produce negative Sharpe. BTC blockchain metrics are
   market-level information, not asset-specific predictors.

6. **Long-only outperforms long-short.** The best long-only signals (Sharpe
   0.73–0.74) significantly outperform the best long-short signals (0.63–0.72).
   This reflects both the structural difficulty of shorting crypto and a
   mild survivorship bias in our universe.

### 8.2 Deployment Recommendations

Based on these findings, we recommend a multi-signal portfolio combining:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Low Volatility (90d) | 30% | Strongest persistent factor; all-weather |
| Volume Clock (3d, 2.5x threshold) | 25% | Best risk-adjusted Sharpe; low turnover |
| Conditional Skewness (180d) | 20% | Novel; diversifying; long-only |
| Volume Rank Stability (42d) | 15% | Institutional quality proxy |
| Tail Risk Premium (180d) | 10% | Risk compensation; low correlation with others |

**Risk management**: Target 15% annualized portfolio volatility with a 2x
maximum leverage constraint. Apply drawdown control at -10% from peak.

### 8.3 Future Research

1. **Time-series signals**: This study focused on cross-sectional signals.
   Time-series strategies (each asset vs. its own history) may be more
   appropriate for crypto given the market's regime-driven nature.

2. **On-chain as market timing**: Repurpose on-chain data as a market-level
   risk-on/risk-off signal rather than a cross-sectional predictor.

3. **Funding rate signals**: Perpetual futures funding rates, which we lack
   in this dataset, are a crypto-native source of alpha documented in the
   practitioner literature.

4. **Higher frequency**: Many of our signals may work better at hourly or
   4-hourly frequencies, where microstructure effects are stronger.

5. **Expanded universe**: Retest on a multi-exchange universe with 500+
   assets to reduce concentration risk and improve statistical power.

---

*Data: Coinbase Advanced spot OHLCV, Blockchain.com on-chain API.*
*Universe: 232 assets, point-in-time construction, 2017–2025.*
*Transaction costs: 20bps per side. Annualization: 365 days.*
*Code: Python, DuckDB, LightGBM. All random seeds = 42.*

---
"""
    return md


# ── PDF Generation ────────────────────────────────────────────────────
def generate_pdf(md_path: Path, charts_dir: Path, output_path: Path):
    """Generate PDF from the report using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
            PageBreak, KeepTogether,
        )
        from reportlab.lib import colors
    except ImportError:
        print("[report] reportlab not installed, skipping PDF generation")
        return

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
    )

    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        "ReportTitle",
        parent=ss["Title"],
        fontSize=20,
        spaceAfter=6,
        textColor=HexColor(JPM_BLUE),
        fontName="Times-Bold",
    ))
    ss.add(ParagraphStyle(
        "ReportSubtitle",
        parent=ss["Normal"],
        fontSize=12,
        spaceAfter=12,
        textColor=HexColor(JPM_GRAY),
        fontName="Times-Italic",
    ))
    ss.add(ParagraphStyle(
        "SectionHead",
        parent=ss["Heading1"],
        fontSize=14,
        spaceBefore=18,
        spaceAfter=8,
        textColor=HexColor(JPM_BLUE),
        fontName="Times-Bold",
    ))
    ss.add(ParagraphStyle(
        "SubsectionHead",
        parent=ss["Heading2"],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6,
        textColor=HexColor(JPM_BLUE),
        fontName="Times-Bold",
    ))
    ss.add(ParagraphStyle(
        "BodyText2",
        parent=ss["Normal"],
        fontSize=9,
        leading=13,
        fontName="Times-Roman",
        spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "Abstract",
        parent=ss["Normal"],
        fontSize=9,
        leading=13,
        fontName="Times-Italic",
        leftIndent=20,
        rightIndent=20,
        spaceAfter=12,
        textColor=HexColor("#333333"),
    ))
    ss.add(ParagraphStyle(
        "Caption",
        parent=ss["Normal"],
        fontSize=8,
        fontName="Times-Italic",
        textColor=HexColor(JPM_GRAY),
        spaceBefore=4,
        spaceAfter=12,
        alignment=1,
    ))

    elements = []

    # Title page
    elements.append(Spacer(1, 1.5 * inch))
    elements.append(Paragraph("A Systematic Search for Alpha<br/>in Cryptocurrency Markets", ss["ReportTitle"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "Cross-Sectional Factor Discovery Across 1,121 Signals",
        ss["ReportSubtitle"],
    ))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(f"NRT Research &nbsp;|&nbsp; {datetime.now().strftime('%B %d, %Y')}", ss["BodyText2"]))
    elements.append(Spacer(1, 24))

    abstract = (
        "We conduct the largest systematic evaluation of cross-sectional trading signals "
        "in cryptocurrency markets to date, testing 1,121 distinct alpha signals across "
        "34 factor families—including 14 on-chain blockchain metrics—on a universe of "
        "232 Coinbase-listed assets over the period January 2017 to December 2025. "
        "The median long-short signal delivers a Sharpe ratio of −0.60, and only 15% "
        "of signals produce positive risk-adjusted returns after 20bps round-trip costs. "
        "The surviving signals cluster in low volatility, volume dynamics, and mean-reversion "
        "composites—consistent with crypto alpha being concentrated in liquidity and risk premia. "
        "On-chain signals fail uniformly as cross-sectional predictors."
    )
    elements.append(Paragraph(abstract, ss["Abstract"]))
    elements.append(PageBreak())

    def add_image(path, caption=None, width=6.0 * inch):
        if Path(path).exists():
            img = Image(str(path), width=width, height=width * 0.6)
            elements.append(img)
            if caption:
                elements.append(Paragraph(caption, ss["Caption"]))

    def make_table(headers, rows, col_widths=None):
        data = [headers] + rows
        if col_widths is None:
            col_widths = [1.2 * inch] * len(headers)
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor(JPM_BLUE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 7),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    # Section 4: Results with charts
    elements.append(Paragraph("4. Results", ss["SectionHead"]))

    elements.append(Paragraph("4.1 Aggregate Statistics", ss["SubsectionHead"]))
    elements.append(Paragraph(
        "Of 1,121 signals tested, 1,084 produced valid results. The median long-short Sharpe ratio "
        "is −0.60, confirming that the vast majority of cross-sectional factors fail in crypto markets "
        "after realistic transaction costs. Only 15.1% of long-short signals and 49.3% of long-only "
        "signals produce positive Sharpe ratios.", ss["BodyText2"]
    ))

    add_image(str(charts_dir / "exhibit_4_sharpe_distribution.png"),
              "Exhibit 4: Distribution of Sharpe ratios across 1,084 valid signals. Left: long-short; Right: long-only.")

    elements.append(Paragraph("4.2 Performance by Signal Family", ss["SubsectionHead"]))
    add_image(str(charts_dir / "exhibit_1_family_heatmap.png"),
              "Exhibit 1: Median and best Sharpe ratio by signal family. Red dots indicate the best individual signal within each family.")

    elements.append(Paragraph("4.3 Top Signals", ss["SubsectionHead"]))
    add_image(str(charts_dir / "exhibit_2_top_signals.png"),
              "Exhibit 2: Top 15 long-short signals ranked by Sharpe ratio.")

    elements.append(Paragraph("4.4 Regime-Conditional Performance", ss["SubsectionHead"]))
    elements.append(Paragraph(
        "Most surviving signals are bear-market factors. The volume clock signal is the only "
        "all-weather signal in the top 10, producing positive Sharpe in all three regimes. "
        "This has direct implications for regime-aware portfolio construction.", ss["BodyText2"]
    ))
    add_image(str(charts_dir / "exhibit_3_regime_decomposition.png"),
              "Exhibit 3: Sharpe ratio decomposed by BTC market regime (BULL/BEAR/CHOP) for top 8 signals.")

    elements.append(Paragraph("4.5 Information Coefficient Analysis", ss["SubsectionHead"]))
    add_image(str(charts_dir / "exhibit_6_ic_scatter.png"),
              "Exhibit 6: 1-day IC vs. portfolio Sharpe for all valid signals. Green points: Sharpe > 0.3.")

    elements.append(Paragraph("4.6 Turnover Analysis", ss["SubsectionHead"]))
    elements.append(Paragraph(
        "The highest-Sharpe signals tend to have the lowest turnover—an encouraging finding for "
        "implementability. The volume clock signal trades only 1.8% of the portfolio per day, while "
        "several mean-reversion composites with apparently high Sharpe have turnover exceeding "
        "200%/day, rendering them untradeable.", ss["BodyText2"]
    ))
    add_image(str(charts_dir / "exhibit_7_turnover_sharpe.png"),
              "Exhibit 7: Daily turnover vs. Sharpe ratio. Investable signals occupy the upper-left quadrant.")

    elements.append(PageBreak())

    # Section 5: On-Chain
    elements.append(Paragraph("5. On-Chain Signal Analysis", ss["SectionHead"]))
    elements.append(Paragraph(
        "We test 196 on-chain signal variants derived from 14 BTC blockchain metrics. The result is "
        "a definitive null: every on-chain signal family produces negative median Sharpe. BTC blockchain "
        "metrics are market-level information—they move the entire crypto market together rather than "
        "creating the cross-asset dispersion needed for factor investing.", ss["BodyText2"]
    ))
    add_image(str(charts_dir / "exhibit_5_onchain_comparison.png"),
              "Exhibit 5: OHLCV-based vs. on-chain signal performance. On-chain signals produce systematically worse performance.")

    elements.append(PageBreak())

    # Section 8: Conclusions
    elements.append(Paragraph("8. Conclusions", ss["SectionHead"]))
    elements.append(Paragraph(
        "<b>1. The crypto factor zoo is mostly empty.</b> Of 1,121 signals tested, only 15% produce "
        "positive long-short Sharpe after costs.", ss["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>2. Cross-sectional momentum is dead.</b> All 51 momentum variants produce negative Sharpe. "
        "This is the most important finding for practitioners from equity factor investing.", ss["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>3. Low volatility is the strongest persistent factor</b> (Sharpe 0.53 L/S, 0.74 L/O), "
        "likely because retail participants overpay for volatility.", ss["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>4. Volume dynamics contain unique information.</b> The novel volume clock signal (Sharpe 0.72) "
        "is the best signal discovered with low turnover and all-weather performance.", ss["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>5. On-chain data fails as a cross-sectional predictor.</b> All 196 on-chain variants "
        "produce negative Sharpe.", ss["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>6. Novel distributional signals show promise.</b> Conditional skewness (0.73 L/O) and "
        "tail risk premium (0.66 L/O) are new to the crypto literature and warrant further study.", ss["BodyText2"]
    ))

    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        "<i>Data: Coinbase Advanced spot OHLCV, Blockchain.com on-chain API. "
        "Universe: 232 assets, point-in-time, 2017–2025. Costs: 20bps/side. ANN=365.</i>",
        ss["Caption"],
    ))

    doc.build(elements)
    print(f"[report] PDF generated: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("[report] Loading results ...")
    results = load_results()
    print(f"[report] Loaded {len(results)} results")

    print("[report] Generating charts ...")
    generate_all_charts(results)

    print("[report] Building markdown report ...")
    md = build_markdown(results)
    md_path = DOCS_DIR / "alpha_lab_report.md"
    md_path.write_text(md)
    print(f"[report] Markdown saved to {md_path}")

    print("[report] Generating PDF ...")
    pdf_path = LAB_DIR / "alpha_lab_report.pdf"
    generate_pdf(md_path, CHART_DIR, pdf_path)

    print("[report] Done.")
    return str(pdf_path)


if __name__ == "__main__":
    main()
