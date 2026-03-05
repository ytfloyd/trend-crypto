#!/usr/bin/env python3
"""
Trader-Facing Summary: Trend-Following Risk/Reward Breakdown

Generates a set of clear charts and a summary table for presentation
to a trader evaluating the strategy.

Usage:
    python -m scripts.research.tsmom.generate_trader_summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "artifacts" / "research" / "tsmom"
EXT = SWEEP / "eth_trend_extension"
CROSS = SWEEP / "cross_asset"
FULLUNIV = SWEEP / "full_universe"
OUT = SWEEP / "trader_summary"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GREEN = "#336633"; GRAY = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
BLUE = "#1565C0"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": GRAY,
    "figure.facecolor": "white",
})


def main():
    print("=" * 60)
    print("  TRADER SUMMARY — TREND FOLLOWING RISK/REWARD")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    eth_ens = pd.read_csv(EXT / "task4" / "ensemble_comparison.csv")
    eth_wf = pd.read_csv(EXT / "task5" / "ensemble_comparison.csv")
    btc_wf = pd.read_csv(CROSS / "btc_walkforward_comparison.csv")
    cost = pd.read_csv(FULLUNIV / "cost_sensitivity.csv")
    asset_cls = pd.read_csv(FULLUNIV / "asset_classification.csv")
    drift = pd.read_csv(EXT / "task3" / "drift_sensitivity.csv")
    tier2 = pd.read_csv(CROSS / "tier2_replication_summary.csv")

    # ================================================================
    # CHART 1: Strategy vs Buy & Hold — The Core Tradeoff (ETH)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    strategies = ["Buy & Hold", "TIM Ensemble\n(in-sample)", "Walk-Forward\nEnsemble"]
    sharpes = [1.11, 1.367, 1.429]
    maxdds = [94.0, 48.1, 44.6]
    cagrs = [82.7, 70.3, 72.3]
    skews = [0.365, 1.458, 1.561]

    colors = [RED, TEAL, NAVY]

    bars1 = axes[0].bar(strategies, sharpes, color=colors, width=0.6, edgecolor="white", lw=1.5)
    axes[0].set_ylabel("Sharpe Ratio", fontweight="bold")
    axes[0].set_title("Risk-Adjusted Return", fontweight="bold", fontsize=12)
    axes[0].set_ylim(0, 1.8)
    for bar, val in zip(bars1, sharpes):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.04, f"{val:.2f}",
                     ha="center", va="bottom", fontweight="bold", fontsize=11)

    bars2 = axes[1].bar(strategies, maxdds, color=colors, width=0.6, edgecolor="white", lw=1.5)
    axes[1].set_ylabel("Max Drawdown (%)", fontweight="bold")
    axes[1].set_title("Worst Peak-to-Trough Loss", fontweight="bold", fontsize=12)
    axes[1].set_ylim(0, 110)
    for bar, val in zip(bars2, maxdds):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 2, f"-{val:.0f}%",
                     ha="center", va="bottom", fontweight="bold", fontsize=11)

    bars3 = axes[2].bar(strategies, skews, color=colors, width=0.6, edgecolor="white", lw=1.5)
    axes[2].set_ylabel("Return Skewness", fontweight="bold")
    axes[2].set_title("Convexity (Right Tail)", fontweight="bold", fontsize=12)
    axes[2].set_ylim(0, 2.0)
    for bar, val in zip(bars3, skews):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.04, f"{val:.2f}",
                     ha="center", va="bottom", fontweight="bold", fontsize=11)

    fig.suptitle("ETH-USD: Trend Ensemble vs Buy & Hold (2017–2026)",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "1_core_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [1/6] Core tradeoff chart saved")

    # ================================================================
    # CHART 2: In-Sample vs Out-of-Sample (the honesty chart)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ETH
    labels_eth = ["B&H\n(Full)", "Ensemble\n(Full)", "B&H\n(OOS '22-'26)", "Ensemble\n(OOS '22-'26)"]
    sr_eth = [1.11, 1.429, 0.152, 0.265]
    dd_eth = [94.0, 44.6, 74.0, 37.2]
    colors_eth = [RED, TEAL, RED, TEAL]
    alphas_eth = [1.0, 1.0, 0.5, 0.5]

    bars = axes[0].bar(labels_eth, sr_eth, color=colors_eth, width=0.6, edgecolor="white", lw=1.5)
    for bar, a in zip(bars, alphas_eth):
        bar.set_alpha(a)
    for bar, val in zip(bars, sr_eth):
        axes[0].text(bar.get_x() + bar.get_width()/2, max(val, 0) + 0.04, f"{val:.2f}",
                     ha="center", va="bottom", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Sharpe Ratio", fontweight="bold")
    axes[0].set_title("ETH-USD — Sharpe", fontweight="bold", fontsize=12)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_ylim(-0.2, 1.8)

    # BTC
    labels_btc = ["B&H\n(Full)", "Ensemble\n(Full)", "B&H\n(OOS '22-'26)", "Ensemble\n(OOS '22-'26)"]
    sr_btc = [1.014, 1.228, 0.446, 0.270]
    colors_btc = [RED, NAVY, RED, NAVY]

    bars = axes[1].bar(labels_btc, sr_btc, color=colors_btc, width=0.6, edgecolor="white", lw=1.5)
    for bar, a in zip(bars, alphas_eth):
        bar.set_alpha(a)
    for bar, val in zip(bars, sr_btc):
        axes[1].text(bar.get_x() + bar.get_width()/2, max(val, 0) + 0.04, f"{val:.2f}",
                     ha="center", va="bottom", fontweight="bold", fontsize=10)
    axes[1].set_ylabel("Sharpe Ratio", fontweight="bold")
    axes[1].set_title("BTC-USD — Sharpe", fontweight="bold", fontsize=12)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylim(-0.2, 1.8)

    fig.suptitle("Full Period vs Out-of-Sample (Walk-Forward, No Look-Ahead)",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "2_insample_vs_oos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/6] In-sample vs OOS chart saved")

    # ================================================================
    # CHART 3: Drawdown Compression — The Insurance Value
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5))

    assets = ["ETH\n(Full)", "ETH\n(OOS)", "BTC\n(Full)", "BTC\n(OOS)",
              "LTC\n(Blind)", "LINK\n(Blind)", "ATOM\n(Blind)"]
    bh_dd = [94.0, 74.0, 83.8, 67.0, 93.6, 90.2, 95.9]
    ens_dd = [44.6, 37.2, 51.6, 43.2, 84.6, 78.4, 76.0]

    x = np.arange(len(assets))
    w = 0.35
    bars1 = ax.bar(x - w/2, bh_dd, w, label="Buy & Hold", color=RED, alpha=0.7, edgecolor="white")
    bars2 = ax.bar(x + w/2, ens_dd, w, label="Trend Ensemble", color=TEAL, edgecolor="white")

    for bar, val in zip(bars1, bh_dd):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"-{val:.0f}%",
                ha="center", va="bottom", fontsize=8, color=RED)
    for bar, val in zip(bars2, ens_dd):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"-{val:.0f}%",
                ha="center", va="bottom", fontsize=8, color=TEAL)

    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Max Drawdown (%)", fontweight="bold")
    ax.set_title("Drawdown Compression — Buy & Hold vs Trend Ensemble",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white")
    ax.set_ylim(0, 115)

    fig.tight_layout()
    fig.savefig(OUT / "3_drawdown_compression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [3/6] Drawdown compression chart saved")

    # ================================================================
    # CHART 4: Transaction Cost Sensitivity
    # ================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(cost["cost_bps"], cost["eth_sharpe"], "o-", color=TEAL, lw=2.5, ms=10,
            label="ETH Ensemble", zorder=5)
    ax.plot(cost["cost_bps"], cost["btc_sharpe"], "s-", color=NAVY, lw=2.5, ms=10,
            label="BTC Ensemble", zorder=5)

    ax.axhline(1.11, color=TEAL, ls=":", lw=1.2, alpha=0.6)
    ax.text(62, 1.12, "ETH B&H (1.11)", fontsize=8, color=TEAL, alpha=0.8)
    ax.axhline(1.014, color=NAVY, ls=":", lw=1.2, alpha=0.6)
    ax.text(62, 1.024, "BTC B&H (1.01)", fontsize=8, color=NAVY, alpha=0.8)

    ax.fill_between([15, 65], 0, 0.5, color=RED, alpha=0.05)
    ax.axhline(1.0, color=GRAY, ls="--", lw=0.8, alpha=0.4)

    ax.set_xlabel("Round-Trip Transaction Cost (bps)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontweight="bold", fontsize=11)
    ax.set_title("How Much Edge Survives After Costs?", fontweight="bold", fontsize=13)
    ax.set_xticks([20, 40, 60])
    ax.set_xlim(15, 65)
    ax.set_ylim(0.8, 1.5)
    ax.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white")

    for _, row in cost.iterrows():
        ax.annotate(f"{row['eth_sharpe']:.2f}", (row['cost_bps'], row['eth_sharpe']),
                    textcoords="offset points", xytext=(12, 5), fontsize=9,
                    fontweight="bold", color=TEAL)
        ax.annotate(f"{row['btc_sharpe']:.2f}", (row['cost_bps'], row['btc_sharpe']),
                    textcoords="offset points", xytext=(12, -12), fontsize=9,
                    fontweight="bold", color=NAVY)

    fig.tight_layout()
    fig.savefig(OUT / "4_cost_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [4/6] Cost sensitivity chart saved")

    # ================================================================
    # CHART 5: Asset Universe — What's Tradeable?
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    cls_order = {"STRONG": 0, "MODERATE": 1, "MARGINAL": 2, "WEAK": 3}
    cls_colors = {"STRONG": TEAL, "MODERATE": BLUE, "MARGINAL": GOLD, "WEAK": RED}
    asset_cls_sorted = asset_cls.sort_values(
        by=["class", "wf_oos_sharpe"],
        key=lambda x: x.map(cls_order) if x.name == "class" else x,
        ascending=[True, False]
    )

    syms = [s.replace("-USD", "") for s in asset_cls_sorted["symbol"]]
    oos_sr = asset_cls_sorted["wf_oos_sharpe"].values
    bar_colors = [cls_colors[c] for c in asset_cls_sorted["class"]]

    bars = ax.bar(range(len(syms)), oos_sr, color=bar_colors, edgecolor="white", lw=0.8)
    ax.set_xticks(range(len(syms)))
    ax.set_xticklabels(syms, rotation=60, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Walk-Forward OOS Sharpe", fontweight="bold")
    ax.set_title("31 Assets Ranked by Out-of-Sample Trend Performance",
                 fontweight="bold", fontsize=13)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TEAL, label=f"STRONG ({(asset_cls['class']=='STRONG').sum()})"),
        Patch(facecolor=BLUE, label=f"MODERATE ({(asset_cls['class']=='MODERATE').sum()})"),
        Patch(facecolor=GOLD, label=f"MARGINAL ({(asset_cls['class']=='MARGINAL').sum()})"),
        Patch(facecolor=RED, label=f"WEAK ({(asset_cls['class']=='WEAK').sum()})"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left",
              frameon=True, facecolor="white")

    fig.tight_layout()
    fig.savefig(OUT / "5_asset_universe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [5/6] Asset universe chart saved")

    # ================================================================
    # CHART 6: The Pitch — Risk/Reward Summary Quadrant
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    points = {
        "ETH B&H": (1.11, -94.0, RED, "o", 120),
        "BTC B&H": (1.014, -83.8, RED, "s", 120),
        "ETH Ensemble\n(Walk-Forward)": (1.429, -44.6, TEAL, "o", 200),
        "BTC Ensemble\n(Walk-Forward)": (1.228, -51.6, NAVY, "s", 200),
        "ETH Ensemble\n(OOS only)": (0.265, -37.2, TEAL, "o", 80),
        "BTC Ensemble\n(OOS only)": (0.270, -43.2, NAVY, "s", 80),
    }

    for label, (sr, dd, color, marker, size) in points.items():
        ax.scatter(sr, dd, c=color, marker=marker, s=size, zorder=5, edgecolors="white", lw=1)
        offset = (8, 5) if "B&H" not in label else (-8, -12)
        ax.annotate(label, (sr, dd), textcoords="offset points", xytext=offset,
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xlabel("Sharpe Ratio", fontweight="bold", fontsize=12)
    ax.set_ylabel("Max Drawdown (%)", fontweight="bold", fontsize=12)
    ax.set_title("Risk/Reward Map — Trend Ensembles vs Buy & Hold",
                 fontweight="bold", fontsize=13)
    ax.axhline(-50, color=LGRAY, ls="--", lw=0.8, alpha=0.5)
    ax.axvline(1.0, color=LGRAY, ls="--", lw=0.8, alpha=0.5)

    ax.fill_between([0.8, 2.0], -55, 0, color=GREEN, alpha=0.04)
    ax.text(1.5, -15, "TARGET ZONE\n(SR>1, DD<50%)", fontsize=9, color=GREEN,
            alpha=0.4, ha="center", fontweight="bold")

    ax.set_xlim(-0.1, 1.7)
    ax.set_ylim(-100, 0)

    fig.tight_layout()
    fig.savefig(OUT / "6_risk_reward_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [6/6] Risk/reward map saved")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 90)
    print("  HEADLINE NUMBERS")
    print("=" * 90)

    table_data = [
        ("ETH-USD", "Buy & Hold", "1.11", "82.7%", "-94.0%", "0.37", "100%", "—"),
        ("ETH-USD", "TIM Ensemble (WF)", "1.43", "72.3%", "-44.6%", "1.56", "42%", "706"),
        ("ETH-USD", "Ensemble OOS '22-'26", "0.27", "3.5%", "-37.2%", "1.05", "42%", "711"),
        ("", "", "", "", "", "", "", ""),
        ("BTC-USD", "Buy & Hold", "1.01", "58.8%", "-83.8%", "-0.01", "100%", "—"),
        ("BTC-USD", "TIM Ensemble (WF)", "1.23", "42.8%", "-51.6%", "1.41", "42%", "619"),
        ("BTC-USD", "Ensemble OOS '22-'26", "0.27", "3.7%", "-43.2%", "0.84", "42%", "619"),
    ]

    header = f"  {'Asset':<12s} {'Strategy':<25s} {'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>8s} {'Skew':>6s} {'TIM':>5s} {'N Strats':>8s}"
    print(header)
    print("  " + "─" * 86)
    for row in table_data:
        if row[0] == "":
            print()
            continue
        print(f"  {row[0]:<12s} {row[1]:<25s} {row[2]:>7s} {row[3]:>7s} {row[4]:>8s} {row[5]:>6s} {row[6]:>5s} {row[7]:>8s}")

    print("\n" + "=" * 90)
    print("  COST SENSITIVITY (Sharpe at varying transaction costs)")
    print("=" * 90)
    print(f"  {'Cost (bps)':<12s} {'ETH Sharpe':>11s} {'ETH MaxDD':>10s} {'BTC Sharpe':>11s} {'BTC MaxDD':>10s}")
    print("  " + "─" * 56)
    for _, r in cost.iterrows():
        print(f"  {int(r['cost_bps']):<12d} {r['eth_sharpe']:>10.3f} {r['eth_max_dd']:>9.1%} "
              f"{r['btc_sharpe']:>10.3f} {r['btc_max_dd']:>9.1%}")
    print(f"\n  Edge consumed at ~50 bps (ETH), ~35 bps (BTC)")

    print("\n" + "=" * 90)
    print("  KEY STRUCTURAL FINDINGS")
    print("=" * 90)
    print("  • TIM stability: ρ = 0.99 (signal structure determines time-in-market)")
    print("  • Walk-forward produces ZERO Sharpe decay vs in-sample on ETH")
    print("  • Drawdown compression: works on every asset tested (99% of strategies)")
    print("  • Bonferroni survivors: only ETH and BTC (2 of 31 assets)")
    print("  • Optimal time-in-market: 37-47% (universal across crypto)")
    print("  • Sharpe degrades ~0.10 per 20bp on ETH, ~0.14 per 20bp on BTC")

    print(f"\n  All charts saved to: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
