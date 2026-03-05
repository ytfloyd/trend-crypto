#!/usr/bin/env python3
"""
Task 2: ATR vs Fixed Stop — Deeper Decomposition

Decompose the ATR vs. fixed stop matched-pair win rate by frequency and
signal family. Test whether any subgroup shows a statistically significant
ATR advantage after Bonferroni correction.

Usage:
    python -m scripts.research.tsmom.ext_task2_atr_decomp
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep"
OUT = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_extension" / "task2"
OUT.mkdir(parents=True, exist_ok=True)

NAVY  = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GRAY  = "#808080"; LGRAY = "#D0D0D0"; GREEN = "#336633"; BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRAY,
    "figure.facecolor": "white",
})


def main():
    print("=" * 70)
    print("  TASK 2: ATR VS FIXED STOP — DEEPER DECOMPOSITION")
    print("=" * 70)

    df = pd.read_csv(SWEEP_DIR / "results_v2.csv")
    strats = df[df["label"] != "BUY_AND_HOLD"].copy()
    print(f"  Loaded {len(strats)} strategies")

    # Build matched pairs: for each (label, freq), compare best ATR vs best fixed
    pairs = []
    for (label, freq), group in strats.groupby(["label", "freq"]):
        atr = group[group["stop_type"] == "atr"]
        pct = group[group["stop_type"] == "pct"]
        ns = group[group["stop"] == "none"]
        if atr.empty or pct.empty:
            continue
        best_atr = atr.loc[atr["sharpe"].idxmax()]
        best_pct = pct.loc[pct["sharpe"].idxmax()]
        family = group["signal_family"].iloc[0]
        pairs.append({
            "label": label, "freq": freq, "family": family,
            "atr_sharpe": best_atr["sharpe"],
            "pct_sharpe": best_pct["sharpe"],
            "atr_dd": best_atr["max_dd"],
            "pct_dd": best_pct["max_dd"],
            "atr_skew": best_atr["skewness"],
            "pct_skew": best_pct["skewness"],
            "sharpe_diff": best_atr["sharpe"] - best_pct["sharpe"],
            "dd_diff": best_atr["max_dd"] - best_pct["max_dd"],
        })

    pdf = pd.DataFrame(pairs)
    pdf["atr_wins_sharpe"] = pdf["atr_sharpe"] > pdf["pct_sharpe"]
    pdf["atr_wins_dd"] = pdf["atr_dd"] > pdf["pct_dd"]
    print(f"  Matched pairs: {len(pdf)}")
    print(f"  Overall ATR win rate (Sharpe): {pdf['atr_wins_sharpe'].mean():.1%}")
    print(f"  Overall ATR win rate (MaxDD):  {pdf['atr_wins_dd'].mean():.1%}")

    # ── Decomposition by frequency ──────────────────────────────────
    print(f"\n  {'Freq':<6s} {'N':>5s} {'ATR Win SR':>11s} {'ATR Win DD':>11s} "
          f"{'Med SR Diff':>12s} {'Med DD Diff':>12s}")
    print(f"  {'─'*6} {'─'*5} {'─'*11} {'─'*11} {'─'*12} {'─'*12}")
    freq_rows = []
    for freq in ["1d", "4h", "1h"]:
        sub = pdf[pdf["freq"] == freq]
        if len(sub) < 5:
            continue
        row = {
            "freq": freq, "n_pairs": len(sub),
            "atr_win_sharpe": round(sub["atr_wins_sharpe"].mean(), 4),
            "atr_win_dd": round(sub["atr_wins_dd"].mean(), 4),
            "med_sharpe_diff": round(sub["sharpe_diff"].median(), 4),
            "med_dd_diff": round(sub["dd_diff"].median(), 4),
        }
        _, p = sp_stats.ttest_rel(sub["atr_sharpe"], sub["pct_sharpe"])
        row["ttest_p"] = round(p, 6)
        freq_rows.append(row)
        print(f"  {freq:<6s} {len(sub):>5d} {row['atr_win_sharpe']:>10.1%} "
              f"{row['atr_win_dd']:>10.1%} {row['med_sharpe_diff']:>+11.4f} "
              f"{row['med_dd_diff']:>+11.4f}  p={p:.4f}")

    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(OUT / "atr_vs_fixed_by_freq.csv", index=False)

    # ── Decomposition by frequency × signal family ──────────────────
    cell_rows = []
    for (freq, fam), sub in pdf.groupby(["freq", "family"]):
        if len(sub) < 5:
            continue
        t_stat, p_val = sp_stats.ttest_rel(sub["atr_sharpe"], sub["pct_sharpe"])
        cell_rows.append({
            "freq": freq, "family": fam, "n_pairs": len(sub),
            "atr_win_sharpe": round(sub["atr_wins_sharpe"].mean(), 4),
            "atr_win_dd": round(sub["atr_wins_dd"].mean(), 4),
            "med_sharpe_diff": round(sub["sharpe_diff"].median(), 4),
            "med_dd_diff": round(sub["dd_diff"].median(), 4),
            "mean_sharpe_diff": round(sub["sharpe_diff"].mean(), 4),
            "ttest_t": round(t_stat, 3),
            "ttest_p": round(p_val, 6),
        })

    cell_df = pd.DataFrame(cell_rows)
    n_cells = len(cell_df)
    bonf_threshold = 0.05 / n_cells
    cell_df["bonferroni_sig"] = cell_df["ttest_p"] < bonf_threshold
    cell_df.to_csv(OUT / "atr_vs_fixed_by_freq_family.csv", index=False)

    n_sig = cell_df["bonferroni_sig"].sum()
    print(f"\n  Cells tested: {n_cells}")
    print(f"  Bonferroni threshold: p < {bonf_threshold:.6f}")
    print(f"  Significant cells: {n_sig}")

    if n_sig > 0:
        print(f"\n  Significant cells:")
        for _, row in cell_df[cell_df["bonferroni_sig"]].iterrows():
            print(f"    {row['freq']} × {row['family']}: ATR win {row['atr_win_sharpe']:.0%}, "
                  f"med diff {row['med_sharpe_diff']:+.4f}, p={row['ttest_p']:.6f}")

    # ── Exhibit 1: Win rate heatmap ─────────────────────────────────
    families_to_show = cell_df.groupby("family")["n_pairs"].sum().nlargest(15).index
    heatmap_data = cell_df[cell_df["family"].isin(families_to_show)].pivot_table(
        index="family", columns="freq", values="atr_win_sharpe", aggfunc="first")
    heatmap_data = heatmap_data.reindex(columns=["1d", "4h", "1h"])

    fig, ax = plt.subplots(figsize=(8, 8))
    data_array = heatmap_data.values
    im = ax.imshow(data_array, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8)
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            val = data_array[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.3 or val > 0.7 else "black")
    plt.colorbar(im, ax=ax, label="ATR Win Rate (Sharpe)", shrink=0.8)
    ax.set_title("Task 2: ATR Win Rate by Frequency × Signal Family\n"
                 "(green = ATR wins more often, red = fixed % wins)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "atr_winrate_heatmap.png", dpi=150)
    plt.close(fig)

    # ── Exhibit 2: Sharpe difference distribution ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(-1.0, 1.0, 80)
    ax.hist(pdf["sharpe_diff"].values, bins=bins, color=TEAL, alpha=0.7,
            edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(pdf["sharpe_diff"].mean(), color=RED, lw=1.5, ls="--",
               label=f"Mean: {pdf['sharpe_diff'].mean():+.4f}")
    ax.axvline(pdf["sharpe_diff"].median(), color=NAVY, lw=1.5, ls="--",
               label=f"Median: {pdf['sharpe_diff'].median():+.4f}")

    n_pos_big = (pdf["sharpe_diff"] > 0.10).sum()
    n_neg_big = (pdf["sharpe_diff"] < -0.10).sum()
    ax.text(0.97, 0.92,
            f"|diff| > 0.10 SR:\n"
            f"  ATR better: {n_pos_big} ({n_pos_big/len(pdf):.0%})\n"
            f"  Fixed better: {n_neg_big} ({n_neg_big/len(pdf):.0%})\n"
            f"  Wash (<0.10): {len(pdf)-n_pos_big-n_neg_big} ({(len(pdf)-n_pos_big-n_neg_big)/len(pdf):.0%})",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LGRAY))

    ax.set_xlabel("Sharpe Difference (ATR minus Fixed %)")
    ax.set_ylabel("Density")
    ax.set_title("Task 2: Distribution of Matched-Pair Sharpe Differences (ATR − Fixed %)",
                 fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    fig.savefig(OUT / "sharpe_diff_distribution.png", dpi=150)
    plt.close(fig)

    # ── Exhibit 3: Win rate by frequency (bar chart) ────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    freqs = ["1d", "4h", "1h"]
    sr_wins = [pdf[pdf["freq"] == f]["atr_wins_sharpe"].mean() for f in freqs]
    dd_wins = [pdf[pdf["freq"] == f]["atr_wins_dd"].mean() for f in freqs]

    x = np.arange(3)
    ax1.bar(x, sr_wins, color=[NAVY, TEAL, GOLD], alpha=0.85, edgecolor="white")
    ax1.axhline(0.5, color=RED, lw=1, ls="--", label="50% (fair)")
    ax1.set_xticks(x); ax1.set_xticklabels(["Daily", "4-Hour", "1-Hour"])
    ax1.set_ylabel("ATR Win Rate"); ax1.set_title("A. Sharpe Win Rate", fontweight="bold")
    ax1.set_ylim(0, 0.7); ax1.legend(fontsize=8)
    for i, v in enumerate(sr_wins):
        ax1.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

    ax2.bar(x, dd_wins, color=[NAVY, TEAL, GOLD], alpha=0.85, edgecolor="white")
    ax2.axhline(0.5, color=RED, lw=1, ls="--", label="50% (fair)")
    ax2.set_xticks(x); ax2.set_xticklabels(["Daily", "4-Hour", "1-Hour"])
    ax2.set_ylabel("ATR Win Rate"); ax2.set_title("B. Max DD Win Rate", fontweight="bold")
    ax2.set_ylim(0, 0.7); ax2.legend(fontsize=8)
    for i, v in enumerate(dd_wins):
        ax2.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

    fig.suptitle("Task 2: ATR vs Fixed Stop Win Rate by Frequency", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "winrate_by_freq.png", dpi=150)
    plt.close(fig)

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "total_pairs": len(pdf),
        "overall_atr_win_sharpe": round(pdf["atr_wins_sharpe"].mean(), 4),
        "overall_atr_win_dd": round(pdf["atr_wins_dd"].mean(), 4),
        "mean_sharpe_diff": round(pdf["sharpe_diff"].mean(), 4),
        "median_sharpe_diff": round(pdf["sharpe_diff"].median(), 4),
        "pct_diff_gt_010": round((pdf["sharpe_diff"].abs() > 0.10).mean(), 4),
        "pct_atr_gt_010": round((pdf["sharpe_diff"] > 0.10).mean(), 4),
        "pct_fixed_gt_010": round((pdf["sharpe_diff"] < -0.10).mean(), 4),
        "n_cells_tested": n_cells,
        "bonferroni_threshold": round(bonf_threshold, 6),
        "n_significant_cells": n_sig,
    }
    pd.DataFrame([summary]).to_csv(OUT / "task2_summary.csv", index=False)

    print(f"\n  {'='*60}")
    print(f"  CONCLUSION")
    print(f"  {'='*60}")
    print(f"  The 31% win rate is {'stable' if n_sig == 0 else 'NOT stable'} across "
          f"frequency and signal family.")
    print(f"  {n_sig} of {n_cells} cells show Bonferroni-significant ATR advantage.")
    meaningful = (pdf["sharpe_diff"].abs() > 0.10).mean()
    print(f"  {meaningful:.0%} of pairs have meaningful (>0.10 SR) differences in either "
          f"direction — the result is "
          f"{'a true wash' if meaningful < 0.3 else 'dispersed but averaging to a wash'}.")
    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
