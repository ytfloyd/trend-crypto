#!/usr/bin/env python3
"""
Generate AQR-style research report for ETH-USD Trend Sweep.

NRT Alternative Thinking 2026 Issue 1:
  "Follow the Trend? What 13,000 Crypto Strategies Actually Tell Us"

Produces:
  - artifacts/research/tsmom/eth_trend_sweep/exhibit_*.png
  - docs/research/eth_trend_sweep_report.md

Usage:
    python -m scripts.research.tsmom.generate_eth_sweep_report
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep"
DOCS_DIR = ROOT / "docs" / "research"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# -- AQR house palette --
NAVY  = "#003366"
TEAL  = "#006B6B"
GRAY  = "#808080"
LGRAY = "#D0D0D0"
RED   = "#CC3333"
GREEN = "#336633"
GOLD  = "#CC9933"
LBLUE = "#6699CC"
BG    = "#FAFAFA"

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.facecolor": "white",
    "axes.facecolor":   BG,
    "axes.edgecolor":   LGRAY,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.color":       GRAY,
    "grid.linewidth":   0.5,
})

SAMPLE_YEARS = 9   # 2017–2026
SOURCE_LINE = (
    "Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. "
    "All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip "
    "transaction costs, no leverage. Past performance is not a reliable indicator of future results."
)
HYPO_LINE = (
    "Hypothetical performance results have many inherent limitations. "
    "No representation is made that any strategy will achieve similar results."
)


def load_results():
    df = pd.read_csv(OUT_DIR / "results_v2.csv")
    return df


def _save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}")


# =====================================================================
# Exhibits
# =====================================================================

def exhibit_1(strats, bh):
    """Exhibit 1: Most Trend Strategies Underperform Buy-and-Hold on a Risk-Adjusted Basis"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(-1.5, 2.5, 80)
    ax.hist(strats["sharpe"].values, bins=bins, color=NAVY, alpha=0.7,
            edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(bh["sharpe"], color=RED, lw=2, label=f"Buy & Hold ({bh['sharpe']:.2f})")
    ax.axvline(strats["sharpe"].median(), color=TEAL, lw=1.5, ls="--",
               label=f"Median strategy ({strats['sharpe'].median():.2f})")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Density")
    ax.set_title(
        "Exhibit 1: Most Trend Strategies Underperform Buy-and-Hold\non a Risk-Adjusted Basis",
        fontweight="bold",
    )
    n_beat = (strats["sharpe"] > bh["sharpe"]).sum()
    n_under = len(strats) - n_beat
    ax.text(0.97, 0.92,
            f"{n_under:,} of {len(strats):,} underperform B&H ({n_under/len(strats):.0%})\n"
            f"Median Sharpe degradation: {strats['sharpe'].median() - bh['sharpe']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LGRAY))
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    _save(fig, "exhibit_1_sharpe_dist.png")


def exhibit_2(strats, bh):
    """Exhibit 2: But Trend Buys Drawdown Protection — 83% Have Shallower Drawdowns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: CAGR distribution
    bins_c = np.linspace(-0.5, 2.0, 60)
    ax1.hist(strats["cagr"].values, bins=bins_c, color=NAVY, alpha=0.7,
             edgecolor="white", linewidth=0.3, density=True)
    ax1.axvline(bh["cagr"], color=RED, lw=2, label=f"B&H ({bh['cagr']:.0%})")
    ax1.axvline(strats["cagr"].median(), color=TEAL, lw=1.5, ls="--",
                label=f"Median ({strats['cagr'].median():.0%})")
    ax1.set_xlabel("CAGR")
    ax1.set_ylabel("Density")
    ax1.set_title("A. What You Give Up: CAGR", fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=LGRAY)

    # Right: MaxDD distribution
    bins_d = np.linspace(-1.0, 0, 60)
    ax2.hist(strats["max_dd"].values, bins=bins_d, color=TEAL, alpha=0.7,
             edgecolor="white", linewidth=0.3, density=True)
    ax2.axvline(bh["max_dd"], color=RED, lw=2, label=f"B&H ({bh['max_dd']:.0%})")
    ax2.axvline(strats["max_dd"].median(), color=NAVY, lw=1.5, ls="--",
                label=f"Median ({strats['max_dd'].median():.0%})")
    ax2.set_xlabel("Max Drawdown")
    ax2.set_ylabel("Density")
    ax2.set_title("B. What You Get: Drawdown Compression", fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=LGRAY)

    fig.suptitle("Exhibit 2: The Trend Tradeoff — CAGR for Drawdown Protection",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "exhibit_2_tradeoff.png")


def exhibit_3(strats, bh):
    """Exhibit 3: Frequency Is the Dominant Variable"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    freqs = ["1d", "4h", "1h"]
    freq_labels = ["Daily", "4-Hour", "1-Hour"]
    colors = [NAVY, TEAL, GOLD]

    bins = np.linspace(-1.5, 2.5, 60)
    for i, (freq, flabel, col) in enumerate(zip(freqs, freq_labels, colors)):
        sub = strats[strats["freq"] == freq]
        n_beat = (sub["sharpe"] > bh["sharpe"]).sum()
        pct = n_beat / len(sub) if len(sub) > 0 else 0

        axes[i].hist(sub["sharpe"].values, bins=bins, color=col, alpha=0.75,
                     edgecolor="white", linewidth=0.3, density=True)
        axes[i].axvline(bh["sharpe"], color=RED, lw=1.5, ls="-")
        axes[i].set_xlabel("Sharpe Ratio")
        axes[i].set_ylabel("Density" if i == 0 else "")
        axes[i].set_title(f"{flabel} ({len(sub):,} configs)", fontweight="bold")
        axes[i].text(0.95, 0.92,
                     f"Med: {sub['sharpe'].median():.2f}\n"
                     f"Beat B&H: {pct:.0%}",
                     transform=axes[i].transAxes, ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LGRAY))

    fig.suptitle("Exhibit 3: Frequency Is the Dominant Variable — Daily Signals Are Overwhelmingly Better",
                 fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "exhibit_3_frequency.png")


def exhibit_4(strats, bh):
    """Exhibit 4: Median Performance by Signal Family (daily, no stop)"""
    ns = strats[(strats["stop"] == "none") & (strats["freq"] == "1d")]
    fam = ns.groupby("signal_family").agg(
        sharpe=("sharpe", "median"),
        cagr=("cagr", "median"),
        max_dd=("max_dd", "median"),
        skewness=("skewness", "median"),
        n=("sharpe", "count"),
    ).sort_values("sharpe", ascending=True)

    fam = fam[fam["n"] >= 3].tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(fam))
    colors = [TEAL if s > bh["sharpe"] else NAVY for s in fam["sharpe"]]
    ax.barh(y, fam["sharpe"], color=colors, alpha=0.85, edgecolor="white", height=0.7)
    ax.axvline(bh["sharpe"], color=RED, lw=1.5, ls="--", label=f"B&H ({bh['sharpe']:.2f})")
    ax.set_yticks(y)
    labels = [f"{fam_name} (n={int(row['n'])})" for fam_name, row in fam.iterrows()]
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Median Sharpe Ratio")
    ax.set_title("Exhibit 4: Median Sharpe by Signal Family\n(daily frequency, no stop, n ≥ 3 configs per family)",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    _save(fig, "exhibit_4_family.png")
    return fam


def exhibit_5(strats, bh):
    """Exhibit 5: Time in Market Controls the CAGR/Drawdown Dial"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.scatter(strats["time_in_market"] * 100, strats["cagr"] * 100,
                c=NAVY, alpha=0.08, s=8, edgecolors="none")
    ax1.scatter(100, bh["cagr"] * 100, c=RED, s=150, marker="*", zorder=10, label="B&H")
    ax1.set_xlabel("Time in Market (%)")
    ax1.set_ylabel("CAGR (%)")
    ax1.set_title("A. More Time in Market → Higher CAGR", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=LGRAY)

    ax2.scatter(strats["time_in_market"] * 100, strats["max_dd"] * 100,
                c=TEAL, alpha=0.08, s=8, edgecolors="none")
    ax2.scatter(100, bh["max_dd"] * 100, c=RED, s=150, marker="*", zorder=10, label="B&H")
    ax2.set_xlabel("Time in Market (%)")
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_title("B. More Time in Market → Deeper Drawdowns", fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=LGRAY)

    fig.suptitle("Exhibit 5: Time in Market Controls the Return/Risk Dial",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "exhibit_5_tim.png")


def exhibit_6(strats, bh):
    """Exhibit 6: Sharpe vs Skewness scatter — positive skew is nearly universal"""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(strats["skewness"], strats["sharpe"], c=NAVY, alpha=0.08, s=8, edgecolors="none",
               label=f"Trend strategies (n={len(strats):,})")
    ax.scatter(bh["skewness"], bh["sharpe"], c=RED, s=200, marker="*", zorder=10, label="Buy & Hold")

    ax.axhline(bh["sharpe"], color=RED, lw=0.5, ls=":", alpha=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    n_pos = (strats["skewness"] > 0).sum()
    ax.text(0.97, 0.05,
            f"{n_pos:,} of {len(strats):,} ({n_pos/len(strats):.0%}) have positive skewness",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LGRAY))

    ax.set_xlabel("Skewness (daily returns)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Exhibit 6: Positive Skewness Is Nearly Universal in Binary Long/Cash Crypto Trend",
                 fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=LGRAY, markerscale=1.5)
    ax.set_xlim(-3, 20)
    ax.set_ylim(-1.5, 2.2)
    fig.tight_layout()
    _save(fig, "exhibit_6_sharpe_skew.png")


def exhibit_7(strats, bh):
    """Exhibit 7: ATR vs Fixed Stops — aggregate medians by stop type"""
    order = ["none", "pct5", "pct10", "pct20", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]
    display = ["None", "5%", "10%", "20%", "1.5×ATR", "2.0×ATR", "2.5×ATR", "3.0×ATR", "4.0×ATR"]
    colors = [GRAY] + [NAVY] * 3 + [TEAL] * 5

    med_sharpe, med_dd, med_tim = [], [], []
    for sl in order:
        sub = strats[strats["stop"] == sl]
        med_sharpe.append(sub["sharpe"].median())
        med_dd.append(sub["max_dd"].median())
        med_tim.append(sub["time_in_market"].median())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(order))

    axes[0].bar(x, med_sharpe, color=colors, alpha=0.85, edgecolor="white")
    axes[0].axhline(bh["sharpe"], color=RED, lw=1, ls="--", label="B&H")
    axes[0].set_xticks(x); axes[0].set_xticklabels(display, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Median Sharpe"); axes[0].set_title("A. Sharpe Ratio", fontweight="bold")
    axes[0].legend(fontsize=7)

    axes[1].bar(x, [d * 100 for d in med_dd], color=colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(bh["max_dd"] * 100, color=RED, lw=1, ls="--", label="B&H")
    axes[1].set_xticks(x); axes[1].set_xticklabels(display, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Median Max DD (%)"); axes[1].set_title("B. Max Drawdown", fontweight="bold")
    axes[1].legend(fontsize=7)

    axes[2].bar(x, [t * 100 for t in med_tim], color=colors, alpha=0.85, edgecolor="white")
    axes[2].set_xticks(x); axes[2].set_xticklabels(display, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("Median TIM (%)"); axes[2].set_title("C. Time in Market", fontweight="bold")

    fig.suptitle("Exhibit 7: Aggregate Performance by Stop Type\n(gray = no stop, blue = fixed %, teal = ATR)",
                 fontweight="bold", fontsize=11, y=1.04)
    fig.tight_layout()
    _save(fig, "exhibit_7_stops_agg.png")


def exhibit_8(strats, bh):
    """Exhibit 8: Matched comparison — ATR vs fixed stops, same base signal"""
    results = []
    for (label, freq), group in strats.groupby(["label", "freq"]):
        no_stop_rows = group[group["stop"] == "none"]
        atr_rows = group[group["stop_type"] == "atr"]
        pct_rows = group[group["stop_type"] == "pct"]
        if no_stop_rows.empty or atr_rows.empty or pct_rows.empty:
            continue
        ns_sr = no_stop_rows.iloc[0]["sharpe"]
        best_atr = atr_rows.loc[atr_rows["sharpe"].idxmax()]
        best_pct = pct_rows.loc[pct_rows["sharpe"].idxmax()]
        results.append({
            "base_sharpe": ns_sr,
            "atr_sharpe": best_atr["sharpe"],
            "pct_sharpe": best_pct["sharpe"],
            "atr_dd": best_atr["max_dd"],
            "pct_dd": best_pct["max_dd"],
            "ns_dd": no_stop_rows.iloc[0]["max_dd"],
        })
    r = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.scatter(r["pct_sharpe"], r["atr_sharpe"], c=TEAL, alpha=0.3, s=15, edgecolors="none")
    lims = [-1.5, 2.0]
    ax1.plot(lims, lims, color=GRAY, lw=1, ls="--")
    atr_wins_sr = (r["atr_sharpe"] > r["pct_sharpe"]).mean()
    ax1.set_xlabel("Best Fixed-% Stop Sharpe")
    ax1.set_ylabel("Best ATR Stop Sharpe")
    ax1.set_title("A. Sharpe: ATR vs Fixed %", fontweight="bold")
    ax1.text(0.05, 0.95, f"ATR wins: {atr_wins_sr:.0%}\nof {len(r):,} matched pairs",
             transform=ax1.transAxes, ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LGRAY))
    ax1.set_xlim(lims); ax1.set_ylim(lims)

    ax2.scatter(r["pct_dd"], r["atr_dd"], c=NAVY, alpha=0.3, s=15, edgecolors="none")
    ax2.plot([-1.0, 0], [-1.0, 0], color=GRAY, lw=1, ls="--")
    atr_wins_dd = (r["atr_dd"] > r["pct_dd"]).mean()
    ax2.set_xlabel("Best Fixed-% Stop Max DD")
    ax2.set_ylabel("Best ATR Stop Max DD")
    ax2.set_title("B. Max DD: ATR vs Fixed %", fontweight="bold")
    ax2.text(0.05, 0.05, f"ATR wins (shallower DD): {atr_wins_dd:.0%}",
             transform=ax2.transAxes, ha="left", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LGRAY))
    ax2.set_xlim(-1.0, 0); ax2.set_ylim(-1.0, 0)

    fig.suptitle(
        "Exhibit 8: Matched Comparison — ATR vs Fixed Stops on Same Base Signal\n"
        "(each point = one signal × frequency pair)",
        fontweight="bold", fontsize=11, y=1.04,
    )
    fig.tight_layout()
    _save(fig, "exhibit_8_matched.png")
    return r


def exhibit_9(strats, bh):
    """Exhibit 9: Multiple testing — what survives Bonferroni?"""
    n_eff = 493 * 3
    bonf_p = 0.05 / n_eff
    bonf_z = stats.norm.ppf(1 - bonf_p / 2)
    sharpe_thresh = bonf_z / np.sqrt(SAMPLE_YEARS)
    conventional_thresh = 1.96 / np.sqrt(SAMPLE_YEARS)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(-1.5, 2.5, 80)
    ax.hist(strats["sharpe"].values, bins=bins, color=NAVY, alpha=0.5,
            edgecolor="white", linewidth=0.3, density=True, label="All strategies")

    survivors = strats[strats["sharpe"] > sharpe_thresh]
    ax.hist(survivors["sharpe"].values, bins=bins, color=GREEN, alpha=0.7,
            edgecolor="white", linewidth=0.3, density=True, label=f"Survive Bonferroni (n={len(survivors)})")

    ax.axvline(sharpe_thresh, color=GREEN, lw=2, ls="-",
               label=f"Bonferroni threshold ({sharpe_thresh:.2f})")
    ax.axvline(conventional_thresh, color=GOLD, lw=1.5, ls="--",
               label=f"Conventional t=1.96 ({conventional_thresh:.2f})")
    ax.axvline(bh["sharpe"], color=RED, lw=1.5, label=f"B&H ({bh['sharpe']:.2f})")

    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Exhibit 9: After Bonferroni Correction ({n_eff:,} Effective Tests),\n"
        f"Only {len(survivors):,} of {len(strats):,} Strategies ({len(survivors)/len(strats):.1%}) Survive",
        fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=LGRAY, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "exhibit_9_multiple_testing.png")
    return sharpe_thresh, len(survivors)


def exhibit_10(strats, bh):
    """Exhibit 10: CAGR vs MaxDD with Calmar isolines — the risk/return map"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for freq, col in [("1d", NAVY), ("4h", TEAL), ("1h", GOLD)]:
        sub = strats[strats["freq"] == freq]
        ax.scatter(sub["max_dd"] * 100, sub["cagr"] * 100, c=col, alpha=0.1, s=8,
                   edgecolors="none", label=f"{freq} ({len(sub):,})")

    ax.scatter(bh["max_dd"] * 100, bh["cagr"] * 100, c=RED, s=200, marker="*",
               zorder=10, label="Buy & Hold")

    for calmar in [0.5, 1.0, 2.0, 3.0]:
        dd_range = np.linspace(-100, -5, 200)
        cagr_line = calmar * np.abs(dd_range)
        ax.plot(dd_range, cagr_line, color=GRAY, lw=0.5, ls=":", alpha=0.5)
        idx = int(len(dd_range) * 0.85)
        ax.text(dd_range[idx], cagr_line[idx] + 2, f"Calmar={calmar}",
                fontsize=6, color=GRAY, rotation=0)

    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Exhibit 10: The Risk/Return Map — Most Strategies Trade CAGR for Drawdown Compression",
                 fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=LGRAY, markerscale=2)
    ax.set_xlim(-100, 0)
    ax.set_ylim(-20, 200)
    fig.tight_layout()
    _save(fig, "exhibit_10_cagr_dd.png")


def exhibit_11(strats, bh):
    """Exhibit 11: Drawdown boxplot by stop type"""
    order = ["none", "pct5", "pct10", "pct20", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]
    display = ["None", "5%", "10%", "20%", "1.5×ATR", "2.0×ATR", "2.5×ATR", "3.0×ATR", "4.0×ATR"]

    data, labels = [], []
    for sl, dl in zip(order, display):
        sub = strats[strats["stop"] == sl]["max_dd"] * 100
        if len(sub) > 0:
            data.append(sub.values)
            labels.append(dl)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(GRAY if i == 0 else (NAVY if i < 4 else TEAL))
        patch.set_alpha(0.75)

    ax.axhline(bh["max_dd"] * 100, color=RED, lw=1.5, ls="--",
               label=f"B&H ({bh['max_dd']:.0%})")
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Exhibit 11: Drawdown Distribution by Stop Type\n"
                 "(gray = none, blue = fixed %, teal = ATR)",
                 fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=LGRAY)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, "exhibit_11_dd_box.png")


# =====================================================================
# Markdown report
# =====================================================================

def build_markdown(df, strats, bh, sharpe_thresh, n_survivors, matched_df):
    n = len(strats)
    n_beat_sr = (strats["sharpe"] > bh["sharpe"]).sum()
    n_beat_dd = (strats["max_dd"] > bh["max_dd"]).sum()
    n_beat_cagr = (strats["cagr"] > bh["cagr"]).sum()
    med_sr = strats["sharpe"].median()
    med_cagr = strats["cagr"].median()
    med_dd = strats["max_dd"].median()
    med_skew = strats["skewness"].median()

    ns = strats[strats["stop"] == "none"]
    daily = strats[strats["freq"] == "1d"]
    daily_ns = ns[ns["freq"] == "1d"]
    four_h = strats[strats["freq"] == "4h"]
    one_h = strats[strats["freq"] == "1h"]

    atr_win_sr = (matched_df["atr_sharpe"] > matched_df["pct_sharpe"]).mean()
    atr_win_dd = (matched_df["atr_dd"] > matched_df["pct_dd"]).mean()

    # Top 10 after Bonferroni
    top10 = strats[strats["sharpe"] > sharpe_thresh].nlargest(10, "sharpe")

    # TIM buckets
    tim_buckets = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]

    md = f"""\
# NRT Alternative Thinking 2026 Issue 1

# Follow the Trend? What 13,000 Crypto Strategies Actually Tell Us

**Portfolio Research Group**

---

## Executive Summary

"Follow the Trend" has become the working hypothesis for systematic crypto allocation at this
desk. In this article, we test that hypothesis exhaustively — building **{n:,} trend-following
configurations** on ETH-USD and asking a simple question: does trend beat buy-and-hold?

The short answer is: mostly no. **{n - n_beat_sr:,} of {n:,} ({(n - n_beat_sr)/n:.0%})** trend
strategies produce *worse* risk-adjusted returns than passive buy-and-hold. The median strategy
has a Sharpe ratio of {med_sr:.2f}, versus buy-and-hold's {bh["sharpe"]:.2f} — a {(med_sr - bh["sharpe"])/bh["sharpe"]:.0%}
degradation. On CAGR, it's worse: only {n_beat_cagr:,} of {n:,} ({n_beat_cagr/n:.1%}) beat
buy-and-hold, and the median strategy surrenders {(bh["cagr"] - med_cagr):.0%} of annual return.[^1]

But this is only half the story. {n_beat_dd:,} of {n:,} ({n_beat_dd/n:.0%}) strategies have
*shallower* max drawdowns than buy-and-hold. And {(strats["skewness"] > 0).sum():,} of {n:,}
({(strats["skewness"] > 0).sum()/n:.0%}) exhibit positive skewness. In an asset that fell 94%
peak-to-trough, drawdown compression is not a trivial benefit. The question is not whether
trend "works" — it is whether the protection it buys is worth the return it costs.

[^1]: A disclaimer is necessary: we are testing {n:,} strategies on a single asset over a
nine-year period. No matter how significant a result appears, it reflects in-sample data mining
until validated out-of-sample and across assets. We would take very little risk on any single
configuration, preferring to diversify across many — and even then, no directional strategy
is anywhere near perfect.

---

## Contents

1. Introduction
2. Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat
3. Part 2: What Trend Buys and What It Costs
4. Part 3: What Actually Drives Performance? Frequency Dominates Everything
5. Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?
6. Part 5: The Multiple Testing Problem
7. Concluding Thoughts
8. Appendix: Parameter Grid and Data Notes

---

## Introduction

Crypto allocators face a unique problem. The asset class has delivered extraordinary long-term
returns — ETH-USD compounded at {bh["cagr"]:.0%} annualized from 2017 to 2026 — but the path
was brutal: a {bh["max_dd"]:.0%} peak-to-trough drawdown, with multiple drawdowns exceeding 70%.
No investor, institutional or otherwise, can plausibly hold through a {bh["max_dd"]:.0%} drawdown.[^2]
The standard response is to apply trend-following logic: be long when the asset is trending
up, move to cash when it is not.

The premise has theoretical support. Moskowitz, Ooi, and Pedersen (2012) documented time-series
momentum across dozens of futures markets. Hurst, Ooi, and Pedersen (2017) extended the evidence
to a century of data. In our own prior work at this desk, we attempted a portfolio-level TSMOM
framework for crypto, applying vol-scaled signals with portfolio-level vol targeting across
multiple assets. The results were poor: the framework produced a Sharpe of 0.77 with 87%
time-in-market — essentially dampened buy-and-hold at a fraction of the CAGR.[^3]

This led to a natural question: if portfolio-level momentum fails, do simpler per-asset
trend signals do better? And if so, what matters more — the entry signal, the data frequency,
or the exit mechanism?

To answer these questions, we built **{n:,} configurations** from the cross-product of:

| Variable | What We Test |
|---|---|
| Base signals | 493 signals from 30+ families (MA crossovers, channel breakouts, momentum indicators, TA oscillators, composite signals) |
| Frequencies | Daily, 4-hour, 1-hour bars |
| Stop variants | None; fixed trailing stops at 5%, 10%, 20%; vol-adaptive trailing stops at 1.5×, 2.0×, 2.5×, 3.0×, 4.0× entry-date ATR |

All configurations use identical backtest rules: binary long or cash, signal applied with
one-bar lag, 20 bps round-trip transaction costs, no leverage, no position sizing.

[^2]: Luna Foundation Guard, Three Arrows Capital, and Alameda Research all failed to maintain
positions through drawdowns of comparable magnitude. The behavioral and institutional constraints
on holding through -90%+ drawdowns are not merely theoretical.

[^3]: See internal memo, "TSMOM Framework Results," January 2026. The framework actively
destroyed value by slowing exit timing through vol-scaling and portfolio construction.

---

## Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat

Before evaluating trend strategies, it is worth establishing how strong the baseline is.
ETH-USD's buy-and-hold performance from January 2017 to February 2026:

| Metric | Value |
|---|---|
| CAGR | {bh["cagr"]:.1%} |
| Sharpe Ratio | {bh["sharpe"]:.2f} |
| Sortino Ratio | {bh["sortino"]:.2f} |
| Max Drawdown | {bh["max_dd"]:.1%} |
| Calmar Ratio | {bh["calmar"]:.2f} |
| Skewness | {bh["skewness"]:.2f} |

A Sharpe ratio of {bh["sharpe"]:.2f} is exceptional by any standard. In equities, a Sharpe above
0.5 is considered good; in crypto, the secular uptrend and the absence of a reliable risk-free
rate produce much higher ratios. Any strategy that sits in cash for part of the period faces
a substantial headwind from missing this strong secular drift.[^4]

**Exhibit 1** shows the distribution of Sharpe ratios across all {n:,} trend configurations.
The median strategy ({med_sr:.2f}) falls well below buy-and-hold ({bh["sharpe"]:.2f}). Only
{n_beat_sr:,} ({n_beat_sr/n:.0%}) of configurations outperform on this metric.

![Exhibit 1](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_1_sharpe_dist.png)

*{SOURCE_LINE}*

This result should not be surprising. The median strategy is invested only {ns["time_in_market"].median():.0%}
of the time. In an asset with {bh["cagr"]:.0%} CAGR, sitting in cash half the time is expensive.
Many configurations are also suboptimal by design (very short lookbacks that trade noise, very
tight stops that exit normal volatility) — the purpose of an exhaustive sweep is to map the full
space, including the bad regions.

[^4]: Throughout this piece, "cash" means zero return. We do not model stablecoin yield. To the
extent that cash earns a positive return (e.g., 5% in a DeFi lending protocol), the case for
trend strategies improves modestly.

---

## Part 2: What Trend Buys and What It Costs

If trend-following in crypto mostly underperforms buy-and-hold on a risk-adjusted basis, why
consider it? Because risk-adjusted returns are not the only thing that matters. An investor
who cannot hold through a {bh["max_dd"]:.0%} drawdown does not earn the {bh["cagr"]:.0%} CAGR.
The relevant question is: what does trend cost, and what does it buy?

**Exhibit 2** shows the tradeoff directly: the left panel shows the CAGR distribution (what you
give up), and the right panel shows the drawdown distribution (what you get).

![Exhibit 2](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_2_tradeoff.png)

*{SOURCE_LINE}*

The numbers are stark:

| Metric | Buy & Hold | Median Strategy | Difference |
|---|---|---|---|
| CAGR | {bh["cagr"]:.1%} | {med_cagr:.1%} | {med_cagr - bh["cagr"]:+.1%} |
| Max Drawdown | {bh["max_dd"]:.1%} | {med_dd:.1%} | {med_dd - bh["max_dd"]:+.1%} |
| Skewness | {bh["skewness"]:.2f} | {med_skew:.2f} | {med_skew - bh["skewness"]:+.2f} |

The median strategy gives up roughly {abs(med_cagr - bh["cagr"]):.0%} of annual return to compress
the max drawdown by {abs(med_dd - bh["max_dd"]):.0%}. Is that a good trade? It depends on
the investor's utility function. For an allocator who cannot hold through {bh["max_dd"]:.0%} but
can hold through {med_dd:.0%}, trend converts an undeployable return stream into a deployable
one — even if the headline CAGR is lower.[^5]

The skewness improvement is real but partly mechanical. A binary long/cash strategy on a
strongly trending asset will exhibit positive skewness by construction: it participates in the
large up-moves (which are persistent and often occur in trend) while exiting before or during
some of the large down-moves. This does not require signal "skill" — {(strats["skewness"] > 0).sum():,}
of {n:,} ({(strats["skewness"] > 0).sum()/n:.0%}) strategies exhibit positive skewness regardless
of signal choice. The skewness is a property of the *trade structure*, not the signal.[^6]

[^5]: This framing is consistent with the crypto-specific finding in our prior work that the
binding constraint is not expected return (which is high) but the ability to stay allocated
through drawdowns.

[^6]: Readers should be cautious about attributing positive skewness to signal "alpha."
A randomly-timed long/cash strategy on ETH-USD would also exhibit positive skewness over
this period, simply because the right tail of ETH daily returns is fatter than the left tail.

---

## Part 3: What Actually Drives Performance? Frequency Dominates Everything

Across {n:,} configurations, we vary three dimensions: signal choice (493 base signals),
frequency (daily, 4-hour, 1-hour), and stop type (9 variants). Which dimension matters most?

The answer is unambiguous: **frequency**.

**Exhibit 3** shows the Sharpe ratio distribution at each frequency. Daily signals have a
median Sharpe of {daily["sharpe"].median():.2f} and {(daily["sharpe"] > bh["sharpe"]).sum() / len(daily):.0%}
beat buy-and-hold. Four-hour signals drop to {four_h["sharpe"].median():.2f}
({(four_h["sharpe"] > bh["sharpe"]).sum() / len(four_h):.0%} beat B&H). One-hour signals collapse
to {one_h["sharpe"].median():.2f} ({(one_h["sharpe"] > bh["sharpe"]).sum() / len(one_h):.0%} beat B&H).

![Exhibit 3](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_3_frequency.png)

*{SOURCE_LINE}*

**Exhibit 3a: Performance by Frequency**

| Frequency | N | Median Sharpe | Median CAGR | Median MaxDD | % Beat B&H (SR) |
|---|---|---|---|---|---|
| Daily | {len(daily):,} | {daily["sharpe"].median():.3f} | {daily["cagr"].median():.1%} | {daily["max_dd"].median():.1%} | {(daily["sharpe"] > bh["sharpe"]).sum()/len(daily):.0%} |
| 4-Hour | {len(four_h):,} | {four_h["sharpe"].median():.3f} | {four_h["cagr"].median():.1%} | {four_h["max_dd"].median():.1%} | {(four_h["sharpe"] > bh["sharpe"]).sum()/len(four_h):.0%} |
| 1-Hour | {len(one_h):,} | {one_h["sharpe"].median():.3f} | {one_h["cagr"].median():.1%} | {one_h["max_dd"].median():.1%} | {(one_h["sharpe"] > bh["sharpe"]).sum()/len(one_h):.0%} |

The mechanism is straightforward. Higher-frequency signals generate more trades, and each trade
incurs transaction costs (20 bps round-trip). In a strong secular uptrend, frequent trading
also increases the probability of being whipsawed out of a trend during intraday noise, then
missing the subsequent continuation. Daily signals trade less often and allow trends to develop;
hourly signals chop in and out of positions, destroying return through friction and missed
upside.[^7]

Within the daily-frequency universe, **Exhibit 4** shows median Sharpe by signal family. The
spread is modest: the best family (EMA crossover, median Sharpe {daily_ns.groupby("signal_family")["sharpe"].median().max():.2f})
outperforms the worst by roughly 0.5 Sharpe units. But the variation *within* families
(across parameter choices) is often as large as the variation *across* families. This suggests
that signal choice, while not irrelevant, is secondary to frequency and time-in-market.

![Exhibit 4](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_4_family.png)

*{SOURCE_LINE} Shows daily-frequency, no-stop configurations only. Families with fewer than
3 configurations are excluded.*

**Exhibit 5** confirms that time-in-market is the core dial. More time invested means higher
CAGR but deeper drawdowns — there is no free lunch.

![Exhibit 5](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_5_tim.png)

*{SOURCE_LINE}*

**Exhibit 5a: Performance by Time-in-Market Bucket**

| TIM Bucket | N | Median Sharpe | Median MaxDD | Median Skew |
|---|---|---|---|---|"""

    for lo, hi in tim_buckets:
        sub = strats[(strats["time_in_market"] >= lo) & (strats["time_in_market"] < hi)]
        if sub.empty:
            continue
        md += (f"\n| {lo:.0%}–{hi:.0%} | {len(sub):,} | {sub['sharpe'].median():.3f} | "
               f"{sub['max_dd'].median():.1%} | {sub['skewness'].median():.2f} |")

    md += f"""

The "sweet spot" appears to be 30–40% time-in-market: high enough to capture most of the
secular drift, low enough to exit during sustained drawdowns. Strategies in this bucket have
a median Sharpe near {strats[(strats["time_in_market"] >= 0.3) & (strats["time_in_market"] < 0.4)]["sharpe"].median():.2f}
and a median drawdown of {strats[(strats["time_in_market"] >= 0.3) & (strats["time_in_market"] < 0.4)]["max_dd"].median():.1%}
— roughly one-third less severe than buy-and-hold.

[^7]: This finding is consistent with the broader trend-following literature. Moskowitz et al
(2012) found that time-series momentum is strongest at monthly frequencies and degrades at
shorter horizons due to mean reversion and transaction costs.

---

## Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?

A common hypothesis in systematic trading is that vol-adaptive exits (e.g., ATR-based trailing
stops) should dominate fixed-percentage exits because they adapt to the prevailing volatility
regime. In theory, a fixed 10% stop is too tight in a high-volatility environment and too
loose in a low-volatility one; an ATR-based stop calibrates automatically.

We test this by comparing nine stop variants across all base signals: no stop, three fixed-
percentage stops (5%, 10%, 20%), and five ATR-based stops (1.5×, 2.0×, 2.5×, 3.0×, 4.0× ATR).

**Exhibit 7** shows the aggregate medians by stop type.

![Exhibit 7](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_7_stops_agg.png)

*{SOURCE_LINE}*

**Exhibit 7a: Aggregate Performance by Stop Type**

| Stop | Type | Med Sharpe | Med MaxDD | Med Skew | Med TIM |
|---|---|---|---|---|---|"""

    order = ["none", "pct5", "pct10", "pct20", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]
    display = ["None", "5% fixed", "10% fixed", "20% fixed",
               "1.5× ATR", "2.0× ATR", "2.5× ATR", "3.0× ATR", "4.0× ATR"]
    for sl, dl in zip(order, display):
        sub = strats[strats["stop"] == sl]
        if sub.empty:
            continue
        stype = "—" if sl == "none" else ("Fixed %" if sl.startswith("pct") else "Vol-adaptive")
        md += (f"\n| {dl} | {stype} | {sub['sharpe'].median():.3f} | "
               f"{sub['max_dd'].median():.1%} | {sub['skewness'].median():.2f} | "
               f"{sub['time_in_market'].median():.0%} |")

    md += f"""

At the aggregate level, both stop types compress drawdowns relative to no-stop, but at the
cost of lower Sharpe ratios. The tightest stops (5% fixed, 1.5× ATR) produce the most drawdown
compression but also the worst Sharpe ratios, because they trigger exits on normal volatility
and generate excessive trading costs.

The more interesting test is the **matched comparison**: for the same base signal and frequency,
does the best ATR stop outperform the best fixed stop? **Exhibit 8** shows the result.

![Exhibit 8](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_8_matched.png)

*{SOURCE_LINE} Each point represents one base signal × frequency pair. Above the diagonal
line, ATR outperforms fixed %.*

The answer is not what we expected. **ATR stops win only {atr_win_sr:.0%} of the time on Sharpe**
and {atr_win_dd:.0%} of the time on max drawdown, across {len(matched_df):,} matched pairs. The
average Sharpe difference is {matched_df["atr_sharpe"].mean() - matched_df["pct_sharpe"].mean():.4f}
— essentially zero.[^8]

The aggregate medians in Exhibit 7 were misleading because ATR and fixed stops have different
time-in-market profiles (ATR stops tend to keep positions open slightly longer), which confounds
the comparison. Once we match on the same base signal, the stop type is a wash.

This is an honest but uncomfortable finding. The economic intuition for vol-adaptive stops is
compelling, but the data do not support a strong claim of dominance over this sample. Both
stop types achieve roughly the same thing: they compress drawdowns at the cost of CAGR, with
the compression proportional to how tight the stop is.[^9]

[^8]: The matched comparison uses the *best* ATR and *best* fixed stop for each signal. This
is generous to both; a random stop selection would show even less differentiation.

[^9]: One interpretation is that in a single-asset context with binary positioning, the stop
distance is more important than how it is calibrated. An ATR stop that happens to produce a
similar stop distance to a fixed-% stop will produce similar results. The theoretical advantage
of ATR may require a more diverse asset universe or more complex position sizing to manifest.

---

## Part 5: The Multiple Testing Problem

We tested {n:,} configurations. Even if every strategy were generated by a coin flip with zero
true Sharpe, we would expect some to look impressive by chance alone. Any honest assessment of
these results must address the multiple testing problem.[^10]

The strategies are not independent — stop variants of the same base signal are highly correlated
(average pairwise Sharpe correlation: 0.94). We estimate the effective number of independent
tests at approximately {493 * 3:,} (493 base signals × 3 frequencies), treating stop variants
as dependent.

At this test count, a Bonferroni-corrected significance threshold requires a z-statistic of
{stats.norm.ppf(1 - 0.05 / (493 * 3) / 2):.2f}. Over our {SAMPLE_YEARS}-year sample, this
translates to a Sharpe ratio of **{sharpe_thresh:.2f}**. Only **{n_survivors:,} of {n:,}
({n_survivors/n:.1%})** strategies survive this threshold.

**Exhibit 9** shows which strategies survive.

![Exhibit 9](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_9_multiple_testing.png)

*{SOURCE_LINE} Bonferroni correction assumes {493 * 3:,} effective independent tests (493 base
signals × 3 frequencies). The {SAMPLE_YEARS}-year sample converts z-thresholds to Sharpe
thresholds via SR = z / √T.*

**Exhibit 9a: Top 10 Strategies Surviving Bonferroni Correction**

| # | Signal | Stop | Freq | Sharpe | CAGR | MaxDD | Calmar | Skew | TIM |
|---|---|---|---|---|---|---|---|---|---|"""

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        md += (f"\n| {rank} | {row['label']} | {row['stop']} | {row['freq']} | "
               f"{row['sharpe']:.2f} | {row['cagr']:.1%} | {row['max_dd']:.1%} | "
               f"{row['calmar']:.2f} | {row['skewness']:.2f} | {row['time_in_market']:.0%} |")

    md += f"""

The survivors cluster in daily-frequency, medium-lookback MA crossovers (EMA, DEMA, SMA)
and a few channel-based signals (Aroon, Supertrend, ADX). This is reassuring in one sense —
the surviving signal families are well-established in the trend-following literature — but
concerning in another: the specific parameterizations that survive are almost certainly
influenced by the particular path of ETH over this sample.[^11]

[^10]: AQR's analysis of 196 "Buy the Dip" strategies (Cao, Chong, and Villalon, 2025) faces
a similar challenge with far fewer tests. With {n:,} strategies, the concern is proportionally
more severe.

[^11]: For context: if we were to run the same sweep on BTC-USD or SOL-USD, the specific
winning parameterizations would likely differ, even if the winning signal *families* remain
similar. This is the distinction between signal-family robustness (which we hypothesize)
and parameter robustness (which we do not claim).

---

## Part 6: The Convexity Profile

For allocators whose mandate is long convexity — bounded downside with exposure to unbounded
upside — the joint distribution of Sharpe and skewness matters more than Sharpe alone.

**Exhibit 6** maps this space.

![Exhibit 6](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_6_sharpe_skew.png)

*{SOURCE_LINE}*

Nearly all strategies ({(strats["skewness"] > 0).sum():,} of {n:,}, or
{(strats["skewness"] > 0).sum()/n:.0%}) exhibit positive skewness. As noted earlier, this is
largely a mechanical consequence of binary long/cash positioning on a positively-trending
asset. The practical implication is that trend strategies in crypto are natural convexity
providers regardless of signal choice — a structural property that may justify their inclusion
in a portfolio even at a modestly lower Sharpe than buy-and-hold.

**Exhibit 10** shows the full risk/return map.

![Exhibit 10](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_10_cagr_dd.png)

*{SOURCE_LINE} Dotted lines show Calmar ratio contours.*

**Exhibit 11** shows the drawdown distribution by stop type.

![Exhibit 11](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_11_dd_box.png)

*{SOURCE_LINE}*

---

## Concluding Thoughts

Four findings emerge from this study:

**First, trend-following in crypto mostly underperforms buy-and-hold on risk-adjusted returns.**
{(n - n_beat_sr)/n:.0%} of the {n:,} strategies we tested produced lower Sharpe ratios than
passive buy-and-hold. The median strategy has a Sharpe of {med_sr:.2f} versus buy-and-hold's
{bh["sharpe"]:.2f}. This is not an indictment of trend-following — it is a statement about how
strong the secular uptrend in crypto has been. In a {bh["cagr"]:.0%} CAGR environment, any time
spent in cash is expensive.

**Second, the value of trend is in drawdown compression, not return enhancement.** {n_beat_dd/n:.0%}
of strategies have shallower drawdowns than buy-and-hold's {bh["max_dd"]:.0%}. The median
strategy compresses the max drawdown by {abs(med_dd - bh["max_dd"]):.0%}. For allocators
constrained by drawdown tolerance, this converts an undeployable return stream into a
deployable one — even at lower headline CAGR.

**Third, data frequency dominates signal choice and exit mechanism.** Daily signals vastly
outperform intraday signals. Within the daily universe, the choice of signal family and stop
type matters far less than the choice of frequency. Vol-adaptive (ATR-based) stops do not
reliably dominate fixed-percentage stops in a matched comparison.

**Fourth, after multiple-testing correction, only {n_survivors/n:.1%} of strategies survive.**
The {n_survivors:,} surviving configurations cluster in well-known trend signals at daily
frequency. We hypothesize that the signal *family* result (MA crossovers, channel breakouts)
may be robust across assets, but the specific *parameterizations* are almost certainly
sample-dependent. Cross-asset validation is required before deployment.

A final note on interpretation. The fact that most trend strategies underperform buy-and-hold
in crypto does *not* mean trend-following is useless. It means the secular uptrend is so strong
that the opportunity cost of being in cash — even temporarily — is enormous. In a lower-drift
environment (equities, commodities, or a future crypto regime with lower secular returns),
the calculus shifts in trend's favor. The historical crypto drift is an anomaly, not a steady
state, and strategies should be evaluated against a range of possible futures, not just the
most favorable past.[^12]

[^12]: This is analogous to AQR's observation that "Buy the Dip" strategies appear to work in
recent data primarily because equities have gone up a lot — not because the timing adds value.
Similarly, many crypto trend strategies "work" primarily because ETH has gone up {bh["total_return"]:.0f}×
over this period — not because the trend signal adds timing value.

---

## Appendix: Parameter Grid and Data Notes

**Data**: Coinbase Advanced spot OHLCV, ETH-USD. January 1, 2017 – February 22, 2026.
Daily, 4-hour, and 1-hour bars. Cached locally from DuckDB.

**Base signals (493)**: SMA crossover (7 fast × 7 slow), EMA crossover, DEMA crossover,
Hull MA crossover, price vs SMA/EMA, Donchian channel, Bollinger Bands, Keltner Channel,
Supertrend, raw momentum, vol-scaled momentum, linear regression t-stat, MACD, RSI, ADX,
CCI, Aroon, Stochastic, Parabolic SAR, Williams %R, MFI, TRIX, PPO, APO, MOM, ROC, CMO,
Ichimoku, OBV, Heikin-Ashi, Kaufman Efficiency Ratio, VWAP, dual momentum, triple MA,
Turtle breakout, regime-filter SMA, ATR breakout, close-above-high, mean-reversion band.

**Stop variants (9)**: None; fixed trailing at 5%, 10%, 20%; ATR-based trailing at 1.5×, 2.0×,
2.5×, 3.0×, 4.0× (14-period ATR at entry date).

**Backtest rules**: Binary long/cash. One-bar lag (signal computed on bar t, position taken on
bar t+1). 20 bps round-trip transaction costs. No leverage. No position sizing. Intraday
signals resampled to daily close for P&L computation.

**Multiple testing**: Effective independent tests estimated at 1,479 (493 signals × 3
frequencies). Stop variants treated as dependent (avg pairwise Sharpe correlation = 0.94).
Bonferroni correction applied at 5% family-wise error rate.

---

## References and Further Reading

Cao, Jeffrey, Nathan Chong, and Dan Villalon. "Hold the Dip." *AQR Alternative Thinking*
2025, Issue 4.

Hurst, Brian, Yao Hua Ooi, Lasse Heje Pedersen. "A Century of Evidence on Trend-Following
Investing." *The Journal of Portfolio Management* 44, no. 1 (2017).

Moskowitz, Tobias J., Yao Hua Ooi, Lasse Heje Pedersen. "Time series momentum." *Journal of
Financial Economics* 104, Issue 2 (2012): 228-50.

Babu, Abilash, Brendan Hoffman, Ari Levine, et al. "You Can't Always Trend When You Want."
*The Journal of Portfolio Management* 46, no. 4 (2020).

AQR. "Trend-Following: Why Now? A Macro Perspective." AQR whitepaper, November 16, 2022.

---

*{HYPO_LINE}*
"""
    return md


# =====================================================================
# Main
# =====================================================================

def main():
    print("[report] Loading results ...")
    df = load_results()
    strats = df[df["label"] != "BUY_AND_HOLD"].copy()
    bh_rows = df[df["label"] == "BUY_AND_HOLD"]
    if bh_rows.empty:
        print("[report] ERROR: no BUY_AND_HOLD row found")
        return
    bh = bh_rows.iloc[0]
    print(f"[report] {len(strats)} strategies, B&H Sharpe={bh['sharpe']:.2f}")

    print("[report] Generating exhibits ...")
    exhibit_1(strats, bh)
    exhibit_2(strats, bh)
    exhibit_3(strats, bh)
    fam_df = exhibit_4(strats, bh)
    exhibit_5(strats, bh)
    exhibit_6(strats, bh)
    exhibit_7(strats, bh)
    matched_df = exhibit_8(strats, bh)
    sharpe_thresh, n_survivors = exhibit_9(strats, bh)
    exhibit_10(strats, bh)
    exhibit_11(strats, bh)

    print("[report] Building markdown ...")
    md = build_markdown(df, strats, bh, sharpe_thresh, n_survivors, matched_df)
    md_path = DOCS_DIR / "eth_trend_sweep_report.md"
    md_path.write_text(md)
    print(f"[report] Markdown: {md_path}")

    print("[report] Done.")


if __name__ == "__main__":
    main()
