#!/usr/bin/env python3
"""
Full Universe Analysis — Cross-universe aggregation.

Reads outputs from full_universe_sweep.py and produces:
  - TIM optimum distribution across Tier A assets
  - Tier B replication statistics
  - Family survival heatmap (Tier A)
  - Walk-forward TIM stability
  - Asset tradeability classification

Usage:
    python -m scripts.research.tsmom.full_universe_analysis
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

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "artifacts" / "research" / "tsmom"
UNIV = SWEEP / "full_universe"
UNIV.mkdir(parents=True, exist_ok=True)

NAVY = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GREEN = "#336633"; GRAY = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
PURPLE = "#663399"; ORANGE = "#CC6633"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRAY,
    "figure.facecolor": "white",
})

N_EFF = 493 * 3
BONF_Z = sp_stats.norm.ppf(1 - 0.05 / N_EFF / 2)


def main():
    print("=" * 70)
    print("  FULL UNIVERSE ANALYSIS")
    print("=" * 70)

    ta = pd.read_csv(UNIV / "tier_a_summary.csv")
    tb = pd.read_csv(UNIV / "tier_b_replication.csv")
    wf = pd.read_csv(UNIV / "tier_a_walkforward.csv")
    tiers = pd.read_csv(UNIV / "universe_tiers.csv")

    # ==================================================================
    # 1. TIM OPTIMUM DISTRIBUTION
    # ==================================================================
    print(f"\n  1. TIM OPTIMUM DISTRIBUTION ({len(ta)} Tier A assets)")

    tim_vals = ta["tim_optimum"].dropna()
    print(f"     Range: [{tim_vals.min():.0%}, {tim_vals.max():.0%}]")
    print(f"     Median: {tim_vals.median():.0%}")
    print(f"     Mean: {tim_vals.mean():.0%}")
    print(f"     Std: {tim_vals.std():.0%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(tim_vals, bins=15, color=NAVY, alpha=0.7, edgecolor="white")
    ax1.axvline(tim_vals.median(), color=RED, ls="--", lw=1.5, label=f"Median: {tim_vals.median():.0%}")
    ax1.axvspan(0.37, 0.47, alpha=0.1, color=GOLD, label="ETH optimal band [37%-47%]")
    ax1.set_xlabel("TIM Optimum")
    ax1.set_ylabel("Count")
    ax1.set_title("A. TIM Optimum Distribution (31 Tier A Assets)", fontweight="bold")
    ax1.legend(fontsize=7, frameon=True, facecolor="white", edgecolor=LGRAY)

    # TIM vs B&H Sharpe
    ax2.scatter(ta["bh_sharpe"], ta["tim_optimum"], c=NAVY, s=40, alpha=0.7)
    for _, r in ta.iterrows():
        if r["tim_optimum"] is not None and not pd.isna(r["tim_optimum"]):
            ax2.annotate(r["symbol"].replace("-USD", ""),
                        (r["bh_sharpe"], r["tim_optimum"]),
                        fontsize=6, alpha=0.6, ha="center", va="bottom")
    ax2.set_xlabel("B&H Sharpe")
    ax2.set_ylabel("TIM Optimum")
    ax2.set_title("B. TIM Optimum vs Asset Drift", fontweight="bold")
    ax2.axhspan(0.37, 0.47, alpha=0.08, color=GOLD)

    fig.tight_layout()
    fig.savefig(UNIV / "tim_optimum_distribution.png", dpi=150)
    plt.close(fig)

    # ==================================================================
    # 2. TIER B REPLICATION STATISTICS
    # ==================================================================
    print(f"\n  2. TIER B REPLICATION ({len(tb)} assets)")

    print(f"     Median % positive Sharpe:  {tb['pct_positive_sharpe'].median():.0%}")
    print(f"     Median % beat B&H:         {tb['pct_beat_bh'].median():.0%}")
    print(f"     Median % better drawdown:  {tb['pct_better_dd'].median():.0%}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].hist(tb["pct_positive_sharpe"], bins=20, color=TEAL, alpha=0.7, edgecolor="white")
    axes[0].axvline(tb["pct_positive_sharpe"].median(), color=RED, ls="--", lw=1.5)
    axes[0].set_xlabel("% Strategies with Positive Sharpe")
    axes[0].set_title(f"A. Positive Sharpe Rate\n(median: {tb['pct_positive_sharpe'].median():.0%})",
                      fontweight="bold")

    axes[1].hist(tb["pct_beat_bh"], bins=20, color=NAVY, alpha=0.7, edgecolor="white")
    axes[1].axvline(tb["pct_beat_bh"].median(), color=RED, ls="--", lw=1.5)
    axes[1].set_xlabel("% Strategies Beating B&H Sharpe")
    axes[1].set_title(f"B. B&H Beat Rate\n(median: {tb['pct_beat_bh'].median():.0%})",
                      fontweight="bold")

    axes[2].hist(tb["pct_better_dd"], bins=20, color=GREEN, alpha=0.7, edgecolor="white")
    axes[2].axvline(tb["pct_better_dd"].median(), color=RED, ls="--", lw=1.5)
    axes[2].set_xlabel("% Strategies with Better Drawdown")
    axes[2].set_title(f"C. Drawdown Compression Rate\n(median: {tb['pct_better_dd'].median():.0%})",
                      fontweight="bold")

    fig.suptitle(f"Tier B: ETH Survivor Replication on {len(tb)} Assets (Zero Re-optimization)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(UNIV / "tier_b_replication_histograms.png", dpi=150)
    plt.close(fig)

    # ==================================================================
    # 3. WALK-FORWARD TIM STABILITY
    # ==================================================================
    print(f"\n  3. WALK-FORWARD TIM STABILITY ({len(wf)} assets)")

    if len(wf) > 0:
        print(f"     Median TIM ρ:       {wf['tim_corr'].median():.3f}")
        print(f"     Min TIM ρ:          {wf['tim_corr'].min():.3f}")
        print(f"     Max TIM ρ:          {wf['tim_corr'].max():.3f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.hist(wf["tim_corr"].dropna(), bins=15, color=NAVY, alpha=0.7, edgecolor="white")
        ax1.axvline(wf["tim_corr"].median(), color=RED, ls="--", lw=1.5,
                    label=f"Median: {wf['tim_corr'].median():.3f}")
        ax1.set_xlabel("TIM Correlation (train vs full)")
        ax1.set_ylabel("Count")
        ax1.set_title("A. TIM Stability Across 31 Assets", fontweight="bold")
        ax1.legend(fontsize=7, frameon=True, facecolor="white", edgecolor=LGRAY)
        ax1.set_xlim(0.95, 1.0)

        oos_sr = wf["oos_sharpe"].dropna()
        full_sr = wf["full_sharpe"].dropna()
        ax2.scatter(full_sr, oos_sr, c=TEAL, s=40, alpha=0.7)
        for _, r in wf.iterrows():
            if pd.notna(r.get("oos_sharpe")):
                ax2.annotate(r["symbol"].replace("-USD", ""),
                            (r["full_sharpe"], r["oos_sharpe"]),
                            fontsize=6, alpha=0.6, ha="center", va="bottom")
        ax2.axhline(0, color=GRAY, ls=":", lw=0.8)
        ax2.axvline(0, color=GRAY, ls=":", lw=0.8)
        lims = [min(full_sr.min(), oos_sr.min()) - 0.1, max(full_sr.max(), oos_sr.max()) + 0.1]
        ax2.plot(lims, lims, color=LGRAY, ls=":", lw=0.8)
        ax2.set_xlabel("Full-Period Sharpe")
        ax2.set_ylabel("Out-of-Sample Sharpe")
        ax2.set_title("B. Walk-Forward Sharpe Decay", fontweight="bold")

        fig.tight_layout()
        fig.savefig(UNIV / "walkforward_stability.png", dpi=150)
        plt.close(fig)

    # ==================================================================
    # 4. FAMILY SURVIVAL HEATMAP (Tier A with survivors)
    # ==================================================================
    print(f"\n  4. FAMILY SURVIVAL HEATMAP")

    fam_data = {}
    for _, r in ta.iterrows():
        sym = r["symbol"]
        slug = sym.replace("-", "").lower()
        path = SWEEP / f"{slug}_trend_sweep" / "results_v2.csv"
        if not path.exists() and sym == "ETH-USD":
            path = SWEEP / "eth_trend_sweep" / "results_v2.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        daily = df[(df["label"] != "BUY_AND_HOLD") & (df["freq"] == "1d")]
        bonf = BONF_Z / np.sqrt(r["years"])
        fam_surv = daily.groupby("signal_family").apply(
            lambda g: (g["sharpe"] >= bonf).mean(), include_groups=False
        )
        fam_data[sym] = fam_surv

    if fam_data:
        fam_matrix = pd.DataFrame(fam_data).fillna(0)
        # Top families by max survival across any asset
        top_fams = fam_matrix.max(axis=1).sort_values(ascending=False).head(20).index
        plot_data = fam_matrix.loc[top_fams]

        # Only show assets with >0 survivors somewhere
        assets_with_surv = plot_data.columns[plot_data.sum() > 0]
        if len(assets_with_surv) > 0:
            plot_data = plot_data[assets_with_surv]

            fig, ax = plt.subplots(figsize=(max(8, len(assets_with_surv) * 0.8), 10))
            im = ax.imshow(plot_data.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
            ax.set_xticks(range(len(assets_with_surv)))
            ax.set_xticklabels([s.replace("-USD", "") for s in assets_with_surv],
                              fontsize=8, rotation=45, ha="right")
            ax.set_yticks(range(len(top_fams)))
            ax.set_yticklabels(top_fams, fontsize=8)
            for i in range(len(top_fams)):
                for j in range(len(assets_with_surv)):
                    val = plot_data.values[i, j]
                    if val > 0.01:
                        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                                fontsize=6, color="white" if val > 0.25 else "black")
            plt.colorbar(im, ax=ax, label="Bonferroni Survival Rate", shrink=0.7)
            ax.set_title("Family Survival Rate Across All Tier A Assets (Daily)", fontweight="bold")
            fig.tight_layout()
            fig.savefig(UNIV / "family_survival_full_heatmap.png", dpi=150)
            plt.close(fig)
            print(f"     Heatmap: {len(assets_with_surv)} assets with survivors, {len(top_fams)} families")
        else:
            print(f"     No families with survivors outside ETH/BTC")

    # ==================================================================
    # 5. ASSET TRADEABILITY CLASSIFICATION
    # ==================================================================
    print(f"\n  5. ASSET CLASSIFICATION")

    classification = []
    for _, r in ta.iterrows():
        wf_row = wf[wf["symbol"] == r["symbol"]]
        oos_sr = wf_row["oos_sharpe"].values[0] if len(wf_row) > 0 else None

        if r["n_survivors"] > 0 and oos_sr is not None and oos_sr > 0:
            cls = "STRONG"
        elif oos_sr is not None and oos_sr > 0:
            cls = "MODERATE"
        elif r["tim_optimum"] is not None and 0.30 <= r["tim_optimum"] <= 0.60:
            cls = "MARGINAL"
        else:
            cls = "WEAK"

        classification.append({
            "symbol": r["symbol"], "class": cls,
            "years": r["years"], "bh_sharpe": r["bh_sharpe"],
            "n_survivors": r["n_survivors"],
            "tim_optimum": r["tim_optimum"],
            "wf_oos_sharpe": round(oos_sr, 3) if oos_sr is not None else None,
        })

    cls_df = pd.DataFrame(classification)
    cls_df.to_csv(UNIV / "asset_classification.csv", index=False)

    for c in ["STRONG", "MODERATE", "MARGINAL", "WEAK"]:
        subset = cls_df[cls_df["class"] == c]
        syms = ", ".join(subset["symbol"].tolist())
        print(f"     {c}: {len(subset)} assets — {syms}")

    # ==================================================================
    # 6. SUMMARY STATISTICS
    # ==================================================================
    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")

    summary = {
        "n_tier_a": len(ta),
        "n_tier_b": len(tb),
        "n_tier_c": len(tiers[tiers["tier"] == "C"]),
        "tier_a_med_bh_sharpe": round(ta["bh_sharpe"].median(), 3),
        "tier_a_med_survivors": int(ta["n_survivors"].median()),
        "tier_a_tim_median": round(tim_vals.median(), 2) if len(tim_vals) > 0 else None,
        "tier_a_tim_std": round(tim_vals.std(), 2) if len(tim_vals) > 0 else None,
        "tier_b_med_pct_pos_sr": round(tb["pct_positive_sharpe"].median(), 3),
        "tier_b_med_pct_beat_bh": round(tb["pct_beat_bh"].median(), 3),
        "tier_b_med_pct_better_dd": round(tb["pct_better_dd"].median(), 3),
        "wf_med_tim_corr": round(wf["tim_corr"].median(), 3) if len(wf) > 0 else None,
        "wf_min_tim_corr": round(wf["tim_corr"].min(), 3) if len(wf) > 0 else None,
        "wf_med_oos_sharpe": round(wf["oos_sharpe"].dropna().median(), 3) if len(wf) > 0 else None,
        "n_strong": len(cls_df[cls_df["class"] == "STRONG"]),
        "n_moderate": len(cls_df[cls_df["class"] == "MODERATE"]),
        "n_marginal": len(cls_df[cls_df["class"] == "MARGINAL"]),
        "n_weak": len(cls_df[cls_df["class"] == "WEAK"]),
    }
    pd.DataFrame([summary]).to_csv(UNIV / "full_universe_summary.csv", index=False)

    for k, v in summary.items():
        print(f"     {k}: {v}")

    print(f"\n  Outputs: {UNIV}")
    print("  Done.")


if __name__ == "__main__":
    main()
