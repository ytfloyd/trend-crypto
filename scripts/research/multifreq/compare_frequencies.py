"""
Cross-Frequency Comparison
===========================
Reads all ML and momentum results from the multifreq artifacts
and generates a unified comparison report with plots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "multifreq" / "ml"
MOM_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "multifreq" / "momentum"
OUT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "multifreq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

FREQ_ORDER = ["5m", "30m", "1h", "4h", "8h", "1d"]


def load_ml_results() -> pd.DataFrame:
    dfs = []
    for freq in FREQ_ORDER:
        p = ML_DIR / f"eval_{freq}.csv"
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_momentum_results() -> pd.DataFrame:
    dfs = []
    for freq in FREQ_ORDER:
        p = MOM_DIR / f"momentum_{freq}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["freq"] = freq
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    print("=" * 70)
    print("CROSS-FREQUENCY COMPARISON — ML & MOMENTUM")
    print("=" * 70)

    ml_df = load_ml_results()
    mom_df = load_momentum_results()

    ml_freqs = sorted(ml_df["freq"].unique().tolist(), key=lambda x: FREQ_ORDER.index(x)) if len(ml_df) else []
    mom_freqs = sorted(mom_df["freq"].unique().tolist(), key=lambda x: FREQ_ORDER.index(x)) if len(mom_df) else []

    print(f"\nML frequencies: {ml_freqs}")
    print(f"Momentum frequencies: {mom_freqs}")

    # ===================================================================
    # ML Summary Table
    # ===================================================================
    print("\n" + "=" * 70)
    print("ML RESULTS BY FREQUENCY")
    print("=" * 70)
    print(f"\n{'Freq':<6s} {'Model':<14s} {'IC':>8s} {'IC(mean)':>10s} {'IC(std)':>9s} "
          f"{'t-stat':>8s} {'IC>0%':>7s} {'Hit':>7s} {'N_obs':>10s}")
    print("-" * 85)
    for freq in ml_freqs:
        sub = ml_df[ml_df["freq"] == freq].sort_values("ic", ascending=False)
        for _, row in sub.iterrows():
            print(f"{freq:<6s} {row['model']:<14s} {row['ic']:+8.4f} {row['ic_mean']:+10.4f} "
                  f"{row['ic_std']:9.4f} {row['ic_t']:8.2f} {row['ic_hit_pct']:6.0%} "
                  f"{row['hit_rate']:6.1%} {int(row['n_obs']):>10,}")

    # Best ML model per frequency
    print(f"\n  BEST ML MODEL PER FREQUENCY:")
    print(f"  {'Freq':<6s} {'Model':<14s} {'IC(mean)':>10s} {'t-stat':>8s}")
    print(f"  {'-'*42}")
    best_ml = []
    for freq in ml_freqs:
        sub = ml_df[ml_df["freq"] == freq].sort_values("ic_mean", ascending=False)
        best = sub.iloc[0]
        best_ml.append(best)
        print(f"  {freq:<6s} {best['model']:<14s} {best['ic_mean']:+10.4f} {best['ic_t']:8.2f}")

    # ===================================================================
    # Momentum Summary Table
    # ===================================================================
    print("\n" + "=" * 70)
    print("MOMENTUM RESULTS BY FREQUENCY")
    print("=" * 70)
    print(f"\n{'Freq':<6s} {'Signal':<12s} {'Sharpe':>8s} {'CAGR':>8s} {'Vol':>8s} "
          f"{'MaxDD':>8s} {'Hit%':>7s} {'TO':>8s}")
    print("-" * 70)
    for freq in mom_freqs:
        sub = mom_df[mom_df["freq"] == freq].sort_values("sharpe", ascending=False)
        for _, row in sub.iterrows():
            print(f"{freq:<6s} {row['signal']:<12s} {row['sharpe']:>8.2f} {row['cagr']:>7.1%} "
                  f"{row['vol']:>7.1%} {row['max_dd']:>7.1%} {row['hit_rate']:>6.1%} "
                  f"{row.get('avg_turnover', 0):>7.2%}")

    # Best momentum signal per frequency
    print(f"\n  BEST MOMENTUM SIGNAL PER FREQUENCY:")
    print(f"  {'Freq':<6s} {'Signal':<10s} {'Sharpe':>8s} {'CAGR':>8s}")
    print(f"  {'-'*36}")
    best_mom = []
    for freq in mom_freqs:
        sub = mom_df[mom_df["freq"] == freq].sort_values("sharpe", ascending=False)
        best = sub.iloc[0]
        best_mom.append(best)
        print(f"  {freq:<6s} {best['signal']:<10s} {best['sharpe']:>8.2f} {best['cagr']:>7.1%}")

    # ===================================================================
    # Combined Plots
    # ===================================================================
    print("\n--- Generating comparison plots ---")

    # 1. ML IC Mean by frequency (grouped bar)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax = axes[0, 0]
    model_colors = {
        "Ridge": "#42A5F5", "XGBoost": "#FFA726", "XGB_Clf": "#EC407A",
        "LightGBM": "#FFCA28", "MLP": "#EF5350",
    }
    models = ml_df["model"].unique()
    x = np.arange(len(ml_freqs))
    width = 0.15
    for i, model in enumerate(models):
        ics = []
        for freq in ml_freqs:
            sub = ml_df[(ml_df["freq"] == freq) & (ml_df["model"] == model)]
            ics.append(sub["ic_mean"].values[0] if len(sub) > 0 else 0)
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, ics, width, label=model,
               color=model_colors.get(model, "#9E9E9E"), alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(ml_freqs, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean OOS IC", fontsize=11)
    ax.set_title("ML Signal Quality by Frequency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)

    # 2. Momentum Sharpe by frequency (grouped bar)
    ax = axes[0, 1]
    sig_colors = {"RET": "#3b82f6", "MAC": "#22c55e", "EMAC": "#10b981",
                  "BRK": "#f59e0b", "LREG": "#8b5cf6"}
    signals = mom_df["signal"].unique()
    x2 = np.arange(len(mom_freqs))
    width2 = 0.15
    for i, sig in enumerate(signals):
        sharpes = []
        for freq in mom_freqs:
            sub = mom_df[(mom_df["freq"] == freq) & (mom_df["signal"] == sig)]
            sharpes.append(sub["sharpe"].values[0] if len(sub) > 0 else 0)
        offset = (i - len(signals) / 2 + 0.5) * width2
        ax.bar(x2 + offset, sharpes, width2, label=sig,
               color=sig_colors.get(sig, "#9E9E9E"), alpha=0.85, edgecolor="white")
    ax.set_xticks(x2)
    ax.set_xticklabels(mom_freqs, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Momentum Signal Quality by Frequency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)

    # 3. Best ML IC and Best Mom Sharpe side by side
    ax = axes[1, 0]
    best_ml_ics = [b["ic_mean"] for b in best_ml]
    best_mom_sharpes = [b["sharpe"] for b in best_mom]

    all_freqs = sorted(set(ml_freqs + mom_freqs), key=lambda x: FREQ_ORDER.index(x))
    x3 = np.arange(len(all_freqs))

    # ML bars
    ml_vals = []
    for freq in all_freqs:
        match = [b for b in best_ml if b["freq"] == freq]
        ml_vals.append(match[0]["ic_mean"] if match else 0)

    # Momentum bars (Sharpe on different scale, so use twin axis)
    mom_vals = []
    for freq in all_freqs:
        match = [b for b in best_mom if b["freq"] == freq]
        mom_vals.append(match[0]["sharpe"] if match else 0)

    w3 = 0.35
    ax.bar(x3 - w3/2, ml_vals, w3, label="Best ML IC(mean)", color="#EC407A", alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x3 + w3/2, mom_vals, w3, label="Best Mom Sharpe", color="#3b82f6", alpha=0.85)
    ax.set_xticks(x3)
    ax.set_xticklabels(all_freqs, fontsize=11)
    ax.set_ylabel("ML IC (mean)", color="#EC407A", fontsize=11)
    ax2.set_ylabel("Momentum Sharpe", color="#3b82f6", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Best ML vs Best Momentum by Frequency", fontsize=13, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    # 4. ML t-stat heatmap
    ax = axes[1, 1]
    heatmap = np.full((len(models), len(ml_freqs)), np.nan)
    for i, model in enumerate(models):
        for j, freq in enumerate(ml_freqs):
            sub = ml_df[(ml_df["freq"] == freq) & (ml_df["model"] == model)]
            if len(sub) > 0:
                heatmap[i, j] = sub.iloc[0]["ic_t"]

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=5)
    ax.set_xticks(range(len(ml_freqs)))
    ax.set_xticklabels(ml_freqs, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    for i in range(len(models)):
        for j in range(len(ml_freqs)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                        fontweight="bold" if val >= 2.0 else "normal",
                        color="white" if val < 0 else "black")
    plt.colorbar(im, ax=ax, label="t-statistic")
    ax.set_title("ML IC t-Statistics (Model × Frequency)", fontsize=13, fontweight="bold")

    fig.suptitle("Multi-Frequency Analysis — ML & Momentum Strategies in Crypto",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_frequency_comparison.png", dpi=150)
    plt.close(fig)
    print("  [1/2] Main comparison panel saved")

    # 5. Detailed IC-by-model heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap2 = np.full((len(models), len(ml_freqs)), np.nan)
    for i, model in enumerate(models):
        for j, freq in enumerate(ml_freqs):
            sub = ml_df[(ml_df["freq"] == freq) & (ml_df["model"] == model)]
            if len(sub) > 0:
                heatmap2[i, j] = sub.iloc[0]["ic_mean"]

    im = ax.imshow(heatmap2, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=0.08)
    ax.set_xticks(range(len(ml_freqs)))
    ax.set_xticklabels(ml_freqs, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    for i in range(len(models)):
        for j in range(len(ml_freqs)):
            val = heatmap2[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=10,
                        fontweight="bold" if val > 0.04 else "normal")
    plt.colorbar(im, ax=ax, label="Mean OOS IC")
    ax.set_title("ML Mean IC Heatmap (Model × Frequency)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ml_ic_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [2/2] IC heatmap saved")

    # ===================================================================
    # Key Findings
    # ===================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Best ML across all
    best_ml_overall = ml_df.sort_values("ic_mean", ascending=False).iloc[0]
    print(f"\n  1. BEST ML SIGNAL overall: {best_ml_overall['model']} at {best_ml_overall['freq']}")
    print(f"     IC(mean)={best_ml_overall['ic_mean']:+.4f}, t={best_ml_overall['ic_t']:.2f}")

    # Best momentum across all
    best_mom_overall = mom_df.sort_values("sharpe", ascending=False).iloc[0]
    print(f"\n  2. BEST MOMENTUM SIGNAL overall: {best_mom_overall['signal']} at {best_mom_overall['freq']}")
    print(f"     Sharpe={best_mom_overall['sharpe']:.2f}, CAGR={best_mom_overall['cagr']:.1%}")

    # Frequency ranking for ML
    freq_ranking_ml = []
    for freq in ml_freqs:
        sub = ml_df[ml_df["freq"] == freq]
        freq_ranking_ml.append((freq, sub["ic_mean"].max()))
    freq_ranking_ml.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  3. FREQUENCY RANKING (ML, by best IC):")
    for i, (freq, ic) in enumerate(freq_ranking_ml, 1):
        print(f"     {i}. {freq:<6s} IC(mean)={ic:+.4f}")

    # Frequency ranking for Momentum
    freq_ranking_mom = []
    for freq in mom_freqs:
        sub = mom_df[mom_df["freq"] == freq]
        freq_ranking_mom.append((freq, sub["sharpe"].max()))
    freq_ranking_mom.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  4. FREQUENCY RANKING (Momentum, by best Sharpe):")
    for i, (freq, sharpe) in enumerate(freq_ranking_mom, 1):
        print(f"     {i}. {freq:<6s} Sharpe={sharpe:.2f}")

    # Statistical significance
    print(f"\n  5. STATISTICALLY SIGNIFICANT ML SIGNALS (t >= 2.0):")
    sig = ml_df[ml_df["ic_t"] >= 2.0].sort_values("ic_t", ascending=False)
    if len(sig) == 0:
        print("     None")
    else:
        for _, row in sig.iterrows():
            print(f"     {row['model']:<14s} at {row['freq']:<6s} IC={row['ic_mean']:+.4f} t={row['ic_t']:.2f}")

    print(f"\nArtifacts saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
