#!/usr/bin/env python3
"""
Cross-Asset Comparison Analysis

Tests three hypotheses across ETH, BTC, SOL (Tier 1) and LTC, LINK, ATOM (Tier 2):
  H1: Signal family dominance is structural (Spearman rho of family survival rankings)
  H2: TIM optimum is portable (~42% across assets)
  H3: Stop-type microstructure is structural (ATR vs fixed by family)

Usage:
    python -m scripts.research.tsmom.cross_asset_analysis
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
OUT = SWEEP / "cross_asset"
OUT.mkdir(parents=True, exist_ok=True)

NAVY  = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
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
SEED = 42
np.random.seed(SEED)


def bonf_threshold(n_years):
    return BONF_Z / np.sqrt(n_years)


def load_tier1():
    """Load full sweep results for ETH, BTC, SOL."""
    assets = {}
    for name, path in [
        ("ETH-USD", SWEEP / "eth_trend_sweep" / "results_v2.csv"),
        ("BTC-USD", SWEEP / "btcusd_trend_sweep" / "results_v2.csv"),
        ("SOL-USD", SWEEP / "solusd_trend_sweep" / "results_v2.csv"),
    ]:
        df = pd.read_csv(path)
        assets[name] = df[df["label"] != "BUY_AND_HOLD"].copy()
    return assets


def main():
    print("=" * 70)
    print("  CROSS-ASSET COMPARISON ANALYSIS")
    print("=" * 70)

    tier1 = load_tier1()
    asset_years = {"ETH-USD": 9.1, "BTC-USD": 9.1, "SOL-USD": 4.7}

    # ================================================================
    # H1: FAMILY SURVIVAL RANKINGS
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  H1: FAMILY SURVIVAL RANKINGS")
    print(f"  {'='*60}")

    family_survival = {}
    for asset, df in tier1.items():
        thresh = bonf_threshold(asset_years[asset])
        daily = df[df["freq"] == "1d"] if "freq" in df.columns else df
        fam_stats = daily.groupby("signal_family").apply(
            lambda g: pd.Series({
                "n_total": len(g),
                "n_survive": (g["sharpe"] >= thresh).sum(),
                "survival_rate": (g["sharpe"] >= thresh).mean(),
                "med_sharpe": g["sharpe"].median(),
            }), include_groups=False
        )
        family_survival[asset] = fam_stats
        n_surv = (daily["sharpe"] >= thresh).sum()
        print(f"\n  {asset}: threshold={thresh:.2f}, survivors={n_surv}/{len(daily)}")
        print(f"  Top 5 families:")
        for fam, row in fam_stats.sort_values("survival_rate", ascending=False).head(5).iterrows():
            print(f"    {fam:<18s} {row['survival_rate']:.0%} ({row['n_survive']:.0f}/{row['n_total']:.0f})")

    # Spearman rank correlation of family survival rates
    all_families = set()
    for fs in family_survival.values():
        all_families.update(fs.index)

    survival_matrix = pd.DataFrame(index=sorted(all_families))
    for asset, fs in family_survival.items():
        survival_matrix[asset] = fs["survival_rate"]
    survival_matrix = survival_matrix.fillna(0)
    survival_matrix.to_csv(OUT / "family_survival_rates.csv")

    asset_names = list(tier1.keys())
    spearman_results = []
    print(f"\n  Spearman rank correlations of family survival rates:")
    for i in range(len(asset_names)):
        for j in range(i + 1, len(asset_names)):
            a1, a2 = asset_names[i], asset_names[j]
            common = survival_matrix[[a1, a2]].dropna()
            if len(common) < 5:
                continue
            rho, pval = sp_stats.spearmanr(common[a1], common[a2])
            spearman_results.append({
                "asset_1": a1, "asset_2": a2,
                "spearman_rho": round(rho, 3), "p_value": round(pval, 4),
            })
            print(f"    {a1} vs {a2}: ρ = {rho:.3f} (p = {pval:.4f})")

    pd.DataFrame(spearman_results).to_csv(OUT / "spearman_family_rankings.csv", index=False)

    # Family survival heatmap
    top_families = survival_matrix.max(axis=1).sort_values(ascending=False).head(20).index
    heatmap_data = survival_matrix.loc[top_families]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(asset_names)))
    ax.set_xticklabels(asset_names, fontsize=9)
    ax.set_yticks(range(len(top_families)))
    ax.set_yticklabels(top_families, fontsize=8)
    for i in range(len(top_families)):
        for j in range(len(asset_names)):
            val = heatmap_data.values[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Bonferroni Survival Rate", shrink=0.7)
    ax.set_title("H1: Family Survival Rate Across Assets (Daily Frequency)\n"
                 "Top 20 families by max survival rate", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "family_survival_heatmap.png", dpi=150)
    plt.close(fig)

    # ================================================================
    # H2: TIM OPTIMUM PORTABILITY
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  H2: TIM OPTIMUM PORTABILITY")
    print(f"  {'='*60}")

    asset_colors = {"ETH-USD": NAVY, "BTC-USD": TEAL, "SOL-USD": GOLD}
    tim_optima = {}

    fig, ax = plt.subplots(figsize=(12, 6))

    for asset, df in tier1.items():
        daily = df[df["freq"] == "1d"] if "freq" in df.columns else df
        tim = daily["time_in_market"]
        sharpe = daily["sharpe"]

        bins = np.arange(0, 1.01, 0.01)
        bin_idx = np.digitize(tim, bins) - 1
        bin_centers = []
        bin_medians = []
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if mask.sum() >= 5:
                bin_centers.append((bins[b] + bins[b + 1]) / 2)
                bin_medians.append(sharpe[mask].median())

        bc = np.array(bin_centers)
        bm = np.array(bin_medians)
        if len(bc) > 5:
            from scipy.ndimage import uniform_filter1d
            smooth = uniform_filter1d(bm, size=5)
            ax.plot(bc, smooth, color=asset_colors[asset], lw=2, label=asset)
            opt_idx = np.argmax(smooth)
            opt_tim = bc[opt_idx]
            opt_sharpe = smooth[opt_idx]
            tim_optima[asset] = {"optimal_tim": round(opt_tim, 2),
                                 "optimal_sharpe": round(opt_sharpe, 3)}
            ax.axvline(opt_tim, color=asset_colors[asset], ls=":", alpha=0.5, lw=1)
            print(f"  {asset}: TIM optimum = {opt_tim:.0%}, peak Sharpe = {opt_sharpe:.3f}")

    # Bootstrap CIs for ETH and BTC
    for asset in ["ETH-USD", "BTC-USD"]:
        daily = tier1[asset]
        daily = daily[daily["freq"] == "1d"] if "freq" in daily.columns else daily
        tim_vals = daily["time_in_market"].values
        sharpe_vals = daily["sharpe"].values

        boot_optima = []
        for _ in range(1000):
            idx = np.random.choice(len(daily), len(daily), replace=True)
            b_tim = tim_vals[idx]
            b_sr = sharpe_vals[idx]
            bins = np.arange(0, 1.01, 0.02)
            b_idx = np.digitize(b_tim, bins) - 1
            bc, bm = [], []
            for b in range(len(bins) - 1):
                mask = b_idx == b
                if mask.sum() >= 3:
                    bc.append((bins[b] + bins[b+1]) / 2)
                    bm.append(np.median(b_sr[mask]))
            if bc:
                from scipy.ndimage import uniform_filter1d
                sm = uniform_filter1d(np.array(bm), size=3)
                boot_optima.append(np.array(bc)[np.argmax(sm)])

        ci_lo = np.percentile(boot_optima, 5)
        ci_hi = np.percentile(boot_optima, 95)
        tim_optima[asset]["ci_90_lo"] = round(ci_lo, 2)
        tim_optima[asset]["ci_90_hi"] = round(ci_hi, 2)
        ax.axvspan(ci_lo, ci_hi, alpha=0.08, color=asset_colors[asset])
        print(f"  {asset}: 90% CI = [{ci_lo:.0%}, {ci_hi:.0%}]")

    ax.set_xlabel("Time in Market")
    ax.set_ylabel("Median Sharpe (5-bin smoothed)")
    ax.set_title("H2: Sharpe vs TIM Across Assets (Daily Frequency)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=LGRAY)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "tim_optimum_comparison.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(tim_optima).T.to_csv(OUT / "tim_optima.csv")

    # ================================================================
    # H3: STOP-TYPE MICROSTRUCTURE
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  H3: STOP-TYPE MICROSTRUCTURE")
    print(f"  {'='*60}")

    stop_results = []
    for asset, df in tier1.items():
        daily = df[df["freq"] == "1d"] if "freq" in df.columns else df
        for (label,), group in daily.groupby(["label"]):
            atr = group[group["stop_type"] == "atr"]
            pct = group[group["stop_type"] == "pct"]
            if atr.empty or pct.empty:
                continue
            atr_best = atr["sharpe"].max()
            pct_best = pct["sharpe"].max()
            fam = group["signal_family"].iloc[0]
            stop_results.append({
                "asset": asset, "label": label, "family": fam,
                "atr_best_sharpe": atr_best, "pct_best_sharpe": pct_best,
                "atr_wins": atr_best > pct_best,
            })

    stop_df = pd.DataFrame(stop_results)

    # Per-asset, per-family ATR win rate
    fam_stop = stop_df.groupby(["asset", "family"]).agg(
        n_pairs=("atr_wins", "count"),
        atr_win_rate=("atr_wins", "mean"),
    ).reset_index()
    fam_stop_pivot = fam_stop.pivot_table(
        index="family", columns="asset", values="atr_win_rate"
    ).dropna(thresh=2)
    fam_stop_pivot.to_csv(OUT / "stop_type_by_family_asset.csv")

    # Key families from Part 8 findings
    key_families = ["EMA", "ADX", "DEMA", "Supertrend", "CCI", "Aroon", "SMA", "Hull"]
    print(f"\n  ATR win rate by key family and asset:")
    print(f"  {'Family':<15s}", end="")
    for a in asset_names:
        print(f"  {a:>10s}", end="")
    print()
    for fam in key_families:
        print(f"  {fam:<15s}", end="")
        for a in asset_names:
            sub = stop_df[(stop_df["asset"] == a) & (stop_df["family"] == fam)]
            if len(sub) > 0:
                wr = sub["atr_wins"].mean()
                print(f"  {wr:>10.0%}", end="")
            else:
                print(f"  {'—':>10s}", end="")
        print()

    # Consistency: does the same family prefer the same stop type across assets?
    consistency = []
    for fam in fam_stop_pivot.index:
        vals = fam_stop_pivot.loc[fam].dropna()
        if len(vals) >= 2:
            all_atr = (vals > 0.5).all()
            all_pct = (vals < 0.5).all()
            consistency.append({
                "family": fam,
                "consistent": all_atr or all_pct,
                "direction": "ATR" if all_atr else ("fixed" if all_pct else "mixed"),
                "mean_atr_wr": round(vals.mean(), 3),
            })
    cons_df = pd.DataFrame(consistency)
    cons_df.to_csv(OUT / "stop_consistency.csv", index=False)
    n_consistent = cons_df["consistent"].sum()
    n_total_fams = len(cons_df)
    print(f"\n  Stop consistency: {n_consistent}/{n_total_fams} families consistent "
          f"({n_consistent/n_total_fams:.0%})")

    # Stop heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    plot_fams = fam_stop_pivot.loc[
        fam_stop_pivot.index.isin(
            fam_stop_pivot.max(axis=1).sort_values(ascending=False).head(20).index
        )
    ]
    im = ax.imshow(plot_fams.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(asset_names)))
    ax.set_xticklabels(asset_names, fontsize=9)
    ax.set_yticks(range(len(plot_fams)))
    ax.set_yticklabels(plot_fams.index, fontsize=8)
    for i in range(len(plot_fams)):
        for j in range(len(asset_names)):
            val = plot_fams.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.7 or val < 0.3 else "black")
    plt.colorbar(im, ax=ax, label="ATR Win Rate (>50% = ATR preferred)", shrink=0.7)
    ax.set_title("H3: ATR vs Fixed Stop Win Rate by Family and Asset\n"
                 "(Green = ATR dominates, Red = Fixed dominates)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "stop_type_heatmap.png", dpi=150)
    plt.close(fig)

    # ================================================================
    # SUMMARY & DECISION
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  DECISION FRAMEWORK")
    print(f"  {'='*60}")

    rhos = [r["spearman_rho"] for r in spearman_results]
    mean_rho = np.mean(rhos) if rhos else 0
    print(f"  Mean Spearman ρ (family rankings): {mean_rho:.3f}")

    if mean_rho > 0.6:
        outcome = "A"
        print(f"  → OUTCOME A: Family dominance is structural (ρ > 0.6)")
    elif mean_rho < 0.3:
        outcome = "C"
        print(f"  → OUTCOME C: ETH findings are path dependence (ρ < 0.3)")
    else:
        outcome = "B"
        print(f"  → OUTCOME B: Ambiguous range (0.3 < ρ < 0.6) — requires judgment")
        print(f"    Note: These thresholds are pre-specified decision points, not theory-derived.")

    # TIM overlap
    tim_assets = [a for a in ["ETH-USD", "BTC-USD"] if a in tim_optima and "ci_90_lo" in tim_optima[a]]
    if len(tim_assets) == 2:
        a1, a2 = tim_assets
        overlap = (tim_optima[a1]["ci_90_lo"] <= tim_optima[a2]["ci_90_hi"] and
                   tim_optima[a2]["ci_90_lo"] <= tim_optima[a1]["ci_90_hi"])
        print(f"\n  TIM CIs overlap: {'Yes' if overlap else 'No'}")
        print(f"    {a1}: [{tim_optima[a1]['ci_90_lo']:.0%}, {tim_optima[a1]['ci_90_hi']:.0%}]")
        print(f"    {a2}: [{tim_optima[a2]['ci_90_lo']:.0%}, {tim_optima[a2]['ci_90_hi']:.0%}]")

    summary = {
        "mean_spearman_rho": round(mean_rho, 3),
        "outcome": outcome,
        "n_consistent_stop_families": n_consistent,
        "n_total_stop_families": n_total_fams,
        "pct_consistent": round(n_consistent / n_total_fams, 3) if n_total_fams > 0 else 0,
    }
    for asset, opt in tim_optima.items():
        key = asset.replace("-", "_").lower()
        for k, v in opt.items():
            summary[f"{key}_{k}"] = v

    pd.DataFrame([summary]).to_csv(OUT / "cross_asset_summary.csv", index=False)

    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
