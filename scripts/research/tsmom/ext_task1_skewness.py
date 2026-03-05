#!/usr/bin/env python3
"""
Task 1: Mechanical Skewness Baseline

Generate 10,000 randomly-timed binary long/cash strategies on ETH-USD daily
bars and compare their skewness distribution to the 13,293 real trend strategies.
Quantify how much of the observed positive skewness is structural (trade form)
vs attributable to signal timing skill.

Usage:
    python -m scripts.research.tsmom.ext_task1_skewness
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

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, ANN_FACTOR

ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep"
OUT = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_extension" / "task1"
OUT.mkdir(parents=True, exist_ok=True)

NAVY  = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GRAY  = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRAY,
    "figure.facecolor": "white",
})

COST_BPS = 20
N_RANDOM = 10_000
SEED = 42
TIM_BUCKETS = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]


def load_eth_daily():
    panel = load_daily_bars()
    eth = panel[panel["symbol"] == "ETH-USD"].copy()
    eth = eth.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    eth = eth[["close"]].astype(float)
    return eth["close"]


def backtest_random(signal: np.ndarray, returns: np.ndarray, cost_bps: int = COST_BPS):
    pos = np.zeros_like(signal)
    pos[1:] = signal[:-1]
    trades = np.abs(np.diff(pos, prepend=0))
    cost = trades * (cost_bps / 10_000)
    net_ret = pos * returns - cost
    equity = np.cumprod(1 + net_ret)
    return equity, net_ret, pos


def compute_perf(equity, net_ret, pos):
    ret = net_ret[~np.isnan(net_ret)]
    if len(ret) < 60 or np.std(ret) < 1e-12:
        return None
    sharpe = float(np.mean(ret) / np.std(ret) * np.sqrt(ANN_FACTOR))
    skewness = float(pd.Series(ret).skew())
    total_days = len(ret)
    cagr = float(equity[-1] ** (ANN_FACTOR / total_days) - 1)
    running_max = np.maximum.accumulate(equity)
    maxdd = float(np.min(equity / running_max - 1))
    tim = float(np.mean(np.abs(pos) > 1e-6))
    return {"sharpe": sharpe, "skewness": skewness, "cagr": cagr, "max_dd": maxdd, "tim": tim}


def main():
    print("=" * 70)
    print("  TASK 1: MECHANICAL SKEWNESS BASELINE")
    print("=" * 70)

    close = load_eth_daily()
    returns = close.pct_change(fill_method=None).values
    returns[0] = 0.0
    n_bars = len(returns)
    print(f"  ETH-USD daily bars: {n_bars}")

    # Load real sweep results
    real = pd.read_csv(SWEEP_DIR / "results_v2.csv")
    real = real[real["label"] != "BUY_AND_HOLD"]
    print(f"  Real trend strategies: {len(real)}")

    # Generate 10,000 random strategies
    rng = np.random.default_rng(SEED)
    random_results = []

    print(f"  Generating {N_RANDOM:,} random strategies ...")
    for i in range(N_RANDOM):
        target_tim = rng.uniform(0.05, 0.95)
        signal = rng.binomial(1, target_tim, size=n_bars).astype(float)

        equity, net_ret, pos = backtest_random(signal, returns)
        perf = compute_perf(equity, net_ret, pos)
        if perf is not None:
            random_results.append(perf)

        if (i + 1) % 2000 == 0:
            print(f"    {i+1:,}/{N_RANDOM:,}")

    rand_df = pd.DataFrame(random_results)
    rand_df.to_csv(OUT / "random_strategies.csv", index=False)
    print(f"  Valid random strategies: {len(rand_df)}")

    # ── Exhibit: Side-by-side skewness distributions ────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(-2, 20, 100)
    ax.hist(rand_df["skewness"].values, bins=bins, color=GOLD, alpha=0.6,
            edgecolor="white", linewidth=0.3, density=True, label=f"Random (n={len(rand_df):,})")
    ax.hist(real["skewness"].values, bins=bins, color=NAVY, alpha=0.5,
            edgecolor="white", linewidth=0.3, density=True, label=f"Real trend (n={len(real):,})")
    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(rand_df["skewness"].median(), color=GOLD, lw=2, ls="--",
               label=f"Random median ({rand_df['skewness'].median():.2f})")
    ax.axvline(real["skewness"].median(), color=NAVY, lw=2, ls="--",
               label=f"Real median ({real['skewness'].median():.2f})")
    ax.set_xlabel("Skewness (daily returns)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Task 1: Skewness of Random vs Real Trend Strategies on ETH-USD",
        fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=LGRAY)
    ax.set_xlim(-2, 20)
    fig.tight_layout()
    fig.savefig(OUT / "skewness_distributions.png", dpi=150)
    plt.close(fig)

    # ── Exhibit: Skewness by TIM bucket ─────────────────────────────
    n_pos_random = (rand_df["skewness"] > 0).sum()
    n_pos_real = (real["skewness"] > 0).sum()
    print(f"\n  Positive skewness: random {n_pos_random/len(rand_df):.1%}, "
          f"real {n_pos_real/len(real):.1%}")
    print(f"  Median skewness: random {rand_df['skewness'].median():.3f}, "
          f"real {real['skewness'].median():.3f}")

    bucket_rows = []
    print(f"\n  {'TIM Bucket':<12s} {'Rand Med':>9s} {'Real Med':>9s} {'Diff':>7s} "
          f"{'W stat':>10s} {'p-value':>10s} {'Signif':>7s}")
    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*7} {'─'*10} {'─'*10} {'─'*7}")

    for lo, hi in TIM_BUCKETS:
        rand_bucket = rand_df[(rand_df["tim"] >= lo) & (rand_df["tim"] < hi)]
        real_bucket = real[(real["time_in_market"] >= lo) & (real["time_in_market"] < hi)]

        if len(rand_bucket) < 10 or len(real_bucket) < 10:
            continue

        rand_med = rand_bucket["skewness"].median()
        real_med = real_bucket["skewness"].median()
        stat, pval = sp_stats.mannwhitneyu(
            real_bucket["skewness"].values,
            rand_bucket["skewness"].values,
            alternative="two-sided",
        )
        sig = "Yes" if pval < 0.05 else "No"

        print(f"  {lo:.0%}–{hi:.0%}      {rand_med:>9.3f} {real_med:>9.3f} "
              f"{real_med - rand_med:>+7.3f} {stat:>10.0f} {pval:>10.4f} {sig:>7s}")

        bucket_rows.append({
            "tim_lo": lo, "tim_hi": hi,
            "n_random": len(rand_bucket), "n_real": len(real_bucket),
            "random_median_skew": round(rand_med, 4),
            "real_median_skew": round(real_med, 4),
            "difference": round(real_med - rand_med, 4),
            "wilcoxon_U": round(stat, 1),
            "p_value": round(pval, 6),
            "significant_p05": sig,
        })

    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(OUT / "skewness_by_tim_bucket.csv", index=False)

    # ── Exhibit: TIM bucket comparison chart ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(bucket_df))
    w = 0.35
    ax.bar(x - w/2, bucket_df["random_median_skew"], w, color=GOLD, alpha=0.8,
           label="Random", edgecolor="white")
    ax.bar(x + w/2, bucket_df["real_median_skew"], w, color=NAVY, alpha=0.8,
           label="Real trend", edgecolor="white")
    for i, row in bucket_df.iterrows():
        if row["significant_p05"] == "Yes":
            ax.text(i + w/2, row["real_median_skew"] + 0.1, "*",
                    ha="center", fontsize=12, fontweight="bold", color=RED)
    labels = [f"{int(r['tim_lo']*100)}–{int(r['tim_hi']*100)}%"
              for _, r in bucket_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time-in-Market Bucket")
    ax.set_ylabel("Median Skewness")
    ax.set_title("Task 1: Median Skewness by TIM Bucket — Random vs Real Trend\n"
                 "(* = significant difference at p < 0.05, Wilcoxon rank-sum)",
                 fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=LGRAY)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "skewness_by_tim.png", dpi=150)
    plt.close(fig)

    # ── Decomposition: structural vs signal skill ───────────────────
    overall_rand_med = rand_df["skewness"].median()
    overall_real_med = real["skewness"].median()
    bh_skew = 0.3646
    total_improvement = overall_real_med - bh_skew
    structural_component = overall_rand_med - bh_skew
    signal_component = overall_real_med - overall_rand_med

    print(f"\n  {'='*60}")
    print(f"  SKEWNESS DECOMPOSITION")
    print(f"  {'='*60}")
    print(f"  B&H skewness:          {bh_skew:.3f}")
    print(f"  Random median:         {overall_rand_med:.3f}")
    print(f"  Real trend median:     {overall_real_med:.3f}")
    print(f"  Total improvement:     {total_improvement:+.3f}")
    print(f"  Structural (random):   {structural_component:+.3f} "
          f"({abs(structural_component/total_improvement)*100:.0f}% of total)")
    print(f"  Signal skill:          {signal_component:+.3f} "
          f"({abs(signal_component/total_improvement)*100:.0f}% of total)")

    summary = {
        "bh_skewness": bh_skew,
        "random_median_skewness": round(overall_rand_med, 4),
        "real_median_skewness": round(overall_real_med, 4),
        "total_improvement_vs_bh": round(total_improvement, 4),
        "structural_component": round(structural_component, 4),
        "signal_component": round(signal_component, 4),
        "pct_structural": round(abs(structural_component / total_improvement) * 100, 1),
        "pct_signal": round(abs(signal_component / total_improvement) * 100, 1),
        "pct_random_positive_skew": round(n_pos_random / len(rand_df) * 100, 1),
        "pct_real_positive_skew": round(n_pos_real / len(real) * 100, 1),
    }
    pd.DataFrame([summary]).to_csv(OUT / "skewness_decomposition.csv", index=False)

    print(f"\n  Conclusion: {summary['pct_structural']:.0f}% of the skewness improvement is "
          f"structural (trade form), {summary['pct_signal']:.0f}% from signal timing.")
    print(f"  {summary['pct_random_positive_skew']:.0f}% of random strategies have positive "
          f"skewness vs {summary['pct_real_positive_skew']:.0f}% of real trend strategies.")
    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
