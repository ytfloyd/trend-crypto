#!/usr/bin/env python3
"""
Task 4: The Time-in-Market Dial — Optimal TIM Targeting

Identify the empirically optimal TIM level, test whether Bonferroni survivors
cluster there, identify which signal families target it, and construct
a TIM-filtered ensemble.

Usage:
    python -m scripts.research.tsmom.ext_task4_tim
"""
from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, ANN_FACTOR

ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep"
OUT = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_extension" / "task4"
OUT.mkdir(parents=True, exist_ok=True)

NAVY  = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GREEN = "#336633"; GRAY  = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRAY,
    "figure.facecolor": "white",
})

SYMBOL = "ETH-USD"
COST_BPS = 20
SHARPE_THRESH = 1.38
N_BOOTSTRAP = 1000
SEED = 42

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eth_trend_sweep_v2 import (
    dispatch_signal, apply_trailing_stop, apply_atr_trailing_stop,
    backtest_signal, compute_perf,
)


def load_eth_daily():
    panel = load_daily_bars()
    eth = panel[panel["symbol"] == SYMBOL].copy()
    eth = eth.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    eth = eth[["open", "high", "low", "close", "volume"]].astype(float)
    return eth


def _label_to_sig_type(label):
    mapping = {
        "SMA": "sma_cross", "EMA": "ema_cross", "DEMA": "dema_cross",
        "Hull": "hull_cross", "Donchian": "donchian",
        "Boll": "bollinger", "Keltner": "keltner", "Supertrend": "supertrend",
        "Momentum": "momentum", "MomThresh": "mom_threshold",
        "VolScaledMom": "vol_scaled_mom", "LREG": "lreg", "MACD": "macd",
        "RSI": "rsi", "ADX": "adx", "CCI": "cci", "Aroon": "aroon",
        "Stoch": "stoch", "SAR": "sar", "WilliamsR": "williams_r",
        "MFI": "mfi", "TRIX": "trix", "PPO": "ppo", "APO": "apo",
        "ADOSC": "adosc", "MOM": "talib_mom", "ROC": "talib_roc",
        "CMO": "talib_cmo", "Ichimoku": "ichimoku", "OBV": "obv_trend",
        "HeikinAshi": "heikin_ashi", "Kaufman": "kaufman", "VWAP": "vwap",
        "DualMom": "dual_mom", "TripleMA": "triple_ma", "Turtle": "turtle",
        "RegimeSMA": "regime_sma", "ATR": "atr_breakout",
        "Close": "close_above_high", "MeanRevBand": "mean_rev_band",
    }
    if label.startswith("Price_above_SMA"):
        return "price_sma"
    if label.startswith("Price_above_EMA"):
        return "price_ema"
    prefix = label.split("_")[0]
    return mapping.get(prefix)


def get_daily_positions(config_row, eth, daily_returns):
    """Return daily position series for a strategy."""
    close = eth["close"]
    high = eth["high"]
    low = eth["low"]
    open_ = eth["open"]
    volume = eth["volume"]

    label = config_row["label"]
    stop = config_row["stop"]
    params = ast.literal_eval(config_row["params"])
    sig_type = _label_to_sig_type(label)
    if sig_type is None:
        return None

    try:
        raw_signal = dispatch_signal(sig_type, close, high, low, open_, volume, **params)
    except Exception:
        return None

    base_signal = raw_signal.dropna()
    if len(base_signal) < 30:
        return None

    if stop == "none":
        signal = base_signal
    elif stop.startswith("pct"):
        pct_val = {"pct5": 0.05, "pct10": 0.10, "pct20": 0.20}[stop]
        signal = apply_trailing_stop(base_signal, close, pct_val)
    elif stop.startswith("atr"):
        atr_val = float(stop.replace("atr", ""))
        signal = apply_atr_trailing_stop(base_signal, close, high, low, atr_val, 14)
    else:
        return None

    sig = signal.reindex(daily_returns.index).fillna(0)
    pos = sig.shift(1).fillna(0)
    return pos


def main():
    print("=" * 70)
    print("  TASK 4: TIM OPTIMIZATION — OPTIMAL TIM TARGETING")
    print("=" * 70)

    eth = load_eth_daily()
    close = eth["close"]
    daily_returns = close.pct_change(fill_method=None).dropna()
    print(f"  ETH-USD: {len(eth)} bars")

    results = pd.read_csv(SWEEP_DIR / "results_v2.csv")
    strats = results[results["label"] != "BUY_AND_HOLD"].copy()
    survivors = strats[strats["sharpe"] >= SHARPE_THRESH].copy()
    daily_strats = strats[strats["freq"] == "1d"].copy()
    print(f"  All strategies: {len(strats)}")
    print(f"  Bonferroni survivors: {len(survivors)}")
    print(f"  Daily strategies: {len(daily_strats)}")

    # ================================================================
    # 4.1: Smoothed Sharpe vs TIM curve with bootstrap CI
    # ================================================================
    print(f"\n  4.1: SHARPE VS TIM CURVE")

    tim_bins = np.arange(0.05, 0.95, 0.01)
    med_sharpe_by_tim = []
    for center in tim_bins:
        lo, hi = center - 0.025, center + 0.025
        bucket = strats[(strats["time_in_market"] >= lo) & (strats["time_in_market"] < hi)]
        med_sharpe_by_tim.append(bucket["sharpe"].median() if len(bucket) >= 10 else np.nan)

    med_sharpe_arr = np.array(med_sharpe_by_tim)
    valid = ~np.isnan(med_sharpe_arr)
    opt_idx = np.nanargmax(med_sharpe_arr)
    opt_tim = tim_bins[opt_idx]
    opt_sharpe = med_sharpe_arr[opt_idx]
    print(f"  Empirical optimum: TIM = {opt_tim:.0%}, Median Sharpe = {opt_sharpe:.3f}")

    # Bootstrap CI for optimal TIM
    rng = np.random.default_rng(SEED)
    boot_opt_tims = []
    all_tims = strats["time_in_market"].values
    all_sharpes = strats["sharpe"].values

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(len(strats), size=len(strats), replace=True)
        b_tims = all_tims[idx]
        b_sharpes = all_sharpes[idx]
        b_meds = []
        for center in tim_bins:
            lo, hi = center - 0.025, center + 0.025
            mask = (b_tims >= lo) & (b_tims < hi)
            if mask.sum() >= 10:
                b_meds.append(np.median(b_sharpes[mask]))
            else:
                b_meds.append(np.nan)
        b_arr = np.array(b_meds)
        if np.any(~np.isnan(b_arr)):
            boot_opt_tims.append(tim_bins[np.nanargmax(b_arr)])

    ci_lo = np.percentile(boot_opt_tims, 5)
    ci_hi = np.percentile(boot_opt_tims, 95)
    print(f"  90% CI for optimal TIM: [{ci_lo:.0%}, {ci_hi:.0%}]")

    # Bootstrap CI for the Sharpe curve
    boot_curves = np.full((N_BOOTSTRAP, len(tim_bins)), np.nan)
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(len(strats), size=len(strats), replace=True)
        b_tims = all_tims[idx]
        b_sharpes = all_sharpes[idx]
        for j, center in enumerate(tim_bins):
            lo, hi = center - 0.025, center + 0.025
            mask = (b_tims >= lo) & (b_tims < hi)
            if mask.sum() >= 10:
                boot_curves[b, j] = np.median(b_sharpes[mask])

    ci_lo_curve = np.nanpercentile(boot_curves, 5, axis=0)
    ci_hi_curve = np.nanpercentile(boot_curves, 95, axis=0)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tim_bins[valid] * 100, med_sharpe_arr[valid], color=NAVY, lw=2,
            label="Median Sharpe (5% TIM bins)")
    ax.fill_between(tim_bins * 100, ci_lo_curve, ci_hi_curve, alpha=0.15, color=NAVY,
                    label="90% bootstrap CI")
    ax.axvline(opt_tim * 100, color=TEAL, lw=1.5, ls="--",
               label=f"Optimum: {opt_tim:.0%} (SR={opt_sharpe:.2f})")
    ax.axvspan(ci_lo * 100, ci_hi * 100, alpha=0.1, color=TEAL)
    bh_sharpe = 1.11
    ax.axhline(bh_sharpe, color=RED, lw=1, ls=":", label=f"B&H ({bh_sharpe:.2f})")
    ax.set_xlabel("Time in Market (%)")
    ax.set_ylabel("Median Sharpe Ratio")
    ax.set_title("Task 4.1: Sharpe vs Time-in-Market with 90% Bootstrap CI",
                 fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    fig.savefig(OUT / "sharpe_vs_tim_curve.png", dpi=150)
    plt.close(fig)

    # ================================================================
    # 4.2: Survivor TIM distribution
    # ================================================================
    print(f"\n  4.2: SURVIVOR TIM DISTRIBUTION")

    ks_stat, ks_p = sp_stats.ks_2samp(
        survivors["time_in_market"].values,
        strats["time_in_market"].values,
    )
    print(f"  KS test (survivors vs all): stat={ks_stat:.4f}, p={ks_p:.6f}")
    print(f"  Survivor median TIM: {survivors['time_in_market'].median():.0%}")
    print(f"  All-strategy median TIM: {strats['time_in_market'].median():.0%}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, 1, 40)
    ax.hist(strats["time_in_market"].values, bins=bins, color=NAVY, alpha=0.4,
            density=True, edgecolor="white", linewidth=0.3, label=f"All (n={len(strats):,})")
    ax.hist(survivors["time_in_market"].values, bins=bins, color=TEAL, alpha=0.7,
            density=True, edgecolor="white", linewidth=0.3, label=f"Survivors (n={len(survivors)})")
    ax.axvline(opt_tim, color=RED, lw=1.5, ls="--", label=f"Optimal TIM ({opt_tim:.0%})")
    ax.set_xlabel("Time in Market")
    ax.set_ylabel("Density")
    ax.set_title(f"Task 4.2: TIM Distribution — Survivors vs All Strategies\n"
                 f"(KS p={ks_p:.4f})",
                 fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=LGRAY)
    fig.tight_layout()
    fig.savefig(OUT / "survivor_tim_distribution.png", dpi=150)
    plt.close(fig)

    # ================================================================
    # 4.3: Signal families in optimal TIM band
    # ================================================================
    print(f"\n  4.3: SIGNAL FAMILIES IN OPTIMAL TIM BAND")
    opt_band_lo = opt_tim - 0.05
    opt_band_hi = opt_tim + 0.05
    print(f"  Optimal band: {opt_band_lo:.0%} – {opt_band_hi:.0%}")

    fam_rows = []
    for fam, group in daily_strats.groupby("signal_family"):
        if len(group) < 3:
            continue
        in_band = group[(group["time_in_market"] >= opt_band_lo) &
                        (group["time_in_market"] <= opt_band_hi)]
        out_band = group[(group["time_in_market"] < opt_band_lo) |
                         (group["time_in_market"] > opt_band_hi)]
        fam_rows.append({
            "family": fam,
            "n_total": len(group),
            "mean_tim": round(group["time_in_market"].mean(), 3),
            "n_in_band": len(in_band),
            "pct_in_band": round(len(in_band) / len(group), 3),
            "med_sharpe_in_band": round(in_band["sharpe"].median(), 3) if len(in_band) > 0 else np.nan,
            "med_sharpe_out_band": round(out_band["sharpe"].median(), 3) if len(out_band) > 0 else np.nan,
        })

    fam_df = pd.DataFrame(fam_rows).sort_values("pct_in_band", ascending=False)
    fam_df.to_csv(OUT / "family_tim_targeting.csv", index=False)

    print(f"\n  {'Family':<16s} {'N':>4s} {'Mean TIM':>9s} {'In Band':>8s} "
          f"{'% Band':>7s} {'SR in':>6s} {'SR out':>7s}")
    print(f"  {'─'*16} {'─'*4} {'─'*9} {'─'*8} {'─'*7} {'─'*6} {'─'*7}")
    for _, row in fam_df.head(15).iterrows():
        print(f"  {row['family']:<16s} {row['n_total']:>4d} "
              f"{row['mean_tim']:>8.0%} {row['n_in_band']:>8d} "
              f"{row['pct_in_band']:>6.0%} {row['med_sharpe_in_band']:>6.2f} "
              f"{row['med_sharpe_out_band']:>7.2f}")

    # ================================================================
    # 4.4: TIM-filtered ensemble
    # ================================================================
    print(f"\n  4.4: TIM-FILTERED ENSEMBLE")

    band_strats = daily_strats[
        (daily_strats["time_in_market"] >= opt_band_lo) &
        (daily_strats["time_in_market"] <= opt_band_hi)
    ]
    print(f"  Strategies in optimal band: {len(band_strats)}")

    all_positions = []
    valid_labels = []
    t0 = __import__("time").time()

    for i, (_, row) in enumerate(band_strats.iterrows()):
        pos = get_daily_positions(row, eth, daily_returns)
        if pos is not None:
            all_positions.append(pos.values)
            valid_labels.append(row["label"])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(band_strats)} signals computed "
                  f"({__import__('time').time()-t0:.0f}s)")

    print(f"  Valid position series: {len(all_positions)}")

    if len(all_positions) > 0:
        pos_matrix = np.array(all_positions)
        ensemble_pos = pd.Series(pos_matrix.mean(axis=0), index=daily_returns.index)

        # Ensemble returns
        trades = ensemble_pos.diff().abs()
        cost = trades * (COST_BPS / 10_000)
        ens_ret = ensemble_pos * daily_returns - cost
        ens_equity = (1 + ens_ret).cumprod()

        ens_perf = {
            "sharpe": round(float(ens_ret.mean() / ens_ret.std() * np.sqrt(ANN_FACTOR)), 3),
            "cagr": round(float(ens_equity.iloc[-1] ** (ANN_FACTOR / len(ens_ret)) - 1), 3),
            "max_dd": round(float((ens_equity / ens_equity.cummax() - 1).min()), 3),
            "skewness": round(float(ens_ret.skew()), 3),
            "avg_tim": round(float((ensemble_pos > 1e-6).mean()), 3),
            "turnover": round(float(trades.sum()), 1),
        }

        # B&H
        bh_eq = (1 + daily_returns).cumprod()
        bh_perf = {
            "sharpe": 1.11, "cagr": 0.827, "max_dd": -0.940,
            "skewness": 0.365, "avg_tim": 1.0, "turnover": 0,
        }

        # Top survivor
        top_surv = strats.nlargest(1, "sharpe").iloc[0]

        # Median single strategy
        med_strat = {
            "sharpe": round(strats["sharpe"].median(), 3),
            "cagr": round(strats["cagr"].median(), 3),
            "max_dd": round(strats["max_dd"].median(), 3),
            "skewness": round(strats["skewness"].median(), 3),
            "avg_tim": round(strats["time_in_market"].median(), 3),
        }

        # Comparison table
        comp_data = [
            {"strategy": "Buy & Hold", **bh_perf},
            {"strategy": "Median Single Strategy", **med_strat},
            {"strategy": f"Top Survivor ({top_surv['label']})",
             "sharpe": round(top_surv["sharpe"], 3),
             "cagr": round(top_surv["cagr"], 3),
             "max_dd": round(top_surv["max_dd"], 3),
             "skewness": round(top_surv["skewness"], 3),
             "avg_tim": round(top_surv["time_in_market"], 3)},
            {"strategy": f"TIM-Filtered Ensemble (n={len(all_positions)})", **ens_perf},
        ]
        comp_df = pd.DataFrame(comp_data)
        comp_df.to_csv(OUT / "ensemble_comparison.csv", index=False)

        print(f"\n  {'Strategy':<40s} {'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>7s} "
              f"{'Skew':>6s} {'TIM':>5s}")
        print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*5}")
        for _, row in comp_df.iterrows():
            print(f"  {row['strategy']:<40s} {row['sharpe']:>7.2f} "
                  f"{row.get('cagr', 'N/A'):>7} "
                  f"{row.get('max_dd', 'N/A'):>7} "
                  f"{row.get('skewness', 'N/A'):>6} {row.get('avg_tim', 'N/A'):>5}")

        # Ensemble equity curve chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

        ax1.plot(bh_eq.index, bh_eq.values, color=RED, lw=1, alpha=0.6, label="Buy & Hold")
        ax1.plot(ens_equity.index, ens_equity.values, color=NAVY, lw=1.5, label="TIM Ensemble")
        ax1.set_yscale("log")
        ax1.set_ylabel("Equity (log scale)")
        ax1.set_title(f"Task 4.4: TIM-Filtered Ensemble (n={len(all_positions)}) vs Buy & Hold",
                      fontweight="bold")
        ax1.legend(frameon=True, facecolor="white", edgecolor=LGRAY)

        ax2.fill_between(ensemble_pos.index, 0, ensemble_pos.values, color=TEAL, alpha=0.5)
        ax2.set_ylabel("Ensemble Position")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(OUT / "ensemble_equity.png", dpi=150)
        plt.close(fig)

    # Save task summary
    summary = {
        "optimal_tim": round(opt_tim, 3),
        "optimal_sharpe": round(opt_sharpe, 3),
        "ci_90_lo": round(ci_lo, 3),
        "ci_90_hi": round(ci_hi, 3),
        "ks_stat": round(ks_stat, 4),
        "ks_p": round(ks_p, 6),
        "n_in_band": len(band_strats),
        "ensemble_sharpe": ens_perf["sharpe"] if len(all_positions) > 0 else np.nan,
        "ensemble_cagr": ens_perf["cagr"] if len(all_positions) > 0 else np.nan,
        "ensemble_dd": ens_perf["max_dd"] if len(all_positions) > 0 else np.nan,
    }
    pd.DataFrame([summary]).to_csv(OUT / "task4_summary.csv", index=False)

    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
