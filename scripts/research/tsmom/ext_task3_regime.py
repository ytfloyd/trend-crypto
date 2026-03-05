#!/usr/bin/env python3
"""
Task 3: Regime Sensitivity — Drift Dependency on ETH-USD

Sub-period analysis, synthetic drift adjustment, and bear regime isolation
for the 534 Bonferroni-surviving strategies.

Usage:
    python -m scripts.research.tsmom.ext_task3_regime
"""
from __future__ import annotations

import ast
import sys
import time
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
OUT = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_extension" / "task3"
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

SUB_PERIODS = [
    ("2017–2018", "2017-01-01", "2018-12-31"),
    ("2019–2020", "2019-01-01", "2020-12-31"),
    ("2021–2022", "2021-01-01", "2022-12-31"),
    ("2023–2026", "2023-01-01", "2026-12-31"),
]
DRIFT_TARGETS = [0.0, 0.10, 0.20, 0.30, 0.50, 0.83, 1.00, 1.50, 2.00, 3.00]
BEAR_START = "2021-11-10"
BEAR_END = "2022-11-21"


def load_eth_daily():
    panel = load_daily_bars()
    eth = panel[panel["symbol"] == SYMBOL].copy()
    eth = eth.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    eth = eth[["open", "high", "low", "close", "volume"]].astype(float)
    return eth


# Import signal dispatch from sweep script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eth_trend_sweep_v2 import (
    dispatch_signal, build_configs,
    apply_trailing_stop, apply_atr_trailing_stop,
    backtest_signal, compute_perf,
    PCT_STOP_LEVELS, PCT_STOP_LABELS, ATR_MULTS, ATR_LABELS,
)


def run_single_strategy(config_row, eth, daily_returns):
    """Run a single strategy defined by a results_v2.csv row on given data."""
    close = eth["close"]
    high = eth["high"]
    low = eth["low"]
    open_ = eth["open"]
    volume = eth["volume"]

    label = config_row["label"]
    stop = config_row["stop"]
    freq = config_row["freq"]
    params = ast.literal_eval(config_row["params"])

    # Determine signal type from label
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

    # Apply stop
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

    equity, net_ret, pos = backtest_signal(signal, daily_returns, COST_BPS)
    return compute_perf(equity, net_ret, pos)


def _label_to_sig_type(label):
    """Map a label like 'EMA_cross_5_150' back to the dispatch key."""
    mapping = {
        "SMA": "sma_cross", "EMA": "ema_cross", "DEMA": "dema_cross",
        "Hull": "hull_cross", "Price": None, "Donchian": "donchian",
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


def main():
    print("=" * 70)
    print("  TASK 3: REGIME SENSITIVITY — DRIFT DEPENDENCY")
    print("=" * 70)

    eth = load_eth_daily()
    close = eth["close"]
    daily_returns = close.pct_change(fill_method=None).dropna()
    print(f"  ETH-USD: {len(eth)} daily bars, {eth.index.min().date()} to {eth.index.max().date()}")

    # Load survivors
    results = pd.read_csv(SWEEP_DIR / "results_v2.csv")
    survivors = results[
        (results["label"] != "BUY_AND_HOLD") &
        (results["sharpe"] >= SHARPE_THRESH) &
        (results["freq"] == "1d")
    ].copy()
    print(f"  Bonferroni survivors (daily, SR≥{SHARPE_THRESH}): {len(survivors)}")

    # ================================================================
    # 3.1: Historical sub-period analysis
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  3.1: SUB-PERIOD ANALYSIS")
    print(f"  {'='*60}")

    subperiod_rows = []
    for name, start, end in SUB_PERIODS:
        sub_ret = daily_returns[(daily_returns.index >= start) & (daily_returns.index <= end)]
        if len(sub_ret) < 30:
            continue

        bh_eq = (1 + sub_ret).cumprod()
        bh_sharpe = float(sub_ret.mean() / sub_ret.std() * np.sqrt(ANN_FACTOR))
        bh_cagr = float(bh_eq.iloc[-1] ** (ANN_FACTOR / len(sub_ret)) - 1)
        bh_dd = float((bh_eq / bh_eq.cummax() - 1).min())

        sub_eth = eth.loc[(eth.index >= start) & (eth.index <= end)]

        strat_sharpes = []
        strat_cagrs = []
        strat_dds = []
        n_run = 0

        for _, row in survivors.iterrows():
            perf = run_single_strategy(row, sub_eth, sub_ret)
            if perf is not None:
                strat_sharpes.append(perf["sharpe"])
                strat_cagrs.append(perf["cagr"])
                strat_dds.append(perf["max_dd"])
                n_run += 1

        if n_run == 0:
            continue

        n_beat = sum(1 for s in strat_sharpes if s > bh_sharpe)
        med_sr = np.median(strat_sharpes)
        med_cagr = np.median(strat_cagrs)
        med_dd = np.median(strat_dds)

        subperiod_rows.append({
            "period": name, "n_strategies": n_run,
            "bh_sharpe": round(bh_sharpe, 3), "bh_cagr": round(bh_cagr, 3),
            "bh_dd": round(bh_dd, 3),
            "med_sharpe": round(med_sr, 3), "med_cagr": round(med_cagr, 3),
            "med_dd": round(med_dd, 3),
            "pct_beat_bh": round(n_beat / n_run, 3),
            "cagr_cost": round(med_cagr - bh_cagr, 3),
            "dd_compression": round(med_dd - bh_dd, 3),
        })

        print(f"\n  {name}: B&H SR={bh_sharpe:.2f} CAGR={bh_cagr:.0%} DD={bh_dd:.0%}")
        print(f"    Strategies run: {n_run}, Beat B&H: {n_beat} ({n_beat/n_run:.0%})")
        print(f"    Median: SR={med_sr:.2f} CAGR={med_cagr:.0%} DD={med_dd:.0%}")

    sub_df = pd.DataFrame(subperiod_rows)
    sub_df.to_csv(OUT / "subperiod_analysis.csv", index=False)

    # Sub-period chart
    if len(sub_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = np.arange(len(sub_df))
        w = 0.35

        axes[0].bar(x - w/2, sub_df["bh_sharpe"], w, color=RED, alpha=0.8, label="B&H")
        axes[0].bar(x + w/2, sub_df["med_sharpe"], w, color=NAVY, alpha=0.8, label="Median Survivor")
        axes[0].set_xticks(x); axes[0].set_xticklabels(sub_df["period"], fontsize=8)
        axes[0].set_ylabel("Sharpe Ratio"); axes[0].set_title("A. Sharpe", fontweight="bold")
        axes[0].legend(fontsize=7); axes[0].axhline(0, color="black", lw=0.5)

        axes[1].bar(x - w/2, sub_df["bh_cagr"] * 100, w, color=RED, alpha=0.8, label="B&H")
        axes[1].bar(x + w/2, sub_df["med_cagr"] * 100, w, color=NAVY, alpha=0.8, label="Median Survivor")
        axes[1].set_xticks(x); axes[1].set_xticklabels(sub_df["period"], fontsize=8)
        axes[1].set_ylabel("CAGR (%)"); axes[1].set_title("B. CAGR", fontweight="bold")
        axes[1].legend(fontsize=7)

        axes[2].bar(x, sub_df["pct_beat_bh"] * 100, color=TEAL, alpha=0.8)
        axes[2].axhline(50, color=RED, lw=1, ls="--")
        axes[2].set_xticks(x); axes[2].set_xticklabels(sub_df["period"], fontsize=8)
        axes[2].set_ylabel("% Beat B&H"); axes[2].set_title("C. Win Rate", fontweight="bold")

        fig.suptitle("Task 3.1: Bonferroni Survivors vs B&H by Sub-Period",
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / "subperiod_chart.png", dpi=150)
        plt.close(fig)

    # ================================================================
    # 3.2: Synthetic drift adjustment
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  3.2: SYNTHETIC DRIFT ADJUSTMENT")
    print(f"  {'='*60}")

    actual_daily_drift = daily_returns.mean()
    print(f"  Actual daily drift: {actual_daily_drift:.6f} "
          f"(≈{(1+actual_daily_drift)**365-1:.0%} ann.)")

    drift_rows = []
    for target_ann in DRIFT_TARGETS:
        target_daily = (1 + target_ann) ** (1/365) - 1
        drift_adj = target_daily - actual_daily_drift

        synth_ret = daily_returns + drift_adj
        synth_close = (1 + synth_ret).cumprod() * close.iloc[0]
        synth_close.iloc[0] = close.iloc[0]

        ratio = synth_close / close.reindex(synth_close.index)
        synth_high = eth["high"].reindex(synth_close.index) * ratio
        synth_low = eth["low"].reindex(synth_close.index) * ratio
        synth_open = eth["open"].reindex(synth_close.index) * ratio
        synth_vol = eth["volume"].reindex(synth_close.index)

        synth_eth = pd.DataFrame({
            "open": synth_open, "high": synth_high, "low": synth_low,
            "close": synth_close, "volume": synth_vol,
        })

        bh_eq = (1 + synth_ret).cumprod()
        bh_sharpe = float(synth_ret.mean() / synth_ret.std() * np.sqrt(ANN_FACTOR))

        strat_sharpes = []
        for _, row in survivors.iterrows():
            perf = run_single_strategy(row, synth_eth, synth_ret)
            if perf is not None:
                strat_sharpes.append(perf["sharpe"])

        n_beat = sum(1 for s in strat_sharpes if s > bh_sharpe) if strat_sharpes else 0
        n_run = len(strat_sharpes)
        med_sr = np.median(strat_sharpes) if strat_sharpes else np.nan

        drift_rows.append({
            "target_drift_ann": target_ann,
            "bh_sharpe": round(bh_sharpe, 3),
            "n_strategies": n_run,
            "med_sharpe": round(med_sr, 3) if not np.isnan(med_sr) else np.nan,
            "pct_beat_bh": round(n_beat / n_run, 3) if n_run > 0 else np.nan,
        })

        print(f"  Drift={target_ann:.0%}: B&H SR={bh_sharpe:.2f}, "
              f"Med Survivor SR={med_sr:.2f}, Beat B&H={n_beat}/{n_run} "
              f"({n_beat/n_run:.0%})" if n_run > 0 else f"  Drift={target_ann:.0%}: no results")

    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(OUT / "drift_sensitivity.csv", index=False)

    # Drift sensitivity chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    x_drift = [r["target_drift_ann"] * 100 for r in drift_rows]

    ax1.plot(x_drift, drift_df["bh_sharpe"], "o-", color=RED, lw=2, label="B&H")
    ax1.plot(x_drift, drift_df["med_sharpe"], "s-", color=NAVY, lw=2, label="Median Survivor")
    ax1.set_xlabel("Assumed Annual Drift (%)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("A. Sharpe vs Drift Level", fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor=LGRAY)
    ax1.axhline(0, color="black", lw=0.5)

    ax2.plot(x_drift, [r * 100 for r in drift_df["pct_beat_bh"]], "D-", color=TEAL, lw=2)
    ax2.axhline(50, color=RED, lw=1, ls="--", label="50% threshold")
    ax2.set_xlabel("Assumed Annual Drift (%)")
    ax2.set_ylabel("% Survivors Beating B&H")
    ax2.set_title("B. Win Rate vs Drift Level", fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", edgecolor=LGRAY)
    ax2.set_ylim(0, 105)

    fig.suptitle("Task 3.2: Drift Sensitivity — At What Drift Does Trend Reliably Add Value?",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "drift_sensitivity_chart.png", dpi=150)
    plt.close(fig)

    # Find crossover drift
    crossover = None
    for i in range(len(drift_df) - 1):
        if drift_df.iloc[i]["pct_beat_bh"] >= 0.5 and drift_df.iloc[i+1]["pct_beat_bh"] < 0.5:
            crossover = (drift_df.iloc[i]["target_drift_ann"] + drift_df.iloc[i+1]["target_drift_ann"]) / 2
            break
        elif drift_df.iloc[i]["pct_beat_bh"] < 0.5 and drift_df.iloc[i+1]["pct_beat_bh"] >= 0.5:
            crossover = (drift_df.iloc[i]["target_drift_ann"] + drift_df.iloc[i+1]["target_drift_ann"]) / 2
            break

    if crossover is not None:
        print(f"\n  Approximate crossover drift: {crossover:.0%} annualized")
    else:
        above_50 = drift_df[drift_df["pct_beat_bh"] >= 0.5]
        if len(above_50) == 0:
            print(f"\n  Trend never reliably beats B&H (>50% win rate) at any drift tested")
        elif len(above_50) == len(drift_df):
            print(f"\n  Trend beats B&H at all drift levels tested")
        else:
            print(f"\n  Crossover between {above_50['target_drift_ann'].max():.0%} and "
                  f"{drift_df[drift_df['pct_beat_bh'] < 0.5]['target_drift_ann'].min():.0%}")

    # ================================================================
    # 3.3: Bear regime isolation (2022)
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  3.3: BEAR REGIME ISOLATION (Nov 2021 – Nov 2022)")
    print(f"  {'='*60}")

    bear_ret = daily_returns[(daily_returns.index >= BEAR_START) & (daily_returns.index <= BEAR_END)]
    bear_eth = eth.loc[(eth.index >= BEAR_START) & (eth.index <= BEAR_END)]

    bh_bear_eq = (1 + bear_ret).cumprod()
    bh_bear_sharpe = float(bear_ret.mean() / bear_ret.std() * np.sqrt(ANN_FACTOR))
    bh_bear_return = float(bh_bear_eq.iloc[-1] - 1)
    bh_bear_dd = float((bh_bear_eq / bh_bear_eq.cummax() - 1).min())

    print(f"  Bear period: {bear_ret.index.min().date()} to {bear_ret.index.max().date()}")
    print(f"  B&H: return={bh_bear_return:.1%}, SR={bh_bear_sharpe:.2f}, DD={bh_bear_dd:.1%}")

    bear_results = []
    for _, row in survivors.iterrows():
        perf = run_single_strategy(row, bear_eth, bear_ret)
        if perf is not None:
            perf["label"] = row["label"]
            perf["stop"] = row["stop"]
            bear_results.append(perf)

    bear_df = pd.DataFrame(bear_results)
    bear_df.to_csv(OUT / "bear_regime_results.csv", index=False)

    if len(bear_df) > 0:
        n_beat = (bear_df["sharpe"] > bh_bear_sharpe).sum()
        n_positive = (bear_df["cagr"] > 0).sum()
        n_less_dd = (bear_df["max_dd"] > bh_bear_dd).sum()

        print(f"  Strategies tested: {len(bear_df)}")
        print(f"  Beat B&H Sharpe: {n_beat} ({n_beat/len(bear_df):.0%})")
        print(f"  Positive return: {n_positive} ({n_positive/len(bear_df):.0%})")
        print(f"  Less DD than B&H: {n_less_dd} ({n_less_dd/len(bear_df):.0%})")
        print(f"  Median Sharpe: {bear_df['sharpe'].median():.2f}")
        print(f"  Median CAGR: {bear_df['cagr'].median():.1%}")
        print(f"  Median MaxDD: {bear_df['max_dd'].median():.1%}")

        # Bear chart
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bins = np.linspace(-3, 3, 50)
        ax.hist(bear_df["sharpe"].values, bins=bins, color=NAVY, alpha=0.7,
                edgecolor="white", linewidth=0.3, density=True)
        ax.axvline(bh_bear_sharpe, color=RED, lw=2,
                   label=f"B&H ({bh_bear_sharpe:.2f})")
        ax.axvline(bear_df["sharpe"].median(), color=TEAL, lw=1.5, ls="--",
                   label=f"Median survivor ({bear_df['sharpe'].median():.2f})")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Sharpe Ratio (bear period)")
        ax.set_ylabel("Density")
        ax.set_title("Task 3.3: Bonferroni Survivors During 2022 Bear Market\n"
                     f"(Nov 2021 – Nov 2022, B&H return = {bh_bear_return:.0%})",
                     fontweight="bold")
        ax.legend(frameon=True, facecolor="white", edgecolor=LGRAY)
        fig.tight_layout()
        fig.savefig(OUT / "bear_regime_chart.png", dpi=150)
        plt.close(fig)

    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
