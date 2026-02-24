#!/usr/bin/env python3
"""
Transaction Cost Sensitivity Analysis

Reconstructs the ETH TIM-filtered ensemble and BTC walk-forward ensemble
daily positions, then evaluates performance at 20, 40, and 60 bps round-trip
transaction costs.

Usage:
    python -m scripts.research.tsmom.cost_sensitivity
"""
from __future__ import annotations

import ast
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, ANN_FACTOR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eth_trend_sweep_v2 import (
    load_asset, dispatch_signal, apply_trailing_stop, apply_atr_trailing_stop,
)

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "artifacts" / "research" / "tsmom"
OUT = SWEEP / "full_universe"
OUT.mkdir(parents=True, exist_ok=True)

COST_LEVELS = [20, 40, 60]
OPT_TIM = 0.42
OPT_BAND = 0.05
SPLIT_DATE = "2021-12-31"

NAVY = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GRAY = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRAY,
    "figure.facecolor": "white",
})

_LABEL_MAP = {
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


def _label_to_sig_type(label):
    if label.startswith("Price_above_SMA"):
        return "price_sma"
    if label.startswith("Price_above_EMA"):
        return "price_ema"
    return _LABEL_MAP.get(label.split("_")[0])


def get_positions(row, asset_df, returns):
    close = asset_df["close"]
    high = asset_df["high"]
    low = asset_df["low"]
    open_ = asset_df["open"]
    volume = asset_df["volume"]

    label = row["label"]
    stop = row["stop"]
    params = ast.literal_eval(row["params"])
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
    sig = signal.reindex(returns.index).fillna(0)
    pos = sig.shift(1).fillna(0)
    return pos


def build_ensemble_positions(strategy_df, asset_df, returns, desc=""):
    """Build mean-weight ensemble position series from a set of strategies."""
    positions = []
    t0 = time.time()
    for i, (_, row) in enumerate(strategy_df.iterrows()):
        pos = get_positions(row, asset_df, returns)
        if pos is not None:
            positions.append(pos.values)
        if (i + 1) % 200 == 0:
            print(f"    {desc} {i+1}/{len(strategy_df)} ({time.time()-t0:.0f}s)")
    if not positions:
        return None
    pos_matrix = np.array(positions)
    ens_pos = pd.Series(pos_matrix.mean(axis=0), index=returns.index)
    print(f"    {desc} done: {len(positions)} valid strategies ({time.time()-t0:.0f}s)")
    return ens_pos


def evaluate_at_cost(ens_pos, returns, cost_bps):
    trades = ens_pos.diff().abs()
    cost = trades * (cost_bps / 10_000)
    net_ret = ens_pos * returns - cost
    equity = (1 + net_ret).cumprod()
    sharpe = float(net_ret.mean() / net_ret.std() * np.sqrt(ANN_FACTOR))
    cagr = float(equity.iloc[-1] ** (ANN_FACTOR / len(net_ret)) - 1)
    maxdd = float((equity / equity.cummax() - 1).min())
    skewness = float(net_ret.skew())
    turnover = float(trades.sum())
    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 3),
        "max_dd": round(maxdd, 3),
        "skewness": round(skewness, 3),
        "turnover": round(turnover, 1),
    }


def main():
    print("=" * 70)
    print("  TRANSACTION COST SENSITIVITY ANALYSIS")
    print("=" * 70)

    band_lo = OPT_TIM - OPT_BAND
    band_hi = OPT_TIM + OPT_BAND

    # ------------------------------------------------------------------
    # ETH: TIM-filtered ensemble (daily, TIM in [37%, 47%])
    # ------------------------------------------------------------------
    print("\n  [1/4] Loading ETH-USD data and sweep results...")
    eth = load_asset("1d", symbol="ETH-USD")
    eth_close = eth["close"]
    eth_returns = eth_close.pct_change(fill_method=None).dropna()

    eth_results = pd.read_csv(SWEEP / "eth_trend_sweep" / "results_v2.csv")
    eth_daily = eth_results[
        (eth_results["label"] != "BUY_AND_HOLD") &
        (eth_results["freq"] == "1d")
    ].copy()

    eth_band = eth_daily[
        (eth_daily["time_in_market"] >= band_lo) &
        (eth_daily["time_in_market"] <= band_hi)
    ]
    print(f"  ETH daily strategies in TIM band [{band_lo:.0%}-{band_hi:.0%}]: {len(eth_band)}")

    print("\n  [2/4] Reconstructing ETH ensemble positions...")
    eth_ens_pos = build_ensemble_positions(eth_band, eth, eth_returns, desc="ETH")

    # ------------------------------------------------------------------
    # BTC: Walk-forward ensemble (training-period TIM selection)
    # ------------------------------------------------------------------
    print("\n  [3/4] Loading BTC-USD data and sweep results...")
    btc = load_asset("1d", symbol="BTC-USD")
    btc_close = btc["close"]
    btc_returns = btc_close.pct_change(fill_method=None).dropna()

    btc_results = pd.read_csv(SWEEP / "btcusd_trend_sweep" / "results_v2.csv")
    btc_daily = btc_results[
        (btc_results["label"] != "BUY_AND_HOLD") &
        (btc_results["freq"] == "1d")
    ].copy()

    train_mask = btc_returns.index <= SPLIT_DATE
    train_ret = btc_returns[train_mask]
    train_btc = btc.loc[btc.index <= SPLIT_DATE]

    print(f"  Computing training-period TIM for {len(btc_daily)} BTC strategies...")
    t0 = time.time()
    train_tims = {}
    for i, (idx, row) in enumerate(btc_daily.iterrows()):
        pos = get_positions(row, train_btc, train_ret)
        if pos is not None:
            pos_train = pos.reindex(train_ret.index).fillna(0)
            train_tims[idx] = float((pos_train.abs() > 1e-6).mean())
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(btc_daily)} ({time.time()-t0:.0f}s)")

    train_tim_series = pd.Series(train_tims)
    btc_with_tim = btc_daily.loc[train_tim_series.index].copy()
    btc_with_tim["train_tim"] = train_tim_series.values

    wf_selected = btc_with_tim[
        (btc_with_tim["train_tim"] >= band_lo) &
        (btc_with_tim["train_tim"] <= band_hi)
    ]
    print(f"  BTC walk-forward selected: {len(wf_selected)}")

    print("\n  [4/4] Reconstructing BTC ensemble positions...")
    btc_ens_pos = build_ensemble_positions(wf_selected, btc, btc_returns, desc="BTC")

    # ------------------------------------------------------------------
    # Evaluate at each cost level
    # ------------------------------------------------------------------
    print("\n  RESULTS")
    print("  " + "=" * 66)

    rows = []
    for cost_bps in COST_LEVELS:
        row = {"cost_bps": cost_bps}
        if eth_ens_pos is not None:
            eth_perf = evaluate_at_cost(eth_ens_pos, eth_returns, cost_bps)
            row["eth_sharpe"] = eth_perf["sharpe"]
            row["eth_max_dd"] = eth_perf["max_dd"]
            row["eth_cagr"] = eth_perf["cagr"]
            row["eth_skewness"] = eth_perf["skewness"]
        if btc_ens_pos is not None:
            btc_perf = evaluate_at_cost(btc_ens_pos, btc_returns, cost_bps)
            row["btc_sharpe"] = btc_perf["sharpe"]
            row["btc_max_dd"] = btc_perf["max_dd"]
            row["btc_cagr"] = btc_perf["cagr"]
            row["btc_skewness"] = btc_perf["skewness"]
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Sharpe degradation per 20 bps
    if "eth_sharpe" in results_df.columns:
        results_df["eth_sharpe_delta"] = results_df["eth_sharpe"].diff()
    if "btc_sharpe" in results_df.columns:
        results_df["btc_sharpe_delta"] = results_df["btc_sharpe"].diff()

    results_df.to_csv(OUT / "cost_sensitivity.csv", index=False)

    print(f"\n  {'Cost (bps)':<12s} {'ETH Sharpe':>11s} {'ETH MaxDD':>10s} "
          f"{'BTC Sharpe':>11s} {'BTC MaxDD':>10s} {'ETH Δ/20bp':>11s} {'BTC Δ/20bp':>11s}")
    print(f"  {'─'*12} {'─'*11} {'─'*10} {'─'*11} {'─'*10} {'─'*11} {'─'*11}")
    for _, r in results_df.iterrows():
        eth_s = f"{r.get('eth_sharpe', 0):>10.3f}" if "eth_sharpe" in r else f"{'N/A':>10s}"
        eth_d = f"{r.get('eth_max_dd', 0):>9.1%}" if "eth_max_dd" in r else f"{'N/A':>9s}"
        btc_s = f"{r.get('btc_sharpe', 0):>10.3f}" if "btc_sharpe" in r else f"{'N/A':>10s}"
        btc_d = f"{r.get('btc_max_dd', 0):>9.1%}" if "btc_max_dd" in r else f"{'N/A':>9s}"
        eth_delta = f"{r.get('eth_sharpe_delta', float('nan')):>10.3f}" if pd.notna(r.get("eth_sharpe_delta")) else f"{'—':>10s}"
        btc_delta = f"{r.get('btc_sharpe_delta', float('nan')):>10.3f}" if pd.notna(r.get("btc_sharpe_delta")) else f"{'—':>10s}"
        print(f"  {int(r['cost_bps']):<12d} {eth_s} {eth_d} {btc_s} {btc_d} {eth_delta} {btc_delta}")

    # Also add B&H rows for context
    bh_rows = []
    for cost_bps in COST_LEVELS:
        bh_rows.append({
            "cost_bps": cost_bps,
            "eth_bh_sharpe": round(float(eth_returns.mean() / eth_returns.std() * np.sqrt(ANN_FACTOR)), 3),
            "btc_bh_sharpe": round(float(btc_returns.mean() / btc_returns.std() * np.sqrt(ANN_FACTOR)), 3),
        })
    bh_df = pd.DataFrame(bh_rows)
    print(f"\n  B&H reference — ETH Sharpe: {bh_df['eth_bh_sharpe'].iloc[0]:.3f}, "
          f"BTC Sharpe: {bh_df['btc_bh_sharpe'].iloc[0]:.3f} (cost-invariant, always invested)")

    # ------------------------------------------------------------------
    # Chart: Sharpe vs Cost Level
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    if "eth_sharpe" in results_df.columns:
        ax.plot(results_df["cost_bps"], results_df["eth_sharpe"],
                "o-", color=TEAL, lw=2, ms=8, label="ETH TIM Ensemble")
        ax.axhline(bh_df["eth_bh_sharpe"].iloc[0], color=TEAL, ls=":", lw=1, alpha=0.5)
        ax.text(62, bh_df["eth_bh_sharpe"].iloc[0], " ETH B&H", va="bottom",
                fontsize=7, color=TEAL, alpha=0.7)

    if "btc_sharpe" in results_df.columns:
        ax.plot(results_df["cost_bps"], results_df["btc_sharpe"],
                "s-", color=NAVY, lw=2, ms=8, label="BTC WF Ensemble")
        ax.axhline(bh_df["btc_bh_sharpe"].iloc[0], color=NAVY, ls=":", lw=1, alpha=0.5)
        ax.text(62, bh_df["btc_bh_sharpe"].iloc[0], " BTC B&H", va="bottom",
                fontsize=7, color=NAVY, alpha=0.7)

    ax.set_xlabel("Round-Trip Transaction Cost (bps)", fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.set_title("Transaction Cost Sensitivity — TIM Ensembles", fontweight="bold", fontsize=11)
    ax.set_xticks(COST_LEVELS)
    ax.legend(loc="best", fontsize=9, frameon=True, facecolor="white", edgecolor=LGRAY)

    for cost_bps in COST_LEVELS:
        ax.axvline(cost_bps, color=LGRAY, lw=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "cost_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Outputs saved to {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
