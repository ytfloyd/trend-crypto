#!/usr/bin/env python3
"""
Cross-Asset Walk-Forward TIM Ensemble on BTC-USD

Replicates the Part 11 walk-forward methodology on BTC-USD:
- Train TIM on 2017-2021, select strategies in [37%, 47%] band
- Deploy ensemble on 2022-2026 out-of-sample
- Compare to BTC B&H and to ETH walk-forward ensemble

BTC-only by design: identical 9-year history and regime alignment to ETH.
SOL lacks the 2017-2021 training window.

Usage:
    python -m scripts.research.tsmom.cross_asset_walkforward
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
    backtest_signal, compute_perf, COST_BPS,
)

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "artifacts" / "research" / "tsmom"
OUT = SWEEP / "cross_asset"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#003366"; TEAL = "#006B6B"; RED = "#CC3333"; GOLD = "#CC9933"
GRAY = "#808080"; LGRAY = "#D0D0D0"; BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.facecolor": BG, "axes.edgecolor": LGRAY,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRAY,
    "figure.facecolor": "white",
})

OPT_TIM = 0.42
OPT_BAND = 0.05
SPLIT_DATE = "2021-12-31"


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
    return mapping.get(label.split("_")[0])


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


def eval_ensemble(positions_list, returns, label):
    if not positions_list:
        return None
    pos_matrix = np.array([p.values for p in positions_list])
    ens_pos = pd.Series(pos_matrix.mean(axis=0), index=returns.index)
    trades = ens_pos.diff().abs()
    cost = trades * (COST_BPS / 10_000)
    net_ret = ens_pos * returns - cost
    equity = (1 + net_ret).cumprod()
    sharpe = float(net_ret.mean() / net_ret.std() * np.sqrt(ANN_FACTOR))
    cagr = float(equity.iloc[-1] ** (ANN_FACTOR / len(net_ret)) - 1)
    maxdd = float((equity / equity.cummax() - 1).min())
    skewness = float(net_ret.skew())
    tim = float((ens_pos > 1e-6).mean())
    mean_pos = float(ens_pos.mean())
    return {"strategy": label, "sharpe": round(sharpe, 3), "cagr": round(cagr, 3),
            "max_dd": round(maxdd, 3), "skewness": round(skewness, 3),
            "tim": round(tim, 3), "mean_pos": round(mean_pos, 3),
            "n_strategies": len(positions_list)}


def main():
    print("=" * 70)
    print("  CROSS-ASSET WALK-FORWARD TIM ENSEMBLE: BTC-USD")
    print("=" * 70)

    btc = load_asset("1d", symbol="BTC-USD")
    close = btc["close"]
    daily_returns = close.pct_change(fill_method=None).dropna()
    print(f"  BTC-USD: {len(btc)} bars, {btc.index.min().date()} to {btc.index.max().date()}")

    # Load BTC sweep results (daily only)
    btc_results = pd.read_csv(SWEEP / "btcusd_trend_sweep" / "results_v2.csv")
    btc_daily = btc_results[
        (btc_results["label"] != "BUY_AND_HOLD") &
        (btc_results["freq"] == "1d")
    ].copy()
    print(f"  Daily strategies: {len(btc_daily)}")

    # Split
    train_mask = daily_returns.index <= SPLIT_DATE
    test_mask = daily_returns.index > SPLIT_DATE
    train_ret = daily_returns[train_mask]
    test_ret = daily_returns[test_mask]
    train_btc = btc.loc[btc.index <= SPLIT_DATE]
    print(f"  Train: {train_ret.index.min().date()} to {train_ret.index.max().date()} ({len(train_ret)} bars)")
    print(f"  Test:  {test_ret.index.min().date()} to {test_ret.index.max().date()} ({len(test_ret)} bars)")

    # Compute training-period TIM
    band_lo, band_hi = OPT_TIM - OPT_BAND, OPT_TIM + OPT_BAND
    print(f"\n  Computing training-period TIM for {len(btc_daily)} strategies...")
    train_tims = {}
    t0 = time.time()
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
    btc_with_tim["full_tim"] = btc_with_tim["time_in_market"]
    print(f"  Valid: {len(btc_with_tim)}")

    tim_corr = btc_with_tim[["train_tim", "full_tim"]].corr().iloc[0, 1]
    print(f"  Correlation(train TIM, full TIM): {tim_corr:.3f}")

    # Select walk-forward band
    wf_selected = btc_with_tim[
        (btc_with_tim["train_tim"] >= band_lo) & (btc_with_tim["train_tim"] <= band_hi)
    ]
    full_in_band = (wf_selected["full_tim"] >= band_lo) & (wf_selected["full_tim"] <= band_hi)
    wf_precision = full_in_band.mean()
    print(f"\n  Walk-forward band [{band_lo:.0%}–{band_hi:.0%}]:")
    print(f"    Selected: {len(wf_selected)}")
    print(f"    Precision: {wf_precision:.0%}")

    # Build walk-forward ensemble
    print(f"\n  Building walk-forward ensemble...")
    wf_positions = []
    for i, (idx, row) in enumerate(wf_selected.iterrows()):
        pos = get_positions(row, btc, daily_returns)
        if pos is not None:
            wf_positions.append(pos)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(wf_selected)}")

    print(f"  Walk-forward ensemble: {len(wf_positions)} strategies")

    # Evaluate
    full_wf = eval_ensemble(wf_positions, daily_returns, "BTC Walk-Forward Ensemble")
    test_wf_pos = [p.reindex(test_ret.index).fillna(0) for p in wf_positions]
    test_wf = eval_ensemble(test_wf_pos, test_ret, "BTC Walk-Forward (OOS)")

    # B&H
    bh_eq_full = (1 + daily_returns).cumprod()
    bh_sharpe_full = float(daily_returns.mean() / daily_returns.std() * np.sqrt(ANN_FACTOR))
    bh_eq_test = (1 + test_ret).cumprod()
    bh_sharpe_test = float(test_ret.mean() / test_ret.std() * np.sqrt(ANN_FACTOR))

    bh_full = {"strategy": "BTC Buy & Hold", "sharpe": round(bh_sharpe_full, 3),
               "cagr": round(float(bh_eq_full.iloc[-1] ** (ANN_FACTOR / len(daily_returns)) - 1), 3),
               "max_dd": round(float((bh_eq_full / bh_eq_full.cummax() - 1).min()), 3),
               "skewness": round(float(daily_returns.skew()), 3),
               "tim": 1.0, "mean_pos": 1.0, "n_strategies": 1}
    bh_test = {"strategy": "BTC Buy & Hold (OOS)", "sharpe": round(bh_sharpe_test, 3),
               "cagr": round(float(bh_eq_test.iloc[-1] ** (ANN_FACTOR / len(test_ret)) - 1), 3),
               "max_dd": round(float((bh_eq_test / bh_eq_test.cummax() - 1).min()), 3),
               "skewness": round(float(test_ret.skew()), 3),
               "tim": 1.0, "mean_pos": 1.0, "n_strategies": 1}

    comp_rows = [bh_full, full_wf, bh_test, test_wf]
    comp_rows = [r for r in comp_rows if r is not None]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT / "btc_walkforward_comparison.csv", index=False)

    print(f"\n  {'Strategy':<35s} {'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>7s} "
          f"{'Skew':>6s} {'MeanPos':>8s}")
    print(f"  {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*8}")
    for _, r in comp_df.iterrows():
        print(f"  {r['strategy']:<35s} {r['sharpe']:>7.2f} {r['cagr']:>7.1%} "
              f"{r['max_dd']:>7.1%} {r['skewness']:>6.2f} {r['mean_pos']:>8.1%}")

    # Equity curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    if wf_positions:
        wf_ens_pos = pd.Series(np.array([p.values for p in wf_positions]).mean(axis=0),
                               index=daily_returns.index)
        wf_trades = wf_ens_pos.diff().abs()
        wf_net = wf_ens_pos * daily_returns - wf_trades * (COST_BPS / 10_000)
        wf_equity = (1 + wf_net).cumprod()

        ax1.plot(wf_equity.index, wf_equity.values, color=TEAL, lw=1.2,
                 label=f"WF Ensemble (SR={full_wf['sharpe']:.2f})")
    ax1.plot(bh_eq_full.index, bh_eq_full.values, color=RED, lw=0.8, alpha=0.5,
             label=f"BTC B&H (SR={bh_full['sharpe']:.2f})")
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log)")
    ax1.set_title("A. BTC Full Period (2017–2026)", fontweight="bold")
    ax1.legend(fontsize=7, frameon=True, facecolor="white", edgecolor=LGRAY)
    ax1.axvline(pd.Timestamp(SPLIT_DATE), color=GRAY, lw=1, ls=":", alpha=0.5)

    if wf_positions:
        wf_test_eq = (1 + wf_net[test_mask]).cumprod()
        ax2.plot(wf_test_eq.index, wf_test_eq.values, color=TEAL, lw=1.2,
                 label=f"WF Ensemble (SR={test_wf['sharpe']:.2f})" if test_wf else "WF")
    ax2.plot(bh_eq_test.index, bh_eq_test.values, color=RED, lw=0.8, alpha=0.5,
             label=f"BTC B&H (SR={bh_test['sharpe']:.2f})")
    ax2.set_yscale("log")
    ax2.set_ylabel("Equity (log)")
    ax2.set_title("B. BTC Out-of-Sample (2022–2026)", fontweight="bold")
    ax2.legend(fontsize=7, frameon=True, facecolor="white", edgecolor=LGRAY)

    fig.suptitle("BTC-USD Walk-Forward TIM Ensemble vs Buy & Hold", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "btc_walkforward_equity.png", dpi=150)
    plt.close(fig)

    # Summary
    summary = {
        "btc_tim_corr": round(tim_corr, 3),
        "btc_wf_selected": len(wf_selected),
        "btc_wf_precision": round(wf_precision, 3),
        "btc_wf_full_sharpe": full_wf["sharpe"] if full_wf else None,
        "btc_wf_oos_sharpe": test_wf["sharpe"] if test_wf else None,
        "btc_bh_oos_sharpe": bh_test["sharpe"],
        "btc_wf_oos_maxdd": test_wf["max_dd"] if test_wf else None,
        "btc_bh_oos_maxdd": bh_test["max_dd"],
    }
    pd.DataFrame([summary]).to_csv(OUT / "btc_walkforward_summary.csv", index=False)

    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
