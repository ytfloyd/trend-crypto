#!/usr/bin/env python3
"""
Cross-Asset Replication: Run ETH daily Bonferroni survivors on Tier 2 assets
(LTC-USD, LINK-USD, ATOM-USD) without any re-optimization.

This is the sharpest test of structural signal: strategies selected on one
asset are applied to completely different assets with zero per-asset tuning.

Usage:
    python -m scripts.research.tsmom.cross_asset_replication
"""
from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, ANN_FACTOR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eth_trend_sweep_v2 import (
    dispatch_signal, apply_trailing_stop, apply_atr_trailing_stop,
    backtest_signal, compute_perf, COST_BPS,
)

ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = ROOT / "artifacts" / "research" / "tsmom"
OUT = SWEEP_DIR / "cross_asset"
OUT.mkdir(parents=True, exist_ok=True)

TIER2_ASSETS = ["LTC-USD", "LINK-USD", "ATOM-USD"]
SAMPLE_YEARS_ETH = 9
N_EFF = 493 * 3
BONF_Z = sp_stats.norm.ppf(1 - 0.05 / N_EFF / 2)
SHARPE_THRESH_ETH = BONF_Z / np.sqrt(SAMPLE_YEARS_ETH)


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


def run_strategy_on_asset(row, asset_df):
    """Run a single strategy configuration on an arbitrary asset's OHLCV data."""
    close = asset_df["close"]
    high = asset_df["high"]
    low = asset_df["low"]
    open_ = asset_df["open"]
    volume = asset_df["volume"]
    daily_returns = close.pct_change(fill_method=None).dropna()

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

    equity, net_ret, pos = backtest_signal(signal, daily_returns)
    perf = compute_perf(equity, net_ret, pos)
    if perf is None:
        return None

    perf["label"] = label
    perf["stop"] = stop
    perf["stop_type"] = row.get("stop_type", "none")
    perf["full_label"] = row["full_label"]
    perf["signal_family"] = row["signal_family"]
    perf["freq"] = row["freq"]
    perf["params"] = row["params"]
    return perf


def main():
    print("=" * 70)
    print("  CROSS-ASSET REPLICATION: ETH SURVIVORS ON TIER 2 ASSETS")
    print("=" * 70)

    # Load ETH results and identify daily Bonferroni survivors
    eth_results = pd.read_csv(SWEEP_DIR / "eth_trend_sweep" / "results_v2.csv")
    eth_strats = eth_results[eth_results["label"] != "BUY_AND_HOLD"].copy()
    eth_daily = eth_strats[eth_strats["freq"] == "1d"]
    survivors = eth_daily[eth_daily["sharpe"] >= SHARPE_THRESH_ETH].copy()
    print(f"  ETH daily Bonferroni survivors: {len(survivors)} (threshold={SHARPE_THRESH_ETH:.2f})")

    # Load daily bars for all assets
    panel = load_daily_bars()

    all_replication = []

    for asset_sym in TIER2_ASSETS:
        print(f"\n  {'='*60}")
        print(f"  {asset_sym}")
        print(f"  {'='*60}")

        asset_df = panel[panel["symbol"] == asset_sym].copy()
        asset_df = asset_df.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
        asset_df = asset_df[["open", "high", "low", "close", "volume"]].astype(float)
        print(f"  {len(asset_df)} bars, {asset_df.index.min().date()} to {asset_df.index.max().date()}")

        # B&H for this asset
        asset_ret = asset_df["close"].pct_change(fill_method=None).dropna()
        bh_eq = (1 + asset_ret).cumprod()
        bh_perf = compute_perf(bh_eq, asset_ret, pd.Series(1.0, index=asset_ret.index))
        bh_sharpe = bh_perf["sharpe"] if bh_perf else 0.0
        print(f"  B&H Sharpe: {bh_sharpe:.3f}")

        results = []
        for i, (idx, row) in enumerate(survivors.iterrows()):
            perf = run_strategy_on_asset(row, asset_df)
            if perf is not None:
                results.append(perf)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(survivors)} strategies run ({len(results)} valid)")

        results_df = pd.DataFrame(results)
        n_valid = len(results_df)
        n_positive_sharpe = (results_df["sharpe"] > 0).sum()
        n_beat_bh = (results_df["sharpe"] > bh_sharpe).sum()
        n_beat_bh_dd = (results_df["max_dd"] > bh_perf["max_dd"]).sum() if bh_perf else 0
        med_sharpe = results_df["sharpe"].median()
        med_dd = results_df["max_dd"].median()
        med_skew = results_df["skewness"].median()

        print(f"\n  Results on {asset_sym}:")
        print(f"    Valid strategies: {n_valid}")
        print(f"    Positive Sharpe: {n_positive_sharpe} ({n_positive_sharpe/n_valid:.0%})")
        print(f"    Beat B&H Sharpe: {n_beat_bh} ({n_beat_bh/n_valid:.0%})")
        print(f"    Better DD than B&H: {n_beat_bh_dd} ({n_beat_bh_dd/n_valid:.0%})")
        print(f"    Median Sharpe: {med_sharpe:.3f} (B&H: {bh_sharpe:.3f})")
        print(f"    Median MaxDD: {med_dd:.1%} (B&H: {bh_perf['max_dd']:.1%})")
        print(f"    Median Skewness: {med_skew:.3f}")

        # Family-level breakdown
        fam_agg = results_df.groupby("signal_family").agg(
            n=("sharpe", "count"),
            med_sharpe=("sharpe", "median"),
            pct_beat_bh=("sharpe", lambda x: (x > bh_sharpe).mean()),
            med_dd=("max_dd", "median"),
            med_skew=("skewness", "median"),
        ).sort_values("med_sharpe", ascending=False)

        results_df["asset"] = asset_sym
        results_df.to_csv(OUT / f"replication_{asset_sym.replace('-','_').lower()}.csv", index=False)
        fam_agg.to_csv(OUT / f"replication_family_{asset_sym.replace('-','_').lower()}.csv")

        all_replication.append({
            "asset": asset_sym,
            "n_bars": len(asset_df),
            "years": len(asset_df) / 365,
            "bh_sharpe": round(bh_sharpe, 3),
            "bh_max_dd": round(bh_perf["max_dd"], 3) if bh_perf else None,
            "n_survivors_tested": n_valid,
            "pct_positive_sharpe": round(n_positive_sharpe / n_valid, 3),
            "pct_beat_bh_sharpe": round(n_beat_bh / n_valid, 3),
            "pct_better_dd": round(n_beat_bh_dd / n_valid, 3),
            "med_sharpe": round(med_sharpe, 3),
            "med_max_dd": round(med_dd, 3),
            "med_skewness": round(med_skew, 3),
        })

        print(f"\n  Top 5 families on {asset_sym}:")
        for fam, row in fam_agg.head(5).iterrows():
            print(f"    {fam:<18s} n={row['n']:3.0f} Sharpe={row['med_sharpe']:.3f} "
                  f"Beat B&H={row['pct_beat_bh']:.0%}")

    # Summary table
    summary = pd.DataFrame(all_replication)
    summary.to_csv(OUT / "tier2_replication_summary.csv", index=False)

    print(f"\n  {'='*70}")
    print(f"  TIER 2 REPLICATION SUMMARY")
    print(f"  {'='*70}")
    print(f"  {'Asset':<12s} {'Bars':>5s} {'B&H SR':>7s} {'%Pos':>6s} {'%Beat':>6s} "
          f"{'%DD↑':>6s} {'MedSR':>7s} {'MedDD':>7s}")
    for _, r in summary.iterrows():
        print(f"  {r['asset']:<12s} {r['n_bars']:>5.0f} {r['bh_sharpe']:>7.3f} "
              f"{r['pct_positive_sharpe']:>6.0%} {r['pct_beat_bh_sharpe']:>6.0%} "
              f"{r['pct_better_dd']:>6.0%} {r['med_sharpe']:>7.3f} {r['med_max_dd']:>7.1%}")

    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
