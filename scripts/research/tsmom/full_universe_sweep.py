#!/usr/bin/env python3
"""
Full Universe Trend Expansion — Orchestrator

Tiers the full DuckDB universe by history length and runs:
  Tier A (5+ years): full daily sweep + walk-forward TIM ensemble
  Tier B (3-5 years): ETH Bonferroni survivor replication only
  Tier C (<3 years): skipped (insufficient statistical power)

Usage:
    python -m scripts.research.tsmom.full_universe_sweep
"""
from __future__ import annotations

import ast
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter1d

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
OUT = SWEEP_DIR / "full_universe"
OUT.mkdir(parents=True, exist_ok=True)

TIER_A_MIN_YEARS = 5
TIER_B_MIN_YEARS = 3
N_EFF = 493 * 3
BONF_Z = sp_stats.norm.ppf(1 - 0.05 / N_EFF / 2)
OPT_TIM_LO, OPT_TIM_HI = 0.37, 0.47

ALREADY_SWEPT = {"ETH-USD", "BTC-USD", "SOL-USD"}


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


def run_strategy_on_asset(row, asset_df):
    close = asset_df["close"]
    high = asset_df["high"]
    low = asset_df["low"]
    open_ = asset_df["open"]
    volume = asset_df["volume"]
    daily_returns = close.pct_change(fill_method=None).dropna()

    label = row["label"]
    stop = row["stop"]
    params = ast.literal_eval(row["params"]) if isinstance(row["params"], str) else row["params"]
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
    perf["full_label"] = row.get("full_label", label)
    perf["signal_family"] = row.get("signal_family", label.split("_")[0])
    perf["freq"] = row.get("freq", "1d")
    perf["params"] = str(row["params"])
    return perf


def get_positions(row, asset_df, returns):
    close = asset_df["close"]
    high = asset_df["high"]
    low = asset_df["low"]
    open_ = asset_df["open"]
    volume = asset_df["volume"]
    label = row["label"]
    stop = row["stop"]
    params = ast.literal_eval(row["params"]) if isinstance(row["params"], str) else row["params"]
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
    return sig.shift(1).fillna(0)


def eval_ensemble(positions_list, returns, label):
    if not positions_list:
        return None
    pos_matrix = np.array([p.values for p in positions_list])
    ens_pos = pd.Series(pos_matrix.mean(axis=0), index=returns.index)
    trades = ens_pos.diff().abs()
    cost = trades * (COST_BPS / 10_000)
    net_ret = ens_pos * returns - cost
    equity = (1 + net_ret).cumprod()
    sharpe = float(net_ret.mean() / net_ret.std() * np.sqrt(ANN_FACTOR)) if net_ret.std() > 0 else 0
    cagr = float(equity.iloc[-1] ** (ANN_FACTOR / len(net_ret)) - 1) if len(net_ret) > 0 else 0
    maxdd = float((equity / equity.cummax() - 1).min())
    skewness = float(net_ret.skew())
    mean_pos = float(ens_pos.mean())
    return {"strategy": label, "sharpe": round(sharpe, 3), "cagr": round(cagr, 3),
            "max_dd": round(maxdd, 3), "skewness": round(skewness, 3),
            "mean_pos": round(mean_pos, 3), "n_strategies": len(positions_list)}


def compute_bh(asset_df):
    close = asset_df["close"]
    ret = close.pct_change(fill_method=None).dropna()
    eq = (1 + ret).cumprod()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(ANN_FACTOR)) if ret.std() > 0 else 0
    cagr = float(eq.iloc[-1] ** (ANN_FACTOR / len(ret)) - 1) if len(ret) > 0 else 0
    maxdd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr, 3), "max_dd": round(maxdd, 3)}


def compute_tim_optimum(df):
    """Find TIM optimum from a sweep results DataFrame (daily strategies)."""
    daily = df[(df["freq"] == "1d") & (df["label"] != "BUY_AND_HOLD")] if "freq" in df.columns else df
    tim = daily["time_in_market"].values
    sharpe = daily["sharpe"].values
    bins = np.arange(0, 1.01, 0.02)
    bc, bm = [], []
    for i in range(len(bins) - 1):
        mask = (tim >= bins[i]) & (tim < bins[i + 1])
        if mask.sum() >= 5:
            bc.append((bins[i] + bins[i + 1]) / 2)
            bm.append(np.median(sharpe[mask]))
    if len(bc) < 3:
        return None, None
    bc, bm = np.array(bc), np.array(bm)
    smooth = uniform_filter1d(bm, size=3)
    opt_idx = np.argmax(smooth)
    return round(float(bc[opt_idx]), 2), round(float(smooth[opt_idx]), 3)


def run_walkforward(results_df, asset_df, symbol):
    """Run walk-forward TIM ensemble on an asset. Split at midpoint of history."""
    daily = results_df[(results_df["freq"] == "1d") & (results_df["label"] != "BUY_AND_HOLD")]
    close = asset_df["close"]
    returns = close.pct_change(fill_method=None).dropna()
    mid = returns.index[len(returns) // 2]

    train_ret = returns[returns.index <= mid]
    test_ret = returns[returns.index > mid]
    train_df = asset_df.loc[asset_df.index <= mid]

    if len(train_ret) < 365 or len(test_ret) < 180:
        return None

    # Compute training-period TIM
    train_tims = {}
    for idx_row, row in daily.iterrows():
        pos = get_positions(row, train_df, train_ret)
        if pos is not None:
            p_train = pos.reindex(train_ret.index).fillna(0)
            train_tims[idx_row] = float((p_train.abs() > 1e-6).mean())

    if not train_tims:
        return None

    train_tim_s = pd.Series(train_tims)
    selected = daily.loc[train_tim_s.index].copy()
    selected["train_tim"] = train_tim_s.values
    wf_sel = selected[(selected["train_tim"] >= OPT_TIM_LO) & (selected["train_tim"] <= OPT_TIM_HI)]

    if len(wf_sel) < 5:
        return None

    # Build ensemble
    wf_positions = []
    for _, row in wf_sel.iterrows():
        pos = get_positions(row, asset_df, returns)
        if pos is not None:
            wf_positions.append(pos)

    if len(wf_positions) < 5:
        return None

    full_ens = eval_ensemble(wf_positions, returns, f"{symbol} WF Ensemble")
    test_pos = [p.reindex(test_ret.index).fillna(0) for p in wf_positions]
    test_ens = eval_ensemble(test_pos, test_ret, f"{symbol} WF OOS")

    tim_corr = float(selected[["train_tim", "time_in_market"]].corr().iloc[0, 1]) if "time_in_market" in selected.columns else None

    return {
        "symbol": symbol,
        "n_selected": len(wf_sel),
        "n_ensemble": len(wf_positions),
        "tim_corr": round(tim_corr, 3) if tim_corr else None,
        "full_sharpe": full_ens["sharpe"] if full_ens else None,
        "full_maxdd": full_ens["max_dd"] if full_ens else None,
        "oos_sharpe": test_ens["sharpe"] if test_ens else None,
        "oos_maxdd": test_ens["max_dd"] if test_ens else None,
        "split_date": str(mid.date()),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    t_global = time.time()
    print("=" * 70)
    print("  FULL UNIVERSE TREND EXPANSION")
    print("=" * 70)

    panel = load_daily_bars()

    # Compute per-symbol stats
    sym_stats = panel.groupby("symbol").agg(
        n_bars=("ts", "count"),
        first_date=("ts", "min"),
        last_date=("ts", "max"),
    ).sort_values("n_bars", ascending=False)
    sym_stats["years"] = sym_stats["n_bars"] / 365.0

    tier_a = sym_stats[sym_stats["years"] >= TIER_A_MIN_YEARS].index.tolist()
    tier_b = sym_stats[(sym_stats["years"] >= TIER_B_MIN_YEARS) & (sym_stats["years"] < TIER_A_MIN_YEARS)].index.tolist()
    tier_c = sym_stats[sym_stats["years"] < TIER_B_MIN_YEARS].index.tolist()

    print(f"\n  Universe: {len(sym_stats)} symbols")
    print(f"  Tier A (5+ yrs, full sweep):   {len(tier_a)}")
    print(f"  Tier B (3-5 yrs, replication): {len(tier_b)}")
    print(f"  Tier C (<3 yrs, skip):         {len(tier_c)}")

    # Save tiering
    tier_table = []
    for sym in tier_a:
        tier_table.append({"symbol": sym, "tier": "A", "years": round(sym_stats.loc[sym, "years"], 1),
                           "n_bars": int(sym_stats.loc[sym, "n_bars"]),
                           "first_date": str(sym_stats.loc[sym, "first_date"].date())})
    for sym in tier_b:
        tier_table.append({"symbol": sym, "tier": "B", "years": round(sym_stats.loc[sym, "years"], 1),
                           "n_bars": int(sym_stats.loc[sym, "n_bars"]),
                           "first_date": str(sym_stats.loc[sym, "first_date"].date())})
    for sym in tier_c:
        tier_table.append({"symbol": sym, "tier": "C", "years": round(sym_stats.loc[sym, "years"], 1),
                           "n_bars": int(sym_stats.loc[sym, "n_bars"]),
                           "first_date": str(sym_stats.loc[sym, "first_date"].date())})
    pd.DataFrame(tier_table).to_csv(OUT / "universe_tiers.csv", index=False)

    # ==================================================================
    # TIER A: Full daily sweeps
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  TIER A: FULL DAILY SWEEPS ({len(tier_a)} symbols)")
    print(f"{'='*70}")

    tier_a_new = [s for s in tier_a if s not in ALREADY_SWEPT]
    tier_a_done = [s for s in tier_a if s in ALREADY_SWEPT]
    print(f"  Already swept: {tier_a_done}")
    print(f"  New sweeps: {len(tier_a_new)}")

    for i, sym in enumerate(tier_a_new):
        t0 = time.time()
        print(f"\n  [{i+1}/{len(tier_a_new)}] Running sweep on {sym}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.research.tsmom.run_eth_trend_sweep_v2",
                 "--symbol", sym, "--daily-only"],
                capture_output=True, text=True, cwd=str(ROOT), timeout=180,
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines[-5:]:
                    print(f"    {line.strip()}")
                print(f"    Done in {elapsed:.0f}s")
            else:
                print(f"    FAILED (rc={result.returncode}): {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after 180s, skipping")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Collect all Tier A results
    tier_a_summary = []
    tier_a_walkforward = []

    for sym in tier_a:
        slug = sym.replace("-", "").lower()
        # Try both naming conventions
        results_path = SWEEP_DIR / f"{slug}_trend_sweep" / "results_v2.csv"
        if not results_path.exists() and sym == "ETH-USD":
            results_path = SWEEP_DIR / "eth_trend_sweep" / "results_v2.csv"
        if not results_path.exists():
            print(f"  [WARN] No results for {sym} at {results_path}")
            continue

        df = pd.read_csv(results_path)
        strats = df[df["label"] != "BUY_AND_HOLD"]
        n_years = sym_stats.loc[sym, "years"]
        bonf_thresh = BONF_Z / np.sqrt(n_years)
        daily = strats[strats["freq"] == "1d"] if "freq" in strats.columns else strats
        n_survivors = (daily["sharpe"] >= bonf_thresh).sum()
        n_daily = len(daily)

        # B&H
        asset_df = panel[panel["symbol"] == sym].copy()
        asset_df = asset_df.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
        asset_df = asset_df[["open", "high", "low", "close", "volume"]].astype(float)
        bh = compute_bh(asset_df)

        # TIM optimum
        tim_opt, tim_sharpe = compute_tim_optimum(daily)

        # Family survival
        if n_survivors > 0:
            surv_df = daily[daily["sharpe"] >= bonf_thresh]
            top_fam = surv_df["signal_family"].value_counts()
            top3 = "/".join(top_fam.head(3).index.tolist())
        else:
            top3 = "—"

        tier_a_summary.append({
            "symbol": sym, "years": round(n_years, 1),
            "n_bars": int(sym_stats.loc[sym, "n_bars"]),
            "bh_sharpe": bh["sharpe"], "bh_maxdd": bh["max_dd"],
            "n_daily_strats": n_daily, "bonf_threshold": round(bonf_thresh, 2),
            "n_survivors": n_survivors,
            "pct_survivors": round(n_survivors / n_daily, 3) if n_daily > 0 else 0,
            "med_sharpe": round(daily["sharpe"].median(), 3),
            "tim_optimum": tim_opt, "tim_peak_sharpe": tim_sharpe,
            "top_families": top3,
        })
        print(f"  {sym:<12s} {n_years:.1f}yr  B&H={bh['sharpe']:.2f}  "
              f"Survivors={n_survivors}/{n_daily}  TIM_opt={tim_opt}  Fams={top3}")

        # Walk-forward (only for assets with enough history for a meaningful split)
        if n_years >= 4:
            print(f"    Running walk-forward for {sym}...")
            wf = run_walkforward(df, asset_df, sym)
            if wf:
                tier_a_walkforward.append(wf)
                print(f"    WF: n={wf['n_ensemble']}, full SR={wf['full_sharpe']}, "
                      f"OOS SR={wf['oos_sharpe']}, TIM ρ={wf['tim_corr']}")
            else:
                print(f"    WF: insufficient data or selections")

    pd.DataFrame(tier_a_summary).to_csv(OUT / "tier_a_summary.csv", index=False)
    pd.DataFrame(tier_a_walkforward).to_csv(OUT / "tier_a_walkforward.csv", index=False)

    # ==================================================================
    # TIER B: ETH survivor replication
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  TIER B: ETH SURVIVOR REPLICATION ({len(tier_b)} symbols)")
    print(f"{'='*70}")

    eth_results = pd.read_csv(SWEEP_DIR / "eth_trend_sweep" / "results_v2.csv")
    eth_strats = eth_results[eth_results["label"] != "BUY_AND_HOLD"]
    eth_daily = eth_strats[eth_strats["freq"] == "1d"]
    eth_bonf = BONF_Z / np.sqrt(sym_stats.loc["ETH-USD", "years"])
    survivors = eth_daily[eth_daily["sharpe"] >= eth_bonf].copy()
    print(f"  ETH daily Bonferroni survivors: {len(survivors)} (threshold={eth_bonf:.2f})")

    # Include Tier A symbols that haven't been replicated yet (for completeness)
    tier_b_all = tier_b.copy()
    tier_b_summary = []

    for i, sym in enumerate(tier_b_all):
        asset_df = panel[panel["symbol"] == sym].copy()
        asset_df = asset_df.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
        asset_df = asset_df[["open", "high", "low", "close", "volume"]].astype(float)

        if len(asset_df) < 365:
            continue

        bh = compute_bh(asset_df)
        bh_sharpe = bh["sharpe"]

        results = []
        for _, row in survivors.iterrows():
            perf = run_strategy_on_asset(row, asset_df)
            if perf is not None:
                results.append(perf)

        if not results:
            continue

        rdf = pd.DataFrame(results)
        n_valid = len(rdf)
        n_pos = (rdf["sharpe"] > 0).sum()
        n_beat = (rdf["sharpe"] > bh_sharpe).sum()
        n_better_dd = (rdf["max_dd"] > bh["max_dd"]).sum()

        tier_b_summary.append({
            "symbol": sym,
            "years": round(sym_stats.loc[sym, "years"], 1),
            "n_bars": int(sym_stats.loc[sym, "n_bars"]),
            "bh_sharpe": bh_sharpe,
            "bh_maxdd": bh["max_dd"],
            "n_tested": n_valid,
            "pct_positive_sharpe": round(n_pos / n_valid, 3),
            "pct_beat_bh": round(n_beat / n_valid, 3),
            "pct_better_dd": round(n_better_dd / n_valid, 3),
            "med_sharpe": round(rdf["sharpe"].median(), 3),
            "med_maxdd": round(rdf["max_dd"].median(), 3),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(tier_b_all):
            print(f"  [{i+1}/{len(tier_b_all)}] {sym}: {n_pos}/{n_valid} positive SR, "
                  f"{n_beat}/{n_valid} beat B&H ({bh_sharpe:.2f})")

    pd.DataFrame(tier_b_summary).to_csv(OUT / "tier_b_replication.csv", index=False)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed_total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  FULL UNIVERSE SWEEP COMPLETE ({elapsed_total:.0f}s)")
    print(f"{'='*70}")

    ta = pd.DataFrame(tier_a_summary)
    tb = pd.DataFrame(tier_b_summary)

    print(f"\n  TIER A ({len(ta)} assets):")
    if len(ta) > 0:
        print(f"    Median B&H Sharpe:     {ta['bh_sharpe'].median():.2f}")
        print(f"    Median survivors:      {ta['n_survivors'].median():.0f}")
        tim_vals = ta["tim_optimum"].dropna()
        if len(tim_vals) > 0:
            print(f"    TIM optimum range:     [{tim_vals.min():.0%}, {tim_vals.max():.0%}]")
            print(f"    TIM optimum median:    {tim_vals.median():.0%}")

    print(f"\n  TIER B ({len(tb)} assets):")
    if len(tb) > 0:
        print(f"    Median % positive SR:  {tb['pct_positive_sharpe'].median():.0%}")
        print(f"    Median % beat B&H:     {tb['pct_beat_bh'].median():.0%}")
        print(f"    Median % better DD:    {tb['pct_better_dd'].median():.0%}")

    if len(tier_a_walkforward) > 0:
        wf_df = pd.DataFrame(tier_a_walkforward)
        print(f"\n  WALK-FORWARD ({len(wf_df)} assets):")
        print(f"    Median TIM ρ:          {wf_df['tim_corr'].median():.3f}")
        print(f"    Median full Sharpe:    {wf_df['full_sharpe'].median():.2f}")
        oos = wf_df["oos_sharpe"].dropna()
        if len(oos) > 0:
            print(f"    Median OOS Sharpe:     {oos.median():.2f}")

    print(f"\n  Outputs: {OUT}")
    print("  Done.")


if __name__ == "__main__":
    main()
