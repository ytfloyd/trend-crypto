"""
Multi-Frequency Momentum Shootout
===================================
Runs the core momentum signal types from the JPM Momentum study at
multiple intraday frequencies: 5m, 30m, 1h, 4h, 8h (plus daily baseline).

Signal types (from Step 4 of the momentum study):
  1. RET  -- Raw trailing return
  2. MAC  -- SMA crossover (fast = lookback/4, slow = lookback)
  3. EMAC -- EMA crossover
  4. BRK  -- Breakout channel position
  5. LREG -- Linear regression t-stat

Each signal is tested in relative (cross-sectional) mode:
  - Top quintile, inverse-volatility weighted, weekly rebalance

Lookback is always 10 bars (the best from the daily study).
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

from scripts.research.common.data import (
    ANN_FACTOR,
    BARS_PER_DAY,
    load_bars,
)
from scripts.research.common.backtest import simple_backtest, DEFAULT_COST_BPS
from scripts.research.common.metrics import compute_metrics, format_metrics_table
from scripts.research.jpm_bigdata_ai.helpers import (
    filter_universe,
    compute_btc_benchmark,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FREQUENCIES = ["8h", "4h", "1h", "30m", "5m"]
LOOKBACK = 10
REBAL_BARS = 5       # rebalance every 5 bars (≈ weekly at daily freq)
WEIGHT_METHOD = "invvol"
VOL_WINDOW = 63
COST_BPS = DEFAULT_COST_BPS

START = "2018-01-01"
END = "2025-12-31"
DATA_START = "2016-06-01"

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "multifreq" / "momentum"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Signal implementations (same as jpm_momentum step_04)
# ---------------------------------------------------------------------------
def signal_ret(group: pd.DataFrame, lookback: int) -> pd.Series:
    close = group["close"]
    return close.shift(1) / close.shift(1 + lookback) - 1.0


def signal_mac(group: pd.DataFrame, lookback: int) -> pd.Series:
    close = group["close"]
    fast = max(2, lookback // 4)
    fast_ma = close.shift(1).rolling(fast, min_periods=fast).mean()
    slow_ma = close.shift(1).rolling(lookback, min_periods=lookback).mean()
    return (fast_ma - slow_ma) / slow_ma


def signal_emac(group: pd.DataFrame, lookback: int) -> pd.Series:
    close = group["close"]
    fast = max(2, lookback // 4)
    fast_ema = close.shift(1).ewm(span=fast, min_periods=fast).mean()
    slow_ema = close.shift(1).ewm(span=lookback, min_periods=lookback).mean()
    return (fast_ema - slow_ema) / slow_ema


def signal_brk(group: pd.DataFrame, lookback: int) -> pd.Series:
    close = group["close"].shift(1)
    high = close.rolling(lookback, min_periods=lookback).max()
    low = close.rolling(lookback, min_periods=lookback).min()
    rng = high - low
    return ((close - low) / rng.replace(0, np.nan)).fillna(0.5)


def signal_lreg(group: pd.DataFrame, lookback: int) -> pd.Series:
    log_close = np.log(group["close"].shift(1))

    def _tstat(window):
        if window.isna().any() or len(window) < lookback:
            return np.nan
        y = window.values
        x = np.arange(len(y))
        slope, _, _, _, std_err = sp_stats.linregress(x, y)
        return slope / std_err if std_err > 1e-10 else 0.0

    return log_close.rolling(lookback, min_periods=lookback).apply(_tstat, raw=False)


SIGNAL_FUNCS = {
    "RET": signal_ret,
    "MAC": signal_mac,
    "EMAC": signal_emac,
    "BRK": signal_brk,
    "LREG": signal_lreg,
}


def _compute_vol(panel: pd.DataFrame, vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    def _per_sym(g):
        g = g.copy()
        ret = np.log(g["close"] / g["close"].shift(1))
        g["realized_vol"] = ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(ANN_FACTOR)
        return g
    return panel.groupby("symbol", group_keys=False).apply(_per_sym)


# ---------------------------------------------------------------------------
# Per-frequency runner
# ---------------------------------------------------------------------------
def run_frequency(freq: str) -> dict:
    t0 = time.time()
    bpd = BARS_PER_DAY[freq]

    # Scale lookback and rebalance to cover the same calendar time
    # 10 daily bars ≈ 10 days. At freq bars, 10 days ≈ 10 × bpd bars.
    lookback_bars = max(3, int(LOOKBACK * bpd))
    rebal_bars = max(1, int(REBAL_BARS * bpd))
    vol_window_bars = max(10, int(VOL_WINDOW * bpd))

    print(f"\n{'='*70}")
    print(f"FREQUENCY: {freq}  (bars/day={bpd:.0f})")
    print(f"  lookback={lookback_bars} bars (~{LOOKBACK}d), rebal={rebal_bars} bars, vol_window={vol_window_bars}")
    print(f"{'='*70}")

    # 1. Load data
    panel = load_bars(freq, start=DATA_START, end=END)
    print(f"  Raw panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    # 2. Universe filter
    adv_window = max(20, int(20 * bpd))
    panel = filter_universe(panel, min_adv_usd=1_000_000,
                            min_history_days=int(90 * bpd),
                            adv_window=adv_window)

    # 3. Filter to eligible symbols only (critical for high-freq perf)
    eligible_syms = panel.loc[panel["in_universe"], "symbol"].unique()
    panel = panel[panel["symbol"].isin(eligible_syms)].copy()
    print(f"  {len(eligible_syms)} eligible symbols, {len(panel):,} rows (filtered)")

    # 4. Vol
    panel = _compute_vol(panel, vol_window=vol_window_bars)

    # 5. Returns for backtest
    panel_bt = panel.loc[panel["ts"] >= START].copy()
    panel_bt["ret_oc"] = panel_bt["close"] / panel_bt["open"] - 1.0
    returns_wide = panel_bt.pivot_table(
        index="ts", columns="symbol", values="ret_oc", aggfunc="first"
    ).fillna(0.0)

    # 6. Run each signal
    # Skip LREG at very high frequencies (rolling regression infeasible)
    active_signals = dict(SIGNAL_FUNCS)
    if lookback_bars > 500:
        print(f"  [NOTE] Skipping LREG at {freq} (lookback={lookback_bars} too large for rolling regression)")
        active_signals.pop("LREG", None)

    all_metrics = []
    equity_curves = {}

    for sig_name, sig_fn in active_signals.items():
        print(f"  {sig_name} (lookback={lookback_bars}) ...", end="", flush=True)

        def _per_sym(g, _fn=sig_fn, _lb=lookback_bars):
            g = g.copy()
            g["signal"] = _fn(g, _lb)
            return g

        p = panel.groupby("symbol", group_keys=False).apply(_per_sym)
        p_active = p.loc[
            (p["ts"] >= START) & p["in_universe"] & p["signal"].notna()
        ].copy()

        all_dates = sorted(p_active["ts"].unique())
        rebal_dates = set(all_dates[::rebal_bars])

        current_weights = {}
        dates_list = []
        weights_list = []

        for dt in all_dates:
            day_data = p_active.loc[p_active["ts"] == dt].copy()
            if day_data.empty:
                dates_list.append(dt)
                weights_list.append({})
                continue

            if dt in rebal_dates:
                ranked = day_data.sort_values("signal", ascending=False)
                n_select = max(1, len(ranked) // 5)
                selected = ranked.head(n_select)

                if len(selected) > 0 and WEIGHT_METHOD == "invvol" and "realized_vol" in selected.columns:
                    vols = selected["realized_vol"].replace(0, np.nan).dropna()
                    if len(vols) > 0:
                        inv_vol = 1.0 / vols.clip(lower=0.10)
                        wts = inv_vol / inv_vol.sum()
                        current_weights = dict(zip(selected.loc[vols.index, "symbol"], wts))
                    else:
                        n_s = len(selected)
                        current_weights = {s: 1.0 / n_s for s in selected["symbol"]}
                elif len(selected) > 0:
                    n_s = len(selected)
                    current_weights = {s: 1.0 / n_s for s in selected["symbol"]}
                else:
                    current_weights = {}

            all_syms = day_data["symbol"].tolist()
            row = {s: current_weights.get(s, 0.0) for s in all_syms}
            dates_list.append(dt)
            weights_list.append(row)

        weights_wide = pd.DataFrame(weights_list, index=pd.DatetimeIndex(dates_list)).fillna(0.0)
        common_cols = weights_wide.columns.intersection(returns_wide.columns)
        w = weights_wide[common_cols]
        r = returns_wide[common_cols]

        result_bt = simple_backtest(w, r, cost_bps=COST_BPS)
        result_bt["ts"] = pd.to_datetime(result_bt["ts"])
        equity = result_bt.set_index("ts")["portfolio_equity"]

        m = compute_metrics(equity)
        m["label"] = f"{sig_name} {lookback_bars}b XS"
        m["signal"] = sig_name
        m["freq"] = freq
        m["lookback_bars"] = lookback_bars
        m["avg_turnover"] = float(result_bt.set_index("ts")["turnover"].mean())
        m["avg_n_holdings"] = float((w > 0).sum(axis=1).mean())
        all_metrics.append(m)
        equity_curves[sig_name] = equity

        print(f"  Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.1%}  MaxDD={m['max_dd']:.1%}")

    # BTC benchmark
    btc_eq = compute_btc_benchmark(panel)
    if equity_curves:
        first_eq = list(equity_curves.values())[0]
        btc_c = btc_eq.reindex(first_eq.index).ffill().bfill()
        if len(btc_c) > 0:
            btc_c = btc_c / btc_c.iloc[0]
            btc_m = compute_metrics(btc_c)
            btc_m["label"] = "BTC Buy & Hold"
            btc_m["signal"] = "BTC"
            btc_m["freq"] = freq

    elapsed = time.time() - t0

    # Save
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(ARTIFACT_DIR / f"momentum_{freq}.csv", index=False, float_format="%.4f")

    print(f"\n  --- {freq} Results ---")
    print(format_metrics_table(all_metrics))
    print(f"  Elapsed: {elapsed:.0f}s")

    return {
        "freq": freq,
        "status": "ok",
        "metrics": all_metrics,
        "elapsed": elapsed,
        "metrics_df": metrics_df,
    }


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("MULTI-FREQUENCY MOMENTUM SHOOTOUT")
    print("=" * 70)
    print(f"Frequencies: {FREQUENCIES}")
    print(f"Signals: {list(SIGNAL_FUNCS.keys())}")
    print(f"Lookback: {LOOKBACK}d equivalent (scaled per frequency)")
    print(f"Mode: relative (top quintile), inv-vol weighted")
    print()

    all_results = []

    for freq in FREQUENCIES + ["1d"]:
        try:
            result = run_frequency(freq)
            all_results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {freq}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"freq": freq, "status": "error", "reason": str(e)})

    # ===================================================================
    # Cross-frequency comparison
    # ===================================================================
    print("\n" + "=" * 70)
    print("CROSS-FREQUENCY MOMENTUM COMPARISON")
    print("=" * 70)

    cross_records = []
    for r in all_results:
        if r.get("status") != "ok":
            continue
        for m in r["metrics"]:
            cross_records.append(m)

    if not cross_records:
        print("No results to compare.")
        return

    cross_df = pd.DataFrame(cross_records)
    cross_df.to_csv(ARTIFACT_DIR / "cross_frequency_momentum.csv", index=False, float_format="%.4f")

    # Best signal per frequency
    print(f"\n  Best signal per frequency:")
    print(f"  {'Freq':<8s} {'Signal':<10s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Elapsed':>10s}")
    print(f"  {'-'*56}")
    for r in all_results:
        if r.get("status") != "ok":
            print(f"  {r['freq']:<8s} --- skipped ({r.get('reason', '')})")
            continue
        best = sorted(r["metrics"], key=lambda x: x.get("sharpe", -99), reverse=True)[0]
        print(f"  {r['freq']:<8s} {best['signal']:<10s} {best['sharpe']:>8.2f} "
              f"{best['cagr']:>7.1%} {best['max_dd']:>7.1%} {r['elapsed']:>9.0f}s")

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n--- Generating plots ---")

    ok_freqs = [r["freq"] for r in all_results if r.get("status") == "ok"]
    sig_colors = {"RET": "#3b82f6", "MAC": "#22c55e", "EMAC": "#10b981",
                  "BRK": "#f59e0b", "LREG": "#8b5cf6"}

    # 1. Sharpe by frequency and signal (grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 7))
    signals = list(SIGNAL_FUNCS.keys())
    x = np.arange(len(ok_freqs))
    width = 0.15
    for i, sig in enumerate(signals):
        sharpes = []
        for freq in ok_freqs:
            sub = cross_df[(cross_df["freq"] == freq) & (cross_df["signal"] == sig)]
            sharpes.append(sub["sharpe"].values[0] if len(sub) > 0 else 0)
        offset = (i - len(signals) / 2 + 0.5) * width
        ax.bar(x + offset, sharpes, width, label=sig,
               color=sig_colors.get(sig, "#9E9E9E"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(ok_freqs, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_xlabel("Bar Frequency", fontsize=11)
    ax.set_title("Momentum Signal Quality Across Frequencies", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "sharpe_by_frequency.png")
    plt.close(fig)
    print("  [1/3] Sharpe by frequency")

    # 2. Sharpe heatmap (signal x frequency)
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = np.full((len(signals), len(ok_freqs)), np.nan)
    for i, sig in enumerate(signals):
        for j, freq in enumerate(ok_freqs):
            sub = cross_df[(cross_df["freq"] == freq) & (cross_df["signal"] == sig)]
            if len(sub) > 0:
                heatmap[i, j] = sub.iloc[0]["sharpe"]

    vabs = max(0.5, np.nanmax(np.abs(heatmap)))
    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(ok_freqs)))
    ax.set_xticklabels(ok_freqs, fontsize=10)
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels(signals, fontsize=10)
    for i in range(len(signals)):
        for j in range(len(ok_freqs)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Sharpe Ratio")
    ax.set_title("Momentum Sharpe: Signal x Frequency", fontsize=13)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "sharpe_heatmap.png")
    plt.close(fig)
    print("  [2/3] Sharpe heatmap")

    # 3. MaxDD heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    dd_map = np.full((len(signals), len(ok_freqs)), np.nan)
    for i, sig in enumerate(signals):
        for j, freq in enumerate(ok_freqs):
            sub = cross_df[(cross_df["freq"] == freq) & (cross_df["signal"] == sig)]
            if len(sub) > 0:
                dd_map[i, j] = sub.iloc[0]["max_dd"] * 100

    im = ax.imshow(dd_map, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ok_freqs)))
    ax.set_xticklabels(ok_freqs, fontsize=10)
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels(signals, fontsize=10)
    for i in range(len(signals)):
        for j in range(len(ok_freqs)):
            val = dd_map[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Max Drawdown (%)")
    ax.set_title("Momentum Max Drawdown: Signal x Frequency", fontsize=13)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "maxdd_heatmap.png")
    plt.close(fig)
    print("  [3/3] MaxDD heatmap")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("MULTI-FREQUENCY MOMENTUM — SUMMARY")
    print("=" * 70)
    for r in all_results:
        if r.get("status") != "ok":
            print(f"  {r['freq']}: {r.get('status', 'unknown')}")
            continue
        best = sorted(r["metrics"], key=lambda x: x.get("sharpe", -99), reverse=True)[0]
        print(f"  {r['freq']}: best={best['signal']} Sharpe={best['sharpe']:.2f} "
              f"CAGR={best['cagr']:.1%} ({r['elapsed']:.0f}s)")

    total_time = sum(r.get("elapsed", 0) for r in all_results if r.get("status") == "ok")
    print(f"\nTotal elapsed: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
