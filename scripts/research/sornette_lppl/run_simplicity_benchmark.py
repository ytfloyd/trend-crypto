#!/usr/bin/env python3
"""
Simplicity Benchmark — isolate LPPLS value-add.

Identical plumbing to run_hf_backtest.py:
  - Same universe (top-50 by ADV, filtered to 35 with history)
  - Same BTC dual-SMA regime filter
  - Same position sizing (inverse-vol, top-K=10)
  - Same costs (30 bps one-way)
  - Same rebalance cadence (6h)
  - Same max-hold (168h)

Only the entry/exit logic changes:
  LPPLS:      Super-exponential detection → tc-based predictive exit
  Benchmark:  Donchian channel breakout   → ATR trailing stop reactive exit

If Sharpe ≈ LPPLS Sharpe → LPPLS is over-engineered beta.
If Sharpe << LPPLS Sharpe → LPPLS provides genuine timing alpha.

Usage:
    python -m scripts.research.sornette_lppl.run_simplicity_benchmark
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_hf import load_hourly_bars, filter_universe_hourly, ANN_FACTOR_HOURLY
from .portfolio_hf import backtest_hourly, hourly_performance_summary
from .regime import compute_regime, regime_mask as make_regime_mask


OUT_DIR = Path(__file__).resolve().parent / "output"


# ===================================================================
# Donchian Entry Signal
# ===================================================================

def compute_donchian_signals(
    panel: pd.DataFrame,
    breakout_window: int = 24,
    eval_every: int = 6,
) -> pd.DataFrame:
    """Flag symbols making new N-hour highs.

    Returns a DataFrame with columns [symbol, ts, signal_strength]
    where signal_strength = (close - lower) / (upper - lower).
    """
    results = []
    symbols = sorted(panel["symbol"].unique())
    panel = panel.sort_values(["symbol", "ts"])

    for sym in symbols:
        sdf = panel[panel["symbol"] == sym].copy()
        if len(sdf) < breakout_window + 10:
            continue

        sdf = sdf.set_index("ts").sort_index()
        highs = sdf["high"].rolling(breakout_window, min_periods=breakout_window).max()
        lows = sdf["low"].rolling(breakout_window, min_periods=breakout_window).min()
        close = sdf["close"]

        breakout = close >= highs
        rng = highs - lows
        strength = ((close - lows) / rng).clip(0, 1).fillna(0)

        eval_mask = np.zeros(len(sdf), dtype=bool)
        eval_mask[breakout_window::eval_every] = True

        triggered = breakout & pd.Series(eval_mask, index=sdf.index)
        triggered = triggered.fillna(False)

        for ts in sdf.index[triggered]:
            results.append({
                "symbol": sym,
                "ts": ts,
                "signal_strength": float(strength.loc[ts]),
            })

    return pd.DataFrame(results)


# ===================================================================
# ATR Computation
# ===================================================================

def compute_atr(panel: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ATR for each symbol, returns wide-format (ts × symbol)."""
    panel = panel.sort_values(["symbol", "ts"])
    atrs = {}

    for sym in panel["symbol"].unique():
        sdf = panel[panel["symbol"] == sym].set_index("ts").sort_index()
        high = sdf["high"]
        low = sdf["low"]
        prev_close = sdf["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atrs[sym] = tr.rolling(period, min_periods=period).mean()

    atr_wide = pd.DataFrame(atrs)
    return atr_wide


# ===================================================================
# Portfolio Construction — ATR trailing stop + time exit
# ===================================================================

def build_benchmark_portfolio(
    signals: pd.DataFrame,
    returns_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    high_wide: pd.DataFrame,
    atr_wide: pd.DataFrame,
    *,
    top_k: int = 10,
    rebalance_every_hours: int = 6,
    min_signal: float = 0.5,
    atr_multiplier: float = 3.0,
    max_hold_hours: int = 168,
    ivol_weight: bool = True,
    vol_lookback: int = 48,
    regime_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build benchmark portfolio with Donchian entry + ATR trailing stop.

    Entry:  Donchian breakout (new 24h high) with signal_strength > min_signal
    Exit 1: Close < highest_high_since_entry - atr_multiplier × ATR(14)  [vol stop]
    Exit 2: hours_held >= max_hold_hours                                  [time decay]
    Exit 3: Regime flips to bear                                          [hard stop]
    """
    all_hours = returns_wide.index.sort_values()
    symbols = returns_wide.columns

    if not signals.empty:
        sig_wide = signals.pivot_table(
            index="ts", columns="symbol", values="signal_strength", aggfunc="last"
        ).reindex(all_hours).ffill(limit=rebalance_every_hours)
    else:
        sig_wide = pd.DataFrame(0.0, index=all_hours, columns=symbols)

    vol = returns_wide.rolling(vol_lookback, min_periods=vol_lookback // 2).std()

    weights = pd.DataFrame(0.0, index=all_hours, columns=symbols)
    holdings: dict[str, dict] = {}
    trades = []
    hour_counter = 0

    for dt in all_hours:
        hour_counter += 1

        # Regime hard stop
        if regime_mask is not None and dt in regime_mask.index and not regime_mask.loc[dt]:
            for sym in list(holdings.keys()):
                h = holdings[sym]
                trades.append({
                    "ts": dt, "symbol": sym, "action": "exit_regime",
                    "hours_held": h.get("hours_held", 0),
                    "cum_ret": h.get("cum_ret", 0),
                })
            holdings.clear()
            continue

        r = returns_wide.loc[dt].fillna(0.0)

        # Update tracking for existing holdings
        for sym in list(holdings.keys()):
            h = holdings[sym]
            h["hours_held"] = h.get("hours_held", 0) + 1
            h["cum_ret"] = (1 + h.get("cum_ret", 0)) * (1 + r.get(sym, 0)) - 1

            # Track highest high since entry for trailing stop
            if dt in high_wide.index and sym in high_wide.columns:
                current_high = high_wide.loc[dt].get(sym, np.nan)
                if not np.isnan(current_high):
                    h["peak_price"] = max(h.get("peak_price", 0), current_high)

            # EXIT 1: ATR trailing stop (the reactive vol-stop)
            if dt in close_wide.index and sym in close_wide.columns:
                current_close = close_wide.loc[dt].get(sym, np.nan)
                current_atr = atr_wide.loc[dt].get(sym, np.nan) if dt in atr_wide.index else np.nan

                if (not np.isnan(current_close)
                        and not np.isnan(current_atr)
                        and h.get("peak_price", 0) > 0
                        and current_atr > 0):
                    stop_level = h["peak_price"] - atr_multiplier * current_atr
                    if current_close < stop_level:
                        trades.append({
                            "ts": dt, "symbol": sym, "action": "exit_atr_stop",
                            "hours_held": h["hours_held"],
                            "cum_ret": h["cum_ret"],
                        })
                        del holdings[sym]
                        continue

            # EXIT 2: Time decay (max hold)
            if h["hours_held"] >= max_hold_hours:
                trades.append({
                    "ts": dt, "symbol": sym, "action": "exit_maxhold",
                    "hours_held": h["hours_held"],
                    "cum_ret": h["cum_ret"],
                })
                del holdings[sym]
                continue

        # ENTRY: every rebalance_every_hours, scan for Donchian breakouts
        if hour_counter % rebalance_every_hours == 0:
            if dt in sig_wide.index:
                s = sig_wide.loc[dt].dropna()
                s = s[s >= min_signal]
                s = s.drop(labels=[sym for sym in holdings if sym in s.index], errors="ignore")

                n_open = top_k - len(holdings)
                if n_open > 0 and len(s) > 0:
                    new_entries = s.sort_values(ascending=False).head(n_open)
                    for sym in new_entries.index:
                        entry_price = (
                            close_wide.loc[dt].get(sym, np.nan)
                            if dt in close_wide.index
                            else np.nan
                        )
                        holdings[sym] = {
                            "entry_hour": dt,
                            "hours_held": 0,
                            "cum_ret": 0.0,
                            "peak_price": entry_price if not np.isnan(entry_price) else 0.0,
                        }
                        trades.append({
                            "ts": dt, "symbol": sym, "action": "entry",
                            "signal": float(new_entries[sym]),
                        })

        # Compute weights
        if holdings:
            held = list(holdings.keys())
            if ivol_weight and dt in vol.index:
                v = vol.loc[dt].reindex(held).fillna(vol.loc[dt].median()).clip(lower=1e-6)
                raw_w = 1.0 / v
            else:
                raw_w = pd.Series(1.0, index=held)
            raw_w = raw_w / raw_w.sum()
            for sym in held:
                if sym in raw_w.index:
                    weights.loc[dt, sym] = raw_w[sym]

    trades_df = pd.DataFrame(trades)
    return weights, trades_df


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Simplicity Benchmark")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--rebal-every", type=int, default=6)
    parser.add_argument("--tc-bps", type=float, default=30.0)
    parser.add_argument("--atr-mult", type=float, default=3.0)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--donchian-window", type=int, default=24)
    parser.add_argument("--max-hold", type=int, default=168)
    parser.add_argument("--no-regime", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("SIMPLICITY BENCHMARK — DONCHIAN + ATR TRAILING STOP")
    print("=" * 70)
    print(f"  Donchian window:  {args.donchian_window}h")
    print(f"  ATR period:       {args.atr_period}h")
    print(f"  ATR multiplier:   {args.atr_mult}×")
    print(f"  Max hold:         {args.max_hold}h")
    print(f"  TC:               {args.tc_bps} bps")
    print(f"  Top-K:            {args.top_k}")

    # 1. Load SAME data as LPPLS backtest
    panel = load_hourly_bars(
        start=args.start, end=args.end,
        min_adv_usd=5_000_000, max_symbols=50,
    )
    if panel.empty:
        print("No data. Aborting.")
        return
    panel = filter_universe_hourly(panel)
    panel = panel[panel["in_universe"]].copy()
    symbols = sorted(panel["symbol"].unique())
    print(f"\nUniverse: {len(symbols)} symbols (same as LPPLS)")
    print(f"Range: {panel['ts'].min()} → {panel['ts'].max()}")

    # 2. Compute returns
    df = panel.sort_values(["symbol", "ts"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)

    # Wide-format price data for trailing stop logic
    close_wide = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    high_wide = df.pivot(index="ts", columns="symbol", values="high").sort_index()

    # 3. Compute ATR
    print(f"\nComputing ATR({args.atr_period}) ...")
    atr_wide = compute_atr(panel, period=args.atr_period)
    atr_wide = atr_wide.reindex(returns_wide.index)

    # 4. Regime filter (IDENTICAL to LPPLS)
    reg_mask = None
    if not args.no_regime and "BTC-USD" in panel["symbol"].values:
        btc = panel[panel["symbol"] == "BTC-USD"].sort_values("ts")
        btc_daily = btc.set_index("ts")["close"].resample("D").last().dropna()
        regime_df = compute_regime(btc_daily)
        regime_daily = make_regime_mask(regime_df, btc_daily.index)
        reg_mask = regime_daily.reindex(returns_wide.index, method="ffill")
        bull_pct = reg_mask.mean() * 100 if reg_mask is not None else 0
        print(f"[regime] BTC dual-SMA: {bull_pct:.0f}% of hours risk-on/bull")

    # 5. Compute Donchian breakout signals
    print(f"\nScanning Donchian({args.donchian_window}h) breakouts ...")
    t0 = time.time()
    signals = compute_donchian_signals(
        panel,
        breakout_window=args.donchian_window,
        eval_every=args.eval_every,
    )
    elapsed = time.time() - t0
    print(f"  {len(signals):,} breakout signals in {elapsed:.1f}s")
    if not signals.empty:
        print(f"  Unique symbols: {signals['symbol'].nunique()}")
        print(f"  Date range: {signals['ts'].min()} → {signals['ts'].max()}")

    # 6. Build benchmark portfolio
    print(f"\n[portfolio] top-{args.top_k}, rebal={args.rebal_every}h, "
          f"ATR stop={args.atr_mult}×, max-hold={args.max_hold}h")

    weights, trades = build_benchmark_portfolio(
        signals, returns_wide, close_wide, high_wide, atr_wide,
        top_k=args.top_k,
        rebalance_every_hours=args.rebal_every,
        atr_multiplier=args.atr_mult,
        max_hold_hours=args.max_hold,
        regime_mask=reg_mask,
    )

    # 7. Backtest (IDENTICAL engine to LPPLS)
    bt = backtest_hourly(weights, returns_wide, tc_bps=args.tc_bps)
    stats = hourly_performance_summary(bt)

    print(f"\n{'='*70}")
    print("SIMPLICITY BENCHMARK — PERFORMANCE")
    print(f"{'='*70}")
    for k, v in stats.items():
        if isinstance(v, float):
            if any(x in k for x in ["vol", "cagr", "return", "dd", "turnover"]):
                print(f"  {k:<25s} {v:.1%}")
            else:
                print(f"  {k:<25s} {v:.2f}")
        else:
            print(f"  {k:<25s} {v}")

    # Trade analysis
    if not trades.empty:
        print(f"\n{'='*70}")
        print("TRADE ANALYSIS")
        print(f"{'='*70}")
        action_counts = trades["action"].value_counts()
        for action, cnt in action_counts.items():
            print(f"  {action:<20s} {cnt:>5d}")

        exits = trades[trades["action"].str.startswith("exit")]
        if "cum_ret" in exits.columns and len(exits) > 0:
            exits_pnl = exits[exits["cum_ret"].notna()]
            if len(exits_pnl) > 0:
                hit_rate = (exits_pnl["cum_ret"] > 0).mean()
                avg_ret = exits_pnl["cum_ret"].mean()
                print(f"\n  Hit rate:            {hit_rate:.1%}")
                print(f"  Avg return/trade:    {avg_ret:.1%}")

        # ATR stop analysis
        atr_exits = trades[trades["action"] == "exit_atr_stop"]
        if len(atr_exits) > 0 and "cum_ret" in atr_exits.columns:
            atr_pnl = atr_exits[atr_exits["cum_ret"].notna()]
            if len(atr_pnl) > 0:
                print(f"\n  ATR stop exits:      {len(atr_exits)}")
                print(f"  ATR stop hit rate:   {(atr_pnl['cum_ret'] > 0).mean():.1%}")
                print(f"  ATR stop avg return: {atr_pnl['cum_ret'].mean():.1%}")

    # 8. Load LPPLS results for comparison
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD: LPPLS vs SIMPLICITY BENCHMARK")
    print(f"{'='*70}")

    lppls_bt = pd.read_parquet(OUT_DIR / "hf_backtest.parquet")
    lppls_bt["ts"] = pd.to_datetime(lppls_bt["ts"])
    lppls_stats = hourly_performance_summary(lppls_bt)

    # Compute arithmetic Sharpe for both
    ann = 365.0 * 24
    bench_sharpe_arith = (
        bt["net_ret"].mean() / bt["net_ret"].std() * np.sqrt(ann)
        if bt["net_ret"].std() > 1e-12 else 0
    )
    lppls_sharpe_arith = (
        lppls_bt["net_ret"].mean() / lppls_bt["net_ret"].std() * np.sqrt(ann)
        if lppls_bt["net_ret"].std() > 1e-12 else 0
    )

    comparison = {
        "CAGR":           (stats.get("cagr", 0),        lppls_stats.get("cagr", 0)),
        "Sharpe (arith)": (bench_sharpe_arith,           lppls_sharpe_arith),
        "Sharpe (CAGR/V)":(stats.get("sharpe", 0),      lppls_stats.get("sharpe", 0)),
        "Max DD":         (stats.get("max_dd", 0),       lppls_stats.get("max_dd", 0)),
        "Calmar":         (stats.get("calmar", 0),       lppls_stats.get("calmar", 0)),
        "Total Return":   (stats.get("total_return", 0), lppls_stats.get("total_return", 0)),
        "Avg Holdings":   (stats.get("avg_holdings", 0), lppls_stats.get("avg_holdings", 0)),
    }

    print(f"  {'Metric':<20s} {'Benchmark':>12s} {'LPPLS':>12s} {'Delta':>12s}")
    print(f"  {'-'*56}")
    for metric, (bench_v, lppls_v) in comparison.items():
        delta = lppls_v - bench_v
        if "sharpe" in metric.lower() or "calmar" in metric.lower():
            print(f"  {metric:<20s} {bench_v:>12.2f} {lppls_v:>12.2f} {delta:>+12.2f}")
        elif "holdings" in metric.lower():
            print(f"  {metric:<20s} {bench_v:>12.1f} {lppls_v:>12.1f} {delta:>+12.1f}")
        else:
            print(f"  {metric:<20s} {bench_v:>11.1%} {lppls_v:>11.1%} {delta:>+11.1%}")

    sharpe_pct_diff = (
        (lppls_sharpe_arith - bench_sharpe_arith) / bench_sharpe_arith * 100
        if bench_sharpe_arith != 0 else float("inf")
    )

    print(f"\n  Sharpe delta: {lppls_sharpe_arith - bench_sharpe_arith:+.2f} "
          f"({sharpe_pct_diff:+.1f}%)")

    if abs(sharpe_pct_diff) < 15:
        print("  VERDICT: Within 15% → LPPLS is likely Over-Engineered Beta.")
    elif sharpe_pct_diff > 15:
        print("  VERDICT: LPPLS outperforms by >15% → Possible Timing Alpha.")
    else:
        print("  VERDICT: Benchmark outperforms LPPLS → LPPLS is value-destructive.")

    # 9. Save artifacts
    bt.to_parquet(OUT_DIR / "benchmark_backtest.parquet", index=False)
    trades.to_parquet(OUT_DIR / "benchmark_trades.parquet", index=False)
    weights.to_parquet(OUT_DIR / "benchmark_weights.parquet")

    comparison_json = {
        "benchmark": {
            "cagr": stats.get("cagr", 0),
            "sharpe_arithmetic": float(bench_sharpe_arith),
            "sharpe_geometric": stats.get("sharpe", 0),
            "max_dd": stats.get("max_dd", 0),
            "calmar": stats.get("calmar", 0),
            "total_return": stats.get("total_return", 0),
            "avg_holdings": stats.get("avg_holdings", 0),
            "n_trades": int(len(trades[trades["action"] == "entry"])),
        },
        "lppls": {
            "cagr": lppls_stats.get("cagr", 0),
            "sharpe_arithmetic": float(lppls_sharpe_arith),
            "sharpe_geometric": lppls_stats.get("sharpe", 0),
            "max_dd": lppls_stats.get("max_dd", 0),
            "calmar": lppls_stats.get("calmar", 0),
            "total_return": lppls_stats.get("total_return", 0),
            "avg_holdings": lppls_stats.get("avg_holdings", 0),
        },
        "delta_sharpe_arithmetic": float(lppls_sharpe_arith - bench_sharpe_arith),
        "delta_sharpe_pct": float(sharpe_pct_diff),
        "verdict": (
            "Within 15%: LPPLS is Over-Engineered Beta"
            if abs(sharpe_pct_diff) < 15
            else (
                "LPPLS outperforms >15%: Possible Timing Alpha"
                if sharpe_pct_diff > 15
                else "Benchmark outperforms: LPPLS is value-destructive"
            )
        ),
        "parameters": {
            "donchian_window": args.donchian_window,
            "atr_period": args.atr_period,
            "atr_multiplier": args.atr_mult,
            "max_hold": args.max_hold,
            "top_k": args.top_k,
            "tc_bps": args.tc_bps,
            "rebal_every": args.rebal_every,
        },
    }

    with open(OUT_DIR / "benchmark_comparison.json", "w") as f:
        json.dump(comparison_json, f, indent=2, default=str)

    print(f"\n  Saved to {OUT_DIR}/benchmark_*.parquet")
    print(f"  Comparison: {OUT_DIR}/benchmark_comparison.json")

    return bt, stats, trades, comparison_json


if __name__ == "__main__":
    main()
