#!/usr/bin/env python3
"""
Medallion Lite — cross-sectional factor model + ensemble regime, event-driven.

Renaissance-inspired architecture adapted for crypto cost structure:
  - Factor model selects tokens (momentum, volume, vol, proximity, Sharpe)
  - Ensemble regime times entries and scales exposure
  - Event-driven holding (enter/hold/exit) instead of continuous rebalance
  - Inverse-vol sizing with position caps

This is the LPPLS/Donchian framework with a better signal engine —
multi-factor cross-sectional selection instead of single-indicator entry,
and continuous regime instead of binary gate.

Usage:
    python -m scripts.research.medallion_lite.run_backtest
    python -m scripts.research.medallion_lite.run_backtest --max-positions 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SORNETTE_DIR = Path(__file__).resolve().parent.parent / "sornette_lppl"
sys.path.insert(0, str(SORNETTE_DIR.parent))
from sornette_lppl.data_hf import load_hourly_bars, filter_universe_hourly

from .regime_ensemble import compute_ensemble_regime
from .factors import compute_factors, compute_composite_score
from .portfolio import (
    build_factor_portfolio,
    backtest_portfolio,
    performance_summary,
    ANN_FACTOR,
)


OUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    parser = argparse.ArgumentParser(description="Medallion Lite Backtest")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--rebal-every", type=int, default=24,
                        help="Evaluate entry/exit every N hours")
    parser.add_argument("--tc-bps", type=float, default=30.0)
    parser.add_argument("--entry-threshold", type=float, default=0.65,
                        help="Composite score threshold for entry")
    parser.add_argument("--exit-threshold", type=float, default=0.40,
                        help="Composite score threshold for exit")
    parser.add_argument("--max-positions", type=int, default=25,
                        help="Max concurrent positions (no fixed top-K)")
    parser.add_argument("--max-hold", type=int, default=336,
                        help="Max holding period (hours); 336 = 14 days")
    parser.add_argument("--trailing-stop", type=float, default=0.15)
    parser.add_argument("--max-weight", type=float, default=0.10,
                        help="Per-name position cap")
    parser.add_argument("--no-regime", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("MEDALLION LITE — FACTOR-TIMED MOMENTUM + ENSEMBLE REGIME")
    print("=" * 70)
    print(f"  Entry threshold:   {args.entry_threshold} (top {100*(1-args.entry_threshold):.0f}th pctile)")
    print(f"  Exit threshold:    {args.exit_threshold}")
    print(f"  Max positions:     {args.max_positions} (unconstrained top-K)")
    print(f"  Max hold:          {args.max_hold}h ({args.max_hold/24:.0f}d)")
    print(f"  Trailing stop:     {args.trailing_stop:.0%}")
    print(f"  Max weight:        {args.max_weight:.0%}")
    print(f"  Rebalance:         {args.rebal_every}h")
    print(f"  TC:                {args.tc_bps} bps")

    # ── 1. Load data (same pipeline as LPPLS) ──────────────────────────
    t0 = time.time()
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
    print(f"\nUniverse: {len(symbols)} symbols")
    print(f"Range: {panel['ts'].min()} → {panel['ts'].max()}")
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # ── 2. Build wide-format matrices ──────────────────────────────────
    df = panel.sort_values(["symbol", "ts"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()

    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
    close_wide = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    high_wide = df.pivot(index="ts", columns="symbol", values="high").sort_index()
    volume_wide = df.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)

    # ── 3. Ensemble regime ─────────────────────────────────────────────
    print("\n[regime] Computing ensemble regime score ...")
    if args.no_regime or "BTC-USD" not in panel["symbol"].values:
        regime = pd.Series(1.0, index=returns_wide.index)
        print("  Regime disabled — full exposure at all times")
    else:
        btc_h = panel[panel["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
        btc_d = btc_h.resample("D").last().dropna()
        regime = compute_ensemble_regime(btc_d, btc_h, returns_wide)
        print(f"  Mean regime score:  {regime.mean():.2f}")
        print(f"  % hours > 0.5:     {(regime > 0.5).mean():.1%}")
        print(f"  % hours > 0.8:     {(regime > 0.8).mean():.1%}")
        print(f"  % hours ≥ entry min (0.45): {(regime >= 0.45).mean():.1%}")

    # ── 4. Cross-sectional factors ─────────────────────────────────────
    print("\n[factors] Computing cross-sectional factors ...")
    t1 = time.time()
    factors = compute_factors(close_wide, volume_wide, high_wide)
    composite = compute_composite_score(factors)
    print(f"  5 factors computed in {time.time() - t1:.1f}s")

    avg_entry_eligible = (composite > args.entry_threshold).sum(axis=1).mean()
    print(f"  Avg tokens > entry threshold ({args.entry_threshold}): {avg_entry_eligible:.1f}")

    # ── 5. Build portfolio ─────────────────────────────────────────────
    print(f"\n[portfolio] Building factor-timed portfolio ...")
    t2 = time.time()
    weights, trades = build_factor_portfolio(
        composite, returns_wide, regime,
        entry_threshold=args.entry_threshold,
        exit_score_threshold=args.exit_threshold,
        max_hold_hours=args.max_hold,
        trailing_stop_pct=args.trailing_stop,
        rebalance_every_hours=args.rebal_every,
        max_positions=args.max_positions,
        max_weight=args.max_weight,
    )
    print(f"  Portfolio built in {time.time() - t2:.1f}s")

    # Trade analysis
    if not trades.empty:
        entries = trades[trades["action"] == "entry"]
        exits = trades[trades["action"].str.startswith("exit")]
        print(f"\n  Total entries:       {len(entries):,}")
        print(f"  Total exits:         {len(exits):,}")
        print(f"  Unique symbols:      {entries['symbol'].nunique() if len(entries) > 0 else 0}")

        action_counts = trades["action"].value_counts()
        for action, cnt in action_counts.items():
            print(f"    {action:<20s} {cnt:>5d}")

        if "cum_ret" in exits.columns and len(exits) > 0:
            exits_pnl = exits[exits["cum_ret"].notna()]
            if len(exits_pnl) > 0:
                hit_rate = (exits_pnl["cum_ret"] > 0).mean()
                avg_ret = exits_pnl["cum_ret"].mean()
                avg_hold = exits_pnl["hours_held"].mean()
                print(f"\n  Hit rate:            {hit_rate:.1%}")
                print(f"  Avg return/trade:    {avg_ret:.1%}")
                print(f"  Avg holding period:  {avg_hold:.0f}h ({avg_hold/24:.1f}d)")

    # ── 6. Backtest ────────────────────────────────────────────────────
    print(f"\n[backtest] Running backtest ({args.tc_bps} bps one-way) ...")
    bt = backtest_portfolio(weights, returns_wide, tc_bps=args.tc_bps)
    stats = performance_summary(bt)

    print(f"\n{'='*70}")
    print("MEDALLION LITE — PERFORMANCE")
    print(f"{'='*70}")
    for k, v in stats.items():
        if isinstance(v, float):
            if any(x in k for x in ["vol", "cagr", "return", "dd", "turnover"]):
                print(f"  {k:<25s} {v:.1%}")
            else:
                print(f"  {k:<25s} {v:.2f}")
        else:
            print(f"  {k:<25s} {v}")

    # Gross return for diagnosis
    if not bt.empty:
        gross_cum = (1 + bt["gross_ret"]).cumprod().iloc[-1]
        net_cum = bt["cum_ret"].iloc[-1]
        total_cost = gross_cum - net_cum
        print(f"\n  Gross cumulative:    {gross_cum:.2f}x")
        print(f"  Net cumulative:      {net_cum:.2f}x")
        print(f"  Total cost drag:     {total_cost:.2f}x")

    # ── 7. BTC benchmark ──────────────────────────────────────────────
    btc_ret = returns_wide.get("BTC-USD", pd.Series(0.0, index=returns_wide.index))
    btc_cum = (1 + btc_ret.fillna(0)).cumprod()
    n_years = len(btc_cum) / ANN_FACTOR
    btc_cagr = btc_cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    btc_vol = btc_ret.std() * np.sqrt(ANN_FACTOR)
    btc_sharpe = (
        btc_ret.mean() / btc_ret.std() * np.sqrt(ANN_FACTOR)
        if btc_ret.std() > 1e-12
        else 0
    )
    print(f"\n  BTC B&H:  CAGR={btc_cagr:.1%}  Vol={btc_vol:.1%}  Sharpe={btc_sharpe:.2f}")

    # ── 8. 3-way comparison ──────────────────────────────────────────
    comparison = _three_way_comparison(stats, bt, SORNETTE_DIR / "output", args)

    # ── 9. Save artifacts ─────────────────────────────────────────────
    bt.to_parquet(OUT_DIR / "medallion_backtest.parquet", index=False)
    weights.to_parquet(OUT_DIR / "medallion_weights.parquet")
    trades.to_parquet(OUT_DIR / "medallion_trades.parquet", index=False)

    with open(OUT_DIR / "medallion_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n  Artifacts saved to {OUT_DIR}/")
    return bt, stats, trades


def _three_way_comparison(
    medal_stats: dict,
    medal_bt: pd.DataFrame,
    lppls_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Head-to-head: Medallion Lite vs LPPLS vs Simplicity Benchmark."""
    print(f"\n{'='*70}")
    print("THREE-WAY COMPARISON")
    print(f"{'='*70}")

    strategies: dict[str, dict] = {"medallion": medal_stats}

    lppls_bt_path = lppls_dir / "hf_backtest.parquet"
    if lppls_bt_path.exists():
        lppls_bt = pd.read_parquet(lppls_bt_path)
        lppls_bt["ts"] = pd.to_datetime(lppls_bt["ts"])
        lppls_sharpe = (
            lppls_bt["net_ret"].mean() / lppls_bt["net_ret"].std() * np.sqrt(ANN_FACTOR)
            if lppls_bt["net_ret"].std() > 1e-12 else 0
        )
        n_h = len(lppls_bt)
        n_y = n_h / ANN_FACTOR
        cum = lppls_bt["cum_ret"].iloc[-1]
        dd = lppls_bt["cum_ret"] / lppls_bt["cum_ret"].cummax() - 1
        strategies["lppls"] = dict(
            cagr=cum ** (1 / n_y) - 1 if n_y > 0 else 0,
            sharpe=float(lppls_sharpe),
            max_dd=float(dd.min()),
            total_return=float(cum - 1),
            avg_holdings=float(lppls_bt["n_holdings"].mean()),
            avg_hourly_turnover=float(lppls_bt["turnover"].mean()),
        )

    bench_bt_path = lppls_dir / "benchmark_backtest.parquet"
    if bench_bt_path.exists():
        bench_bt = pd.read_parquet(bench_bt_path)
        bench_bt["ts"] = pd.to_datetime(bench_bt["ts"])
        bench_sharpe = (
            bench_bt["net_ret"].mean() / bench_bt["net_ret"].std() * np.sqrt(ANN_FACTOR)
            if bench_bt["net_ret"].std() > 1e-12 else 0
        )
        n_h = len(bench_bt)
        n_y = n_h / ANN_FACTOR
        cum = bench_bt["cum_ret"].iloc[-1]
        dd = bench_bt["cum_ret"] / bench_bt["cum_ret"].cummax() - 1
        strategies["benchmark"] = dict(
            cagr=cum ** (1 / n_y) - 1 if n_y > 0 else 0,
            sharpe=float(bench_sharpe),
            max_dd=float(dd.min()),
            total_return=float(cum - 1),
            avg_holdings=float(bench_bt["n_holdings"].mean()),
            avg_hourly_turnover=float(bench_bt["turnover"].mean()),
        )

    metrics = ["cagr", "sharpe", "max_dd", "total_return", "avg_holdings", "avg_hourly_turnover"]
    header = f"  {'Metric':<25s}"
    for name in strategies:
        header += f" {name:>14s}"
    print(header)
    print(f"  {'-' * (25 + 15 * len(strategies))}")

    for m in metrics:
        row = f"  {m:<25s}"
        for name in strategies:
            v = strategies[name].get(m, float("nan"))
            if isinstance(v, (int, float)):
                if any(x in m for x in ["vol", "cagr", "return", "dd", "turnover"]):
                    row += f" {v:>13.1%}"
                elif "holdings" in m:
                    row += f" {v:>13.1f}"
                else:
                    row += f" {v:>13.2f}"
            else:
                row += f" {str(v):>13s}"
        print(row)

    sharpe_vals = {n: s.get("sharpe", 0) for n, s in strategies.items()}
    best = max(sharpe_vals, key=sharpe_vals.get)
    print(f"\n  Best Sharpe: {best} ({sharpe_vals[best]:.2f})")

    return {
        "strategies": {
            name: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in s.items()}
            for name, s in strategies.items()
        },
        "best_sharpe": best,
        "parameters": {
            "entry_threshold": args.entry_threshold,
            "exit_threshold": args.exit_threshold,
            "max_positions": args.max_positions,
            "max_hold": args.max_hold,
            "trailing_stop": args.trailing_stop,
            "max_weight": args.max_weight,
            "rebal_every": args.rebal_every,
            "tc_bps": args.tc_bps,
        },
    }


if __name__ == "__main__":
    main()
