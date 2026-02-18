#!/usr/bin/env python3
"""
High-frequency (hourly) LPPLS Jumpers backtest.

Usage:
    python -m scripts.research.sornette_lppl.run_hf_backtest [--start 2023-01-01]

Pipeline:
  1. Load hourly bars from market.duckdb
  2. Filter universe by ADV and listing age
  3. Run fast super-exponential scanner every eval_every hours
  4. For triggered assets, run LPPLS confirmation + tc estimation
  5. Build portfolio with tc-based exits
  6. Backtest with realistic HF costs
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_hf import load_hourly_bars, filter_universe_hourly, ANN_FACTOR_HOURLY
from .scanner_hf import fast_scan_single, lppl_confirm, FAST_WINDOWS_HOURS
from .portfolio_hf import (
    build_hourly_portfolio,
    backtest_hourly,
    hourly_performance_summary,
)
from .regime import compute_regime, regime_mask as make_regime_mask


def _build_hourly_signals(
    panel: pd.DataFrame,
    eval_every: int = 6,
    start_ts: pd.Timestamp | None = None,
    cache_path: Path | None = None,
    lppl_every: int = 4,
) -> pd.DataFrame:
    """Fast-layer scan over hourly bars with LPPLS on triggers.

    Uses pre-sorted numpy arrays and bisect for O(log N) lookups
    instead of per-timestamp DataFrame filtering.
    """
    if cache_path and cache_path.exists():
        print(f"[hf] Loading cached signals from {cache_path}")
        return pd.read_parquet(cache_path)

    symbols = sorted(panel["symbol"].unique())
    all_hours = sorted(panel["ts"].unique())
    if start_ts:
        all_hours = [h for h in all_hours if h >= start_ts]

    eval_hours = all_hours[::eval_every]
    print(f"[hf] Scanning {len(eval_hours)} timestamps × "
          f"{len(symbols)} symbols (eval every {eval_every}h) ...")

    # Pre-sort and convert to numpy for speed
    sym_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym in symbols:
        sdf = panel[panel["symbol"] == sym].sort_values("ts")
        ts_arr = sdf["ts"].values.astype("datetime64[ns]")
        lp_arr = np.log(sdf["close"].values)
        sym_data[sym] = (ts_arr, lp_arr)

    results = []
    t0 = time.time()
    n_triggers = 0
    n_scanned = 0

    for i, ts in enumerate(eval_hours):
        ts_ns = np.datetime64(ts, "ns")

        for sym in symbols:
            ts_arr, lp_arr = sym_data[sym]

            # Binary search for cutoff
            idx = np.searchsorted(ts_arr, ts_ns, side="right")
            if idx < 72:
                continue

            lp = lp_arr[:idx]
            n_scanned += 1

            # Stage 1: fast scan (~0.1ms)
            fast = fast_scan_single(lp)
            if not fast["triggered"]:
                continue

            n_triggers += 1

            # Stage 2: LPPLS confirmation (only every lppl_every triggers
            # or if the signal is strong)
            run_lppl = (n_triggers % lppl_every == 0) or fast["se_score"] > 0.2
            if run_lppl:
                lppl = lppl_confirm(lp)
            else:
                lppl = {"confirmed": False, "tc_hours": float("nan"),
                        "bubble_conf": 0.0, "best_fit": None}

            results.append({
                "symbol": sym,
                "ts": pd.Timestamp(ts),
                "se_score": fast["se_score"],
                "burst": fast["burst"],
                "move_pct": fast["move_pct"],
                "best_window_h": fast["best_window"],
                "n_convex": fast["n_convex"],
                "lppl_confirmed": lppl["confirmed"],
                "tc_hours": lppl["tc_hours"],
                "bubble_conf": lppl["bubble_conf"],
            })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = n_scanned / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(eval_hours)}] {ts} | "
                  f"{n_triggers} triggers | {len(results)} signals | "
                  f"{rate:.0f} scans/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"[hf] Scan done in {elapsed:.0f}s | "
          f"{n_scanned:,} total scans | "
          f"{n_triggers} triggers → {len(results)} signals")

    df = pd.DataFrame(results) if results else pd.DataFrame()

    if cache_path and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[hf] Cached signals → {cache_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="HF LPPLS Jumpers Backtest")
    parser.add_argument("--start", default="2023-01-01",
                        help="Start date for backtest")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=6,
                        help="Scan frequency in hours")
    parser.add_argument("--rebal-every", type=int, default=6,
                        help="Rebalance frequency in hours")
    parser.add_argument("--tc-bps", type=float, default=30.0,
                        help="One-way cost in bps (wider for HF)")
    parser.add_argument("--tc-exit", type=float, default=4.0,
                        help="Exit when LPPLS tc < this many hours")
    parser.add_argument("--max-hold", type=int, default=168,
                        help="Max holding period (hours)")
    parser.add_argument("--trailing-stop", type=float, default=0.15,
                        help="Trailing stop loss (%)")
    parser.add_argument("--no-regime", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("SORNETTE LPPL — HIGH-FREQUENCY JUMPERS (HOURLY)")
    print("=" * 70)

    # 1. Load hourly data (SQL-side ADV pre-filter for speed)
    panel = load_hourly_bars(
        start=args.start, end=args.end,
        min_adv_usd=5_000_000, max_symbols=50,
    )
    if panel.empty:
        print("[hf] No data loaded. Aborting.")
        return
    panel = filter_universe_hourly(panel)
    panel = panel[panel["in_universe"]].copy()
    symbols = sorted(panel["symbol"].unique())
    print(f"Universe: {len(symbols)} symbols")

    hr_range = panel["ts"].agg(["min", "max"])
    print(f"Range: {hr_range['min']} to {hr_range['max']}")

    # 2. Build hourly signals (fast scan + LPPLS on triggers)
    cache_path = out_dir / "hf_signals.parquet" if not args.recompute else None
    if args.recompute:
        for f in out_dir.glob("hf_*.parquet"):
            f.unlink()

    signals = _build_hourly_signals(
        panel,
        eval_every=args.eval_every,
        start_ts=pd.Timestamp(args.start),
        cache_path=out_dir / "hf_signals.parquet",
    )

    if signals.empty:
        print("[hf] No signals detected. Aborting.")
        return

    n_confirmed = signals["lppl_confirmed"].sum() if "lppl_confirmed" in signals.columns else 0
    print(f"\n[hf] Signal summary:")
    print(f"  Total triggers:    {len(signals):,}")
    print(f"  LPPLS confirmed:   {n_confirmed:,} ({100*n_confirmed/len(signals):.0f}%)")
    print(f"  Unique symbols:    {signals['symbol'].nunique()}")

    # 3. Compute hourly returns
    df = panel.sort_values(["symbol", "ts"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index()
    returns_wide = returns_wide.fillna(0.0)

    # 4. Regime filter (upsampled from daily BTC)
    reg_mask = None
    if not args.no_regime and "BTC-USD" in panel["symbol"].values:
        btc = panel[panel["symbol"] == "BTC-USD"].sort_values("ts")
        # Build daily close from hourly (last bar of each day)
        btc_daily = btc.set_index("ts")["close"].resample("D").last().dropna()
        regime_df = compute_regime(btc_daily)

        # Upsample to hourly
        regime_daily = make_regime_mask(regime_df, btc_daily.index)
        reg_mask = regime_daily.reindex(returns_wide.index, method="ffill")
        bull_pct = reg_mask.mean() * 100 if reg_mask is not None else 0
        print(f"\n[regime] BTC dual-SMA: {bull_pct:.0f}% of hours risk-on/bull")

    # 5. Build portfolio
    print(f"\n[portfolio] top-{args.top_k}, rebal={args.rebal_every}h, "
          f"tc={args.tc_bps}bps, tc-exit={args.tc_exit}h, "
          f"max-hold={args.max_hold}h, stop={args.trailing_stop:.0%}")

    weights, trades = build_hourly_portfolio(
        signals, returns_wide,
        top_k=args.top_k,
        rebalance_every_hours=args.rebal_every,
        tc_exit_hours=args.tc_exit,
        max_hold_hours=args.max_hold,
        trailing_stop_pct=args.trailing_stop,
        regime_mask=reg_mask,
    )

    # 6. Backtest
    bt = backtest_hourly(weights, returns_wide, tc_bps=args.tc_bps)

    # 7. Results
    stats = hourly_performance_summary(bt)

    print(f"\n{'='*70}")
    print("HF JUMPERS PORTFOLIO — PERFORMANCE SUMMARY")
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
            hit_rate = (exits["cum_ret"] > 0).mean()
            avg_ret = exits["cum_ret"].mean()
            med_ret = exits["cum_ret"].median()
            avg_hold = exits["hours_held"].mean()
            print(f"\n  Hit rate:            {hit_rate:.1%}")
            print(f"  Avg return/trade:    {avg_ret:.1%}")
            print(f"  Median return/trade: {med_ret:.1%}")
            print(f"  Avg holding period:  {avg_hold:.0f}h ({avg_hold/24:.1f}d)")

        # tc-exit analysis
        tc_exits = trades[trades["action"] == "exit_tc"]
        if len(tc_exits) > 0:
            tc_hit = (tc_exits["cum_ret"] > 0).mean()
            tc_avg = tc_exits["cum_ret"].mean()
            print(f"\n  tc-based exits:      {len(tc_exits)}")
            print(f"  tc-exit hit rate:    {tc_hit:.1%}")
            print(f"  tc-exit avg return:  {tc_avg:.1%}")

    # Benchmarks
    btc_ret = returns_wide.get("BTC-USD",
                                pd.Series(0.0, index=returns_wide.index)).fillna(0)
    btc_cum = (1 + btc_ret).cumprod()
    n_years = len(btc_cum) / ANN_FACTOR_HOURLY
    btc_cagr = btc_cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    btc_vol = btc_ret.std() * np.sqrt(ANN_FACTOR_HOURLY)
    btc_sharpe = btc_cagr / btc_vol if btc_vol > 0 else 0

    print(f"\n{'='*70}")
    print("BENCHMARK")
    print(f"{'='*70}")
    print(f"  BTC B&H:  CAGR={btc_cagr:.1%}  Vol={btc_vol:.1%}  "
          f"Sharpe={btc_sharpe:.2f}")

    # Save
    bt.to_parquet(out_dir / "hf_backtest.parquet", index=False)
    trades.to_parquet(out_dir / "hf_trades.parquet", index=False)
    weights.to_parquet(out_dir / "hf_weights.parquet")
    print(f"\n  Saved to {out_dir}/")

    return bt, stats, trades


if __name__ == "__main__":
    main()
