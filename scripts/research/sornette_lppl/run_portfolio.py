#!/usr/bin/env python3
"""
Backtest the "Jumpers" portfolio: go long crypto assets with the
strongest LPPL + super-exponential signals.

Usage:
    python -m scripts.research.sornette_lppl.run_portfolio [--top-k 10]

Computes both LPPL bubble indicators AND super-exponential growth
scores, blends them, and constructs a portfolio of "jumpers".
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_daily_bars, filter_universe, ANN_FACTOR
from .bubble_indicator import compute_bubble_panel
from .superexp import compute_superexp_panel
from .signals import blend_signals
from .portfolio import build_portfolio_weights, backtest_portfolio, performance_summary
from .regime import compute_regime, regime_mask as make_regime_mask


def _load_or_compute(
    panel: pd.DataFrame,
    out_dir: Path,
    lppl_eval_every: int = 20,
    se_eval_every: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached indicators or compute from scratch.

    LPPL is expensive → eval every 20 days.
    Super-exponential is cheap → eval every 5 days.
    """
    lppl_path = out_dir / "bubble_indicators.parquet"
    se_path = out_dir / "superexp_indicators.parquet"

    symbols = sorted(panel["symbol"].unique())

    if lppl_path.exists():
        print(f"[signals] Loading cached LPPL indicators")
        lppl_df = pd.read_parquet(lppl_path)
    else:
        print(f"[signals] Computing LPPL indicators for {len(symbols)} symbols "
              f"(eval every {lppl_eval_every}d) ...")
        t0 = time.time()
        lppl_df = compute_bubble_panel(
            panel, symbols=symbols, eval_every=lppl_eval_every,
        )
        elapsed = time.time() - t0
        print(f"  LPPL done in {elapsed:.0f}s ({len(lppl_df):,} obs)")
        lppl_df.to_parquet(lppl_path, index=False)

    if se_path.exists():
        print(f"[signals] Loading cached super-exponential indicators")
        se_df = pd.read_parquet(se_path)
    else:
        print(f"[signals] Computing super-exponential indicators "
              f"(eval every {se_eval_every}d) ...")
        t0 = time.time()
        se_df = compute_superexp_panel(
            panel, eval_every=se_eval_every, min_history=90,
        )
        elapsed = time.time() - t0
        print(f"  SuperExp done in {elapsed:.0f}s ({len(se_df):,} obs)")
        se_df.to_parquet(se_path, index=False)

    return lppl_df, se_df


def main():
    parser = argparse.ArgumentParser(description="LPPL Jumpers Portfolio Backtest")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rebal", type=int, default=5)
    parser.add_argument("--vol-target", type=float, default=0.40)
    parser.add_argument("--tc-bps", type=float, default=20.0)
    parser.add_argument("--no-ivol", action="store_true")
    parser.add_argument("--min-adv", type=float, default=1_000_000)
    parser.add_argument("--lppl-eval", type=int, default=20,
                        help="LPPL eval frequency (days, expensive)")
    parser.add_argument("--se-eval", type=int, default=5,
                        help="Super-exponential eval frequency (days, cheap)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--no-regime", action="store_true",
                        help="Disable regime filter")
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation of indicators")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    if args.recompute:
        for f in out_dir.glob("*.parquet"):
            f.unlink()

    # 1. Load data
    print("=" * 70)
    print("SORNETTE LPPL — JUMPERS PORTFOLIO BACKTEST (Blended Signals)")
    print("=" * 70)
    panel = load_daily_bars(start=args.start, end=args.end)
    panel = filter_universe(panel, min_adv_usd=args.min_adv)
    panel = panel[panel["in_universe"]].copy()
    symbols = sorted(panel["symbol"].unique())
    print(f"Universe: {len(symbols)} symbols")

    date_range = panel["ts"].agg(["min", "max"])
    print(f"Date range: {date_range['min'].date()} to {date_range['max'].date()}")

    # 2. Compute / load indicators
    lppl_df, se_df = _load_or_compute(
        panel, out_dir,
        lppl_eval_every=args.lppl_eval,
        se_eval_every=args.se_eval,
    )

    # 3. Blend signals
    print("[signals] Blending LPPL + super-exponential signals ...")
    sig_df = blend_signals(lppl_df, se_df)
    sig_df.to_parquet(out_dir / "blended_signals.parquet", index=False)

    active = sig_df[sig_df["signal"] > 0]
    print(f"  Total signal observations: {len(sig_df):,}")
    print(f"  Active (signal > 0):       {len(active):,}")
    print(f"  Signal type distribution:")
    if not active.empty:
        for st, cnt in active["signal_type"].value_counts().items():
            print(f"    {st}: {cnt}")

    # 4. Compute returns (close-to-close)
    df = panel.sort_values(["symbol", "ts"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index()
    returns_wide = returns_wide.fillna(0.0)

    # 5. Regime filter
    reg_mask = None
    if not args.no_regime and "BTC-USD" in returns_wide.columns:
        btc_close = panel[panel["symbol"] == "BTC-USD"].set_index("ts")["close"]
        regime_df = compute_regime(btc_close)
        reg_mask = make_regime_mask(regime_df, returns_wide.index)

        bull_pct = (reg_mask).mean() * 100
        print(f"\n[regime] BTC dual-SMA filter: {bull_pct:.0f}% of days risk-on/bull")
    elif args.no_regime:
        print("\n[regime] DISABLED")

    # 6. Build portfolio
    print(f"Portfolio: top-{args.top_k}, rebal={args.rebal}d, "
          f"vol-target={args.vol_target}, tc={args.tc_bps}bps, "
          f"ivol={'OFF' if args.no_ivol else 'ON'}")

    weights = build_portfolio_weights(
        sig_df, returns_wide,
        top_k=args.top_k,
        rebalance_every=args.rebal,
        ivol_weight=not args.no_ivol,
        regime_mask=reg_mask,
    )

    # 7. Backtest
    vol_target = args.vol_target if args.vol_target > 0 else None
    bt = backtest_portfolio(
        weights, returns_wide,
        vol_target=vol_target,
        tc_bps=args.tc_bps,
    )

    # 7. Results
    stats = performance_summary(bt, ann_factor=ANN_FACTOR)

    print(f"\n{'='*70}")
    print("JUMPERS PORTFOLIO — PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    for k, v in stats.items():
        if isinstance(v, float):
            if "vol" in k or "cagr" in k or "return" in k or "dd" in k or "turnover" in k:
                print(f"  {k:<22s} {v:.1%}")
            else:
                print(f"  {k:<22s} {v:.2f}")
        else:
            print(f"  {k:<22s} {v}")

    # Benchmarks
    ew_ret = returns_wide[symbols].mean(axis=1)
    ew_cum = (1 + ew_ret).cumprod()
    ew_years = len(ew_cum) / ANN_FACTOR
    ew_cagr = ew_cum.iloc[-1] ** (1 / ew_years) - 1 if ew_years > 0 else 0
    ew_vol = ew_ret.std() * np.sqrt(ANN_FACTOR)
    ew_sharpe = ew_cagr / ew_vol if ew_vol > 0 else 0

    btc_ret = returns_wide.get("BTC-USD", pd.Series(0.0, index=returns_wide.index)).fillna(0)
    btc_cum = (1 + btc_ret).cumprod()
    btc_cagr = btc_cum.iloc[-1] ** (1 / ew_years) - 1 if ew_years > 0 else 0
    btc_vol = btc_ret.std() * np.sqrt(ANN_FACTOR)
    btc_sharpe = btc_cagr / btc_vol if btc_vol > 0 else 0

    print(f"\n{'='*70}")
    print("BENCHMARKS")
    print(f"{'='*70}")
    print(f"  EW Basket:  CAGR={ew_cagr:.1%}  Vol={ew_vol:.1%}  Sharpe={ew_sharpe:.2f}")
    print(f"  BTC B&H:    CAGR={btc_cagr:.1%}  Vol={btc_vol:.1%}  Sharpe={btc_sharpe:.2f}")

    # Holdings analysis
    if not bt.empty:
        invested = bt[bt["n_holdings"] > 0]
        pct_invested = len(invested) / len(bt) * 100
        print(f"\n  % days invested:  {pct_invested:.0f}%")
        print(f"  Avg holdings:     {bt['n_holdings'].mean():.1f}")

    # Save
    bt.to_parquet(out_dir / "jumpers_backtest.parquet", index=False)
    weights.to_parquet(out_dir / "jumpers_weights.parquet")
    print(f"\n✓ Saved to {out_dir}/")

    return bt, stats


if __name__ == "__main__":
    main()
