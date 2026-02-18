#!/usr/bin/env python3
"""
Scan all crypto tokens for LPPL bubble signatures.

Usage:
    python -m scripts.research.sornette_lppl.run_bubble_scan [--top 30]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_daily_bars, filter_universe
from .bubble_indicator import compute_bubble_panel, WINDOWS
from .signals import compute_signals


def main():
    parser = argparse.ArgumentParser(description="LPPL Bubble Scanner")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top symbols to display")
    parser.add_argument("--min-adv", type=float, default=1_000_000,
                        help="Minimum 20d ADV in USD")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate every N days")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-12-31")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 1. Load data
    print("=" * 70)
    print("SORNETTE LPPL BUBBLE SCANNER — Crypto Digital Assets")
    print("=" * 70)
    panel = load_daily_bars(start=args.start, end=args.end)
    panel = filter_universe(panel, min_adv_usd=args.min_adv)
    panel = panel[panel["in_universe"]].copy()
    symbols = sorted(panel["symbol"].unique())
    print(f"\nUniverse: {len(symbols)} symbols after filtering")

    # 2. Compute bubble indicators
    t0 = time.time()
    print(f"\nFitting LPPL on windows {WINDOWS} (eval every {args.eval_every}d)...")
    bubble_df = compute_bubble_panel(
        panel, symbols=symbols, eval_every=args.eval_every,
    )
    elapsed = time.time() - t0
    print(f"\nLPPL scan completed in {elapsed:.0f}s  ({len(bubble_df):,} observations)")

    # Save raw indicators
    bubble_df.to_parquet(out_dir / "bubble_indicators.parquet", index=False)
    print(f"Saved → {out_dir / 'bubble_indicators.parquet'}")

    # 3. Generate signals
    sig_df = compute_signals(bubble_df)
    sig_df.to_parquet(out_dir / "bubble_signals.parquet", index=False)

    # 4. Display latest snapshot
    latest = sig_df["ts"].max()
    snap = sig_df[sig_df["ts"] == latest].sort_values("signal", ascending=False)
    snap = snap.head(args.top)

    print(f"\n{'='*70}")
    print(f"TOP {args.top} BUBBLE SIGNALS — {latest}")
    print(f"{'='*70}")
    print(f"{'Symbol':<14} {'Signal':>8} {'Type':<22} {'BubConf':>8} "
          f"{'ABConf':>8} {'tc_days':>8} {'#Win':>5}")
    print("-" * 80)
    for _, row in snap.iterrows():
        print(f"{row['symbol']:<14} {row['signal']:>8.3f} "
              f"{row['signal_type']:<22} {row['bubble_conf']:>8.3f} "
              f"{row['antibubble_conf']:>8.3f} {row['tc_days']:>8.1f} "
              f"{row['n_valid']:>5}")

    print(f"\n✓ Full results saved to {out_dir}/")
    return sig_df


if __name__ == "__main__":
    main()
