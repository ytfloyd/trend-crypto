#!/usr/bin/env python
"""
Chapter 3: Long-Only Optimised Portfolio (paper p.69).

Key adaptations:
- All weights clipped to >= 0 (long-only)
- Absolute momentum naturally produces long-only (cash when signal negative)
- For relative momentum: long top quintile only (no short leg)
- Compare long-only absolute vs. long-only relative vs. combined

Usage:
    python -m scripts.research.jpm_momentum.run_ch3_long_only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_daily_bars, compute_returns_wide
from .signals import compute_signal
from .universe import filter_universe
from .weights import build_weights
from .backtest import simple_backtest
from .metrics import compute_metrics, format_metrics_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ch3: Long-only optimised portfolio")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default="output/jpm_momentum/ch3_long_only")
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--signal-type", default="RET")
    p.add_argument("--lookback", type=int, default=21)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--weight-method", default="inv_vol")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Ch3: Long-Only Portfolio Comparison ===")
    panel = load_daily_bars(start=args.start, end=args.end)
    panel_u = filter_universe(panel)
    returns_wide = compute_returns_wide(panel)

    sig_panel = compute_signal(panel_u, signal_type=args.signal_type, lookback=args.lookback)
    eligible = sig_panel[sig_panel["in_universe"]].copy()

    results = []

    # --- Absolute (TSMOM): long when signal > 0 ---
    abs_sel = eligible.copy()
    abs_sel["selected"] = abs_sel["signal"] > 0
    mask_abs = abs_sel.pivot(index="ts", columns="symbol", values="selected").fillna(False)
    w_abs = build_weights(mask_abs, returns_wide, method=args.weight_method)
    bt_abs = simple_backtest(w_abs, returns_wide, cost_bps=args.cost_bps)
    eq_abs = pd.Series(bt_abs["portfolio_equity"].values, index=pd.to_datetime(bt_abs["ts"]))
    m_abs = compute_metrics(eq_abs)
    m_abs["label"] = "absolute_long_only"
    results.append(m_abs)

    # --- Relative (XSMOM): top-K by signal ---
    rel_sel = eligible.copy()
    ranked = rel_sel.groupby("ts")["signal"].rank(ascending=False)
    rel_sel["selected"] = ranked <= args.top_k
    mask_rel = rel_sel.pivot(index="ts", columns="symbol", values="selected").fillna(False)
    w_rel = build_weights(mask_rel, returns_wide, method=args.weight_method)
    bt_rel = simple_backtest(w_rel, returns_wide, cost_bps=args.cost_bps)
    eq_rel = pd.Series(bt_rel["portfolio_equity"].values, index=pd.to_datetime(bt_rel["ts"]))
    m_rel = compute_metrics(eq_rel)
    m_rel["label"] = "relative_long_only"
    results.append(m_rel)

    # --- Combined: 50/50 blend of absolute + relative ---
    w_combined = 0.5 * w_abs.reindex_like(w_rel).fillna(0.0) + 0.5 * w_rel
    bt_comb = simple_backtest(w_combined, returns_wide, cost_bps=args.cost_bps)
    eq_comb = pd.Series(bt_comb["portfolio_equity"].values, index=pd.to_datetime(bt_comb["ts"]))
    m_comb = compute_metrics(eq_comb)
    m_comb["label"] = "combined_50_50"
    results.append(m_comb)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "long_only_results.csv", index=False)
    print(f"\n{format_metrics_table(results)}")


if __name__ == "__main__":
    main()
