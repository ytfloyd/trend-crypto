#!/usr/bin/env python
"""
Chapter 3: Multi-Signal Diversified Portfolio (paper p.73).

The paper's key insight: blending multiple momentum speeds and signal types
dramatically improves Sharpe and reduces drawdown.

Steps:
1. Blend signals: equal-weight across signal types (RET, MAC, BRK, LREG) at multiple lookbacks.
2. Portfolio of momentum portfolios: combine absolute and relative.
3. Compute correlation matrix across signals to verify diversification benefit.
4. Final portfolio: EMV (Equal Marginal Volatility) weighting across sub-strategies.

Usage:
    python -m scripts.research.jpm_momentum.run_ch3_diversified
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_LOOKBACKS
from .data import load_daily_bars, compute_returns_wide, ANN_FACTOR
from .signals import compute_signal
from .universe import filter_universe
from .weights import build_weights
from .backtest import simple_backtest
from .metrics import compute_metrics, format_metrics_table


BLEND_SIGNALS = ("RET", "MAC", "BRK", "LREG")
BLEND_LOOKBACKS = (10, 21, 63, 126)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ch3: Multi-signal diversified portfolio")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default="output/jpm_momentum/ch3_diversified")
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--weight-method", default="inv_vol")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Ch3: Multi-Signal Diversified Portfolio ===")
    panel = load_daily_bars(start=args.start, end=args.end)
    panel_u = filter_universe(panel)
    returns_wide = compute_returns_wide(panel)

    # --- Build sub-strategy equity curves ---
    sub_equities: dict[str, pd.Series] = {}
    sub_weights: dict[str, pd.DataFrame] = {}

    for sig_type in BLEND_SIGNALS:
        for lb in BLEND_LOOKBACKS:
            label = f"{sig_type}_L{lb}"
            sig_panel = compute_signal(panel_u, signal_type=sig_type, lookback=lb)
            eligible = sig_panel[sig_panel["in_universe"]].copy()
            eligible["selected"] = eligible["signal"] > 0
            mask = eligible.pivot(index="ts", columns="symbol", values="selected").fillna(False)
            w = build_weights(mask, returns_wide, method=args.weight_method)
            bt = simple_backtest(w, returns_wide, cost_bps=args.cost_bps)

            eq = pd.Series(bt["portfolio_equity"].values, index=pd.to_datetime(bt["ts"]))
            sub_equities[label] = eq
            sub_weights[label] = w

    # --- Correlation across sub-strategies ---
    eq_df = pd.DataFrame(sub_equities)
    ret_df = eq_df.pct_change().dropna()
    corr = ret_df.corr()
    corr.to_csv(out_dir / "sub_strategy_correlation.csv")
    print(f"\nMean pairwise correlation: {corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().mean():.3f}")

    # --- EMV (Equal Marginal Volatility) blend ---
    vol_ann = ret_df.std() * np.sqrt(ANN_FACTOR)
    inv_vol = 1.0 / vol_ann.clip(lower=0.05)
    emv_weights = inv_vol / inv_vol.sum()
    print(f"\nEMV weights:\n{emv_weights.round(3).to_string()}")

    # Composite equity = EMV-weighted average of sub-strategy returns
    composite_ret = (ret_df * emv_weights).sum(axis=1)
    composite_equity = (1 + composite_ret).cumprod()
    composite_equity.name = "diversified"

    m = compute_metrics(composite_equity)
    m["label"] = "diversified_emv"

    # --- Individual sub-strategy metrics for comparison ---
    all_results = []
    for label, eq in sub_equities.items():
        mi = compute_metrics(eq)
        mi["label"] = label
        all_results.append(mi)
    all_results.append(m)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "diversified_results.csv", index=False)
    print(f"\n{format_metrics_table(all_results)}")


if __name__ == "__main__":
    main()
