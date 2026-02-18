#!/usr/bin/env python
"""
ETF — Chapter 3: Multi-Signal Diversified Portfolio.

Blends multiple momentum speeds and signal types across the ETF universe.
EMV (Equal Marginal Volatility) weighting across sub-strategies.

This is the paper's key insight adapted for ETFs: blending multiple
signal types and lookbacks dramatically improves Sharpe and reduces drawdown.

Usage:
    python -m scripts.research.jpm_momentum.run_etf_ch3_diversified
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_etf_daily_bars, compute_returns_wide, ANN_FACTOR_ETF, compute_benchmark
from .signals import compute_signal
from .universe import filter_universe
from .weights import build_weights
from .backtest import simple_backtest
from .metrics import compute_metrics, format_metrics_table
from scripts.research.etf_data.universe import get_core_universe


BLEND_SIGNALS = ("RET", "MAC", "BRK", "LREG")
BLEND_LOOKBACKS = (10, 21, 63, 126)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF Ch3: Multi-signal diversified portfolio")
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default="output/jpm_momentum_etf/ch3_diversified")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--weight-method", default="inv_vol")
    p.add_argument("--universe", choices=["full", "core"], default="core")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    af = ANN_FACTOR_ETF

    if args.universe == "core":
        tickers = get_core_universe()
    else:
        from scripts.research.etf_data.universe import get_full_universe
        tickers = get_full_universe()

    print("=== ETF Ch3: Multi-Signal Diversified Portfolio ===")
    panel = load_etf_daily_bars(start=args.start, end=args.end, tickers=tickers)
    panel_u = filter_universe(panel, min_adv_usd=5_000_000, min_history_days=252)
    returns_wide = compute_returns_wide(panel, method="close_to_close")

    # --- Build sub-strategy equity curves ---
    sub_equities: dict[str, pd.Series] = {}

    for sig_type in BLEND_SIGNALS:
        for lb in BLEND_LOOKBACKS:
            label = f"{sig_type}_L{lb}"
            sig_panel = compute_signal(panel_u, signal_type=sig_type, lookback=lb)
            eligible = sig_panel[sig_panel["in_universe"]].copy()
            eligible["selected"] = eligible["signal"] > 0
            mask = eligible.pivot(index="ts", columns="symbol", values="selected").fillna(False)
            w = build_weights(mask, returns_wide, method=args.weight_method, ann_factor=af)
            bt = simple_backtest(w, returns_wide, cost_bps=args.cost_bps, ann_factor=af)

            eq = pd.Series(bt["portfolio_equity"].values, index=pd.to_datetime(bt["ts"]))
            sub_equities[label] = eq

    # --- Correlation across sub-strategies ---
    eq_df = pd.DataFrame(sub_equities)
    ret_df = eq_df.pct_change().dropna()
    corr = ret_df.corr()
    corr.to_csv(out_dir / "sub_strategy_correlation.csv")
    mean_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().mean()
    print(f"\nMean pairwise correlation: {mean_corr:.3f}")

    # --- EMV (Equal Marginal Volatility) blend ---
    vol_ann = ret_df.std() * np.sqrt(af)
    inv_vol = 1.0 / vol_ann.clip(lower=0.01)
    emv_weights = inv_vol / inv_vol.sum()
    print(f"\nEMV weights:\n{emv_weights.round(3).to_string()}")

    # Composite equity = EMV-weighted average of sub-strategy returns
    composite_ret = (ret_df * emv_weights).sum(axis=1)
    composite_equity = (1 + composite_ret).cumprod()
    composite_equity.name = "diversified"

    m = compute_metrics(composite_equity, ann_factor=af)
    m["label"] = "diversified_emv"

    # --- Individual sub-strategy metrics + benchmark ---
    all_results = []
    for label, eq in sub_equities.items():
        mi = compute_metrics(eq, ann_factor=af)
        mi["label"] = label
        all_results.append(mi)
    all_results.append(m)

    # SPY benchmark
    try:
        spy_eq = compute_benchmark(panel, ticker="SPY")
        spy_m = compute_metrics(spy_eq, ann_factor=af)
        spy_m["label"] = "SPY_buy_hold"
        all_results.append(spy_m)
    except ValueError:
        pass

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "diversified_results.csv", index=False)
    print(f"\n{format_metrics_table(all_results)}")


if __name__ == "__main__":
    main()
