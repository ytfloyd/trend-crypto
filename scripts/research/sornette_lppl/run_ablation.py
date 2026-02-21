#!/usr/bin/env python3
"""
Ablation studies and robustness checks for the Jumpers portfolio.

1. Cost sensitivity: 20/30/50/100/150 bps
2. Fast-layer-only ablation (no LPPLS)
3. BTC-SMA-gated Equal-Weight benchmark
4. Full 2021-2026 primary results

Used to generate tables for the research report.
"""
from __future__ import annotations

import json
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


def _load_indicators(panel: pd.DataFrame, out_dir: Path):
    """Load or compute indicators."""
    lppl_path = out_dir / "bubble_indicators.parquet"
    se_path = out_dir / "superexp_indicators.parquet"
    symbols = sorted(panel["symbol"].unique())

    if lppl_path.exists():
        lppl_df = pd.read_parquet(lppl_path)
    else:
        print(f"  Computing LPPL ({len(symbols)} symbols) ...")
        t0 = time.time()
        lppl_df = compute_bubble_panel(panel, symbols=symbols, eval_every=20)
        print(f"  LPPL done in {time.time()-t0:.0f}s")
        lppl_df.to_parquet(lppl_path, index=False)

    if se_path.exists():
        se_df = pd.read_parquet(se_path)
    else:
        print(f"  Computing SuperExp ...")
        t0 = time.time()
        se_df = compute_superexp_panel(panel, eval_every=5, min_history=90)
        print(f"  SuperExp done in {time.time()-t0:.0f}s")
        se_df.to_parquet(se_path, index=False)

    return lppl_df, se_df


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    START, END = "2021-01-01", "2026-12-31"

    print("=" * 70)
    print("ABLATION STUDIES — JUMPERS PORTFOLIO")
    print("=" * 70)

    # Load data
    panel = load_daily_bars(start=START, end=END)
    panel = filter_universe(panel, min_adv_usd=1_000_000)
    panel = panel[panel["in_universe"]].copy()
    symbols = sorted(panel["symbol"].unique())
    print(f"Universe: {len(symbols)} symbols, "
          f"{panel['ts'].min().date()} to {panel['ts'].max().date()}")

    # Indicators
    lppl_df, se_df = _load_indicators(panel, out_dir)

    # Blended signals (also save for PDF generator)
    sig_df = blend_signals(lppl_df, se_df)
    sig_df.to_parquet(out_dir / "blended_signals.parquet", index=False)

    # Returns
    df = panel.sort_values(["symbol", "ts"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index()
    returns_wide = returns_wide.fillna(0.0)

    # Regime
    btc_close = panel[panel["symbol"] == "BTC-USD"].set_index("ts")["close"]
    regime_df = compute_regime(btc_close)
    reg_mask = make_regime_mask(regime_df, returns_wide.index)

    # =========================================================================
    # 1. COST SENSITIVITY
    # =========================================================================
    print(f"\n{'='*70}")
    print("1. TRANSACTION COST SENSITIVITY")
    print(f"{'='*70}")

    weights = build_portfolio_weights(
        sig_df, returns_wide,
        top_k=10, rebalance_every=5, ivol_weight=True,
        regime_mask=reg_mask,
    )

    cost_results = []
    for tc_bps in [0, 20, 30, 50, 100, 150]:
        bt = backtest_portfolio(weights, returns_wide, tc_bps=tc_bps)
        stats = performance_summary(bt, ann_factor=ANN_FACTOR)
        stats["tc_bps"] = tc_bps
        cost_results.append(stats)
        print(f"  {tc_bps:3d} bps:  CAGR={stats['cagr']:.1%}  "
              f"Sharpe={stats['sharpe']:.2f}  MaxDD={stats['max_dd']:.1%}")

    # Save the canonical backtest (single source of truth for the PDF)
    bt_blend = backtest_portfolio(weights, returns_wide, tc_bps=20)
    bt_blend.to_parquet(out_dir / "jumpers_backtest.parquet", index=False)
    weights.to_parquet(out_dir / "jumpers_weights.parquet")
    print(f"  -> Saved canonical backtest + weights to {out_dir}/")

    # =========================================================================
    # 2. FAST-LAYER-ONLY ABLATION (no LPPLS)
    # =========================================================================
    print(f"\n{'='*70}")
    print("2. FAST-LAYER-ONLY ABLATION")
    print(f"{'='*70}")

    # Build fast-only signals (100% weight on fast, 0% on LPPLS)
    sig_fast_only = blend_signals(lppl_df, se_df, w_fast=1.0, w_lppl=0.0)
    weights_fast = build_portfolio_weights(
        sig_fast_only, returns_wide,
        top_k=10, rebalance_every=5, ivol_weight=True,
        regime_mask=reg_mask,
    )
    bt_fast = backtest_portfolio(weights_fast, returns_wide, tc_bps=20)
    stats_fast = performance_summary(bt_fast, ann_factor=ANN_FACTOR)
    print(f"  Fast-only:   CAGR={stats_fast['cagr']:.1%}  "
          f"Sharpe={stats_fast['sharpe']:.2f}  MaxDD={stats_fast['max_dd']:.1%}")

    # Full blend (baseline) — already computed above as canonical backtest
    stats_blend = performance_summary(bt_blend, ann_factor=ANN_FACTOR)
    print(f"  Blended:     CAGR={stats_blend['cagr']:.1%}  "
          f"Sharpe={stats_blend['sharpe']:.2f}  MaxDD={stats_blend['max_dd']:.1%}")

    # LPPL-only
    sig_lppl_only = blend_signals(lppl_df, se_df, w_fast=0.0, w_lppl=1.0)
    weights_lppl = build_portfolio_weights(
        sig_lppl_only, returns_wide,
        top_k=10, rebalance_every=5, ivol_weight=True,
        regime_mask=reg_mask,
    )
    bt_lppl = backtest_portfolio(weights_lppl, returns_wide, tc_bps=20)
    stats_lppl = performance_summary(bt_lppl, ann_factor=ANN_FACTOR)
    print(f"  LPPL-only:   CAGR={stats_lppl['cagr']:.1%}  "
          f"Sharpe={stats_lppl['sharpe']:.2f}  MaxDD={stats_lppl['max_dd']:.1%}")

    lppl_marginal = stats_blend['sharpe'] - stats_fast['sharpe']
    print(f"\n  LPPL marginal Sharpe contribution: {lppl_marginal:+.2f}")

    # =========================================================================
    # 3. BTC-SMA-GATED EQUAL-WEIGHT BENCHMARK
    # =========================================================================
    print(f"\n{'='*70}")
    print("3. BTC-SMA-GATED EQUAL-WEIGHT BENCHMARK")
    print(f"{'='*70}")

    # EW basket of full universe, gated by BTC regime
    ew_ret = returns_wide[symbols].mean(axis=1)
    gated_ew = ew_ret.copy()
    daily_cash = (1 + 0.04) ** (1 / 365) - 1
    gated_ew[~reg_mask] = daily_cash
    gated_cum = (1 + gated_ew).cumprod()
    n_years = len(gated_cum) / ANN_FACTOR
    gated_cagr = gated_cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    gated_vol = gated_ew.std() * np.sqrt(ANN_FACTOR)
    gated_daily_std = gated_ew.std()
    gated_sharpe = (gated_ew.mean() / gated_daily_std * np.sqrt(ANN_FACTOR)) if gated_daily_std > 1e-12 else 0
    gated_dd = gated_cum / gated_cum.cummax() - 1
    gated_maxdd = gated_dd.min()

    # Ungated EW
    ew_cum = (1 + ew_ret).cumprod()
    ew_cagr = ew_cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    ew_vol = ew_ret.std() * np.sqrt(ANN_FACTOR)
    ew_daily_std = ew_ret.std()
    ew_sharpe = (ew_ret.mean() / ew_daily_std * np.sqrt(ANN_FACTOR)) if ew_daily_std > 1e-12 else 0

    # BTC
    btc_ret = returns_wide.get("BTC-USD", pd.Series(0, index=returns_wide.index))
    btc_cum = (1 + btc_ret).cumprod()
    btc_cagr = btc_cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    btc_vol = btc_ret.std() * np.sqrt(ANN_FACTOR)
    btc_daily_std = btc_ret.std()
    btc_sharpe = (btc_ret.mean() / btc_daily_std * np.sqrt(ANN_FACTOR)) if btc_daily_std > 1e-12 else 0

    print(f"  BTC-SMA-gated EW: CAGR={gated_cagr:.1%}  "
          f"Sharpe={gated_sharpe:.2f}  MaxDD={gated_maxdd:.1%}")
    print(f"  Ungated EW:       CAGR={ew_cagr:.1%}  "
          f"Sharpe={ew_sharpe:.2f}")
    print(f"  BTC B&H:          CAGR={btc_cagr:.1%}  "
          f"Sharpe={btc_sharpe:.2f}")
    print(f"  Jumpers (20bps):  CAGR={stats_blend['cagr']:.1%}  "
          f"Sharpe={stats_blend['sharpe']:.2f}  MaxDD={stats_blend['max_dd']:.1%}")

    # =========================================================================
    # Save all results
    # =========================================================================
    results = {
        "cost_sensitivity": cost_results,
        "ablation": {
            "fast_only": stats_fast,
            "blended": stats_blend,
            "lppl_only": stats_lppl,
            "lppl_marginal_sharpe": lppl_marginal,
        },
        "benchmarks": {
            "btc_sma_gated_ew": {
                "cagr": gated_cagr, "vol": gated_vol,
                "sharpe": gated_sharpe, "max_dd": gated_maxdd,
            },
            "ungated_ew": {"cagr": ew_cagr, "vol": ew_vol, "sharpe": ew_sharpe},
            "btc": {"cagr": btc_cagr, "vol": btc_vol, "sharpe": btc_sharpe},
        },
    }

    results_path = out_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
