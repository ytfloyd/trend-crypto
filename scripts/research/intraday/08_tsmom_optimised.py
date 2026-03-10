#!/usr/bin/env python3
"""
Strategy 8: Optimised TSMOM — the best signal scaled for production.

Round 2 findings:
  - TSMOM Slow (3-28d) has Sharpe 0.42, MaxDD -6.6%
  - But only 0.9% CAGR because vol targeting is too conservative
  - Max leverage cap of 2.0 constrains the strategy

This script:
  1. Sweeps vol targets from 10% to 40%
  2. Tests universe sizes (5, 10, 15, 20 assets)
  3. Computes year-by-year performance
  4. Identifies the optimal risk/return trade-off
  5. Builds a production-ready combined strategy
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import numpy as np
import pandas as pd

from common.data import load_bars, ANN_FACTOR
from common.backtest import simple_backtest
from common.metrics import compute_metrics, format_metrics_table
from common.risk_overlays import apply_vol_targeting, apply_position_limit_wide

RESULTS_DIR = Path(__file__).resolve().parent / "_results"
RESULTS_DIR.mkdir(exist_ok=True)
COST_BPS = 15.0


def load_4h_topN(n, start="2020-01-01", end="2025-12-31"):
    panel = load_bars("4h", start=start, end=end)
    panel["dollar_vol"] = panel["close"] * panel["volume"]
    avg_dv = panel.groupby("symbol")["dollar_vol"].mean().sort_values(ascending=False)
    top = avg_dv.head(n).index.tolist()
    panel = panel[panel["symbol"].isin(top)]
    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    returns_wide = np.log(close_wide / close_wide.shift(1)).replace([np.inf, -np.inf], np.nan)
    return close_wide, returns_wide


def tsmom_signal(returns_wide, horizons=None):
    if horizons is None:
        horizons = {18: 0.20, 42: 0.30, 84: 0.30, 168: 0.20}
    vol = returns_wide.rolling(42, min_periods=12).std()
    signal = pd.DataFrame(0.0, index=returns_wide.index, columns=returns_wide.columns)
    for h, w in horizons.items():
        cum_ret = returns_wide.rolling(h).sum()
        norm_ret = cum_ret / (vol * np.sqrt(h)).replace(0, np.nan)
        signal += w * norm_ret.fillna(0)
    return 2 / (1 + np.exp(-signal)) - 1


def vol_sized_weights(signal, returns_wide, vol_lookback=42):
    """Continuous signal → vol-normalised weights (long-only)."""
    sig = signal.clip(lower=0)
    vol = returns_wide.rolling(vol_lookback, min_periods=12).std()
    ann_vol = vol * np.sqrt(6 * 365)
    # Target 20% vol per fully-on position; signal scales [0, 1]
    raw = sig / ann_vol.replace(0, np.nan) * 0.20
    raw = raw.fillna(0).clip(upper=1.0)
    return raw


def full_metrics(equity):
    m = compute_metrics(equity)
    ret = equity.pct_change().dropna()
    gains = ret[ret > 0].sum()
    pains = ret[ret < 0].abs().sum()
    m["gain_to_pain"] = float(gains / pains) if pains > 0 else np.inf
    p95 = ret.quantile(0.95)
    p05 = abs(ret.quantile(0.05))
    m["tail_ratio"] = float(p95 / p05) if p05 > 0 else np.inf
    try:
        monthly = ret.groupby(ret.index.to_period("M")).sum()
        m["monthly_hit_rate"] = float((monthly > 0).mean())
        m["worst_month"] = float(monthly.min())
        m["best_month"] = float(monthly.max())
    except Exception:
        m["monthly_hit_rate"] = np.nan
    return m


def yearly_breakdown(equity):
    """Year-by-year Sharpe and return."""
    ret = equity.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    years = sorted(ret.index.year.unique())
    rows = []
    for yr in years:
        yr_ret = ret[ret.index.year == yr]
        if len(yr_ret) < 100:
            continue
        ann_ret = yr_ret.mean() * 6 * 365
        ann_vol = yr_ret.std() * np.sqrt(6 * 365)
        sr = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = (1 + yr_ret).cumprod()
        max_dd = float((dd / dd.cummax() - 1).min())
        rows.append({"year": yr, "return": ann_ret, "vol": ann_vol, "sharpe": sr, "max_dd": max_dd})
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("  STRATEGY 8: OPTIMISED TSMOM")
    print("=" * 80)

    # ================================================================
    # Part 1: Vol target sweep
    # ================================================================
    print("\n--- Part 1: Vol Target Sweep (10 assets, Slow horizons) ---\n")
    close_wide, returns_wide = load_4h_topN(n=10)
    print(f"  {len(returns_wide):,} bars × {len(returns_wide.columns)} assets")
    print(f"  Assets: {list(returns_wide.columns)}")

    sig = tsmom_signal(returns_wide)
    w_base = vol_sized_weights(sig, returns_wide)

    results_vt = []
    for vol_target in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        w = apply_position_limit_wide(w_base.copy(), max_wt=0.30)
        w = apply_vol_targeting(w, returns_wide, vol_target=vol_target, lookback=42, max_leverage=3.0)
        bt = simple_backtest(w, returns_wide, cost_bps=COST_BPS)
        eq = bt.set_index("ts")["portfolio_equity"]
        m = full_metrics(eq)
        m["label"] = f"VT={vol_target:.0%}"
        results_vt.append(m)
        print(f"  VT={vol_target:.0%}:  CAGR={m['cagr']:.1%}  SR={m['sharpe']:.2f}  "
              f"MaxDD={m['max_dd']:.1%}  Sortino={m['sortino']:.2f}  Calmar={m['calmar']:.2f}  "
              f"Skew={m['skewness']:.2f}  Monthly_hit={m.get('monthly_hit_rate', 0):.0%}")

    print(f"\n{format_metrics_table(results_vt)}")

    # ================================================================
    # Part 2: Universe size sweep
    # ================================================================
    print("\n\n--- Part 2: Universe Size Sweep (VT=25%, Slow horizons) ---\n")
    results_univ = []
    for n_assets in [5, 10, 15, 20]:
        c, r = load_4h_topN(n=n_assets)
        s = tsmom_signal(r)
        w = vol_sized_weights(s, r)
        w = apply_position_limit_wide(w, max_wt=0.30)
        w = apply_vol_targeting(w, r, vol_target=0.25, lookback=42, max_leverage=3.0)
        bt = simple_backtest(w, r, cost_bps=COST_BPS)
        eq = bt.set_index("ts")["portfolio_equity"]
        m = full_metrics(eq)
        m["label"] = f"Top-{n_assets}"
        results_univ.append(m)
        print(f"  Top-{n_assets}: CAGR={m['cagr']:.1%}  SR={m['sharpe']:.2f}  "
              f"MaxDD={m['max_dd']:.1%}  Skew={m['skewness']:.2f}  "
              f"Calmar={m['calmar']:.2f}  Monthly={m.get('monthly_hit_rate', 0):.0%}")

    print(f"\n{format_metrics_table(results_univ)}")

    # ================================================================
    # Part 3: Best config — yearly breakdown
    # ================================================================
    print("\n\n--- Part 3: Production Config Year-by-Year ---")
    print("  Config: Top-10, VT=25%, Slow horizons, MaxLev=3.0\n")

    c10, r10 = load_4h_topN(n=10)
    s10 = tsmom_signal(r10)
    w10 = vol_sized_weights(s10, r10)
    w10 = apply_position_limit_wide(w10, max_wt=0.30)
    w10 = apply_vol_targeting(w10, r10, vol_target=0.25, lookback=42, max_leverage=3.0)

    bt10 = simple_backtest(w10, r10, cost_bps=COST_BPS)
    eq10 = bt10.set_index("ts")["portfolio_equity"]

    yb = yearly_breakdown(eq10)
    print(f"  {'Year':<8s} {'Return':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s}")
    print("  " + "-" * 40)
    for _, row in yb.iterrows():
        print(f"  {int(row['year']):<8d} {row['return']:>7.1%} {row['vol']:>7.1%} "
              f"{row['sharpe']:>8.2f} {row['max_dd']:>7.1%}")

    m_prod = full_metrics(eq10)
    print(f"\n  Full-period:")
    print(f"    CAGR:            {m_prod['cagr']:.1%}")
    print(f"    Sharpe:          {m_prod['sharpe']:.2f}")
    print(f"    Sortino:         {m_prod['sortino']:.2f}")
    print(f"    Calmar:          {m_prod['calmar']:.2f}")
    print(f"    Max DD:          {m_prod['max_dd']:.1%}")
    print(f"    Skewness:        {m_prod['skewness']:.3f}")
    print(f"    Tail ratio:      {m_prod['tail_ratio']:.3f}")
    print(f"    Gain/pain:       {m_prod['gain_to_pain']:.3f}")
    print(f"    Hit rate:        {m_prod['hit_rate']:.1%}")
    print(f"    Monthly hit:     {m_prod.get('monthly_hit_rate', 0):.1%}")
    print(f"    Worst month:     {m_prod.get('worst_month', 0):.1%}")
    print(f"    Best month:      {m_prod.get('best_month', 0):.1%}")

    bt10.to_parquet(RESULTS_DIR / "tsmom_production.parquet", index=False)

    # ================================================================
    # Part 4: Turnover and execution analysis
    # ================================================================
    print(f"\n\n--- Part 4: Execution Analysis ---")
    avg_turnover = bt10["turnover"].mean()
    avg_exposure = bt10["gross_exposure"].mean()
    avg_cost = bt10["cost_ret"].mean() * 6 * 365
    print(f"  Avg gross exposure:  {avg_exposure:.2f}")
    print(f"  Avg daily turnover:  {avg_turnover * 6:.4f}")
    print(f"  Ann. cost drag:      {avg_cost:.2%}")
    print(f"  Trades per day:      ~{avg_turnover * 6 / 0.01:.1f} (est. at 1% per trade)")


if __name__ == "__main__":
    main()
