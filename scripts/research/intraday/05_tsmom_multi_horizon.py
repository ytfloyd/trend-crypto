#!/usr/bin/env python3
"""
Strategy 5: Time-Series Momentum across multiple horizons (4h)

The most well-documented anomaly in crypto: trend persistence.
Strong academic evidence (Moskowitz, Ooi, Pedersen 2012) and
crypto-specific (Liu & Tsyvinski 2021).

Design:
  - 4h bars (less noise than 1h, still intraday)
  - Top 10 liquid assets only (concentrated bets)
  - Multi-horizon momentum: blend 1d, 3d, 7d, 14d signals
  - Vol-normalised position sizing (1/vol per asset)
  - Gentle vol targeting (20%) and position limits (25%)
  - NO aggressive DD control (proven to create death spirals)

Each asset gets a signal = weighted average of normalised returns
at 4 horizons. Signal is mapped to position [-1, +1] via sigmoid.
Long-only constraint applied.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import numpy as np
import pandas as pd

from common.data import load_bars, filter_universe, ANN_FACTOR
from common.backtest import simple_backtest
from common.metrics import compute_metrics, format_metrics_table
from common.risk_overlays import apply_vol_targeting, apply_position_limit_wide

RESULTS_DIR = Path(__file__).resolve().parent / "_results"
RESULTS_DIR.mkdir(exist_ok=True)
COST_BPS = 15.0


def load_4h_top_liquid(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    n_assets: int = 10,
    min_adv: float = 10_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 4h bars for the top N most liquid assets."""
    panel = load_bars("4h", start=start, end=end)

    # Compute average dollar volume per symbol
    panel["dollar_vol"] = panel["close"] * panel["volume"]
    avg_dv = panel.groupby("symbol")["dollar_vol"].mean().sort_values(ascending=False)
    top_syms = avg_dv[avg_dv > min_adv].head(n_assets).index.tolist()

    panel = panel[panel["symbol"].isin(top_syms)]
    print(f"  Universe: {top_syms}")

    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    returns_wide = np.log(close_wide / close_wide.shift(1)).replace([np.inf, -np.inf], np.nan)

    return panel, returns_wide, close_wide


def tsmom_signal(
    returns_wide: pd.DataFrame,
    horizons: dict[int, float] = None,
) -> pd.DataFrame:
    """Multi-horizon time-series momentum signal.

    For each asset, compute normalised return (return / vol) at each horizon,
    then blend with given weights. Clip to [-1, 1].
    """
    if horizons is None:
        # Horizons in 4h bars: 6=1d, 18=3d, 42=7d, 84=14d
        horizons = {6: 0.15, 18: 0.25, 42: 0.35, 84: 0.25}

    vol = returns_wide.rolling(42, min_periods=12).std()
    signal = pd.DataFrame(0.0, index=returns_wide.index, columns=returns_wide.columns)

    for h, w in horizons.items():
        cum_ret = returns_wide.rolling(h).sum()
        norm_ret = cum_ret / (vol * np.sqrt(h)).replace(0, np.nan)
        signal += w * norm_ret.fillna(0)

    # Sigmoid scaling
    signal = 2 / (1 + np.exp(-signal)) - 1
    return signal


def position_sizing(
    signal: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 42,
    target_vol_per_asset: float = 0.02,  # ~15% annualised per asset (4h)
) -> pd.DataFrame:
    """Convert signals to vol-normalised weights (long-only)."""
    # Only go long
    sig = signal.clip(lower=0)

    # Vol-normalise: each position sized so its contribution = target_vol
    vol = returns_wide.rolling(vol_lookback, min_periods=12).std()
    ann_vol = vol * np.sqrt(6 * 365)  # 6 bars per day * 365 days
    vol_scalar = (target_vol_per_asset / ann_vol).clip(upper=5.0).fillna(0)

    weights = sig * vol_scalar
    return weights


def convexity_metrics(equity: pd.Series) -> dict:
    """Extended metrics focused on convexity."""
    m = compute_metrics(equity)
    ret = equity.pct_change().dropna()

    gains = ret[ret > 0].sum()
    pains = ret[ret < 0].abs().sum()
    m["gain_to_pain"] = float(gains / pains) if pains > 0 else np.inf

    p95 = ret.quantile(0.95)
    p05 = abs(ret.quantile(0.05))
    m["tail_ratio"] = float(p95 / p05) if p05 > 0 else np.inf

    dd = equity / equity.cummax() - 1
    m["pct_time_in_dd"] = float((dd < -0.01).mean())

    # Monthly returns for consistency check
    monthly = ret.resample("ME").sum() if hasattr(ret.index, 'freq') or True else ret
    try:
        monthly = ret.groupby(ret.index.to_period("M")).sum()
        m["monthly_hit_rate"] = float((monthly > 0).mean())
        m["worst_month"] = float(monthly.min())
        m["best_month"] = float(monthly.max())
    except Exception:
        m["monthly_hit_rate"] = np.nan
        m["worst_month"] = np.nan
        m["best_month"] = np.nan

    return m


def main():
    print("=" * 70)
    print("  STRATEGY 5: TIME-SERIES MOMENTUM (4h)")
    print("=" * 70)

    panel, returns_wide, close_wide = load_4h_top_liquid(n_assets=10)
    print(f"  {len(returns_wide):,} bars × {len(returns_wide.columns)} assets")

    # ---- Configuration sweep ----
    configs = [
        {
            "name": "TSMOM Fast (1-7d)",
            "horizons": {6: 0.30, 18: 0.40, 42: 0.30},
        },
        {
            "name": "TSMOM Balanced (1-14d)",
            "horizons": {6: 0.15, 18: 0.25, 42: 0.35, 84: 0.25},
        },
        {
            "name": "TSMOM Slow (3-28d)",
            "horizons": {18: 0.20, 42: 0.30, 84: 0.30, 168: 0.20},
        },
        {
            "name": "TSMOM Ultra-Slow (7-56d)",
            "horizons": {42: 0.25, 84: 0.35, 168: 0.25, 336: 0.15},
        },
    ]

    all_results = []
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        sig = tsmom_signal(returns_wide, horizons=cfg["horizons"])
        w_raw = position_sizing(sig, returns_wide)

        # Light risk overlays only
        w = apply_position_limit_wide(w_raw, max_wt=0.25)
        w = apply_vol_targeting(w, returns_wide, vol_target=0.20, lookback=42, max_leverage=2.0)

        bt = simple_backtest(w, returns_wide, cost_bps=COST_BPS)
        eq = bt.set_index("ts")["portfolio_equity"]
        m = convexity_metrics(eq)
        m["label"] = cfg["name"]

        print(f"  CAGR={m['cagr']:.1%}  Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']:.1%}  "
              f"Sortino={m['sortino']:.2f}  Skew={m['skewness']:.2f}  Calmar={m['calmar']:.2f}")
        print(f"  Tail ratio={m['tail_ratio']:.2f}  Gain/pain={m['gain_to_pain']:.2f}  "
              f"Monthly hit={m.get('monthly_hit_rate', 0):.1%}")

        all_results.append(m)

        # Save best
        bt.to_parquet(RESULTS_DIR / f"tsmom_{cfg['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.parquet", index=False)

    print(f"\n\n{'='*70}")
    print("  TSMOM SUMMARY")
    print(f"{'='*70}")
    print(format_metrics_table(all_results))

    # ---- Ensemble: blend all horizons ----
    print(f"\n\n--- TSMOM Ensemble (all horizons blended) ---")
    all_horizons = {6: 0.10, 18: 0.15, 42: 0.25, 84: 0.25, 168: 0.15, 336: 0.10}
    sig_ens = tsmom_signal(returns_wide, horizons=all_horizons)
    w_ens = position_sizing(sig_ens, returns_wide)
    w_ens = apply_position_limit_wide(w_ens, max_wt=0.25)
    w_ens = apply_vol_targeting(w_ens, returns_wide, vol_target=0.20, lookback=42, max_leverage=2.0)

    bt_ens = simple_backtest(w_ens, returns_wide, cost_bps=COST_BPS)
    eq_ens = bt_ens.set_index("ts")["portfolio_equity"]
    m_ens = convexity_metrics(eq_ens)
    m_ens["label"] = "TSMOM Ensemble"

    print(f"  CAGR={m_ens['cagr']:.1%}  Sharpe={m_ens['sharpe']:.2f}  MaxDD={m_ens['max_dd']:.1%}  "
          f"Sortino={m_ens['sortino']:.2f}  Skew={m_ens['skewness']:.2f}  Calmar={m_ens['calmar']:.2f}")
    print(f"  Tail ratio={m_ens['tail_ratio']:.2f}  Gain/pain={m_ens['gain_to_pain']:.2f}  "
          f"Monthly hit={m_ens.get('monthly_hit_rate', 0):.1%}")

    bt_ens.to_parquet(RESULTS_DIR / "tsmom_ensemble.parquet", index=False)


if __name__ == "__main__":
    main()
