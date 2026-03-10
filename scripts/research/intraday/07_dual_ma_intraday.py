#!/usr/bin/env python3
"""
Strategy 7: Dual MA Crossover (4h) — the proven approach scaled down.

The repo already has strong results with MA 5/40 on daily bars.
This tests whether the same signal works at 4h (intraday-ish)
with appropriate parameter scaling.

MA fast/slow crossover with:
  - Vol-normalised sizing
  - Position limits
  - Multiple MA combinations tested
  - Top 10 liquid assets

This is the "boring but profitable" baseline against which all
exotic strategies should be compared.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import numpy as np
import pandas as pd

from common.data import load_bars
from common.backtest import simple_backtest
from common.metrics import compute_metrics, format_metrics_table
from common.risk_overlays import apply_vol_targeting, apply_position_limit_wide

RESULTS_DIR = Path(__file__).resolve().parent / "_results"
RESULTS_DIR.mkdir(exist_ok=True)
COST_BPS = 15.0


def load_4h_top(n=10, start="2020-01-01", end="2025-12-31"):
    panel = load_bars("4h", start=start, end=end)
    panel["dollar_vol"] = panel["close"] * panel["volume"]
    avg_dv = panel.groupby("symbol")["dollar_vol"].mean().sort_values(ascending=False)
    top = avg_dv.head(n).index.tolist()
    panel = panel[panel["symbol"].isin(top)]
    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    returns_wide = np.log(close_wide / close_wide.shift(1)).replace([np.inf, -np.inf], np.nan)
    return close_wide, returns_wide


def ma_crossover_signal(
    close_wide: pd.DataFrame,
    fast: int,
    slow: int,
) -> pd.DataFrame:
    """Binary signal: 1 when fast MA > slow MA, 0 otherwise."""
    ma_fast = close_wide.rolling(fast).mean()
    ma_slow = close_wide.rolling(slow).mean()
    return (ma_fast > ma_slow).astype(float)


def vol_normalised_weights(
    signal: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 42,
) -> pd.DataFrame:
    vol = returns_wide.rolling(vol_lookback, min_periods=12).std()
    raw = signal / vol.replace(0, np.nan)
    raw = raw.fillna(0)
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    return raw.div(row_sum, axis=0).fillna(0)


def convexity_metrics(equity: pd.Series) -> dict:
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


def main():
    print("=" * 70)
    print("  STRATEGY 7: DUAL MA CROSSOVER (4h)")
    print("=" * 70)

    close_wide, returns_wide = load_4h_top(n=10)
    print(f"  {len(returns_wide):,} bars × {len(returns_wide.columns)} assets")
    print(f"  Assets: {list(returns_wide.columns)}")

    # MA combos scaled from daily (×6 bars per day)
    configs = [
        {"fast": 18, "slow": 72, "name": "MA 3d/12d"},      # ~= daily 3/12
        {"fast": 30, "slow": 120, "name": "MA 5d/20d"},      # ~= daily 5/20
        {"fast": 30, "slow": 240, "name": "MA 5d/40d"},      # ~= daily 5/40 (repo standard)
        {"fast": 60, "slow": 240, "name": "MA 10d/40d"},
        {"fast": 60, "slow": 480, "name": "MA 10d/80d"},
        {"fast": 120, "slow": 600, "name": "MA 20d/100d"},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n--- {cfg['name']} (fast={cfg['fast']}, slow={cfg['slow']}) ---")
        sig = ma_crossover_signal(close_wide, cfg["fast"], cfg["slow"])
        w = vol_normalised_weights(sig, returns_wide)
        w = apply_position_limit_wide(w, max_wt=0.25)
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

        bt.to_parquet(RESULTS_DIR / f"ma_{cfg['name'].lower().replace(' ', '_').replace('/', '-')}.parquet", index=False)

    print(f"\n\n{'='*70}")
    print("  MA CROSSOVER SUMMARY")
    print(f"{'='*70}")
    print(format_metrics_table(all_results))


if __name__ == "__main__":
    main()
