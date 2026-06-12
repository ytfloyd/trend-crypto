"""Run the FLAGSHIP medallion_lite strategy and measure Sortino vs BTC buy-and-hold.

Drives the real pipeline (scripts/research/medallion_lite: 5-factor composite +
ensemble regime + event-driven portfolio + 30bps costs) against the coinbase
hourly lake, resamples net returns to DAILY, and computes Sortino on the FULL
sample and OUT-OF-SAMPLE (2023+) for both the strategy and BTC B&H.

Run:  PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_sortino.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.metrics import compute_metrics  # noqa: E402
from medallion_lite.factors import compute_composite_score, compute_factors  # noqa: E402
from medallion_lite.portfolio import (  # noqa: E402
    backtest_portfolio,
    build_factor_portfolio,
    performance_summary,
)
from medallion_lite.regime_ensemble import compute_ensemble_regime  # noqa: E402
from sornette_lppl.data_hf import filter_universe_hourly, load_hourly_bars  # noqa: E402

LAKE = str(ROOT.parent / "data" / "coinbase_crypto_ohlcv_lake.duckdb")
START, END = "2021-01-01", "2026-06-01"
TC_BPS = 30.0


def daily_metrics(hourly_ret: pd.Series, lo: str | None) -> dict:
    """Resample an hourly return series to daily, slice, and compute Sortino/Sharpe."""
    r = hourly_ret.copy()
    r.index = pd.to_datetime(r.index)
    daily = (1.0 + r.fillna(0.0)).resample("D").prod() - 1.0
    if lo:
        daily = daily[daily.index >= lo]
    daily = daily.dropna()
    if len(daily) < 30:
        return {"sortino": np.nan, "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan}
    eq = (1.0 + daily).cumprod()
    m = compute_metrics(eq)
    return {k: m[k] for k in ("sortino", "sharpe", "cagr", "max_dd")}


def main() -> None:
    print(f"loading hourly bars {START}..{END} from {Path(LAKE).name} ...")
    panel = load_hourly_bars(db_path=LAKE, start=START, end=END)
    panel = filter_universe_hourly(panel)
    panel = panel[panel["in_universe"]].copy()
    print(f"  {panel['symbol'].nunique()} symbols in universe")

    df = panel.sort_values(["symbol", "ts"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
    close_wide = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    high_wide = df.pivot(index="ts", columns="symbol", values="high").sort_index()
    volume_wide = df.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)

    if "BTC-USD" in panel["symbol"].values:
        btc_h = panel[panel["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
        regime = compute_ensemble_regime(btc_h.resample("D").last().dropna(), btc_h, returns_wide)
    else:
        regime = pd.Series(1.0, index=returns_wide.index)

    factors = compute_factors(close_wide, volume_wide, high_wide)
    composite = compute_composite_score(factors)
    weights, trades = build_factor_portfolio(
        composite, returns_wide, regime,
        entry_threshold=0.65, exit_score_threshold=0.40, max_hold_hours=336,
        trailing_stop_pct=0.15, rebalance_every_hours=24, max_positions=25, max_weight=0.10,
    )
    bt = backtest_portfolio(weights, returns_wide, tc_bps=TC_BPS)
    stats = performance_summary(bt)

    net = pd.Series(bt["net_ret"].values, index=pd.to_datetime(bt["ts"]))
    btc = returns_wide.get("BTC-USD", pd.Series(0.0, index=returns_wide.index))
    btc.index = pd.to_datetime(btc.index)

    print(f"\n  (hourly Sharpe from pipeline: {stats.get('sharpe', float('nan')):.2f}, "
          f"max_dd {stats.get('max_dd', float('nan')):.1%})\n")
    hdr = f"{'series':<22} {'Sortino':>8} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}"
    for lo, name in ((None, "FULL 2021-2026"), ("2023-01-01", "OOS 2023-2026")):
        print(f"=== {name} (daily) ===")
        print(hdr)
        for label, series in (("Medallion Lite", net), ("BTC buy&hold", btc)):
            m = daily_metrics(series, lo)
            print(f"{label:<22} {m['sortino']:>8.2f} {m['sharpe']:>8.2f} {m['cagr']:>8.0%} {m['max_dd']:>8.0%}")
        print()


if __name__ == "__main__":
    main()
