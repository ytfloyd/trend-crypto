#!/usr/bin/env python3
"""
Generate an interactive HTML tear sheet for the Sornette LPPLS hourly system.

Reads pre-computed backtest artifacts and produces a comprehensive,
institutional-quality tear sheet via the shared HTML tearsheet generator.

Usage:
    python -m scripts.research.sornette_lppl.generate_tearsheet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from tearsheet_common_v0 import (  # noqa: E402
    build_standard_html_tearsheet,
    compute_comprehensive_stats,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
ANN_FACTOR_HOURLY = 365.0 * 24


def _load_hf_backtest() -> pd.DataFrame:
    path = OUT_DIR / "hf_backtest.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Run the hourly backtest first: {path}\n"
            "  python -m scripts.research.sornette_lppl.run_hf_backtest"
        )
    bt = pd.read_parquet(path)
    bt["ts"] = pd.to_datetime(bt["ts"])
    return bt


def _build_daily_equity(bt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Resample hourly backtest to daily equity + turnover for tearsheet."""
    bt = bt.set_index("ts").sort_index()

    daily_ret = (1 + bt["net_ret"]).resample("D").prod() - 1.0
    daily_ret = daily_ret.dropna()
    daily_ret = daily_ret[daily_ret.index >= bt.index[0]]

    equity = (1 + daily_ret).cumprod()
    equity.name = "equity"

    daily_turnover = bt["turnover"].resample("D").sum()
    daily_turnover = daily_turnover.reindex(equity.index).fillna(0.0)
    daily_turnover.name = "turnover"

    return equity, daily_turnover


def _build_btc_benchmark(equity_index: pd.DatetimeIndex) -> pd.Series | None:
    """Load BTC buy & hold from the cached hourly bars."""
    cache_dir = Path(__file__).resolve().parent / "_cache"
    candidates = sorted(cache_dir.glob("bars_1h_*.parquet"))
    if not candidates:
        print("[tearsheet] No cached hourly bars found for BTC benchmark.")
        return None

    bars = pd.read_parquet(candidates[0])
    btc = bars[bars["symbol"] == "BTC-USD"].copy()
    if btc.empty:
        print("[tearsheet] No BTC-USD data in cache.")
        return None

    btc["ts"] = pd.to_datetime(btc["ts"])
    btc = btc.sort_values("ts").set_index("ts")

    btc_daily = btc["close"].resample("D").last().dropna()
    btc_ret = btc_daily.pct_change().dropna()

    common = btc_ret.index.intersection(equity_index)
    if len(common) < 10:
        print("[tearsheet] Insufficient BTC overlap.")
        return None

    btc_ret = btc_ret.reindex(equity_index).fillna(0.0)
    btc_eq = (1 + btc_ret).cumprod()
    btc_eq.name = "benchmark_equity"
    return btc_eq


def _patch_log_scale(html_path: Path) -> None:
    """Patch the equity chart to use log scale (60x vs 2.3x needs it)."""
    text = html_path.read_text(encoding="utf-8")
    old = '"title": "Equity"}, "legend"'
    new = '"title": "Equity (log)", "type": "log"}, "legend"'
    if old in text:
        text = text.replace(old, new, 1)
        html_path.write_text(text, encoding="utf-8")
        print("[patch] Equity chart set to log scale.")
    else:
        print("[patch] Could not find equity yaxis to patch for log scale.")


def main():
    print("=" * 60)
    print("SORNETTE LPPLS HOURLY — TEAR SHEET GENERATOR")
    print("=" * 60)

    bt = _load_hf_backtest()
    print(f"Loaded {len(bt):,} hourly bars  ({bt['ts'].min()} → {bt['ts'].max()})")

    equity, turnover = _build_daily_equity(bt)
    print(f"Daily equity: {len(equity)} days  ({equity.index[0].date()} → {equity.index[-1].date()})")

    bench = _build_btc_benchmark(equity.index)
    if bench is not None:
        print(f"BTC benchmark loaded: {len(bench)} days")

    stats = compute_comprehensive_stats(
        equity=equity,
        benchmark_equity=bench,
        turnover=turnover,
    )

    out_html = OUT_DIR / "sornette_lppl_hourly_tearsheet.html"

    build_standard_html_tearsheet(
        out_html=out_html,
        strategy_label="Sornette LPPLS Hourly",
        strategy_equity=equity,
        strategy_stats=stats,
        benchmark_equity=bench,
        benchmark_label="BTC Buy & Hold",
        equity_csv_path=str(OUT_DIR / "hf_backtest.parquet"),
        metrics_csv_path=str(OUT_DIR / "hf_robustness.json"),
        subtitle=(
            "Hourly bubble-riding strategy using super-exponential detection "
            "and LPPLS tc-based exit timing. 30 bps one-way costs, "
            "top-10 holdings, BTC dual-SMA regime filter."
        ),
        turnover=turnover,
        auto_open=False,
    )

    _patch_log_scale(out_html)

    print(f"\nTear sheet saved: {out_html}")
    print(f"Size: {out_html.stat().st_size / 1024:.0f} KB")

    import webbrowser
    webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
