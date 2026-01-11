#!/usr/bin/env python
from __future__ import annotations

"""
CLI runner for Growth Sleeve V1.5 (daily-only v0).

Loads bars from DuckDB (Top50 ADV>10M view), runs the Growth Sleeve backtest,
and writes standard artifacts (equity, weights, turnover, trades, debug).
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import duckdb
import pandas as pd

from alpha_ensemble_v15_growth_lib_v0 import GrowthSleeveConfig, run_growth_sleeve_backtest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Growth Sleeve V1.5 backtest (daily-only v0).")
    p.add_argument("--db", required=True, help="Path to DuckDB database.")
    p.add_argument(
        "--price_table",
        default="bars_1d_usd_universe_clean_top50_adv10m",
        help="Daily bars view/table to use.",
    )
    p.add_argument("--start", default="2023-01-01", help="Start date (inclusive, YYYY-MM-DD).")
    p.add_argument("--end", default="2024-12-31", help="End date (inclusive, YYYY-MM-DD).")
    p.add_argument("--out_dir", default="artifacts/research/alpha_ensemble_v15_growth", help="Output directory.")
    p.add_argument("--config_name", default="v0", help="Config name for filenames (default: v0).")

    # Key knobs
    p.add_argument("--adx_thresh", type=float, default=25.0)
    p.add_argument("--ichimoku_tenkan", type=int, default=9)
    p.add_argument("--ichimoku_kijun", type=int, default=26)
    p.add_argument("--ichimoku_senkou", type=int, default=52)
    p.add_argument("--ichimoku_disp", type=int, default=26)
    p.add_argument("--corr_threshold", type=float, default=0.7)
    p.add_argument("--cluster_cap", type=float, default=0.40)
    p.add_argument("--target_vol", type=float, default=0.20)
    p.add_argument("--max_scalar", type=float, default=1.5)
    p.add_argument("--max_name_weight", type=float, default=0.08)
    p.add_argument("--gap_atr_mult", type=float, default=1.0)
    p.add_argument("--slippage_bps", type=float, default=25.0)

    return p.parse_args()


def load_bars(db_path: Path, table: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily bars for all symbols in table within [start, end].
    """
    con = duckdb.connect(str(db_path))
    query = f"""
        SELECT ts, symbol, open, high, low, close, volume
        FROM {table}
        WHERE ts BETWEEN $1 AND $2
        ORDER BY ts, symbol
    """
    df = con.execute(query, [start, end]).fetch_df()
    con.close()
    if df.empty:
        raise SystemExit(f"No data returned from {table} between {start} and {end}")
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def build_config_from_args(args: argparse.Namespace) -> GrowthSleeveConfig:
    slippage = args.slippage_bps / 10000.0
    cfg_overrides: Dict[str, Any] = dict(
        adx_threshold=args.adx_thresh,
        ichimoku_tenkan=args.ichimoku_tenkan,
        ichimoku_kijun=args.ichimoku_kijun,
        ichimoku_senkou=args.ichimoku_senkou,
        ichimoku_disp=args.ichimoku_disp,
        corr_threshold=args.corr_threshold,
        cluster_cap=args.cluster_cap,
        target_vol=args.target_vol,
        max_scalar=args.max_scalar,
        max_name_weight=args.max_name_weight,
        gap_atr_mult=args.gap_atr_mult,
        gap_slippage=slippage,
    )
    return GrowthSleeveConfig(**cfg_overrides)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = load_bars(db_path, args.price_table, args.start, args.end)
    cfg = build_config_from_args(args)

    results = run_growth_sleeve_backtest(
        bars_df=bars,
        start_ts=pd.to_datetime(args.start),
        end_ts=pd.to_datetime(args.end),
        params=cfg.__dict__,
    )

    prefix = "growth"
    suffix = args.config_name

    equity_path = out_dir / f"{prefix}_equity_{suffix}.csv"
    weights_path = out_dir / f"{prefix}_weights_{suffix}.parquet"
    turnover_path = out_dir / f"{prefix}_turnover_{suffix}.csv"
    trades_path = out_dir / f"{prefix}_trades_{suffix}.parquet"
    debug_path = out_dir / f"{prefix}_debug_{suffix}.parquet"

    results["equity_df"].to_csv(equity_path, index=False)
    results["weights_df"].to_parquet(weights_path, index=False)
    # turnover from equity_df (already present)
    results["equity_df"][["ts", "turnover"]].to_csv(turnover_path, index=False)
    if not results["trades_df"].empty:
        results["trades_df"].to_parquet(trades_path, index=False)
    if not results["debug_df"].empty:
        results["debug_df"].to_parquet(debug_path, index=False)

    print(f"[growth_runner] Wrote equity to {equity_path}")
    print(f"[growth_runner] Wrote weights to {weights_path}")
    print(f"[growth_runner] Wrote turnover to {turnover_path}")
    if not results["trades_df"].empty:
        print(f"[growth_runner] Wrote trades to {trades_path}")
    if not results["debug_df"].empty:
        print(f"[growth_runner] Wrote debug to {debug_path}")

    # Quick diagnostics to avoid silent zero-trade runs
    weights = results["weights_df"]
    gross = weights.groupby("ts")["weight"].sum()
    debug_df = results["debug_df"]
    trades_df = results["trades_df"]
    print("\n[growth_runner] Diagnostics:")
    print(f"  Days with gross>0: {(gross.abs() > 1e-6).mean():.3f}")
    if not debug_df.empty:
        print(f"  Regime_on rate: {debug_df['regime_on'].mean():.3f}")
        print(f"  Slow_on rate:   {debug_df['slow_on'].mean():.3f}")
        print(f"  Fast_on rate:   {debug_df['fast_on'].mean():.3f}")
    print(f"  Entries: {len(trades_df[trades_df['side'].str.contains('ENTRY')]) if not trades_df.empty else 0}")
    print(f"  Exits:   {len(trades_df[trades_df['side'].str.contains('EXIT')]) if not trades_df.empty else 0}")
    print(f"  Traded symbols: {trades_df['symbol'].nunique() if not trades_df.empty else 0}")


if __name__ == "__main__":
    main()
