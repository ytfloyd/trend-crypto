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
import sys

import duckdb
import pandas as pd

from alpha_ensemble_v15_growth_lib_v0 import GrowthSleeveConfig, run_growth_sleeve_backtest
from run_manifest_v0 import build_base_manifest, fingerprint_file, write_run_manifest, hash_config_blob


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
    equity_df = results["equity_df"]
    print("\n[growth_runner] Diagnostics:")
    print(f"  Days with gross>0: {(gross.abs() > 1e-6).mean():.3f}")
    if not debug_df.empty:
        print(f"  Regime_on rate: {debug_df['regime_on'].mean():.3f}")
        print(f"  Slow_on rate:   {debug_df['slow_on'].mean():.3f}")
        print(f"  Fast_on rate:   {debug_df['fast_on'].mean():.3f}")
    print(f"  Entries: {len(trades_df[trades_df['side'].str.contains('ENTRY')]) if not trades_df.empty else 0}")
    print(f"  Exits:   {len(trades_df[trades_df['side'].str.contains('EXIT')]) if not trades_df.empty else 0}")
    print(f"  Traded symbols: {trades_df['symbol'].nunique() if not trades_df.empty else 0}")

    # Exposure health summary CSV
    summary = {}
    if not equity_df.empty:
        summary["start"] = equity_df["ts"].min()
        summary["end"] = equity_df["ts"].max()
        summary["n_days"] = equity_df["ts"].nunique()
    summary["pct_gross_gt0"] = float((gross.abs() > 1e-6).mean())
    summary["gross_mean"] = float(gross.mean())
    summary["gross_median"] = float(gross.median())
    summary["gross_p10"] = float(gross.quantile(0.10))
    summary["gross_p90"] = float(gross.quantile(0.90))
    summary["gross_max"] = float(gross.max())

    active_counts = weights.groupby("ts")["symbol"].nunique()
    summary["active_mean"] = float(active_counts.mean())
    summary["active_median"] = float(active_counts.median())
    summary["active_p90"] = float(active_counts.quantile(0.90))
    summary["active_max"] = float(active_counts.max())

    if not debug_df.empty:
        vol_scalar_ts = debug_df.drop_duplicates(subset=["ts"])[["ts", "vol_scalar"]].set_index("ts")["vol_scalar"]
        summary["scalar_mean"] = float(vol_scalar_ts.mean())
        summary["scalar_median"] = float(vol_scalar_ts.median())
        summary["scalar_p10"] = float(vol_scalar_ts.quantile(0.10))
        summary["scalar_p90"] = float(vol_scalar_ts.quantile(0.90))
        summary["scalar_max"] = float(vol_scalar_ts.max())
        summary["scalar_pct_cap"] = float((vol_scalar_ts >= 0.999 * 1.5).mean())

        if "exp_vol_ann" in debug_df.columns:
            exp_vol_ts = debug_df.drop_duplicates(subset=["ts"])[["ts", "exp_vol_ann"]].set_index("ts")["exp_vol_ann"]
            summary["exp_vol_mean"] = float(exp_vol_ts.mean())
            summary["exp_vol_median"] = float(exp_vol_ts.median())
            summary["exp_vol_p10"] = float(exp_vol_ts.quantile(0.10))
            summary["exp_vol_p90"] = float(exp_vol_ts.quantile(0.90))
            summary["exp_vol_max"] = float(exp_vol_ts.max())

        summary["regime_on_rate"] = float(debug_df["regime_on"].mean())
        summary["slow_on_rate"] = float(debug_df["slow_on"].mean())
        summary["fast_on_rate"] = float(debug_df["fast_on"].mean())

    summary["entries"] = int(len(trades_df[trades_df["side"].str.contains("ENTRY")])) if not trades_df.empty else 0
    summary["exits"] = int(len(trades_df[trades_df["side"].str.contains("EXIT")])) if not trades_df.empty else 0
    summary["traded_symbols"] = int(trades_df["symbol"].nunique()) if not trades_df.empty else 0

    summary_path = out_dir / f"{prefix}_debug_summary_{suffix}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[growth_runner] Wrote exposure summary to {summary_path}")

    # Realized vol warning vs target
    if not equity_df.empty:
        ret = equity_df["portfolio_equity"].pct_change().dropna()
        realized_vol = ret.std() * (365 ** 0.5) if not ret.empty else 0.0
        target_vol = GrowthSleeveConfig().target_vol
        if realized_vol < 0.25 * target_vol and summary["pct_gross_gt0"] > 0.30:
            print(
                f"[growth_runner][WARNING] Realized vol {realized_vol:.4f} is <25% of target {target_vol:.4f} "
                "despite material gross exposure; check sizing/units."
            )

    # Write run manifest
    manifest = build_base_manifest(strategy_id="alpha_ensemble_v15_growth_sleeve", argv=sys.argv, repo_root=Path(__file__).resolve().parents[2])
    manifest.update(
        {
            "config": cfg.__dict__,
            "config_hash": hash_config_blob(cfg.__dict__),
            "data_sources": {
                "duckdb": fingerprint_file(db_path),
                "price_table": price_table,
            },
            "universe": price_table,
            "artifacts_written": {
                "equity_csv": str(equity_path),
                "weights_parquet": str(weights_path),
                "turnover_csv": str(turnover_path),
                "trades_parquet": str(trades_path),
                "debug_parquet": str(debug_path),
                "debug_summary_csv": str(summary_path),
            },
        }
    )
    manifest_path = out_dir / f"{prefix}_run_manifest_{suffix}.json"
    write_run_manifest(manifest_path, manifest)
    print(f"[growth_runner] Wrote run manifest to {manifest_path}")


if __name__ == "__main__":
    main()
