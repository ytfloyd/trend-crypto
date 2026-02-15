#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import duckdb
import pandas as pd

from run_manifest_v0 import build_base_manifest, fingerprint_file, hash_config_blob, write_run_manifest
from timeseries_bundle_v0 import write_timeseries_bundle
from universes import get_universe
from kuma_01_lib_v0 import (
    apply_dynamic_atr_trailing_stop,
    build_inverse_vol_weights,
    compute_atr30,
    compute_breakout_and_ma_filter,
    compute_vol31,
    simulate_portfolio,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run kuma_01 backtest (research-only).")
    p.add_argument("--db", default="../data/coinbase_daily_121025.duckdb", help="Path to DuckDB file.")
    p.add_argument(
        "--table",
        default="bars_1d_usd_universe_clean",
        help="Table/view with daily bars.",
    )
    p.add_argument("--start", required=True, help="Start date (inclusive, e.g., 2017-01-01)")
    p.add_argument("--end", required=True, help="End date (inclusive, e.g., 2025-01-01)")
    p.add_argument(
        "--out_dir",
        default="artifacts/research/kuma_01",
        help="Output directory for artifacts.",
    )
    p.add_argument("--cost_bps", type=float, default=20.0, help="Cost in bps per turnover.")
    p.add_argument("--universe", type=str, default="kuma_live_universe")
    p.add_argument("--universe_lookback_days", type=int, default=7)
    p.add_argument("--write_bundle", action="store_true", default=True, help="Write timeseries bundle.")
    p.add_argument("--no_write_bundle", action="store_false", dest="write_bundle", help="Disable bundle.")
    p.add_argument(
        "--bundle_format",
        choices=["parquet", "csvgz", "both"],
        default="csvgz",
        help="Bundle output format.",
    )
    p.add_argument("--no_html", action="store_true", help="Skip HTML tearsheet generation.")
    return p.parse_args()


def load_panel(db_path: str, table: str, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not symbols:
        raise ValueError("symbols list is empty")
    placeholders = ",".join(["?"] * len(symbols))
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE symbol IN ({placeholders})
              AND ts >= ? AND ts <= ?
            ORDER BY ts, symbol
            """,
            symbols + [start, end],
        ).fetch_df()
    finally:
        con.close()

    required = {"symbol", "ts", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["open", "close"])
    return df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = get_universe(
        args.universe,
        db_path=args.db,
        table=args.table,
        lookback_days=args.universe_lookback_days,
    )

    panel = load_panel(args.db, args.table, symbols, args.start, args.end)
    panel = compute_breakout_and_ma_filter(panel)
    panel = compute_atr30(panel)
    panel = compute_vol31(panel)
    panel = apply_dynamic_atr_trailing_stop(panel)

    weights_signal = build_inverse_vol_weights(panel)
    equity_df, weights_held = simulate_portfolio(panel, weights_signal, cost_bps=args.cost_bps, execution_lag_bars=1)

    weights_signal.to_parquet(out_dir / "weights_signal.parquet", index=False)
    weights_held.to_parquet(out_dir / "weights_held.parquet", index=False)
    equity_df.to_csv(out_dir / "equity.csv", index=False)
    equity_df[["ts", "turnover_one_sided", "turnover_two_sided"]].to_csv(
        out_dir / "turnover.csv", index=False
    )

    bundle_info = None
    if args.write_bundle:
        bars_df = panel[["ts", "symbol", "open", "high", "low", "close", "volume"]].copy()
        features_df = panel[
            [
                "ts",
                "symbol",
                "breakout20",
                "sma5",
                "sma40",
                "trend_ok",
                "entry_signal",
                "atr30",
                "vol31",
                "stop_level",
                "stop_hit",
                "stop_block",
            ]
        ].copy()
        signals_df = panel[["ts", "symbol", "entry_signal"]].copy()
        stops_df = panel[["ts", "symbol", "stop_level", "stop_hit", "stop_block"]].copy()
        write_parquet = args.bundle_format in ("parquet", "both")
        write_csvgz = args.bundle_format in ("csvgz", "both", "parquet")
        bundle_info = write_timeseries_bundle(
            str(out_dir),
            bars_df=bars_df,
            features_df=features_df,
            signals_df=signals_df,
            weights_signal_df=weights_signal,
            weights_held_df=weights_held,
            stops_df=stops_df,
            portfolio_df=equity_df,
            write_parquet=write_parquet,
            write_csvgz=write_csvgz,
        )
        bundle_path = bundle_info.get("parquet") or bundle_info.get("csvgz")
        print(f"[kuma_01] Wrote timeseries bundle: {bundle_path} (rows={bundle_info.get('rows')})")

    config_blob = {
        "breakout_lookback": 20,
        "fast_ma": 5,
        "slow_ma": 40,
        "atr_window": 30,
        "atr_k": 3.0,
        "vol_window": 31,
        "cost_bps": args.cost_bps,
        "universe": args.universe,
        "resolved_universe": symbols,
        "universe_lookback_days": args.universe_lookback_days,
        "execution_lag_bars": 1,
    }
    artifacts_written = [
        str(out_dir / "weights_signal.parquet"),
        str(out_dir / "weights_held.parquet"),
        str(out_dir / "equity.csv"),
        str(out_dir / "turnover.csv"),
    ]
    if bundle_info:
        if bundle_info.get("parquet"):
            artifacts_written.append(bundle_info["parquet"])
        if bundle_info.get("csvgz"):
            artifacts_written.append(bundle_info["csvgz"])

    manifest = build_base_manifest("kuma_01_v0", sys.argv)
    manifest.update(
        {
            "config": config_blob,
            "config_hash": hash_config_blob(config_blob),
            "data_sources": {
                "duckdb": fingerprint_file(args.db, with_hash=False),
                "price_table": args.table,
            },
            "time_range": {"start": args.start, "end": args.end},
            "universe": symbols,
            "artifacts_written": artifacts_written,
        }
    )
    write_run_manifest(out_dir / "run_manifest.json", manifest)

    print(f"[kuma_01] Wrote artifacts to {out_dir}")

    # --- HTML tearsheet ---
    if not args.no_html:
        from tearsheet_common_v0 import build_standard_html_tearsheet, load_equity_csv
        strat_eq = load_equity_csv(str(out_dir / "equity.csv"))
        build_standard_html_tearsheet(
            out_html=out_dir / "tearsheet.html",
            strategy_label="kuma_01 (Breakout + MA5/40 + ATR30 Stop)",
            strategy_equity=strat_eq,
            equity_csv_path=str(out_dir / "equity.csv"),
            manifest_path=str(out_dir / "run_manifest.json"),
            subtitle="Breakout20 + MA5/40; dynamic ATR30 stop (3x); inverse vol31 sizing",
        )


if __name__ == "__main__":
    main()
