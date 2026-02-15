#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import duckdb
import pandas as pd

from run_manifest_v0 import (
    build_base_manifest,
    fingerprint_file,
    hash_config_blob,
    write_run_manifest,
)
from timeseries_bundle_v0 import write_timeseries_bundle
from transtrend_crypto_lib_v0 import (
    HorizonSpec,
    TranstrendConfig,
    build_target_weights,
    compute_trend_scores,
    simulate_portfolio,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto v0 (spot long-only)")
    p.add_argument("--db", type=str, default="../data/coinbase_daily_121025.duckdb")
    p.add_argument("--table", type=str, default="bars_1d_usd_universe_clean_adv10m")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts/research/transtrend_crypto_v0")
    p.add_argument("--target_vol_annual", type=float, default=0.20)
    p.add_argument("--danger_gross", type=float, default=0.25)
    p.add_argument("--cost_bps", type=float, default=20.0)
    p.add_argument("--cash_yield_annual", type=float, default=0.04)
    p.add_argument("--cash_buffer", type=float, default=0.05)
    p.add_argument("--max_gross", type=float, default=1.0)
    p.add_argument("--danger_btc_vol_threshold", type=float, default=0.80)
    p.add_argument("--danger_btc_dd20_threshold", type=float, default=-0.20)
    p.add_argument("--danger_btc_ret5_threshold", type=float, default=-0.10)
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


def load_panel(db_path: str, table: str, start: str, end: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE ts >= ? AND ts <= ?
            ORDER BY ts, symbol
            """,
            [start, end],
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

    horizons = [
        HorizonSpec("fast", 10, 2, 20),
        HorizonSpec("mid", 20, 5, 40),
        HorizonSpec("slow", 50, 10, 200),
    ]

    fee_bps = args.cost_bps / 2.0
    slippage_bps = args.cost_bps / 2.0

    cfg = TranstrendConfig(
        horizons=horizons,
        target_vol_annual=args.target_vol_annual,
        danger_gross=args.danger_gross,
        cost_bps=args.cost_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cash_yield_annual=args.cash_yield_annual,
        cash_buffer=args.cash_buffer,
        max_gross=args.max_gross,
        danger_btc_vol_threshold=args.danger_btc_vol_threshold,
        danger_btc_dd20_threshold=args.danger_btc_dd20_threshold,
        danger_btc_ret5_threshold=args.danger_btc_ret5_threshold,
    )

    panel = load_panel(args.db, args.table, args.start, args.end)
    panel = compute_trend_scores(panel, cfg)

    weights_signal, danger = build_target_weights(panel, cfg)
    equity_df, weights_held = simulate_portfolio(panel, weights_signal, cfg, danger)

    weights_signal.to_parquet(out_dir / "weights_signal.parquet", index=False)
    weights_held.to_parquet(out_dir / "weights_held.parquet", index=False)
    equity_df.to_csv(out_dir / "equity.csv", index=False)

    turnover_df = equity_df[["ts", "turnover_one_sided", "turnover_two_sided"]]
    turnover_df.to_csv(out_dir / "turnover.csv", index=False)

    bundle_info = None
    if args.write_bundle:
        bars_df = panel[["ts", "symbol", "open", "high", "low", "close", "volume"]].copy()
        features_df = panel[["ts", "symbol", "score", "vol_ann"]].copy()
        write_parquet = args.bundle_format in ("parquet", "both")
        write_csvgz = args.bundle_format in ("csvgz", "both", "parquet")
        bundle_info = write_timeseries_bundle(
            str(out_dir),
            bars_df=bars_df,
            features_df=features_df,
            weights_signal_df=weights_signal,
            weights_held_df=weights_held,
            portfolio_df=equity_df,
            write_parquet=write_parquet,
            write_csvgz=write_csvgz,
        )
        bundle_path = bundle_info.get("parquet") or bundle_info.get("csvgz")
        print(f"[transtrend_crypto_v0] Wrote timeseries bundle: {bundle_path} (rows={bundle_info.get('rows')})")

    config_blob = cfg.to_dict()
    config_hash = hash_config_blob(config_blob)

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

    manifest = build_base_manifest("transtrend_crypto_v0_spot", sys.argv)
    manifest.update(
        {
            "config": config_blob,
            "config_hash": config_hash,
            "data_sources": {
                "duckdb": fingerprint_file(args.db, with_hash=False),
                "price_table": args.table,
            },
            "time_range": {"start": args.start, "end": args.end},
            "universe": args.table,
            "artifacts_written": artifacts_written,
        }
    )

    write_run_manifest(out_dir / "run_manifest.json", manifest)

    print(f"[transtrend_crypto_v0] Wrote artifacts to {out_dir}")

    # --- HTML tearsheet ---
    if not args.no_html:
        from tearsheet_common_v0 import build_standard_html_tearsheet, load_equity_csv
        strat_eq = load_equity_csv(str(out_dir / "equity.csv"))
        build_standard_html_tearsheet(
            out_html=out_dir / "tearsheet.html",
            strategy_label="Transtrend Crypto v0 (Spot Long-Only)",
            strategy_equity=strat_eq,
            equity_csv_path=str(out_dir / "equity.csv"),
            manifest_path=str(out_dir / "run_manifest.json"),
            subtitle="Spot long-only trend-following with dual MA crossover",
        )


if __name__ == "__main__":
    main()
