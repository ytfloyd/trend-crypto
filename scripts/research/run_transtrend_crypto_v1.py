#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

from run_manifest_v0 import (
    build_base_manifest,
    fingerprint_file,
    hash_config_blob,
    write_run_manifest,
)
from transtrend_crypto_lib_v1 import (
    HorizonSpec,
    TranstrendConfigV1,
    build_target_weights,
    compute_trend_scores,
    simulate_portfolio,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto v1 (spot long-only + ATR stops)")
    p.add_argument("--db", type=str, default="../data/coinbase_daily_121025.duckdb")
    p.add_argument("--table", type=str, default="bars_1d_usd_universe_clean_adv10m")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts/research/transtrend_crypto_v1")
    p.add_argument("--target_vol_annual", type=float, default=0.20)
    p.add_argument("--danger_gross", type=float, default=0.25)
    p.add_argument("--cost_bps", type=float, default=20.0)
    p.add_argument("--cash_yield_annual", type=float, default=0.04)
    p.add_argument("--cash_buffer", type=float, default=0.05)
    p.add_argument("--max_gross", type=float, default=1.0)
    p.add_argument("--danger_btc_vol_threshold", type=float, default=0.80)
    p.add_argument("--danger_btc_dd20_threshold", type=float, default=-0.20)
    p.add_argument("--danger_btc_ret5_threshold", type=float, default=-0.10)
    p.add_argument("--atr_window", type=int, default=20)
    p.add_argument("--atr_k", type=float, default=3.0)
    p.add_argument("--stop_cooldown_days", type=int, default=5)
    p.add_argument("--stop_use_atr_entry", type=bool, default=True)
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

    cfg = TranstrendConfigV1(
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
        atr_window=args.atr_window,
        atr_k=args.atr_k,
        stop_cooldown_days=args.stop_cooldown_days,
        stop_use_atr_entry=args.stop_use_atr_entry,
    )

    panel = load_panel(args.db, args.table, args.start, args.end)
    panel = compute_trend_scores(panel, cfg)

    weights_signal, danger, stop_levels, stop_events = build_target_weights(panel, cfg)
    equity_df, weights_held = simulate_portfolio(panel, weights_signal, cfg, danger)

    weights_signal.to_parquet(out_dir / "weights_signal.parquet", index=False)
    weights_held.to_parquet(out_dir / "weights_held.parquet", index=False)
    equity_df.to_csv(out_dir / "equity.csv", index=False)

    turnover_df = equity_df[["ts", "turnover_one_sided", "turnover_two_sided"]]
    turnover_df.to_csv(out_dir / "turnover.csv", index=False)

    stop_levels.to_parquet(out_dir / "stop_levels.parquet", index=False)
    stop_events.to_csv(out_dir / "stop_events.csv", index=False)

    config_blob = cfg.to_dict()
    config_hash = hash_config_blob(config_blob)

    manifest = build_base_manifest("transtrend_crypto_v1_spot_atr", sys.argv)
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
            "artifacts_written": [
                str(out_dir / "weights_signal.parquet"),
                str(out_dir / "weights_held.parquet"),
                str(out_dir / "equity.csv"),
                str(out_dir / "turnover.csv"),
                str(out_dir / "stop_levels.parquet"),
                str(out_dir / "stop_events.csv"),
            ],
        }
    )

    write_run_manifest(out_dir / "run_manifest.json", manifest)

    print(f"[transtrend_crypto_v1] Wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
