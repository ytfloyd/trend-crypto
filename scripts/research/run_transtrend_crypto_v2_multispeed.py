#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import duckdb
import pandas as pd

from run_manifest_v0 import build_base_manifest, fingerprint_file, hash_config_blob, write_run_manifest
from timeseries_bundle_v0 import write_timeseries_bundle
from transtrend_crypto_lib_v0 import HorizonSpec, TranstrendConfig
from transtrend_crypto_lib_v2_multispeed import combine_sleeves, run_sleeve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto v2 multi-speed runner")
    p.add_argument("--db", type=str, default="../data/coinbase_daily_121025.duckdb")
    p.add_argument("--table", type=str, default="bars_1d_usd_universe_clean_adv10m")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts/research/transtrend_crypto_v2_multispeed")
    p.add_argument("--target_vol_annual", type=float, default=0.20)
    p.add_argument("--danger_gross", type=float, default=0.25)
    p.add_argument("--cost_bps", type=float, default=20.0)
    p.add_argument("--cash_yield_annual", type=float, default=0.04)
    p.add_argument("--cash_buffer", type=float, default=0.05)
    p.add_argument("--max_gross", type=float, default=1.0)
    p.add_argument("--fast_weight", type=float, default=0.30)
    p.add_argument("--slow_weight", type=float, default=0.70)

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


def _fingerprint_db_rows(db_path: str, table: str) -> dict:
    con = duckdb.connect(db_path, read_only=True)
    try:
        count = con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()[0]
        min_max = con.execute(f"SELECT MIN(ts), MAX(ts) FROM {table}").fetchone()
        return {"rows": int(count), "ts_min": str(min_max[0]), "ts_max": str(min_max[1])}
    finally:
        con.close()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    fast_dir = out_dir / "fast"
    slow_dir = out_dir / "slow"
    combined_dir = out_dir / "combined"
    fast_dir.mkdir(parents=True, exist_ok=True)
    slow_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.db, args.table, args.start, args.end)

    fast_horizons = [HorizonSpec("fast", 10, 2, 20)]
    slow_horizons = [HorizonSpec("slow", 50, 10, 200)]

    cfg = TranstrendConfig(
        horizons=slow_horizons,
        target_vol_annual=args.target_vol_annual,
        danger_gross=args.danger_gross,
        cost_bps=args.cost_bps,
        fee_bps=args.cost_bps / 2.0,
        slippage_bps=args.cost_bps / 2.0,
        cash_yield_annual=args.cash_yield_annual,
        cash_buffer=args.cash_buffer,
        max_gross=args.max_gross,
        danger_btc_vol_threshold=args.danger_btc_vol_threshold,
        danger_btc_dd20_threshold=args.danger_btc_dd20_threshold,
        danger_btc_ret5_threshold=args.danger_btc_ret5_threshold,
    )

    fast_out = run_sleeve(panel, cfg, fast_horizons, "fast")
    slow_out = run_sleeve(panel, cfg, slow_horizons, "slow")

    fast_out["weights_signal"].to_parquet(fast_dir / "weights_signal.parquet", index=False)
    fast_out["weights_held"].to_parquet(fast_dir / "weights_held.parquet", index=False)
    fast_out["equity_df"].to_csv(fast_dir / "equity.csv", index=False)
    fast_out["equity_df"][["ts", "turnover_one_sided", "turnover_two_sided"]].to_csv(
        fast_dir / "turnover.csv", index=False
    )

    slow_out["weights_signal"].to_parquet(slow_dir / "weights_signal.parquet", index=False)
    slow_out["weights_held"].to_parquet(slow_dir / "weights_held.parquet", index=False)
    slow_out["equity_df"].to_csv(slow_dir / "equity.csv", index=False)
    slow_out["equity_df"][["ts", "turnover_one_sided", "turnover_two_sided"]].to_csv(
        slow_dir / "turnover.csv", index=False
    )

    combined_equity = combine_sleeves(
        fast_out["equity_df"],
        slow_out["equity_df"],
        args.fast_weight,
        args.slow_weight,
    )
    combined_equity.to_csv(combined_dir / "equity.csv", index=False)
    combined_equity[["ts", "turnover_one_sided", "turnover_two_sided"]].to_csv(
        combined_dir / "turnover.csv", index=False
    )

    fast_bundle = None
    slow_bundle = None
    combined_bundle = None
    if args.write_bundle:
        bars_df = panel[["ts", "symbol", "open", "high", "low", "close", "volume"]].copy()
        write_parquet = args.bundle_format in ("parquet", "both")
        write_csvgz = args.bundle_format in ("csvgz", "both", "parquet")
        fast_bundle = write_timeseries_bundle(
            str(fast_dir),
            bars_df=bars_df,
            weights_signal_df=fast_out["weights_signal"],
            weights_held_df=fast_out["weights_held"],
            portfolio_df=fast_out["equity_df"],
            write_parquet=write_parquet,
            write_csvgz=write_csvgz,
        )
        slow_bundle = write_timeseries_bundle(
            str(slow_dir),
            bars_df=bars_df,
            weights_signal_df=slow_out["weights_signal"],
            weights_held_df=slow_out["weights_held"],
            portfolio_df=slow_out["equity_df"],
            write_parquet=write_parquet,
            write_csvgz=write_csvgz,
        )
        combined_bundle = write_timeseries_bundle(
            str(combined_dir),
            bars_df=bars_df,
            portfolio_df=combined_equity,
            write_parquet=write_parquet,
            write_csvgz=write_csvgz,
        )
        print(
            f\"[transtrend_crypto_v2_multispeed] Wrote timeseries bundles: "
            f\"fast_rows={fast_bundle.get('rows')} slow_rows={slow_bundle.get('rows')} combined_rows={combined_bundle.get('rows')}\"
        )

    summary = {
        "fast_weight": args.fast_weight,
        "slow_weight": args.slow_weight,
        "target_vol_annual": args.target_vol_annual,
        "danger_gross": args.danger_gross,
        "cost_bps": args.cost_bps,
        "cash_yield_annual": args.cash_yield_annual,
        "cash_buffer": args.cash_buffer,
        "max_gross": args.max_gross,
        "fast_horizon": {"breakout_lookback": 10, "fast_ma": 2, "slow_ma": 20},
        "slow_horizon": {"breakout_lookback": 50, "fast_ma": 10, "slow_ma": 200},
    }
    (combined_dir / "combined_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    config_payload = {
        **summary,
        "danger_btc_vol_threshold": args.danger_btc_vol_threshold,
        "danger_btc_dd20_threshold": args.danger_btc_dd20_threshold,
        "danger_btc_ret5_threshold": args.danger_btc_ret5_threshold,
        "start": args.start,
        "end": args.end,
        "db": args.db,
        "table": args.table,
    }
    manifest = build_base_manifest("transtrend_crypto_v2_multispeed_spot", __import__("sys").argv)
    artifacts_written = {
        "fast": {
            "equity": str((fast_dir / "equity.csv").resolve()),
            "weights_signal": str((fast_dir / "weights_signal.parquet").resolve()),
            "weights_held": str((fast_dir / "weights_held.parquet").resolve()),
            "turnover": str((fast_dir / "turnover.csv").resolve()),
        },
        "slow": {
            "equity": str((slow_dir / "equity.csv").resolve()),
            "weights_signal": str((slow_dir / "weights_signal.parquet").resolve()),
            "weights_held": str((slow_dir / "weights_held.parquet").resolve()),
            "turnover": str((slow_dir / "turnover.csv").resolve()),
        },
        "combined": {
            "equity": str((combined_dir / "equity.csv").resolve()),
            "turnover": str((combined_dir / "turnover.csv").resolve()),
            "summary": str((combined_dir / "combined_summary.json").resolve()),
        },
    }
    if fast_bundle:
        if fast_bundle.get("parquet"):
            artifacts_written["fast"]["bundle_parquet"] = fast_bundle["parquet"]
        if fast_bundle.get("csvgz"):
            artifacts_written["fast"]["bundle_csvgz"] = fast_bundle["csvgz"]
    if slow_bundle:
        if slow_bundle.get("parquet"):
            artifacts_written["slow"]["bundle_parquet"] = slow_bundle["parquet"]
        if slow_bundle.get("csvgz"):
            artifacts_written["slow"]["bundle_csvgz"] = slow_bundle["csvgz"]
    if combined_bundle:
        if combined_bundle.get("parquet"):
            artifacts_written["combined"]["bundle_parquet"] = combined_bundle["parquet"]
        if combined_bundle.get("csvgz"):
            artifacts_written["combined"]["bundle_csvgz"] = combined_bundle["csvgz"]

    manifest.update(
        {
            "config": config_payload,
            "config_hash": hash_config_blob(config_payload),
            "data_sources": {
                "duckdb": fingerprint_file(args.db, with_hash=False),
                "price_table": args.table,
                "duckdb_rows": _fingerprint_db_rows(args.db, args.table),
            },
            "time_range": {"start": args.start, "end": args.end},
            "universe": args.table,
            "artifacts_written": artifacts_written,
        }
    )
    write_run_manifest(combined_dir / "run_manifest.json", manifest)

    print(f"[transtrend_crypto_v2_multispeed] Wrote {combined_dir}")

    # --- HTML tearsheet ---
    if not args.no_html:
        from tearsheet_common_v0 import build_standard_html_tearsheet, load_equity_csv
        strat_eq = load_equity_csv(str(combined_dir / "equity.csv"))
        build_standard_html_tearsheet(
            out_html=combined_dir / "tearsheet.html",
            strategy_label="Transtrend Crypto v2 Multi-Speed",
            strategy_equity=strat_eq,
            equity_csv_path=str(combined_dir / "equity.csv"),
            manifest_path=str(combined_dir / "run_manifest.json"),
            subtitle="Multi-speed trend-following with fast and slow MA crossover channels",
        )


if __name__ == "__main__":
    main()
