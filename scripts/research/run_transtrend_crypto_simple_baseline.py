#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from run_manifest_v0 import build_base_manifest, fingerprint_file, hash_config_blob, write_run_manifest
from timeseries_bundle_v0 import write_timeseries_bundle
from transtrend_crypto_simple_baseline_lib import (
    SimpleBaselineConfig,
    UNIVERSE,
    build_equal_weights,
    compute_signals,
    load_panel,
    simulate_portfolio,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto simple baseline (fixed universe)")
    p.add_argument("--db", type=str, default="../data/coinbase_daily_121025.duckdb")
    p.add_argument("--table", type=str, default="bars_1d_usd_universe_clean_adv10m")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts/research/transtrend_crypto_simple_baseline")
    p.add_argument("--cost_bps", type=float, default=0.0)
    p.add_argument("--execution_lag_bars", type=int, default=1)
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimpleBaselineConfig(
        fast_ma=20,
        slow_ma=100,
        cost_bps=args.cost_bps,
        execution_lag_bars=args.execution_lag_bars,
    )

    panel = load_panel(args.db, args.table, args.start, args.end)
    panel = compute_signals(panel, cfg.fast_ma, cfg.slow_ma)

    weights_signal = build_equal_weights(panel)
    equity_df, weights_held = simulate_portfolio(panel, weights_signal, cfg)

    weights_signal.to_parquet(out_dir / "weights_signal.parquet", index=False)
    weights_held.to_parquet(out_dir / "weights_held.parquet", index=False)
    equity_df.to_csv(out_dir / "equity.csv", index=False)
    equity_df[["ts", "turnover_one_sided", "turnover_two_sided"]].to_csv(
        out_dir / "turnover.csv", index=False
    )

    bundle_info = None
    if args.write_bundle:
        bars_df = panel[["ts", "symbol", "open", "high", "low", "close", "volume"]].copy()
        signals_df = panel[["ts", "symbol", "signal"]].copy()
        write_parquet = args.bundle_format in ("parquet", "both")
        write_csvgz = args.bundle_format in ("csvgz", "both", "parquet")
        bundle_info = write_timeseries_bundle(
            str(out_dir),
            bars_df=bars_df,
            signals_df=signals_df,
            weights_signal_df=weights_signal,
            weights_held_df=weights_held,
            portfolio_df=equity_df,
            write_parquet=write_parquet,
            write_csvgz=write_csvgz,
        )
        bundle_path = bundle_info.get("parquet") or bundle_info.get("csvgz")
        print(f"[transtrend_crypto_simple_baseline] Wrote timeseries bundle: {bundle_path} (rows={bundle_info.get('rows')})")

    config_blob = {
        "fast_ma": cfg.fast_ma,
        "slow_ma": cfg.slow_ma,
        "cost_bps": cfg.cost_bps,
        "execution_lag_bars": cfg.execution_lag_bars,
        "universe": UNIVERSE,
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

    manifest = build_base_manifest("transtrend_crypto_simple_baseline", sys.argv)
    manifest.update(
        {
            "config": config_blob,
            "config_hash": hash_config_blob(config_blob),
            "data_sources": {
                "duckdb": fingerprint_file(args.db, with_hash=False),
                "price_table": args.table,
            },
            "time_range": {"start": args.start, "end": args.end},
            "universe": UNIVERSE,
            "artifacts_written": artifacts_written,
        }
    )
    write_run_manifest(out_dir / "run_manifest.json", manifest)

    print(f"[transtrend_crypto_simple_baseline] Wrote artifacts to {out_dir}")

    # --- HTML tearsheet ---
    if not args.no_html:
        from tearsheet_common_v0 import build_standard_html_tearsheet, load_equity_csv
        strat_eq = load_equity_csv(str(out_dir / "equity.csv"))
        build_standard_html_tearsheet(
            out_html=out_dir / "tearsheet.html",
            strategy_label="Simple Baseline (Fixed Universe)",
            strategy_equity=strat_eq,
            equity_csv_path=str(out_dir / "equity.csv"),
            manifest_path=str(out_dir / "run_manifest.json"),
            subtitle="Simple dual MA crossover baseline on fixed cryptocurrency universe",
        )


if __name__ == "__main__":
    main()
