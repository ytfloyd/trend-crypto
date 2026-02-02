#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_manifest_v0 import build_base_manifest, fingerprint_file, hash_config_blob, write_run_manifest
from transtrend_crypto_ma_5_40_fixed_universe_lib import (
    MA540FixedUniverseConfig,
    UNIVERSE,
    build_equal_weights,
    compute_signals,
    load_panel,
    simulate_portfolio,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto MA(5/40) fixed-universe baseline")
    p.add_argument("--db", type=str, default="../data/coinbase_daily_121025.duckdb")
    p.add_argument("--table", type=str, default="bars_1d_usd_universe_clean")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts/research/transtrend_crypto_ma_5_40_fixed_universe")
    p.add_argument("--cost_bps", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MA540FixedUniverseConfig(
        fast_ma=5,
        slow_ma=40,
        cost_bps=args.cost_bps,
        execution_lag_bars=1,
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

    config_blob = {
        "fast_ma": cfg.fast_ma,
        "slow_ma": cfg.slow_ma,
        "cost_bps": cfg.cost_bps,
        "execution_lag_bars": cfg.execution_lag_bars,
        "universe": UNIVERSE,
    }
    manifest = build_base_manifest("transtrend_crypto_ma_5_40_fixed_universe", sys.argv)
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
            "artifacts_written": [
                str(out_dir / "weights_signal.parquet"),
                str(out_dir / "weights_held.parquet"),
                str(out_dir / "equity.csv"),
                str(out_dir / "turnover.csv"),
            ],
        }
    )
    write_run_manifest(out_dir / "run_manifest.json", manifest)

    print(f"[transtrend_crypto_ma_5_40_fixed_universe] Wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
