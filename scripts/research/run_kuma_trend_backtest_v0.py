#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import duckdb
import pandas as pd

from kuma_trend_lib_v0 import KumaConfig, run_kuma_trend_backtest
from run_manifest_v0 import build_base_manifest, fingerprint_file, hash_config_blob, write_run_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run kuma_trend breakout trend-following backtest.")
    p.add_argument("--db", default="../data/coinbase_daily_121025.duckdb", help="Path to DuckDB file.")
    p.add_argument(
        "--table",
        default="bars_1d_usd_universe_clean",
        help="Table/view with daily bars.",
    )
    p.add_argument(
        "--symbols",
        default=" ".join(
            [
                "BTC-USD",
                "ETH-USD",
                "LTC-USD",
                "BCH-USD",
                "OXT-USD",
                "XLM-USD",
                "XTZ-USD",
                "ETC-USD",
                "LINK-USD",
                "ZRX-USD",
                "KNC-USD",
                "DASH-USD",
                "MKR-USD",
                "ATOM-USD",
                "ALGO-USD",
                "COMP-USD",
                "BAND-USD",
                "NMR-USD",
                "CGLD-USD",
                "UMA-USD",
                "LRC-USD",
                "YFI-USD",
                "UNI-USD",
                "SOL-USD",
                "SUI-USD",
            ]
        ),
        help="Space-separated list of symbols.",
    )
    p.add_argument("--start", required=True, help="Start date (inclusive, e.g., 2017-01-01)")
    p.add_argument("--end", required=True, help="End date (inclusive, e.g., 2025-01-01)")
    p.add_argument(
        "--cash_yield_annual",
        type=float,
        default=0.04,
        help="Annual cash yield (e.g., 0.04 for 4%).",
    )
    p.add_argument(
        "--out_dir",
        default="artifacts/research/kuma_trend",
        help="Output directory for artifacts.",
    )
    return p.parse_args()


def load_panel(db_path: Path, table: str, symbols: List[str], start: str, end: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    df = con.execute(
        f"""
        SELECT symbol, ts, open, high, low, close, volume, vwap
        FROM {table}
        WHERE symbol IN ({','.join(['?']*len(symbols))})
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts, symbol
        """,
        symbols + [start, end],
    ).fetchdf()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in args.symbols.split() if s.strip()]
    if not symbols:
        symbols = [
            "BTC-USD",
            "ETH-USD",
            "LTC-USD",
            "BCH-USD",
            "EOS-USD",
            "OXT-USD",
            "XLM-USD",
            "XTZ-USD",
            "ETC-USD",
            "LINK-USD",
            "REP-USD",
            "ZRX-USD",
            "KNC-USD",
            "DASH-USD",
            "MKR-USD",
            "ATOM-USD",
            "OMG-USD",
            "ALGO-USD",
            "COMP-USD",
            "BAND-USD",
            "NMR-USD",
            "CGLD-USD",
            "UMA-USD",
            "LRC-USD",
            "YFI-USD",
            "UNI-USD",
            "REN-USD",
            "SOL-USD",
            "SUI-USD",
        ]
    panel = load_panel(db_path, args.table, symbols, args.start, args.end)
    available = set(panel["symbol"].unique())
    requested = set(symbols)
    missing = sorted(requested - available)
    if missing:
        print(f"[kuma_trend] WARNING: symbols not found in {args.table}: {missing}")
    symbols = sorted(available & requested)
    if panel.empty or not symbols:
        raise RuntimeError("No data loaded for the requested symbols/date range.")
    panel = panel[panel["symbol"].isin(symbols)]

    cfg = KumaConfig(
        breakout_lookback=20,
        fast_ma=5,
        slow_ma=40,
        atr_window=20,
        vol_window=20,
        cash_yield_annual=args.cash_yield_annual,
        cash_buffer=0.05,
    )

    weights_df, equity_df, positions = run_kuma_trend_backtest(panel.set_index(["symbol", "ts"]), cfg)

    weights_out = out_dir / "kuma_trend_weights_v0.parquet"
    equity_out = out_dir / "kuma_trend_equity_v0.csv"
    positions_out = out_dir / "kuma_trend_positions_v0.parquet"
    turnover_out = out_dir / "kuma_trend_turnover_v0.csv"

    weights_df.reset_index().to_parquet(weights_out, index=False)
    equity_df.to_csv(equity_out, index=False)
    positions.reset_index().to_parquet(positions_out, index=False)
    equity_df[["ts", "turnover"]].to_csv(turnover_out, index=False)

    print(f"[run_kuma_trend_backtest_v0] Wrote weights to {weights_out}")
    print(f"[run_kuma_trend_backtest_v0] Wrote equity to {equity_out}")
    print(f"[run_kuma_trend_backtest_v0] Wrote positions to {positions_out}")
    print(f"[run_kuma_trend_backtest_v0] Wrote turnover to {turnover_out}")

    manifest = build_base_manifest(
        strategy_id="kuma_trend_v0",
        argv=sys.argv,
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest.update(
        {
            "config": vars(args) | {"symbols": symbols},
            "config_hash": hash_config_blob(vars(args)),
            "data_sources": {
                "duckdb": fingerprint_file(db_path),
                "price_table": args.table,
            },
            "universe": args.table,
            "artifacts_written": {
                "weights_parquet": str(weights_out),
                "equity_csv": str(equity_out),
                "positions_parquet": str(positions_out),
                "turnover_csv": str(turnover_out),
            },
        }
    )
    manifest_path = out_dir / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)
    print(f"[run_kuma_trend_backtest_v0] Wrote run manifest to {manifest_path}")


if __name__ == "__main__":
    main()

