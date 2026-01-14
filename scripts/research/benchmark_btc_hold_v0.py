#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BTC-USD buy-and-hold benchmark equity aligned to a strategy equity CSV.")
    p.add_argument("--db", required=True, help="Path to DuckDB database.")
    p.add_argument("--price_table", required=True, help="Table/view with daily prices (ts, symbol, close).")
    p.add_argument("--symbol", default="BTC-USD", help="Benchmark symbol (default: BTC-USD).")
    p.add_argument("--equity_csv", required=True, help="Strategy equity CSV with ts column to align on.")
    p.add_argument("--out_csv", required=True, help="Output CSV for benchmark equity.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eq_path = Path(args.equity_csv)
    out_path = Path(args.out_csv)

    eq = pd.read_csv(eq_path, parse_dates=["ts"])
    if eq.empty or "ts" not in eq.columns:
        raise SystemExit(f"equity_csv invalid or empty: {eq_path}")
    eq_dates = eq["ts"].sort_values()
    start, end = eq_dates.min(), eq_dates.max()

    con = duckdb.connect(str(args.db))
    query = f"""
        SELECT ts, close
        FROM {args.price_table}
        WHERE symbol = $1
          AND ts BETWEEN $2 AND $3
        ORDER BY ts
    """
    btc = con.execute(query, [args.symbol, start, end]).fetch_df()
    con.close()

    if btc.empty:
        raise SystemExit("No BTC data found in the requested window.")

    px = btc.sort_values("ts").set_index("ts")["close"].astype(float)
    px = px.reindex(eq_dates).ffill()
    equity_btc = px / px.iloc[0]

    out = pd.DataFrame({"ts": eq_dates, "equity": equity_btc.values})
    out.to_csv(out_path, index=False)
    print(f"[benchmark_btc_hold_v0] Wrote BTC benchmark equity to {out_path} (columns: ts, equity)")

    # Helper snippet (manual stats check):
    # import numpy as np
    # ret = equity_btc.pct_change().dropna()
    # ann_mu = (1 + ret.mean()) ** 365 - 1
    # ann_vol = ret.std() * np.sqrt(365)
    # sharpe = ann_mu / ann_vol
    # dd = equity_btc / equity_btc.cummax() - 1
    # print(ann_mu, ann_vol, sharpe, dd.min())


if __name__ == "__main__":
    main()
