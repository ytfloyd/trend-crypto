"""CLI: parse an IBKR activity-report CSV into silver ledger artifacts.

Usage:
    python scripts/build_ledger.py --csv /path/to/activity.csv

Writes (under research/silver_replicator/artifacts/):
    silver_ledger.parquet                  -- trades_aggregate
    silver_perf_by_underlying.csv          -- perf_by_underlying
    silver_monthly_account_returns.csv     -- monthly_returns
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when running this file directly.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_ROOT))

from src.ledger import build_silver_ledger  # noqa: E402


def _print_summary(trades: pd.DataFrame, perf: pd.DataFrame) -> None:
    print("=" * 60)
    print("Silver ledger summary")
    print("=" * 60)

    if trades.empty:
        print("No silver rows found in Trade Summary.")
    else:
        n_symbols = trades["Symbol"].nunique()
        is_fut = trades["instrument_class"].eq("future")
        is_opt = trades["instrument_class"].eq("future_option")
        gross = float(trades["gross_qty"].sum())
        net = float(trades["net_qty"].sum())
        print(f"Unique silver symbols traded     : {n_symbols}")
        print(f"  futures rows                   : {int(is_fut.sum())}")
        print(f"  future-option rows             : {int(is_opt.sum())}")
        print(f"Gross contracts (buy+sell abs)   : {gross:,.0f}")
        print(f"Net contracts (buy-sell)         : {net:,.0f}")

    if not perf.empty and "is_total_row" in perf.columns:
        non_total = perf[~perf["is_total_row"]]
        totals = perf[perf["is_total_row"]]
        realized = float(non_total["Realized_P&L"].sum())
        print(f"Total silver realized P&L (USD)  : ${realized:,.2f}")
        print("\nPer-underlying totals:")
        for _, row in totals.iterrows():
            label = row["Underlying"]
            r = row.get("Realized_P&L", float("nan"))
            print(f"  {label:30s} ${r:,.2f}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to IBKR activity-report CSV.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=_ROOT / "artifacts",
        help="Output directory for artifact files.",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    result = build_silver_ledger(args.csv)

    trades = result["trades_aggregate"]
    perf = result["perf_by_underlying"]
    monthly = result["monthly_returns"]

    trades_path = args.artifacts / "silver_ledger.parquet"
    perf_path = args.artifacts / "silver_perf_by_underlying.csv"
    monthly_path = args.artifacts / "silver_monthly_account_returns.csv"

    trades.to_parquet(trades_path, index=False)
    perf.to_csv(perf_path, index=False)
    monthly.to_csv(monthly_path, index=False)

    _print_summary(trades, perf)
    print(f"\nWrote: {trades_path}")
    print(f"Wrote: {perf_path}")
    print(f"Wrote: {monthly_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
