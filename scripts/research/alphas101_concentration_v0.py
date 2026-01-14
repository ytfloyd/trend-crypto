#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concentration diagnostics for 101_alphas ensemble weights.")
    p.add_argument(
        "--weights",
        required=True,
        help="Path to ensemble weights parquet (e.g. artifacts/research/101_alphas/ensemble_weights_v0.parquet)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV path for top symbols by avg |weight| "
             "(e.g. artifacts/research/101_alphas/alphas101_concentration_v0.csv)",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top symbols to include (default: 20).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    out_path = Path(args.out)

    w = pd.read_parquet(weights_path)
    if "symbol" not in w.columns or "weight" not in w.columns:
        raise ValueError("Expected 'symbol' and 'weight' columns in weights parquet.")

    avg_abs = w.groupby("symbol")["weight"].apply(lambda s: s.abs().mean())
    avg_abs = avg_abs.sort_values(ascending=False)

    top = avg_abs.head(args.top_n).reset_index()
    top.columns = ["symbol", "avg_abs_weight"]

    total_abs = avg_abs.sum()
    btc_eth = avg_abs.get("BTC-USD", 0.0) + avg_abs.get("ETH-USD", 0.0)
    share_btc_eth = btc_eth / total_abs if total_abs > 0 else 0.0

    top["total_abs_sum"] = total_abs
    top["btc_eth_abs"] = btc_eth
    top["btc_eth_share_total"] = share_btc_eth

    top.to_csv(out_path, index=False)
    print(f"[alphas101_concentration_v0] Wrote top-{args.top_n} concentration metrics to {out_path}")
    print(f"[alphas101_concentration_v0] BTC+ETH share of avg gross: {share_btc_eth:.3f}")


if __name__ == "__main__":
    main()

