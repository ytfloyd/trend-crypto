#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def symbol_to_stub(symbol: str) -> str:
    # "SOL-USD" -> "sol_usd"
    return symbol.lower().replace("-", "_")


def find_run_dir_for_symbol(
    run_root: Path, prefix: str, symbol: str
) -> Path:
    stub = symbol_to_stub(symbol)
    candidates: List[Path] = []
    for child in run_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(f"{prefix}{stub}_"):
            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No run dir found for symbol={symbol} with prefix={prefix} under {run_root}"
        )
    if len(candidates) > 1:
        # if multiple, pick latest by name (timestamp suffix) and warn
        candidates = sorted(candidates)
        print(
            f"[WARN] Multiple run dirs for {symbol}; using latest: {candidates[-1].name}"
        )
    return candidates[-1]


def load_returns_from_equity(equity_path: Path) -> pd.Series:
    df = pd.read_parquet(equity_path)
    # expected schema from engine: ts + nav and per-bar returns
    # e.g. ['ts', 'nav', 'gross_ret', 'net_ret', ...]
    if "ts" not in df.columns:
        raise ValueError(f"Missing 'ts' column in {equity_path}: {df.columns}")

    df = df.sort_values("ts")

    if "net_ret" in df.columns:
        # preferred: use net_ret directly
        df = df.set_index("ts")
        return df["net_ret"].astype(float)

    if "nav" in df.columns:
        # fallback: derive returns from nav
        df["ret"] = df["nav"].astype(float).pct_change().fillna(0.0)
        df = df.set_index("ts")
        return df["ret"]

    raise ValueError(
        f"Unexpected equity schema in {equity_path}: {df.columns} "
        "(need either 'net_ret' or 'nav')."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate midcap momentum v0 runs into an equal-weight portfolio."
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default="artifacts/runs",
        help="Root directory containing run_id subdirectories.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="midcap_momentum_v0_",
        help="Run_id prefix for midcap momentum v0 runs.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        required=True,
        help="Symbols to include, e.g. SOL-USD ETC-USD ...",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/research/midcap_momentum_v0/portfolio.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_path = Path(args.out)

    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    series_map: Dict[str, pd.Series] = {}

    for symbol in args.symbols:
        run_dir = find_run_dir_for_symbol(run_root, args.prefix, symbol)
        equity_path = run_dir / "equity.parquet"
        if not equity_path.exists():
            raise FileNotFoundError(f"Missing equity.parquet in {run_dir}")
        ret = load_returns_from_equity(equity_path)
        series_map[symbol] = ret
        print(f"[aggregate_midcap_portfolio_v0] Loaded {symbol} from {run_dir.name}")

    # align on ts index
    returns_df = pd.DataFrame(series_map).sort_index()

    # equal weight among symbols that are active on each date
    active = ~returns_df.isna()
    weights = active.astype(float)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)

    portfolio_ret = (returns_df.fillna(0.0) * weights).sum(axis=1)
    portfolio_equity = (1.0 + portfolio_ret).cumprod()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = (
        pd.DataFrame(
            {
                "ts": portfolio_ret.index,
                "portfolio_ret": portfolio_ret.values,
                "portfolio_equity": portfolio_equity.values,
            }
        )
        .sort_values("ts")
    )
    out_df.to_csv(out_path, index=False)
    print(
        f"[aggregate_midcap_portfolio_v0] Wrote portfolio series with "
        f"{len(out_df)} rows to {out_path}"
    )


if __name__ == "__main__":
    main()

