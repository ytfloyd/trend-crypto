#!/usr/bin/env python
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beta/correlation of 101_alphas ensemble vs a benchmark (e.g. BTC)."
    )
    p.add_argument(
        "--equity",
        required=True,
        help="Ensemble equity CSV (e.g. artifacts/research/101_alphas/ensemble_equity_v0.csv)",
    )
    p.add_argument(
        "--db",
        required=True,
        help="DuckDB path (e.g. ../data/coinbase_daily_121025.duckdb)",
    )
    p.add_argument(
        "--price_table",
        default="bars_1d_usd_universe_clean",
        help="Daily bars table/view (default: bars_1d_usd_universe_clean)",
    )
    p.add_argument(
        "--benchmark_symbol",
        default="BTC-USD",
        help="Benchmark symbol (default: BTC-USD)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV for beta/correlation summary.",
    )
    return p.parse_args()


def ols_beta(y: pd.Series, x: pd.Series):
    """
    Regress y = alpha + beta * x + eps.

    Returns:
        alpha, beta, r2, t_beta, n_obs
    """
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float)
    mask = y.notna() & x.notna()
    y = y[mask]
    x = x[mask]

    n = len(y)
    if n < 5:
        raise ValueError(f"Not enough observations for regression: n={n}")

    X = np.column_stack([np.ones(n), x.values])
    beta_hat, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
    alpha, beta = beta_hat

    y_hat = X @ beta_hat
    resid = y.values - y_hat

    # R^2
    tss = ((y.values - y.mean()) ** 2).sum()
    rss = (resid ** 2).sum()
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    # Var-cov of beta_hat
    dof = max(n - 2, 1)
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    var_beta1 = sigma2 * xtx_inv[1, 1]
    se_beta1 = np.sqrt(var_beta1) if var_beta1 > 0 else np.nan
    t_beta = beta / se_beta1 if se_beta1 and se_beta1 > 0 else np.nan

    return alpha, beta, r2, t_beta, n


def main() -> None:
    args = parse_args()

    equity_path = Path(args.equity)
    db_path = Path(args.db)
    out_path = Path(args.out)

    # Strategy returns
    eq = pd.read_csv(equity_path, parse_dates=["ts"])
    if "portfolio_ret" not in eq.columns:
        raise ValueError("Expected 'portfolio_ret' in equity file.")
    eq = eq.sort_values("ts")
    strat_ret = eq.set_index("ts")["portfolio_ret"]

    # Benchmark returns from DuckDB
    con = duckdb.connect(str(db_path))
    prices = con.execute(
        f"""
        SELECT symbol, ts, close
        FROM {args.price_table}
        WHERE symbol = ?
        ORDER BY ts
        """,
        [args.benchmark_symbol],
    ).df()
    if prices.empty:
        raise ValueError(f"No rows for benchmark symbol {args.benchmark_symbol}.")

    prices["ts"] = pd.to_datetime(prices["ts"])
    prices = prices.sort_values("ts")
    prices["bench_ret"] = prices.groupby("symbol")["close"].pct_change()
    bench_ret = (
        prices.set_index("ts")
        .sort_index()["bench_ret"]
        .dropna()
    )

    # Align on common dates
    df = pd.concat(
        {
            "strat_ret": strat_ret,
            "bench_ret": bench_ret,
        },
        axis=1,
        join="inner",
    ).dropna()
    if df.empty:
        raise ValueError("No overlapping dates between strategy and benchmark.")

    corr = df["strat_ret"].corr(df["bench_ret"])
    alpha, beta, r2, t_beta, n_obs = ols_beta(df["strat_ret"], df["bench_ret"])

    ann = 365
    vol_strat = df["strat_ret"].std() * np.sqrt(ann)
    vol_bench = df["bench_ret"].std() * np.sqrt(ann)

    out = pd.DataFrame.from_records(
        [
            dict(
                label=f"ensemble_vs_{args.benchmark_symbol}",
                n_obs=n_obs,
                corr=corr,
                beta=beta,
                alpha=alpha,
                r2=r2,
                t_beta=t_beta,
                vol_strat=vol_strat,
                vol_bench=vol_bench,
            )
        ]
    )
    out.to_csv(out_path, index=False)
    print(f"[alphas101_beta_analysis_v0] Wrote beta summary to {out_path}")


if __name__ == "__main__":
    main()

