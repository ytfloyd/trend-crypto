#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import duckdb
import numpy as np
import pandas as pd

from alphas101_lib_v0 import cs_rank

ENSEMBLE_ALPHA_NAMES = [
    # Base alphas currently traded
    "alpha_001",
    "alpha_002",
    "alpha_003",
    "alpha_004",
    "alpha_005",
    "alpha_006",
    "alpha_007",
    "alpha_008",
    "alpha_009",
    "alpha_010",
    # Custom C-alphas (tradeable subset)
    "alpha_c01",
    "alpha_c02",
    "alpha_c03",
    "alpha_c04",
    "alpha_c05",
    "alpha_c06",
    "alpha_c07",
    "alpha_c08",
    "alpha_c09",
    "alpha_c10",
    "alpha_c11",
    "alpha_c12",
    "alpha_c13",
    "alpha_c14",
    "alpha_c15",
]

DEFAULT_DB_PATH = Path("data/market.duckdb")


def resolve_db_path(args: argparse.Namespace) -> Path:
    if getattr(args, "db", None):
        db_path = Path(args.db)
    else:
        env_path = os.getenv("TREND_CRYPTO_DUCKDB_PATH")
        if env_path:
            db_path = Path(env_path)
        else:
            db_path = DEFAULT_DB_PATH
    db_path = db_path.expanduser().resolve()
    if not db_path.exists():
        candidates = list(Path("data").glob("*.duckdb")) if Path("data").exists() else []
        msg = [
            f"[run_101_alphas_ensemble_v0] ERROR: DuckDB file not found at: {db_path}",
            "",
            "Hints:",
            "- set TREND_CRYPTO_DUCKDB_PATH=/path/to/your.duckdb, or",
            "  pass --db /path/to/your.duckdb, or",
            "  symlink/copy it to data/market.duckdb",
        ]
        if candidates:
            msg.append("")
            msg.append("Discovered .duckdb files under ./data:")
            for c in candidates:
                msg.append(f"  - {c.resolve()}")
        raise FileNotFoundError("\n".join(msg))
    return db_path


def load_alphas(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Alphas parquet must have ts and symbol columns.")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["ts", "symbol"])
    return df


def load_returns(db_path: Path, table: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone='UTC';")
    df = con.execute(
        f"""
        SELECT ts, symbol, close
        FROM {table}
        ORDER BY ts, symbol;
        """
    ).fetch_df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["symbol", "ts"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    # forward return for weight at t -> return at t+1
    df["fwd_ret"] = df.groupby("symbol")["ret"].shift(-1)
    return df[["ts", "symbol", "fwd_ret"]]


def apply_turnover_cap(
    prev_w: "pd.Series",
    target_w: "pd.Series",
    turnover_cap: float,
) -> tuple["pd.Series", float]:
    """
    Enforce a max daily turnover constraint between prev_w and target_w.
    Returns capped weights and realized turnover (0.5 * sum|delta|).
    """
    if turnover_cap is None or turnover_cap <= 0:
        all_syms = prev_w.index.union(target_w.index)
        prev = prev_w.reindex(all_syms).fillna(0.0)
        target = target_w.reindex(all_syms).fillna(0.0)
        delta = target - prev
        raw_turnover = 0.5 * np.abs(delta).sum()
        return target, float(raw_turnover)

    all_syms = prev_w.index.union(target_w.index)
    prev = prev_w.reindex(all_syms).fillna(0.0)
    target = target_w.reindex(all_syms).fillna(0.0)
    delta = target - prev
    raw_turnover = 0.5 * np.abs(delta).sum()

    if raw_turnover <= turnover_cap or raw_turnover == 0:
        return target, float(raw_turnover)

    scale = turnover_cap / raw_turnover
    capped = prev + delta * scale
    realized_turnover = 0.5 * np.abs(capped - prev).sum()

    return capped, float(realized_turnover)


def build_weights(alpha_df: pd.DataFrame, target_gross: float) -> pd.DataFrame:
    available = [c for c in ENSEMBLE_ALPHA_NAMES if c in alpha_df.columns]
    if not available:
        raise ValueError("No ensemble alpha columns found to ensemble.")

    # Sanity check: drop alpha_c05 if highly collinear with alpha_c01
    if "alpha_c01" in available and "alpha_c05" in available:
        flat = alpha_df[["alpha_c01", "alpha_c05"]].dropna()
        if not flat.empty:
            corr = flat["alpha_c01"].corr(flat["alpha_c05"])
            if corr is not None and abs(corr) > 0.7:
                print(
                    f"[run_101_alphas_ensemble_v0] |corr(alpha_c01, alpha_c05)|={corr:.3f} > 0.7; dropping alpha_c05 from ensemble"
                )
                available.remove("alpha_c05")

    # Build combined signed signal:
    # 1) time-series z-score each alpha per symbol (keeps sign)
    # 2) average across alphas to get a signed signal
    alpha_panel = alpha_df.set_index(["ts", "symbol"])[available]

    def _zscore_time(s: pd.Series) -> pd.Series:
        m = s.mean()
        v = s.std(ddof=0)
        return (s - m) / (v + 1e-8)

    alpha_z = alpha_panel.groupby(level="symbol").transform(_zscore_time)
    signal = alpha_z.mean(axis=1)
    signal.name = "signal"
    signal = signal.dropna()

    weights_list: List[tuple] = []
    for ts, s in signal.groupby(level="ts"):
        s = s.droplevel("ts")  # index: symbol

        if s.empty:
            continue

        s_pos = s[s > 0].copy()
        s_neg = s[s < 0].copy()

        sum_pos = float(s_pos.sum()) if not s_pos.empty else 0.0
        sum_neg_abs = float((-s_neg).sum()) if not s_neg.empty else 0.0
        denom = sum_pos + sum_neg_abs

        if denom > 0.0 and sum_pos > 0.0:
            tilt = sum_pos / denom  # in [0,1]
            gross_long_ts = target_gross * tilt
            raw_w = s_pos / sum_pos
            w_ts = raw_w * gross_long_ts
        else:
            gross_long_ts = 0.0
            w_ts = s * 0.0

        for sym, w_sym in w_ts.items():
            weights_list.append((ts, sym, float(w_sym)))

    weights_df = pd.DataFrame(weights_list, columns=["ts", "symbol", "weight"])
    weights_df = weights_df.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return weights_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct equal-weight ensemble across implemented 101 Alphas subset."
    )
    parser.add_argument(
        "--alphas",
        type=str,
        required=True,
        help="Path to alphas parquet (output of run_101_alphas_compute_v0).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to DuckDB database.",
    )
    parser.add_argument(
        "--price_table",
        type=str,
        default="bars_1d_usd_universe_clean",
        help="Table/view for prices to compute returns.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/research/101_alphas",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--target_gross",
        type=float,
        default=1.0,
        help="Target gross long exposure (<= 1.0). Default 1.0.",
    )
    parser.add_argument(
        "--cash_yield_annual",
        type=float,
        default=0.04,
        help="Annualized cash yield (e.g. 0.04 = 4%).",
    )
    parser.add_argument(
        "--turnover_cap",
        type=float,
        default=0.15,
        help="Max daily turnover (fraction of notional). Default: 0.15",
    )
    parser.add_argument(
        "--regime_csv",
        default=None,
        help="Optional regime labels CSV (ts,regime) from alphas101_regime_labels_v0.py.",
    )
    parser.add_argument(
        "--regime_mode",
        default="none",
        choices=["none", "danger_cash"],
        help=(
            "Regime gating mode. "
            "'none': ignore regimes. "
            "'danger_cash': on days with regime=='danger', set all weights to 0 and go 100% cash."
        ),
    )

    args = parser.parse_args()
    db_path = resolve_db_path(args)
    alpha_path = Path(args.alphas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes = None
    if args.regime_csv:
        regimes = pd.read_csv(args.regime_csv, parse_dates=["ts"])
        regimes = (
            regimes[["ts", "regime"]]
            .drop_duplicates(subset=["ts"])
            .sort_values("ts")
        )

    alpha_df = load_alphas(alpha_path)
    weights_df = build_weights(alpha_df, args.target_gross)

    returns_df = load_returns(db_path, args.price_table)

    # Apply turnover cap sequentially
    weights_list = []
    turn_rows = []
    prev_weights = pd.Series(dtype=float)

    for ts in sorted(weights_df["ts"].unique()):
        target_w = (
            weights_df[weights_df["ts"] == ts]
            .set_index("symbol")["weight"]
        )
        realized_w, realized_turn = apply_turnover_cap(
            prev_weights, target_w, turnover_cap=args.turnover_cap
        )
        for sym, w_sym in realized_w.items():
            weights_list.append((ts, sym, float(w_sym)))
        turn_rows.append({"ts": ts, "turnover": realized_turn})
        prev_weights = realized_w

    weights_df = pd.DataFrame(weights_list, columns=["ts", "symbol", "weight"]).sort_values(
        ["ts", "symbol"]
    )
    turnover_df = pd.DataFrame(turn_rows).sort_values("ts")

    if regimes is not None and args.regime_mode != "none":
        weights_df = weights_df.merge(regimes, on="ts", how="left")
        if args.regime_mode == "danger_cash":
            danger_mask = weights_df["regime"] == "danger"
            if danger_mask.any():
                weights_df.loc[danger_mask, "weight"] = 0.0

    weights_to_save = weights_df[["ts", "symbol", "weight"]].copy()

    weights_out = out_dir / "ensemble_weights_v0.parquet"
    weights_to_save.to_parquet(weights_out, index=False)
    print(
        f"[run_101_alphas_ensemble_v0] Wrote weights (per ts,symbol) to {weights_out}"
    )

    rf_daily = (1.0 + args.cash_yield_annual) ** (1.0 / 365.0) - 1.0
    fwd_ret = returns_df.set_index(["ts", "symbol"])["fwd_ret"]

    eq_rows = []
    nav = 1.0
    for ts, g in weights_to_save.groupby("ts"):
        w = g.set_index("symbol")["weight"]
        try:
            r = fwd_ret.xs(ts, level="ts")
        except Exception:
            r = pd.Series(dtype=float)
        r = r.reindex(w.index).fillna(0.0)

        active_ret = float((w * r).sum())
        gross_long_ts = float(w.sum())
        cash_weight_ts = max(0.0, 1.0 - gross_long_ts)

        total_ret = active_ret + cash_weight_ts * rf_daily
        nav *= (1.0 + total_ret)

        eq_rows.append(
            {
                "ts": ts,
                "portfolio_ret": total_ret,
                "portfolio_equity": nav,
                "gross_long": gross_long_ts,
                "cash_weight": cash_weight_ts,
            }
        )

    equity_df = pd.DataFrame(eq_rows).sort_values("ts")
    equity_path = out_dir / "ensemble_equity_v0.csv"
    equity_df.to_csv(equity_path, index=False)
    print(
        f"[run_101_alphas_ensemble_v0] Wrote ensemble equity with {len(equity_df)} rows to {equity_path}"
    )

    turnover_path = out_dir / "ensemble_turnover_v0.csv"
    turnover_df.to_csv(turnover_path, index=False)
    print(
        f"[run_101_alphas_ensemble_v0] Wrote turnover with {len(turnover_df)} rows to {turnover_path}"
    )


if __name__ == "__main__":
    main()

