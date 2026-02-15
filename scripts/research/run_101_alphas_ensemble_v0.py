#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
from typing import List

import duckdb
import numpy as np
import pandas as pd
from run_manifest_v0 import build_base_manifest, fingerprint_file, write_run_manifest, hash_config_blob
from scripts.research.groupby_utils import apply_by_ts

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
    Returns capped weights and realized turnover (two-sided equity turnover: 0.5 * sum|delta|).
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


def build_weights_with_concentration(
    df: pd.DataFrame,
    target_gross: float = 1.0,
    top_k: int = 0,
    frac_conc: float = 0.0,
    max_weight_core: float = 0.0025,
    max_weight_top: float = 0.02,
) -> pd.DataFrame:
    """
    Two-sleeve construction:

    1) Concentrated sleeve: top K signals get `frac_conc` of the target gross,
       with a looser cap `max_weight_top`.
    2) Core sleeve: everyone else gets (1 - frac_conc) of target gross,
       with a tighter cap `max_weight_core`.

    Assumes df has columns: ts, symbol, signal (and possibly others).
    Returns df with a new 'weight' column.
    """
    df = df.copy()
    df["signal_pos"] = df["signal"].clip(lower=0.0)

    def _per_ts(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("signal_pos", ascending=False)

        total_sig = g["signal_pos"].sum()
        if total_sig <= 0:
            g["weight"] = 0.0
            return g

        if top_k > 0:
            leaders_idx = g.index[:top_k]
            mask_leader = g.index.isin(leaders_idx)
        else:
            mask_leader = np.zeros(len(g), dtype=bool)

        sig_leader_sum = g.loc[mask_leader, "signal_pos"].sum()
        sig_core_sum = g.loc[~mask_leader, "signal_pos"].sum()

        if sig_leader_sum > 0 and top_k > 0:
            alloc_conc = target_gross * frac_conc
            alloc_core = target_gross * (1.0 - frac_conc)
        else:
            alloc_conc = 0.0
            alloc_core = target_gross

        g["w_raw"] = 0.0

        if sig_leader_sum > 0 and alloc_conc > 0:
            g.loc[mask_leader, "w_raw"] = alloc_conc * (
                g.loc[mask_leader, "signal_pos"] / sig_leader_sum
            )

        if sig_core_sum > 0 and alloc_core > 0:
            g.loc[~mask_leader, "w_raw"] = alloc_core * (
                g.loc[~mask_leader, "signal_pos"] / sig_core_sum
            )

        g.loc[mask_leader, "w_raw"] = g.loc[mask_leader, "w_raw"].clip(
            upper=max_weight_top
        )
        g.loc[~mask_leader, "w_raw"] = g.loc[~mask_leader, "w_raw"].clip(
            upper=max_weight_core
        )

        g["weight"] = g["w_raw"]
        return g

    out = apply_by_ts(df, _per_ts)
    return out


def build_weights(
    alpha_df: pd.DataFrame,
    target_gross: float,
    alpha_cols: list[str] | None = None,
    selection_signs: dict[str, float] | None = None,
) -> pd.DataFrame:
    if alpha_cols is None:
        available = [c for c in ENSEMBLE_ALPHA_NAMES if c in alpha_df.columns]
    else:
        available = [c for c in alpha_cols if c in alpha_df.columns]
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

    if selection_signs is not None:
        for col in available:
            sign = selection_signs.get(col, 1.0)
            if sign < 0:
                alpha_panel[col] = -alpha_panel[col]

    def _zscore_time(s: pd.Series) -> pd.Series:
        m = s.mean()
        v = s.std(ddof=0)
        return (s - m) / (v + 1e-8)

    alpha_z = alpha_panel.groupby(level="symbol").transform(_zscore_time)
    signal = alpha_z.mean(axis=1)
    signal.name = "signal"
    signal = signal.dropna()

    signal_df = signal.reset_index()
    weights_df = build_weights_with_concentration(
        signal_df,
        target_gross=target_gross,
        top_k=0,
        frac_conc=0.0,
        max_weight_core=0.0025,
        max_weight_top=0.02,
    )
    weights_df = weights_df[["ts", "symbol", "weight"]].sort_values(
        ["ts", "symbol"]
    ).reset_index(drop=True)
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
    parser.add_argument(
        "--alpha_selection_csv",
        help=(
            "Optional CSV with alpha selection (alpha,sign,...) from alphas101_select_v0.py. "
            "If provided, only these alphas are used; if 'sign' is present, each alpha "
            "is multiplied by that sign before cross-sectional ranking."
        ),
    )
    parser.add_argument(
        "--concentration_top_k",
        type=int,
        default=0,
        help="Number of assets in concentrated sleeve (0 = disabled).",
    )
    parser.add_argument(
        "--concentration_fraction",
        type=float,
        default=0.20,
        help="Fraction of target gross dedicated to top-K leaders.",
    )
    parser.add_argument(
        "--max_weight_core",
        type=float,
        default=0.0025,
        help="Max weight per core asset.",
    )
    parser.add_argument(
        "--max_weight_top",
        type=float,
        default=0.02,
        help="Max weight per top-K leader.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="v1_base",
        help="Suffix used in ensemble output filenames.",
    )
    parser.add_argument("--no_html", action="store_true", help="Skip HTML tearsheet generation.")

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
    full_alpha_cols = [c for c in alpha_df.columns if c.startswith("alpha_")]

    selection_signs = None
    alpha_cols = None
    if args.alpha_selection_csv:
        sel = pd.read_csv(args.alpha_selection_csv)
        if "alpha" not in sel.columns:
            raise ValueError("Selection CSV must have an 'alpha' column")
        selected = set(sel["alpha"])
        alpha_cols = [c for c in full_alpha_cols if c in selected]
        if not alpha_cols:
            raise ValueError("No overlap between selected alphas and available columns")
        if "sign" in sel.columns:
            selection_signs = dict(zip(sel["alpha"], sel["sign"]))
        print(
            f"[run_101_alphas_ensemble_v0] Using {len(alpha_cols)} selected alphas from {args.alpha_selection_csv}"
        )
    else:
        alpha_cols = [c for c in ENSEMBLE_ALPHA_NAMES if c in full_alpha_cols]

    # Build signal then apply concentration-aware weights
    # Reuse build_weights for collinearity + z-scoring but override sleeve params
    available = [c for c in alpha_cols if c in alpha_df.columns]

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

    alpha_panel = alpha_df.set_index(["ts", "symbol"])[available]

    if selection_signs is not None:
        for col in available:
            sign = selection_signs.get(col, 1.0)
            if sign < 0:
                alpha_panel[col] = -alpha_panel[col]

    def _zscore_time(s: pd.Series) -> pd.Series:
        m = s.mean()
        v = s.std(ddof=0)
        return (s - m) / (v + 1e-8)

    alpha_z = alpha_panel.groupby(level="symbol").transform(_zscore_time)
    signal = alpha_z.mean(axis=1)
    signal.name = "signal"
    signal = signal.dropna()
    signal_df = signal.reset_index()

    weights_df = build_weights_with_concentration(
        signal_df,
        target_gross=args.target_gross,
        top_k=args.concentration_top_k,
        frac_conc=args.concentration_fraction,
        max_weight_core=args.max_weight_core,
        max_weight_top=args.max_weight_top,
    )

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

    weights_out = out_dir / f"ensemble_weights_{args.output_suffix}.parquet"
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
    equity_path = out_dir / f"ensemble_equity_{args.output_suffix}.csv"
    equity_df.to_csv(equity_path, index=False)
    print(
        f"[run_101_alphas_ensemble_v0] Wrote ensemble equity with {len(equity_df)} rows to {equity_path}"
    )

    turnover_path = out_dir / f"ensemble_turnover_{args.output_suffix}.csv"
    turnover_df.to_csv(turnover_path, index=False)
    print(
        f"[run_101_alphas_ensemble_v0] Wrote turnover with {len(turnover_df)} rows to {turnover_path}"
    )

    # Run manifest
    manifest = build_base_manifest(
        strategy_id="alphas101_ensemble",
        argv=sys.argv,
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest.update(
        {
            "config": vars(args),
            "config_hash": hash_config_blob(vars(args)),
            "data_sources": {
                "duckdb": fingerprint_file(db_path),
                "price_table": args.price_table,
            },
            "universe": args.price_table,
            "artifacts_written": {
                "weights_parquet": str(weights_out),
                "equity_csv": str(equity_path),
                "turnover_csv": str(turnover_path),
            },
        }
    )
    manifest_path = out_dir / f"ensemble_run_manifest_{args.output_suffix}.json"
    write_run_manifest(manifest_path, manifest)
    print(f"[run_101_alphas_ensemble_v0] Wrote run manifest to {manifest_path}")

    # --- HTML tearsheet ---
    if not args.no_html:
        from tearsheet_common_v0 import build_standard_html_tearsheet, load_equity_csv
        strat_eq = load_equity_csv(str(equity_path))
        build_standard_html_tearsheet(
            out_html=out_dir / "tearsheet.html",
            strategy_label="101 Alphas Ensemble",
            strategy_equity=strat_eq,
            equity_csv_path=str(equity_path),
            manifest_path=str(manifest_path),
            subtitle="Equal-weight ensemble across implemented 101 Alphas subset",
        )


if __name__ == "__main__":
    main()

