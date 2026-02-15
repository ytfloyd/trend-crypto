#!/usr/bin/env python
from __future__ import annotations

"""
CLI runner for Growth Sleeve V1.5 (daily-only v0).

Loads bars from DuckDB (Top50 ADV>10M view), runs the Growth Sleeve backtest,
and writes standard artifacts (equity, weights, turnover, trades, debug).
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from typing import Any, Dict

import duckdb
import pandas as pd

from alpha_ensemble_v15_growth_lib_v0 import GrowthSleeveConfig, run_growth_sleeve_backtest
from run_manifest_v0 import (
    build_base_manifest,
    fingerprint_file,
    write_run_manifest,
    hash_config_blob,
    update_run_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Growth Sleeve V1.5 backtest (daily-only v0).")
    p.add_argument("--db", required=True, help="Path to DuckDB database.")
    p.add_argument(
        "--price_table",
        default="bars_1d_usd_universe_clean_top50_adv10m",
        help="Daily bars view/table to use.",
    )
    p.add_argument("--start", default="2023-01-01", help="Start date (inclusive, YYYY-MM-DD).")
    p.add_argument("--end", default="2024-12-31", help="End date (inclusive, YYYY-MM-DD).")
    p.add_argument("--out_dir", default="artifacts/research/alpha_ensemble_v15_growth", help="Output directory.")
    p.add_argument("--config_name", default="v0", help="Config name for filenames (default: v0).")

    # Key knobs
    p.add_argument("--adx_thresh", type=float, default=25.0)
    p.add_argument("--ichimoku_tenkan", type=int, default=9)
    p.add_argument("--ichimoku_kijun", type=int, default=26)
    p.add_argument("--ichimoku_senkou", type=int, default=52)
    p.add_argument("--ichimoku_disp", type=int, default=26)
    p.add_argument("--corr_threshold", type=float, default=0.7)
    p.add_argument("--cluster_cap", type=float, default=0.40)
    p.add_argument("--target_vol", type=float, default=0.20)
    p.add_argument("--max_scalar", type=float, default=1.5)
    p.add_argument("--max_name_weight", type=float, default=0.08)
    p.add_argument("--gap_atr_mult", type=float, default=1.0)
    p.add_argument("--slippage_bps", type=float, default=25.0)
    p.add_argument(
        "--allow_percent_returns_for_debug",
        action="store_true",
        help="Allow percent-unit returns without raising (for debug only). Default: False.",
    )
    p.add_argument(
        "--allow_low_exposure",
        action="store_true",
        help="Allow very low gross exposure without raising (warn only). Default: False.",
    )
    p.add_argument(
        "--fail_on_low_exposure",
        action="store_true",
        help="If set, low exposure triggers a hard failure.",
    )

    p.add_argument("--no_html", action="store_true", help="Skip HTML tearsheet generation.")
    return p.parse_args()


def load_bars(db_path: Path, table: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily bars for all symbols in table within [start, end].
    """
    con = duckdb.connect(str(db_path))
    query = f"""
        SELECT ts, symbol, open, high, low, close, volume
        FROM {table}
        WHERE ts BETWEEN $1 AND $2
        ORDER BY ts, symbol
    """
    df = con.execute(query, [start, end]).fetch_df()
    con.close()
    if df.empty:
        raise SystemExit(f"No data returned from {table} between {start} and {end}")
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def build_config_from_args(args: argparse.Namespace) -> GrowthSleeveConfig:
    slippage = args.slippage_bps / 10000.0
    cfg_overrides: Dict[str, Any] = dict(
        adx_threshold=args.adx_thresh,
        ichimoku_tenkan=args.ichimoku_tenkan,
        ichimoku_kijun=args.ichimoku_kijun,
        ichimoku_senkou=args.ichimoku_senkou,
        ichimoku_disp=args.ichimoku_disp,
        corr_threshold=args.corr_threshold,
        cluster_cap=args.cluster_cap,
        target_vol=args.target_vol,
        max_scalar=args.max_scalar,
        max_name_weight=args.max_name_weight,
        gap_atr_mult=args.gap_atr_mult,
        gap_slippage=slippage,
        allow_percent_returns_for_debug=args.allow_percent_returns_for_debug,
        allow_low_exposure=args.allow_low_exposure,
        fail_on_low_exposure=args.fail_on_low_exposure,
    )
    return GrowthSleeveConfig(**cfg_overrides)


def _write_diagnostics(diag_rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(diag_rows)
    required = {
        "date",
        "universe_n",
        "eligible_n",
        "active_n",
        "gross_exposure",
        "net_exposure",
        "vol_scalar",
        "expected_vol_ann",
        "target_vol",
        "max_scalar",
        "pre_cluster_gross",
        "post_cluster_gross",
        "cluster_scale",
        "n_capped_single_name",
        "n_capped_cluster",
        "turnover",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Diagnostics missing required columns: {missing}")
    df.to_csv(path, index=False)
    return df


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = load_bars(db_path, args.price_table, args.start, args.end)
    cfg = build_config_from_args(args)

    results = run_growth_sleeve_backtest(
        bars_df=bars,
        start_ts=pd.to_datetime(args.start),
        end_ts=pd.to_datetime(args.end),
        params=cfg.__dict__,
    )

    prefix = "growth"
    suffix = args.config_name

    equity_path = out_dir / f"{prefix}_equity_{suffix}.csv"
    weights_path = out_dir / f"{prefix}_weights_{suffix}.parquet"
    turnover_path = out_dir / f"{prefix}_turnover_{suffix}.csv"
    trades_path = out_dir / f"{prefix}_trades_{suffix}.parquet"
    debug_path = out_dir / f"{prefix}_debug_{suffix}.parquet"

    results["equity_df"].to_csv(equity_path, index=False)
    results["weights_df"].to_parquet(weights_path, index=False)
    # turnover from equity_df (already present)
    results["equity_df"][["ts", "turnover"]].to_csv(turnover_path, index=False)
    if not results["trades_df"].empty:
        results["trades_df"].to_parquet(trades_path, index=False)
    if not results["debug_df"].empty:
        results["debug_df"].to_parquet(debug_path, index=False)

    print(f"[growth_runner] Wrote equity to {equity_path}")
    print(f"[growth_runner] Wrote weights to {weights_path}")
    print(f"[growth_runner] Wrote turnover to {turnover_path}")
    if not results["trades_df"].empty:
        print(f"[growth_runner] Wrote trades to {trades_path}")
    if not results["debug_df"].empty:
        print(f"[growth_runner] Wrote debug to {debug_path}")

    # Quick diagnostics to avoid silent zero-trade runs
    weights = results["weights_df"]
    gross = weights.groupby("ts")["weight"].sum()
    debug_df = results["debug_df"]
    trades_df = results["trades_df"]
    equity_df = results["equity_df"]
    print("\n[growth_runner] Diagnostics:")
    print(f"  Days with gross>0: {(gross.abs() > 1e-6).mean():.3f}")
    if not debug_df.empty:
        print(f"  Regime_on rate: {debug_df['regime_on'].mean():.3f}")
        print(f"  Slow_on rate:   {debug_df['slow_on'].mean():.3f}")
        print(f"  Fast_on rate:   {debug_df['fast_on'].mean():.3f}")
    print(f"  Entries: {len(trades_df[trades_df['side'].str.contains('ENTRY')]) if not trades_df.empty else 0}")
    print(f"  Exits:   {len(trades_df[trades_df['side'].str.contains('EXIT')]) if not trades_df.empty else 0}")
    print(f"  Traded symbols: {trades_df['symbol'].nunique() if not trades_df.empty else 0}")

    # Diagnostics per day
    diag_rows = []
    for ts in equity_df["ts"]:
        w_day = weights[weights["ts"] == ts]
        gross_exposure = float(w_day["weight"].sum())
        net_exposure = gross_exposure
        active_n = int((w_day["weight"] > 0).sum())
        universe_n = int(w_day["symbol"].nunique())
        turnover_day = float(equity_df.loc[equity_df["ts"] == ts, "turnover"].iloc[0]) if "turnover" in equity_df else None
        dbg_day = debug_df[debug_df["ts"] == ts] if not debug_df.empty else pd.DataFrame()
        vol_scalar = float(dbg_day["vol_scalar"].dropna().iloc[0]) if not dbg_day.empty and "vol_scalar" in dbg_day else None
        exp_vol_ann = float(dbg_day["exp_vol_ann"].dropna().iloc[0]) if not dbg_day.empty and "exp_vol_ann" in dbg_day else None
        pre_cluster_gross = float(dbg_day["pre_cluster_gross"].dropna().iloc[0]) if not dbg_day.empty and "pre_cluster_gross" in dbg_day else None
        post_cluster_gross = float(dbg_day["post_cluster_gross"].dropna().iloc[0]) if not dbg_day.empty and "post_cluster_gross" in dbg_day else None
        cluster_scale = float(dbg_day["cluster_scale"].dropna().iloc[0]) if not dbg_day.empty and "cluster_scale" in dbg_day else None
        n_capped_single = int(dbg_day["n_capped_single"].dropna().iloc[0]) if not dbg_day.empty and "n_capped_single" in dbg_day else 0
        n_capped_cluster = int(dbg_day["n_capped_cluster"].dropna().iloc[0]) if not dbg_day.empty and "n_capped_cluster" in dbg_day else 0
        regime_on_rate = float(dbg_day["regime_on"].mean()) if not dbg_day.empty and "regime_on" in dbg_day else None
        slow_on_rate = float(dbg_day["slow_on"].mean()) if not dbg_day.empty and "slow_on" in dbg_day else None
        fast_on_rate = float(dbg_day["fast_on"].mean()) if not dbg_day.empty and "fast_on" in dbg_day else None
        diag_rows.append(
            {
                "date": ts,
                "universe_n": universe_n,
                "eligible_n": universe_n,
                "active_n": active_n,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "turnover": turnover_day,
                "vol_scalar": vol_scalar,
                "expected_vol_ann": exp_vol_ann,
                "target_vol": cfg.target_vol,
                "max_scalar": cfg.max_scalar,
                "pre_cluster_gross": pre_cluster_gross,
                "post_cluster_gross": post_cluster_gross,
                "cluster_scale": cluster_scale,
                "n_capped_single_name": n_capped_single,
                "n_capped_cluster": n_capped_cluster,
                "regime_on_rate": regime_on_rate,
                "slow_on_rate": slow_on_rate,
                "fast_on_rate": fast_on_rate,
            }
        )

    diagnostics_path = out_dir / f"{prefix}_diagnostics_{suffix}.csv"
    diag_df = _write_diagnostics(diag_rows, diagnostics_path)
    print(f"[growth_runner] Wrote diagnostics to {diagnostics_path}")

    # Exposure health summary
    summary = {}
    if not equity_df.empty:
        summary["start"] = equity_df["ts"].min()
        summary["end"] = equity_df["ts"].max()
        summary["n_days"] = equity_df["ts"].nunique()
    summary["pct_gross_gt0"] = float((gross.abs() > 1e-6).mean())
    summary["gross_mean"] = float(diag_df["gross_exposure"].mean())
    summary["gross_median"] = float(diag_df["gross_exposure"].median())
    summary["gross_p10"] = float(diag_df["gross_exposure"].quantile(0.10))
    summary["gross_p90"] = float(diag_df["gross_exposure"].quantile(0.90))
    summary["gross_max"] = float(diag_df["gross_exposure"].max())

    summary["active_mean"] = float(diag_df["active_n"].mean())
    summary["active_median"] = float(diag_df["active_n"].median())
    summary["active_p90"] = float(diag_df["active_n"].quantile(0.90))
    summary["active_max"] = float(diag_df["active_n"].max())

    if not debug_df.empty:
        vol_scalar_ts = diag_df["vol_scalar"]
        summary["scalar_mean"] = float(vol_scalar_ts.mean())
        summary["scalar_median"] = float(vol_scalar_ts.median())
        summary["scalar_p10"] = float(vol_scalar_ts.quantile(0.10))
        summary["scalar_p90"] = float(vol_scalar_ts.quantile(0.90))
        summary["scalar_max"] = float(vol_scalar_ts.max())
        summary["scalar_pct_cap"] = float((vol_scalar_ts >= 0.999 * cfg.max_scalar).mean())

        exp_vol_ts = diag_df["expected_vol_ann"]
        summary["exp_vol_mean"] = float(exp_vol_ts.mean())
        summary["exp_vol_median"] = float(exp_vol_ts.median())
        summary["exp_vol_p10"] = float(exp_vol_ts.quantile(0.10))
        summary["exp_vol_p90"] = float(exp_vol_ts.quantile(0.90))
        summary["exp_vol_max"] = float(exp_vol_ts.max())

        if "regime_on_rate" in diag_df:
            summary["regime_on_rate"] = float(diag_df["regime_on_rate"].mean())
        if "slow_on_rate" in diag_df:
            summary["slow_on_rate"] = float(diag_df["slow_on_rate"].mean())
        if "fast_on_rate" in diag_df:
            summary["fast_on_rate"] = float(diag_df["fast_on_rate"].mean())

    summary["entries"] = int(len(trades_df[trades_df["side"].str.contains("ENTRY")])) if not trades_df.empty else 0
    summary["exits"] = int(len(trades_df[trades_df["side"].str.contains("EXIT")])) if not trades_df.empty else 0
    summary["traded_symbols"] = int(trades_df["symbol"].nunique()) if not trades_df.empty else 0

    summary_path = out_dir / f"{prefix}_debug_summary_{suffix}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[growth_runner] Wrote exposure summary to {summary_path}")

    # Realized vol warning vs target
    if not equity_df.empty:
        ret = equity_df["portfolio_equity"].pct_change().dropna()
        realized_vol = ret.std() * (365 ** 0.5) if not ret.empty else 0.0
        target_vol = cfg.target_vol
        if realized_vol < 0.25 * target_vol and summary["pct_gross_gt0"] > 0.30:
            print(
                f"[growth_runner][WARNING] Realized vol {realized_vol:.4f} is <25% of target {target_vol:.4f} "
                "despite material gross exposure; check sizing/units."
            )

    # Exposure health guardrail
    gross_mean = summary.get("gross_mean", 0.0)
    if gross_mean < 0.02:
        msg = f"[growth_runner][ERROR] Gross exposure mean {gross_mean:.4f} is near zero; see {diagnostics_path}"
        if cfg.fail_on_low_exposure or not cfg.allow_low_exposure:
            raise RuntimeError(msg)
        else:
            print(msg)
    elif gross_mean < 0.15:
        print(f"[growth_runner][WARNING] Gross exposure mean {gross_mean:.4f} is low; see {diagnostics_path}")

    # Write run manifest
    manifest = build_base_manifest(strategy_id="alpha_ensemble_v15_growth_sleeve", argv=sys.argv, repo_root=Path(__file__).resolve().parents[2])
    manifest.update(
        {
            "config": cfg.__dict__,
            "config_hash": hash_config_blob(cfg.__dict__),
            "data_sources": {
                "duckdb": fingerprint_file(db_path),
                "price_table": price_table,
            },
            "universe": price_table,
            "artifacts_written": {
                "equity_csv": str(equity_path),
                "weights_parquet": str(weights_path),
                "turnover_csv": str(turnover_path),
                "trades_parquet": str(trades_path),
                "debug_parquet": str(debug_path),
                "debug_summary_csv": str(summary_path),
                "diagnostics_csv": str(diagnostics_path),
            },
        }
    )
    manifest_path = out_dir / f"{prefix}_run_manifest_{suffix}.json"
    write_run_manifest(manifest_path, manifest)
    print(f"[growth_runner] Wrote run manifest to {manifest_path}")

    # --- HTML tearsheet ---
    if not args.no_html:
        from tearsheet_common_v0 import build_standard_html_tearsheet, load_equity_csv
        strat_eq = load_equity_csv(str(equity_path))
        build_standard_html_tearsheet(
            out_html=out_dir / "tearsheet.html",
            strategy_label="Alpha Ensemble v1.5 Growth Sleeve",
            strategy_equity=strat_eq,
            equity_csv_path=str(equity_path),
            manifest_path=str(manifest_path),
            subtitle="Daily growth sleeve using 101 alpha ensemble with top-K momentum overlay",
        )


if __name__ == "__main__":
    main()
