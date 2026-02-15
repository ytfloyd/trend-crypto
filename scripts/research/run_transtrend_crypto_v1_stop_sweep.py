#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import duckdb
import pandas as pd

from transtrend_crypto_lib_v1 import (
    HorizonSpec,
    TranstrendConfigV1,
    build_target_weights,
    compute_trend_scores,
    simulate_portfolio,
)
from transtrend_crypto_metrics_v1 import compute_metrics
from timeseries_bundle_v0 import write_timeseries_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto v1 ATR stop sweep")
    p.add_argument("--db", type=str, default="../data/coinbase_daily_121025.duckdb")
    p.add_argument("--table", type=str, default="bars_1d_usd_universe_clean_adv10m")
    p.add_argument("--start", type=str, default="2017-01-01")
    p.add_argument("--end", type=str, default="2025-01-01")
    p.add_argument("--out_dir", type=str, default="artifacts/research/transtrend_crypto_v1_sweeps")
    p.add_argument("--target_vol_annual", type=float, default=0.20)
    p.add_argument("--danger_gross", type=float, default=0.25)
    p.add_argument("--cost_bps", type=float, default=20.0)
    p.add_argument("--cash_yield_annual", type=float, default=0.04)
    p.add_argument("--atr_window", type=int, default=20)
    p.add_argument("--atr_k_grid", type=str, default="2.0,2.5,3.0,3.5,4.0")
    p.add_argument("--cooldown_grid", type=str, default="0,3,5,10")
    p.add_argument("--write_bundle", action="store_true", default=True, help="Write timeseries bundle.")
    p.add_argument("--no_write_bundle", action="store_false", dest="write_bundle", help="Disable bundle.")
    p.add_argument(
        "--bundle_format",
        choices=["parquet", "csvgz", "both"],
        default="csvgz",
        help="Bundle output format.",
    )
    p.add_argument("--no_html", action="store_true", help="Skip HTML tearsheet generation.")
    return p.parse_args()


def _parse_float_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_grid(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_panel(db_path: str, table: str, start: str, end: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE ts >= ? AND ts <= ?
            ORDER BY ts, symbol
            """,
            [start, end],
        ).fetch_df()
    finally:
        con.close()

    required = {"symbol", "ts", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["open", "close"])
    return df


def run_one(panel: pd.DataFrame, cfg: TranstrendConfigV1, out_dir: Path, *, write_bundle: bool, bundle_format: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_scored = compute_trend_scores(panel, cfg)
    weights_signal, danger, stop_levels, stop_events = build_target_weights(panel_scored, cfg)
    equity_df, weights_held = simulate_portfolio(panel_scored, weights_signal, cfg, danger)

    weights_signal.to_parquet(out_dir / "weights_signal.parquet", index=False)
    weights_held.to_parquet(out_dir / "weights_held.parquet", index=False)
    equity_df.to_csv(out_dir / "equity.csv", index=False)
    equity_df[["ts", "turnover_one_sided", "turnover_two_sided"]].to_csv(out_dir / "turnover.csv", index=False)
    stop_levels.to_parquet(out_dir / "stop_levels.parquet", index=False)
    stop_events.to_csv(out_dir / "stop_events.csv", index=False)

    if write_bundle:
        bars_df = panel_scored[["ts", "symbol", "open", "high", "low", "close", "volume"]].copy()
        features_df = panel_scored[["ts", "symbol", "score", "vol_ann", "atr"]].copy()
        write_parquet = bundle_format in ("parquet", "both")
        write_csvgz = bundle_format in ("csvgz", "both", "parquet")
        bundle_info = write_timeseries_bundle(
            str(out_dir),
            bars_df=bars_df,
            features_df=features_df,
            weights_signal_df=weights_signal,
            weights_held_df=weights_held,
            stops_df=stop_levels,
            portfolio_df=equity_df,
            write_parquet=write_parquet,
            write_csvgz=write_csvgz,
        )
        bundle_path = bundle_info.get("parquet") or bundle_info.get("csvgz")
        print(f"[transtrend_crypto_v1_stop_sweep] Wrote timeseries bundle: {bundle_path} (rows={bundle_info.get('rows')})")

    metrics = compute_metrics(equity_df)

    stop_hits = len(stop_events)
    n_days = metrics.loc[0, "n_days"] if not metrics.empty else 0
    stop_rate = stop_hits / (n_days / 365.0) if n_days > 0 else 0.0

    danger_pct = float(equity_df["danger"].mean()) if "danger" in equity_df.columns else 0.0
    avg_gross = float(equity_df["gross_exposure"].mean()) if "gross_exposure" in equity_df.columns else 0.0

    metrics.loc[0, "stop_hits"] = stop_hits
    metrics.loc[0, "stop_hit_rate_per_year"] = stop_rate
    metrics.loc[0, "danger_pct"] = danger_pct
    metrics.loc[0, "avg_gross"] = avg_gross

    metrics_path = out_dir / "metrics_transtrend_crypto_v1.csv"
    metrics.to_csv(metrics_path, index=False)

    return {
        "total_return": float(metrics.loc[0, "total_return"]),
        "cagr": float(metrics.loc[0, "cagr"]),
        "vol": float(metrics.loc[0, "vol"]),
        "sharpe": float(metrics.loc[0, "sharpe"]),
        "sortino": float(metrics.loc[0, "sortino"]),
        "calmar": float(metrics.loc[0, "calmar"]),
        "max_dd": float(metrics.loc[0, "max_dd"]),
        "avg_turnover_one_sided": float(metrics.loc[0, "avg_turnover_one_sided"]),
        "danger_pct": danger_pct,
        "avg_gross": avg_gross,
        "stop_hits": stop_hits,
        "stop_hit_rate_per_year": stop_rate,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atr_k_grid = _parse_float_grid(args.atr_k_grid)
    cooldown_grid = _parse_int_grid(args.cooldown_grid)

    panel = load_panel(args.db, args.table, args.start, args.end)

    horizons = [
        HorizonSpec("fast", 10, 2, 20),
        HorizonSpec("mid", 20, 5, 40),
        HorizonSpec("slow", 50, 10, 200),
    ]

    results = []
    for atr_k in atr_k_grid:
        for cooldown in cooldown_grid:
            run_dir = out_dir / f"k{atr_k}_cd{cooldown}"
            record = {
                "atr_k": atr_k,
                "cooldown": cooldown,
                "out_dir_run": str(run_dir),
                "success": False,
                "error": None,
            }
            try:
                cfg = TranstrendConfigV1(
                    horizons=horizons,
                    target_vol_annual=args.target_vol_annual,
                    danger_gross=args.danger_gross,
                    cost_bps=args.cost_bps,
                    fee_bps=args.cost_bps / 2.0,
                    slippage_bps=args.cost_bps / 2.0,
                    cash_yield_annual=args.cash_yield_annual,
                    atr_window=args.atr_window,
                    atr_k=atr_k,
                    stop_cooldown_days=cooldown,
                )
                metrics = run_one(panel, cfg, run_dir, write_bundle=args.write_bundle, bundle_format=args.bundle_format)
                record.update(metrics)
                record["success"] = True
                # --- HTML tearsheet per sweep run ---
                if not args.no_html:
                    try:
                        from tearsheet_common_v0 import build_standard_html_tearsheet, load_equity_csv
                        strat_eq = load_equity_csv(str(run_dir / "equity.csv"))
                        build_standard_html_tearsheet(
                            out_html=run_dir / "tearsheet.html",
                            strategy_label=f"Transtrend v1 Stop Sweep (k={atr_k}, cd={cooldown})",
                            strategy_equity=strat_eq,
                            equity_csv_path=str(run_dir / "equity.csv"),
                            subtitle=f"ATR stop k={atr_k}, cooldown={cooldown} days",
                        )
                    except Exception as exc:
                        print(f"[stop_sweep] HTML tearsheet failed for {run_dir}: {exc}")
            except Exception as exc:
                record["error"] = str(exc)
            results.append(record)

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "stop_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best_df = summary_df[summary_df["success"]].copy()
    if not best_df.empty:
        best_df = best_df.sort_values(["calmar", "sharpe"], ascending=False).head(10)
    best_path = out_dir / "stop_sweep_best.csv"
    best_df.to_csv(best_path, index=False)

    print(f"[transtrend_crypto_v1_stop_sweep] Wrote {summary_path}")
    print(f"[transtrend_crypto_v1_stop_sweep] Wrote {best_path}")


if __name__ == "__main__":
    main()
