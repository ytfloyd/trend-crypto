from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import duckdb
import polars as pl
import yaml


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_temp_config(cfg: dict) -> Path:
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    Path(tmp_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return Path(tmp_path)


def ensure_views(db_path: Path, base_table: str) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone='UTC';")
    con.execute(
        """
        CREATE OR REPLACE VIEW hourly_bars_clean AS
        SELECT *
        FROM hourly_bars
        WHERE open > 0 AND high > 0 AND low > 0 AND close > 0
          AND (high / low) <= 10
          AND abs(close / open - 1) <= 2.0;
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW bars_4h AS
        WITH base AS (
          SELECT
            symbol,
            CAST(to_timestamp(floor(epoch(ts) / 14400) * 14400) AS TIMESTAMP) AS ts_bucket,
            ts,
            open, high, low, close, volume
          FROM {base_table}
        )
        SELECT
            symbol,
            ts_bucket AS ts,
            arg_min(open, ts) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, ts) AS close,
            sum(volume) AS volume
        FROM base
        GROUP BY symbol, ts_bucket;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW bars_4h_clean AS
        WITH base AS (
          SELECT
            symbol,
            CAST(to_timestamp(floor(epoch(ts) / 14400) * 14400) AS TIMESTAMP) AS ts_bucket,
            ts,
            open, high, low, close, volume
          FROM hourly_bars_clean
        )
        SELECT
            symbol,
            ts_bucket AS ts,
            arg_min(open, ts) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, ts) AS close,
            sum(volume) AS volume
        FROM base
        GROUP BY symbol, ts_bucket;
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW bars_1d AS
        WITH base AS (
          SELECT
            symbol,
            CAST(date_trunc('day', ts) AS TIMESTAMP) AS ts_bucket,
            ts,
            open, high, low, close, volume
          FROM {base_table}
        )
        SELECT
            symbol,
            ts_bucket AS ts,
            arg_min(open, ts) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, ts) AS close,
            sum(volume) AS volume
        FROM base
        GROUP BY symbol, ts_bucket;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW bars_1d_clean AS
        WITH base AS (
          SELECT
            symbol,
            CAST(date_trunc('day', ts) AS TIMESTAMP) AS ts_bucket,
            ts,
            open, high, low, close, volume
          FROM hourly_bars_clean
        )
        SELECT
            symbol,
            ts_bucket AS ts,
            arg_min(open, ts) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, ts) AS close,
            sum(volume) AS volume
        FROM base
        GROUP BY symbol, ts_bucket;
        """
    )

    dup_4h = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT symbol, ts, COUNT(*) AS n
            FROM bars_4h
            GROUP BY symbol, ts
            HAVING COUNT(*) > 1
        ) t;
        """
    ).fetchone()[0]
    if dup_4h and dup_4h > 0:
        con.close()
        raise RuntimeError(f"bars_4h has duplicate (symbol, ts) groups: {dup_4h}")

    dup_1d = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT symbol, ts, COUNT(*) AS n
            FROM bars_1d
            GROUP BY symbol, ts
            HAVING COUNT(*) > 1
        ) t;
        """
    ).fetchone()[0]
    if dup_1d and dup_1d > 0:
        con.close()
        raise RuntimeError(f"bars_1d has duplicate (symbol, ts) groups: {dup_1d}")
    con.close()


def run_backtest(cfg_path: Path) -> tuple[str, float, int]:
    result = subprocess.run(
        ["python", "scripts/run_backtest.py", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[run_backtest] FAILED for config: {cfg_path}")
        print("---- STDOUT ----")
        print(result.stdout)
        print("---- STDERR ----")
        print(result.stderr)
        raise RuntimeError(f"run_backtest failed with code {result.returncode}")

    stdout = result.stdout.splitlines()
    run_id = None
    sharpe = None
    for line in stdout:
        if line.startswith("Run ID:"):
            run_id = line.split("Run ID:")[1].strip()
        if line.startswith("Sharpe:"):
            try:
                sharpe = float(line.split("Sharpe:")[1].strip())
            except Exception:
                pass
    if not run_id:
        raise RuntimeError("Could not parse Run ID from run_backtest output.")
    summary_path = Path("artifacts") / "runs" / run_id / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        sharpe = data.get("sharpe", sharpe)
    trades_path = Path("artifacts") / "runs" / run_id / "trades.parquet"
    trades_count = 0
    if trades_path.exists():
        trades_count = pl.read_parquet(trades_path).height
    return run_id, float(sharpe) if sharpe is not None else float("nan"), trades_count


def run_hours(run_id: str) -> float:
    manifest_path = Path("artifacts") / "runs" / run_id / "manifest.json"
    if not manifest_path.exists():
        return 0.0
    data = json.loads(manifest_path.read_text())
    tr = data.get("time_range", {})
    try:
        start = tr.get("start")
        end = tr.get("end")
        if start and end:
            start_dt = pl.from_datetime(pl.Series([start])).item()
            end_dt = pl.from_datetime(pl.Series([end])).item()
            return (end_dt - start_dt).total_seconds() / 3600
    except Exception:
        return 0.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run timeframe sensitivity for BTC/ETH sleeves.")
    parser.add_argument("--timeframes", default="1h,4h,1d", help="Comma-separated timeframes")
    parser.add_argument("--slippage_grid_bps", default="1,3,5,10,15,20", help="Comma-separated slippage bps grid")
    parser.add_argument("--fee_bps", type=float, default=10.0, help="Fee bps to apply to all runs")
    parser.add_argument("--base_config_btc", required=True, help="Path to base BTC config YAML")
    parser.add_argument("--base_config_eth", required=True, help="Path to base ETH config YAML")
    parser.add_argument("--breakeven_sharpe", type=float, default=0.0, help="Sharpe threshold for breakeven slippage")
    parser.add_argument("--out_csv", default="artifacts/compare/timeframe_sensitivity.csv", help="Output CSV path")
    args = parser.parse_args()

    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    slippage_grid = [float(x) for x in args.slippage_grid_bps.split(",") if x.strip()]

    cfg_btc_base = load_config(Path(args.base_config_btc))
    cfg_eth_base = load_config(Path(args.base_config_eth))
    db_path = Path(cfg_btc_base["data"]["db_path"])
    base_table = cfg_btc_base["data"]["table"]

    ensure_views(db_path, base_table)

    rows: List[Dict] = []

    for tf in timeframes:
        table = "hourly_bars_clean" if tf == "1h" else ("bars_4h_clean" if tf == "4h" else "bars_1d_clean")
        breakeven_btc = None
        breakeven_eth = None
        last_run_btc = None
        last_run_eth = None
        for slip in slippage_grid:
            cfg_btc = json.loads(json.dumps(cfg_btc_base))
            cfg_eth = json.loads(json.dumps(cfg_eth_base))
            for cfg in (cfg_btc, cfg_eth):
                cfg["data"]["table"] = table
                cfg["data"]["timeframe"] = tf
                cfg["execution"]["slippage_bps"] = slip
                cfg["execution"]["fee_bps"] = args.fee_bps
                cfg["strategy"]["window_units"] = "hours"
                cfg["risk"]["window_units"] = "hours"
            cfg_btc_path = write_temp_config(cfg_btc)
            cfg_eth_path = write_temp_config(cfg_eth)
            run_id_btc, sharpe_btc, trades_btc = run_backtest(cfg_btc_path)
            run_id_eth, sharpe_eth, trades_eth = run_backtest(cfg_eth_path)
            last_run_btc = run_id_btc
            last_run_eth = run_id_eth
            rows.append(
                {
                    "timeframe": tf,
                    "slippage_bps": slip,
                    "sharpe_btc": sharpe_btc,
                    "sharpe_eth": sharpe_eth,
                    "trades_btc": trades_btc,
                    "trades_eth": trades_eth,
                    "run_btc": run_id_btc,
                    "run_eth": run_id_eth,
                }
            )
            if sharpe_btc > args.breakeven_sharpe:
                breakeven_btc = slip if breakeven_btc is None else max(breakeven_btc, slip)
            if sharpe_eth > args.breakeven_sharpe:
                breakeven_eth = slip if breakeven_eth is None else max(breakeven_eth, slip)

        trades_year_btc = None
        trades_year_eth = None
        if last_run_btc:
            hours = run_hours(last_run_btc)
            trades_year_btc = trades_btc / (hours / 8760) if hours > 0 else None
        if last_run_eth:
            hours = run_hours(last_run_eth)
            trades_year_eth = trades_eth / (hours / 8760) if hours > 0 else None

        rows.append(
            {
                "timeframe": tf,
                "slippage_bps": "breakeven",
                "sharpe_btc": breakeven_btc,
                "sharpe_eth": breakeven_eth,
                "trades_btc": trades_year_btc,
                "trades_eth": trades_year_eth,
                "run_btc": None,
                "run_eth": None,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

