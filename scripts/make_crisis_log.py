from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl


CRISES = [
    ("China Ban Crash", "2021-05-19T00:00:00Z"),
    ("FTX Collapse", "2022-11-08T00:00:00Z"),
]


def parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_portfolio(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "ts" not in df.columns or "nav" not in df.columns:
        raise ValueError("portfolio_parquet must contain columns ts and nav")
    ts_dtype = df.schema["ts"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
        df = df.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    elif isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        df = df.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
    return df.sort("ts")


def nearest_leq(df: pl.DataFrame, ts: datetime) -> pl.DataFrame:
    sub = df.filter(pl.col("ts") <= ts)
    if sub.is_empty():
        return pl.DataFrame()
    return sub.tail(1)


def max_drawdown(nav_series: pl.Series) -> float:
    running_max = nav_series.cum_max()
    dd = nav_series / running_max - 1
    return dd.min()


def analyze_event(df: pl.DataFrame, name: str, ts_event: datetime, window_hours: int) -> dict:
    start = ts_event - timedelta(hours=window_hours)
    end = ts_event + timedelta(hours=window_hours)
    window_df = df.filter((pl.col("ts") >= start) & (pl.col("ts") <= end)).sort("ts")
    if window_df.is_empty():
        return {
            "event": name,
            "event_ts": ts_event.isoformat(),
            "nav_start": None,
            "nav_event": None,
            "nav_end": None,
            "max_drawdown": None,
            "panic_triggered": False,
            "corr_override_triggered": False,
            "funding_exit_triggered": False,
            "avg_leverage": None,
            "leverage_at_event": None,
        }

    nav_start = window_df.select(pl.col("nav").first()).item()
    nav_end = window_df.select(pl.col("nav").last()).item()
    event_row = nearest_leq(window_df, ts_event)
    nav_event = event_row["nav"].item() if not event_row.is_empty() else None

    max_dd = max_drawdown(window_df["nav"])

    panic_flag = (
        window_df["panic_triggered"].any() if "panic_triggered" in window_df.columns else False
    )
    corr_flag = (
        window_df["corr_override_triggered"].any()
        if "corr_override_triggered" in window_df.columns
        else False
    )
    funding_flag = (
        window_df["funding_exit_triggered"].any()
        if "funding_exit_triggered" in window_df.columns
        else False
    )

    avg_leverage = window_df["leverage_scalar_final"].mean() if "leverage_scalar_final" in window_df.columns else None
    lev_event = (
        event_row["leverage_scalar_final"].item()
        if "leverage_scalar_final" in event_row.columns and not event_row.is_empty()
        else None
    )

    return {
        "event": name,
        "event_ts": ts_event.isoformat(),
        "nav_start": nav_start,
        "nav_event": nav_event,
        "nav_end": nav_end,
        "max_drawdown": max_dd,
        "panic_triggered": panic_flag,
        "corr_override_triggered": corr_flag,
        "funding_exit_triggered": funding_flag,
        "avg_leverage": avg_leverage,
        "leverage_at_event": lev_event,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Make crisis log table from portfolio parquet.")
    parser.add_argument("--portfolio_parquet", required=True, help="Path to portfolio_equity.parquet")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--window_hours", type=int, default=72, help="Hours before/after event to analyze")
    args = parser.parse_args()

    df = load_portfolio(Path(args.portfolio_parquet))
    results = []
    for name, ts in CRISES:
        ts_dt = parse_ts(ts)
        res = analyze_event(df, name, ts_dt, args.window_hours)
        results.append(res)

    out_df = pl.DataFrame(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)

    print(out_df)
    print(f"Wrote crisis log to {out_path}")


if __name__ == "__main__":
    main()

