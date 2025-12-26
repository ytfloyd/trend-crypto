from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from math import sqrt

import polars as pl
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet


def load_equity(run_dir: Path) -> pl.DataFrame:
    path = run_dir / "equity.parquet"
    df = pl.read_parquet(path).sort("ts")
    if "net_ret" not in df.columns:
        df = df.with_columns((pl.col("nav") / pl.col("nav").shift(1) - 1).alias("net_ret"))
    return df


def metrics(df: pl.DataFrame) -> dict:
    df = df.sort("ts")
    nav = df["nav"]
    start = nav.item(0)
    end = nav.item(nav.len() - 1)
    total_return = (end / start) - 1 if start else 0.0
    returns = df["net_ret"]
    mean = returns.mean()
    std = returns.std(ddof=1)
    diffs = df.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds = diffs.median() if diffs.len() > 0 else 0
    periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
    sharpe = (mean / std) * (periods_per_year ** 0.5) if std and std > 0 else 0.0
    n_periods = returns.len()
    cagr = (end / start) ** (periods_per_year / n_periods) - 1 if start and n_periods > 0 else 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_dd = drawdowns.min()
    max_dd_idx = int(drawdowns.arg_min()) if drawdowns.len() > 0 else 0
    max_dd_ts = df["ts"].item(max_dd_idx) if drawdowns.len() > 0 else None
    vol_ann = std * sqrt(periods_per_year) if std is not None else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_ts": max_dd_ts.isoformat() if max_dd_ts else None,
        "vol_annual": vol_ann,
        "calmar": calmar,
    }


def draw_equity_and_dd(df: pl.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    eq_path = out_dir / "equity_plot.png"
    dd_path = out_dir / "dd_plot.png"
    nav = df["nav"]
    ts = df["ts"]
    plt.figure(figsize=(7, 3))
    plt.plot(ts.to_list(), nav.to_list(), label="Combined NAV")
    plt.yscale("log")
    plt.title("Equity Curve (log)")
    plt.tight_layout()
    eq_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(eq_path, dpi=150)
    plt.close()

    running_max = nav.cum_max()
    dd = (nav / running_max) - 1
    plt.figure(figsize=(7, 2.5))
    plt.plot(ts.to_list(), dd.to_list(), label="Drawdown")
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(dd_path, dpi=150)
    plt.close()
    return eq_path, dd_path


def monthly_table(df: pl.DataFrame) -> pl.DataFrame:
    if "net_ret" not in df.columns:
        df = df.with_columns((pl.col("nav") / pl.col("nav").shift(1) - 1).alias("net_ret"))
    df = df.with_columns(
        [
            pl.col("ts").dt.year().alias("year"),
            pl.col("ts").dt.month().alias("month"),
        ]
    )
    monthly = (
        df.group_by(["year", "month"])
        .agg((pl.col("net_ret").add(1).product() - 1).alias("ret"))
        .sort(["year", "month"])
    )
    return monthly


def crisis_stats(df: pl.DataFrame, start: str, end: str) -> dict:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    ts_schema = df.schema.get("ts")
    tz_aware = hasattr(ts_schema, "time_zone") and ts_schema.time_zone is not None
    if tz_aware:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    window = df.filter((pl.col("ts") >= pl.lit(start_dt)) & (pl.col("ts") <= pl.lit(end_dt)))
    if window.is_empty():
        return {"return": None, "max_dd": None, "worst_day": None, "days": 0}
    total_ret = window["nav"][-1] / window["nav"][0] - 1
    running_max = window["nav"].cum_max()
    dd = (window["nav"] / running_max) - 1
    max_dd = dd.min()
    worst_day = window["net_ret"].min()
    return {"return": total_ret, "max_dd": max_dd, "worst_day": worst_day, "days": len(window)}


def rolling_corr(series_a: pl.Series, series_b: pl.Series, window: int) -> pl.Series:
    # compute rolling correlation using covariance/variance
    df = pl.DataFrame({"a": series_a, "b": series_b})
    df = df.with_columns(
        [
            pl.col("a").rolling_mean(window_size=window, min_samples=window).alias("ma"),
            pl.col("b").rolling_mean(window_size=window, min_samples=window).alias("mb"),
            pl.col("a").rolling_var(window_size=window, min_samples=window).alias("va"),
            pl.col("b").rolling_var(window_size=window, min_samples=window).alias("vb"),
            (pl.col("a") * pl.col("b")).rolling_mean(window_size=window, min_samples=window).alias("mab"),
        ]
    )
    df = df.with_columns(
        (
            (pl.col("mab") - pl.col("ma") * pl.col("mb")) / (pl.col("va").sqrt() * pl.col("vb").sqrt())
        ).alias("corr")
    )
    return df["corr"]


def dt_lit(date_str: str, tz_aware: bool, end_of_day: bool = False) -> pl.Expr:
    dt = datetime.fromisoformat(date_str)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    if tz_aware:
        dt = dt.replace(tzinfo=timezone.utc)
    return pl.lit(dt)


def correlation_matrix(df_comb: pl.DataFrame, df_btc: pl.DataFrame, df_eth: pl.DataFrame, df_bench: pl.DataFrame) -> dict:
    joined = (
        df_comb.select(["ts", "net_ret"]).rename({"net_ret": "comb"})
        .join(df_btc.select(["ts", "net_ret"]).rename({"net_ret": "btc"}), on="ts", how="inner")
        .join(df_eth.select(["ts", "net_ret"]).rename({"net_ret": "eth"}), on="ts", how="inner")
        .join(df_bench.select(["ts", "net_ret"]).rename({"net_ret": "btc_bh"}), on="ts", how="inner")
    )
    corr = {}
    for a, b in [("comb", "btc_bh"), ("btc", "btc_bh"), ("eth", "btc_bh"), ("btc", "eth")]:
        if joined.height <= 2:
            corr[f"{a}_{b}"] = None
            continue
        val = (
            joined.select(pl.corr(pl.col(a), pl.col(b)).alias("corr"))
            .to_series()
            .item()
        )
        corr[f"{a}_{b}"] = float(val) if val is not None else None
    return corr


def build_pdf(args, comb_summary: dict, corr_stats: dict, eq_path: Path, dd_path: Path, monthly: pl.DataFrame, crisis: dict, rolling_corr_path: Path) -> None:
    doc = SimpleDocTemplate(args.out_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph("Core Trend Book V2.5 (TV=60 + Cash Yield)", styles["Title"]))
    elems.append(Paragraph(f"Report generated: {datetime.utcnow().isoformat()}Z", styles["Normal"]))
    elems.append(
        Paragraph(
            f"Runs: BTC={args.run_btc} | ETH={args.run_eth} | Combined={args.combined_dir} | Benchmark={args.benchmark_btc_bh}",
            styles["Normal"],
        )
    )
    elems.append(Spacer(1, 0.2 * inch))

    # Summary table
    summary_data = [
        ["Total Return", f"{comb_summary.get('total_return',0):.4f}", "CAGR", f"{comb_summary.get('cagr',0):.4f}"],
        ["Sharpe", f"{comb_summary.get('sharpe',0):.4f}", "Vol Ann", f"{comb_summary.get('vol_annual',0):.4f}"],
        ["Max DD", f"{comb_summary.get('max_drawdown',0):.4f}", "Calmar", f"{comb_summary.get('calmar',0):.4f}"],
        ["Turnover", f"{comb_summary.get('turnover_total',0):.4f}", "Turn Events", f"{comb_summary.get('turnover_events',0)}"],
        ["Edge/Turn bps", f"{comb_summary.get('edge_per_turnover_bps','n/a')}"],
    ]
    tbl = Table(summary_data)
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    elems.append(tbl)
    elems.append(Spacer(1, 0.2 * inch))

    # Equity and DD charts
    elems.append(Image(str(eq_path), width=6 * inch, height=3 * inch))
    elems.append(Spacer(1, 0.1 * inch))
    elems.append(Image(str(dd_path), width=6 * inch, height=2.5 * inch))
    elems.append(Spacer(1, 0.2 * inch))

    # Monthly table
    month_pivot = monthly.pivot(values="ret", index="year", columns="month")
    cols = ["Year"] + [str(m) for m in range(1, 13)]
    data = [cols]
    for row in month_pivot.iter_rows(named=True):
        line = [row["year"]]
        for m in range(1, 13):
            val = row.get(str(m))
            line.append(f"{val:.4f}" if val is not None else "")
        data.append(line)
    mt = Table(data)
    mt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    elems.append(Paragraph("Monthly Returns", styles["Heading3"]))
    elems.append(mt)
    elems.append(Spacer(1, 0.2 * inch))

    # Rolling correlation chart
    elems.append(Paragraph("Rolling Correlation vs BTC Buy&Hold", styles["Heading3"]))
    elems.append(Image(str(rolling_corr_path), width=6 * inch, height=2.5 * inch))
    elems.append(Spacer(1, 0.2 * inch))

    # Crisis stats
    elems.append(Paragraph("Crisis Windows", styles["Heading3"]))
    crisis_data = [
        ["Window", "Return", "MaxDD", "Worst Day", "Days", "Throttle (dd_scaler<1)"],
        ["Covid (2020-03-01 to 2020-04-30)", f"{crisis['covid']['return']:.4f}" if crisis['covid']['return'] is not None else "n/a",
         f"{crisis['covid']['max_dd']:.4f}" if crisis['covid']['max_dd'] is not None else "n/a",
         f"{crisis['covid']['worst_day']:.4f}" if crisis['covid']['worst_day'] is not None else "n/a",
         crisis['covid']['days'],
         crisis['covid'].get("throttle_info","n/a")],
        ["FTX (2022-11-01 to 2022-12-31)", f"{crisis['ftx']['return']:.4f}" if crisis['ftx']['return'] is not None else "n/a",
         f"{crisis['ftx']['max_dd']:.4f}" if crisis['ftx']['max_dd'] is not None else "n/a",
         f"{crisis['ftx']['worst_day']:.4f}" if crisis['ftx']['worst_day'] is not None else "n/a",
         crisis['ftx']['days'],
         crisis['ftx'].get("throttle_info","n/a")],
    ]
    ct = Table(crisis_data)
    ct.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    elems.append(ct)

    # Correlation matrix
    elems.append(Spacer(1, 0.2 * inch))
    elems.append(Paragraph("Correlation Matrix", styles["Heading3"]))
    corr_data = [["Pair", "Corr"]]
    for k, v in corr_stats.items():
        corr_data.append([k, f"{v:.4f}" if v is not None else "n/a"])
    corr_table = Table(corr_data)
    corr_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    elems.append(corr_table)

    doc.build(elems)


def main_pdf() -> None:
    parser = argparse.ArgumentParser(description="Generate tear sheet PDF for V2.5.")
    parser.add_argument("--run_btc", required=True)
    parser.add_argument("--run_eth", required=True)
    parser.add_argument("--combined_dir", required=True)
    parser.add_argument("--out_pdf", required=True)
    parser.add_argument("--benchmark_btc_bh", required=True)
    parser.add_argument("--rf_apy", type=float, default=0.04)
    parser.add_argument("--roll_corr_days", type=int, default=90)
    args = parser.parse_args()
    out_path = Path(args.out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    comb_eq = load_equity(Path(args.combined_dir))
    btc_eq = load_equity(Path(args.run_btc))
    eth_eq = load_equity(Path(args.run_eth))
    bench_eq = load_equity(Path(args.benchmark_btc_bh))

    comb_metrics = metrics(comb_eq)

    tz_aware = hasattr(comb_eq.schema.get("ts"), "time_zone") and comb_eq.schema.get("ts").time_zone is not None
    # crisis windows
    crisis = {
        "covid": crisis_stats(comb_eq, "2020-03-01", "2020-04-30"),
        "ftx": crisis_stats(comb_eq, "2022-11-01", "2022-12-31"),
    }
    # throttle info from combined (dd_scaler) if present
    for key, bounds in {
        "covid": ("2020-03-01", "2020-04-30"),
        "ftx": ("2022-11-01", "2022-12-31"),
    }.items():
        start_expr = dt_lit(bounds[0], tz_aware, end_of_day=False)
        end_expr = dt_lit(bounds[1], tz_aware, end_of_day=True)
        slice_eq = comb_eq.filter((pl.col("ts") >= start_expr) & (pl.col("ts") <= end_expr))
        throttle_days = 0
        min_dd_scaler = None
        if "dd_scaler" in slice_eq.columns and not slice_eq.is_empty():
            throttle_days = int((slice_eq["dd_scaler"] < 1.0).sum())
            min_dd_scaler = float(slice_eq["dd_scaler"].min())
        crisis[key]["throttle_info"] = f"throttle_days={throttle_days}, min_dd_scaler={min_dd_scaler}"

    # rolling correlation
    df_rc = comb_eq.join(bench_eq.select(["ts", "net_ret"]).rename({"net_ret": "r_bench"}), on="ts", how="inner")
    window = args.roll_corr_days
    roll_corr_series = rolling_corr(df_rc["net_ret"], df_rc["r_bench"], window)
    rc_path = Path(args.combined_dir) / "rolling_corr.png"
    plt.figure(figsize=(7, 2.5))
    plt.plot(df_rc["ts"].to_list(), roll_corr_series.to_list(), label="Rolling Corr")
    plt.title(f"Rolling Corr ({window}d) vs BTC B&H")
    plt.tight_layout()
    plt.savefig(rc_path, dpi=150)
    plt.close()

    corr_stats = correlation_matrix(comb_eq, btc_eq, eth_eq, bench_eq)

    monthly = monthly_table(comb_eq)
    eq_path, dd_path = draw_equity_and_dd(comb_eq, Path(args.combined_dir))

    args.out_pdf = str(out_path)
    build_pdf(args, comb_metrics, corr_stats, eq_path, dd_path, monthly, crisis, rc_path)


if __name__ == "__main__":
    main_pdf()

