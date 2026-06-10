#!/usr/bin/env python
"""Long/short MA crossover sweep on ETF and US stock daily lakes.

Backtest mechanics mirror ``scripts/research/cl_ma_crossover_multihorizon.py``:

  - signal at close[t]: +1 if fast_ma > slow_ma, -1 if fast_ma < slow_ma, else 0
  - held weight: w_held = w_signal.shift(1) (one-bar execution lag)
  - per-bar return: open[t+1] / open[t] - 1
  - turnover (one-sided): abs(w_held.diff())
  - cost: turnover * cost_bps / 1e4 charged on the open-to-open return

We use the daily ``open`` series for the realised return (close-driven signal,
next open execution) so we never look ahead. Daily bars assume ~252 bars/yr.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ETF_DB = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/etf_market.duckdb"
DEFAULT_STOCKS_DB = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/stocks_market.duckdb"
DEFAULT_OUT_DIR = "artifacts/research/etf_stocks_ma_crossover"

DEFAULT_PAIRS: list[tuple[int, int]] = [
    (5, 20),
    (5, 40),
    (10, 50),
    (20, 100),
    (50, 200),
]
DEFAULT_COSTS: list[float] = [0.0, 1.0, 5.0]
TARGET_PAIR: tuple[int, int] = (5, 40)
TARGET_COST: float = 1.0
DAILY_BARS_PER_YEAR: float = 252.0


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, object]
    equity: pd.Series  # indexed by ts, name = symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily long/short MA crossover sweep on ETF + US stock lakes."
    )
    parser.add_argument("--etf_db", default=DEFAULT_ETF_DB)
    parser.add_argument("--stocks_db", default=DEFAULT_STOCKS_DB)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--pairs",
        default=",".join(f"{f}:{s}" for f, s in DEFAULT_PAIRS),
        help="Comma-separated fast:slow MA pairs.",
    )
    parser.add_argument(
        "--costs",
        default=",".join(str(c) for c in DEFAULT_COSTS),
        help="Comma-separated cost grid in bps per side.",
    )
    parser.add_argument(
        "--max_symbols",
        type=int,
        default=0,
        help="Optional cap on symbols per asset class (0 = no cap; for smoke tests).",
    )
    return parser.parse_args()


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        fast_str, slow_str = chunk.split(":")
        fast, slow = int(fast_str), int(slow_str)
        if fast >= slow:
            raise ValueError(f"fast must be < slow, got {fast}:{slow}")
        pairs.append((fast, slow))
    return pairs


def parse_costs(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_panel(db_path: str) -> pd.DataFrame:
    """Load the bars_1d panel as a tidy DataFrame indexed by (symbol, ts)."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            "SELECT symbol, ts, open, close FROM bars_1d ORDER BY symbol, ts"
        ).fetchdf()
    finally:
        con.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def backtest_symbol(
    bars: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    cost_bps: float,
) -> BacktestResult | None:
    """Vectorized MA crossover backtest for a single symbol's daily bars.

    ``bars`` is a frame with columns ``ts, open, close`` sorted by ts. Returns
    ``None`` if there are insufficient bars or the strategy never trades.
    """
    if len(bars) < 2 * slow:
        return None

    ts = bars["ts"].to_numpy()
    open_px = bars["open"].astype(float).to_numpy()
    close_px = bars["close"].astype(float).to_numpy()

    # Use shifted close so that fast_ma / slow_ma at index t use info up to t-1.
    # Matches the reference implementation (close.shift(1).rolling(...)).
    close_s = pd.Series(close_px).shift(1)
    fast_ma = close_s.rolling(fast, min_periods=fast).mean().to_numpy()
    slow_ma = close_s.rolling(slow, min_periods=slow).mean().to_numpy()

    signal = np.zeros(len(bars), dtype=float)
    valid = ~np.isnan(fast_ma) & ~np.isnan(slow_ma)
    signal[valid & (fast_ma > slow_ma)] = 1.0
    signal[valid & (fast_ma < slow_ma)] = -1.0

    # Hold the signal one bar later (we observe at close, execute on next open).
    w_held = np.empty_like(signal)
    w_held[0] = 0.0
    w_held[1:] = signal[:-1]

    # Per-bar return: open[t+1]/open[t] - 1; final bar has no t+1, set 0.
    next_open = np.concatenate([open_px[1:], [np.nan]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = next_open / open_px - 1.0
    ret = np.where(np.isfinite(ret), ret, 0.0)

    # Turnover (one-sided): |w_held - w_held.shift(1)|, leading element vs 0.
    prev_w = np.concatenate([[0.0], w_held[:-1]])
    turnover = np.abs(w_held - prev_w)
    cost_ret = turnover * (cost_bps / 1e4)
    portfolio_ret = w_held * ret - cost_ret

    equity = (1.0 + portfolio_ret).cumprod()
    if equity.size == 0:
        return None
    drawdown = equity / np.maximum.accumulate(equity) - 1.0

    elapsed_days = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).total_seconds() / 86_400
    elapsed_years = elapsed_days / 365.25 if elapsed_days > 0 else 0.0
    bars_per_year = (
        len(bars) / elapsed_years if elapsed_years > 0 else DAILY_BARS_PER_YEAR
    )
    ret_std = float(np.std(portfolio_ret, ddof=0))
    ret_mean = float(np.mean(portfolio_ret))
    final_eq = float(equity[-1])
    total_return = final_eq - 1.0

    # Drop symbols that never traded (signal never flipped from 0).
    sig_changes = int((np.diff(np.concatenate([[0.0], signal])) != 0).sum())
    if sig_changes == 0 or total_return == 0.0:
        return None

    if ret_std > 0 and bars_per_year > 0:
        sharpe = ret_mean / ret_std * math.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    if elapsed_years > 0 and final_eq > 0:
        cagr = final_eq ** (1.0 / elapsed_years) - 1.0
    else:
        cagr = float("nan")

    metrics = {
        "n_bars": int(len(bars)),
        "start": str(pd.Timestamp(ts[0]).date()),
        "end": str(pd.Timestamp(ts[-1]).date()),
        "elapsed_years": float(elapsed_years),
        "bars_per_year_observed": float(bars_per_year),
        "final_equity": final_eq,
        "total_return": float(total_return),
        "cagr": float(cagr) if not math.isnan(cagr) else None,
        "vol": float(ret_std * math.sqrt(bars_per_year)) if bars_per_year > 0 else 0.0,
        "sharpe": float(sharpe),
        "max_dd": float(np.min(drawdown)),
        "avg_abs_exposure": float(np.mean(np.abs(w_held))),
        "long_pct": float((w_held > 0).mean()),
        "short_pct": float((w_held < 0).mean()),
        "active_pct": float((np.abs(w_held) > 0).mean()),
        "avg_turnover_one_sided": float(np.mean(turnover)),
        "signal_changes": sig_changes,
    }
    eq_series = pd.Series(equity, index=pd.DatetimeIndex(ts), name=str(bars.name))
    return BacktestResult(metrics=metrics, equity=eq_series)


def run_sweep(
    panel: pd.DataFrame,
    asset_class: str,
    pairs: list[tuple[int, int]],
    costs: list[float],
    max_symbols: int = 0,
) -> tuple[pd.DataFrame, dict[tuple[int, int, float, str], pd.Series]]:
    """Run all (pair, cost, symbol) backtests for one asset class.

    Returns a tidy metrics DataFrame and a dict mapping
    ``(fast, slow, cost_bps, symbol) -> equity series`` for the target pair/cost
    combo only (we don't keep equity curves for all 9k+ runs).
    """
    rows: list[dict[str, object]] = []
    equity_cache: dict[tuple[int, int, float, str], pd.Series] = {}
    symbols = panel["symbol"].unique().tolist()
    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]

    for sym in symbols:
        bars = panel.loc[panel["symbol"] == sym, ["ts", "open", "close"]].copy()
        bars = bars.dropna(subset=["open", "close"]).sort_values("ts")
        bars.name = sym  # used inside backtest_symbol for equity series name
        if bars.empty:
            continue
        for fast, slow in pairs:
            if len(bars) < 2 * slow:
                continue
            for cost in costs:
                res = backtest_symbol(bars, fast=fast, slow=slow, cost_bps=cost)
                if res is None:
                    continue
                row = {
                    "asset_class": asset_class,
                    "symbol": sym,
                    "fast": fast,
                    "slow": slow,
                    "cost_bps": cost,
                    **res.metrics,
                }
                rows.append(row)
                if (fast, slow) == TARGET_PAIR and cost == TARGET_COST:
                    equity_cache[(fast, slow, cost, sym)] = res.equity

    metrics = pd.DataFrame(rows)
    return metrics, equity_cache


def write_top10(metrics: pd.DataFrame, out_dir: Path) -> Path:
    target = metrics[metrics["cost_bps"] == TARGET_COST].copy()
    if target.empty:
        path = out_dir / "top10_per_pair_per_class.csv"
        target.to_csv(path, index=False)
        return path
    target = target.sort_values(
        ["asset_class", "fast", "slow", "sharpe"], ascending=[True, True, True, False]
    )
    top = (
        target.groupby(["asset_class", "fast", "slow"], group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )
    path = out_dir / "top10_per_pair_per_class.csv"
    top.to_csv(path, index=False)
    return path


def plot_sharpe_distribution(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """Boxplot of Sharpe by (fast, slow) pair, one box per asset class."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    target = metrics[metrics["cost_bps"] == TARGET_COST].copy()
    pairs = (
        target.drop_duplicates(["fast", "slow"])
        .sort_values(["slow", "fast"])
        .loc[:, ["fast", "slow"]]
        .to_records(index=False)
        .tolist()
    )
    asset_classes = ["etf", "stocks"]
    colors = {"etf": "#1f77b4", "stocks": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35
    positions: list[float] = []
    labels: list[str] = []
    for i, (fast, slow) in enumerate(pairs):
        labels.append(f"{fast}/{slow}")
        for j, ac in enumerate(asset_classes):
            sub = target[
                (target["fast"] == fast)
                & (target["slow"] == slow)
                & (target["asset_class"] == ac)
            ]["sharpe"].dropna()
            if sub.empty:
                continue
            pos = i + (j - 0.5) * width
            bp = ax.boxplot(
                sub.values,
                positions=[pos],
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
            )
            for box in bp["boxes"]:
                box.set_facecolor(colors[ac])
                box.set_alpha(0.55)
            for med in bp["medians"]:
                med.set_color("black")
            positions.append(pos)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("MA pair (fast/slow)")
    ax.set_ylabel(f"Annualized Sharpe (cost={TARGET_COST:g} bps)")
    ax.set_title("Sharpe distribution by MA pair (ETFs vs US stocks)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[ac], alpha=0.55) for ac in asset_classes
    ]
    ax.legend(handles, [ac.upper() for ac in asset_classes], loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = fig_dir / "01_sharpe_distribution_by_pair.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_top5_equity(
    metrics: pd.DataFrame,
    equity_cache: dict[tuple[int, int, float, str], pd.Series],
    asset_class: str,
    out_dir: Path,
    fname: str,
) -> Path | None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sub = metrics[
        (metrics["asset_class"] == asset_class)
        & (metrics["fast"] == TARGET_PAIR[0])
        & (metrics["slow"] == TARGET_PAIR[1])
        & (metrics["cost_bps"] == TARGET_COST)
    ].copy()
    if sub.empty:
        return None
    top5 = sub.sort_values("sharpe", ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(13, 7))
    for _, row in top5.iterrows():
        sym = row["symbol"]
        eq = equity_cache.get((TARGET_PAIR[0], TARGET_PAIR[1], TARGET_COST, sym))
        if eq is None or eq.empty:
            continue
        label = f"{sym} (Sharpe={row['sharpe']:.2f}, CAGR={row['cagr']:.2%})"
        ax.plot(eq.index, eq.values, linewidth=1.2, label=label)
    ax.set_yscale("log")
    ax.set_title(
        f"Top-5 {asset_class.upper()} {TARGET_PAIR[0]}/{TARGET_PAIR[1]} MA equity curves"
        f" (cost={TARGET_COST:g} bps)"
    )
    ax.set_ylabel("Strategy equity (log)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = fig_dir / fname
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_cost_sensitivity(metrics: pd.DataFrame, out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sub = metrics[
        (metrics["fast"] == TARGET_PAIR[0]) & (metrics["slow"] == TARGET_PAIR[1])
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    summary_rows = []
    for ac, color in [("etf", "#1f77b4"), ("stocks", "#d62728")]:
        agg = (
            sub[sub["asset_class"] == ac]
            .groupby("cost_bps")["sharpe"]
            .agg(["median", "mean", "count"])
            .reset_index()
        )
        if agg.empty:
            continue
        ax.plot(
            agg["cost_bps"],
            agg["median"],
            marker="o",
            color=color,
            label=f"{ac.upper()} median",
        )
        ax.plot(
            agg["cost_bps"],
            agg["mean"],
            marker="x",
            color=color,
            linestyle="--",
            alpha=0.6,
            label=f"{ac.upper()} mean",
        )
        for _, r in agg.iterrows():
            summary_rows.append(
                {
                    "asset_class": ac,
                    "cost_bps": r["cost_bps"],
                    "median_sharpe": r["median"],
                    "mean_sharpe": r["mean"],
                    "n_symbols": r["count"],
                }
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cost (bps per side)")
    ax.set_ylabel("Sharpe")
    ax.set_title(
        f"Cost sensitivity for {TARGET_PAIR[0]}/{TARGET_PAIR[1]} MA crossover"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "04_cost_sensitivity_pair_5_40.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "cost_sensitivity_pair_5_40.csv", index=False
    )
    return path


def write_readme(
    metrics: pd.DataFrame,
    universe_counts: dict[str, int],
    out_dir: Path,
    pairs: list[tuple[int, int]],
    costs: list[float],
) -> Path:
    target = metrics[metrics["cost_bps"] == TARGET_COST]
    median_sharpe = (
        target.groupby(["asset_class", "fast", "slow"])["sharpe"]
        .median()
        .reset_index()
        .sort_values(["asset_class", "sharpe"], ascending=[True, False])
    )

    best_lines: list[str] = []
    top_tables: dict[str, pd.DataFrame] = {}
    for ac in ["etf", "stocks"]:
        ms_ac = median_sharpe[median_sharpe["asset_class"] == ac]
        if ms_ac.empty:
            continue
        best_row = ms_ac.iloc[0]
        best_pair = (int(best_row["fast"]), int(best_row["slow"]))
        best_lines.append(
            f"- **{ac.upper()}** best pair by median Sharpe at "
            f"{TARGET_COST:g} bps: **{best_pair[0]}/{best_pair[1]}** "
            f"(median Sharpe {best_row['sharpe']:.2f}, "
            f"n={int((target['asset_class']==ac).sum() / max(len(pairs),1))})"
        )
        top5 = (
            target[
                (target["asset_class"] == ac)
                & (target["fast"] == TARGET_PAIR[0])
                & (target["slow"] == TARGET_PAIR[1])
            ]
            .sort_values("sharpe", ascending=False)
            .head(5)[["symbol", "sharpe", "cagr", "max_dd", "n_bars"]]
        )
        top_tables[ac] = top5

    pair_lines = "\n".join(
        f"- {f}/{s}" for f, s in sorted(pairs, key=lambda p: (p[1], p[0]))
    )
    cost_str = ", ".join(str(c) for c in costs)

    lines: list[str] = []
    lines.append("# ETF + US Stocks Long/Short MA Crossover Sweep")
    lines.append("")
    lines.append("## Universe and data")
    lines.append("")
    lines.append("- ETFs: `bars_1d` from `etf_market.duckdb` (read-only).")
    lines.append("- US stocks: `bars_1d` from `stocks_market.duckdb` (read-only).")
    lines.append(
        f"- After filtering symbols with fewer than `2*slow` bars or zero total "
        f"return: {universe_counts.get('etf', 0)} ETFs, "
        f"{universe_counts.get('stocks', 0)} stocks "
        f"(largest slow MA = {max(s for _, s in pairs)})."
    )
    lines.append("")
    lines.append("## Strategy")
    lines.append("")
    lines.append("Mechanics mirror `scripts/research/cl_ma_crossover_multihorizon.py`:")
    lines.append("")
    lines.append("- Signal at close[t]: `+1` if `fast_ma > slow_ma`, `-1` if `<`, else `0`.")
    lines.append("- Held weight: `w_held = w_signal.shift(1)` (no lookahead).")
    lines.append("- Per-bar return: `open[t+1]/open[t] - 1`.")
    lines.append("- Turnover one-sided: `|w_held.diff()|`, charged as `turnover * cost_bps / 1e4`.")
    lines.append("- Annualization: 252 bars/year on daily bars (using observed bars/year per symbol).")
    lines.append("")
    lines.append("Symbols are dropped if `n_bars < 2*slow` or if the strategy never traded "
                 "(`signal_changes == 0` or `total_return == 0`).")
    lines.append("")
    lines.append("### Parameter grid")
    lines.append("")
    lines.append("Fast/slow pairs:")
    lines.append("")
    lines.append(pair_lines)
    lines.append("")
    lines.append(f"Costs (bps per side): {cost_str}")
    lines.append("")
    lines.append("## Headline results (cost = 1 bps per side)")
    lines.append("")
    lines.extend(best_lines)
    lines.append("")
    lines.append("### Top-5 Sharpe at 5/40, 1 bps")
    for ac, top5 in top_tables.items():
        if top5.empty:
            continue
        lines.append("")
        lines.append(f"#### {ac.upper()}")
        lines.append("")
        lines.append("| symbol | sharpe | cagr | max_dd | n_bars |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in top5.iterrows():
            cagr_str = (
                f"{r['cagr']:.2%}" if pd.notna(r['cagr']) else "n/a"
            )
            lines.append(
                f"| {r['symbol']} | {r['sharpe']:.2f} | {cagr_str} | "
                f"{r['max_dd']:.2%} | {int(r['n_bars'])} |"
            )

    lines.append("")
    lines.append("## Median Sharpe by pair / asset class @ 1 bps")
    lines.append("")
    lines.append("| asset_class | fast | slow | median_sharpe |")
    lines.append("|---|---:|---:|---:|")
    for _, r in median_sharpe.iterrows():
        lines.append(
            f"| {r['asset_class']} | {int(r['fast'])} | {int(r['slow'])} | "
            f"{r['sharpe']:.2f} |"
        )

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `metrics.csv` - tidy per (asset_class, symbol, fast, slow, cost_bps) metrics.")
    lines.append("- `top10_per_pair_per_class.csv` - top-10 Sharpe per pair per class @ 1 bps.")
    lines.append("- `cost_sensitivity_pair_5_40.csv` - aggregate Sharpe vs cost at 5/40.")
    lines.append("- `figures/01_sharpe_distribution_by_pair.png` - Sharpe boxplots.")
    lines.append("- `figures/02_top5_equity_curves_etf_pair_5_40_cost1.png` - top ETFs at 5/40.")
    lines.append("- `figures/03_top5_equity_curves_stocks_pair_5_40_cost1.png` - top stocks at 5/40.")
    lines.append("- `figures/04_cost_sensitivity_pair_5_40.png` - median Sharpe vs cost at 5/40.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- **Survivor bias.** The stock universe is the current `bars_1d` "
                 "membership in `stocks_market.duckdb`; delisted names are not present, "
                 "so per-symbol Sharpe will be biased upward versus a true point-in-time "
                 "universe. ETF survivorship should be milder but is also not corrected.")
    lines.append("- **Short selling assumed frictionless.** Hard-to-borrow rates and locate "
                 "constraints are ignored. The per-side cost grid is a generic proxy.")
    lines.append("- **Daily open execution.** Returns are computed open-to-open with a "
                 "one-bar execution lag; intraday slippage and queue dynamics are not modelled.")
    lines.append("- **Single-name results, not a portfolio.** Sharpe ratios reported here are "
                 "per-symbol; a diversified portfolio would have different (typically higher) "
                 "Sharpe driven by cross-sectional dispersion.")
    lines.append("- **Variable history.** Some stocks have very short histories "
                 "(e.g. recent IPOs / spinoffs); the `2*slow` filter prunes the worst cases "
                 "but per-symbol stats are still noisier for short series.")

    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(args.pairs)
    costs = parse_costs(args.costs)
    print(f"[ma_sweep] pairs={pairs} costs={costs}")

    print(f"[ma_sweep] loading ETF panel from {args.etf_db} ...")
    etf_panel = load_panel(args.etf_db)
    print(
        f"[ma_sweep]   etf rows={len(etf_panel):,} symbols={etf_panel['symbol'].nunique()}"
    )

    print(f"[ma_sweep] loading stocks panel from {args.stocks_db} ...")
    stk_panel = load_panel(args.stocks_db)
    print(
        f"[ma_sweep]   stocks rows={len(stk_panel):,} "
        f"symbols={stk_panel['symbol'].nunique()}"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        etf_metrics, etf_equity = run_sweep(
            etf_panel, "etf", pairs, costs, max_symbols=args.max_symbols
        )
        stk_metrics, stk_equity = run_sweep(
            stk_panel, "stocks", pairs, costs, max_symbols=args.max_symbols
        )

    metrics = pd.concat([etf_metrics, stk_metrics], ignore_index=True)
    metrics_path = out_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    universe_counts = {
        "etf": metrics[metrics["asset_class"] == "etf"]["symbol"].nunique(),
        "stocks": metrics[metrics["asset_class"] == "stocks"]["symbol"].nunique(),
    }

    top10_path = write_top10(metrics, out_dir)
    fig1 = plot_sharpe_distribution(metrics, out_dir)
    fig2 = plot_top5_equity(
        metrics,
        etf_equity,
        "etf",
        out_dir,
        "02_top5_equity_curves_etf_pair_5_40_cost1.png",
    )
    fig3 = plot_top5_equity(
        metrics,
        stk_equity,
        "stocks",
        out_dir,
        "03_top5_equity_curves_stocks_pair_5_40_cost1.png",
    )
    fig4 = plot_cost_sensitivity(metrics, out_dir)
    readme_path = write_readme(metrics, universe_counts, out_dir, pairs, costs)

    print("[ma_sweep] universe counts (post-filter):")
    for ac, n in universe_counts.items():
        print(f"  {ac}: {n}")
    print(f"[ma_sweep] metrics: {metrics_path}")
    print(f"[ma_sweep] top10:   {top10_path}")
    print(f"[ma_sweep] figs:    {fig1}, {fig2}, {fig3}, {fig4}")
    print(f"[ma_sweep] readme:  {readme_path}")


if __name__ == "__main__":
    main()
