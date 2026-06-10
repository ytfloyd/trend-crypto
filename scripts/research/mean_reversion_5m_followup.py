#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


BARS_PER_YEAR_5M = 365.0 * 24 * 12
VARIANT = "mr_12h_z2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow-up diagnostics for cleaned 5m mean reversion.")
    parser.add_argument("--run_dir", default="artifacts/research/mean_reversion_5m_universe_clean_v4")
    parser.add_argument("--signals_dir", default="artifacts/research/trend_fixed_atr_5m_universe/signals_by_symbol")
    parser.add_argument("--out_dir", default="artifacts/research/mean_reversion_5m_followup")
    parser.add_argument("--cost_grid_bps", default="0,1,2,5,10,20,30")
    parser.add_argument("--liquidity_tiers", default="25,50,100,200")
    return parser.parse_args()


def _metrics_from_returns(
    frame: pd.DataFrame,
    ret_col: str = "portfolio_ret",
    *,
    variant: str | None = None,
    label: str | None = None,
) -> dict[str, object]:
    if frame.empty:
        return {}
    g = frame.sort_values("ts").copy()
    ret = g[ret_col].astype(float)
    eq = (1.0 + ret).cumprod()
    dd = eq / eq.cummax() - 1.0
    elapsed_years = (pd.to_datetime(g["ts"]).iloc[-1] - pd.to_datetime(g["ts"]).iloc[0]).total_seconds()
    elapsed_years = max(elapsed_years / (365.0 * 24 * 3600), len(g) / BARS_PER_YEAR_5M)
    ret_std = ret.std(ddof=0)
    final_equity = float(eq.iloc[-1])
    return {
        "variant": variant if variant is not None else str(g["variant"].iloc[0]) if "variant" in g else "",
        "label": label or "",
        "start": str(pd.to_datetime(g["ts"]).iloc[0]),
        "end": str(pd.to_datetime(g["ts"]).iloc[-1]),
        "n_bars": int(len(g)),
        "final_equity": final_equity,
        "cagr": final_equity ** (1.0 / elapsed_years) - 1.0 if final_equity > 0 else float("nan"),
        "vol": float(ret_std * math.sqrt(BARS_PER_YEAR_5M)),
        "sharpe": float(ret.mean() / ret_std * math.sqrt(BARS_PER_YEAR_5M)) if ret_std > 0 else 0.0,
        "max_dd": float(dd.min()),
        "avg_gross": float(g["gross_exposure"].mean()) if "gross_exposure" in g else float("nan"),
        "avg_n_held": float(g["n_held"].mean()) if "n_held" in g else float("nan"),
        "avg_turnover_one_sided": float(g["turnover_one_sided"].mean()) if "turnover_one_sided" in g else float("nan"),
    }


def cost_sensitivity(portfolio: pd.DataFrame, cost_grid: list[float], out_dir: Path) -> pd.DataFrame:
    records = []
    for cost_bps in cost_grid:
        cost_rate = cost_bps / 10_000.0
        for variant, group in portfolio.groupby("variant", sort=True):
            g = group.copy()
            g["net_ret"] = g["gross_asset_ret"] - g["turnover_one_sided"] * cost_rate
            rec = _metrics_from_returns(g, "net_ret", variant=variant, label=f"{cost_bps:g}bps")
            rec["cost_bps"] = cost_bps
            records.append(rec)
    result = pd.DataFrame(records)
    result.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    return result


def annual_breakdown(portfolio: pd.DataFrame, cost_bps: float, out_dir: Path) -> pd.DataFrame:
    records = []
    cost_rate = cost_bps / 10_000.0
    focus = portfolio[portfolio["variant"] == VARIANT].copy()
    focus["net_ret"] = focus["gross_asset_ret"] - focus["turnover_one_sided"] * cost_rate
    focus["year"] = pd.to_datetime(focus["ts"]).dt.year
    for year, group in focus.groupby("year", sort=True):
        rec = _metrics_from_returns(group, "net_ret", variant=VARIANT, label=str(year))
        rec["year"] = int(year)
        rec["cost_bps"] = cost_bps
        records.append(rec)
    result = pd.DataFrame(records)
    result.to_csv(out_dir / f"annual_breakdown_{cost_bps:g}bps.csv", index=False)
    return result


def event_breakdown(portfolio: pd.DataFrame, cost_bps: float, out_dir: Path) -> pd.DataFrame:
    periods = [
        ("active_start", "2020-08-13", "2020-12-31"),
        ("2021_bull_and_may_crash", "2021-01-01", "2021-06-30"),
        ("2021_h2", "2021-07-01", "2021-12-31"),
        ("2022_bear", "2022-01-01", "2022-12-31"),
        ("2023_chop", "2023-01-01", "2023-12-31"),
        ("2024_bull", "2024-01-01", "2024-12-31"),
        ("2025_2026", "2025-01-01", "2026-05-24"),
    ]
    records = []
    cost_rate = cost_bps / 10_000.0
    focus = portfolio[portfolio["variant"] == VARIANT].copy()
    focus["net_ret"] = focus["gross_asset_ret"] - focus["turnover_one_sided"] * cost_rate
    ts = pd.to_datetime(focus["ts"])
    for name, start, end in periods:
        mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
        rec = _metrics_from_returns(focus.loc[mask], "net_ret", variant=VARIANT, label=name)
        if rec:
            rec["period"] = name
            rec["cost_bps"] = cost_bps
            records.append(rec)
    result = pd.DataFrame(records)
    result.to_csv(out_dir / f"event_breakdown_{cost_bps:g}bps.csv", index=False)
    return result


def compute_symbol_liquidity(signals_dir: Path, symbols: set[str], out_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(signals_dir.glob("*.parquet")):
        df0 = pd.read_parquet(path, columns=["symbol"])
        symbol = str(df0.iloc[0, 0])
        if symbol not in symbols:
            continue
        df = pd.read_parquet(path, columns=["ts", "symbol", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df[df["ts"] >= pd.Timestamp("2020-08-13")]
        dollar_volume = df["close"].astype(float) * df["volume"].astype(float)
        rolling_24h = dollar_volume.rolling(288, min_periods=144).sum().shift(1)
        records.append(
            {
                "symbol": symbol,
                "median_rolling_24h_dollar_volume": float(rolling_24h.median()),
                "mean_rolling_24h_dollar_volume": float(rolling_24h.mean()),
                "bars": int(len(df)),
            }
        )
    result = pd.DataFrame(records).sort_values("median_rolling_24h_dollar_volume", ascending=False)
    result.to_csv(out_dir / "symbol_liquidity.csv", index=False)
    return result


def liquidity_tier_backtests(
    run_dir: Path,
    signals_dir: Path,
    tiers: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    positions_glob = (run_dir / "positions_by_symbol" / "*.parquet").as_posix()
    con = duckdb.connect()
    symbols = {
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT symbol FROM read_parquet('{positions_glob}', union_by_name = TRUE)"
        ).fetchall()
    }
    liquidity = compute_symbol_liquidity(signals_dir, symbols, out_dir)
    records = []
    equity_frames = []

    pos = con.execute(
        f"""
        SELECT ts, symbol, variant, ret_oc, active
        FROM read_parquet('{positions_glob}', union_by_name = TRUE)
        WHERE variant = '{VARIANT}'
        """
    ).fetch_df()
    pos["ts"] = pd.to_datetime(pos["ts"])

    for tier in tiers:
        top_symbols = set(liquidity.head(tier)["symbol"])
        p = pos[pos["symbol"].isin(top_symbols)].copy()
        p["n_active"] = p.groupby("ts")["active"].transform("sum")
        p["w_signal"] = 0.0
        mask = p["n_active"] >= 10
        p.loc[mask, "w_signal"] = p.loc[mask, "active"] * (1.0 / p.loc[mask, "n_active"]).clip(upper=0.05)
        p = p.sort_values(["symbol", "ts"])
        p["w_held"] = p.groupby("symbol")["w_signal"].shift(1).fillna(0.0)
        p["w_prev_held"] = p.groupby("symbol")["w_signal"].shift(2).fillna(0.0)
        by_ts = (
            p.groupby("ts")
            .apply(
                lambda g: pd.Series(
                    {
                        "gross_asset_ret": float((g["w_held"] * g["ret_oc"].fillna(0.0)).sum()),
                        "gross_exposure": float(g["w_held"].abs().sum()),
                        "turnover_one_sided": float((g["w_held"] - g["w_prev_held"]).abs().sum()),
                        "n_held": float((g["w_held"] > 0).sum()),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        by_ts["variant"] = VARIANT
        by_ts["tier"] = f"top_{tier}"
        by_ts["portfolio_ret"] = by_ts["gross_asset_ret"]
        by_ts["portfolio_equity"] = (1.0 + by_ts["portfolio_ret"]).cumprod()
        rec = _metrics_from_returns(by_ts, "portfolio_ret", variant=VARIANT, label=f"top_{tier}")
        rec["tier"] = tier
        records.append(rec)
        equity_frames.append(by_ts)

    metrics = pd.DataFrame(records).sort_values("tier")
    metrics.to_csv(out_dir / "liquidity_tier_metrics.csv", index=False)
    pd.concat(equity_frames, ignore_index=True).to_csv(out_dir / "liquidity_tier_equity.csv", index=False)
    con.close()
    return metrics


def write_figures(costs: pd.DataFrame, tiers: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    focus = costs[costs["variant"] == VARIANT].sort_values("cost_bps")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(focus["cost_bps"], focus["sharpe"], marker="o")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("mr_12h_z2 Sharpe vs cost")
    axes[0].set_xlabel("One-sided cost (bps)")
    axes[0].set_ylabel("Annualized Sharpe")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(focus["cost_bps"], focus["cagr"], marker="o")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("mr_12h_z2 CAGR vs cost")
    axes[1].set_xlabel("One-sided cost (bps)")
    axes[1].set_ylabel("CAGR")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_cost_sensitivity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f"top {int(x)}" for x in tiers["tier"]], tiers["sharpe"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("mr_12h_z2 Sharpe by liquidity tier, zero cost")
    ax.set_ylabel("Annualized Sharpe")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_liquidity_tiers.png", dpi=160)
    plt.close(fig)


def render_report(
    out_dir: Path,
    costs: pd.DataFrame,
    annual0: pd.DataFrame,
    annual10: pd.DataFrame,
    events10: pd.DataFrame,
    tiers: pd.DataFrame,
) -> None:
    focus = costs[costs["variant"] == VARIANT].sort_values("cost_bps")
    best_zero = focus.iloc[0]
    cost10 = focus[focus["cost_bps"] == 10].iloc[0]
    lines = [
        "# 5-Minute Mean-Reversion Follow-Up",
        "",
        "**Source run:** `artifacts/research/mean_reversion_5m_universe_clean_v4/`",
        f"**Focus variant:** `{VARIANT}`",
        "",
        "## Headline",
        "",
        f"- Zero-cost Sharpe: `{best_zero['sharpe']:.2f}`, CAGR: `{best_zero['cagr']:+.1%}`, max DD: `{best_zero['max_dd']:+.1%}`.",
        f"- At 10 bps one-sided cost: Sharpe `{cost10['sharpe']:.2f}`, CAGR `{cost10['cagr']:+.1%}`, max DD `{cost10['max_dd']:+.1%}`.",
        "- Cost sensitivity, annual/event splits, and liquidity-tier checks are written as CSV artifacts.",
        "",
        "## Interpretation",
        "",
        "The signal remains interesting at zero cost but is cost-sensitive. Liquidity tiers help identify whether the edge is broad or a long-tail artifact; any production version should be judged on top-liquidity subsets and conservative execution costs.",
        "",
        "## Artifacts",
        "",
        "- `cost_sensitivity.csv`",
        "- `annual_breakdown_0bps.csv`, `annual_breakdown_10bps.csv`",
        "- `event_breakdown_10bps.csv`",
        "- `liquidity_tier_metrics.csv`, `liquidity_tier_equity.csv`",
        "- `symbol_liquidity.csv`",
        "- `figures/01_cost_sensitivity.png`, `figures/02_liquidity_tiers.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    summary = {
        "zero_cost": best_zero.to_dict(),
        "cost_10bps": cost10.to_dict(),
        "annual_0bps": annual0.to_dict(orient="records"),
        "annual_10bps": annual10.to_dict(orient="records"),
        "events_10bps": events10.to_dict(orient="records"),
        "liquidity_tiers": tiers.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_grid = [float(x) for x in args.cost_grid_bps.split(",") if x.strip()]
    tiers = [int(x) for x in args.liquidity_tiers.split(",") if x.strip()]

    con = duckdb.connect(str(run_dir / "portfolio.duckdb"), read_only=True)
    portfolio = con.execute("SELECT * FROM portfolio ORDER BY variant, ts").fetch_df()
    con.close()
    portfolio["ts"] = pd.to_datetime(portfolio["ts"])

    costs = cost_sensitivity(portfolio, cost_grid, out_dir)
    annual0 = annual_breakdown(portfolio, 0.0, out_dir)
    annual10 = annual_breakdown(portfolio, 10.0, out_dir)
    events10 = event_breakdown(portfolio, 10.0, out_dir)
    tier_metrics = liquidity_tier_backtests(run_dir, Path(args.signals_dir), tiers, out_dir)
    write_figures(costs, tier_metrics, out_dir)
    render_report(out_dir, costs, annual0, annual10, events10, tier_metrics)

    print(f"[mean_reversion_5m_followup] wrote artifacts to {out_dir}")
    print("\nCost sensitivity:")
    print(costs[costs["variant"] == VARIANT].to_string(index=False))
    print("\nLiquidity tiers:")
    print(tier_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
