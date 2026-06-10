#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


BARS_PER_DAY_5M = 24 * 12
BARS_PER_YEAR_5M = 365.0 * BARS_PER_DAY_5M
VARIANT = "mr_12h_z2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime-gate cleaned 5m mean-reversion weights.")
    parser.add_argument("--run_dir", default="artifacts/research/mean_reversion_5m_universe_clean_v4")
    parser.add_argument(
        "--btc_file",
        default="artifacts/research/trend_fixed_atr_5m_universe/signals_by_symbol/BTC_USD.parquet",
    )
    parser.add_argument("--out_dir", default="artifacts/research/mean_reversion_5m_regime_gating")
    parser.add_argument("--cost_grid_bps", default="0,5,10,20")
    return parser.parse_args()


def compute_btc_gates(btc_file: Path) -> pd.DataFrame:
    btc = pd.read_parquet(btc_file, columns=["ts", "close"])
    btc["ts"] = pd.to_datetime(btc["ts"])
    btc = btc.sort_values("ts").reset_index(drop=True)
    close = btc["close"].astype(float)

    sma_7d = close.shift(1).rolling(7 * BARS_PER_DAY_5M, min_periods=3 * BARS_PER_DAY_5M).mean()
    sma_30d = close.shift(1).rolling(30 * BARS_PER_DAY_5M, min_periods=10 * BARS_PER_DAY_5M).mean()
    max_30d = close.shift(1).rolling(30 * BARS_PER_DAY_5M, min_periods=10 * BARS_PER_DAY_5M).max()
    ret_24h = close / close.shift(BARS_PER_DAY_5M) - 1.0
    ret_7d = close / close.shift(7 * BARS_PER_DAY_5M) - 1.0
    dd_30d = close / max_30d - 1.0

    gates = pd.DataFrame({"ts": btc["ts"]})
    gates["ungated"] = True
    gates["btc_7d_up"] = close > sma_7d
    gates["btc_30d_up"] = close > sma_30d
    gates["drawdown_guard"] = (dd_30d > -0.20) & (ret_24h > -0.08) & (ret_7d > -0.20)
    gates["btc_7d_up_or_rebound"] = gates["btc_7d_up"] | (ret_24h > 0.03)
    gates["btc_30d_up_and_guard"] = gates["btc_30d_up"] & gates["drawdown_guard"]
    for col in gates.columns:
        if col != "ts":
            gates[col] = gates[col].fillna(False).astype(bool)
    return gates


def metrics_from_returns(frame: pd.DataFrame, ret_col: str, *, gate: str, cost_bps: float) -> dict[str, object]:
    g = frame.sort_values("ts").copy()
    ret = g[ret_col].astype(float)
    eq = (1.0 + ret).cumprod()
    dd = eq / eq.cummax() - 1.0
    elapsed_years = (pd.to_datetime(g["ts"]).iloc[-1] - pd.to_datetime(g["ts"]).iloc[0]).total_seconds()
    elapsed_years = max(elapsed_years / (365.0 * 24 * 3600), len(g) / BARS_PER_YEAR_5M)
    ret_std = ret.std(ddof=0)
    final_equity = float(eq.iloc[-1])
    return {
        "gate": gate,
        "cost_bps": cost_bps,
        "start": str(pd.to_datetime(g["ts"]).iloc[0]),
        "end": str(pd.to_datetime(g["ts"]).iloc[-1]),
        "n_bars": int(len(g)),
        "final_equity": final_equity,
        "cagr": final_equity ** (1.0 / elapsed_years) - 1.0 if final_equity > 0 else float("nan"),
        "vol": float(ret_std * math.sqrt(BARS_PER_YEAR_5M)),
        "sharpe": float(ret.mean() / ret_std * math.sqrt(BARS_PER_YEAR_5M)) if ret_std > 0 else 0.0,
        "max_dd": float(dd.min()),
        "avg_gross": float(g["gross_exposure"].mean()),
        "active_avg_gross": float(g.loc[g["gross_exposure"] > 0, "gross_exposure"].mean()),
        "avg_n_held": float(g["n_held"].mean()),
        "avg_turnover_one_sided": float(g["turnover_one_sided"].mean()),
        "gate_on_pct": float(g["gate_on"].mean()),
    }


def build_gated_portfolio(
    run_dir: Path,
    gates: pd.DataFrame,
    gate_col: str,
    out_dir: Path,
) -> pd.DataFrame:
    con = duckdb.connect(str(run_dir / "portfolio.duckdb"), read_only=True)
    gate_df = gates[["ts", gate_col]].rename(columns={gate_col: "gate_on"}).copy()
    con.register("gate_df", gate_df)
    portfolio = con.execute(
        f"""
        WITH base AS (
            SELECT w.ts, w.symbol, w.ret_oc, w.w_signal, COALESCE(g.gate_on, FALSE) AS gate_on
            FROM weights w
            LEFT JOIN gate_df g USING(ts)
            WHERE w.variant = '{VARIANT}'
        ),
        signal AS (
            SELECT ts, symbol, ret_oc, gate_on,
                   CASE WHEN gate_on THEN w_signal ELSE 0.0 END AS w_signal_gated
            FROM base
        ),
        held AS (
            SELECT ts, symbol, ret_oc, gate_on,
                   LAG(w_signal_gated, 1, 0.0) OVER (PARTITION BY symbol ORDER BY ts) AS w_held,
                   LAG(w_signal_gated, 2, 0.0) OVER (PARTITION BY symbol ORDER BY ts) AS w_prev_held
            FROM signal
        )
        SELECT
            ts,
            SUM(w_held * COALESCE(ret_oc, 0.0)) AS gross_asset_ret,
            SUM(ABS(w_held)) AS gross_exposure,
            SUM(ABS(w_held - w_prev_held)) AS turnover_one_sided,
            SUM(CASE WHEN w_held > 0 THEN 1 ELSE 0 END) AS n_held,
            BOOL_OR(gate_on) AS gate_on
        FROM held
        GROUP BY ts
        ORDER BY ts
        """
    ).fetch_df()
    con.close()
    portfolio["gate"] = gate_col
    portfolio["portfolio_ret"] = portfolio["gross_asset_ret"]
    portfolio["portfolio_equity"] = (1.0 + portfolio["portfolio_ret"]).cumprod()
    return portfolio


def cost_metrics(portfolios: pd.DataFrame, cost_grid: list[float], out_dir: Path) -> pd.DataFrame:
    records = []
    for cost_bps in cost_grid:
        cost_rate = cost_bps / 10_000.0
        for gate, group in portfolios.groupby("gate", sort=True):
            g = group.copy()
            g["net_ret"] = g["gross_asset_ret"] - g["turnover_one_sided"] * cost_rate
            records.append(metrics_from_returns(g, "net_ret", gate=gate, cost_bps=cost_bps))
    result = pd.DataFrame(records).sort_values(["cost_bps", "sharpe"], ascending=[True, False])
    result.to_csv(out_dir / "regime_gate_cost_metrics.csv", index=False)
    return result


def annual_metrics(portfolios: pd.DataFrame, cost_bps: float, out_dir: Path) -> pd.DataFrame:
    records = []
    cost_rate = cost_bps / 10_000.0
    p = portfolios.copy()
    p["net_ret"] = p["gross_asset_ret"] - p["turnover_one_sided"] * cost_rate
    p["year"] = pd.to_datetime(p["ts"]).dt.year
    for (gate, year), group in p.groupby(["gate", "year"], sort=True):
        rec = metrics_from_returns(group, "net_ret", gate=gate, cost_bps=cost_bps)
        rec["year"] = int(year)
        records.append(rec)
    result = pd.DataFrame(records)
    result.to_csv(out_dir / f"regime_gate_annual_{cost_bps:g}bps.csv", index=False)
    return result


def write_figures(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for cost_bps in [0.0, 10.0]:
        m = metrics[metrics["cost_bps"] == cost_bps].sort_values("sharpe")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(m["gate"], m["sharpe"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Regime gate Sharpe at {cost_bps:g} bps one-sided cost")
        ax.set_xlabel("Annualized Sharpe")
        fig.tight_layout()
        fig.savefig(fig_dir / f"gate_sharpe_{cost_bps:g}bps.png", dpi=160)
        plt.close(fig)


def render_report(metrics: pd.DataFrame, out_dir: Path) -> None:
    m0 = metrics[metrics["cost_bps"] == 0].sort_values("sharpe", ascending=False)
    m10 = metrics[metrics["cost_bps"] == 10].sort_values("sharpe", ascending=False)
    best0 = m0.iloc[0]
    best10 = m10.iloc[0]
    ungated10 = m10[m10["gate"] == "ungated"].iloc[0]
    lines = [
        "# 5-Minute Mean-Reversion Regime Gating",
        "",
        f"**Focus variant:** `{VARIANT}`",
        "**Source:** cleaned v4 mean-reversion weights",
        "",
        "## Summary",
        "",
        f"- Best zero-cost gate: `{best0['gate']}` with Sharpe `{best0['sharpe']:.2f}`, CAGR `{best0['cagr']:+.1%}`, max DD `{best0['max_dd']:+.1%}`.",
        f"- Best 10 bps gate: `{best10['gate']}` with Sharpe `{best10['sharpe']:.2f}`, CAGR `{best10['cagr']:+.1%}`, max DD `{best10['max_dd']:+.1%}`.",
        f"- Ungated at 10 bps: Sharpe `{ungated10['sharpe']:.2f}`, CAGR `{ungated10['cagr']:+.1%}`, max DD `{ungated10['max_dd']:+.1%}`.",
        "",
        "## Artifacts",
        "",
        "- `regime_gate_cost_metrics.csv`",
        "- `regime_gate_annual_0bps.csv`, `regime_gate_annual_10bps.csv`",
        "- `regime_gate_equity.csv`",
        "- `btc_regime_gates.csv`",
        "- `figures/gate_sharpe_0bps.png`, `figures/gate_sharpe_10bps.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_zero_cost": best0.to_dict(),
                "best_10bps": best10.to_dict(),
                "ungated_10bps": ungated10.to_dict(),
            },
            indent=2,
            default=str,
        )
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_grid = [float(x) for x in args.cost_grid_bps.split(",") if x.strip()]

    gates = compute_btc_gates(Path(args.btc_file))
    gates.to_csv(out_dir / "btc_regime_gates.csv", index=False)
    gate_cols = [c for c in gates.columns if c != "ts"]

    frames = []
    for gate_col in gate_cols:
        print(f"[regime_gating] building {gate_col}")
        frames.append(build_gated_portfolio(run_dir, gates, gate_col, out_dir))
    portfolios = pd.concat(frames, ignore_index=True)
    portfolios.to_csv(out_dir / "regime_gate_equity.csv", index=False)

    metrics = cost_metrics(portfolios, cost_grid, out_dir)
    annual_metrics(portfolios, 0.0, out_dir)
    annual_metrics(portfolios, 10.0, out_dir)
    write_figures(metrics, out_dir)
    render_report(metrics, out_dir)

    print(f"[mean_reversion_5m_regime_gating] wrote artifacts to {out_dir}")
    print(metrics[metrics["cost_bps"].isin([0.0, 10.0])].to_string(index=False))


if __name__ == "__main__":
    main()
