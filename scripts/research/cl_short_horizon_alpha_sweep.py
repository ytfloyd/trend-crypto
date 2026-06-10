#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMEFRAMES: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
}


@dataclass(frozen=True)
class AlphaSpec:
    alpha_id: str
    family: str
    timeframe: str
    params: dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep short-horizon alpha rules on CL institutional continuous.")
    parser.add_argument("--continuous_dir", default="artifacts/research/cl_institutional_continuous")
    parser.add_argument("--out_dir", default="artifacts/research/cl_short_horizon_alpha_sweep")
    parser.add_argument("--cost_grid_bps", default="0,1,2,5,10")
    return parser.parse_args()


def load_valid_1m(continuous_dir: Path) -> pd.DataFrame:
    bars = pd.read_parquet(continuous_dir / "bars_1m.parquet")
    bars["ts"] = pd.to_datetime(bars["ts"])
    bars = bars[bars["front_month_valid"]].sort_values("ts").copy()
    if bars.empty:
        raise RuntimeError("No front-month-valid institutional CL rows found.")
    return bars


def resample_bars(bars_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    bars = bars_1m.set_index("ts").resample(TIMEFRAMES[timeframe]).agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum", "active_expiry": "last"}
    )
    bars = bars.dropna(subset=["o", "c"]).reset_index()
    bars["timeframe"] = timeframe
    return bars


def build_specs() -> list[AlphaSpec]:
    specs: list[AlphaSpec] = []
    for tf in TIMEFRAMES:
        # Return continuation/reversal over short horizons.
        for lookback in (3, 6, 12, 24, 48):
            for mode in ("trend", "reversal"):
                specs.append(
                    AlphaSpec(
                        f"{tf}_ret_{mode}_{lookback}",
                        f"return_{mode}",
                        tf,
                        {"lookback": lookback},
                    )
                )

        # Shifted z-score event rules.
        for lookback in (24, 72, 144):
            for threshold in (1.0, 1.5, 2.0):
                for mode in ("trend", "reversal"):
                    specs.append(
                        AlphaSpec(
                            f"{tf}_z_{mode}_lb{lookback}_th{threshold:g}",
                            f"zscore_{mode}",
                            tf,
                            {"lookback": lookback, "threshold": threshold},
                        )
                    )

        # Moving average crossovers.
        for fast, slow in ((3, 12), (5, 20), (5, 40), (10, 40), (20, 80)):
            specs.append(
                AlphaSpec(
                    f"{tf}_ma_{fast}_{slow}",
                    "ma_crossover",
                    tf,
                    {"fast": fast, "slow": slow},
                )
            )

        # Donchian channel trend and fade.
        for lookback in (12, 24, 48, 96):
            for mode in ("breakout", "fade"):
                specs.append(
                    AlphaSpec(
                        f"{tf}_donchian_{mode}_{lookback}",
                        f"donchian_{mode}",
                        tf,
                        {"lookback": lookback},
                    )
                )

        # RSI-like mean reversion and trend.
        for lookback in (7, 14, 28):
            specs.append(AlphaSpec(f"{tf}_rsi_reversal_{lookback}", "rsi_reversal", tf, {"lookback": lookback}))
            specs.append(AlphaSpec(f"{tf}_rsi_trend_{lookback}", "rsi_trend", tf, {"lookback": lookback}))
    return specs


def _rsi(close: pd.Series, lookback: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(lookback, min_periods=lookback).mean()
    loss = (-delta.clip(upper=0)).rolling(lookback, min_periods=lookback).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _event_zscore_position(close: pd.Series, lookback: int, threshold: float, mode: str) -> pd.Series:
    mean = close.shift(1).rolling(lookback, min_periods=lookback).mean()
    std = close.shift(1).rolling(lookback, min_periods=lookback).std()
    z = (close - mean) / std.replace(0.0, np.nan)
    position = 0.0
    out: list[float] = []
    for value in z:
        if pd.notna(value):
            if position == 0.0:
                if mode == "trend":
                    if value >= threshold:
                        position = 1.0
                    elif value <= -threshold:
                        position = -1.0
                else:
                    if value <= -threshold:
                        position = 1.0
                    elif value >= threshold:
                        position = -1.0
            elif position > 0.0 and value <= 0.0:
                position = 0.0
            elif position < 0.0 and value >= 0.0:
                position = 0.0
        out.append(position)
    return pd.Series(out, index=close.index)


def signal_for_spec(bars: pd.DataFrame, spec: AlphaSpec) -> pd.Series:
    close = bars["c"].astype(float)
    high = bars["h"].astype(float)
    low = bars["l"].astype(float)
    params = spec.params

    if spec.family == "return_trend":
        lookback = int(params["lookback"])
        past_ret = close / close.shift(lookback) - 1.0
        return np.sign(past_ret).fillna(0.0)

    if spec.family == "return_reversal":
        lookback = int(params["lookback"])
        past_ret = close / close.shift(lookback) - 1.0
        return (-np.sign(past_ret)).fillna(0.0)

    if spec.family == "zscore_trend":
        return _event_zscore_position(close, int(params["lookback"]), float(params["threshold"]), "trend")

    if spec.family == "zscore_reversal":
        return _event_zscore_position(close, int(params["lookback"]), float(params["threshold"]), "reversal")

    if spec.family == "ma_crossover":
        fast = close.shift(1).rolling(int(params["fast"]), min_periods=int(params["fast"])).mean()
        slow = close.shift(1).rolling(int(params["slow"]), min_periods=int(params["slow"])).mean()
        signal = pd.Series(0.0, index=close.index)
        signal = signal.mask(fast > slow, 1.0)
        signal = signal.mask(fast < slow, -1.0)
        return signal.fillna(0.0)

    if spec.family in {"donchian_breakout", "donchian_fade"}:
        lookback = int(params["lookback"])
        upper = high.shift(1).rolling(lookback, min_periods=lookback).max()
        lower = low.shift(1).rolling(lookback, min_periods=lookback).min()
        mid = (upper + lower) / 2.0
        position = 0.0
        out: list[float] = []
        for c, u, lo, m in zip(close, upper, lower, mid):
            if pd.notna(u) and pd.notna(lo):
                if position == 0.0:
                    if spec.family == "donchian_breakout":
                        if c > u:
                            position = 1.0
                        elif c < lo:
                            position = -1.0
                    else:
                        if c < lo:
                            position = 1.0
                        elif c > u:
                            position = -1.0
                elif position > 0.0 and c <= m:
                    position = 0.0
                elif position < 0.0 and c >= m:
                    position = 0.0
            out.append(position)
        return pd.Series(out, index=close.index)

    if spec.family in {"rsi_reversal", "rsi_trend"}:
        lookback = int(params["lookback"])
        rsi = _rsi(close, lookback).shift(1)
        signal = pd.Series(0.0, index=close.index)
        if spec.family == "rsi_reversal":
            signal = signal.mask(rsi < 30.0, 1.0)
            signal = signal.mask(rsi > 70.0, -1.0)
        else:
            signal = signal.mask(rsi > 70.0, 1.0)
            signal = signal.mask(rsi < 30.0, -1.0)
        return signal.fillna(0.0)

    raise ValueError(f"unsupported family: {spec.family}")


def backtest(bars: pd.DataFrame, spec: AlphaSpec, cost_bps: float = 0.0) -> pd.DataFrame:
    df = bars.copy()
    df["alpha_id"] = spec.alpha_id
    df["family"] = spec.family
    df["w_signal"] = signal_for_spec(df, spec).astype(float).clip(-1.0, 1.0)
    df["w_held"] = df["w_signal"].shift(1).fillna(0.0)
    df["ret_open_to_next_open"] = df["o"].shift(-1) / df["o"] - 1.0
    df["ret_open_to_next_open"] = df["ret_open_to_next_open"].fillna(0.0)
    df["turnover_one_sided"] = (df["w_held"] - df["w_held"].shift(1).fillna(0.0)).abs()
    df["cost_ret"] = df["turnover_one_sided"] * (cost_bps / 10_000.0)
    df["portfolio_ret"] = df["w_held"] * df["ret_open_to_next_open"] - df["cost_ret"]
    df["portfolio_equity"] = (1.0 + df["portfolio_ret"]).cumprod()
    df["drawdown"] = df["portfolio_equity"] / df["portfolio_equity"].cummax() - 1.0
    return df


def metrics(frame: pd.DataFrame, spec: AlphaSpec, cost_bps: float) -> dict[str, object]:
    ret = frame["portfolio_ret"].astype(float)
    eq = frame["portfolio_equity"].astype(float)
    elapsed_years = (frame["ts"].iloc[-1] - frame["ts"].iloc[0]).total_seconds() / (365.0 * 24 * 3600)
    bars_per_year = len(frame) / elapsed_years if elapsed_years > 0 else 0.0
    ret_std = ret.std(ddof=0)
    signal_changes = int((frame["w_signal"] != frame["w_signal"].shift(1)).sum())
    return {
        "alpha_id": spec.alpha_id,
        "family": spec.family,
        "timeframe": spec.timeframe,
        "params": json.dumps(spec.params, sort_keys=True),
        "cost_bps": cost_bps,
        "n_bars": int(len(frame)),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": float(eq.iloc[-1] ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 and eq.iloc[-1] > 0 else None,
        "vol": float(ret_std * math.sqrt(bars_per_year)) if bars_per_year > 0 else 0.0,
        "sharpe": float(ret.mean() / ret_std * math.sqrt(bars_per_year)) if ret_std > 0 and bars_per_year > 0 else 0.0,
        "max_dd": float(frame["drawdown"].min()),
        "avg_abs_exposure": float(frame["w_held"].abs().mean()),
        "avg_turnover_one_sided": float(frame["turnover_one_sided"].mean()),
        "signal_changes": signal_changes,
        "long_pct": float((frame["w_held"] > 0).mean()),
        "short_pct": float((frame["w_held"] < 0).mean()),
    }


def write_figures(results: pd.DataFrame, top_equity: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    top = results[results["cost_bps"] == 0.0].sort_values("sharpe", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["alpha_id"][::-1], top["sharpe"][::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 20 CL short-horizon alphas by zero-cost Sharpe")
    ax.set_xlabel("Annualized Sharpe")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_top20_sharpe.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    for alpha_id, group in top_equity.groupby("alpha_id", sort=False):
        ax.plot(group["ts"], group["portfolio_equity"], linewidth=1.0, label=alpha_id)
    ax.set_yscale("log")
    ax.set_title("Top CL short-horizon alpha equity curves")
    ax.set_ylabel("Equity (log)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_top_equity_curves.png", dpi=160)
    plt.close(fig)


def render_report(results: pd.DataFrame, out_dir: Path) -> None:
    top0 = results[results["cost_bps"] == 0.0].sort_values("sharpe", ascending=False).head(5)
    top2 = results[results["cost_bps"] == 2.0].sort_values("sharpe", ascending=False).head(5)
    best = top0.iloc[0]
    lines = [
        "# CL Short-Horizon Alpha Sweep",
        "",
        "**Source:** institutional CL continuous valid front-month series",
        "",
        "## Headline",
        "",
        f"- Best zero-cost alpha: `{best['alpha_id']}` with Sharpe `{best['sharpe']:.2f}`, CAGR `{best['cagr']:+.1%}`, max DD `{best['max_dd']:+.1%}`.",
        "- Cost sensitivity is included for 0, 1, 2, 5, and 10 bps one-sided costs.",
        "",
        "## Top Zero-Cost Alphas",
        "",
    ]
    for _, row in top0.iterrows():
        lines.append(
            f"- `{row['alpha_id']}`: Sharpe `{row['sharpe']:.2f}`, CAGR `{row['cagr']:+.1%}`, max DD `{row['max_dd']:+.1%}`."
        )
    lines += ["", "## Top 2 bps Alphas", ""]
    for _, row in top2.iterrows():
        lines.append(
            f"- `{row['alpha_id']}`: Sharpe `{row['sharpe']:.2f}`, CAGR `{row['cagr']:+.1%}`, max DD `{row['max_dd']:+.1%}`."
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `results.csv`",
        "- `top_equity.csv`",
        "- `figures/01_top20_sharpe.png`",
        "- `figures/02_top_equity_curves.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    continuous_dir = Path(args.continuous_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_grid = [float(x) for x in args.cost_grid_bps.split(",") if x.strip()]

    bars_1m = load_valid_1m(continuous_dir)
    bars_by_tf = {tf: resample_bars(bars_1m, tf) for tf in TIMEFRAMES}
    specs = build_specs()

    records = []
    zero_cost_frames: dict[str, pd.DataFrame] = {}
    for i, spec in enumerate(specs, start=1):
        bars = bars_by_tf[spec.timeframe]
        for cost_bps in cost_grid:
            frame = backtest(bars, spec, cost_bps=cost_bps)
            records.append(metrics(frame, spec, cost_bps))
            if cost_bps == 0.0:
                zero_cost_frames[spec.alpha_id] = frame[
                    ["ts", "alpha_id", "family", "timeframe", "portfolio_ret", "portfolio_equity", "drawdown", "w_held"]
                ].copy()
        if i % 50 == 0:
            print(f"[cl_short_horizon_alpha_sweep] evaluated {i}/{len(specs)} specs")

    results = pd.DataFrame(records).sort_values(["cost_bps", "sharpe"], ascending=[True, False])
    results.to_csv(out_dir / "results.csv", index=False)

    top_ids = results[results["cost_bps"] == 0.0].sort_values("sharpe", ascending=False).head(10)["alpha_id"].tolist()
    top_equity = pd.concat([zero_cost_frames[x] for x in top_ids], ignore_index=True)
    top_equity.to_csv(out_dir / "top_equity.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "continuous_dir": str(continuous_dir),
                "timeframes": list(TIMEFRAMES),
                "n_specs": len(specs),
                "cost_grid_bps": cost_grid,
                "return_model": "signal_at_close_t_fill_next_open_hold_open_to_next_open",
            },
            indent=2,
        )
    )
    write_figures(results, top_equity, out_dir)
    render_report(results, out_dir)
    print(results[results["cost_bps"] == 0.0].head(20).to_string(index=False))
    print(f"[cl_short_horizon_alpha_sweep] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
