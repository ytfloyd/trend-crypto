#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BARS_PER_DAY_5M = 24 * 12
BARS_PER_YEAR_5M = 365.0 * BARS_PER_DAY_5M


@dataclass(frozen=True)
class Variant:
    label: str
    lookback: int
    entry_z: float
    exit_z: float
    stop_z: float | None = None
    max_hold_bars: int | None = None


@dataclass(frozen=True)
class StudyConfig:
    bars_per_year: float = BARS_PER_YEAR_5M
    cash_yield_annual: float = 0.04
    max_gross: float = 1.0
    max_name_weight: float = 0.05
    min_active_positions: int = 10
    cost_bps: float = 0.0
    min_rolling_24h_dollar_volume: float = 10_000_000.0
    max_abs_5m_return: float = 0.25
    variants: tuple[Variant, ...] = (
        Variant("mr_12h_z2", 12 * 12, -2.0, 0.0),
        Variant("mr_24h_z2", 24 * 12, -2.0, 0.0),
        Variant("mr_48h_z2", 48 * 12, -2.0, 0.0),
        Variant("mr_24h_z25", 24 * 12, -2.5, 0.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-minute full-universe mean-reversion study.")
    parser.add_argument(
        "--signals_dir",
        default="artifacts/research/trend_fixed_atr_5m_universe/signals_by_symbol",
        help="Directory of per-symbol 5m parquet files containing ts, symbol, open, close, ret_oc.",
    )
    parser.add_argument("--out_dir", default="artifacts/research/mean_reversion_5m_universe")
    parser.add_argument("--cost_bps", type=float, default=0.0)
    parser.add_argument("--cash_yield_annual", type=float, default=0.04)
    parser.add_argument("--max_name_weight", type=float, default=0.05)
    parser.add_argument("--min_active_positions", type=int, default=10)
    parser.add_argument("--min_rolling_24h_dollar_volume", type=float, default=10_000_000.0)
    parser.add_argument("--max_abs_5m_return", type=float, default=0.25)
    parser.add_argument("--canonical_by_base", action="store_true", default=True)
    parser.add_argument("--no_canonical_by_base", action="store_false", dest="canonical_by_base")
    parser.add_argument("--quote_preference", default="USD,USDC")
    parser.add_argument("--max_symbols", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_symbol_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["ts", "symbol", "open", "close", "volume", "ret_oc"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)


def _base_and_quote(symbol: str) -> tuple[str, str]:
    if "-" not in symbol:
        return symbol, ""
    return tuple(symbol.rsplit("-", 1))  # type: ignore[return-value]


def select_canonical_files(
    signals_dir: Path,
    *,
    canonical_by_base: bool,
    quote_preference: list[str],
    max_symbols: int | None,
) -> list[Path]:
    files = sorted(signals_dir.glob("*.parquet"))
    if not canonical_by_base:
        return files[:max_symbols] if max_symbols is not None else files

    candidates: dict[str, tuple[int, str, Path]] = {}
    quote_rank = {quote: idx for idx, quote in enumerate(quote_preference)}
    for path in files:
        # File stems are symbol names with '-' converted to '_'. Read only metadata when needed
        # because some bases contain underscores or digits.
        symbol = str(pd.read_parquet(path, columns=["symbol"]).iloc[0, 0])
        base, quote = _base_and_quote(symbol)
        rank = quote_rank.get(quote, len(quote_rank))
        current = candidates.get(base)
        if current is None or (rank, symbol) < (current[0], current[1]):
            candidates[base] = (rank, symbol, path)

    selected = [item[2] for item in sorted(candidates.values(), key=lambda x: x[1])]
    return selected[:max_symbols] if max_symbols is not None else selected


def add_cleaning_columns(df: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    out = df.copy()
    expected_step = pd.Timedelta(minutes=5)
    continuous_bar = out["ts"].diff().eq(expected_step).fillna(False)
    dollar_volume = out["close"].astype(float) * out["volume"].astype(float)
    rolling_24h_dollar_volume = dollar_volume.rolling(
        BARS_PER_DAY_5M, min_periods=BARS_PER_DAY_5M // 2
    ).sum().shift(1)
    raw_ret = out["ret_oc"].astype(float)
    valid_bar = raw_ret.abs() <= cfg.max_abs_5m_return
    liquid = rolling_24h_dollar_volume >= cfg.min_rolling_24h_dollar_volume
    out["eligible"] = (continuous_bar & valid_bar & liquid).fillna(False)
    out["ret_clean"] = raw_ret.where(continuous_bar & valid_bar, 0.0).fillna(0.0)
    out["rolling_24h_dollar_volume"] = rolling_24h_dollar_volume
    out["invalid_bar"] = (~(continuous_bar & valid_bar)).fillna(True)
    return out


def compute_variant_positions(df: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    close = df["close"].astype(float)
    mean = close.shift(1).rolling(variant.lookback, min_periods=variant.lookback).mean()
    std = close.shift(1).rolling(variant.lookback, min_periods=variant.lookback).std()
    zscore = (close - mean) / std.replace(0.0, np.nan)
    eligible = df["eligible"].fillna(False).to_numpy()

    pos = np.zeros(len(df), dtype=float)
    holding = False
    hold_bars = 0
    entries = 0
    exits = 0
    stop_exits = 0

    for i, z in enumerate(zscore.to_numpy()):
        if not eligible[i]:
            holding = False
            hold_bars = 0
            pos[i] = 0.0
            continue
        if not np.isfinite(z):
            pos[i] = 1.0 if holding else 0.0
            continue
        if not holding and z <= variant.entry_z:
            holding = True
            hold_bars = 0
            entries += 1
        elif holding:
            hold_bars += 1
            exit_now = z >= variant.exit_z
            stop_now = variant.stop_z is not None and z <= variant.stop_z
            timeout_now = variant.max_hold_bars is not None and hold_bars >= variant.max_hold_bars
            if exit_now or stop_now or timeout_now:
                holding = False
                exits += 1
                if stop_now:
                    stop_exits += 1
                hold_bars = 0
        pos[i] = 1.0 if holding else 0.0

    out = df[["ts", "symbol"]].copy()
    out["ret_oc"] = df["ret_clean"]
    out["variant"] = variant.label
    out["zscore"] = zscore
    out["active"] = pos
    out["eligible"] = df["eligible"].astype(bool)
    out["invalid_bar"] = df["invalid_bar"].astype(bool)
    out.attrs["entries"] = entries
    out.attrs["exits"] = exits
    out.attrs["stop_exits"] = stop_exits
    return out


def write_parquet(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, path: Path) -> None:
    con.register("tmp_write_df", df)
    con.execute(f"COPY tmp_write_df TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("tmp_write_df")


def process_symbols(
    signals_dir: Path,
    out_dir: Path,
    cfg: StudyConfig,
    max_symbols: int | None,
    force: bool,
    *,
    canonical_by_base: bool,
    quote_preference: list[str],
) -> pd.DataFrame:
    positions_dir = out_dir / "positions_by_symbol"
    positions_dir.mkdir(parents=True, exist_ok=True)
    files = select_canonical_files(
        signals_dir,
        canonical_by_base=canonical_by_base,
        quote_preference=quote_preference,
        max_symbols=max_symbols,
    )
    if not files:
        raise RuntimeError(f"No parquet files found under {signals_dir}")

    con = duckdb.connect()
    records: list[dict[str, object]] = []
    for idx, path in enumerate(files, start=1):
        out_path = positions_dir / path.name
        if out_path.exists() and not force:
            print(f"[{idx}/{len(files)}] skip existing {path.stem}")
            continue

        raw_df = read_symbol_frame(path)
        df = add_cleaning_columns(raw_df, cfg)
        frames = []
        symbol = str(df["symbol"].iloc[0])
        for variant in cfg.variants:
            variant_df = compute_variant_positions(df, variant)
            frames.append(variant_df)
            records.append(
                {
                    "symbol": symbol,
                    "variant": variant.label,
                    "bars": len(variant_df),
                    "entries": int(variant_df.attrs["entries"]),
                    "exits": int(variant_df.attrs["exits"]),
                    "stop_exits": int(variant_df.attrs["stop_exits"]),
                    "active_bars": int(variant_df["active"].sum()),
                    "eligible_bars": int(variant_df["eligible"].sum()),
                    "invalid_bars": int(variant_df["invalid_bar"].sum()),
                    "quote": _base_and_quote(symbol)[1],
                    "base": _base_and_quote(symbol)[0],
                }
            )
        write_parquet(con, pd.concat(frames, ignore_index=True), out_path)
        print(f"[{idx}/{len(files)}] wrote {symbol}: bars={len(df):,}")
    con.close()
    summary = pd.DataFrame(records)
    summary.to_csv(out_dir / "symbol_variant_summary.csv", index=False)
    return summary


def build_portfolios(out_dir: Path, cfg: StudyConfig) -> pd.DataFrame:
    positions_glob = (out_dir / "positions_by_symbol" / "*.parquet").as_posix()
    db_path = out_dir / "portfolio.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSTALL parquet")
        con.execute("LOAD parquet")
    except Exception:
        pass

    rf_bar = (1.0 + cfg.cash_yield_annual) ** (1.0 / cfg.bars_per_year) - 1.0
    cost_rate = cfg.cost_bps / 10_000.0
    con.execute("DROP TABLE IF EXISTS weights")
    con.execute(
        f"""
        CREATE TABLE weights AS
        WITH pos AS (
            SELECT * FROM read_parquet('{positions_glob}', union_by_name = TRUE)
        ),
        active AS (
            SELECT
                *,
                SUM(active) OVER (PARTITION BY variant, ts) AS n_active
            FROM pos
        )
        SELECT
            ts,
            symbol,
            variant,
            ret_oc,
            zscore,
            CASE
                WHEN n_active >= {cfg.min_active_positions}
                THEN active * LEAST({cfg.max_gross} / n_active, {cfg.max_name_weight})
                ELSE 0.0
            END AS w_signal
        FROM active
        """
    )
    con.execute("DROP TABLE IF EXISTS portfolio")
    con.execute(
        f"""
        CREATE TABLE portfolio AS
        WITH held AS (
            SELECT
                ts,
                symbol,
                variant,
                ret_oc,
                w_signal,
                LAG(w_signal, 1, 0.0) OVER (PARTITION BY variant, symbol ORDER BY ts) AS w_held,
                LAG(w_signal, 2, 0.0) OVER (PARTITION BY variant, symbol ORDER BY ts) AS w_prev_held
            FROM weights
        ),
        by_ts AS (
            SELECT
                variant,
                ts,
                SUM(w_held * COALESCE(ret_oc, 0.0)) AS gross_asset_ret,
                SUM(ABS(w_held)) AS gross_exposure,
                SUM(ABS(w_held - w_prev_held)) AS turnover_one_sided,
                SUM(CASE WHEN w_held > 0 THEN 1 ELSE 0 END) AS n_held
            FROM held
            GROUP BY variant, ts
        ),
        ret AS (
            SELECT
                variant,
                ts,
                gross_asset_ret,
                gross_exposure,
                GREATEST(0.0, 1.0 - gross_exposure) AS cash_weight,
                turnover_one_sided,
                n_held,
                turnover_one_sided * {cost_rate} AS cost_ret,
                gross_asset_ret
                    + GREATEST(0.0, 1.0 - gross_exposure) * {rf_bar}
                    - turnover_one_sided * {cost_rate} AS portfolio_ret
            FROM by_ts
        )
        SELECT
            *,
            EXP(
                SUM(LN(GREATEST(1e-12, 1.0 + portfolio_ret)))
                OVER (PARTITION BY variant ORDER BY ts)
            ) AS portfolio_equity
        FROM ret
        ORDER BY variant, ts
        """
    )
    con.execute(f"COPY portfolio TO '{(out_dir / 'equity.csv').as_posix()}' (HEADER, DELIMITER ',')")
    con.execute(
        f"""
        COPY (
            SELECT variant, symbol, SUM(ABS(w_signal)) AS weight_abs_sum, AVG(w_signal) AS avg_weight
            FROM weights
            GROUP BY variant, symbol
            ORDER BY variant, weight_abs_sum DESC
        ) TO '{(out_dir / 'symbol_weights.csv').as_posix()}' (HEADER, DELIMITER ',')
        """
    )
    equity = con.execute("SELECT * FROM portfolio ORDER BY variant, ts").fetch_df()
    con.close()
    return equity


def compute_metrics(equity: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    rows = []
    for variant, group in equity.groupby("variant", sort=True):
        group = group.sort_values("ts")
        ret = group["portfolio_ret"].astype(float)
        eq = group["portfolio_equity"].astype(float)
        running_max = eq.cummax()
        drawdown = eq / running_max - 1.0
        elapsed_years = (
            pd.to_datetime(group["ts"]).iloc[-1] - pd.to_datetime(group["ts"]).iloc[0]
        ).total_seconds() / (365.0 * 24 * 3600)
        elapsed_years = max(elapsed_years, len(group) / cfg.bars_per_year)
        ret_std = ret.std(ddof=0)
        final_equity = float(eq.iloc[-1])
        rows.append(
            {
                "variant": variant,
                "start": str(pd.to_datetime(group["ts"]).iloc[0]),
                "end": str(pd.to_datetime(group["ts"]).iloc[-1]),
                "n_bars": int(len(group)),
                "elapsed_years": elapsed_years,
                "final_equity": final_equity,
                "total_return": final_equity - 1.0,
                "cagr": final_equity ** (1.0 / elapsed_years) - 1.0 if final_equity > 0 else np.nan,
                "vol": float(ret_std * math.sqrt(cfg.bars_per_year)),
                "sharpe": float(ret.mean() / ret_std * math.sqrt(cfg.bars_per_year)) if ret_std > 0 else 0.0,
                "max_dd": float(drawdown.min()),
                "avg_gross": float(group["gross_exposure"].mean()),
                "avg_n_held": float(group["n_held"].mean()),
                "avg_turnover_one_sided": float(group["turnover_one_sided"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)


def write_figures(equity: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for variant, group in equity.groupby("variant", sort=True):
        group = group.sort_values("ts")
        ts = pd.to_datetime(group["ts"])
        eq = group["portfolio_equity"].astype(float)
        axes[0].plot(ts, eq, linewidth=1.0, label=variant)
        dd = eq / eq.cummax() - 1.0
        axes[1].plot(ts, dd, linewidth=0.8, label=variant)
    axes[0].set_yscale("log")
    axes[0].set_title("5m Mean-Reversion Variants, Zero Transaction Costs")
    axes[0].set_ylabel("Portfolio equity (log)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_equity_drawdown.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = metrics.sort_values("sharpe", ascending=True)
    ax.barh(ordered["variant"], ordered["sharpe"])
    ax.set_title("Sharpe by 5m Mean-Reversion Variant")
    ax.set_xlabel("Annualized Sharpe")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_sharpe_by_variant.png", dpi=160)
    plt.close(fig)


def render_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    cfg: StudyConfig,
    n_symbols: int,
    *,
    canonical_by_base: bool,
) -> None:
    best = metrics.iloc[0]
    lines = [
        "# 5-Minute Full-Universe Mean Reversion",
        "",
        "**Status:** internal research note",
        f"**Artifacts:** `{out_dir.as_posix()}/`",
        "",
        "## Method",
        "",
        f"- Universe: `{n_symbols}` canonical symbols from the existing 5-minute per-symbol files.",
        f"- Canonical by base asset: `{canonical_by_base}`.",
        f"- Point-in-time liquidity filter: trailing 24h dollar volume >= `${cfg.min_rolling_24h_dollar_volume:,.0f}`.",
        "- Continuity filter: a bar is tradable only when it directly follows the previous 5-minute timestamp for that symbol.",
        f"- Data-quality filter: 5-minute open-to-close return capped at `+/-{cfg.max_abs_5m_return:.0%}`; excluded bars earn zero return and cannot hold positions.",
        "- Signal: long-only Bollinger z-score mean reversion.",
        "- Entry: close below rolling mean by the configured z-score threshold.",
        "- Exit: close reverts to the prior-bar rolling mean (`z >= 0`).",
        f"- Transaction costs: `{cfg.cost_bps}` bps.",
        "- Portfolio: equal weight across active positions, one-bar execution lag via held weights.",
        f"- Portfolio constraints: require at least `{cfg.min_active_positions}` active names and cap each name at `{cfg.max_name_weight:.1%}`.",
        "",
        "## Best Variant",
        "",
        f"- Variant: `{best['variant']}`",
        f"- Final equity: `{best['final_equity']:.4f}`",
        f"- CAGR: `{best['cagr']:+.1%}`",
        f"- Volatility: `{best['vol']:+.1%}`",
        f"- Sharpe: `{best['sharpe']:.2f}`",
        f"- Max drawdown: `{best['max_dd']:+.1%}`",
        f"- Average positions held: `{best['avg_n_held']:.1f}`",
        f"- Average one-sided turnover per 5m bar: `{best['avg_turnover_one_sided']:.1%}`",
        "",
        "![Equity and drawdown](../../artifacts/research/mean_reversion_5m_universe/figures/01_equity_drawdown.png)",
        "",
        "## Artifacts",
        "",
        "- Metrics: `metrics.csv`",
        "- Portfolio equity: `equity.csv`",
        "- Per-symbol positions: `positions_by_symbol/*.parquet`",
        "- Symbol/variant entry counts: `symbol_variant_summary.csv`",
        "- Symbol weights: `symbol_weights.csv`",
        "- Config: `config.json`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = StudyConfig(
        cost_bps=args.cost_bps,
        cash_yield_annual=args.cash_yield_annual,
        max_name_weight=args.max_name_weight,
        min_active_positions=args.min_active_positions,
        min_rolling_24h_dollar_volume=args.min_rolling_24h_dollar_volume,
        max_abs_5m_return=args.max_abs_5m_return,
    )
    quote_preference = [q.strip() for q in args.quote_preference.split(",") if q.strip()]

    summary = process_symbols(
        Path(args.signals_dir),
        out_dir,
        cfg,
        args.max_symbols,
        args.force,
        canonical_by_base=args.canonical_by_base,
        quote_preference=quote_preference,
    )
    equity = build_portfolios(out_dir, cfg)
    metrics = compute_metrics(equity, cfg)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    config_blob = asdict(cfg)
    config_blob["canonical_by_base"] = args.canonical_by_base
    config_blob["quote_preference"] = quote_preference
    config_blob["signals_dir"] = args.signals_dir
    (out_dir / "config.json").write_text(json.dumps(config_blob, indent=2, default=str))
    write_figures(equity, metrics, out_dir)
    render_report(out_dir, metrics, cfg, summary["symbol"].nunique(), canonical_by_base=args.canonical_by_base)
    print(f"[mean_reversion_5m_universe] wrote artifacts to {out_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
