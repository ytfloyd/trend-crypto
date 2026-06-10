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
STABLE_BASES = {
    "DAI",
    "EURC",
    "GYEN",
    "PAX",
    "PAXG",
    "PYUSD",
    "USDC",
    "USDT",
    "USD",
}


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    breakout_lookback: int
    fast_ma: int
    slow_ma: int


@dataclass(frozen=True)
class StudyConfig:
    timeframe: str = "5m"
    bars_per_year: float = BARS_PER_YEAR_5M
    horizons: tuple[HorizonSpec, ...] = (
        HorizonSpec("fast", 10, 2, 20),
        HorizonSpec("mid", 20, 5, 40),
        HorizonSpec("slow", 50, 10, 200),
    )
    atr_window: int = 20
    atr_k: float = 3.0
    stop_cooldown_bars: int = 5 * BARS_PER_DAY_5M
    vol_window: int = 20
    vol_floor_annual: float = 0.10
    target_vol_annual: float = 0.20
    cash_yield_annual: float = 0.04
    cash_buffer: float = 0.05
    max_gross: float = 1.0
    cost_bps: float = 20.0
    danger_gross: float = 0.25
    danger_btc_vol_threshold: float = 0.80
    danger_btc_dd20_threshold: float = -0.20
    danger_btc_ret5_threshold: float = -0.10
    disable_danger: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-universe 5-minute trend study with entry-fixed ATR stops."
    )
    parser.add_argument(
        "--db",
        default="/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb",
    )
    parser.add_argument("--source_table", default="candles_1m")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--quotes", default="USD,USDC")
    parser.add_argument("--include_stables", action="store_true")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbol override.")
    parser.add_argument("--max_symbols", type=int, default=None, help="Debug cap after sorting symbols.")
    parser.add_argument("--min_5m_bars", type=int, default=250)
    parser.add_argument(
        "--out_dir",
        default="artifacts/research/trend_fixed_atr_5m_universe",
    )
    parser.add_argument("--atr_k", type=float, default=3.0)
    parser.add_argument("--atr_window", type=int, default=20)
    parser.add_argument("--vol_window", type=int, default=20)
    parser.add_argument("--target_vol_annual", type=float, default=0.20)
    parser.add_argument("--cost_bps", type=float, default=20.0)
    parser.add_argument("--disable_danger", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite per-symbol signal files.")
    return parser.parse_args()


def symbol_base(symbol: str) -> str:
    return symbol.rsplit("-", 1)[0]


def resolve_symbols(
    con: duckdb.DuckDBPyConnection,
    table: str,
    quotes: set[str],
    include_stables: bool,
    symbols_override: str | None,
    max_symbols: int | None,
) -> list[str]:
    if symbols_override:
        symbols = sorted({s.strip() for s in symbols_override.split(",") if s.strip()})
    else:
        rows = con.execute(f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol").fetchall()
        symbols = []
        for (symbol,) in rows:
            if "-" not in symbol:
                continue
            base, quote = symbol.rsplit("-", 1)
            if quote not in quotes:
                continue
            if not include_stables and base in STABLE_BASES:
                continue
            symbols.append(symbol)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    return symbols


def load_5m_bars(
    con: duckdb.DuckDBPyConnection,
    table: str,
    symbol: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    clauses = ["symbol = ?"]
    params: list[object] = [symbol]
    if start:
        clauses.append("ts >= CAST(? AS TIMESTAMPTZ)")
        params.append(start)
    if end:
        clauses.append("ts <= CAST(? AS TIMESTAMPTZ)")
        params.append(end)
    where = " AND ".join(clauses)
    df = con.execute(
        f"""
        SELECT
            time_bucket(INTERVAL '5 minutes', ts) AS ts,
            FIRST(open ORDER BY ts) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close ORDER BY ts) AS close,
            SUM(volume) AS volume
        FROM {table}
        WHERE {where}
        GROUP BY 1
        HAVING FIRST(open ORDER BY ts) > 0
           AND LAST(close ORDER BY ts) > 0
           AND MAX(high) > 0
           AND MIN(low) > 0
        ORDER BY 1
        """,
        params,
    ).fetch_df()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def compute_symbol_frame(bars: pd.DataFrame, symbol: str, cfg: StudyConfig) -> pd.DataFrame:
    df = bars.copy()
    df["symbol"] = symbol
    close = df["close"]
    high = df["high"]
    low = df["low"]

    signal_sum = pd.Series(0.0, index=df.index)
    for horizon in cfg.horizons:
        breakout_max = close.shift(1).rolling(
            horizon.breakout_lookback, min_periods=horizon.breakout_lookback
        ).max()
        breakout = (close > breakout_max).astype(float)
        fast_ma = close.shift(1).rolling(horizon.fast_ma, min_periods=horizon.fast_ma).mean()
        slow_ma = close.shift(1).rolling(horizon.slow_ma, min_periods=horizon.slow_ma).mean()
        signal_sum += breakout * (fast_ma > slow_ma).astype(float)

    ret_cc = close.pct_change()
    vol_ann = ret_cc.shift(1).rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
    vol_ann = vol_ann * math.sqrt(cfg.bars_per_year)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(cfg.atr_window, min_periods=cfg.atr_window).mean().shift(1)

    df["score"] = signal_sum / float(len(cfg.horizons))
    df["ret_oc"] = df["close"] / df["open"] - 1.0
    df["vol_ann"] = vol_ann
    df["atr"] = atr

    stop_block, stop_level, atr_entry, stop_events = apply_entry_fixed_atr_stops(df, cfg)
    df["stop_block"] = stop_block
    df["stop_level"] = stop_level
    df["atr_entry"] = atr_entry
    vol_for_weight = df["vol_ann"].where(np.isfinite(df["vol_ann"]), cfg.vol_floor_annual)
    vol_for_weight = vol_for_weight.clip(lower=cfg.vol_floor_annual)
    df["w_raw"] = (df["score"].clip(lower=0.0) / vol_for_weight).where(~df["stop_block"], 0.0)
    df["w_raw"] = df["w_raw"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    events_df = pd.DataFrame.from_records(stop_events)
    return df[
        [
            "ts",
            "symbol",
            "open",
            "close",
            "volume",
            "ret_oc",
            "score",
            "vol_ann",
            "atr",
            "atr_entry",
            "stop_level",
            "stop_block",
            "w_raw",
        ]
    ], events_df


def apply_entry_fixed_atr_stops(
    df: pd.DataFrame, cfg: StudyConfig
) -> tuple[list[bool], list[float], list[float], list[dict[str, object]]]:
    stop_block: list[bool] = []
    stop_level_series: list[float] = []
    atr_entry_series: list[float] = []
    stop_events: list[dict[str, object]] = []

    in_pos = False
    atr_entry = np.nan
    max_close = np.nan
    stop_level = np.nan
    cooldown = 0

    for row in df.itertuples(index=False):
        close = float(row.close)
        atr = float(row.atr) if np.isfinite(row.atr) else np.nan
        allow_signal = bool(row.score > 0)
        stop_hit = False

        if in_pos and np.isfinite(stop_level):
            max_close = max(max_close, close) if np.isfinite(max_close) else close
            stop_level = max_close - cfg.atr_k * atr_entry
            if close <= stop_level:
                stop_hit = True
                cooldown = cfg.stop_cooldown_bars
                stop_events.append(
                    {
                        "ts": row.ts,
                        "symbol": row.symbol,
                        "close": close,
                        "stop_level": stop_level,
                        "atr_entry": atr_entry,
                    }
                )

        allowed = allow_signal and cooldown <= 0 and not stop_hit
        if allowed and not in_pos:
            if np.isfinite(atr):
                in_pos = True
                atr_entry = atr
                max_close = close
                stop_level = close - cfg.atr_k * atr_entry
            else:
                allowed = False

        if not allowed and (not allow_signal or stop_hit):
            in_pos = False
            atr_entry = np.nan
            max_close = np.nan
            stop_level = np.nan

        stop_block.append(bool(cooldown > 0 or stop_hit))
        stop_level_series.append(float(stop_level) if np.isfinite(stop_level) else np.nan)
        atr_entry_series.append(float(atr_entry) if np.isfinite(atr_entry) else np.nan)

        if cooldown > 0:
            cooldown -= 1

    return stop_block, stop_level_series, atr_entry_series, stop_events


def write_parquet(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, path: Path) -> None:
    con.register("tmp_write_df", df)
    con.execute(f"COPY tmp_write_df TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("tmp_write_df")


def build_portfolio_outputs(out_dir: Path, cfg: StudyConfig) -> pd.DataFrame:
    signals_glob = (out_dir / "signals_by_symbol" / "*.parquet").as_posix()
    db_path = out_dir / "portfolio.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSTALL parquet")
        con.execute("LOAD parquet")
    except Exception:
        pass

    gross_target = min(cfg.max_gross, 1.0 - cfg.cash_buffer)
    rf_bar = (1.0 + cfg.cash_yield_annual) ** (1.0 / cfg.bars_per_year) - 1.0
    cost_rate = cfg.cost_bps / 10_000.0
    danger_sql = "FALSE" if cfg.disable_danger else "COALESCE(d.danger, FALSE)"

    con.execute("DROP TABLE IF EXISTS weights")
    con.execute(
        f"""
        CREATE TABLE weights AS
        WITH sig AS (
            SELECT * FROM read_parquet('{signals_glob}', union_by_name = TRUE)
        ),
        base AS (
            SELECT
                *,
                SUM(w_raw) OVER (PARTITION BY ts) AS gross_raw
            FROM sig
        ),
        w0 AS (
            SELECT
                *,
                CASE
                    WHEN gross_raw > 0 THEN w_raw / gross_raw * {gross_target}
                    ELSE 0.0
                END AS w_base
            FROM base
        ),
        risk AS (
            SELECT
                ts,
                SQRT(SUM(POWER(w_base * COALESCE(vol_ann, {cfg.vol_floor_annual}), 2))) AS port_vol
            FROM w0
            GROUP BY ts
        ),
        scaled AS (
            SELECT
                w0.*,
                CASE
                    WHEN r.port_vol > 0 THEN w0.w_base * LEAST(1.0, {cfg.target_vol_annual} / r.port_vol)
                    ELSE 0.0
                END AS w_scaled
            FROM w0
            JOIN risk r USING (ts)
        ),
        danger AS (
            SELECT
                ts,
                (
                    COALESCE(vol_ann, 0.0) > {cfg.danger_btc_vol_threshold}
                    OR close / MAX(close) OVER (
                        ORDER BY ts ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                    ) - 1.0 < {cfg.danger_btc_dd20_threshold}
                    OR close / LAG(close, 5) OVER (ORDER BY ts) - 1.0 < {cfg.danger_btc_ret5_threshold}
                ) AS danger
            FROM sig
            WHERE symbol = 'BTC-USD'
        ),
        gross_scaled AS (
            SELECT
                scaled.*,
                SUM(w_scaled) OVER (PARTITION BY ts) AS gross_scaled,
                {danger_sql} AS danger
            FROM scaled
            LEFT JOIN danger d USING (ts)
        )
        SELECT
            ts,
            symbol,
            ret_oc,
            score,
            stop_block,
            CASE
                WHEN danger AND gross_scaled > 0 THEN w_scaled * ({cfg.danger_gross} / gross_scaled)
                ELSE w_scaled
            END AS w_signal,
            danger
        FROM gross_scaled
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
                ret_oc,
                w_signal,
                LAG(w_signal, 1, 0.0) OVER (PARTITION BY symbol ORDER BY ts) AS w_held,
                LAG(w_signal, 2, 0.0) OVER (PARTITION BY symbol ORDER BY ts) AS w_prev_held,
                danger
            FROM weights
        ),
        by_ts AS (
            SELECT
                ts,
                SUM(w_held * COALESCE(ret_oc, 0.0)) AS gross_asset_ret,
                SUM(ABS(w_held)) AS gross_exposure,
                SUM(ABS(w_held - w_prev_held)) AS turnover_one_sided,
                BOOL_OR(danger) AS danger
            FROM held
            GROUP BY ts
        ),
        ret AS (
            SELECT
                ts,
                gross_asset_ret,
                gross_exposure,
                GREATEST(0.0, 1.0 - gross_exposure) AS cash_weight,
                turnover_one_sided,
                turnover_one_sided * {cost_rate} AS cost_ret,
                danger,
                gross_asset_ret
                    + GREATEST(0.0, 1.0 - gross_exposure) * {rf_bar}
                    - turnover_one_sided * {cost_rate} AS portfolio_ret
            FROM by_ts
        )
        SELECT
            *,
            EXP(SUM(LN(GREATEST(1e-12, 1.0 + portfolio_ret))) OVER (ORDER BY ts)) AS portfolio_equity
        FROM ret
        ORDER BY ts
        """
    )
    con.execute(f"COPY portfolio TO '{(out_dir / 'equity.csv').as_posix()}' (HEADER, DELIMITER ',')")
    con.execute(
        f"""
        COPY (
            SELECT symbol, SUM(ABS(w_signal)) AS weight_abs_sum, AVG(w_signal) AS avg_weight
            FROM weights
            GROUP BY symbol
            ORDER BY weight_abs_sum DESC
            LIMIT 50
        ) TO '{(out_dir / 'top_weighted_symbols.csv').as_posix()}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT symbol, COUNT(*) AS stop_hits
            FROM read_parquet('{(out_dir / 'stop_events_by_symbol' / '*.parquet').as_posix()}', union_by_name = TRUE)
            GROUP BY symbol
            ORDER BY stop_hits DESC
            LIMIT 50
        ) TO '{(out_dir / 'top_stop_hit_symbols.csv').as_posix()}' (HEADER, DELIMITER ',')
        """
    )
    equity = con.execute("SELECT * FROM portfolio ORDER BY ts").fetch_df()
    con.close()
    return equity


def compute_metrics(equity: pd.DataFrame, cfg: StudyConfig) -> dict[str, float | int | str]:
    if equity.empty:
        return {}
    ret = equity["portfolio_ret"].astype(float)
    eq = equity["portfolio_equity"].astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    elapsed_years = (
        pd.to_datetime(equity["ts"]).iloc[-1] - pd.to_datetime(equity["ts"]).iloc[0]
    ).total_seconds() / (365.0 * 24 * 3600)
    elapsed_years = max(elapsed_years, len(equity) / cfg.bars_per_year)
    vol = float(ret.std(ddof=0) * math.sqrt(cfg.bars_per_year))
    sharpe = float(ret.mean() / ret.std(ddof=0) * math.sqrt(cfg.bars_per_year)) if ret.std(ddof=0) > 0 else 0.0
    final_equity = float(eq.iloc[-1])
    cagr = final_equity ** (1.0 / elapsed_years) - 1.0 if elapsed_years > 0 and final_equity > 0 else np.nan
    return {
        "start": str(pd.to_datetime(equity["ts"]).iloc[0]),
        "end": str(pd.to_datetime(equity["ts"]).iloc[-1]),
        "n_bars": int(len(equity)),
        "elapsed_years": float(elapsed_years),
        "final_equity": final_equity,
        "total_return": final_equity - 1.0,
        "cagr": float(cagr),
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(drawdown.min()),
        "avg_gross": float(equity["gross_exposure"].mean()),
        "avg_turnover_one_sided": float(equity["turnover_one_sided"].mean()),
        "danger_pct": float(equity["danger"].mean()) if "danger" in equity else 0.0,
    }


def write_figures(equity: pd.DataFrame, out_dir: Path) -> None:
    if equity.empty:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(equity["ts"])
    eq = equity["portfolio_equity"].astype(float)
    dd = eq / eq.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(ts, eq, linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("5m Full-Universe Trend with Entry-Fixed ATR Stops")
    axes[0].set_ylabel("Portfolio equity (log)")
    axes[1].fill_between(ts, dd, 0.0, alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_equity_drawdown.png", dpi=160)
    plt.close(fig)


def render_report(
    out_dir: Path,
    cfg: StudyConfig,
    metrics: dict[str, object],
    n_symbols: int,
    skipped: list[dict[str, object]],
) -> None:
    def pct(value: object) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):+.1%}"

    lines = [
        "# 5-Minute Full-Universe Trend with Entry-Fixed ATR Stops",
        "",
        "**Status:** internal research note",
        f"**Artifacts:** `{out_dir.as_posix()}/`",
        "",
        "## Method",
        "",
        "- Source: Coinbase `candles_1m`, resampled to 5-minute OHLCV bars.",
        f"- Universe: `{n_symbols}` USD/USDC spot pairs after quote/stable filters.",
        "- Signal: replicated Transtrend v1 multi-horizon breakout plus MA filter.",
        f"- Stops: entry-fixed ATR({cfg.atr_window}) trailing stop at `{cfg.atr_k}x` ATR.",
        f"- Cooldown: `{cfg.stop_cooldown_bars}` 5-minute bars after a stop hit.",
        f"- Costs: `{cfg.cost_bps}` bps on one-sided turnover.",
        "",
        "## Headline Results",
        "",
        f"- Active window: `{metrics.get('start')}` to `{metrics.get('end')}`",
        f"- Final equity: `{float(metrics.get('final_equity', float('nan'))):.4f}`",
        f"- CAGR: `{pct(metrics.get('cagr'))}`",
        f"- Volatility: `{pct(metrics.get('vol'))}`",
        f"- Sharpe: `{float(metrics.get('sharpe', float('nan'))):.2f}`",
        f"- Max drawdown: `{pct(metrics.get('max_dd'))}`",
        f"- Average gross exposure: `{pct(metrics.get('avg_gross'))}`",
        f"- Average one-sided turnover per bar: `{pct(metrics.get('avg_turnover_one_sided'))}`",
        f"- Danger-gated bars: `{pct(metrics.get('danger_pct'))}`",
        "",
        "![Equity and drawdown](../../artifacts/research/trend_fixed_atr_5m_universe/figures/01_equity_drawdown.png)",
        "",
        "## Artifacts",
        "",
        "- Metrics: `metrics.json`",
        "- Portfolio equity: `equity.csv`",
        "- Per-symbol signal files: `signals_by_symbol/*.parquet`",
        "- Per-symbol stop events: `stop_events_by_symbol/*.parquet`",
        "- Top weighted symbols: `top_weighted_symbols.csv`",
        "- Top stop-hit symbols: `top_stop_hit_symbols.csv`",
        "- Config and universe audit: `config.json`, `universe.csv`, `skipped_symbols.csv`",
    ]
    if skipped:
        lines.extend(["", "## Coverage Notes", "", f"- Skipped `{len(skipped)}` symbols with insufficient 5-minute history."])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    signals_dir = out_dir / "signals_by_symbol"
    events_dir = out_dir / "stop_events_by_symbol"
    signals_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    cfg = StudyConfig(
        atr_window=args.atr_window,
        atr_k=args.atr_k,
        vol_window=args.vol_window,
        target_vol_annual=args.target_vol_annual,
        cost_bps=args.cost_bps,
        disable_danger=args.disable_danger,
    )

    con = duckdb.connect(args.db, read_only=True)
    write_con = duckdb.connect()
    quotes = {q.strip() for q in args.quotes.split(",") if q.strip()}
    symbols = resolve_symbols(
        con,
        args.source_table,
        quotes,
        args.include_stables,
        args.symbols,
        args.max_symbols,
    )

    skipped: list[dict[str, object]] = []
    processed: list[dict[str, object]] = []
    for idx, symbol in enumerate(symbols, start=1):
        safe_symbol = symbol.replace("/", "_").replace("-", "_")
        signal_path = signals_dir / f"{safe_symbol}.parquet"
        event_path = events_dir / f"{safe_symbol}.parquet"
        if signal_path.exists() and event_path.exists() and not args.force:
            print(f"[{idx}/{len(symbols)}] skip existing {symbol}")
            processed.append({"symbol": symbol, "signal_path": str(signal_path), "event_path": str(event_path)})
            continue

        bars = load_5m_bars(con, args.source_table, symbol, args.start, args.end)
        if len(bars) < args.min_5m_bars:
            skipped.append({"symbol": symbol, "reason": "insufficient_5m_bars", "bars": len(bars)})
            print(f"[{idx}/{len(symbols)}] skip {symbol}: {len(bars)} bars")
            continue

        frame, events = compute_symbol_frame(bars, symbol, cfg)
        write_parquet(write_con, frame, signal_path)
        if events.empty:
            events = pd.DataFrame(columns=["ts", "symbol", "close", "stop_level", "atr_entry"])
        write_parquet(write_con, events, event_path)
        processed.append(
            {
                "symbol": symbol,
                "bars": int(len(frame)),
                "start": str(frame["ts"].iloc[0]),
                "end": str(frame["ts"].iloc[-1]),
                "stop_hits": int(len(events)),
                "signal_path": str(signal_path),
                "event_path": str(event_path),
            }
        )
        print(f"[{idx}/{len(symbols)}] wrote {symbol}: bars={len(frame):,} stops={len(events):,}")

    con.close()
    write_con.close()

    pd.DataFrame(processed).to_csv(out_dir / "universe.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_symbols.csv", index=False)
    config_blob = asdict(cfg)
    config_blob["horizons"] = [asdict(h) for h in cfg.horizons]
    config_blob.update(
        {
            "db": args.db,
            "source_table": args.source_table,
            "start": args.start,
            "end": args.end,
            "quotes": sorted(quotes),
            "include_stables": args.include_stables,
            "min_5m_bars": args.min_5m_bars,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config_blob, indent=2, default=str))

    if not processed:
        raise RuntimeError("No symbols were processed; cannot build portfolio outputs.")

    equity = build_portfolio_outputs(out_dir, cfg)
    metrics = compute_metrics(equity, cfg)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_figures(equity, out_dir)
    render_report(out_dir, cfg, metrics, len(processed), skipped)
    print(f"[trend_fixed_atr_5m_universe] wrote artifacts to {out_dir}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
