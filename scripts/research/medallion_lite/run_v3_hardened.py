#!/usr/bin/env python3
"""
Run Medallion Lite V3 through the hardened PortfolioEngine.

Signal generation (regime, V3 factors, composite) is precomputed from
the research pipeline.  The MedallionLiteV3Strategy adapts those signals
into the PortfolioStrategy interface.  All execution, cost accounting,
drift tracking, and risk management are handled by PortfolioEngine.

Usage:
    python -m scripts.research.medallion_lite.run_v3_hardened
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from backtest.portfolio_engine import PortfolioEngine
from backtest.portfolio_result import PortfolioResult
from common.config import (
    DataConfig,
    EngineConfig,
    ExecutionConfig,
    PortfolioConfig,
    RiskConfigRaw,
    RunConfigRaw,
    StrategyConfigRaw,
    compile_config,
)
from risk.risk_manager import RiskManager
from strategy.medallion_lite_v3 import MedallionLiteV3Config, MedallionLiteV3Strategy

# Research signal modules (relative import won't work here)
sys.path.insert(0, str(ROOT / "scripts" / "research" / "medallion_lite"))
from data_v2 import load_full_universe
from factors_v3 import compute_composite_v3, compute_factors_v3
from regime_ensemble import compute_ensemble_regime

sys.path.insert(0, str(ROOT / "scripts" / "research"))
from tearsheet_common_v0 import build_standard_html_tearsheet, compute_comprehensive_stats

OUT_DIR = ROOT / "scripts" / "research" / "medallion_lite" / "output"


def _pandas_to_polars_bars(df_sym: pd.DataFrame, symbol: str) -> pl.DataFrame:
    """Convert a pandas OHLCV slice for one symbol into the Polars schema
    expected by PortfolioEngine: ts, symbol, open, high, low, close, volume.
    """
    df = df_sym.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return pl.DataFrame({
        "ts": df["ts"].tolist(),
        "symbol": [symbol] * len(df),
        "open": df["open"].astype(float).tolist(),
        "high": df["high"].astype(float).tolist(),
        "low": df["low"].astype(float).tolist(),
        "close": df["close"].astype(float).tolist(),
        "volume": df["volume"].astype(float).tolist(),
    }).sort("ts")


def _hourly_to_daily_equity(equity_df: pl.DataFrame) -> pd.Series:
    """Convert hourly engine output to daily equity for the tearsheet."""
    pdf = equity_df.select("ts", "nav").to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"])
    pdf["date"] = pdf["ts"].dt.date
    daily = pdf.groupby("date")["nav"].last()
    daily.index = pd.to_datetime(daily.index)
    initial = daily.iloc[0]
    return (daily / initial).rename("equity")


def _daily_turnover(equity_df: pl.DataFrame) -> pd.Series:
    """Aggregate hourly turnover to daily."""
    pdf = equity_df.select("ts", "turnover").to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"])
    pdf["date"] = pdf["ts"].dt.date
    daily = pdf.groupby("date")["turnover"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def _build_btc_benchmark(panel: pd.DataFrame, equity_index: pd.DatetimeIndex) -> pd.Series:
    btc = panel[panel["symbol"] == "BTC-USD"].copy()
    btc["ts"] = pd.to_datetime(btc["ts"])
    btc_d = btc.set_index("ts")["close"].resample("D").last().dropna()
    btc_eq = btc_d / btc_d.iloc[0]
    btc_eq = btc_eq.reindex(equity_index, method="ffill").bfill()
    btc_eq.name = "benchmark_equity"
    return btc_eq


def _patch_log_scale(html_path: Path) -> None:
    text = html_path.read_text(encoding="utf-8")
    old = '"title": "Equity"}'
    new = '"title": "Equity (log)", "type": "log"}'
    if old in text:
        text = text.replace(old, new, 1)
        html_path.write_text(text, encoding="utf-8")
        print("[patch] Equity chart switched to log scale")
    else:
        print("[patch] WARNING: could not find equity yaxis to patch")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MEDALLION LITE V3 — HARDENED PortfolioEngine BACKTEST")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────
    t0 = time.time()
    panel, adv_wide = load_full_universe(
        start="2021-01-01", end="2026-12-31", min_adv_usd=100_000,
    )
    print(f"[data] Loaded in {time.time() - t0:.1f}s")

    df = panel.sort_values(["symbol", "ts"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    symbols = sorted(df["symbol"].unique().tolist())
    print(f"[data] Universe: {len(symbols)} symbols")

    # Pivot for signal computation
    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
    close_wide = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    high_wide = df.pivot(index="ts", columns="symbol", values="high").sort_index()
    low_wide = df.pivot(index="ts", columns="symbol", values="low").sort_index()
    volume_wide = df.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)

    # ── 2. Precompute regime ─────────────────────────────────────
    print("[regime] Computing ensemble regime ...")
    btc_h = panel[panel["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
    btc_d = btc_h.resample("D").last().dropna()
    regime = compute_ensemble_regime(btc_d, btc_h, returns_wide)
    print(f"[regime] Done — mean score: {regime.mean():.3f}")

    # ── 3. Precompute V3 factors + composite ─────────────────────
    print("[factors] Computing V3 factors ...")
    t1 = time.time()
    factors = compute_factors_v3(close_wide, volume_wide, high_wide, low_wide, adv_wide)
    composite = compute_composite_v3(factors, adv_wide, preset="v3_deep_signal")
    composite_smooth = composite.ewm(span=72, min_periods=24).mean()
    print(f"[factors] Done in {time.time() - t1:.1f}s")

    # ── 4. Find common coverage window ───────────────────────────
    # For inner-join alignment, only symbols with >=80% coverage survive.
    # Cap at top 40 by median ADV for tractable engine runtime.
    all_ts = sorted(df["ts"].unique())
    total_hours = len(all_ts)
    sym_coverage = df.groupby("symbol")["ts"].nunique()
    min_coverage = int(total_hours * 0.80)
    eligible = sym_coverage[sym_coverage >= min_coverage].index.tolist()

    # Also require symbol in composite columns
    eligible = [s for s in eligible if s in composite_smooth.columns]

    # Rank by median ADV and take top N for performance
    MAX_SYMBOLS = 40
    if len(eligible) > MAX_SYMBOLS:
        median_adv = adv_wide[eligible].median().sort_values(ascending=False)
        eligible = median_adv.head(MAX_SYMBOLS).index.tolist()

    eligible = sorted(eligible)
    print(f"[alignment] {len(eligible)}/{len(symbols)} symbols selected "
          f"(>=80% coverage, top {MAX_SYMBOLS} by ADV)")

    # Forward-fill per-symbol gaps for surviving symbols
    bars_by_symbol: dict[str, pl.DataFrame] = {}
    hourly_range = pd.date_range(
        start=pd.Timestamp(all_ts[0]), end=pd.Timestamp(all_ts[-1]), freq="h",
    )

    for sym in eligible:
        sym_df = df[df["symbol"] == sym][["ts", "open", "high", "low", "close", "volume"]].copy()
        sym_df["ts"] = pd.to_datetime(sym_df["ts"])
        sym_df = sym_df.set_index("ts").reindex(hourly_range).ffill().bfill().reset_index()
        sym_df = sym_df.rename(columns={"index": "ts"})
        sym_df["symbol"] = sym
        bars_by_symbol[sym] = _pandas_to_polars_bars(sym_df, sym)

    n_bars = bars_by_symbol[eligible[0]].height
    print(f"[alignment] {n_bars:,} aligned bars for {len(eligible)} symbols")

    # ── 5. Build RunConfig ───────────────────────────────────────
    sample_bars = bars_by_symbol[eligible[0]]
    start_ts = sample_bars[0, "ts"]
    end_ts = sample_bars[-1, "ts"]

    portfolio_cfg = PortfolioConfig(
        symbols=eligible,
        max_gross_leverage=1.0,
        max_net_leverage=1.0,
        max_single_name_weight=0.10,
    )

    raw_cfg = RunConfigRaw(
        run_name="medallion_lite_v3_hardened",
        data=DataConfig(
            db_path=":memory:",
            table="bars_1h",
            symbol=eligible[0],
            start=start_ts,
            end=end_ts,
            timeframe="1h",
        ),
        engine=EngineConfig(
            strict_validation=False,
            lookback=200,
            initial_cash=100_000.0,
        ),
        strategy=StrategyConfigRaw(mode="buy_and_hold", window_units="bars"),
        risk=RiskConfigRaw(
            vol_window=168,
            target_vol_annual=None,
            max_weight=1.0,
            window_units="bars",
        ),
        execution=ExecutionConfig(
            fee_bps=10.0,
            slippage_bps=5.0,
            execution_lag_bars=1,
            rebalance_deadband=0.003,
        ),
        portfolio=portfolio_cfg,
    )
    cfg = compile_config(raw_cfg)
    print(f"[config] Execution: {cfg.execution.fee_bps}bp fee + "
          f"{cfg.execution.slippage_bps}bp slip, "
          f"lag={cfg.execution.execution_lag_bars}, "
          f"deadband={cfg.execution.rebalance_deadband}")

    # ── 6. Build strategy ────────────────────────────────────────
    strat_cfg = MedallionLiteV3Config(
        core_slots=10,
        satellite_slots=25,
        core_capital_share=0.60,
    )
    strategy = MedallionLiteV3Strategy(
        regime_scores=regime,
        composite_scores=composite_smooth,
        cfg=strat_cfg,
    )

    # ── 7. Build risk manager (pass-through, no per-asset vol targeting) ──
    rm = RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor)

    # ── 8. Run through hardened PortfolioEngine ──────────────────
    print("\n[engine] Running PortfolioEngine ...")
    t2 = time.time()
    engine = PortfolioEngine(
        cfg=cfg,
        strategy=strategy,
        risk_manager=rm,
        bars_by_symbol=bars_by_symbol,
        portfolio_cfg=portfolio_cfg,
    )
    result: PortfolioResult = engine.run()
    elapsed = time.time() - t2
    print(f"[engine] Done in {elapsed:.1f}s")

    # ── 9. Print summary ─────────────────────────────────────────
    s = result.summary
    print("\n" + "=" * 70)
    print("  RESULTS (HARDENED ENGINE)")
    print("=" * 70)
    print(f"  Total Return:     {s['total_return_pct']:.1f}%")
    print(f"  Sharpe:           {s['sharpe']:.2f}")
    print(f"  Sortino:          {s['sortino']:.2f}")
    print(f"  Max Drawdown:     {s['max_drawdown']:.1%}")
    print(f"  Trade Count:      {s['trade_count']}")
    print(f"  Total Turnover:   {s['total_turnover']:.1f}")
    print(f"  Total Cost:       {s['total_cost']:.4f}")
    print(f"  Symbols:          {s['n_symbols']}")
    print(f"  Bars:             {s['n_bars']}")

    # Avg exposure
    held = result.weights_df.group_by("ts").agg(
        pl.col("held_weight").abs().sum().alias("gross")
    )
    avg_exp = float(held["gross"].mean())
    print(f"  Avg Exposure:     {avg_exp:.1%}")

    # CAGR
    nav = result.equity_df["nav"]
    years = n_bars / 8760.0
    total_ret = float(nav[-1]) / float(nav[0])
    cagr = total_ret ** (1 / years) - 1 if years > 0 else 0
    print(f"  CAGR:             {cagr:.1%}")
    print(f"  Cumulative:       {total_ret:.2f}x")
    print("=" * 70)

    # ── 10. Save artifacts ───────────────────────────────────────
    eq_path = OUT_DIR / "v3_hardened_equity.parquet"
    result.equity_df.write_parquet(eq_path)
    wt_path = OUT_DIR / "v3_hardened_weights.parquet"
    result.weights_df.write_parquet(wt_path)
    print(f"\n[save] Equity:  {eq_path}")
    print(f"[save] Weights: {wt_path}")

    # ── 11. Generate tearsheet ───────────────────────────────────
    print("\n[tearsheet] Generating HTML tearsheet ...")
    daily_eq = _hourly_to_daily_equity(result.equity_df)
    daily_turn = _daily_turnover(result.equity_df)
    btc_bench = _build_btc_benchmark(panel, daily_eq.index)

    stats = compute_comprehensive_stats(
        equity=daily_eq,
        benchmark_equity=btc_bench,
        turnover=daily_turn,
    )

    subtitle = (
        f"Hardened PortfolioEngine · Model B execution · "
        f"{cfg.execution.fee_bps:.0f}bp fee + {cfg.execution.slippage_bps:.0f}bp slip · "
        f"lag={cfg.execution.execution_lag_bars} · "
        f"Core/Satellite (10+25, 60/40) · "
        f"V3 Deep Signal 15 factors · "
        f"{len(eligible)} symbols, hourly"
    )

    html_path = OUT_DIR / "v3_hardened_tearsheet.html"
    build_standard_html_tearsheet(
        out_html=str(html_path),
        strategy_label="Medallion Lite V3 — Hardened Engine",
        strategy_equity=daily_eq,
        strategy_stats=stats,
        benchmark_equity=btc_bench,
        benchmark_label="BTC Buy & Hold",
        subtitle=subtitle,
        turnover=daily_turn,
        auto_open=True,
        confidential_footer=True,
    )
    _patch_log_scale(html_path)
    print(f"\n[tearsheet] {html_path}")
    print("Done.")


if __name__ == "__main__":
    main()
