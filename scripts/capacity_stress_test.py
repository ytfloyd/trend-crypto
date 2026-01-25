#!/usr/bin/env python
"""Capacity stress test using dynamic slippage impact model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from backtest.engine import BacktestEngine
from common.config import compile_config, load_config_from_yaml, make_run_id, write_manifest
from data.portal import DataPortal
from risk.risk_manager import RiskManager
from strategy.buy_and_hold import BuyAndHoldStrategy
from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy
from strategy.ma_cross_vol_hysteresis import MACrossVolHysteresis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capacity stress test (Sharpe decay vs AUM)")
    p.add_argument("--config", required=True, help="Base YAML config")
    p.add_argument("--assets", required=True, help="Comma-separated symbols")
    p.add_argument("--aum-levels", required=True, help="Comma-separated USD values")
    p.add_argument("--impact-coeff", type=float, default=0.1, help="Impact coefficient")
    p.add_argument("--output", required=True, help="Output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    aum_levels = [float(x.strip()) for x in args.aum_levels.split(",") if x.strip()]

    raw_base = load_config_from_yaml(args.config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for asset in assets:
        for aum in aum_levels:
            raw_cfg = raw_base.model_copy(
                update={
                    "data": raw_base.data.model_copy(update={"symbol": asset}),
                }
            )
            cfg = compile_config(raw_cfg)
            portal = DataPortal(cfg.data, strict_validation=cfg.engine.strict_validation)
            if cfg.strategy.mode == "buy_and_hold":
                strategy = BuyAndHoldStrategy(cfg.strategy)
            elif cfg.strategy.mode == "ma_crossover_long_only":
                strategy = MACrossoverLongOnlyStrategy(
                    fast=cfg.strategy.fast,
                    slow=cfg.strategy.slow,
                    weight_on=cfg.strategy.weight_on,
                    target_vol_annual=cfg.strategy.target_vol_annual,
                    vol_lookback=cfg.strategy.vol_lookback or 20,
                    max_weight=cfg.strategy.max_weight,
                    enable_adx_filter=cfg.strategy.enable_adx_filter,
                    adx_window=cfg.strategy.adx_window,
                    adx_threshold=cfg.strategy.adx_threshold,
                    adx_entry_only=cfg.strategy.adx_entry_only,
                )
            else:
                strategy = MACrossVolHysteresis(cfg.strategy)

            engine = BacktestEngine(
                cfg,
                strategy,
                RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor),
                portal,
                use_dynamic_slippage=True,
                aum_usd=aum,
                impact_coeff=args.impact_coeff,
            )
            bars = portal.load_bars()
            _, summary = engine.run()

            # Ensure manifest exists for each run
            run_id = make_run_id(f"capacity_{asset}_{int(aum)}")
            run_dir = Path("artifacts") / "capacity_stress" / run_id
            write_manifest(
                run_dir,
                cfg,
                bars_start=bars[0, "ts"],
                bars_end=bars.item(bars.height - 1, "ts"),
                data_provenance=portal.last_provenance,
            )

            verdict = []
            if summary.get("sharpe", 0.0) < 1.0:
                verdict.append("FAIL (Capacity Breached)")
            if summary.get("sharpe", 0.0) < 0.75:
                verdict.append("DEAD")
            if summary.get("max_impact_bps", 0.0) > 250.0:
                verdict.append("LIQUIDITY STRESS")

            rows.append(
                {
                    "config_hash": cfg.compute_hash(),
                    "asset": asset,
                    "aum_usd": aum,
                    "sharpe": summary.get("sharpe"),
                    "max_drawdown": summary.get("max_drawdown"),
                    "total_return": summary.get("total_return_decimal"),
                    "trade_count": summary.get("trade_count"),
                    "avg_impact_bps": summary.get("avg_impact_bps"),
                    "max_impact_bps": summary.get("max_impact_bps"),
                    "p99_impact_bps": summary.get("p99_impact_bps"),
                    "verdict": "; ".join(verdict) if verdict else "OK",
                }
            )

    # Sort by AUM
    rows.sort(key=lambda r: (r["asset"], r["aum_usd"]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Console table
    print("asset,aum_usd,sharpe,max_dd,avg_bps,max_bps,p99_bps,verdict")
    for r in rows:
        print(
            f"{r['asset']},{r['aum_usd']:.0f},{r['sharpe']:.3f},"
            f"{r['max_drawdown']:.3f},{r['avg_impact_bps']:.1f},"
            f"{r['max_impact_bps']:.1f},{r['p99_impact_bps']:.1f},"
            f"{r['verdict']}"
        )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
