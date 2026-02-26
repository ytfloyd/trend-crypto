#!/usr/bin/env python3
"""
TSMOM Long Convexity Engine — Experiment Runner
=================================================

Pre-registered primary specification:
  Signal     : VOL_SCALED (trailing return / realized vol)
  Lookback   : 21 days
  Sizing     : binary (equal risk per position)
  Exit       : signal reversal only
  Vol target : 15 % annualised (portfolio-level second pass)
  Max weight : 20 % per asset (excess → cash)

Runs the primary spec first, then a full sensitivity grid.

Usage:
    python -m scripts.research.tsmom.run_tsmom
    python -m scripts.research.tsmom.run_tsmom --primary-only
    python -m scripts.research.tsmom.run_tsmom --db ../data/market.duckdb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, filter_universe, compute_btc_benchmark, ANN_FACTOR
from common.backtest import simple_backtest, DEFAULT_COST_BPS
from common.metrics import compute_metrics, format_metrics_table

from .signals import compute_signal, SIGNAL_FUNCTIONS
from .weights import (
    build_tsmom_weights,
    apply_portfolio_vol_target,
    apply_vol_adaptive_trailing_stop,
    apply_vol_spike_exit,
)
from .convexity_metrics import (
    compute_convexity_metrics,
    classify_regime,
    conditional_correlation,
    time_in_market_by_regime,
    regime_sharpe_skew,
    participation_ratio_portfolio,
    participation_ratio_per_asset,
    bootstrap_sharpe,
    bootstrap_skewness,
    extract_crisis_timeline,
    CRISIS_EPISODES,
)


# ── Config ────────────────────────────────────────────────────────────

# Primary spec (pre-registered)
PRIMARY = {
    "signal": "VOL_SCALED",
    "lookback": 21,
    "vol_lookback": 63,
    "sizing": "binary",
    "exit": "signal_reversal",
    "vol_target": 0.15,
    "max_weight": 0.20,
}

# Sensitivity grid
SIGNAL_NAMES = list(SIGNAL_FUNCTIONS.keys())
LOOKBACKS = [5, 10, 21, 42, 63, 126, 252]
SIZING_METHODS = ["binary", "proportional", "capped"]
EXIT_METHODS = ["signal_reversal", "trailing_stop_1.5", "trailing_stop_2.0",
                "trailing_stop_2.5", "vol_spike", "combined_2.0"]
VOL_TARGETS = [0.10, 0.15, 0.20]

START = "2017-01-01"
END = "2025-12-15"
COST_BPS = DEFAULT_COST_BPS

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "tsmom"


# ── Data loading ──────────────────────────────────────────────────────

def prepare_data(
    db_path: str | None = None,
    start: str = START,
    end: str = END,
) -> tuple:
    """Load and prepare wide-format matrices."""
    print(f"[tsmom] Loading data ({start} to {end}) ...")

    if db_path:
        panel = load_daily_bars(db_path=db_path, start=start, end=end)
    else:
        panel = load_daily_bars(start=start, end=end)

    panel = filter_universe(panel, min_adv_usd=500_000, min_history_days=90)
    panel = panel.sort_values(["ts", "symbol"])

    close_wide = panel.pivot(index="ts", columns="symbol", values="close")
    volume_wide = panel.pivot(index="ts", columns="symbol", values="volume")
    returns_wide = close_wide.pct_change(fill_method=None)
    universe_wide = (
        panel.pivot(index="ts", columns="symbol", values="in_universe")
        .fillna(False).infer_objects(copy=False).astype(bool)
    )

    btc_equity = compute_btc_benchmark(panel)

    # BTC close for crisis timelines
    btc_close = close_wide[[c for c in close_wide.columns if "BTC" in c.upper()]]
    btc_col = btc_close.columns[0] if len(btc_close.columns) > 0 else None
    btc_close_series = btc_close[btc_col] if btc_col else pd.Series(dtype=float)

    n_assets = universe_wide.sum(axis=1).median()
    print(f"[tsmom] Data ready: {len(close_wide)} days, ~{n_assets:.0f} assets in universe")

    return (close_wide, volume_wide, returns_wide, universe_wide,
            btc_equity, btc_close_series, btc_col)


# ── Single configuration runner ──────────────────────────────────────

def run_single_config(
    config: dict,
    close_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    universe_wide: pd.DataFrame,
    btc_equity: pd.Series,
    btc_close_series: pd.Series,
    btc_col: str | None,
    cost_bps: float = COST_BPS,
) -> dict:
    """Run a single TSMOM configuration and return full results."""
    t0 = time.time()

    sig_name = config["signal"]
    lookback = config["lookback"]
    vol_lookback = config.get("vol_lookback", 63)
    sizing = config["sizing"]
    exit_method = config["exit"]
    vol_target = config["vol_target"]
    max_weight = config["max_weight"]

    label = f"{sig_name}_{lookback}d_{sizing}_{exit_method}_vt{int(vol_target*100)}"

    # 1. Compute signal
    signal = compute_signal(sig_name, close_wide, returns_wide, lookback,
                            vol_lookback=vol_lookback)

    # 2. Build long-or-cash weights
    weights = build_tsmom_weights(
        signal, universe_wide, returns_wide,
        sizing=sizing, vol_target=vol_target,
        vol_lookback=vol_lookback, max_weight=max_weight,
    )

    # 3. Apply exit overlay
    if exit_method == "signal_reversal":
        pass  # Already embedded in weight construction
    elif exit_method.startswith("trailing_stop_"):
        k = float(exit_method.split("_")[-1])
        weights = apply_vol_adaptive_trailing_stop(
            weights, close_wide, returns_wide, atr_multiple=k,
        )
    elif exit_method == "vol_spike":
        weights = apply_vol_spike_exit(weights, returns_wide)
    elif exit_method.startswith("combined_"):
        k = float(exit_method.split("_")[-1])
        weights = apply_vol_adaptive_trailing_stop(
            weights, close_wide, returns_wide, atr_multiple=k,
        )

    # 4. Portfolio-level vol targeting (second pass)
    weights_vt = apply_portfolio_vol_target(
        weights, returns_wide, vol_target=vol_target,
    )

    # 5. Backtest
    bt = simple_backtest(weights_vt, returns_wide, cost_bps=cost_bps)
    if bt.empty or len(bt) < 30:
        return {"label": label, "config": config, "error": "insufficient data",
                "elapsed_sec": round(time.time() - t0, 1)}

    equity = bt.set_index("ts")["portfolio_equity"]
    port_ret = bt.set_index("ts")["portfolio_ret"]

    # 6. Convexity metrics
    metrics = compute_convexity_metrics(equity, weights_vt)
    metrics["label"] = label
    metrics["avg_turnover"] = round(float(bt["turnover"].mean()), 5)
    metrics["avg_gross_exposure"] = round(float(bt["gross_exposure"].mean()), 3)

    # 7. Regime analysis
    regime = classify_regime(returns_wide, btc_col)
    metrics["regime_sharpe_skew"] = regime_sharpe_skew(port_ret, regime)
    metrics["time_in_market_by_regime"] = time_in_market_by_regime(weights_vt, regime)

    # 8. Conditional correlation
    if btc_col and btc_col in returns_wide.columns:
        btc_ret = returns_wide[btc_col]
        metrics["conditional_corr"] = conditional_correlation(port_ret, btc_ret, regime)
    else:
        metrics["conditional_corr"] = {}

    # 9. Participation ratios
    metrics["participation_portfolio"] = participation_ratio_portfolio(equity, btc_equity)
    metrics["participation_per_asset"] = participation_ratio_per_asset(
        weights_vt, returns_wide, universe_wide,
    )

    metrics["config"] = config
    metrics["elapsed_sec"] = round(time.time() - t0, 1)

    return {
        "metrics": metrics,
        "equity": equity,
        "port_ret": port_ret,
        "weights": weights_vt,
        "bt": bt,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TSMOM Long Convexity Engine")
    parser.add_argument("--db", type=str, default=None, help="Path to DuckDB file")
    parser.add_argument("--primary-only", action="store_true",
                        help="Run only the pre-registered primary spec")
    parser.add_argument("--start", type=str, default=START)
    parser.add_argument("--end", type=str, default=END)
    parser.add_argument("--cost-bps", type=float, default=COST_BPS)
    parser.add_argument("--output", type=str, default=str(ARTIFACT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TSMOM LONG CONVEXITY ENGINE")
    print("=" * 70)

    (close_wide, volume_wide, returns_wide, universe_wide,
     btc_equity, btc_close_series, btc_col) = prepare_data(
        db_path=args.db, start=args.start, end=args.end,
    )

    # =================================================================
    # PRIMARY SPECIFICATION
    # =================================================================
    print("\n" + "=" * 70)
    print("  PRIMARY SPEC (pre-registered)")
    print("  " + " | ".join(f"{k}={v}" for k, v in PRIMARY.items()))
    print("=" * 70)

    primary_result = run_single_config(
        PRIMARY, close_wide, returns_wide, universe_wide,
        btc_equity, btc_close_series, btc_col, cost_bps=args.cost_bps,
    )

    if "error" in primary_result:
        print(f"  [ERROR] Primary spec failed: {primary_result['error']}")
    else:
        pm = primary_result["metrics"]
        print(f"\n  --- PRIMARY SPEC RESULTS ---")
        print(f"  Sharpe:     {pm.get('sharpe', np.nan):>8.3f}")
        print(f"  Skewness:   {pm.get('skewness', np.nan):>8.3f}")
        print(f"  CAGR:       {pm.get('cagr', np.nan):>8.1%}")
        print(f"  MaxDD:      {pm.get('max_dd', np.nan):>8.1%}")
        print(f"  Hit Rate:   {pm.get('hit_rate', np.nan):>8.1%}")
        print(f"  Win/Loss:   {pm.get('win_loss_ratio', np.nan):>8.2f}")
        print(f"  Time in Mkt:{pm.get('time_in_market', np.nan):>8.1%}")
        print(f"  Turnover:   {pm.get('avg_turnover', np.nan):>8.4f}")
        print(f"  Part.(port):{pm.get('participation_portfolio', np.nan):>8.1%}")
        print(f"  Part.(asset):{pm.get('participation_per_asset', np.nan):>8.1%}")

        # Regime breakdown
        print(f"\n  --- REGIME ANALYSIS ---")
        for r_label in ["BULL", "BEAR", "CHOP"]:
            rs = pm.get("regime_sharpe_skew", {}).get(r_label, {})
            tim = pm.get("time_in_market_by_regime", {}).get(r_label, np.nan)
            cc = pm.get("conditional_corr", {}).get(r_label, np.nan)
            print(f"  {r_label:5s}: Sharpe={rs.get('sharpe', np.nan):>6.2f}  "
                  f"Skew={rs.get('skewness', np.nan):>6.2f}  "
                  f"TimeIn={tim:>5.1%}  BTC_corr={cc:>6.3f}")

        # Bootstrap CIs
        print(f"\n  --- BOOTSTRAP 95% CI ---")
        port_ret = primary_result["port_ret"]
        invested_ret = port_ret[port_ret.abs() > 1e-10]
        if len(invested_ret) > 50:
            sh_pt, sh_lo, sh_hi = bootstrap_sharpe(invested_ret)
            sk_pt, sk_lo, sk_hi = bootstrap_skewness(invested_ret)
            print(f"  Sharpe:   {sh_pt:>6.3f}  [{sh_lo:>6.3f}, {sh_hi:>6.3f}]")
            print(f"  Skewness: {sk_pt:>6.3f}  [{sk_lo:>6.3f}, {sk_hi:>6.3f}]")

        # Crisis timelines
        print(f"\n  --- CRISIS TIMELINES ---")
        crisis_data = {}
        for ep_name in CRISIS_EPISODES:
            ct = extract_crisis_timeline(
                btc_close_series, primary_result["weights"],
                primary_result["port_ret"], ep_name,
            )
            if ct is not None:
                crisis_data[ep_name] = ct
                avg_wt = ct["total_weight"].mean()
                cum_pnl = (1 + ct["daily_pnl"]).prod() - 1
                btc_move = ct["btc_price"].iloc[-1] / ct["btc_price"].iloc[0] - 1
                print(f"  {ep_name:<20s}: avg_weight={avg_wt:.1%}  "
                      f"strat_return={cum_pnl:.1%}  btc_return={btc_move:.1%}")

        # Save primary results
        primary_output = {
            "config": PRIMARY,
            "metrics": {k: v for k, v in pm.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame))},
        }
        with open(output_dir / "primary_spec_results.json", "w") as f:
            json.dump(primary_output, f, indent=2, default=str)

        primary_result["equity"].to_csv(output_dir / "primary_equity.csv")
        primary_result["weights"].to_parquet(output_dir / "primary_weights.parquet")

        for ep_name, ct in crisis_data.items():
            safe_name = ep_name.replace(" ", "_").replace("(", "").replace(")", "")
            ct.to_csv(output_dir / f"crisis_{safe_name}.csv")

    # =================================================================
    # PASS / FAIL CHECK ON PRIMARY
    # =================================================================
    if "error" not in primary_result:
        pm = primary_result["metrics"]
        print(f"\n  --- PRIMARY SPEC PASS/FAIL ---")
        checks = {
            "Skewness > 0":       pm.get("skewness", -99) > 0,
            "Sharpe > 0":         pm.get("sharpe", -99) > 0,
            "MaxDD > -30%":       pm.get("max_dd", -99) > -0.30,
            "BEAR corr < 0.5":    pm.get("conditional_corr", {}).get("BEAR", 99) < 0.5,
            "Participation > 20%": pm.get("participation_per_asset", 0) > 0.20,
        }
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check_name}")

        all_pass = all(checks.values())
        print(f"\n  {'>>> DEPLOY CANDIDATE <<<' if all_pass else '>>> NO DEPLOY — see failures above'}")

    if args.primary_only:
        print(f"\n[tsmom] Primary-only mode. Done.")
        print(f"[tsmom] Artifacts: {output_dir}")
        return

    # =================================================================
    # SENSITIVITY GRID
    # =================================================================
    print("\n" + "=" * 70)
    print("  SENSITIVITY GRID")
    print("=" * 70)

    configs = []
    # Signal x Lookback (primary sizing/exit/vol_target)
    for sig in SIGNAL_NAMES:
        for lb in LOOKBACKS:
            configs.append({
                "signal": sig, "lookback": lb,
                "vol_lookback": 63, "sizing": "binary",
                "exit": "signal_reversal", "vol_target": 0.15,
                "max_weight": 0.20,
            })

    # Sizing variants (primary signal/lookback)
    for sizing in SIZING_METHODS:
        if sizing == "binary":
            continue
        configs.append({
            "signal": "VOL_SCALED", "lookback": 21,
            "vol_lookback": 63, "sizing": sizing,
            "exit": "signal_reversal", "vol_target": 0.15,
            "max_weight": 0.20,
        })

    # Exit variants (primary signal/lookback/sizing)
    for exit_m in EXIT_METHODS:
        if exit_m == "signal_reversal":
            continue
        configs.append({
            "signal": "VOL_SCALED", "lookback": 21,
            "vol_lookback": 63, "sizing": "binary",
            "exit": exit_m, "vol_target": 0.15,
            "max_weight": 0.20,
        })

    # Vol target variants (primary everything else)
    for vt in VOL_TARGETS:
        if vt == 0.15:
            continue
        configs.append({
            "signal": "VOL_SCALED", "lookback": 21,
            "vol_lookback": 63, "sizing": "binary",
            "exit": "signal_reversal", "vol_target": vt,
            "max_weight": 0.20,
        })

    # Deduplicate (primary spec is already in the signal x lookback grid)
    seen = set()
    unique_configs = []
    for c in configs:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    n_total = len(unique_configs)
    print(f"  Total configurations: {n_total}")

    all_results = []
    t_grid_start = time.time()

    for i, config in enumerate(unique_configs):
        label = (f"{config['signal']}_{config['lookback']}d_"
                 f"{config['sizing']}_{config['exit']}_vt{int(config['vol_target']*100)}")
        print(f"  [{i+1}/{n_total}] {label} ...", end="", flush=True)

        try:
            result = run_single_config(
                config, close_wide, returns_wide, universe_wide,
                btc_equity, btc_close_series, btc_col, cost_bps=args.cost_bps,
            )
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                all_results.append({"label": label, "config": config, "error": result["error"]})
            else:
                m = result["metrics"]
                print(f"  Sharpe={m.get('sharpe', np.nan):.2f}  "
                      f"Skew={m.get('skewness', np.nan):.2f}  "
                      f"CAGR={m.get('cagr', np.nan):.1%}  "
                      f"MaxDD={m.get('max_dd', np.nan):.1%}  "
                      f"({m.get('elapsed_sec', 0):.1f}s)")
                flat_m = {}
                for k, v in m.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, dict):
                                for sub_sub_k, sub_sub_v in sub_v.items():
                                    flat_m[f"{k}_{sub_k}_{sub_sub_k}"] = sub_sub_v
                            else:
                                flat_m[f"{k}_{sub_k}"] = sub_v
                    elif isinstance(v, (int, float, str, bool, type(None))):
                        flat_m[k] = v
                all_results.append(flat_m)

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            all_results.append({"label": label, "config": config, "error": str(e)})

    grid_elapsed = time.time() - t_grid_start
    print(f"\n  Grid complete: {n_total} configs in {grid_elapsed:.0f}s "
          f"({grid_elapsed/60:.1f}min)")

    # Save grid results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "sensitivity_grid.csv", index=False, float_format="%.5f")
    print(f"  Saved: {output_dir / 'sensitivity_grid.csv'}")

    # ── Summary tables ────────────────────────────────────────────────
    valid = results_df.dropna(subset=["sharpe"])
    if len(valid) > 0:
        print(f"\n  --- TOP 10 BY SKEWNESS ---")
        top_skew = valid.nlargest(10, "skewness")
        for _, row in top_skew.iterrows():
            print(f"  {row.get('label', '?'):<50s}  "
                  f"Skew={row.get('skewness', np.nan):>6.2f}  "
                  f"Sharpe={row.get('sharpe', np.nan):>6.2f}  "
                  f"MaxDD={row.get('max_dd', np.nan):>7.1%}")

        print(f"\n  --- TOP 10 BY SHARPE ---")
        top_sharpe = valid.nlargest(10, "sharpe")
        for _, row in top_sharpe.iterrows():
            print(f"  {row.get('label', '?'):<50s}  "
                  f"Sharpe={row.get('sharpe', np.nan):>6.2f}  "
                  f"Skew={row.get('skewness', np.nan):>6.2f}  "
                  f"MaxDD={row.get('max_dd', np.nan):>7.1%}")

    # ── BTC benchmark ─────────────────────────────────────────────────
    btc_aligned = btc_equity.reindex(close_wide.index).ffill().dropna()
    if len(btc_aligned) > 30:
        btc_m = compute_metrics(btc_aligned / btc_aligned.iloc[0])
        print(f"\n  --- BTC BUY & HOLD ---")
        print(f"  Sharpe={btc_m.get('sharpe', np.nan):.2f}  "
              f"Skew={btc_m.get('skewness', np.nan):.2f}  "
              f"CAGR={btc_m.get('cagr', np.nan):.1%}  "
              f"MaxDD={btc_m.get('max_dd', np.nan):.1%}")

    total_elapsed = time.time() - t_grid_start
    print(f"\n{'='*70}")
    print(f"  TSMOM EXPERIMENT COMPLETE")
    print(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Artifacts: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
