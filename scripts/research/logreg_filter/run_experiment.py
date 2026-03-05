"""
End-to-end experiment runner for the logistic regression probability filter.

Pipeline: data → features → labels → walk-forward model → overlay variants
→ backtest each → metrics → report.

Usage:
    python -m scripts.research.logreg_filter.run_experiment \
        --config configs/research/logreg_filter_v0.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.research.common.backtest import simple_backtest
from scripts.research.common.data import (
    compute_btc_benchmark,
    filter_universe,
    load_daily_bars,
)
from scripts.research.common.metrics import compute_metrics, compute_regime

from .features import (
    FeatureConfig,
    compute_cross_asset_features,
    compute_features_panel,
    get_all_feature_columns,
)
from .labels import (
    BarrierLabelConfig,
    ForwardReturnLabelConfig,
    LabelType,
    UnresolvedPolicy,
    compute_labels_panel,
)
from .model import ModelConfig, ModelOutput, WalkForwardConfig, train_walk_forward
from .overlay import OverlayConfig, build_overlay_variants


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
def _parse_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_feature_cfg(raw: dict) -> FeatureConfig:
    return FeatureConfig(**{k: v for k, v in raw.items() if k in FeatureConfig.__dataclass_fields__})


def _make_barrier_cfg(raw: dict) -> BarrierLabelConfig:
    policy = raw.get("unresolved_policy", "conservative")
    return BarrierLabelConfig(
        tp_atr=raw.get("tp_atr", 2.0),
        sl_atr=raw.get("sl_atr", 1.0),
        horizon=raw.get("horizon", 20),
        atr_window=raw.get("atr_window", 14),
        unresolved_policy=UnresolvedPolicy(policy),
    )


def _make_fwd_cfg(raw: dict) -> ForwardReturnLabelConfig:
    return ForwardReturnLabelConfig(**{
        k: v for k, v in raw.items()
        if k in ForwardReturnLabelConfig.__dataclass_fields__
    })


def _make_model_cfg(raw: dict) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})


def _make_wf_cfg(raw: dict) -> WalkForwardConfig:
    return WalkForwardConfig(**{
        k: v for k, v in raw.items() if k in WalkForwardConfig.__dataclass_fields__
    })


def _make_overlay_cfg(raw: dict) -> OverlayConfig:
    return OverlayConfig(**{k: v for k, v in raw.items() if k in OverlayConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Base strategy: top-N momentum
# ---------------------------------------------------------------------------
def build_base_weights(
    panel: pd.DataFrame,
    momentum_window: int = 20,
    top_n: int = 10,
    rebalance_freq: int = 1,
) -> pd.DataFrame:
    """Build equal-weight top-N momentum portfolio weights.

    At each rebalance date, select the top_n assets (within universe)
    by trailing momentum and assign equal weight = 1/top_n.

    Returns wide-format weights (index=ts, columns=symbols).
    """
    df = panel.loc[panel["in_universe"]].copy()
    df = df.sort_values(["symbol", "ts"])
    df["mom"] = df.groupby("symbol")["close"].transform(
        lambda s: s / s.shift(momentum_window) - 1.0,
    )

    dates = sorted(df["ts"].unique())
    rebalance_dates = dates[::rebalance_freq]

    all_symbols = sorted(df["symbol"].unique())
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=all_symbols)

    for dt in rebalance_dates:
        day = df.loc[df["ts"] == dt].dropna(subset=["mom"])
        if len(day) == 0:
            continue
        top = day.nlargest(top_n, "mom")["symbol"].tolist()
        w = 1.0 / max(len(top), 1)
        for sym in top:
            weights.at[dt, sym] = w

    weights = weights.ffill()
    return weights


def build_returns_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Build wide-format open-to-close returns from panel data."""
    ret_parts = []
    for sym, g in panel.groupby("symbol"):
        g = g.sort_values("ts").set_index("ts")
        r = g["close"].pct_change(fill_method=None)
        r.name = sym
        ret_parts.append(r)
    ret_wide = pd.concat(ret_parts, axis=1).sort_index()
    return ret_wide


# ---------------------------------------------------------------------------
# Regime heuristic
# ---------------------------------------------------------------------------
def compute_regime_heuristic(
    panel: pd.DataFrame, ma_window: int = 50,
) -> pd.Series:
    """BTC MA slope heuristic: p_regime=1 if slope > 0, else 0."""
    btc = panel.loc[panel["symbol"] == "BTC-USD"].sort_values("ts").set_index("ts")
    if len(btc) < ma_window + 5:
        return pd.Series(0.5, index=panel["ts"].unique())
    ma = btc["close"].rolling(ma_window, min_periods=ma_window).mean()
    slope = ma - ma.shift(5)
    p_regime = (slope > 0).astype(float)
    p_regime.name = "p_regime"
    return p_regime


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(config_path: str) -> dict:
    """Execute the full experiment pipeline.

    Returns dict with metrics for each variant.
    """
    cfg = _parse_config(config_path)
    out_dir = Path(cfg.get("output", {}).get("dir", "artifacts/research/logreg_filter"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    shutil.copy2(config_path, out_dir / "config_snapshot.yaml")

    data_cfg = cfg.get("data", {})
    exec_cfg = cfg.get("execution", {})
    cost_bps = exec_cfg.get("cost_bps", 20.0)
    execution_lag = exec_cfg.get("execution_lag", 1)

    # ── Step 1: Load data ──
    print("=" * 70)
    print("Logistic Regression Probability Filter — Experiment")
    print("=" * 70)
    print("\n[1/7] Loading data ...")
    panel = load_daily_bars(
        start=data_cfg.get("start", "2017-01-01"),
        end=data_cfg.get("end", "2026-12-31"),
    )
    panel = filter_universe(
        panel,
        min_adv_usd=data_cfg.get("min_adv_usd", 500_000),
        min_history_days=data_cfg.get("min_history_days", 90),
    )
    print(f"  Universe: {panel['symbol'].nunique()} assets, "
          f"{panel['ts'].nunique()} dates")

    # ── Step 2: Features ──
    print("\n[2/7] Computing features ...")
    feat_cfg = _make_feature_cfg(cfg.get("features", {}))
    featured = compute_features_panel(panel, feat_cfg)
    feature_cols = get_all_feature_columns(feat_cfg)
    print(f"  {len(feature_cols)} features: {feature_cols}")

    # ── Step 3: Labels ──
    print("\n[3/7] Computing labels ...")
    label_cfg = cfg.get("label", {})
    label_type = LabelType(label_cfg.get("type", "barrier"))

    barrier_cfg = _make_barrier_cfg(label_cfg.get("barrier", {}))
    fwd_cfg = _make_fwd_cfg(label_cfg.get("forward_return", {}))

    labels = compute_labels_panel(
        panel, label_type=label_type,
        barrier_cfg=barrier_cfg, fwd_cfg=fwd_cfg,
    )
    featured = featured.merge(labels, on=["ts", "symbol"], how="left")
    n_labels = featured["label"].notna().sum()
    base_rate = featured["label"].mean()
    print(f"  {n_labels:,} labelled observations, base rate = {base_rate:.1%}")

    # ── Step 4: Walk-forward model training ──
    print("\n[4/7] Training walk-forward logistic regression ...")
    model_cfg = _make_model_cfg(cfg.get("model", {}))
    wf_cfg = _make_wf_cfg(cfg.get("walk_forward", {}))
    model_output: ModelOutput = train_walk_forward(
        featured, feature_cols, label_col="label",
        model_cfg=model_cfg, wf_cfg=wf_cfg,
    )
    n_folds = len(model_output.fold_results)
    mean_auc = np.nanmean([fr.auc for fr in model_output.fold_results])
    print(f"  {n_folds} folds, mean AUC = {mean_auc:.3f}")

    # ── Step 5: Base strategy + overlay variants ──
    print("\n[5/7] Building base strategy and overlay variants ...")
    base_cfg = cfg.get("base_strategy", {})
    base_weights = build_base_weights(
        panel,
        momentum_window=base_cfg.get("momentum_window", 20),
        top_n=base_cfg.get("top_n", 10),
        rebalance_freq=base_cfg.get("rebalance_freq", 1),
    )

    # Regime
    regime_cfg = cfg.get("regime", {})
    p_regime = None
    if regime_cfg.get("enabled", True):
        if regime_cfg.get("mode", "heuristic") == "heuristic":
            p_regime = compute_regime_heuristic(
                panel, ma_window=regime_cfg.get("heuristic_ma_window", 50),
            )
        print(f"  Regime mode: {regime_cfg.get('mode', 'heuristic')}")

    overlay_cfg = _make_overlay_cfg(cfg.get("overlay", {}))
    variants = build_overlay_variants(
        base_weights, model_output.predictions,
        p_regime=p_regime, cfg=overlay_cfg,
    )
    print(f"  Variants: {list(variants.keys())}")

    # ── Step 6: Backtest each variant ──
    print("\n[6/7] Backtesting variants ...")
    returns_wide = build_returns_wide(panel)

    results = {}
    equity_curves = {}
    backtest_dfs = {}
    for name, w in variants.items():
        bt = simple_backtest(w, returns_wide, cost_bps=cost_bps,
                             execution_lag=execution_lag)
        eq = bt.set_index("ts")["portfolio_equity"]
        eq.name = name
        metrics = compute_metrics(eq)
        metrics["label"] = name
        metrics["avg_turnover"] = bt["turnover"].mean()
        metrics["total_fees"] = bt["cost_ret"].sum()
        metrics["avg_exposure"] = bt["gross_exposure"].mean()
        results[name] = metrics
        equity_curves[name] = eq
        backtest_dfs[name] = bt
        print(f"  {name:30s} Sharpe={metrics['sharpe']:.2f}  "
              f"CAGR={metrics['cagr']:.1%}  MaxDD={metrics['max_dd']:.1%}  "
              f"Turnover={metrics['avg_turnover']:.4f}")

    # ── Step 7: Threshold sweep ──
    print("\n[7/7] Running threshold sweep ...")
    sweep_cfg = cfg.get("threshold_sweep", {})
    p_values = sweep_cfg.get("p_enter_values", [0.50, 0.55, 0.60, 0.65, 0.70])
    sweep_results = []
    for p_val in p_values:
        sweep_overlay = OverlayConfig(
            p_enter=p_val,
            max_weight=overlay_cfg.max_weight,
            target_gross=overlay_cfg.target_gross,
            max_positions=overlay_cfg.max_positions,
        )
        sweep_variants = build_overlay_variants(
            base_weights, model_output.predictions,
            p_regime=p_regime, cfg=sweep_overlay,
        )
        best_key = "filter_sizing_regime" if "filter_sizing_regime" in sweep_variants else "filter_sizing"
        w = sweep_variants[best_key]
        bt = simple_backtest(w, returns_wide, cost_bps=cost_bps,
                             execution_lag=execution_lag)
        eq = bt.set_index("ts")["portfolio_equity"]
        m = compute_metrics(eq)
        m["p_enter"] = p_val
        m["avg_turnover"] = bt["turnover"].mean()
        m["total_fees"] = bt["cost_ret"].sum()
        sweep_results.append(m)
        print(f"  p_enter={p_val:.2f}  Sharpe={m['sharpe']:.2f}  "
              f"CAGR={m['cagr']:.1%}  MaxDD={m['max_dd']:.1%}")

    # ── Save artifacts ──
    print(f"\nSaving artifacts to {out_dir} ...")

    # Metrics summary
    pd.DataFrame(results).T.to_csv(out_dir / "variant_metrics.csv")
    pd.DataFrame(sweep_results).to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Model diagnostics
    model_output.fold_metrics_df().to_csv(out_dir / "fold_metrics.csv", index=False)
    model_output.coefficient_summary().to_csv(out_dir / "coefficient_summary.csv", index=False)

    # Predictions
    if cfg.get("output", {}).get("save_predictions", True):
        model_output.predictions.to_csv(out_dir / "predictions.csv", index=False)

    # Equity curves
    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_csv(out_dir / "equity_curves.csv")

    # Generate PDF report
    if cfg.get("output", {}).get("generate_pdf", True):
        try:
            from .report import generate_report
            generate_report(
                results=results,
                sweep_results=sweep_results,
                equity_curves=equity_curves,
                backtest_dfs=backtest_dfs,
                model_output=model_output,
                cfg=cfg,
                out_path=out_dir / "logreg_filter_report.pdf",
            )
        except Exception as e:
            print(f"  [warn] PDF generation failed: {e}")

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "n_folds": n_folds,
        "mean_auc": float(mean_auc),
        "variants": list(results.keys()),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    print(f"\nDone. Artifacts saved to {out_dir}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Logistic Regression Probability Filter Experiment",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/research/logreg_filter_v0.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
