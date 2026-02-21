#!/usr/bin/env python
"""Phase B: Smart Predict-then-Optimize (SPO) comparison.

Based on: "Smart Predict-then-Optimize for Portfolio Optimization in Real Markets"
          (Wang Yi & Hasuike, 2026)

Pre-flight: mandatory sanity checks on the two HIGH-priority factors from Phase A.
Then: for each factor, train MSE and SPO-loss LightGBM models with strict
walk-forward, evaluate on held-out test set, and produce comparison report.

Data source: DuckDB daily bars via ``scripts/research/common/data``.
Depends on: Phase A outputs (research_queue.json, factor construction code).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.common.data import ANN_FACTOR, filter_universe
from scripts.research.common.backtest import simple_backtest
from scripts.research.common.metrics import compute_metrics
from scripts.research.paper_strategies.phase_a_decay import (
    _load_from_table,
    compute_factor_signal,
    compute_long_short_returns,
    FACTOR_DEFS,
    MIN_ASSETS_FOR_QUINTILE,
    QUINTILE_PCT,
)

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("artifacts/research/phase_b_spo")
LOG_PATH = Path("artifacts/research/run_log.txt")
QUEUE_PATH = Path("artifacts/research/alpha_decay/research_queue.json")

TRAIN_END = "2022-12-31"
VAL_END = "2024-03-31"
# Test: 2024-04-01 to end of data

WINDOW_90 = 90
FACTORS = ["VOL_LT", "VOL_RL"]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (shared across sanity checks and model training)
# ═══════════════════════════════════════════════════════════════════════════

def load_data(
    db_path: str, table: str, start: str, end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare data, returning (close, returns, volume) wide-format."""
    panel = _load_from_table(db_path, table, start, end)
    panel = filter_universe(panel, min_adv_usd=500_000, min_history_days=365)
    panel = panel[panel["in_universe"]].copy()

    close = panel.pivot(index="ts", columns="symbol", values="close")
    volume = panel.pivot(index="ts", columns="symbol", values="volume")
    returns = close.pct_change()
    return close, returns, volume


def btc_series(close: pd.DataFrame) -> pd.Series:
    """Extract BTC-USD close series."""
    if "BTC-USD" in close.columns:
        return close["BTC-USD"].dropna()
    raise ValueError("BTC-USD not in universe")


# ═══════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def rolling_sharpe_series(ls_returns: pd.Series, window: int = WINDOW_90) -> pd.Series:
    """Compute continuously rolling annualized Sharpe."""
    mu = ls_returns.rolling(window, min_periods=window).mean()
    sigma = ls_returns.rolling(window, min_periods=window).std()
    return (mu / sigma.replace(0, np.nan)) * np.sqrt(ANN_FACTOR)


def sanity_check_1(
    close: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """Explain the last-90-day Sharpe spike."""
    log("Sanity Check 1: Explaining 90-day Sharpe spike")
    results: dict = {}
    btc = btc_series(close)
    btc_ret = btc.pct_change()

    for fname in FACTORS:
        sig = compute_factor_signal(close, returns, volume, fname)
        ls = compute_long_short_returns(sig, returns).dropna()
        rs = rolling_sharpe_series(ls)

        # Plot rolling 90d Sharpe
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rs.index, rs.values, linewidth=1.0, label=f"{fname} rolling 90d Sharpe")
        ax.axhline(2.0, color="r", linestyle="--", alpha=0.5, label="Sharpe=2.0 threshold")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title(f"{fname} — Rolling 90-Day Sharpe")
        ax.set_ylabel("Annualized Sharpe")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / "sanity" / f"rolling_sharpe_{fname}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Find all periods where 90d Sharpe > 2.0
        high_sharpe = rs[rs > 2.0].dropna()
        episodes = []
        if len(high_sharpe) > 0:
            # Group into contiguous episodes
            dates = high_sharpe.index
            groups = []
            current_group = [dates[0]]
            for d in dates[1:]:
                if (d - current_group[-1]).days <= 7:
                    current_group.append(d)
                else:
                    groups.append(current_group)
                    current_group = [d]
            groups.append(current_group)

            for g in groups:
                ep_start, ep_end = g[0], g[-1]
                btc_mask = (btc.index >= ep_start) & (btc.index <= ep_end)
                btc_chunk = btc_ret.reindex(btc.index[btc_mask]).dropna()
                btc_period_ret = float((1 + btc_chunk).prod() - 1) if len(btc_chunk) > 0 else np.nan
                btc_period_vol = float(btc_chunk.std() * np.sqrt(ANN_FACTOR)) if len(btc_chunk) > 5 else np.nan

                btc_21d = btc.pct_change(21)
                mid_btc_21d = btc_21d.reindex([g[len(g)//2]]).iloc[0] if len(g) > 0 else np.nan
                if mid_btc_21d > 0.15:
                    regime = "BULL"
                elif mid_btc_21d < -0.15:
                    regime = "BEAR"
                else:
                    regime = "CHOP"

                episodes.append({
                    "start": str(ep_start.date()),
                    "end": str(ep_end.date()),
                    "peak_sharpe": float(high_sharpe.loc[g].max()),
                    "btc_return": btc_period_ret,
                    "btc_vol": btc_period_vol,
                    "regime": regime,
                })

        current_sharpe = float(rs.iloc[-1]) if len(rs) > 0 else np.nan
        max_historical = float(rs.max()) if len(rs) > 0 else np.nan
        unprecedented = current_sharpe > max_historical * 0.95 and len(episodes) <= 1

        results[fname] = {
            "current_90d_sharpe": current_sharpe,
            "max_historical_90d_sharpe": max_historical,
            "n_episodes_above_2": len(episodes),
            "episodes": episodes,
            "unprecedented": unprecedented,
        }
        log(f"  {fname}: current 90d Sharpe={current_sharpe:.2f}, "
            f"max historical={max_historical:.2f}, "
            f"{'UNPRECEDENTED' if unprecedented else 'PRECEDENT EXISTS'} "
            f"({len(episodes)} episodes above 2.0)")

    # Universe stability check
    n_assets_recent = close.iloc[-90:].notna().sum(axis=1).mean()
    n_assets_historical = close.notna().sum(axis=1).mean()
    universe_shrinkage = n_assets_recent < n_assets_historical * 0.80
    results["universe"] = {
        "n_recent_90d": float(n_assets_recent),
        "n_historical_avg": float(n_assets_historical),
        "shrinkage": universe_shrinkage,
    }
    log(f"  Universe: recent={n_assets_recent:.0f}, historical avg={n_assets_historical:.0f}, "
        f"{'UNIVERSE_SHRINKAGE' if universe_shrinkage else 'stable'}")

    return results


def sanity_check_2(
    close: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame,
) -> dict:
    """Regime sensitivity test."""
    log("Sanity Check 2: Regime sensitivity")
    btc = btc_series(close)
    btc_21d = btc.pct_change(21)

    terciles = btc_21d.quantile([1/3, 2/3])
    t_low, t_high = float(terciles.iloc[0]), float(terciles.iloc[1])

    results: dict = {}
    for fname in FACTORS:
        sig = compute_factor_signal(close, returns, volume, fname)
        ls = compute_long_short_returns(sig, returns).dropna()

        # Align with BTC regime
        common = ls.index.intersection(btc_21d.index)
        ls_a = ls.reindex(common)
        regime = btc_21d.reindex(common)

        bull_mask = regime > t_high
        bear_mask = regime < t_low
        chop_mask = ~bull_mask & ~bear_mask

        def _sharpe(s: pd.Series) -> float:
            s = s.dropna()
            if len(s) < 30:
                return np.nan
            return float((s.mean() / s.std()) * np.sqrt(ANN_FACTOR)) if s.std() > 1e-12 else np.nan

        sharpes = {
            "BULL": _sharpe(ls_a[bull_mask]),
            "BEAR": _sharpe(ls_a[bear_mask]),
            "CHOP": _sharpe(ls_a[chop_mask]),
        }

        regime_dependent = any(v < -0.3 for v in sharpes.values() if np.isfinite(v))
        regime_robust = all(v > 0.2 for v in sharpes.values() if np.isfinite(v))

        results[fname] = {
            "sharpes": sharpes,
            "regime_dependent": regime_dependent,
            "regime_robust": regime_robust,
        }
        flags = []
        if regime_dependent:
            flags.append("REGIME_DEPENDENT")
        if regime_robust:
            flags.append("REGIME_ROBUST")
        flag_str = ", ".join(flags) if flags else "no flags"
        log(f"  {fname}: BULL={sharpes['BULL']:.2f}, BEAR={sharpes['BEAR']:.2f}, "
            f"CHOP={sharpes['CHOP']:.2f} [{flag_str}]")

    return results


def sanity_check_3(
    close: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame,
) -> dict:
    """Top-contributor concentration analysis."""
    log("Sanity Check 3: PnL concentration")
    results: dict = {}

    for fname in FACTORS:
        sig = compute_factor_signal(close, returns, volume, fname)
        # Per-asset contribution to long-short PnL
        sig_shifted = sig.shift(1)
        ret = returns

        common = sig_shifted.index.intersection(ret.index).sort_values()
        sig_a = sig_shifted.reindex(common)
        ret_a = ret.reindex(common)

        # For each asset: sum of (rank_indicator * return) across time
        top_mask = sig_a >= (1.0 - QUINTILE_PCT)
        bottom_mask = sig_a <= QUINTILE_PCT

        long_contrib = (ret_a * top_mask).sum()
        short_contrib = -(ret_a * bottom_mask).sum()
        total_contrib = long_contrib + short_contrib
        total_pnl = total_contrib.sum()

        if abs(total_pnl) < 1e-12:
            results[fname] = {"top5_pct": np.nan, "concentrated": False, "top5_assets": []}
            continue

        sorted_contrib = total_contrib.abs().sort_values(ascending=False)
        top5 = sorted_contrib.head(5)
        top5_pct = float(top5.sum() / sorted_contrib.sum()) if sorted_contrib.sum() > 0 else 0

        # Check if top5 still in recent universe
        recent_syms = set(close.iloc[-90:].dropna(axis=1, how="all").columns)
        top5_current = [s for s in top5.index if s in recent_syms]

        concentrated = top5_pct > 0.50
        diversified = top5_pct < 0.30

        results[fname] = {
            "top5_pct": top5_pct,
            "concentrated": concentrated,
            "diversified": diversified,
            "top5_assets": list(top5.index),
            "top5_still_active": top5_current,
        }
        flag = "CONCENTRATED" if concentrated else ("DIVERSIFIED" if diversified else "moderate")
        log(f"  {fname}: top5 = {top5_pct:.1%} of PnL [{flag}], "
            f"assets: {list(top5.index[:5])}")

    return results


def write_sanity_report(
    sc1: dict, sc2: dict, sc3: dict, out_dir: Path,
) -> tuple[Path, int]:
    """Write sanity_checks.md and return (path, n_flags)."""
    flags: list[str] = []
    lines = ["# Phase B Sanity Checks", ""]

    # Interpretation paragraph
    lines.append("## Executive Interpretation")
    lines.append("")

    interp_parts = []
    for fname in FACTORS:
        s1 = sc1[fname]
        if s1["unprecedented"]:
            interp_parts.append(
                f"{fname}'s last-90-day Sharpe ({s1['current_90d_sharpe']:.2f}) is historically "
                f"unprecedented — no prior 90-day window reached this level."
            )
            flags.append(f"REGIME_ANOMALY ({fname})")
        else:
            interp_parts.append(
                f"{fname}'s last-90-day Sharpe ({s1['current_90d_sharpe']:.2f}) has historical "
                f"precedent ({s1['n_episodes_above_2']} prior episode(s) above 2.0, "
                f"max={s1['max_historical_90d_sharpe']:.2f})."
            )

    uni = sc1["universe"]
    if uni["shrinkage"]:
        interp_parts.append(
            f"Universe has shrunk: {uni['n_recent_90d']:.0f} recent vs "
            f"{uni['n_historical_avg']:.0f} historical average assets."
        )
        flags.append("UNIVERSE_SHRINKAGE")
    else:
        interp_parts.append(
            f"Universe is stable: {uni['n_recent_90d']:.0f} recent vs "
            f"{uni['n_historical_avg']:.0f} historical average."
        )

    for fname in FACTORS:
        s2 = sc2[fname]
        if s2["regime_dependent"]:
            interp_parts.append(f"{fname} is REGIME_DEPENDENT — performance varies significantly by BTC regime.")
            flags.append(f"REGIME_DEPENDENT ({fname})")
        if s2["regime_robust"]:
            interp_parts.append(f"{fname} is REGIME_ROBUST — positive Sharpe across all regimes.")

    for fname in FACTORS:
        s3 = sc3[fname]
        if s3["concentrated"]:
            interp_parts.append(
                f"{fname} PnL is CONCENTRATED — top 5 assets drive {s3['top5_pct']:.0%} "
                f"of returns ({s3['top5_assets'][:3]})."
            )
            flags.append(f"CONCENTRATED ({fname})")

    lines.append(" ".join(interp_parts))
    lines.append("")

    # Check 1 detail
    lines.extend(["## Sanity Check 1: 90-Day Sharpe Spike", ""])
    for fname in FACTORS:
        s = sc1[fname]
        lines.append(f"### {fname}")
        lines.append(f"- Current 90d Sharpe: **{s['current_90d_sharpe']:.2f}**")
        lines.append(f"- Max historical 90d Sharpe: {s['max_historical_90d_sharpe']:.2f}")
        lines.append(f"- Episodes with 90d Sharpe > 2.0: {s['n_episodes_above_2']}")
        if s["episodes"]:
            lines.append("")
            lines.append("| Start | End | Peak Sharpe | BTC Return | BTC Vol | Regime |")
            lines.append("|---|---|---|---|---|---|")
            for ep in s["episodes"]:
                lines.append(
                    f"| {ep['start']} | {ep['end']} | {ep['peak_sharpe']:.2f} | "
                    f"{ep['btc_return']:.1%} | {ep['btc_vol']:.1%} | {ep['regime']} |"
                )
        lines.append(f"- Classification: **{'UNPRECEDENTED' if s['unprecedented'] else 'PRECEDENT EXISTS'}**")
        lines.append("")

    lines.append(f"### Universe Stability")
    lines.append(f"- Recent 90d average assets: {uni['n_recent_90d']:.0f}")
    lines.append(f"- Historical average assets: {uni['n_historical_avg']:.0f}")
    lines.append(f"- Flag: **{'UNIVERSE_SHRINKAGE' if uni['shrinkage'] else 'STABLE'}**")
    lines.append("")

    # Check 2 detail
    lines.extend(["## Sanity Check 2: Regime Sensitivity", ""])
    lines.append("| Factor | BULL | BEAR | CHOP | Flag |")
    lines.append("|---|---|---|---|---|")
    for fname in FACTORS:
        s = sc2[fname]
        sh = s["sharpes"]
        flag_list = []
        if s["regime_dependent"]:
            flag_list.append("REGIME_DEPENDENT")
        if s["regime_robust"]:
            flag_list.append("REGIME_ROBUST")
        lines.append(
            f"| {fname} | {sh['BULL']:.2f} | {sh['BEAR']:.2f} | {sh['CHOP']:.2f} | "
            f"{', '.join(flag_list) or 'none'} |"
        )
    lines.append("")

    # Check 3 detail
    lines.extend(["## Sanity Check 3: PnL Concentration", ""])
    for fname in FACTORS:
        s = sc3[fname]
        lines.append(f"### {fname}")
        lines.append(f"- Top 5 assets: {s['top5_pct']:.1%} of total factor PnL")
        lines.append(f"- Top 5: {s['top5_assets']}")
        lines.append(f"- Still active: {s.get('top5_still_active', [])}")
        flag = "CONCENTRATED" if s["concentrated"] else ("DIVERSIFIED" if s.get("diversified") else "moderate")
        lines.append(f"- Flag: **{flag}**")
        lines.append("")

    report_path = out_dir / "sanity_checks.md"
    report_path.write_text("\n".join(lines))
    return report_path, len(flags)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE CONSTRUCTION (B1)
# ═══════════════════════════════════════════════════════════════════════════

def build_features(
    close: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame,
    regime_dependent_factors: set[str],
) -> pd.DataFrame:
    """Build daily feature matrix. Returns DataFrame indexed by (ts, symbol)."""
    log_returns = np.log(close / close.shift(1))
    features: dict[str, pd.DataFrame] = {}

    # Returns features
    features["ret_1d"] = log_returns
    features["ret_5d"] = np.log(close / close.shift(5))
    features["ret_21d"] = np.log(close / close.shift(21))
    features["ret_63d"] = np.log(close / close.shift(63))
    features["vol_20d"] = log_returns.rolling(20, min_periods=15).std() * np.sqrt(ANN_FACTOR)
    features["vol_5d"] = log_returns.rolling(5, min_periods=4).std() * np.sqrt(ANN_FACTOR)
    vol_20d = features["vol_20d"]
    features["sharpe_21"] = features["ret_21d"] / vol_20d.replace(0, np.nan)

    # Price structure
    features["dist_20d_high"] = close / close.rolling(20).max() - 1.0
    features["dist_20d_low"] = close / close.rolling(20).min() - 1.0
    vol_5d_avg = volume.rolling(5, min_periods=4).mean()
    vol_60d_avg = volume.rolling(60, min_periods=40).mean()
    features["vol_ratio_5_60"] = vol_5d_avg / vol_60d_avg.replace(0, np.nan)

    # Factor score features
    vol_lt_raw = log_returns.rolling(20, min_periods=15).std()
    features["vol_lt_rank"] = vol_lt_raw.rank(axis=1, pct=True, ascending=True)
    features["vol_rl_rank"] = features["vol_ratio_5_60"].rank(axis=1, pct=True, ascending=True)

    # Regime features (broadcast to all assets)
    btc_close = close["BTC-USD"] if "BTC-USD" in close.columns else close.iloc[:, 0]
    btc_log_ret = np.log(btc_close / btc_close.shift(1))
    btc_ret_21d = np.log(btc_close / btc_close.shift(21))
    btc_vol_20d = btc_log_ret.rolling(20, min_periods=15).std() * np.sqrt(ANN_FACTOR)
    btc_vol_zscore = (btc_vol_20d - btc_vol_20d.rolling(252, min_periods=100).mean()) / \
                     btc_vol_20d.rolling(252, min_periods=100).std().replace(0, np.nan)
    mkt_ret_21d = returns.mean(axis=1).rolling(21).sum()

    # Average pairwise 20d correlation
    avg_corr_vals = []
    for i in range(len(close)):
        if i < 20:
            avg_corr_vals.append(np.nan)
            continue
        chunk = returns.iloc[i-20:i].dropna(axis=1, how="all")
        if chunk.shape[1] < 3:
            avg_corr_vals.append(np.nan)
            continue
        c = chunk.corr().values
        mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
        avg_corr_vals.append(float(np.nanmean(c[mask])))
    mkt_avg_corr = pd.Series(avg_corr_vals, index=close.index)

    for col in close.columns:
        features.setdefault("btc_ret_21d", pd.DataFrame(index=close.index, columns=close.columns))
        features["btc_ret_21d"][col] = btc_ret_21d.values
        features.setdefault("btc_vol_20d", pd.DataFrame(index=close.index, columns=close.columns))
        features["btc_vol_20d"][col] = btc_vol_20d.values
        features.setdefault("btc_vol_zscore", pd.DataFrame(index=close.index, columns=close.columns))
        features["btc_vol_zscore"][col] = btc_vol_zscore.values
        features.setdefault("mkt_avg_corr", pd.DataFrame(index=close.index, columns=close.columns))
        features["mkt_avg_corr"][col] = mkt_avg_corr.values
        features.setdefault("mkt_ret_21d", pd.DataFrame(index=close.index, columns=close.columns))
        features["mkt_ret_21d"][col] = mkt_ret_21d.values

    # Interaction terms for regime-dependent factors
    if "VOL_LT" in regime_dependent_factors:
        features["btc_x_vol_lt"] = features["btc_ret_21d"] * features["vol_lt_rank"]
    if "VOL_RL" in regime_dependent_factors:
        features["btc_x_vol_rl"] = features["btc_ret_21d"] * features["vol_rl_rank"]

    # Lag all features by 1 day
    lagged = {name: df.shift(1) for name, df in features.items() if isinstance(df, pd.DataFrame)}

    # Stack to (ts, symbol) multi-index
    frames = []
    for name, df in lagged.items():
        stacked = df.stack(dropna=False)
        stacked.name = name
        frames.append(stacked)

    feature_df = pd.concat(frames, axis=1)
    feature_df.index.names = ["ts", "symbol"]

    # Forward-fill up to 5 days within each symbol, then drop NaN
    feature_df = feature_df.groupby(level="symbol").ffill(limit=5)
    feature_df = feature_df.dropna()

    return feature_df


# ═══════════════════════════════════════════════════════════════════════════
# TARGET CONSTRUCTION (B2)
# ═══════════════════════════════════════════════════════════════════════════

def build_factor_target(
    close: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame,
    factor_name: str,
) -> pd.Series:
    """Compute daily long-short return series for a factor."""
    sig = compute_factor_signal(close, returns, volume, factor_name)
    ls = compute_long_short_returns(sig, returns)
    return ls.dropna()


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (B3 / B4)
# ═══════════════════════════════════════════════════════════════════════════

def _try_lgbm():
    try:
        import lightgbm as lgb
        return lgb, True
    except ImportError:
        return None, False


def _aggregate_features_daily(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate (ts, symbol) features to daily means for portfolio-level prediction."""
    return feature_df.groupby(level="ts").mean()


def train_model(
    daily_X: pd.DataFrame,
    target: pd.Series,
    train_end: str,
    val_end: str,
    loss_mode: str = "mse",
) -> dict:
    """Train LightGBM (or sklearn fallback) with walk-forward."""
    lgb, has_lgb = _try_lgbm()

    common = daily_X.index.intersection(target.index).sort_values()
    X = daily_X.reindex(common)
    y = target.reindex(common)

    train_mask = X.index <= pd.Timestamp(train_end)
    val_mask = (X.index > pd.Timestamp(train_end)) & (X.index <= pd.Timestamp(val_end))
    test_mask = X.index > pd.Timestamp(val_end)

    X_train, y_train = X[train_mask].values, y[train_mask].values
    X_val, y_val = X[val_mask].values, y[val_mask].values
    X_test, y_test = X[test_mask].values, y[test_mask].values
    test_dates = X.index[test_mask]
    train_dates = X.index[train_mask]

    # Drop NaN rows
    for arr_name in ["train", "val", "test"]:
        _X = locals()[f"X_{arr_name}"]
        _y = locals()[f"y_{arr_name}"]
        mask = ~(np.isnan(_X).any(axis=1) | np.isnan(_y))
        if arr_name == "train":
            X_train, y_train = _X[mask], _y[mask]
        elif arr_name == "val":
            X_val, y_val = _X[mask], _y[mask]
        else:
            test_keep = mask
            X_test, y_test = _X[mask], _y[mask]
            test_dates = test_dates[mask]

    if len(X_train) < 100 or len(X_test) < 50:
        return {"error": f"Insufficient data: train={len(X_train)}, test={len(X_test)}"}

    feature_names = list(daily_X.columns)

    if has_lgb:
        # Hyperparameter random search on validation
        best_val_loss = np.inf
        best_params = {}
        rng = np.random.RandomState(SEED)

        param_space = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
            "min_child_samples": [20, 40, 60, 80, 100],
        }

        for trial in range(50):
            params = {k: rng.choice(v) for k, v in param_space.items()}
            lgb_params = {
                "objective": "regression",
                "metric": "mse",
                "num_leaves": 2 ** int(params["max_depth"]) - 1,
                "learning_rate": params["learning_rate"],
                "min_child_samples": int(params["min_child_samples"]),
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": SEED,
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

            feval_fn = None
            if loss_mode == "spo":
                lgb_params["objective"] = _spo_loss_approx
                lgb_params["metric"] = "None"
                feval_fn = _mse_eval

            callbacks = [lgb.early_stopping(50, verbose=False)]
            model = lgb.train(
                lgb_params, train_data,
                num_boost_round=int(params["n_estimators"]),
                valid_sets=[val_data],
                feval=feval_fn,
                callbacks=callbacks,
            )

            val_pred = model.predict(X_val)
            val_loss = float(np.mean((val_pred - y_val) ** 2))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_model = model

        # Predictions
        y_pred_test = best_model.predict(X_test)
        y_pred_val = best_model.predict(X_val)
        y_pred_train = best_model.predict(X_train)
        importances = dict(zip(feature_names, best_model.feature_importance(importance_type="gain").tolist()))
        backend = "lightgbm"
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        if loss_mode == "spo":
            log("  sklearn fallback — SPO custom objective not available, using MSE")

        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=SEED,
        )
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_val = model.predict(X_val)
        y_pred_train = model.predict(X_train)
        importances = dict(zip(feature_names, model.feature_importances_.tolist()))
        best_model = model
        best_params = {"n_estimators": 300, "max_depth": 5}
        best_val_loss = float(np.mean((y_pred_val - y_val) ** 2))
        backend = "sklearn"

    # In-sample Sharpe (for overfit check)
    train_signal = np.sign(y_pred_train)
    train_strat_ret = train_signal * y_train
    is_sharpe = float(
        (np.mean(train_strat_ret) / np.std(train_strat_ret)) * np.sqrt(ANN_FACTOR)
    ) if np.std(train_strat_ret) > 1e-12 else np.nan

    return {
        "model": best_model,
        "backend": backend,
        "loss_mode": loss_mode,
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in best_params.items()},
        "best_val_mse": best_val_loss,
        "importances": importances,
        "y_pred_test": y_pred_test,
        "y_pred_val": y_pred_val,
        "y_test": y_test,
        "test_dates": test_dates,
        "is_sharpe": is_sharpe,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "feature_names": feature_names,
    }


def _spo_loss_approx(y_pred, y_true_lgbm):
    """SPO approximation: quintile-weighted MSE."""
    y_true = y_true_lgbm.get_label()
    pred_ranks = pd.Series(y_pred).rank(pct=True).values
    weights = 1.0 + 3.0 * np.abs(pred_ranks - 0.5) * 2.0
    residuals = y_pred - y_true
    gradient = 2.0 * weights * residuals
    hessian = 2.0 * weights
    return gradient, hessian


def _mse_eval(y_pred, y_true_lgbm):
    """Custom eval metric (MSE) for early stopping with custom objective."""
    y_true = y_true_lgbm.get_label()
    mse = float(np.mean((y_pred - y_true) ** 2))
    return "mse", mse, False


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION (B6)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    result: dict,
    btc_close: pd.Series,
    factor_name: str,
    label: str,
) -> dict:
    """Convert predictions to portfolio returns and compute metrics."""
    if "error" in result:
        return {"error": result["error"]}

    y_pred = result["y_pred_test"]
    y_test = result["y_test"]
    dates = result["test_dates"]

    # Strategy: go long when predicted factor return > 0, else flat/short
    signal = np.sign(y_pred)
    strat_ret = pd.Series(signal * y_test, index=dates)
    equity = (1 + strat_ret).cumprod()
    metrics = compute_metrics(equity)

    # Regime-conditional Sharpe
    btc_21d = np.log(btc_close / btc_close.shift(21))
    btc_aligned = btc_21d.reindex(dates)
    terciles = btc_21d.quantile([1/3, 2/3])
    t_low, t_high = float(terciles.iloc[0]), float(terciles.iloc[1])

    def _sharpe(s):
        s = s.dropna()
        return float((s.mean() / s.std()) * np.sqrt(ANN_FACTOR)) if len(s) > 20 and s.std() > 1e-12 else np.nan

    regime_sharpes = {
        "BULL": _sharpe(strat_ret[btc_aligned > t_high]),
        "BEAR": _sharpe(strat_ret[btc_aligned < t_low]),
        "CHOP": _sharpe(strat_ret[(btc_aligned >= t_low) & (btc_aligned <= t_high)]),
    }

    # Rolling 90d Sharpe on test set
    roll_sharpe = rolling_sharpe_series(strat_ret)

    return {
        "label": label,
        "factor": factor_name,
        "strat_ret": strat_ret,
        "equity": equity,
        "metrics": metrics,
        "regime_sharpes": regime_sharpes,
        "rolling_sharpe": roll_sharpe,
        "is_sharpe": result.get("is_sharpe", np.nan),
        "n_test": result["n_test"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_equity_comparison(
    raw_eq: pd.Series, mse_eval: dict, spo_eval: dict,
    factor_name: str, out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    if raw_eq is not None and len(raw_eq) > 0:
        ax.plot(raw_eq.index, raw_eq.values, label="Raw Factor", linewidth=1.0, alpha=0.7)
    if "equity" in mse_eval:
        eq = mse_eval["equity"]
        ax.plot(eq.index, eq.values, label="MSE Model", linewidth=1.2)
    if "equity" in spo_eval:
        eq = spo_eval["equity"]
        ax.plot(eq.index, eq.values, label="SPO Model", linewidth=1.2, linestyle="--")
    ax.set_title(f"{factor_name} — Test Period Equity Curves")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "plots" / f"equity_curves_{factor_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(importances: dict, factor_name: str, label: str, out_dir: Path) -> Path:
    top = sorted(importances.items(), key=lambda x: -x[1])[:15]
    names, vals = zip(*top) if top else ([], [])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), vals, align="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title(f"{factor_name} ({label}) — Feature Importance (gain)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = out_dir / "plots" / f"feature_importance_{factor_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_monthly_heatmap(strat_ret: pd.Series, factor_name: str, label: str, out_dir: Path) -> Path:
    monthly = strat_ret.resample("ME").sum()
    df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    pivot = df.pivot(index="year", columns="month", values="ret")
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"M{m}" for m in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{factor_name} ({label}) — Monthly Returns")
    fig.colorbar(im, ax=ax, label="Return")
    plt.tight_layout()
    path = out_dir / "plots" / f"monthly_heatmap_{factor_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# REPORT WRITING (B7 + output)
# ═══════════════════════════════════════════════════════════════════════════

def write_model_report(
    all_results: dict, sc_summary: str, n_flags: int, out_dir: Path,
) -> Path:
    lines = ["# Phase B: SPO Comparison Report", ""]

    # Summary
    lines.append("## Phase B Summary")
    lines.append("")

    # Paste sanity check key findings
    lines.append("## Sanity Check Outcomes")
    lines.append(sc_summary)
    lines.append("")

    for fname in FACTORS:
        r = all_results[fname]
        mse = r["mse_eval"]
        spo = r["spo_eval"]
        raw = r["raw_metrics"]
        pred_corr = r.get("pred_correlation", np.nan)

        mse_m = mse.get("metrics", {})
        spo_m = spo.get("metrics", {})

        lines.extend(["---", f"## Results: {fname}", ""])

        # Raw factor baseline
        lines.append(f"**Raw factor (Phase A baseline):**")
        lines.append(f"- Full-period Sharpe: {r.get('raw_full_sharpe', np.nan):.3f}")
        lines.append(f"- Test-period Sharpe: {raw.get('sharpe', np.nan):.3f}")
        lines.append("")

        # MSE
        mse_sharpe = mse_m.get("sharpe", np.nan)
        mse_rs = mse.get("regime_sharpes", {})
        is_sharpe_mse = mse.get("is_sharpe", np.nan)
        overfit_mse = is_sharpe_mse > 2 * mse_sharpe if np.isfinite(is_sharpe_mse) and np.isfinite(mse_sharpe) and mse_sharpe > 0 else False

        lines.append("**MSE Model:**")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Test Sharpe | {mse_sharpe:.3f} |")
        lines.append(f"| CAGR | {mse_m.get('cagr', np.nan):.1%} |")
        lines.append(f"| Max DD | {mse_m.get('max_dd', np.nan):.1%} |")
        lines.append(f"| Calmar | {mse_m.get('calmar', np.nan):.2f} |")
        lines.append(f"| Sortino | {mse_m.get('sortino', np.nan):.2f} |")
        lines.append(f"| Win Rate | {mse_m.get('hit_rate', np.nan):.1%} |")
        lines.append(f"| Regime — BULL | {mse_rs.get('BULL', np.nan):.2f} |")
        lines.append(f"| Regime — BEAR | {mse_rs.get('BEAR', np.nan):.2f} |")
        lines.append(f"| Regime — CHOP | {mse_rs.get('CHOP', np.nan):.2f} |")
        lines.append(f"| IS/OOS Sharpe ratio | {is_sharpe_mse:.2f} / {mse_sharpe:.2f} = {is_sharpe_mse/mse_sharpe:.1f}x |" if mse_sharpe > 0.01 else f"| IS Sharpe | {is_sharpe_mse:.2f} |")
        if overfit_mse:
            lines.append(f"| **LIKELY_OVERFIT** | IS {is_sharpe_mse:.2f} > 2x OOS {mse_sharpe:.2f} |")
        lines.append("")

        # SPO
        spo_sharpe = spo_m.get("sharpe", np.nan)
        spo_rs = spo.get("regime_sharpes", {})
        improvement = ((spo_sharpe - mse_sharpe) / abs(mse_sharpe) * 100) if abs(mse_sharpe) > 0.01 else np.nan
        if np.isfinite(improvement):
            imp_str = f"{improvement:+.1f}% Sharpe improvement" if improvement > 0 else "DEGRADED"
        else:
            imp_str = "N/A"

        null_flag = pred_corr > 0.98 if np.isfinite(pred_corr) else False

        lines.append("**SPO Model:**")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Test Sharpe | {spo_sharpe:.3f} |")
        lines.append(f"| CAGR | {spo_m.get('cagr', np.nan):.1%} |")
        lines.append(f"| Max DD | {spo_m.get('max_dd', np.nan):.1%} |")
        lines.append(f"| Calmar | {spo_m.get('calmar', np.nan):.2f} |")
        lines.append(f"| Sortino | {spo_m.get('sortino', np.nan):.2f} |")
        lines.append(f"| Win Rate | {spo_m.get('hit_rate', np.nan):.1%} |")
        lines.append(f"| Regime — BULL | {spo_rs.get('BULL', np.nan):.2f} |")
        lines.append(f"| Regime — BEAR | {spo_rs.get('BEAR', np.nan):.2f} |")
        lines.append(f"| Regime — CHOP | {spo_rs.get('CHOP', np.nan):.2f} |")
        lines.append(f"| Pred correlation with MSE | {pred_corr:.3f} {'**NULL — models identical**' if null_flag else ''} |")
        lines.append(f"| **SPO improvement** | **{imp_str}** |")
        lines.append("")

        # Verdict
        best = "RAW FACTOR"
        best_sharpe = raw.get("sharpe", 0)
        if mse_sharpe > best_sharpe + 0.05:
            best = "MSE MODEL"
            best_sharpe = mse_sharpe
        if spo_sharpe > best_sharpe + 0.05:
            best = "SPO MODEL"

        lines.append(f"**VERDICT {fname}:** {best} recommended")
        lines.append("")

        # Feature importance
        best_imp = r.get("best_importances", {})
        top10 = sorted(best_imp.items(), key=lambda x: -x[1])[:10]
        if top10:
            lines.append(f"Top 10 features ({best}):")
            for rank, (feat, val) in enumerate(top10, 1):
                lines.append(f"  {rank}. {feat} ({val:.1f})")
            btc_rank = next((i+1 for i, (f, _) in enumerate(top10) if "btc_ret_21d" in f), None)
            if btc_rank:
                lines.append(f"  -> btc_ret_21d at rank #{btc_rank}: model has learned regime-conditioning")
            interaction_rank = next((i+1 for i, (f, _) in enumerate(top10) if "btc_x_" in f), None)
            if interaction_rank:
                lines.append(f"  -> interaction term at rank #{interaction_rank}: factor is regime-dependent")
        lines.append("")

    # Cross-factor correlation
    if all(fname in all_results and "strat_ret" in all_results[fname].get("best_eval", {}) for fname in FACTORS):
        corr_val = all_results[FACTORS[0]]["best_eval"]["strat_ret"].corr(
            all_results[FACTORS[1]]["best_eval"]["strat_ret"]
        )
        lines.append("## Cross-Factor Correlation")
        lines.append(f"Daily return correlation (best VOL_LT vs best VOL_RL): **{corr_val:.3f}**")
        if corr_val > 0.7:
            lines.append("Factors are highly correlated — limited diversification benefit.")
        elif corr_val < 0.5:
            lines.append("Factors are sufficiently uncorrelated to combine.")
        lines.append("")

    # Null result checks
    lines.append("## Null Result Protocol")
    best_sharpes = []
    for fname in FACTORS:
        r = all_results[fname]
        for key in ["mse_eval", "spo_eval"]:
            m = r[key].get("metrics", {})
            best_sharpes.append(m.get("sharpe", 0))
    if max(best_sharpes) < 0.3:
        lines.append("**TEST A — NULL RESULT:** No model exceeded Sharpe 0.3. "
                     "Raw factors from Phase A are the better strategy.")
    else:
        lines.append(f"TEST A: Best model Sharpe = {max(best_sharpes):.3f} — edge may be real.")
    lines.append("")

    report_path = out_dir / "model_comparison.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B: SPO Comparison")
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--table", type=str, default="bars_1d_usd_universe_clean")
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default="2025-12-15")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "sanity").mkdir(exist_ok=True)
    (out_dir / "best_models").mkdir(exist_ok=True)

    # Load data
    log(f"PHASE B STARTING — factors: {FACTORS}, data: {args.start} to {args.end}")
    close, returns, volume = load_data(args.db, args.table, args.start, args.end)
    n_assets = close.shape[1]
    log(f"Phase B started. Factors: {FACTORS}. Assets: {n_assets}. "
        f"Test window: 2024-04-01 to {close.index.max().date()}.")

    # ── SANITY CHECKS ──
    sc1 = sanity_check_1(close, returns, volume, out_dir)
    sc2 = sanity_check_2(close, returns, volume)
    sc3 = sanity_check_3(close, returns, volume)
    sc_path, n_flags = write_sanity_report(sc1, sc2, sc3, out_dir)

    # Build summary for embedding in final report
    sc_summary_lines = []
    for fname in FACTORS:
        s1 = sc1[fname]
        s2 = sc2[fname]
        s3 = sc3[fname]
        sc_summary_lines.append(
            f"- {fname}: 90d Sharpe={'UNPRECEDENTED' if s1['unprecedented'] else 'precedent exists'}, "
            f"regime={'REGIME_DEPENDENT' if s2['regime_dependent'] else 'robust'}, "
            f"concentration={s3['top5_pct']:.0%} ({'CONCENTRATED' if s3['concentrated'] else 'ok'})"
        )
    sc_summary = "\n".join(sc_summary_lines)

    if n_flags > 0:
        log(f"SANITY CHECKS FLAGGED {n_flags} issues — see sanity_checks.md. "
            f"Proceeding with elevated skepticism.")
    else:
        log("SANITY CHECKS COMPLETE — proceeding to Phase B model training")

    # Determine regime-dependent factors for interaction features
    regime_dep = {f for f in FACTORS if sc2[f]["regime_dependent"]}
    log(f"Sanity check 2: VOL_LT regime flags: "
        f"{'REGIME_DEPENDENT' if 'VOL_LT' in regime_dep else 'robust'}. "
        f"VOL_RL regime flags: {'REGIME_DEPENDENT' if 'VOL_RL' in regime_dep else 'robust'}")

    # ── FEATURE CONSTRUCTION (B1) ──
    log("B1. Building features")
    feature_df = build_features(close, returns, volume, regime_dep)
    daily_X = _aggregate_features_daily(feature_df)
    log(f"  Features: {daily_X.shape[1]} columns, {daily_X.shape[0]} days")

    # ── FACTOR LOOP ──
    btc_close = close["BTC-USD"] if "BTC-USD" in close.columns else close.iloc[:, 0]
    all_results: dict = {}

    for fname in FACTORS:
        log(f"B2. Building target: {fname}")
        target = build_factor_target(close, returns, volume, fname)

        # Raw factor test-period metrics
        test_mask = target.index > pd.Timestamp(VAL_END)
        raw_test = target[test_mask]
        raw_eq = (1 + raw_test).cumprod()
        raw_metrics = compute_metrics(raw_eq)

        # Full-period raw Sharpe
        raw_full_eq = (1 + target.dropna()).cumprod()
        raw_full_metrics = compute_metrics(raw_full_eq)

        # B3: MSE model
        log(f"B3. Training MSE model: {fname}")
        mse_result = train_model(daily_X, target, TRAIN_END, VAL_END, loss_mode="mse")
        if "error" in mse_result:
            log(f"  MSE ERROR: {mse_result['error']}")

        # B4: SPO model
        log(f"B4. Training SPO model: {fname}")
        spo_result = train_model(daily_X, target, TRAIN_END, VAL_END, loss_mode="spo")
        if "error" in spo_result:
            log(f"  SPO ERROR: {spo_result['error']}")

        # Prediction correlation check (B5 critical check)
        pred_corr = np.nan
        if "y_pred_val" in mse_result and "y_pred_val" in spo_result:
            pc = np.corrcoef(mse_result["y_pred_val"], spo_result["y_pred_val"])[0, 1]
            pred_corr = float(pc) if np.isfinite(pc) else np.nan
            if pred_corr > 0.98:
                log(f"  SPO correlation check: {fname}={pred_corr:.3f} — NULL (models identical)")
            else:
                log(f"  SPO correlation check: {fname}={pred_corr:.3f} — DIFFERENTIATED")

        # B6: Evaluate
        log(f"B6. Evaluating: {fname}")
        mse_eval = evaluate_model(mse_result, btc_close, fname, "MSE")
        spo_eval = evaluate_model(spo_result, btc_close, fname, "SPO")

        mse_sharpe = mse_eval.get("metrics", {}).get("sharpe", -99)
        spo_sharpe = spo_eval.get("metrics", {}).get("sharpe", -99)
        log(f"  MSE Sharpe={mse_sharpe:.3f}, SPO Sharpe={spo_sharpe:.3f}, Raw={raw_metrics.get('sharpe', np.nan):.3f}")

        # Pick best model
        best_label = "MSE" if mse_sharpe >= spo_sharpe else "SPO"
        best_eval = mse_eval if mse_sharpe >= spo_sharpe else spo_eval
        best_result = mse_result if mse_sharpe >= spo_sharpe else spo_result
        best_imp = best_result.get("importances", {}) if "error" not in best_result else {}

        # Check if raw factor is actually better
        raw_sharpe = raw_metrics.get("sharpe", -99)
        if raw_sharpe > max(mse_sharpe, spo_sharpe) + 0.05:
            best_label = "RAW"
        log(f"  Best model per factor: {fname}={best_label}")

        # Save predictions
        if "y_pred_test" in mse_result:
            pred_df = pd.DataFrame({
                "date": mse_result["test_dates"],
                "y_true": mse_result["y_test"],
                "mse_pred": mse_result["y_pred_test"],
            })
            if "y_pred_test" in spo_result:
                pred_df["spo_pred"] = spo_result["y_pred_test"]
            pred_df.to_csv(out_dir / "best_models" / f"{fname}_test_predictions.csv", index=False)

        # Save best model
        if "model" in best_result:
            with open(out_dir / "best_models" / f"{fname}_best_model.pkl", "wb") as f:
                pickle.dump(best_result["model"], f)

        # Plots
        plot_equity_comparison(raw_eq, mse_eval, spo_eval, fname, out_dir)
        if best_imp:
            plot_feature_importance(best_imp, fname, best_label, out_dir)
        if "strat_ret" in best_eval:
            plot_monthly_heatmap(best_eval["strat_ret"], fname, best_label, out_dir)

        all_results[fname] = {
            "mse_eval": mse_eval,
            "spo_eval": spo_eval,
            "best_eval": best_eval,
            "raw_metrics": raw_metrics,
            "raw_full_sharpe": raw_full_metrics.get("sharpe", np.nan),
            "pred_correlation": pred_corr,
            "best_importances": best_imp,
        }

    # ── REPORT ──
    log("B7. Writing reports")
    report_path = write_model_report(all_results, sc_summary, n_flags, out_dir)

    log(f"PHASE B COMPLETE — outputs written to {out_dir}")


if __name__ == "__main__":
    main()
