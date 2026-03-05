#!/usr/bin/env python
"""Phase C: Correlation-aware portfolio construction.

Based on: "Forecasting Equity Correlations with Hybrid Transformer GNN"
          (Cheng & Zhu, 2025)

Honest simplification: shrinkage-based correlation forecast for a
two-factor portfolio (VOL_LT + VOL_RL).  The full Transformer-GNN is not
appropriate at this scale; the shrinkage approach captures the core insight
— using a forecast correlation rather than a naive historical estimate —
without overfitting.

Depends on: Phase A (factor construction) and Phase B (regime analysis).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, date as dt_date
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
from scripts.research.common.metrics import compute_metrics
from scripts.research.paper_strategies.phase_a_decay import (
    _load_from_table,
    compute_factor_signal,
    compute_long_short_returns,
    FACTOR_DEFS,
)

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("artifacts/research/phase_c_correlation")
LOG_PATH = Path("artifacts/research/run_log.txt")

TEST_START = "2024-04-01"
FACTORS = ["VOL_LT", "VOL_RL"]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(db_path: str, table: str, start: str, end: str):
    panel = _load_from_table(db_path, table, start, end)
    panel = filter_universe(panel, min_adv_usd=500_000, min_history_days=365)
    panel = panel[panel["in_universe"]].copy()
    close = panel.pivot(index="ts", columns="symbol", values="close")
    volume = panel.pivot(index="ts", columns="symbol", values="volume")
    returns = close.pct_change()
    return close, returns, volume


def build_factor_returns(close, returns, volume, factor_name):
    sig = compute_factor_signal(close, returns, volume, factor_name)
    ls = compute_long_short_returns(sig, returns)
    return ls.dropna()


def btc_series(close):
    return close["BTC-USD"].dropna()


def regime_series(close):
    """BTC 21d return tercile classification."""
    btc = btc_series(close)
    btc_21d = np.log(btc / btc.shift(21))
    t = btc_21d.quantile([1 / 3, 2 / 3])
    t_low, t_high = float(t.iloc[0]), float(t.iloc[1])
    regime = pd.Series("CHOP", index=btc_21d.index)
    regime[btc_21d > t_high] = "BULL"
    regime[btc_21d < t_low] = "BEAR"
    return regime, t_low, t_high


# ═══════════════════════════════════════════════════════════════════════════
# PRE-PHASE C: REGIME-CONDITIONAL CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def regime_correlation_analysis(
    vol_lt: pd.Series, vol_rl: pd.Series,
    close: pd.DataFrame, out_dir: Path,
) -> dict:
    """RC1–RC3: full regime-conditional correlation analysis."""
    log("RC1. Unconditional correlation baseline")
    common = vol_lt.index.intersection(vol_rl.index).sort_values()
    lt = vol_lt.reindex(common)
    rl = vol_rl.reindex(common)

    uncond_corr = float(lt.corr(rl))
    log(f"  Unconditional correlation: {uncond_corr:.3f}")

    # Rolling 60d correlation
    rolling_corr = lt.rolling(60, min_periods=40).corr(rl)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_corr.index, rolling_corr.values, linewidth=0.8)
    ax.axhline(uncond_corr, color="k", linestyle="--", alpha=0.5, label=f"Unconditional ({uncond_corr:.3f})")
    ax.axhline(-0.3, color="r", linestyle=":", alpha=0.5, label="Hedge threshold (-0.3)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title("VOL_LT vs VOL_RL — Rolling 60-Day Correlation")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdir / "rolling_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Periods where rolling corr < -0.3
    hedge_periods = rolling_corr[rolling_corr < -0.3].dropna()
    log(f"  Periods with corr < -0.3: {len(hedge_periods)} days")

    # ── RC2: Regime-conditional correlation ──
    log("RC2. Regime-conditional correlation")
    regime, _, _ = regime_series(close)
    regime_aligned = regime.reindex(common)
    regime_corrs = {}
    for r_label in ["BULL", "BEAR", "CHOP"]:
        mask = regime_aligned == r_label
        if mask.sum() > 30:
            regime_corrs[r_label] = float(lt[mask].corr(rl[mask]))
        else:
            regime_corrs[r_label] = np.nan

    negative_hedge = any(v < -0.2 for v in regime_corrs.values() if np.isfinite(v))
    all_close = all(abs(v - uncond_corr) < 0.15 for v in regime_corrs.values() if np.isfinite(v))

    log(f"  BULL={regime_corrs.get('BULL', np.nan):.3f}, "
        f"BEAR={regime_corrs.get('BEAR', np.nan):.3f}, "
        f"CHOP={regime_corrs.get('CHOP', np.nan):.3f}")
    if negative_hedge:
        log("  Flag: NEGATIVE_HEDGE")
    if all_close:
        log("  Flag: REGIME_INVARIANT")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    regimes = list(regime_corrs.keys())
    vals = [regime_corrs[r] for r in regimes]
    colors = ["#2ecc71" if v < -0.2 else ("#e74c3c" if v > 0.3 else "#3498db") for v in vals]
    ax.bar(regimes, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(uncond_corr, color="k", linestyle="--", label=f"Unconditional ({uncond_corr:.3f})")
    ax.set_title("VOL_LT vs VOL_RL Correlation by Regime")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(pdir / "regime_correlation_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── RC3: Correlation dynamics ──
    log("RC3. Correlation dynamics analysis")
    acf_1 = float(rolling_corr.autocorr(lag=1))
    acf_5 = float(rolling_corr.autocorr(lag=5))
    acf_20 = float(rolling_corr.autocorr(lag=20))
    forecastable = acf_5 > 0.7
    noisy = acf_5 < 0.3

    log(f"  Autocorrelation: lag-1={acf_1:.3f}, lag-5={acf_5:.3f}, lag-20={acf_20:.3f}")
    if forecastable:
        log("  Flag: CORRELATION_FORECASTABLE")
    elif noisy:
        log("  Flag: CORRELATION_NOISY")
    else:
        log("  Correlation persistence: moderate")

    # Regime transition analysis
    regime_changes = regime_aligned[regime_aligned != regime_aligned.shift(1)].dropna()
    transition_delays = []
    for i in range(len(regime_changes)):
        change_date = regime_changes.index[i]
        new_regime = regime_changes.iloc[i]
        prior_corr = regime_corrs.get(new_regime, uncond_corr)
        if not np.isfinite(prior_corr):
            continue
        post_corrs = rolling_corr.loc[change_date:].iloc[:60]
        stabilized = post_corrs[(post_corrs - prior_corr).abs() < 0.10]
        if len(stabilized) > 0:
            days_to_stable = (stabilized.index[0] - change_date).days
            transition_delays.append(days_to_stable)

    avg_transition = float(np.mean(transition_delays)) if transition_delays else np.nan
    log(f"  Average days to correlation stabilization after regime change: {avg_transition:.0f}")

    results = {
        "unconditional_corr": uncond_corr,
        "regime_corrs": regime_corrs,
        "negative_hedge": negative_hedge,
        "regime_invariant": all_close,
        "acf_1": acf_1,
        "acf_5": acf_5,
        "acf_20": acf_20,
        "forecastable": forecastable,
        "noisy": noisy,
        "avg_transition_days": avg_transition,
        "n_hedge_days": len(hedge_periods),
        "rolling_corr": rolling_corr,
    }
    return results


def write_regime_correlation_report(rc: dict, out_dir: Path) -> Path:
    lines = [
        "# Phase C: Regime-Conditional Correlation Analysis",
        "",
        "## RC1: Unconditional Correlation",
        f"- Unconditional correlation (full history): **{rc['unconditional_corr']:.3f}**",
        f"- Days with 60d rolling corr < -0.3: {rc['n_hedge_days']}",
        "",
        "## RC2: Regime-Conditional Correlation",
        f"- BULL: **{rc['regime_corrs'].get('BULL', np.nan):.3f}**",
        f"- BEAR: **{rc['regime_corrs'].get('BEAR', np.nan):.3f}**",
        f"- CHOP: **{rc['regime_corrs'].get('CHOP', np.nan):.3f}**",
        "",
    ]
    flags = []
    if rc["negative_hedge"]:
        flags.append("NEGATIVE_HEDGE")
    if rc["regime_invariant"]:
        flags.append("REGIME_INVARIANT")
    lines.append(f"Flags: **{', '.join(flags) if flags else 'none'}**")
    lines.append("")

    lines.extend([
        "## RC3: Correlation Dynamics",
        f"- Autocorrelation lag-1: {rc['acf_1']:.3f}",
        f"- Autocorrelation lag-5: {rc['acf_5']:.3f}",
        f"- Autocorrelation lag-20: {rc['acf_20']:.3f}",
        f"- Average days to stabilization after regime change: {rc['avg_transition_days']:.0f}",
        "",
    ])
    if rc["forecastable"]:
        lines.append("Flag: **CORRELATION_FORECASTABLE** (lag-5 ACF > 0.7)")
    elif rc["noisy"]:
        lines.append("Flag: **CORRELATION_NOISY** (lag-5 ACF < 0.3)")
    else:
        lines.append(f"Correlation persistence: moderate (lag-5 ACF = {rc['acf_5']:.3f})")
    lines.append("")

    path = out_dir / "regime_correlation.md"
    path.write_text("\n".join(lines))
    return path


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION METHODS (C1)
# ═══════════════════════════════════════════════════════════════════════════

def method1_equal_weight(vol_lt: pd.Series, vol_rl: pd.Series) -> pd.Series:
    """M1: 50/50 fixed allocation, monthly rebalance to 50/50."""
    common = vol_lt.index.intersection(vol_rl.index).sort_values()
    lt = vol_lt.reindex(common).fillna(0)
    rl = vol_rl.reindex(common).fillna(0)
    port_ret = 0.5 * lt + 0.5 * rl
    return port_ret


def method2_vol_parity(vol_lt: pd.Series, vol_rl: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """M2: Inverse-volatility weighting, weekly rebalance."""
    common = vol_lt.index.intersection(vol_rl.index).sort_values()
    lt = vol_lt.reindex(common).fillna(0)
    rl = vol_rl.reindex(common).fillna(0)

    vol_lt_20d = lt.rolling(20, min_periods=10).std()
    vol_rl_20d = rl.rolling(20, min_periods=10).std()

    inv_vol_lt = 1.0 / vol_lt_20d.replace(0, np.nan)
    inv_vol_rl = 1.0 / vol_rl_20d.replace(0, np.nan)
    total_inv = inv_vol_lt + inv_vol_rl
    w_lt = (inv_vol_lt / total_inv).fillna(0.5)
    w_rl = (inv_vol_rl / total_inv).fillna(0.5)

    # Weekly rebalance: hold weights constant within each week
    w_lt_weekly = w_lt.resample("W-FRI").last().reindex(common, method="ffill").fillna(0.5)
    w_rl_weekly = w_rl.resample("W-FRI").last().reindex(common, method="ffill").fillna(0.5)

    port_ret = w_lt_weekly.shift(1).fillna(0.5) * lt + w_rl_weekly.shift(1).fillna(0.5) * rl
    weights = pd.DataFrame({"w_lt": w_lt_weekly, "w_rl": w_rl_weekly}, index=common)
    return port_ret, weights


def method3_regime_switching(
    vol_lt: pd.Series, vol_rl: pd.Series, close: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """M3: Regime-based allocation with 5-day persistence filter."""
    common = vol_lt.index.intersection(vol_rl.index).sort_values()
    lt = vol_lt.reindex(common).fillna(0)
    rl = vol_rl.reindex(common).fillna(0)

    regime_raw, _, _ = regime_series(close)
    regime_raw = regime_raw.reindex(common)

    # 5-day persistence filter: regime must hold for 5 consecutive days
    stable_regime = pd.Series("CHOP", index=common)
    current = "CHOP"
    count = 0
    for i, d in enumerate(common):
        r_today = regime_raw.iloc[i] if i < len(regime_raw) else "CHOP"
        if r_today == current:
            count += 1
        else:
            count = 1
            current = r_today
        if count >= 5:
            stable_regime.iloc[i] = current
        else:
            stable_regime.iloc[i] = stable_regime.iloc[i - 1] if i > 0 else "CHOP"

    alloc = {"BULL": (0.30, 0.70), "BEAR": (0.80, 0.20), "CHOP": (0.55, 0.45)}
    w_lt = stable_regime.map(lambda r: alloc.get(r, (0.55, 0.45))[0]).astype(float)
    w_rl = stable_regime.map(lambda r: alloc.get(r, (0.55, 0.45))[1]).astype(float)

    port_ret = w_lt.shift(1).fillna(0.55) * lt + w_rl.shift(1).fillna(0.45) * rl
    weights = pd.DataFrame({"w_lt": w_lt, "w_rl": w_rl, "regime": stable_regime}, index=common)
    return port_ret, weights


def method4_shrinkage(
    vol_lt: pd.Series, vol_rl: pd.Series, close: pd.DataFrame,
    regime_corrs: dict, forecastable: bool, noisy: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """M4: Shrinkage correlation-aware minimum-variance portfolio."""
    common = vol_lt.index.intersection(vol_rl.index).sort_values()
    lt = vol_lt.reindex(common).fillna(0)
    rl = vol_rl.reindex(common).fillna(0)

    regime_raw, _, _ = regime_series(close)
    regime_raw = regime_raw.reindex(common).fillna("CHOP")

    # Shrinkage factor
    if forecastable:
        shrinkage = 0.35
    elif noisy:
        shrinkage = 0.65
    else:
        shrinkage = 0.50

    vol_lt_20d = lt.rolling(20, min_periods=10).std()
    vol_rl_20d = rl.rolling(20, min_periods=10).std()

    # Weekly computation with 5-day transition smoothing
    rebal_dates = pd.date_range(common[0], common[-1], freq="W-FRI")
    rebal_dates = rebal_dates[rebal_dates.isin(common) | True]  # get closest
    rebal_dates = common[common.to_series().dt.dayofweek == 4]  # Fridays in data
    if len(rebal_dates) < 2:
        rebal_dates = common[::5]

    target_w_lt = pd.Series(0.5, index=common, dtype=float)
    forecast_corrs = pd.Series(np.nan, index=common, dtype=float)

    for d in common:
        loc = common.get_loc(d)
        if loc < 60:
            target_w_lt.iloc[loc] = 0.5
            continue

        # Sample correlation
        window = slice(max(0, loc - 60), loc)
        rho_sample = float(lt.iloc[window].corr(rl.iloc[window]))
        if not np.isfinite(rho_sample):
            rho_sample = 0.0

        # Regime prior
        current_regime = regime_raw.iloc[loc] if loc < len(regime_raw) else "CHOP"
        rho_prior = regime_corrs.get(current_regime, 0.0)
        if not np.isfinite(rho_prior):
            rho_prior = 0.0

        # Shrinkage blend
        rho_forecast = (1 - shrinkage) * rho_sample + shrinkage * rho_prior
        forecast_corrs.iloc[loc] = rho_forecast

        # Min-variance weights
        s_lt = float(vol_lt_20d.iloc[loc]) if np.isfinite(vol_lt_20d.iloc[loc]) else 0.01
        s_rl = float(vol_rl_20d.iloc[loc]) if np.isfinite(vol_rl_20d.iloc[loc]) else 0.01
        s_lt = max(s_lt, 1e-6)
        s_rl = max(s_rl, 1e-6)

        denom = s_lt**2 + s_rl**2 - 2 * rho_forecast * s_lt * s_rl
        if abs(denom) < 1e-12:
            w = 0.5
        else:
            w = (s_rl**2 - rho_forecast * s_lt * s_rl) / denom

        w = np.clip(w, 0.15, 0.85)
        target_w_lt.iloc[loc] = w

    # 5-day transition smoothing
    smooth_w_lt = target_w_lt.rolling(5, min_periods=1).mean()
    smooth_w_rl = 1.0 - smooth_w_lt

    port_ret = smooth_w_lt.shift(1).fillna(0.5) * lt + smooth_w_rl.shift(1).fillna(0.5) * rl
    weights = pd.DataFrame({"w_lt": smooth_w_lt, "w_rl": smooth_w_rl}, index=common)
    forecast_df = pd.DataFrame({"forecast_corr": forecast_corrs, "regime": regime_raw}, index=common)
    return port_ret, weights, forecast_df


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY TARGETING (C2)
# ═══════════════════════════════════════════════════════════════════════════

def apply_vol_target(port_ret: pd.Series, vol_target: float = 0.15, lookback: int = 20) -> pd.Series:
    """Scale portfolio returns to hit target vol."""
    realized = port_ret.rolling(lookback, min_periods=10).std() * np.sqrt(ANN_FACTOR)
    scalar = (vol_target / realized.replace(0, np.nan)).clip(0.25, 2.0).fillna(1.0)
    return port_ret * scalar.shift(1).fillna(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION (C4)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_method(
    port_ret: pd.Series, label: str, regime: pd.Series,
    weights_df: pd.DataFrame | None = None,
    cost_per_switch_bps: float = 0.0,
) -> dict:
    """Full evaluation for one construction method on the test set."""
    test = port_ret[port_ret.index >= pd.Timestamp(TEST_START)].dropna()
    if len(test) < 30:
        return {"label": label, "error": f"Only {len(test)} test days"}

    # Apply vol targeting
    test_vt = apply_vol_target(test)

    # Transaction cost for weight changes
    turnover = 0.0
    if weights_df is not None:
        w_test = weights_df.reindex(test.index)
        if "w_lt" in w_test.columns:
            daily_to = w_test["w_lt"].diff().abs().fillna(0)
            turnover = float(daily_to.mean())
            cost_drag = daily_to * (20 / 10_000)
            if cost_per_switch_bps > 0 and "regime" in w_test.columns:
                regime_changes = (w_test["regime"] != w_test["regime"].shift(1)).astype(float)
                cost_drag += regime_changes * (cost_per_switch_bps / 10_000)
            test_net = test_vt - cost_drag.reindex(test_vt.index).fillna(0)
        else:
            test_net = test_vt
    else:
        test_net = test_vt

    # Metrics
    equity_gross = (1 + test_vt).cumprod()
    equity_net = (1 + test_net).cumprod()
    metrics_gross = compute_metrics(equity_gross)
    metrics_net = compute_metrics(equity_net)

    # Regime-conditional
    regime_aligned = regime.reindex(test_net.index)
    regime_sharpes = {}
    regime_dd = {}
    for r_label in ["BULL", "BEAR", "CHOP"]:
        mask = regime_aligned == r_label
        r_slice = test_net[mask].dropna()
        if len(r_slice) > 20 and r_slice.std() > 1e-12:
            regime_sharpes[r_label] = float((r_slice.mean() / r_slice.std()) * np.sqrt(ANN_FACTOR))
            eq_slice = (1 + r_slice).cumprod()
            dd = eq_slice / eq_slice.cummax() - 1.0
            regime_dd[r_label] = float(dd.min())
        else:
            regime_sharpes[r_label] = np.nan
            regime_dd[r_label] = np.nan

    # Rolling 90d Sharpe
    mu = test_net.rolling(90, min_periods=60).mean()
    sigma = test_net.rolling(90, min_periods=60).std()
    rolling_sharpe = (mu / sigma.replace(0, np.nan)) * np.sqrt(ANN_FACTOR)

    return {
        "label": label,
        "strat_ret_net": test_net,
        "strat_ret_gross": test_vt,
        "equity_net": equity_net,
        "equity_gross": equity_gross,
        "metrics_gross": metrics_gross,
        "metrics_net": metrics_net,
        "regime_sharpes": regime_sharpes,
        "regime_dd": regime_dd,
        "rolling_sharpe": rolling_sharpe,
        "avg_daily_turnover": turnover,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CORRELATION FORECAST QUALITY (C5)
# ═══════════════════════════════════════════════════════════════════════════

def correlation_forecast_quality(
    vol_lt: pd.Series, vol_rl: pd.Series,
    forecast_df: pd.DataFrame, uncond_corr: float,
    out_dir: Path,
) -> dict:
    """Evaluate shrinkage forecast vs baselines on 5-day windows."""
    test_start = pd.Timestamp(TEST_START)
    common = vol_lt.index.intersection(vol_rl.index).sort_values()
    lt = vol_lt.reindex(common)
    rl = vol_rl.reindex(common)
    fc = forecast_df["forecast_corr"].reindex(common)
    sample_corr = lt.rolling(60, min_periods=40).corr(rl)

    test_dates = common[common >= test_start]
    errors_shrink = []
    errors_hist = []
    errors_naive = []
    forecast_vals = []
    realized_vals = []

    for i in range(0, len(test_dates) - 5, 5):
        d = test_dates[i]
        window = test_dates[i:i + 5]
        if len(window) < 5:
            break
        realized = float(lt.reindex(window).corr(rl.reindex(window)))
        if not np.isfinite(realized):
            continue

        f_shrink = float(fc.loc[d]) if d in fc.index and np.isfinite(fc.loc[d]) else uncond_corr
        f_hist = float(sample_corr.loc[d]) if d in sample_corr.index and np.isfinite(sample_corr.loc[d]) else uncond_corr
        f_naive = uncond_corr

        errors_shrink.append(abs(f_shrink - realized))
        errors_hist.append(abs(f_hist - realized))
        errors_naive.append(abs(f_naive - realized))
        forecast_vals.append(f_shrink)
        realized_vals.append(realized)

    mae_shrink = float(np.mean(errors_shrink)) if errors_shrink else np.nan
    mae_hist = float(np.mean(errors_hist)) if errors_hist else np.nan
    mae_naive = float(np.mean(errors_naive)) if errors_naive else np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(forecast_vals, realized_vals, alpha=0.4, s=15, label="5-day windows")
    lims = [-1, 1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect forecast")
    ax.set_xlabel("Shrinkage Forecast Correlation")
    ax.set_ylabel("Realized 5-day Correlation")
    ax.set_title("Correlation Forecast Quality (Test Period)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "plots" / "correlation_forecast_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "mae_shrinkage": mae_shrink,
        "mae_historical": mae_hist,
        "mae_naive": mae_naive,
        "shrinkage_adds_value": mae_shrink < mae_hist,
        "n_windows": len(errors_shrink),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_equity_comparison(results: dict[str, dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    for label, r in results.items():
        if "equity_net" in r:
            eq = r["equity_net"]
            ax.plot(eq.index, eq.values, label=label, linewidth=1.2)
    ax.set_title("Portfolio Construction — Test Period Equity Curves (Net)")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "plots" / "equity_curves_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_sharpe_comparison(results: dict[str, dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, r in results.items():
        if "rolling_sharpe" in r:
            rs = r["rolling_sharpe"]
            ax.plot(rs.index, rs.values, label=label, linewidth=1.0)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("Rolling 90-Day Sharpe — All Construction Methods (Test Period)")
    ax.set_ylabel("Annualized Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "plots" / "rolling_sharpe_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weight_history(weights: pd.DataFrame, label: str, out_dir: Path):
    test_w = weights[weights.index >= pd.Timestamp(TEST_START)]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(test_w.index, 0, test_w["w_lt"], alpha=0.6, label="VOL_LT weight")
    ax.fill_between(test_w.index, test_w["w_lt"], 1, alpha=0.6, label="VOL_RL weight")
    if "regime" in test_w.columns:
        regime_colors = {"BULL": "#2ecc71", "BEAR": "#e74c3c", "CHOP": "#f39c12"}
        for i in range(len(test_w)):
            r = test_w["regime"].iloc[i] if "regime" in test_w.columns else "CHOP"
            if r in regime_colors:
                ax.axvspan(test_w.index[i], test_w.index[min(i + 1, len(test_w) - 1)],
                          alpha=0.08, color=regime_colors[r])
    ax.set_title(f"{label} — Weight History (Test Period)")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "plots" / f"weight_history_{label.lower().replace(' ', '_').replace('-', '')}.png",
               dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmap(strat_ret: pd.Series, label: str, out_dir: Path):
    monthly = strat_ret.resample("ME").sum()
    df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    pivot = df.pivot(index="year", columns="month", values="ret")
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"M{m}" for m in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{label} — Monthly Returns (Test Period)")
    fig.colorbar(im, ax=ax, label="Return")
    plt.tight_layout()
    fig.savefig(out_dir / "plots" / "monthly_heatmap_best_method.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# REPORT WRITING
# ═══════════════════════════════════════════════════════════════════════════

def write_portfolio_report(
    results: dict[str, dict], rc: dict, fc_quality: dict,
    out_dir: Path,
) -> Path:
    lines = ["# Phase C: Portfolio Construction Report", ""]

    # Summary placeholder — filled after analysis
    lines.extend(["## Phase C Summary", ""])

    # Regime correlation findings
    lines.extend(["## Regime Correlation Findings", ""])
    lines.append(f"Unconditional correlation: {rc['unconditional_corr']:.3f}")
    rc_c = rc["regime_corrs"]
    lines.append(f"BULL: {rc_c.get('BULL', np.nan):.3f} | "
                f"BEAR: {rc_c.get('BEAR', np.nan):.3f} | "
                f"CHOP: {rc_c.get('CHOP', np.nan):.3f}")
    if rc["negative_hedge"]:
        lines.append("Negative-correlation hedge detected in at least one regime.")
    else:
        lines.append("No negative-correlation hedge detected.")
    if rc["forecastable"]:
        lines.append(f"Correlation is forecastable (lag-5 ACF = {rc['acf_5']:.3f}).")
    elif rc["noisy"]:
        lines.append(f"Correlation is noisy (lag-5 ACF = {rc['acf_5']:.3f}).")
    else:
        lines.append(f"Correlation persistence is moderate (lag-5 ACF = {rc['acf_5']:.3f}).")
    lines.append("")

    # Comparison table
    lines.extend(["## Portfolio Construction Results", ""])
    labels = ["M1-EqWt", "M2-VolPar", "M3-Regime", "M4-Shrink"]
    header = f"| {'Metric':<20s} | " + " | ".join(f"{l:>10s}" for l in labels) + " |"
    sep = "|" + "-" * 22 + "|" + "|".join("-" * 12 for _ in labels) + "|"
    lines.append(header)
    lines.append(sep)

    def _val(label, key, fmt=".3f", sub="metrics_net"):
        r = results.get(label, {})
        m = r.get(sub, {})
        v = m.get(key, np.nan)
        return f"{v:{fmt}}" if np.isfinite(v) else "N/A"

    def _pct(label, key, sub="metrics_net"):
        r = results.get(label, {})
        m = r.get(sub, {})
        v = m.get(key, np.nan)
        return f"{v:.1%}" if np.isfinite(v) else "N/A"

    for metric, key, fmt_fn in [
        ("Sharpe", "sharpe", _val),
        ("CAGR", "cagr", _pct),
        ("Max Drawdown", "max_dd", _pct),
        ("Calmar", "calmar", lambda l, k: _val(l, k, ".2f")),
        ("Sortino", "sortino", lambda l, k: _val(l, k, ".2f")),
    ]:
        row = f"| {metric:<20s} | " + " | ".join(f"{fmt_fn(l, key):>10s}" for l in labels) + " |"
        lines.append(row)

    # Turnover
    row_to = f"| {'Avg Daily TO':<20s} | "
    for l in labels:
        r = results.get(l, {})
        to = r.get("avg_daily_turnover", 0)
        row_to += f"{to:>10.4f} | "
    lines.append(row_to)

    # Net Sharpe (already using metrics_net)
    lines.append("")

    # Regime breakdown
    lines.extend(["## Regime Breakdown (Net Sharpe)", ""])
    header_r = f"| {'Regime':<10s} | " + " | ".join(f"{l:>10s}" for l in labels) + " |"
    lines.append(header_r)
    lines.append(sep)
    for regime in ["BULL", "BEAR", "CHOP"]:
        row = f"| {regime:<10s} | "
        for l in labels:
            r = results.get(l, {})
            rs = r.get("regime_sharpes", {})
            v = rs.get(regime, np.nan)
            row += f"{v:>10.2f} | " if np.isfinite(v) else f"{'N/A':>10s} | "
        lines.append(row)
    lines.append("")

    # Correlation forecast quality
    lines.extend(["## Correlation Forecast Quality (Method 4)", ""])
    lines.append(f"| Metric | Shrinkage | Historical | Naive Mean |")
    lines.append(f"|--------|-----------|------------|------------|")
    lines.append(f"| MAE | {fc_quality['mae_shrinkage']:.4f} | "
                f"{fc_quality['mae_historical']:.4f} | "
                f"{fc_quality['mae_naive']:.4f} |")
    lines.append(f"| N windows | {fc_quality['n_windows']} | | |")
    if fc_quality["shrinkage_adds_value"]:
        lines.append("Verdict: **SHRINKAGE ADDS VALUE**")
    else:
        lines.append("Verdict: **SHRINKAGE DEGRADES** — regime-conditional prior "
                     "hurts more than it helps at this timescale.")
    lines.append("")

    # Determine winner
    best_label = None
    best_sharpe = -99.0
    m1_sharpe = results.get("M1-EqWt", {}).get("metrics_net", {}).get("sharpe", -99)
    for l in labels:
        r = results.get(l, {})
        s = r.get("metrics_net", {}).get("sharpe", -99)
        if np.isfinite(s) and s > best_sharpe:
            best_sharpe = s
            best_label = l

    margin = best_sharpe - m1_sharpe if np.isfinite(m1_sharpe) else 0

    # Null result checks (C6)
    null_result = margin < 0.05
    regime_fragile = False
    high_turnover = False

    if best_label:
        r = results[best_label]
        rs = r.get("regime_sharpes", {})
        positive_regimes = sum(1 for v in rs.values() if np.isfinite(v) and v > 0)
        regime_fragile = positive_regimes < 2
        gross_sharpe = r.get("metrics_gross", {}).get("sharpe", 0)
        net_sharpe = r.get("metrics_net", {}).get("sharpe", 0)
        if best_label in ["M3-Regime", "M4-Shrink"] and (gross_sharpe - net_sharpe) > 0.1:
            high_turnover = True

    lines.extend(["## Verdict", ""])
    if null_result:
        lines.append(f"**WINNING METHOD: M1-EqWt (Equal Weight)**")
        lines.append(f"No method materially improved on equal-weighting (margin = {margin:+.3f} Sharpe).")
    else:
        lines.append(f"**WINNING METHOD: {best_label}**")
        m1_calmar = results.get("M1-EqWt", {}).get("metrics_net", {}).get("calmar", 0)
        best_calmar = results.get(best_label, {}).get("metrics_net", {}).get("calmar", 0)
        calmar_margin = best_calmar - m1_calmar if np.isfinite(m1_calmar) and np.isfinite(best_calmar) else 0
        lines.append(f"Margin over equal-weight: +{margin:.3f} Sharpe, +{calmar_margin:.2f} Calmar")

    if regime_fragile:
        lines.append("Flag: **REGIME_FRAGILE** — winner only works in 1 of 3 regimes.")
    else:
        lines.append("Regime robustness: **ROBUST** (positive Sharpe in 2+ regimes)")

    if high_turnover:
        lines.append("Flag: **HIGH_TURNOVER_COST** — dynamic method's edge is partially consumed by costs.")
    else:
        lines.append("Turnover concern: **ACCEPTABLE**")

    lines.append("")

    report_path = out_dir / "portfolio_construction_report.md"
    report_path.write_text("\n".join(lines))
    return report_path, null_result, best_label, best_sharpe, m1_sharpe


# ═══════════════════════════════════════════════════════════════════════════
# MASTER REPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_master_report(
    rc: dict, method_results: dict, best_label: str,
    fc_quality: dict, null_result: bool,
    m1_sharpe: float, best_sharpe: float,
    out_dir: Path,
) -> Path:
    today = dt_date.today().isoformat()
    m_best = method_results.get(best_label, {}).get("metrics_net", {})

    lines = [
        "═" * 60,
        "CRYPTO ALPHA RESEARCH PIPELINE — FULL RESULTS",
        f"Run date: {today}",
        "Data: Coinbase Advanced spot, 232 assets, 2017-2025",
        "Papers processed: 111 discovered, 5 passed all filters",
        "═" * 60,
        "",
        "## Executive Summary",
        "",
        "The pipeline discovered two deployable cross-sectional factors in crypto: "
        "VOL_LT (low-volatility, Sharpe 0.59) and VOL_RL (volume-relative, Sharpe 0.55). "
        "Both factors are strengthening and uncorrelated (rho=0.042), providing regime-complementary coverage — "
        "VOL_LT dominates in bear markets (Sharpe 2.55) while VOL_RL dominates in bulls (Sharpe 1.56). "
        "ML overlays (Phase B) added no value over the raw factor signals. "
        f"Portfolio construction (Phase C) tested four allocation methods; "
        f"the recommended approach achieves a combined net Sharpe of {best_sharpe:.2f} on the test period.",
        "",
        "## Pipeline Performance",
        "",
        "| Stage | Count |",
        "|---|---|",
        "| Papers discovered | 111 |",
        "| Passed quality filters | 5 (4.5%) |",
        "| Passed methodology audit | 5 |",
        "| Translated to crypto | 4 (1 untranslatable — microstructure, no order book) |",
        "| Produced deployable edge | 2 factors (VOL_LT, VOL_RL) |",
        "",
        "## Phase A: Alpha Decay Findings",
        "",
        "| Factor | Sharpe | Last 90d | Decay | Priority |",
        "|--------|--------|----------|-------|----------|",
        "| VOL_LT | 0.59 | 3.68 | STRENGTHENING | HIGH |",
        "| VOL_RL | 0.55 | 2.15 | STRENGTHENING | HIGH |",
        "| REV_1W | 0.24 | -1.05 | STRENGTHENING | LOW |",
        "| MOM_1M | -0.46 | 0.53 | DECAYING | AVOID |",
        "| MOM_12M | -0.52 | -1.66 | DECAYING | AVOID |",
        "| MOM_3M | -1.02 | -0.17 | DECAYING | AVOID |",
        "",
        "Key finding: Cross-sectional momentum is dead across all three lookbacks "
        "in crypto. Low-volatility and volume-relative factors are alive and "
        "strengthening. This is consistent with a retail-dominated market where "
        "the low-risk anomaly has not been arbitraged away.",
        "",
        "## Phase B: ML Overlay Findings",
        "",
        "**Key finding 1:** Raw factors beat ML overlays on both signals. "
        "MSE VOL_RL overfit 24x (IS Sharpe 9.21 vs OOS 0.39). ML adds no value "
        "when the underlying signal is already clean and well-specified.",
        "",
        "**Key finding 2:** SPO loss produced genuine differentiation (VOL_RL pred "
        "correlation 0.479) but still underperformed the raw factor.",
        "",
        "**Key finding 3 (most important):** VOL_LT and VOL_RL are regime complements.",
        "",
        "| Factor | BULL | BEAR | CHOP |",
        "|--------|------|------|------|",
        "| VOL_LT | -0.75 | 2.55 | 0.35 |",
        "| VOL_RL | 1.56 | -0.03 | -0.17 |",
        "",
        "Cross-factor correlation: 0.042. Decision: **DEPLOY RAW FACTORS.** No ML overlay.",
        "",
        "## Phase C: Portfolio Construction Findings",
        "",
        f"Regime-conditional correlation analysis: "
        f"BULL={rc['regime_corrs'].get('BULL', np.nan):.3f}, "
        f"BEAR={rc['regime_corrs'].get('BEAR', np.nan):.3f}, "
        f"CHOP={rc['regime_corrs'].get('CHOP', np.nan):.3f}.",
    ]

    if rc["negative_hedge"]:
        lines.append("Negative-correlation hedge confirmed — factors hedge each other in at least one regime.")
    else:
        lines.append("No strong negative-correlation hedge detected.")

    if rc["forecastable"]:
        lines.append(f"Correlation is forecastable (lag-5 ACF = {rc['acf_5']:.3f}).")
    else:
        lines.append(f"Correlation has limited forecastability (lag-5 ACF = {rc['acf_5']:.3f}).")

    lines.append("")
    lines.append("Construction method comparison (test period 2024-04-01 to 2025-12-14):")
    lines.append("")
    labels_all = ["M1-EqWt", "M2-VolPar", "M3-Regime", "M4-Shrink"]
    lines.append("| Method | Net Sharpe | CAGR | Max DD | Calmar |")
    lines.append("|--------|-----------|------|--------|--------|")
    for l in labels_all:
        r = method_results.get(l, {})
        m = r.get("metrics_net", {})
        lines.append(
            f"| {l} | {m.get('sharpe', np.nan):.3f} | "
            f"{m.get('cagr', np.nan):.1%} | "
            f"{m.get('max_dd', np.nan):.1%} | "
            f"{m.get('calmar', np.nan):.2f} |"
        )
    lines.append("")

    if null_result:
        lines.append(f"Decision: **Equal-weight (M1)** — no method materially improved on 50/50.")
        chosen_method = "Equal Weight (50% VOL_LT / 50% VOL_RL)"
        chosen_rebal = "Monthly drift correction"
    else:
        lines.append(f"Decision: **{best_label}** — margin over equal-weight: "
                     f"+{best_sharpe - m1_sharpe:.3f} Sharpe.")
        chosen_method = best_label
        if "Regime" in best_label:
            chosen_method = "Regime-Switching (BULL: 30/70, BEAR: 80/20, CHOP: 55/45)"
            chosen_rebal = "On regime change (5-day persistence filter) + monthly drift correction"
        elif "Shrink" in best_label:
            chosen_method = "Shrinkage Correlation-Aware (minimum-variance)"
            chosen_rebal = "Weekly with 5-day transition smoothing"
        elif "VolPar" in best_label:
            chosen_method = "Volatility Parity (inverse-vol weighted)"
            chosen_rebal = "Weekly"
        else:
            chosen_method = "Equal Weight (50/50)"
            chosen_rebal = "Monthly drift correction"

    fc_verdict = "SHRINKAGE ADDS VALUE" if fc_quality["shrinkage_adds_value"] else "SHRINKAGE DEGRADES"
    lines.append(f"Correlation forecast quality: {fc_verdict} (MAE: shrinkage={fc_quality['mae_shrinkage']:.4f} "
                f"vs historical={fc_quality['mae_historical']:.4f} vs naive={fc_quality['mae_naive']:.4f}).")
    lines.append("")

    # Deployment specification
    lines.extend([
        "═" * 60,
        "## DEPLOYMENT SPECIFICATION",
        "═" * 60,
        "",
        "**STRATEGY:** Combined VOL_LT + VOL_RL Raw Factor Portfolio",
        "",
        "**SIGNALS:**",
        "",
        "VOL_LT: Long bottom-quintile 20d-realized-vol assets, "
        "short top-quintile, equal-weighted within each leg. Rebalance: daily.",
        "",
        "VOL_RL: Long top-quintile (5d-avg-vol / 60d-avg-vol) assets, "
        "short bottom-quintile, equal-weighted. Rebalance: daily.",
        "",
        "**UNIVERSE:**",
        "All assets in bars_1d_usd_universe_clean with: "
        "minimum 90 days of history at signal date, "
        "point-in-time membership (no forward-looking universe), "
        "no stablecoins, wrapped tokens, or LP tokens.",
        "",
        f"**PORTFOLIO CONSTRUCTION:** {chosen_method}",
        f"Rebalance frequency: {chosen_rebal}",
        "",
        "**RISK MANAGEMENT:**",
        "- Volatility target: 15% annualized (ANN_FACTOR=365)",
        "- Vol estimation lookback: 20 days",
        "- Max scale factor: 2.0 (no more than 2x leverage)",
        "- Min scale factor: 0.25 (always maintain at least 25% exposure)",
        "- Max single-asset position: 20% of either leg",
        "",
        "**TRANSACTION COST ASSUMPTION:** 20bps per side",
        f"Net Sharpe at this cost level: {m_best.get('sharpe', np.nan):.3f}",
        "",
        f"**PERFORMANCE EXPECTATIONS (test period 2024-04-01 to 2025-12-14):**",
        f"- Equal-weight baseline Sharpe: {m1_sharpe:.3f}",
        f"- Chosen construction Sharpe: {best_sharpe:.3f}",
        f"- CAGR (net of costs): {m_best.get('cagr', np.nan):.1%}",
        f"- Max drawdown: {m_best.get('max_dd', np.nan):.1%}",
        "",
        "**FORWARD TESTING SPECIFICATION:**",
        "- Duration: 30 days minimum before any live allocation",
        "- Paper trading size: $10,000 notional per factor leg",
        "- Frequency: run signals daily at close, execute at next open",
        "",
        "Metrics to monitor daily:",
        "- Rolling 10-day Sharpe (each factor separately)",
        "- Current regime classification (BTC 21d return)",
        "- Factor correlation (10-day rolling)",
        "- Portfolio drawdown from forward-test peak",
        "",
        "**KILL SWITCH CONDITIONS:**",
        "",
        "Hard stops:",
        "- Portfolio drawdown exceeds 10% from forward-test peak",
        "- Either factor Sharpe (rolling 10d) below -1.0 for 5 consecutive days",
        "- Any single asset moves > 4 standard deviations in one day "
        "-> reduce that position 50% immediately, review before restoring",
        "",
        "Soft review triggers:",
        "- Factor correlation (10d rolling) exceeds 0.6 "
        "(factors losing their complementary structure)",
        "- Universe drops below 50 assets on either factor "
        "(signal becomes concentrated)",
        "- Vol target scale factor hits 0.25 floor for 10 consecutive days "
        "(market regime may have changed structurally)",
        "",
        "**REGIME MONITORING:**",
        "- If BEAR regime persists > 30 days: review VOL_RL sizing "
        "(BEAR Sharpe is -0.03, acceptable but monitor for deterioration).",
        "- If BULL regime persists > 30 days: review VOL_LT sizing "
        "(BULL Sharpe is -0.75, known cost of carrying the hedge).",
        "",
        "## Open Research Questions (Next Pipeline Run)",
        "",
        "1. **Time-series momentum** (each asset vs its own history) was not tested. "
        "Cross-sectional momentum is dead, but TSMOM may behave differently.",
        "",
        "2. **Microstructure signals** require order book data. "
        "Coinbase Advanced exposes L2 via WebSocket. "
        "Estimated infrastructure: 2-3 days to build, ongoing storage. "
        "Estimated value: high.",
        "",
        "3. **Funding rate signals** were not tested (no perp data). "
        "Cross-exchange funding rate arbitrage is a documented crypto-native edge.",
        "",
        "4. **DeePM regime-robust objective** worth stealing as a portfolio-level "
        "risk framework (minimax EVaR could replace vol-targeting overlay).",
        "",
        "5. **Alpha decay model re-run** should be scheduled every 90 days. "
        "Factor half-lives in crypto are short.",
    ])

    report_path = Path("artifacts/research") / f"master_report_{today}.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C: Correlation-Aware Portfolio Construction")
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--table", type=str, default="bars_1d_usd_universe_clean")
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default="2025-12-15")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    log(f"PHASE C STARTING — regime correlation analysis first")
    log(f"Phase C starting. Factors: VOL_LT, VOL_RL. Test: {TEST_START}/{args.end}")

    # Load data
    close, returns, volume = load_data(args.db, args.table, args.start, args.end)
    n_assets = close.shape[1]
    log(f"  Loaded {n_assets} assets, {close.index.min().date()} to {close.index.max().date()}")

    # Build factor returns
    vol_lt = build_factor_returns(close, returns, volume, "VOL_LT")
    vol_rl = build_factor_returns(close, returns, volume, "VOL_RL")
    log(f"  VOL_LT: {len(vol_lt)} days, VOL_RL: {len(vol_rl)} days")

    # ═════════════════════════════════════════════════════════════════
    # PRE-PHASE C: REGIME CORRELATION ANALYSIS
    # ═════════════════════════════════════════════════════════════════
    rc = regime_correlation_analysis(vol_lt, vol_rl, close, out_dir)
    write_regime_correlation_report(rc, out_dir)

    rc_c = rc["regime_corrs"]
    hedge_flag = "NEGATIVE_HEDGE" if rc["negative_hedge"] else "NOT APPLICABLE"
    fc_flag = "FORECASTABLE" if rc["forecastable"] else ("NOISY" if rc["noisy"] else "MODERATE")
    print(f"REGIME CORRELATION ANALYSIS COMPLETE")
    print(f"Key finding: correlation in BULL={rc_c.get('BULL', np.nan):.3f}, "
          f"BEAR={rc_c.get('BEAR', np.nan):.3f}, CHOP={rc_c.get('CHOP', np.nan):.3f}")
    print(f"Hedge flag: {hedge_flag}")
    print(f"Forecastability: {fc_flag}")

    log(f"Regime correlations: BULL={rc_c.get('BULL', np.nan):.3f}, "
        f"BEAR={rc_c.get('BEAR', np.nan):.3f}, CHOP={rc_c.get('CHOP', np.nan):.3f}")

    # Shrinkage factor
    if rc["forecastable"]:
        shrinkage_val = 0.35
    elif rc["noisy"]:
        shrinkage_val = 0.65
    else:
        shrinkage_val = 0.50
    log(f"Forecastability: {fc_flag}. Shrinkage factor set to {shrinkage_val}")

    # ═════════════════════════════════════════════════════════════════
    # PHASE C: BUILD AND EVALUATE FOUR CONSTRUCTION METHODS
    # ═════════════════════════════════════════════════════════════════
    log("C1. Building four construction methods")

    regime, _, _ = regime_series(close)

    # M1: Equal Weight
    m1_ret = method1_equal_weight(vol_lt, vol_rl)

    # M2: Vol Parity
    m2_ret, m2_weights = method2_vol_parity(vol_lt, vol_rl)

    # M3: Regime Switching
    m3_ret, m3_weights = method3_regime_switching(vol_lt, vol_rl, close)

    # M4: Shrinkage
    m4_ret, m4_weights, m4_forecast = method4_shrinkage(
        vol_lt, vol_rl, close, rc["regime_corrs"], rc["forecastable"], rc["noisy"],
    )

    print("CONSTRUCTION METHODS COMPLETE — evaluating")

    # Evaluate all four
    method_results = {}
    for label, ret, w, extra_cost in [
        ("M1-EqWt", m1_ret, None, 0),
        ("M2-VolPar", m2_ret, m2_weights, 0),
        ("M3-Regime", m3_ret, m3_weights, 5),
        ("M4-Shrink", m4_ret, m4_weights, 0),
    ]:
        log(f"C4. Evaluating {label}")
        ev = evaluate_method(ret, label, regime, w, cost_per_switch_bps=extra_cost)
        method_results[label] = ev
        m = ev.get("metrics_net", {})
        log(f"  {label}: Sharpe={m.get('sharpe', np.nan):.3f}, "
            f"CAGR={m.get('cagr', np.nan):.1%}, MaxDD={m.get('max_dd', np.nan):.1%}")

    # C5: Correlation forecast quality
    log("C5. Evaluating correlation forecast quality")
    fc_quality = correlation_forecast_quality(
        vol_lt, vol_rl, m4_forecast, rc["unconditional_corr"], out_dir,
    )
    log(f"  MAE — Shrinkage: {fc_quality['mae_shrinkage']:.4f}, "
        f"Historical: {fc_quality['mae_historical']:.4f}, "
        f"Naive: {fc_quality['mae_naive']:.4f}")

    # Plots
    log("Generating plots")
    plot_equity_comparison(method_results, out_dir)
    plot_rolling_sharpe_comparison(method_results, out_dir)
    plot_weight_history(m3_weights, "M3-Regime", out_dir)
    plot_weight_history(m4_weights, "M4-Shrink", out_dir)

    # Find best method for heatmap
    best_label = max(method_results, key=lambda l: method_results[l].get("metrics_net", {}).get("sharpe", -99))
    best_ret = method_results[best_label].get("strat_ret_net")
    if best_ret is not None:
        plot_monthly_heatmap(best_ret, best_label, out_dir)

    # Write portfolio construction report
    report_path, null_result, winner, best_sharpe, m1_sharpe = write_portfolio_report(
        method_results, rc, fc_quality, out_dir,
    )

    # Log construction results
    net_sharpes = {l: method_results[l].get("metrics_net", {}).get("sharpe", np.nan)
                  for l in ["M1-EqWt", "M2-VolPar", "M3-Regime", "M4-Shrink"]}
    log(f"Construction results: " + ", ".join(f"{k}={v:.3f}" for k, v in net_sharpes.items()) + " (net Sharpe)")
    log(f"Winner: {winner}. Margin over equal-weight: {best_sharpe - m1_sharpe:+.3f}")

    # Null result file
    if null_result:
        null_path = out_dir / "NULL_RESULT.md"
        null_path.write_text(
            "# Phase C Null Result\n\n"
            "No portfolio construction method materially improved on equal-weighting.\n"
            "Likely causes: (1) two-factor portfolio is too small for correlation "
            "structure to matter, (2) the near-zero unconditional correlation means "
            "optimization has little room to improve, (3) test period too short.\n\n"
            "Recommendation: deploy equal-weight combination of VOL_LT and VOL_RL. "
            "Revisit portfolio construction when universe expands to 5+ factors.\n"
        )
        log("Null result: no method beat equal-weight by >= 0.05 Sharpe")

    log("PHASE C COMPLETE — master report generating")
    print("PHASE C COMPLETE — master report generating")

    # ═════════════════════════════════════════════════════════════════
    # MASTER REPORT
    # ═════════════════════════════════════════════════════════════════
    master_path = write_master_report(
        rc, method_results, winner, fc_quality,
        null_result, m1_sharpe, best_sharpe, out_dir,
    )
    log(f"Phase C complete. Master report written to {master_path}")
    print(f"PIPELINE COMPLETE — all outputs at artifacts/research/")
    print(f"Master report: {master_path}")


if __name__ == "__main__":
    main()
