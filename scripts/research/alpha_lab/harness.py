"""
Test harness for the Alpha Lab.

Takes a SignalSpec, computes the signal, builds long-short and long-only
portfolios, backtests with transaction costs, computes metrics + IC + regime
analysis, and returns a structured result dict.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.backtest import simple_backtest, DEFAULT_COST_BPS
from common.data import ANN_FACTOR
from common.metrics import compute_metrics, compute_regime
from common.risk_overlays import apply_position_limit_wide

from .signals import SignalSpec, get_signal_function


@dataclass
class HarnessResult:
    """Structured result from testing a single signal."""
    spec: SignalSpec
    long_short: dict[str, Any] = field(default_factory=dict)
    long_only: dict[str, Any] = field(default_factory=dict)
    ic: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        clean_params = {k: v for k, v in self.spec.params.items() if not k.startswith("_")}
        return {
            "name": self.spec.name,
            "family": self.spec.family,
            "params": clean_params,
            "description": self.spec.description,
            "long_short": self.long_short,
            "long_only": self.long_only,
            "ic": self.ic,
            "regime": self.regime,
            "meta": self.meta,
            "error": self.error,
        }


def _build_weights(
    signal: pd.DataFrame,
    universe: pd.DataFrame,
    mode: str = "long_short",
    quintile_frac: float = 0.2,
    max_wt: float = 0.10,
) -> pd.DataFrame:
    """Convert signal scores to portfolio weights.

    Parameters
    ----------
    signal : wide-format signal DataFrame (index=ts, columns=symbols)
    universe : wide-format boolean mask (True = in universe)
    mode : 'long_short' or 'long_only'
    quintile_frac : fraction of universe for top/bottom quintiles
    max_wt : maximum single-asset weight
    """
    sig = signal.copy()
    sig[~universe] = np.nan

    n_assets = universe.sum(axis=1)
    n_leg = (n_assets * quintile_frac).clip(lower=1).astype(int)

    weights = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)

    for i, ts in enumerate(sig.index):
        row = sig.iloc[i].dropna()
        if len(row) < 5:
            continue
        n = max(1, int(n_leg.iloc[i]))
        ranked = row.rank(ascending=True)

        top = ranked.nlargest(n).index
        bottom = ranked.nsmallest(n).index

        if mode == "long_short":
            weights.loc[ts, top] = 1.0 / n
            weights.loc[ts, bottom] = -1.0 / n
        else:
            weights.loc[ts, top] = 1.0 / n

    weights = apply_position_limit_wide(weights, max_wt)
    return weights


def _compute_regime(
    returns_wide: pd.DataFrame,
    btc_col: str | None = None,
) -> pd.Series:
    """Classify each day as BULL/BEAR/CHOP based on BTC 21d return.

    Delegates to the shared implementation in common.metrics.
    """
    return compute_regime(returns_wide, btc_col=btc_col, window=21)


def _regime_metrics(
    equity: pd.Series,
    regime: pd.Series,
) -> dict[str, dict]:
    """Compute per-regime Sharpe."""
    ret = equity.pct_change().dropna()
    common = ret.index.intersection(regime.index)
    ret = ret.reindex(common)
    regime = regime.reindex(common)

    result = {}
    for r in ["BULL", "BEAR", "CHOP"]:
        mask = regime == r
        r_ret = ret[mask]
        n = len(r_ret)
        if n < 10:
            result[r] = {"sharpe": np.nan, "n_days": n}
            continue
        std = float(r_ret.std())
        sharpe = float((r_ret.mean() / std) * np.sqrt(ANN_FACTOR)) if std > 1e-12 else np.nan
        result[r] = {"sharpe": round(sharpe, 3), "n_days": n}
    return result


def _compute_ic_quick(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 1,
) -> dict:
    """Fast IC computation — Spearman rank correlation at a single horizon."""
    from scipy import stats as sp_stats

    fwd = returns.rolling(horizon).sum().shift(-horizon)
    common_ts = signal.index.intersection(fwd.index)
    common_cols = signal.columns.intersection(fwd.columns)
    if len(common_cols) < 5:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_tstat": np.nan, "n_periods": 0}

    sig = signal.reindex(index=common_ts, columns=common_cols)
    fwd_r = fwd.reindex(index=common_ts, columns=common_cols)

    ic_vals = []
    sample_ts = common_ts[::max(1, len(common_ts) // 200)]
    for ts in sample_ts:
        s = sig.loc[ts].dropna()
        r = fwd_r.loc[ts].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 5:
            continue
        sv, rv = s.loc[common].values, r.loc[common].values
        if np.std(sv) == 0 or np.std(rv) == 0:
            continue
        corr, _ = sp_stats.spearmanr(sv, rv)
        ic_vals.append(corr)

    if len(ic_vals) < 3:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_tstat": np.nan, "n_periods": len(ic_vals)}

    ic_arr = np.array(ic_vals)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr, ddof=1))
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else np.nan

    return {
        "ic_mean": round(ic_mean, 5),
        "ic_std": round(ic_std, 5),
        "ic_tstat": round(ic_tstat, 3),
        "n_periods": len(ic_vals),
    }


def run_signal_test(
    spec: SignalSpec,
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    universe_wide: pd.DataFrame,
    test_start: str | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> HarnessResult:
    """Run the full test harness on a single signal.

    Returns a HarnessResult with metrics for long-short and long-only,
    IC analysis, and regime breakdown.
    """
    result = HarnessResult(spec=spec)
    t0 = time.time()

    try:
        fn = get_signal_function(spec)
        signal = fn(close_wide, volume_wide, returns_wide, spec.params)

        if signal.isna().all().all():
            result.error = "Signal produced all NaN"
            result.meta["elapsed_sec"] = round(time.time() - t0, 1)
            return result

        for mode in ["long_short", "long_only"]:
            weights = _build_weights(signal, universe_wide, mode=mode)

            if test_start:
                ts_mask = weights.index >= pd.Timestamp(test_start)
                weights = weights.loc[ts_mask]
                ret_slice = returns_wide.reindex(weights.index).fillna(0.0)
            else:
                ret_slice = returns_wide

            bt = simple_backtest(weights, ret_slice, cost_bps=cost_bps)
            if bt.empty or len(bt) < 30:
                continue
            eq = bt.set_index("ts")["portfolio_equity"]
            metrics = compute_metrics(eq)
            metrics["avg_turnover"] = round(float(bt["turnover"].mean()), 5)
            metrics["avg_gross_exposure"] = round(float(bt["gross_exposure"].mean()), 3)
            metrics = {k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}

            if mode == "long_short":
                result.long_short = metrics
            else:
                result.long_only = metrics

        ic_1d = _compute_ic_quick(signal, returns_wide, horizon=1)
        ic_5d = _compute_ic_quick(signal, returns_wide, horizon=5)
        result.ic = {"1d": ic_1d, "5d": ic_5d}

        if result.long_short:
            ls_weights = _build_weights(signal, universe_wide, mode="long_short")
            if test_start:
                ls_weights = ls_weights.loc[ls_weights.index >= pd.Timestamp(test_start)]
                ret_slice = returns_wide.reindex(ls_weights.index).fillna(0.0)
            else:
                ret_slice = returns_wide
            bt = simple_backtest(ls_weights, ret_slice, cost_bps=cost_bps)
            eq = bt.set_index("ts")["portfolio_equity"]
            regime = _compute_regime(returns_wide.reindex(eq.index))
            result.regime = _regime_metrics(eq, regime)

    except Exception:
        result.error = traceback.format_exc()

    result.meta["elapsed_sec"] = round(time.time() - t0, 1)
    return result
