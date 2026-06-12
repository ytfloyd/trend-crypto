"""Pipeline stages for time-series (trend-following) alpha evaluation.

Seven stages, each returning a StageResult:

  1. Per-asset time-series IC
  2. Signal persistence (autocorrelation)
  3. IC horizon profile
  4. Vol-targeted portfolio backtest
  5. Walk-forward validation (CPCV + PBO)
  6. Deflated Sharpe ratio
  7. Blend diversification (informational, no gate)

Key difference from the cross-sectional pipeline: signals are evaluated
*per-asset* (does BTC's signal predict BTC's own returns?), and the
backtest proxy is a vol-targeted directional portfolio, not a quintile
long/short book.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ..common.types import StageResult, StageVerdict
from ..common.embargo import _apply_embargo
from .types import TSGateConfig
from .portfolio import vol_targeted_backtest

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _per_asset_ic(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """Compute per-asset Spearman IC between forecast and forward return.

    For each asset, correlate the time series of signal values with
    forward returns.  Returns a Series indexed by symbol.
    """
    fwd = returns.shift(-horizon)
    common_dates = forecasts.index.intersection(fwd.index)
    common_syms = forecasts.columns.intersection(fwd.columns)

    if len(common_dates) < 30 or len(common_syms) < 1:
        return pd.Series(dtype=float)

    fc = forecasts.loc[common_dates, common_syms]
    fr = fwd.loc[common_dates, common_syms]

    ics = {}
    for sym in common_syms:
        s = fc[sym].dropna()
        r = fr[sym].dropna()
        shared = s.index.intersection(r.index)
        if len(shared) < 30:
            continue
        ic = s[shared].rank().corr(r[shared].rank())
        if np.isfinite(ic):
            ics[sym] = ic

    return pd.Series(ics)


def _pooled_tstat(per_asset_ics: pd.Series) -> float:
    """Pooled t-statistic from per-asset ICs (mean / se)."""
    n = len(per_asset_ics)
    if n < 3:
        return 0.0
    mean_ic = per_asset_ics.mean()
    se = per_asset_ics.std() / np.sqrt(n)
    return float(mean_ic / se) if se > 1e-12 else 0.0


def _signal_autocorrelation(
    forecasts: pd.DataFrame,
    lag: int = 1,
) -> pd.Series:
    """Compute per-asset lag-N autocorrelation of the signal.

    Returns a Series indexed by symbol.
    """
    result = {}
    for sym in forecasts.columns:
        s = forecasts[sym].dropna()
        if len(s) < 30:
            continue
        ac = s.autocorr(lag=lag)
        if np.isfinite(ac):
            result[sym] = ac
    return pd.Series(result)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: PER-ASSET TIME-SERIES IC
# ═══════════════════════════════════════════════════════════════════════

def stage_ts_ic(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    config: TSGateConfig,
) -> StageResult:
    """Evaluate per-asset time-series predictive power.

    For each asset, compute Spearman correlation between forecast[t] and
    return[t+h].  Pool across assets to get median IC and t-stat.

    Gate: pooled |t-stat| > min_abs_tstat AND median |IC| > min_abs_median_ic.
    """
    per_asset = _per_asset_ic(forecasts, returns, horizon=config.ic_horizon)

    if len(per_asset) < config.min_ic_assets:
        return StageResult(
            stage="ts_ic", verdict=StageVerdict.SKIP,
            metrics={"n_assets": len(per_asset)},
            detail=f"Only {len(per_asset)} assets with IC (need {config.min_ic_assets})",
        )

    median_ic = float(per_asset.median())
    mean_ic = float(per_asset.mean())
    tstat = _pooled_tstat(per_asset)
    pct_positive = float((per_asset > 0).mean())

    metrics: dict[str, Any] = {
        "median_ic": round(median_ic, 6),
        "mean_ic": round(mean_ic, 6),
        "pooled_tstat": round(tstat, 2),
        "pct_positive": round(pct_positive, 3),
        "n_assets": len(per_asset),
    }

    passed = abs(tstat) >= config.min_abs_tstat and abs(median_ic) >= config.min_abs_median_ic
    detail = ""
    if not passed:
        detail = (
            f"TS IC: |t|={abs(tstat):.2f} (need {config.min_abs_tstat}), "
            f"median |IC|={abs(median_ic):.4f} (need {config.min_abs_median_ic})"
        )

    return StageResult(
        stage="ts_ic",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or f"TS IC: median={median_ic:.4f}, t={tstat:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: SIGNAL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

def stage_persistence(
    forecasts: pd.DataFrame,
    config: TSGateConfig,
) -> StageResult:
    """Evaluate signal persistence via lag-1 autocorrelation.

    Trend signals should be slow-changing (high autocorrelation).
    Fast-decaying signals imply high turnover and are untradable.

    Gate: median autocorrelation > min_median_autocorr.
    """
    ac = _signal_autocorrelation(forecasts, lag=1)

    if len(ac) < 5:
        return StageResult(
            stage="persistence", verdict=StageVerdict.SKIP,
            metrics={"n_assets": len(ac)},
            detail="Insufficient assets for persistence analysis",
        )

    median_ac = float(ac.median())
    mean_ac = float(ac.mean())
    min_ac = float(ac.min())

    metrics: dict[str, Any] = {
        "median_autocorr": round(median_ac, 4),
        "mean_autocorr": round(mean_ac, 4),
        "min_autocorr": round(min_ac, 4),
        "n_assets": len(ac),
    }

    passed = median_ac >= config.min_median_autocorr
    detail = ""
    if not passed:
        detail = (
            f"Persistence: median AC={median_ac:.3f} < {config.min_median_autocorr} "
            f"(signal too noisy / high turnover)"
        )

    return StageResult(
        stage="persistence",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or f"Persistence: median AC={median_ac:.3f}",
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: IC HORIZON PROFILE
# ═══════════════════════════════════════════════════════════════════════

def stage_horizon_profile(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    config: TSGateConfig,
) -> StageResult:
    """Evaluate signal predictive power across multiple forward horizons.

    Unlike CS decay (must decay monotonically), trend signals should
    predict at multiple horizons simultaneously.

    Gate: IC positive at h=1 (if required) AND positive at >= min_positive_horizons.
    """
    horizons = config.ic_horizons
    horizon_ics: dict[int, float] = {}

    for h in horizons:
        per_asset = _per_asset_ic(forecasts, returns, horizon=h)
        if len(per_asset) >= 5:
            horizon_ics[h] = float(per_asset.median())

    if len(horizon_ics) < 3:
        return StageResult(
            stage="horizon_profile", verdict=StageVerdict.SKIP,
            metrics={}, detail="Insufficient data for horizon analysis",
        )

    n_positive = sum(1 for v in horizon_ics.values() if v > 0)
    h1_positive = horizon_ics.get(1, 0) > 0

    metrics: dict[str, Any] = {
        f"ic_h{h}": round(v, 6) for h, v in horizon_ics.items()
    }
    metrics["n_positive_horizons"] = n_positive
    metrics["h1_positive"] = h1_positive

    h1_ok = h1_positive or not config.require_h1_positive
    passed = h1_ok and n_positive >= config.min_positive_horizons
    detail = ""
    if not passed:
        detail = (
            f"Horizon profile: {n_positive}/{len(horizons)} positive "
            f"(need {config.min_positive_horizons}), h1={horizon_ics.get(1, 'N/A')}"
        )

    return StageResult(
        stage="horizon_profile",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or f"Horizon profile: {n_positive}/{len(horizons)} positive",
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: VOL-TARGETED PORTFOLIO BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def stage_portfolio_backtest(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    config: TSGateConfig,
) -> StageResult:
    """Run a vol-targeted trend portfolio backtest.

    Per-asset: weight = forecast / 10 * (vol_target / asset_vol), capped.
    Aggregate with gross leverage constraint.  Deduct estimated costs.

    Gate: net Sharpe > min_net_sharpe.
    """
    result = vol_targeted_backtest(
        forecasts, returns,
        vol_target=config.vol_target,
        vol_lookback=config.vol_lookback,
        max_weight=config.max_weight,
        max_gross_leverage=config.max_gross_leverage,
        cost_bps=config.cost_bps,
        ann_factor=config.ann_factor,
    )

    if result.daily_returns.empty or len(result.daily_returns) < 60:
        return StageResult(
            stage="portfolio_backtest", verdict=StageVerdict.SKIP,
            metrics={}, detail="Insufficient data for portfolio backtest",
        )

    metrics: dict[str, Any] = {
        "net_sharpe": round(result.net_sharpe, 4),
        "net_sortino": round(result.net_sortino, 4),
        "cagr": round(result.cagr, 4),
        "max_drawdown": round(result.max_drawdown, 4),
        "annual_turnover": round(result.annual_turnover, 2),
        "total_cost_drag": round(result.total_cost_drag, 4),
    }

    passed = result.net_sharpe >= config.min_net_sharpe
    detail = ""
    if not passed:
        detail = (
            f"Portfolio: net Sharpe={result.net_sharpe:.3f} < {config.min_net_sharpe} "
            f"(CAGR={result.cagr:.1%}, MaxDD={result.max_drawdown:.1%})"
        )

    return StageResult(
        stage="portfolio_backtest",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or (
            f"Portfolio: net Sharpe={result.net_sharpe:.3f}, "
            f"CAGR={result.cagr:.1%}, MaxDD={result.max_drawdown:.1%}"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5: WALK-FORWARD VALIDATION (CPCV + PBO)
# ═══════════════════════════════════════════════════════════════════════

def stage_walk_forward(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    config: TSGateConfig,
) -> StageResult:
    """Walk-forward validation using CPCV on the vol-targeted portfolio P&L.

    Splits the time series into groups, runs vol-targeted backtests on
    each IS/OOS combination, and computes PBO from the Sharpe ratios.

    Gate: PBO < max_pbo.
    """
    from afml.backtest_stats import probability_of_backtest_overfitting

    common_dates = sorted(forecasts.index.intersection(returns.index))
    n_dates = len(common_dates)

    if n_dates < 120:
        return StageResult(
            stage="walk_forward", verdict=StageVerdict.SKIP,
            metrics={}, detail=f"Only {n_dates} dates (need >=120 for CPCV)",
        )

    n_groups = min(config.cpcv_n_groups, n_dates // 20)
    if n_groups < 3:
        return StageResult(
            stage="walk_forward", verdict=StageVerdict.SKIP,
            metrics={}, detail=f"Too few dates for {config.cpcv_n_groups}-group CPCV",
        )

    block_size = n_dates // n_groups
    blocks = [
        list(range(i * block_size, min((i + 1) * block_size, n_dates)))
        for i in range(n_groups)
    ]

    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []

    for test_combo in combinations(range(n_groups), config.cpcv_n_test_groups):
        train_idx, test_idx = _apply_embargo(
            blocks, test_combo, n_dates, config.cpcv_pct_embargo,
        )

        if len(train_idx) < 30 or len(test_idx) < 15:
            continue

        train_dates = [common_dates[i] for i in train_idx]
        test_dates = [common_dates[i] for i in test_idx]

        is_sharpe = _path_sharpe(forecasts, returns, train_dates, config)
        oos_sharpe = _path_sharpe(forecasts, returns, test_dates, config)

        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)

    if len(is_sharpes) < 3:
        return StageResult(
            stage="walk_forward", verdict=StageVerdict.SKIP,
            metrics={}, detail="Too few valid CPCV paths",
        )

    is_arr = np.array(is_sharpes)
    oos_arr = np.array(oos_sharpes)

    pbo_result = probability_of_backtest_overfitting(is_arr, oos_arr)
    pbo = pbo_result["pbo"]
    rank_corr = pbo_result["rank_corr"]

    metrics: dict[str, Any] = {
        "n_splits": len(is_sharpes),
        "mean_is_sharpe": round(float(is_arr.mean()), 4),
        "mean_oos_sharpe": round(float(oos_arr.mean()), 4),
        "pbo": round(pbo, 4),
        "rank_corr": round(rank_corr, 4),
    }

    passed = pbo < config.max_pbo
    detail = ""
    if not passed:
        detail = f"PBO={pbo:.3f} >= {config.max_pbo} (overfitting likely)"

    return StageResult(
        stage="walk_forward",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or f"PBO={pbo:.3f}, rank_corr={rank_corr:.3f}",
    )


def _path_sharpe(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    dates: list,
    config: TSGateConfig,
) -> float:
    """Compute annualised Sharpe of a vol-targeted portfolio over given dates."""
    date_set = set(dates)
    fc = forecasts.loc[forecasts.index.isin(date_set)]
    ret = returns.loc[returns.index.isin(date_set)]

    result = vol_targeted_backtest(
        fc, ret,
        vol_target=config.vol_target,
        vol_lookback=config.vol_lookback,
        max_weight=config.max_weight,
        max_gross_leverage=config.max_gross_leverage,
        cost_bps=config.cost_bps,
        ann_factor=config.ann_factor,
    )

    if result.daily_returns.empty or len(result.daily_returns) < 10:
        return 0.0

    return result.net_sharpe


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6: DEFLATED SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════

def stage_deflated_sharpe(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    config: TSGateConfig,
    n_candidates_tested: int,
) -> StageResult:
    """Apply the Deflated Sharpe Ratio to account for multiple testing.

    Uses the full-sample vol-targeted portfolio returns as the test statistic.

    Gate: DSR p-value >= min_deflated_sharpe_pval.
    """
    from afml.backtest_stats import (
        deflated_sharpe_ratio,
        expected_max_sharpe,
    )

    result = vol_targeted_backtest(
        forecasts, returns,
        vol_target=config.vol_target,
        vol_lookback=config.vol_lookback,
        max_weight=config.max_weight,
        max_gross_leverage=config.max_gross_leverage,
        cost_bps=config.cost_bps,
        ann_factor=config.ann_factor,
    )

    if result.daily_returns.empty or len(result.daily_returns) < 60:
        return StageResult(
            stage="deflated_sharpe", verdict=StageVerdict.SKIP,
            metrics={}, detail="Insufficient data for DSR",
        )

    rets = result.daily_returns
    n_obs = len(rets)
    observed_sr = result.net_sharpe

    n_trials = config.n_trials_for_deflation or max(n_candidates_tested, 1)
    sr_benchmark = expected_max_sharpe(n_trials)

    skew = float(rets.skew())
    kurt = float(rets.kurtosis())

    dsr_pval = deflated_sharpe_ratio(
        observed_sr=observed_sr,
        sr_benchmark=sr_benchmark,
        n_obs=n_obs,
        skewness=skew,
        excess_kurtosis=kurt,
    )

    metrics: dict[str, Any] = {
        "observed_sharpe": round(observed_sr, 4),
        "sr_benchmark": round(sr_benchmark, 4),
        "n_trials": n_trials,
        "dsr_pval": round(dsr_pval, 4),
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "n_obs": n_obs,
    }

    passed = dsr_pval >= config.min_deflated_sharpe_pval
    detail = ""
    if not passed:
        detail = (
            f"DSR p-val={dsr_pval:.3f} < {config.min_deflated_sharpe_pval} "
            f"(SR={observed_sr:.2f} vs benchmark={sr_benchmark:.2f} "
            f"from {n_trials} trials)"
        )

    return StageResult(
        stage="deflated_sharpe",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or f"DSR p-val={dsr_pval:.3f} (SR={observed_sr:.2f})",
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 7: BLEND DIVERSIFICATION (INFORMATIONAL)
# ═══════════════════════════════════════════════════════════════════════

def stage_blend_diversification(
    forecasts: pd.DataFrame,
    approved_forecasts: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    config: TSGateConfig,
    candidate_name: str,
) -> StageResult:
    """Assess diversification benefit of adding this signal to the approved set.

    Compute pairwise signal correlation with each existing approved signal.
    Flag complementary pairs (low correlation).

    This stage is informational only — it never fails.
    """
    if not approved_forecasts:
        return StageResult(
            stage="blend_diversification",
            verdict=StageVerdict.PASS,
            metrics={"n_approved": 0, "most_correlated": "N/A", "max_corr": 0.0},
            detail="First approved signal — no diversification analysis needed",
        )

    correlations: dict[str, float] = {}
    for name, existing_fc in approved_forecasts.items():
        common_dates = forecasts.index.intersection(existing_fc.index)
        common_syms = forecasts.columns.intersection(existing_fc.columns)
        if len(common_dates) < 30 or len(common_syms) < 3:
            continue
        # Flatten both signal panels into a single series and correlate
        fc_flat = forecasts.loc[common_dates, common_syms].values.flatten()
        ex_flat = existing_fc.loc[common_dates, common_syms].values.flatten()
        mask = np.isfinite(fc_flat) & np.isfinite(ex_flat)
        if mask.sum() < 100:
            continue
        corr = float(np.corrcoef(fc_flat[mask], ex_flat[mask])[0, 1])
        correlations[name] = corr

    if not correlations:
        return StageResult(
            stage="blend_diversification",
            verdict=StageVerdict.PASS,
            metrics={"n_approved": len(approved_forecasts)},
            detail="Could not compute correlations with approved signals",
        )

    max_corr_name = max(correlations, key=lambda k: abs(correlations[k]))
    max_corr = correlations[max_corr_name]
    n_complementary = sum(1 for c in correlations.values() if abs(c) < config.max_blend_correlation)

    metrics: dict[str, Any] = {
        "n_approved": len(approved_forecasts),
        "most_correlated": max_corr_name,
        "max_corr": round(max_corr, 4),
        "n_complementary": n_complementary,
        **{f"corr__{k}": round(v, 4) for k, v in correlations.items()},
    }

    return StageResult(
        stage="blend_diversification",
        verdict=StageVerdict.PASS,
        metrics=metrics,
        detail=(
            f"Max corr with '{max_corr_name}'={max_corr:.3f}, "
            f"{n_complementary}/{len(correlations)} complementary"
        ),
    )
