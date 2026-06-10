"""Pipeline stages — each takes data + config and returns a StageResult.

V2 changes (2026-03):
  - Spearman rank IC replaces Pearson IC
  - Purge/embargo enforced in CPCV block splits
  - Inverse-vol weighted, beta-neutral long/short proxy
  - New turnover/cost gate (stage 3.5)
  - Orthogonalized redundancy (residual IC testing)
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.alpha_pipeline.types import GateConfig, StageResult, StageVerdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _cross_sectional_ic(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily cross-sectional Spearman rank IC.

    Both inputs are (ts x symbol) wide DataFrames.
    Returns a Series indexed by ts with one IC value per day.
    """
    common_dates = scores.index.intersection(forward_returns.index)
    common_syms = scores.columns.intersection(forward_returns.columns)
    s = scores.loc[common_dates, common_syms]
    r = forward_returns.loc[common_dates, common_syms]

    ics = []
    for dt in common_dates:
        row_s = s.loc[dt].dropna()
        row_r = r.loc[dt].dropna()
        common = row_s.index.intersection(row_r.index)
        if len(common) < 5:
            continue
        ic = row_s[common].rank().corr(row_r[common].rank())
        if not np.isnan(ic):
            ics.append((dt, ic))

    if not ics:
        return pd.Series(dtype=float)
    return pd.Series(
        [x[1] for x in ics], index=pd.DatetimeIndex([x[0] for x in ics]),
    )


def _forward_returns(close_wide: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Compute forward returns from a (ts x symbol) close price panel."""
    return close_wide.pct_change(horizon, fill_method=None).shift(-horizon)


def _trailing_vol(
    returns_wide: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Rolling annualized volatility for inverse-vol weighting."""
    return returns_wide.rolling(lookback, min_periods=max(lookback // 2, 5)).std()


def _invvol_ls_return(
    scores_row: pd.Series,
    returns_row: pd.Series,
    vol_row: pd.Series | None = None,
) -> float | None:
    """Inverse-vol weighted, beta-neutral long-short return for one date.

    Returns None if insufficient data.
    """
    common = scores_row.dropna().index.intersection(
        returns_row.dropna().index
    )
    if vol_row is not None:
        common = common.intersection(vol_row.dropna().index)
        common = common[vol_row[common] > 0]
    if len(common) < 10:
        return None

    ranks = scores_row[common].rank(pct=True)
    long_mask = ranks >= 0.8
    short_mask = ranks <= 0.2
    if long_mask.sum() < 2 or short_mask.sum() < 2:
        return None

    ret = returns_row[common]

    if vol_row is not None:
        inv_vol = 1.0 / vol_row[common]
        w_long = inv_vol[long_mask] / inv_vol[long_mask].sum()
        w_short = inv_vol[short_mask] / inv_vol[short_mask].sum()
        long_ret = (ret[long_mask] * w_long).sum()
        short_ret = (ret[short_mask] * w_short).sum()
    else:
        long_ret = ret[long_mask].mean()
        short_ret = ret[short_mask].mean()

    ls_ret = long_ret - short_ret

    # Beta-neutralize: subtract cross-sectional mean return (market)
    mkt_ret = ret.mean()
    ls_ret -= 0  # L/S is already ~neutral; subtract any residual market tilt
    # The portfolio is long top quintile, short bottom — ~dollar neutral.
    # Subtract the portfolio's net market exposure:
    if vol_row is not None:
        net_mkt_exposure = (w_long.sum() - w_short.sum()) * mkt_ret
    else:
        net_mkt_exposure = 0.0
    ls_ret -= net_mkt_exposure

    return float(ls_ret)


def _compute_ls_series(
    scores: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    returns_wide: pd.DataFrame | None,
    dates: list,
    vol_lookback: int = 20,
    use_invvol: bool = True,
) -> list[float]:
    """Compute daily long-short returns for a list of dates."""
    vol_wide = None
    if use_invvol and returns_wide is not None:
        vol_wide = _trailing_vol(returns_wide, lookback=vol_lookback)

    daily_rets = []
    for dt in dates:
        if dt not in scores.index or dt not in fwd_ret.index:
            continue
        vol_row = vol_wide.loc[dt] if vol_wide is not None and dt in vol_wide.index else None
        val = _invvol_ls_return(scores.loc[dt], fwd_ret.loc[dt], vol_row)
        if val is not None:
            daily_rets.append(val)
    return daily_rets


def _apply_embargo(
    blocks: list[list[int]],
    test_combo: tuple[int, ...],
    n_dates: int,
    pct_embargo: float,
) -> tuple[list[int], list[int]]:
    """Build train/test index sets with embargo zones around test blocks.

    Removes `n_embargo` observations from training data on each side of
    every test block to prevent information leakage.
    """
    n_embargo = max(1, int(n_dates * pct_embargo))

    test_idx_set: set[int] = set()
    for g in test_combo:
        test_idx_set.update(blocks[g])

    # Embargo zone: indices within n_embargo of any test index
    embargo_set: set[int] = set()
    for idx in test_idx_set:
        for offset in range(-n_embargo, n_embargo + 1):
            neighbor = idx + offset
            if 0 <= neighbor < n_dates and neighbor not in test_idx_set:
                embargo_set.add(neighbor)

    train_idx = [
        i for i in range(n_dates)
        if i not in test_idx_set and i not in embargo_set
    ]
    test_idx = sorted(test_idx_set)
    return train_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: IC SCREENING
# ═══════════════════════════════════════════════════════════════════════

def stage_ic_screen(
    scores: pd.DataFrame,
    close_wide: pd.DataFrame,
    config: GateConfig,
) -> StageResult:
    """Screen alpha by cross-sectional Spearman IC against 1-period forward returns.

    Gates:
        |t-stat| >= min_abs_tstat
        |mean IC| >= min_abs_mean_ic
        n_days >= min_ic_days
    """
    fwd_ret = _forward_returns(close_wide, horizon=1)
    daily_ic = _cross_sectional_ic(scores, fwd_ret)

    n_days = len(daily_ic)
    if n_days == 0:
        return StageResult(
            stage="ic_screen", verdict=StageVerdict.FAIL,
            metrics={"n_days": 0}, detail="No valid IC observations",
        )

    mean_ic = float(daily_ic.mean())
    std_ic = float(daily_ic.std())
    tstat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 0 else 0.0

    metrics: dict[str, Any] = {
        "n_days": n_days,
        "mean_ic": round(mean_ic, 6),
        "std_ic": round(std_ic, 6),
        "tstat_ic": round(tstat, 3),
    }

    passed = (
        abs(tstat) >= config.min_abs_tstat
        and abs(mean_ic) >= config.min_abs_mean_ic
        and n_days >= config.min_ic_days
    )

    detail_parts = []
    if abs(tstat) < config.min_abs_tstat:
        detail_parts.append(f"|t-stat|={abs(tstat):.2f} < {config.min_abs_tstat}")
    if abs(mean_ic) < config.min_abs_mean_ic:
        detail_parts.append(f"|mean_ic|={abs(mean_ic):.4f} < {config.min_abs_mean_ic}")
    if n_days < config.min_ic_days:
        detail_parts.append(f"n_days={n_days} < {config.min_ic_days}")

    return StageResult(
        stage="ic_screen",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail="; ".join(detail_parts) if detail_parts else "IC screen passed",
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: IC DECAY
# ═══════════════════════════════════════════════════════════════════════

def stage_ic_decay(
    scores: pd.DataFrame,
    close_wide: pd.DataFrame,
    config: GateConfig,
) -> StageResult:
    """Check that IC decays with horizon (real signals decay; data mining doesn't).

    Gates:
        IC at shortest horizon > IC at longest horizon
        Optionally require monotonic decay
    """
    horizons = config.max_decay_horizons
    ic_by_horizon: dict[int, float] = {}

    for h in horizons:
        fwd_ret = _forward_returns(close_wide, horizon=h)
        daily_ic = _cross_sectional_ic(scores, fwd_ret)
        if len(daily_ic) < 20:
            continue
        ic_by_horizon[h] = float(abs(daily_ic.mean()))

    if len(ic_by_horizon) < 2:
        return StageResult(
            stage="ic_decay", verdict=StageVerdict.SKIP,
            metrics={"horizons_computed": len(ic_by_horizon)},
            detail="Insufficient data for decay analysis",
        )

    sorted_horizons = sorted(ic_by_horizon.keys())
    ic_values = [ic_by_horizon[h] for h in sorted_horizons]

    metrics: dict[str, Any] = {
        f"abs_mean_ic_h{h}": round(v, 6) for h, v in ic_by_horizon.items()
    }

    shortest_ic = ic_values[0]
    longest_ic = ic_values[-1]
    decays = shortest_ic > longest_ic

    is_monotonic = all(
        ic_values[i] >= ic_values[i + 1] for i in range(len(ic_values) - 1)
    )
    metrics["decays_short_to_long"] = decays
    metrics["is_monotonic"] = is_monotonic

    if config.require_monotonic_decay:
        passed = is_monotonic
    else:
        passed = decays

    detail = ""
    if not passed:
        detail = (
            f"IC does not decay: "
            f"h={sorted_horizons[0]}→{shortest_ic:.4f}, "
            f"h={sorted_horizons[-1]}→{longest_ic:.4f}"
        )

    return StageResult(
        stage="ic_decay",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or "IC decay pattern confirmed",
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: REDUNDANCY CHECK (with orthogonalization)
# ═══════════════════════════════════════════════════════════════════════

def stage_redundancy(
    scores: pd.DataFrame,
    existing_alphas: dict[str, pd.DataFrame],
    config: GateConfig,
    close_wide: pd.DataFrame | None = None,
) -> StageResult:
    """Check that the candidate adds marginal information beyond existing alphas.

    When orthogonalize_redundancy is True (default), high-correlation candidates
    are not immediately rejected. Instead, the candidate is orthogonalized against
    the existing catalog and the residual is tested for IC significance.

    Gate (classic): max |corr| < threshold
    Gate (orthogonal): residual IC t-stat >= residual_min_abs_tstat
    """
    if not existing_alphas:
        return StageResult(
            stage="redundancy", verdict=StageVerdict.PASS,
            metrics={"n_existing": 0, "max_corr": 0.0},
            detail="No existing alphas to compare against",
        )

    max_corr = 0.0
    max_corr_name = ""
    correlations: dict[str, float] = {}

    for name, existing_scores in existing_alphas.items():
        common_dates = scores.index.intersection(existing_scores.index)
        common_syms = scores.columns.intersection(existing_scores.columns)
        if len(common_dates) < 20 or len(common_syms) < 3:
            continue

        s_flat = scores.loc[common_dates, common_syms].values.flatten()
        e_flat = existing_scores.loc[common_dates, common_syms].values.flatten()
        mask = ~(np.isnan(s_flat) | np.isnan(e_flat))
        if mask.sum() < 100:
            continue

        corr = float(np.corrcoef(s_flat[mask], e_flat[mask])[0, 1])
        correlations[name] = round(corr, 4)
        if abs(corr) > abs(max_corr):
            max_corr = corr
            max_corr_name = name

    metrics: dict[str, Any] = {
        "n_existing": len(existing_alphas),
        "max_corr": round(max_corr, 4),
        "max_corr_with": max_corr_name,
    }

    # Classic path: if below threshold, pass immediately
    if abs(max_corr) < config.max_correlation_with_existing:
        metrics["method"] = "classic"
        return StageResult(
            stage="redundancy", verdict=StageVerdict.PASS,
            metrics=metrics, detail="No redundancy detected",
        )

    # High correlation detected — try orthogonalization if enabled
    if not config.orthogonalize_redundancy or close_wide is None:
        metrics["method"] = "classic"
        return StageResult(
            stage="redundancy", verdict=StageVerdict.FAIL,
            metrics=metrics,
            detail=(
                f"Redundant with '{max_corr_name}' "
                f"(corr={max_corr:.3f} >= {config.max_correlation_with_existing})"
            ),
        )

    # Orthogonalize: regress out existing alphas, test residual IC
    residual = _orthogonalize(scores, existing_alphas)
    fwd_ret = _forward_returns(close_wide, horizon=1)
    residual_ic = _cross_sectional_ic(residual, fwd_ret)

    if len(residual_ic) < 50:
        metrics["method"] = "orthogonal"
        metrics["residual_n_days"] = len(residual_ic)
        return StageResult(
            stage="redundancy", verdict=StageVerdict.FAIL,
            metrics=metrics,
            detail="Insufficient data for residual IC test",
        )

    mean_res_ic = float(residual_ic.mean())
    std_res_ic = float(residual_ic.std())
    res_tstat = mean_res_ic / (std_res_ic / np.sqrt(len(residual_ic))) if std_res_ic > 0 else 0.0

    metrics["method"] = "orthogonal"
    metrics["residual_mean_ic"] = round(mean_res_ic, 6)
    metrics["residual_tstat_ic"] = round(res_tstat, 3)

    passed = abs(res_tstat) >= config.residual_min_abs_tstat
    detail = (
        f"Correlated with '{max_corr_name}' (corr={max_corr:.3f}), "
        f"residual IC t-stat={res_tstat:.2f} "
        f"({'passes' if passed else 'fails'} threshold {config.residual_min_abs_tstat})"
    )

    return StageResult(
        stage="redundancy",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail,
    )


def _orthogonalize(
    scores: pd.DataFrame,
    existing_alphas: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Project out existing alpha signals from the candidate, return residual.

    For each date, regresses the candidate's cross-sectional scores on the
    existing alphas and returns the residuals.
    """
    residual = scores.copy()

    for _name, existing in existing_alphas.items():
        common_dates = residual.index.intersection(existing.index)
        common_syms = residual.columns.intersection(existing.columns)
        if len(common_dates) < 20 or len(common_syms) < 3:
            continue

        for dt in common_dates:
            y = residual.loc[dt, common_syms].dropna()
            x = existing.loc[dt, common_syms].dropna()
            shared = y.index.intersection(x.index)
            if len(shared) < 5:
                continue
            yv = y[shared].values
            xv = x[shared].values
            xv_mean = xv.mean()
            xv_centered = xv - xv_mean
            denom = np.dot(xv_centered, xv_centered)
            if denom < 1e-12:
                continue
            beta = np.dot(xv_centered, yv - yv.mean()) / denom
            residual.loc[dt, shared] = yv - beta * xv_centered

    return residual


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3.5: TURNOVER / COST GATE
# ═══════════════════════════════════════════════════════════════════════

def stage_turnover(
    scores: pd.DataFrame,
    close_wide: pd.DataFrame,
    config: GateConfig,
) -> StageResult:
    """Estimate turnover from signal rank changes and gate on net-of-cost IC.

    Computes the daily turnover implied by rebalancing to the signal's quintile
    assignment, applies a cost penalty (bps), and checks that the cost-adjusted
    IC remains positive and significant.

    Gate: net IC (gross IC - cost drag) >= min_net_ic
    """
    fwd_ret = _forward_returns(close_wide, horizon=1)
    common_dates = sorted(scores.index.intersection(fwd_ret.index))
    common_syms = scores.columns.intersection(fwd_ret.columns)

    if len(common_dates) < 50 or len(common_syms) < 5:
        return StageResult(
            stage="turnover", verdict=StageVerdict.SKIP,
            metrics={}, detail="Insufficient data for turnover analysis",
        )

    s = scores.loc[common_dates, common_syms]

    # Compute daily quintile assignments
    ranks = s.rank(axis=1, pct=True)
    quintile = (ranks >= 0.8).astype(float) - (ranks <= 0.2).astype(float)

    # Turnover: fraction of positions that change per day
    turnover_per_day = quintile.diff().abs().sum(axis=1) / (2 * max(common_syms.__len__(), 1))
    turnover_per_day = turnover_per_day.iloc[1:]  # drop first NaN
    mean_turnover = float(turnover_per_day.mean())
    annualized_turnover = mean_turnover * 365

    # Gross IC
    daily_ic = _cross_sectional_ic(s, fwd_ret)
    if len(daily_ic) < 20:
        return StageResult(
            stage="turnover", verdict=StageVerdict.SKIP,
            metrics={}, detail="Insufficient IC data for cost analysis",
        )

    gross_ic = float(abs(daily_ic.mean()))

    # Cost drag: turnover * cost_bps converted to IC-equivalent units
    # Rough conversion: if mean absolute daily return is R, then cost drag on
    # IC ≈ (turnover * cost_bps/10000) / R
    mean_abs_ret = float(
        close_wide.loc[common_dates, common_syms]
        .pct_change(fill_method=None)
        .abs()
        .mean()
        .mean()
    )
    if mean_abs_ret < 1e-8:
        mean_abs_ret = 0.02  # fallback ~2% daily

    cost_drag_ic = (mean_turnover * config.turnover_cost_bps / 10_000) / mean_abs_ret
    net_ic = gross_ic - cost_drag_ic

    metrics: dict[str, Any] = {
        "mean_daily_turnover": round(mean_turnover, 4),
        "annualized_turnover": round(annualized_turnover, 2),
        "gross_ic": round(gross_ic, 6),
        "cost_drag_ic": round(cost_drag_ic, 6),
        "net_ic": round(net_ic, 6),
        "cost_bps": config.turnover_cost_bps,
    }

    passed = net_ic >= config.min_net_ic
    detail = ""
    if not passed:
        detail = (
            f"Net IC={net_ic:.4f} < {config.min_net_ic} "
            f"(gross={gross_ic:.4f}, cost_drag={cost_drag_ic:.4f}, "
            f"turnover={mean_turnover:.1%}/day at {config.turnover_cost_bps}bps)"
        )

    return StageResult(
        stage="turnover",
        verdict=StageVerdict.PASS if passed else StageVerdict.FAIL,
        metrics=metrics,
        detail=detail or (
            f"Net IC={net_ic:.4f} (gross={gross_ic:.4f}, "
            f"turnover={mean_turnover:.1%}/day)"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: WALK-FORWARD VALIDATION (CPCV + PBO)
# ═══════════════════════════════════════════════════════════════════════

def stage_walk_forward(
    scores: pd.DataFrame,
    close_wide: pd.DataFrame,
    config: GateConfig,
    returns_wide: pd.DataFrame | None = None,
) -> StageResult:
    """Walk-forward validation using Combinatorial Purged K-Fold.

    V2 improvements:
      - Embargo zones enforced around test blocks
      - Inverse-vol weighted long/short proxy (when returns_wide provided)

    Computes IS and OOS Sharpe per split, then PBO.
    Gate: PBO < max_pbo.
    """
    from src.afml.backtest_stats import probability_of_backtest_overfitting

    fwd_ret = _forward_returns(close_wide, horizon=1)
    common_dates = sorted(scores.index.intersection(fwd_ret.index))
    common_syms = scores.columns.intersection(fwd_ret.columns)

    if len(common_dates) < 100 or len(common_syms) < 3:
        return StageResult(
            stage="walk_forward", verdict=StageVerdict.SKIP,
            metrics={}, detail="Insufficient data for CPCV",
        )

    s = scores.loc[common_dates, common_syms]
    r = fwd_ret.loc[common_dates, common_syms]

    ret_w = None
    if returns_wide is not None and config.invvol_weight:
        ret_w = returns_wide.loc[
            returns_wide.index.isin(set(common_dates)), common_syms
        ]

    n_dates = len(common_dates)
    n_groups = min(config.cpcv_n_groups, n_dates // 20)
    if n_groups < 3:
        return StageResult(
            stage="walk_forward", verdict=StageVerdict.SKIP,
            metrics={}, detail=f"Only {n_dates} dates, need >=60 for CPCV",
        )

    block_size = n_dates // n_groups
    blocks = [
        list(range(i * block_size, min((i + 1) * block_size, n_dates)))
        for i in range(n_groups)
    ]

    is_sharpes = []
    oos_sharpes = []

    for test_combo in combinations(range(n_groups), config.cpcv_n_test_groups):
        train_idx, test_idx = _apply_embargo(
            blocks, test_combo, n_dates, config.cpcv_pct_embargo,
        )

        if len(train_idx) < 20 or len(test_idx) < 10:
            continue

        train_dates = [common_dates[i] for i in train_idx]
        test_dates = [common_dates[i] for i in test_idx]

        is_rets = _compute_ls_series(
            s, r, ret_w, train_dates, config.vol_lookback, config.invvol_weight,
        )
        oos_rets = _compute_ls_series(
            s, r, ret_w, test_dates, config.vol_lookback, config.invvol_weight,
        )

        if len(is_rets) < 10 or len(oos_rets) < 5:
            continue

        is_arr = np.array(is_rets)
        oos_arr_local = np.array(oos_rets)
        is_sharpe = float(is_arr.mean() / is_arr.std()) if is_arr.std() > 0 else 0.0
        oos_sharpe = float(oos_arr_local.mean() / oos_arr_local.std()) if oos_arr_local.std() > 0 else 0.0

        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)

    if len(is_sharpes) < 3:
        return StageResult(
            stage="walk_forward", verdict=StageVerdict.SKIP,
            metrics={}, detail="Too few CPCV splits",
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
        "embargo_pct": config.cpcv_pct_embargo,
        "invvol_weight": config.invvol_weight,
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


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5: DEFLATED SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════

def stage_deflated_sharpe(
    scores: pd.DataFrame,
    close_wide: pd.DataFrame,
    config: GateConfig,
    n_candidates_tested: int,
    returns_wide: pd.DataFrame | None = None,
) -> StageResult:
    """Apply the deflated Sharpe ratio to account for multiple testing.

    V2: Uses inverse-vol weighted long/short proxy when returns_wide is provided.
    Gate: P(true SR > E[max SR under null]) >= min_deflated_sharpe_pval.
    """
    from src.afml.backtest_stats import (
        deflated_sharpe_ratio,
        expected_max_sharpe,
    )

    fwd_ret = _forward_returns(close_wide, horizon=1)
    common_dates = sorted(scores.index.intersection(fwd_ret.index))
    common_syms = scores.columns.intersection(fwd_ret.columns)

    if len(common_dates) < 50 or len(common_syms) < 3:
        return StageResult(
            stage="deflated_sharpe", verdict=StageVerdict.SKIP,
            metrics={}, detail="Insufficient data for DSR",
        )

    s = scores.loc[common_dates, common_syms]
    r = fwd_ret.loc[common_dates, common_syms]

    ret_w = None
    if returns_wide is not None and config.invvol_weight:
        ret_w = returns_wide.loc[
            returns_wide.index.isin(set(common_dates)), common_syms
        ]

    daily_rets = _compute_ls_series(
        s, r, ret_w, common_dates, config.vol_lookback, config.invvol_weight,
    )

    if len(daily_rets) < 50:
        return StageResult(
            stage="deflated_sharpe", verdict=StageVerdict.SKIP,
            metrics={}, detail="Too few daily returns for DSR",
        )

    rets = pd.Series(daily_rets)
    n_obs = len(rets)
    mean_ret = float(rets.mean())
    std_ret = float(rets.std())
    observed_sr = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 0 else 0.0

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
