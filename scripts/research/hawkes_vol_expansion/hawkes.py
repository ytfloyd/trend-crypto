"""Exponential Hawkes process with state-dependent background rate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass(frozen=True)
class HawkesParams:
    w0: float
    w_volcomp: float
    w_oibuild: float
    alpha: float
    beta: float

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else np.inf

    def mu(self, volcomp: float, oibuild: float) -> float:
        lin = self.w0 + self.w_volcomp * volcomp + self.w_oibuild * oibuild
        return float(np.exp(np.clip(lin, -20.0, 20.0)))


@dataclass(frozen=True)
class FitResult:
    params: HawkesParams
    log_likelihood: float
    n_events: int
    success: bool
    message: str
    std_errors: dict[str, float]


def fit_hawkes_mle(
    event_times: np.ndarray,
    mu_values: np.ndarray,
    *,
    beta_init: float = 1.0,
) -> FitResult:
    """Joint MLE for background weights and (α, β) with fixed μ(t) feature map.

    μ(t) is supplied exogenously from features × weights via exp(linear).
    We optimize w0, w_volcomp, w_oibuild, log_alpha, log_beta jointly.
    """
    if len(event_times) < 5:
        return _empty_fit("too_few_events")

    t = np.asarray(event_times, dtype=float)
    z_vol = np.asarray(mu_values[:, 0], dtype=float)
    z_oi = np.asarray(mu_values[:, 1], dtype=float)
    t0, t1 = float(t.min()), float(t.max())
    if t1 <= t0:
        return _empty_fit("degenerate_timeline")

    def neg_ll(theta: np.ndarray) -> float:
        w0, wv, wo, log_a, log_b = theta
        alpha = np.exp(log_a)
        beta = np.exp(log_b)
        if alpha >= beta:
            return 1e12
        ll = _log_likelihood(t, z_vol, z_oi, w0, wv, wo, alpha, beta, t1)
        if not np.isfinite(ll):
            return 1e12
        return -ll

    x0 = np.array([0.0, -0.5, 0.0, np.log(0.2), np.log(beta_init)])
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-10, 0), (-5, 5)]
    opt = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    w0, wv, wo, log_a, log_b = opt.x
    alpha, beta = float(np.exp(log_a)), float(np.exp(log_b))
    params = HawkesParams(w0, wv, wo, alpha, beta)
    ll = _log_likelihood(t, z_vol, z_oi, w0, wv, wo, alpha, beta, t1)
    std_errors = _approx_std_errors(opt, neg_ll)
    return FitResult(
        params=params,
        log_likelihood=float(ll),
        n_events=len(t),
        success=bool(opt.success),
        message=str(opt.message),
        std_errors=std_errors,
    )


def intensity_at_events(
    event_times: np.ndarray,
    mu_at_events: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    t = np.asarray(event_times, dtype=float)
    mu = np.asarray(mu_at_events, dtype=float)
    exc = np.zeros(len(t))
    for i in range(1, len(t)):
        exc[i] = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + exc[i - 1])
    return mu + alpha * exc


def compensator(
    event_times: np.ndarray,
    mu_grid: np.ndarray,
    time_grid: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> float:
    """Integrated intensity Λ(T) on [0, T]."""
    t_end = float(time_grid[-1])
    comp_mu = float(np.trapz(mu_grid, time_grid))
    comp_exc = 0.0
    for ti in event_times:
        if ti < t_end:
            comp_exc += (alpha / beta) * (1.0 - np.exp(-beta * (t_end - ti)))
    return comp_mu + comp_exc


def predicted_event_probability(
    lam_t: float,
    horizon: float,
) -> float:
    """P(≥1 event in (t, t+h]) under Poisson with rate λ ≈ constant over short h."""
    return float(1.0 - np.exp(-max(lam_t, 0.0) * horizon))


def time_rescaling_gaps(
    event_times: np.ndarray,
    mu_grid: np.ndarray,
    time_grid: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Rescaled inter-event gaps Λ(t_i) - Λ(t_{i-1}); should be Exp(1) if correct."""
    t = np.asarray(event_times, dtype=float)
    cumulative = []
    for ti in t:
        mask = time_grid <= ti
        cumulative.append(
            compensator(
                t[t <= ti],
                mu_grid[mask],
                time_grid[mask],
                alpha=alpha,
                beta=beta,
            )
        )
    cum = np.array(cumulative)
    return np.diff(cum)


def ks_exp_test(gaps: np.ndarray) -> tuple[float, float]:
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if len(gaps) < 5:
        return np.nan, np.nan
    return stats.kstest(gaps, "expon", args=(0, 1))


def _log_likelihood(
    t: np.ndarray,
    z_vol: np.ndarray,
    z_oi: np.ndarray,
    w0: float,
    wv: float,
    wo: float,
    alpha: float,
    beta: float,
    t_end: float,
) -> float:
    mu = np.exp(np.clip(w0 + wv * z_vol + wo * z_oi, -20, 20))
    exc = np.zeros(len(t))
    ll = 0.0
    for i in range(len(t)):
        lam = mu[i] + alpha * exc[i]
        if lam <= 0:
            return -np.inf
        ll += np.log(lam)
        if i + 1 < len(t):
            exc[i + 1] = np.exp(-beta * (t[i + 1] - t[i])) * (1.0 + exc[i])
    time_grid = np.linspace(t[0], t_end, max(50, len(t) * 4))
    mu_grid = np.interp(time_grid, t, mu)
    ll -= compensator(t, mu_grid, time_grid, alpha=alpha, beta=beta)
    return float(ll)


def _approx_std_errors(
    opt: optimize.OptimizeResult,
    neg_ll,
) -> dict[str, float]:
    names = ["w0", "w_volcomp", "w_oibuild", "log_alpha", "log_beta"]
    try:
        hess = optimize.approx_fprime(opt.x, lambda x: optimize.approx_fprime(x, neg_ll), 1e-5)
        cov = np.linalg.pinv(np.outer(hess, hess) + np.eye(len(opt.x)) * 1e-6)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        return {n: float(s) for n, s in zip(names, se)}
    except Exception:
        return {n: float("nan") for n in names}


def _empty_fit(reason: str) -> FitResult:
    p = HawkesParams(0.0, 0.0, 0.0, 0.0, 1.0)
    return FitResult(
        params=p,
        log_likelihood=float("nan"),
        n_events=0,
        success=False,
        message=reason,
        std_errors={},
    )
