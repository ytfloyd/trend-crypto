"""Bayesian statistics for strategy evaluation.

Provides uncertainty quantification for the metrics that drive research
decisions: Sharpe ratios, hit rates, strategy comparisons, and parameter
robustness.  All methods use conjugate priors or direct simulation --
no MCMC dependencies, just numpy and scipy.

Designed as a drop-in augmentation to ``compute_metrics()``.

Reference: Will Kurt, *Bayesian Statistics the Fun Way*, Chapters 4-12.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .data import ANN_FACTOR
from .metrics import compute_metrics


# =====================================================================
# A. Posterior Sharpe Ratio
# =====================================================================
def posterior_sharpe(
    returns: pd.Series | np.ndarray,
    prior_sr_mean: float = 0.0,
    prior_sr_std: float = 0.5,
    n_samples: int = 10_000,
    ann_factor: float = ANN_FACTOR,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw samples from the posterior distribution of the Sharpe ratio.

    Uses a Normal-Inverse-Gamma conjugate model on (mu, sigma^2) of
    single-period returns, with a prior centred on ``prior_sr_mean``.

    Short backtests produce wide posteriors (honest uncertainty); long
    backtests converge toward the sample Sharpe.

    Parameters
    ----------
    returns : Series or array of single-period arithmetic returns.
    prior_sr_mean : prior mean for the annualised Sharpe ratio.
    prior_sr_std : prior std; wider = more agnostic. Default 0.5
        encodes the belief that most strategies don't work.
    n_samples : number of posterior draws.
    ann_factor : periods per year for annualising (365 for daily crypto).
    rng : optional numpy random Generator for reproducibility.

    Returns
    -------
    1-D array of ``n_samples`` posterior Sharpe draws (annualised).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(prior_sr_mean, prior_sr_std, size=n_samples)

    if rng is None:
        rng = np.random.default_rng()

    x_bar = float(np.mean(r))
    s2 = float(np.var(r, ddof=1))

    # Convert Sharpe-space prior to per-period mu prior.
    # SR = mu / sigma * sqrt(ann_factor) => mu_prior = SR_prior * sigma / sqrt(ann_factor)
    sigma_est = math.sqrt(s2) if s2 > 0 else 1e-8
    mu0 = prior_sr_mean * sigma_est / math.sqrt(ann_factor)
    # Prior precision: kappa0 controls how many "pseudo-observations" the prior is worth.
    # Map prior_sr_std to kappa0 via: var(mu_posterior) ~ sigma^2 / (kappa0 + n)
    # We want prior_std_mu = prior_sr_std * sigma / sqrt(ann_factor)
    prior_std_mu = prior_sr_std * sigma_est / math.sqrt(ann_factor)
    kappa0 = max(1.0, s2 / (prior_std_mu**2 + 1e-30) - n)
    kappa0 = max(kappa0, 0.5)

    # Normal-Inverse-Gamma posterior parameters
    nu0 = max(kappa0, 2.0)
    alpha0 = nu0 / 2.0
    beta0 = alpha0 * s2

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * (n - 1) * s2 + (kappa0 * n * (x_bar - mu0) ** 2) / (2.0 * kappa_n)

    # Sample sigma^2 from Inverse-Gamma(alpha_n, beta_n)
    sigma2_samples = beta_n / rng.gamma(alpha_n, 1.0, size=n_samples)
    sigma2_samples = np.maximum(sigma2_samples, 1e-30)

    # Sample mu | sigma^2 from Normal(mu_n, sigma^2 / kappa_n)
    mu_samples = rng.normal(mu_n, np.sqrt(sigma2_samples / kappa_n))

    # Annualised Sharpe = mu / sigma * sqrt(ann_factor)
    sigma_samples = np.sqrt(sigma2_samples)
    sr_samples = (mu_samples / sigma_samples) * math.sqrt(ann_factor)

    return sr_samples


# =====================================================================
# B. Bayesian Credible Interval for Sharpe
# =====================================================================
def sharpe_credible_interval(
    returns: pd.Series | np.ndarray,
    ci: float = 0.95,
    prior_sr_mean: float = 0.0,
    prior_sr_std: float = 0.5,
    ann_factor: float = ANN_FACTOR,
    n_samples: int = 20_000,
) -> dict:
    """Compute a Bayesian credible interval for the Sharpe ratio.

    Returns
    -------
    dict with keys:
        median    -- posterior median Sharpe
        lower     -- lower bound of CI
        upper     -- upper bound of CI
        p_positive -- P(Sharpe > 0), the probability the strategy makes money
        std       -- posterior standard deviation
    """
    samples = posterior_sharpe(
        returns,
        prior_sr_mean=prior_sr_mean,
        prior_sr_std=prior_sr_std,
        n_samples=n_samples,
        ann_factor=ann_factor,
    )
    alpha = (1.0 - ci) / 2.0
    return {
        "median": float(np.median(samples)),
        "lower": float(np.percentile(samples, 100 * alpha)),
        "upper": float(np.percentile(samples, 100 * (1 - alpha))),
        "p_positive": float(np.mean(samples > 0)),
        "std": float(np.std(samples)),
    }


# =====================================================================
# C. Bayesian Strategy Comparison (A vs B)
# =====================================================================
def p_a_beats_b(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    prior_sr_mean: float = 0.0,
    prior_sr_std: float = 0.5,
    ann_factor: float = ANN_FACTOR,
    n_samples: int = 20_000,
) -> dict:
    """Compute P(Sharpe_A > Sharpe_B) from posterior samples.

    Replaces eyeballing Sharpe differences in ablation tables with a
    proper probabilistic comparison.

    Returns
    -------
    dict with keys:
        p_a_wins    -- P(SR_A > SR_B)
        p_b_wins    -- P(SR_B > SR_A)
        median_diff -- median(SR_A - SR_B)
        ci_diff     -- (lower, upper) 95% CI on SR_A - SR_B
    """
    sa = posterior_sharpe(returns_a, prior_sr_mean, prior_sr_std, n_samples, ann_factor)
    sb = posterior_sharpe(returns_b, prior_sr_mean, prior_sr_std, n_samples, ann_factor)
    diff = sa - sb
    return {
        "p_a_wins": float(np.mean(diff > 0)),
        "p_b_wins": float(np.mean(diff < 0)),
        "median_diff": float(np.median(diff)),
        "ci_diff": (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
    }


# =====================================================================
# D. Beta-Binomial Hit Rate
# =====================================================================
def beta_hit_rate(
    returns: pd.Series | np.ndarray,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    ci: float = 0.95,
) -> dict:
    """Bayesian estimate of the hit rate (fraction of winning periods).

    Conjugate Beta-Binomial model:
        prior  = Beta(alpha, beta)
        data   = k wins out of n trials
        posterior = Beta(alpha + k, beta + n - k)

    This is the central example from Chapters 5-6 of the book.

    Parameters
    ----------
    returns : return series; positive returns count as "wins".
    prior_alpha, prior_beta : Beta prior parameters.
        (1, 1) = uniform = "I have no idea what the hit rate is."
        (50, 50) = informative prior centred at 50% (skeptical).
    ci : credible interval width (default 0.95).

    Returns
    -------
    dict with keys:
        posterior_mean -- posterior mean hit rate
        posterior_mode -- MAP estimate (most likely hit rate)
        lower, upper   -- credible interval bounds
        prior_mean     -- prior mean for reference
        n_wins, n_total -- data counts
        p_above_50     -- P(hit rate > 0.5)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n_total = len(r)
    n_wins = int(np.sum(r > 0))

    a_post = prior_alpha + n_wins
    b_post = prior_beta + (n_total - n_wins)
    dist = sp_stats.beta(a_post, b_post)

    alpha_tail = (1.0 - ci) / 2.0
    mode = (a_post - 1) / (a_post + b_post - 2) if (a_post > 1 and b_post > 1) else dist.mean()

    return {
        "posterior_mean": float(dist.mean()),
        "posterior_mode": float(mode),
        "lower": float(dist.ppf(alpha_tail)),
        "upper": float(dist.ppf(1.0 - alpha_tail)),
        "prior_mean": prior_alpha / (prior_alpha + prior_beta),
        "n_wins": n_wins,
        "n_total": n_total,
        "p_above_50": float(1.0 - dist.cdf(0.5)),
    }


# =====================================================================
# E. Bayes Factor for Positive Edge
# =====================================================================
def bayes_factor_positive_sharpe(
    returns: pd.Series | np.ndarray,
    prior_sr_mean: float = 0.0,
    prior_sr_std: float = 0.5,
    ann_factor: float = ANN_FACTOR,
    n_samples: int = 50_000,
) -> dict:
    """Bayes factor for H1: Sharpe > 0 vs H0: Sharpe <= 0.

    BF > 1 means evidence favours positive Sharpe.

    Interpretation thresholds (Jeffreys' scale, as used in the book):
        BF < 1   : evidence against (strategy likely has no edge)
        1-3      : barely worth mentioning
        3-10     : moderate evidence for positive edge
        10-30    : strong evidence
        30-100   : very strong evidence
        > 100    : decisive

    Parameters
    ----------
    returns : return series.
    prior_sr_mean, prior_sr_std : Sharpe prior.

    Returns
    -------
    dict with keys: bf, interpretation, p_positive.
    """
    samples = posterior_sharpe(
        returns,
        prior_sr_mean=prior_sr_mean,
        prior_sr_std=prior_sr_std,
        n_samples=n_samples,
        ann_factor=ann_factor,
    )
    p_pos = float(np.mean(samples > 0))
    p_neg = 1.0 - p_pos

    if p_neg < 1e-10:
        bf = 1e6
    elif p_pos < 1e-10:
        bf = 1e-6
    else:
        bf = p_pos / p_neg

    # Prior odds under the prior
    prior_samples = np.random.default_rng(42).normal(prior_sr_mean, prior_sr_std, size=n_samples)
    prior_p_pos = float(np.mean(prior_samples > 0))
    prior_odds = prior_p_pos / (1.0 - prior_p_pos) if prior_p_pos < 1.0 else 1e6

    # True Bayes factor = posterior odds / prior odds
    posterior_odds = bf
    if prior_odds > 1e-10:
        bf_true = posterior_odds / prior_odds
    else:
        bf_true = posterior_odds

    interpretation = _interpret_bf(bf_true)

    return {
        "bf": float(bf_true),
        "interpretation": interpretation,
        "p_positive": p_pos,
    }


def _interpret_bf(bf: float) -> str:
    """Jeffreys' scale for Bayes factor interpretation."""
    if bf < 1.0:
        return "evidence_against"
    elif bf < 3.0:
        return "barely_worth_mentioning"
    elif bf < 10.0:
        return "moderate_evidence"
    elif bf < 30.0:
        return "strong_evidence"
    elif bf < 100.0:
        return "very_strong_evidence"
    else:
        return "decisive"


# =====================================================================
# F. Bayesian Parameter Robustness
# =====================================================================
def parameter_robustness(
    metrics_by_param: dict[str, dict],
    sharpe_key: str = "sharpe",
    n_days_key: str = "n_days",
    prior: str = "uniform",
) -> dict:
    """Bayesian model averaging across a parameter sweep.

    Instead of picking the max-Sharpe parameter, compute a posterior-weighted
    average across the full grid.  This penalises isolated peaks (overfitting)
    and rewards broad plateaus (robust).

    Parameters
    ----------
    metrics_by_param : dict mapping parameter label to a metrics dict
        (as returned by ``compute_metrics``).  Must contain ``sharpe_key``.
    sharpe_key : key for the Sharpe ratio in each metrics dict.
    n_days_key : key for the number of observations.
    prior : "uniform" (default) or "shrinkage" (penalises extreme values).

    Returns
    -------
    dict with keys:
        posterior_weighted_sharpe -- BMA-weighted Sharpe across all params
        best_param               -- max-Sharpe parameter label
        best_sharpe              -- max raw Sharpe
        effective_n_params       -- how many params carry meaningful weight
                                    (high = robust, low = fragile)
        concentration            -- Herfindahl index of posterior weights
                                    (1/N = perfectly spread, 1.0 = all on one)
        param_weights            -- dict of parameter -> posterior weight
    """
    params = list(metrics_by_param.keys())
    if not params:
        return {
            "posterior_weighted_sharpe": np.nan,
            "best_param": None,
            "best_sharpe": np.nan,
            "effective_n_params": 0,
            "concentration": 1.0,
            "param_weights": {},
        }

    sharpes = np.array([
        metrics_by_param[p].get(sharpe_key, np.nan) for p in params
    ], dtype=np.float64)
    n_days = np.array([
        metrics_by_param[p].get(n_days_key, 365) for p in params
    ], dtype=np.float64)

    valid = np.isfinite(sharpes)
    if not np.any(valid):
        return {
            "posterior_weighted_sharpe": np.nan,
            "best_param": None,
            "best_sharpe": np.nan,
            "effective_n_params": 0,
            "concentration": 1.0,
            "param_weights": {},
        }

    # Approximate log-likelihood: Sharpe * sqrt(n) is approximately the
    # t-statistic, so exp(0.5 * SR * sqrt(n)) is proportional to likelihood.
    log_lik = np.where(valid, 0.5 * sharpes * np.sqrt(n_days), -1e10)

    if prior == "shrinkage":
        log_prior = -0.5 * sharpes**2
        log_lik = log_lik + np.where(valid, log_prior, 0.0)

    log_lik -= np.max(log_lik[valid])
    weights = np.where(valid, np.exp(log_lik), 0.0)
    w_sum = weights.sum()
    if w_sum > 0:
        weights /= w_sum
    else:
        weights = np.where(valid, 1.0 / valid.sum(), 0.0)

    weighted_sr = float(np.sum(weights * np.where(valid, sharpes, 0.0)))
    best_idx = int(np.nanargmax(sharpes))

    herfindahl = float(np.sum(weights**2))
    eff_n = 1.0 / herfindahl if herfindahl > 0 else 0.0

    return {
        "posterior_weighted_sharpe": weighted_sr,
        "best_param": params[best_idx],
        "best_sharpe": float(sharpes[best_idx]),
        "effective_n_params": eff_n,
        "concentration": herfindahl,
        "param_weights": {p: float(weights[i]) for i, p in enumerate(params)},
    }


# =====================================================================
# G. Augmented Metrics (drop-in extension of compute_metrics)
# =====================================================================
def compute_bayesian_metrics(
    equity: pd.Series,
    prior_sr_mean: float = 0.0,
    prior_sr_std: float = 0.5,
    ci: float = 0.95,
    ann_factor: float = ANN_FACTOR,
) -> dict:
    """Compute standard metrics augmented with Bayesian uncertainty estimates.

    Calls ``compute_metrics()`` then appends:
        sharpe_ci_lower, sharpe_ci_upper, p_positive_sharpe, sharpe_posterior_std,
        hit_rate_ci_lower, hit_rate_ci_upper, p_hit_above_50,
        bayes_factor, bf_interpretation

    Parameters
    ----------
    equity : equity curve (same as ``compute_metrics``).
    prior_sr_mean, prior_sr_std : Sharpe prior.
    ci : credible interval width.

    Returns
    -------
    dict with all keys from ``compute_metrics`` plus the Bayesian fields.
    """
    base = compute_metrics(equity)
    if base.get("n_days", 0) < 10:
        base.update({
            "sharpe_ci_lower": np.nan,
            "sharpe_ci_upper": np.nan,
            "p_positive_sharpe": np.nan,
            "sharpe_posterior_std": np.nan,
            "hit_rate_ci_lower": np.nan,
            "hit_rate_ci_upper": np.nan,
            "p_hit_above_50": np.nan,
            "bayes_factor": np.nan,
            "bf_interpretation": "insufficient_data",
        })
        return base

    ret = equity.pct_change().dropna()

    # Sharpe CI
    sr_ci = sharpe_credible_interval(
        ret, ci=ci, prior_sr_mean=prior_sr_mean,
        prior_sr_std=prior_sr_std, ann_factor=ann_factor,
    )
    base["sharpe_ci_lower"] = sr_ci["lower"]
    base["sharpe_ci_upper"] = sr_ci["upper"]
    base["p_positive_sharpe"] = sr_ci["p_positive"]
    base["sharpe_posterior_std"] = sr_ci["std"]

    # Hit rate CI
    hr = beta_hit_rate(ret, ci=ci)
    base["hit_rate_ci_lower"] = hr["lower"]
    base["hit_rate_ci_upper"] = hr["upper"]
    base["p_hit_above_50"] = hr["p_above_50"]

    # Bayes factor
    bf = bayes_factor_positive_sharpe(
        ret, prior_sr_mean=prior_sr_mean,
        prior_sr_std=prior_sr_std, ann_factor=ann_factor,
    )
    base["bayes_factor"] = bf["bf"]
    base["bf_interpretation"] = bf["interpretation"]

    return base


# =====================================================================
# Formatting helpers
# =====================================================================
def format_bayesian_table(results: list[dict], label_key: str = "label") -> str:
    """Pretty-print metrics with Bayesian uncertainty columns."""
    header = (
        f"{'Strategy':<25s} {'Sharpe':>7s} {'  95% CI':>16s} "
        f"{'P(SR>0)':>8s} {'BF':>7s} {'Hit%':>6s} {'  95% CI':>14s}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lbl = r.get(label_key, "")[:25]
        sr = r.get("sharpe", np.nan)
        lo = r.get("sharpe_ci_lower", np.nan)
        hi = r.get("sharpe_ci_upper", np.nan)
        pp = r.get("p_positive_sharpe", np.nan)
        bf = r.get("bayes_factor", np.nan)
        hr = r.get("hit_rate", np.nan)
        hr_lo = r.get("hit_rate_ci_lower", np.nan)
        hr_hi = r.get("hit_rate_ci_upper", np.nan)

        sr_ci_str = f"[{lo:+.2f}, {hi:+.2f}]" if np.isfinite(lo) else "       n/a"
        hr_ci_str = f"[{hr_lo:.1%},{hr_hi:.1%}]" if np.isfinite(hr_lo) else "      n/a"

        lines.append(
            f"{lbl:<25s} {sr:>7.2f} {sr_ci_str:>16s} "
            f"{pp:>7.1%} {bf:>7.1f} {hr:>5.1%} {hr_ci_str:>14s}"
        )
    return "\n".join(lines)


def comparison_table(
    pairs: list[tuple[str, str, dict]],
) -> str:
    """Pretty-print strategy comparison results from ``p_a_beats_b``.

    Parameters
    ----------
    pairs : list of (name_a, name_b, result_dict) tuples.
    """
    header = f"{'A':>20s}  vs  {'B':<20s} {'P(A>B)':>8s} {'Median diff':>12s} {'95% CI':>20s}"
    lines = [header, "-" * len(header)]
    for name_a, name_b, result in pairs:
        p = result["p_a_wins"]
        md = result["median_diff"]
        ci = result["ci_diff"]
        ci_str = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        lines.append(
            f"{name_a:>20s}  vs  {name_b:<20s} {p:>7.1%} {md:>+11.2f} {ci_str:>20s}"
        )
    return "\n".join(lines)
