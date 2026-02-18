"""
Log-Periodic Power Law Singularity (LPPLS) model fitting.

Implements the Filimonov & Sornette (2013) linearised calibration with
a **vectorised** grid search: for each candidate (tc, m, ω), the model
is linear in (A, B, C1, C2) and solved via batch OLS using numpy.

LPPL formula
------------
    E[ln p(t)] = A + B·(tc - t)^m
                 + C1·(tc - t)^m · cos(ω · ln(tc - t))
                 + C2·(tc - t)^m · sin(ω · ln(tc - t))

References
----------
- Sornette (2003) "Why Stock Markets Crash"
- Filimonov & Sornette (2013) Physica A 392, 3698-3707
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Fit result container
# ---------------------------------------------------------------------------
@dataclass
class LPPLFit:
    """Container for a single LPPL fit result."""

    tc: float
    m: float
    omega: float
    A: float
    B: float
    C1: float
    C2: float
    r_squared: float
    rmse: float
    n_obs: int
    converged: bool = True

    @property
    def C(self) -> float:
        return np.sqrt(self.C1**2 + self.C2**2)

    @property
    def phi(self) -> float:
        return np.arctan2(self.C2, self.C1)

    @property
    def is_positive_bubble(self) -> bool:
        return self.B < 0

    @property
    def is_anti_bubble(self) -> bool:
        return self.B > 0

    @property
    def damping(self) -> float:
        if self.C < 1e-10 or self.omega < 1e-10:
            return float("inf")
        return self.m * abs(self.B) / (self.omega * self.C)

    @property
    def osc_ratio(self) -> float:
        if abs(self.B) < 1e-10:
            return float("inf")
        return self.C / abs(self.B)

    def is_valid(self, min_damping: float = 0.5, max_osc_ratio: float = 1.0) -> bool:
        if not (0.01 <= self.m <= 0.99):
            return False
        if not (2.0 <= self.omega <= 25.0):
            return False
        if self.osc_ratio > max_osc_ratio:
            return False
        if self.damping < min_damping:
            return False
        if self.r_squared < 0.5:
            return False
        return True


_FAILED = LPPLFit(
    tc=float("nan"), m=float("nan"), omega=float("nan"),
    A=float("nan"), B=float("nan"), C1=float("nan"), C2=float("nan"),
    r_squared=-1.0, rmse=float("inf"), n_obs=0, converged=False,
)


# ---------------------------------------------------------------------------
# Vectorised OLS for a batch of (tc, m, omega) triplets
# ---------------------------------------------------------------------------
def _batch_ols(
    log_price: np.ndarray,
    t: np.ndarray,
    tc_arr: np.ndarray,
    m_arr: np.ndarray,
    omega_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve OLS for all combinations of (tc, m, omega) at once.

    Parameters
    ----------
    log_price : (N,)
    t : (N,)
    tc_arr : (K_tc,)
    m_arr : (K_m,)
    omega_arr : (K_w,)

    Returns
    -------
    params : (K, 4)  — [A, B, C1, C2] for each of K=K_tc×K_m×K_w combos
    r2     : (K,)    — R² for each combo
    nlp    : (K, 3)  — [tc, m, omega] for each combo
    """
    N = len(log_price)
    y = log_price  # (N,)
    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)

    # Build meshgrid of all combos
    TC, M, W = np.meshgrid(tc_arr, m_arr, omega_arr, indexing="ij")
    K = TC.size
    tc_flat = TC.ravel()   # (K,)
    m_flat = M.ravel()     # (K,)
    w_flat = W.ravel()     # (K,)

    # dt[k, n] = tc[k] - t[n]  → (K, N)
    dt = tc_flat[:, None] - t[None, :]  # (K, N)
    dt = np.maximum(dt, 1e-10)

    dtm = np.power(dt, m_flat[:, None])         # (K, N)
    log_dt = np.log(dt)                          # (K, N)
    omega_log = w_flat[:, None] * log_dt         # (K, N)

    cos_term = dtm * np.cos(omega_log)  # (K, N)
    sin_term = dtm * np.sin(omega_log)  # (K, N)

    # Design matrix: X[k] = [1, dtm, cos, sin]  shape → (K, N, 4)
    ones = np.ones((K, N))
    X = np.stack([ones, dtm, cos_term, sin_term], axis=-1)  # (K, N, 4)

    # Normal equations: (X^T X) beta = X^T y   for each k
    XtX = np.einsum("knp,knq->kpq", X, X)   # (K, 4, 4)
    Xty = np.einsum("knp,n->kp", X, y)       # (K, 4)

    # Solve with regularisation for stability
    reg = 1e-8 * np.eye(4)[None, :, :]  # (1, 4, 4)
    try:
        # solve expects (K,4,4) @ (K,4,1) → (K,4,1)
        params = np.linalg.solve(XtX + reg, Xty[:, :, None]).squeeze(-1)  # (K, 4)
    except np.linalg.LinAlgError:
        params = np.full((K, 4), np.nan)

    # Fitted values and R²
    fitted = np.einsum("knp,kp->kn", X, params)  # (K, N)
    ss_res = np.sum((y[None, :] - fitted) ** 2, axis=1)  # (K,)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else np.full(K, -1.0)

    nlp = np.column_stack([tc_flat, m_flat, w_flat])  # (K, 3)
    return params, r2, nlp


# ---------------------------------------------------------------------------
# Grid search + optional local refinement
# ---------------------------------------------------------------------------
def fit_lppl(
    log_price: np.ndarray,
    t: np.ndarray | None = None,
    *,
    tc_range: tuple[int, int] = (1, 180),
    n_tc: int = 15,
    m_grid: np.ndarray | None = None,
    omega_grid: np.ndarray | None = None,
    bubble_type: str = "positive",
    refine: bool = True,
) -> LPPLFit:
    """Fit LPPL to a log-price series using vectorised Filimonov-Sornette.

    Parameters
    ----------
    log_price : np.ndarray
        Log of close prices.
    t : np.ndarray | None
        Time indices (default: 0..N-1).
    tc_range : (int, int)
        Range of tc offsets from last observation.
    n_tc : int
        Number of tc grid points.
    m_grid, omega_grid : np.ndarray | None
        Custom grids (defaults: 10 points for m, 12 for omega).
    bubble_type : str
        ``"positive"`` (B < 0) or ``"negative"`` (B > 0).
    refine : bool
        Run Nelder-Mead after grid search.

    Returns
    -------
    LPPLFit
    """
    n = len(log_price)
    if n < 20:
        return LPPLFit(**{**_FAILED.__dict__, "n_obs": n})
    if t is None:
        t = np.arange(n, dtype=float)

    if m_grid is None:
        m_grid = np.linspace(0.01, 0.99, 8)
    if omega_grid is None:
        omega_grid = np.linspace(2.0, 25.0, 8)

    t_last = t[-1]
    tc_arr = np.linspace(t_last + tc_range[0], t_last + tc_range[1], n_tc)

    # Vectorised grid search
    params, r2, nlp = _batch_ols(log_price, t, tc_arr, m_grid, omega_grid)

    # Mask by bubble type
    B_vals = params[:, 1]
    if bubble_type == "positive":
        mask = B_vals < 0
    else:
        mask = B_vals > 0

    # Also mask NaN
    mask &= ~np.any(np.isnan(params), axis=1)
    mask &= np.isfinite(r2)

    if not mask.any():
        return LPPLFit(**{**_FAILED.__dict__, "n_obs": n})

    # Best R²
    r2_masked = np.where(mask, r2, -np.inf)
    best_idx = np.argmax(r2_masked)

    tc_best, m_best, omega_best = nlp[best_idx]
    A, B, C1, C2 = params[best_idx]
    best_r2 = r2[best_idx]

    # Local refinement
    if refine:
        tc_best, m_best, omega_best, A, B, C1, C2, best_r2 = _refine(
            log_price, t, tc_best, m_best, omega_best, bubble_type
        )

    rmse = np.sqrt(np.mean((log_price - _eval(t, tc_best, m_best, omega_best, A, B, C1, C2)) ** 2))

    return LPPLFit(
        tc=tc_best, m=m_best, omega=omega_best,
        A=A, B=B, C1=C1, C2=C2,
        r_squared=best_r2, rmse=rmse, n_obs=n,
    )


def _eval(
    t: np.ndarray, tc: float, m: float, omega: float,
    A: float, B: float, C1: float, C2: float,
) -> np.ndarray:
    dt = np.maximum(tc - t, 1e-10)
    dtm = np.power(dt, m)
    log_dt = np.log(dt)
    return A + B * dtm + C1 * dtm * np.cos(omega * log_dt) + C2 * dtm * np.sin(omega * log_dt)


def _refine(
    log_price: np.ndarray, t: np.ndarray,
    tc0: float, m0: float, omega0: float,
    bubble_type: str,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Local Nelder-Mead refinement on (tc, m, omega)."""
    t_last = t[-1]
    y = log_price
    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)

    def _ols_r2(tc: float, m: float, omega: float) -> tuple[float, np.ndarray]:
        dt = np.maximum(tc - t, 1e-10)
        dtm = np.power(dt, m)
        log_dt = np.log(dt)
        X = np.column_stack([
            np.ones_like(t), dtm,
            dtm * np.cos(omega * log_dt),
            dtm * np.sin(omega * log_dt),
        ])
        try:
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return -1.0, np.full(4, np.nan)
        ss_res = np.sum((y - X @ params) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else -1.0
        return r2, params

    def _cost(x: np.ndarray) -> float:
        tc, m, omega = x
        if tc <= t_last or m < 0.01 or m > 0.99 or omega < 2.0 or omega > 25.0:
            return 1e10
        r2, params = _ols_r2(tc, m, omega)
        if np.any(np.isnan(params)):
            return 1e10
        B = params[1]
        if bubble_type == "positive" and B >= 0:
            return 1e10
        if bubble_type == "negative" and B <= 0:
            return 1e10
        return -r2

    res = minimize(
        _cost, [tc0, m0, omega0],
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 0.05, "fatol": 1e-5},
    )

    tc_r, m_r, omega_r = res.x
    r2, params = _ols_r2(tc_r, m_r, omega_r)

    if np.any(np.isnan(params)) or r2 < 0:
        r2_0, params_0 = _ols_r2(tc0, m0, omega0)
        return tc0, m0, omega0, params_0[0], params_0[1], params_0[2], params_0[3], r2_0

    A, B, C1, C2 = params
    return tc_r, m_r, omega_r, A, B, C1, C2, r2


def fit_both(
    log_price: np.ndarray,
    t: np.ndarray | None = None,
    **kwargs,
) -> dict[str, LPPLFit]:
    """Fit both positive-bubble and anti-bubble LPPL."""
    pos = fit_lppl(log_price, t, bubble_type="positive", **kwargs)
    neg = fit_lppl(log_price, t, bubble_type="negative", **kwargs)
    return {"positive": pos, "negative": neg}
