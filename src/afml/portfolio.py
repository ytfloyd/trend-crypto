"""
Hierarchical Risk Parity (HRP) portfolio construction.

Implements the HRP algorithm from AFML Chapter 16.  HRP is a
tree-based portfolio optimisation that:
  1. Clusters assets by correlation distance
  2. Quasi-diagonalises the covariance matrix
  3. Allocates risk top-down through the dendrogram

Unlike mean-variance (Markowitz), HRP doesn't require inverting the
covariance matrix, making it more stable and better suited to
large cross-sections of correlated assets.

Reference:
    López de Prado, M. (2016) "Building Diversified Portfolios that
    Outperform Out-of-Sample", J. Portfolio Management.
    AFML Chapter 16, pp. 225–237.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# =====================================================================
# Correlation distance
# =====================================================================

def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix to a distance matrix.

    ``d(i,j) = sqrt(0.5 * (1 - ρ(i,j)))``

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix.

    Returns
    -------
    pd.DataFrame — distance matrix with same index/columns.
    """
    dist = ((1 - corr) / 2).clip(lower=0) ** 0.5
    return dist


# =====================================================================
# Quasi-diagonalisation  (AFML Snippet 16.2)
# =====================================================================

def quasi_diagonalise(link: np.ndarray) -> list[int]:
    """Return the order of assets from the hierarchical clustering.

    This is the order that quasi-diagonalises the covariance matrix
    (similar assets are adjacent).
    """
    return list(leaves_list(link))


# =====================================================================
# Recursive bisection  (AFML Snippet 16.3)
# =====================================================================

def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Inverse-variance portfolio variance for a cluster of assets."""
    cov_slice = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(cov_slice)
    ivp /= ivp.sum()
    return float(ivp @ cov_slice @ ivp)


def recursive_bisection(
    cov: np.ndarray,
    sorted_idx: list[int],
) -> np.ndarray:
    """Top-down allocation through the dendrogram.

    Splits the sorted asset list in half, allocates between the two
    halves inversely proportional to their cluster variance, and
    recurses.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    sorted_idx : list[int]
        Asset order from ``quasi_diagonalise()``.

    Returns
    -------
    np.ndarray — weights for each asset (in original order).
    """
    n = cov.shape[0]
    weights = np.ones(n)

    # BFS-style bisection
    clusters = [sorted_idx]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            total = var_left + var_right

            alpha = 1 - var_left / total if total > 0 else 0.5

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1 - alpha)

            new_clusters.append(left)
            new_clusters.append(right)

        clusters = new_clusters

    return weights


# =====================================================================
# Main HRP function
# =====================================================================

def hrp_weights(
    returns: pd.DataFrame,
    method: str = "single",
) -> pd.Series:
    """Compute HRP portfolio weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (columns = assets, rows = time).
    method : str
        Linkage method for hierarchical clustering
        (``"single"``, ``"complete"``, ``"average"``, ``"ward"``).

    Returns
    -------
    pd.Series — portfolio weights indexed by asset name, summing to 1.
    """
    cov = returns.cov().values
    corr = returns.corr()

    dist = correlation_distance(corr)
    dist_condensed = squareform(dist.values, checks=False)
    link = linkage(dist_condensed, method=method)

    sorted_idx = quasi_diagonalise(link)
    weights = recursive_bisection(cov, sorted_idx)
    weights /= weights.sum()

    return pd.Series(weights, index=returns.columns, name="weight")


# =====================================================================
# Inverse-variance (for comparison)
# =====================================================================

def inverse_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """Simple 1/σ² portfolio (diagonal covariance assumption)."""
    var = returns.var()
    w = 1.0 / var
    w /= w.sum()
    return w.rename("weight")


# =====================================================================
# Equal-weight (for comparison)
# =====================================================================

def equal_weights(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio."""
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns, name="weight")
