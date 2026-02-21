from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CovarianceMethod = Literal["sample", "ledoit_wolf"]


@dataclass(frozen=True)
class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) allocator.

    Reference:
        López de Prado, M. (2016) "Building Diversified Portfolios that Outperform"
    """

    @staticmethod
    def get_linkage(corr: pd.DataFrame) -> np.ndarray:
        try:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform
        except ImportError as exc:
            raise ImportError(
                "scipy is required for HRP linkage; install scipy to use this allocator."
            ) from exc
        corr = corr.clip(-1.0, 1.0)
        dist = np.sqrt(0.5 * (1.0 - corr))
        dist = dist.clip(lower=0.0, upper=1.0)
        dist_condensed = squareform(dist.values, checks=False)
        return linkage(dist_condensed, method="ward")

    @staticmethod
    def get_quasi_diag(link: np.ndarray) -> list[int]:
        link = link.astype(int)
        n = link.shape[0] + 1
        root = 2 * n - 2

        def _recurse(node: int) -> list[int]:
            if node < n:
                return [node]
            row = link[node - n]
            return _recurse(int(row[0])) + _recurse(int(row[1]))

        return _recurse(root)

    @staticmethod
    def get_cluster_var(cov: pd.DataFrame, cluster_items: Iterable[str]) -> float:
        cov_slice = cov.loc[cluster_items, cluster_items]
        diag = np.diag(cov_slice.values)
        diag = np.where(np.isfinite(diag), diag, 0.0)
        inv_diag = np.where(diag > 0, 1.0 / diag, 0.0)
        if inv_diag.sum() == 0:
            weights = np.ones_like(inv_diag) / len(inv_diag)
        else:
            weights = inv_diag / inv_diag.sum()
        return float(weights @ cov_slice.values @ weights)

    @staticmethod
    def get_rec_bisection(cov: pd.DataFrame, sort_ix: list[int]) -> pd.Series:
        items = cov.index[sort_ix].tolist()
        weights = pd.Series(1.0, index=items)
        clusters = [items]
        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                split = len(cluster) // 2
                left = cluster[:split]
                right = cluster[split:]
                var_left = HierarchicalRiskParity.get_cluster_var(cov, left)
                var_right = HierarchicalRiskParity.get_cluster_var(cov, right)
                denom = var_left + var_right
                if denom <= 0:
                    alpha = 0.5
                else:
                    alpha = 1.0 - var_left / denom
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha
                new_clusters.extend([left, right])
            clusters = new_clusters
        return weights

    @staticmethod
    def allocate(
        returns: pd.DataFrame,
        covariance_method: CovarianceMethod = "sample",
    ) -> pd.Series:
        """Compute HRP weights from a returns matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            Wide-format returns with symbols as columns.
        covariance_method : ``"sample"`` | ``"ledoit_wolf"``
            Covariance estimation method.  ``"sample"`` uses the standard
            sample covariance (existing behaviour, default).
            ``"ledoit_wolf"`` applies Ledoit-Wolf shrinkage via
            ``sklearn.covariance.LedoitWolf``.
        """
        df = returns.dropna()
        if df.empty:
            raise ValueError("Empty after dropna; cannot allocate.")

        cov, corr = _estimate_covariance(df, covariance_method)

        diag = np.diag(cov.values)
        if not np.all(np.isfinite(diag)):
            raise ValueError("Covariance diagonal contains non-finite values.")
        link = HierarchicalRiskParity.get_linkage(corr)
        sort_ix = HierarchicalRiskParity.get_quasi_diag(link)
        weights = HierarchicalRiskParity.get_rec_bisection(cov, sort_ix)
        weights = weights / weights.sum()
        return weights


def _estimate_covariance(
    df: pd.DataFrame,
    method: CovarianceMethod,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate covariance and correlation matrices.

    Returns (cov, corr) DataFrames indexed/columned by symbol names.
    """
    if method == "sample":
        return df.cov(), df.corr()

    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for Ledoit-Wolf covariance; "
                "install scikit-learn to use covariance_method='ledoit_wolf'."
            ) from exc

        lw = LedoitWolf().fit(df.values)
        cov = pd.DataFrame(lw.covariance_, index=df.columns, columns=df.columns)
        logger.info("Ledoit-Wolf shrinkage coefficient: %.4f", lw.shrinkage_)

        std = np.sqrt(np.diag(lw.covariance_))
        safe_std = np.where(std > 1e-12, std, 1.0)
        outer_std = np.outer(safe_std, safe_std)
        corr_vals = lw.covariance_ / outer_std
        np.fill_diagonal(corr_vals, 1.0)
        corr = pd.DataFrame(corr_vals, index=df.columns, columns=df.columns)

        return cov, corr

    raise ValueError(f"Unknown covariance_method: {method!r}")
