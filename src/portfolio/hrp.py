from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


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
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        while sort_ix.max() >= n:
            sort_ix.index = range(sort_ix.shape[0])
            for i, j in sort_ix[sort_ix >= n].items():
                cluster = link[j - n]
                sort_ix[i] = cluster[0]
                sort_ix = pd.concat(
                    [
                        sort_ix.iloc[: i + 1],
                        pd.Series([cluster[1]]),
                        sort_ix.iloc[i + 1 :],
                    ]
                ).reset_index(drop=True)
        return sort_ix.astype(int).tolist()

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
    def allocate(returns: pd.DataFrame) -> pd.Series:
        df = returns.dropna()
        if df.empty:
            raise ValueError("Empty after dropna; cannot allocate.")
        cov = df.cov()
        corr = df.corr()
        diag = np.diag(cov.values)
        if not np.all(np.isfinite(diag)):
            raise ValueError("Covariance diagonal contains non-finite values.")
        link = HierarchicalRiskParity.get_linkage(corr)
        sort_ix = HierarchicalRiskParity.get_quasi_diag(link)
        weights = HierarchicalRiskParity.get_rec_bisection(cov, sort_ix)
        weights = weights / weights.sum()
        return weights
