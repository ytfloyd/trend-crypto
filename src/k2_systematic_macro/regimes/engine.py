"""First-pass unsupervised regime engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for latent-state research labels."""

    n_states: int = 3
    random_state: int = 42
    min_samples: int = 30
    feature_columns: tuple[str, ...] = (
        "realized_vol_12",
        "realized_vol_24",
        "atr_compression",
        "range_compression",
        "trend_persistence",
        "trend_return_atr",
        "breakout_channel_width_atr",
    )


def fit_regimes(frame: pd.DataFrame, config: RegimeConfig | None = None) -> pd.DataFrame:
    """Assign unsupervised regimes to completed-bar features.

    The preferred implementation is sklearn ``GaussianMixture``. If sklearn is
    unavailable, the deterministic fallback buckets volatility/compression/trend
    scores into quantile regimes and records that limitation in ``regime_method``.
    """
    cfg = config or RegimeConfig()
    out = frame.sort_values("ts").reset_index(drop=True).copy()
    out["regime"] = np.nan
    out["regime_probability"] = np.nan
    out["regime_method"] = "unassigned"

    feature_columns = [col for col in cfg.feature_columns if col in out]
    if not feature_columns:
        return out
    matrix = out[feature_columns].apply(pd.to_numeric, errors="coerce")
    valid = matrix.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.shape[0] < cfg.min_samples:
        return out

    states = min(cfg.n_states, valid.shape[0])
    try:
        labels, probabilities, metadata = _fit_gmm(valid, states, cfg.random_state)
    except Exception as exc:  # pragma: no cover - exercised only without sklearn/runtime support
        labels, probabilities, metadata = _quantile_fallback(valid, states)
        metadata["fallback_reason"] = str(exc)

    ordered_labels = _order_regimes_by_volatility(labels, valid, metadata.get("vol_col"))
    out.loc[valid.index, "regime"] = ordered_labels.astype(float)
    out.loc[valid.index, "regime_probability"] = probabilities
    out.loc[valid.index, "regime_method"] = metadata["method"]
    out.attrs["regime_metadata"] = {
        "method": metadata["method"],
        "feature_columns": feature_columns,
        "n_states": states,
        **{k: v for k, v in metadata.items() if k not in {"method", "vol_col"}},
    }
    return out


def _fit_gmm(
    valid: pd.DataFrame,
    n_states: int,
    random_state: int,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x = scaler.fit_transform(valid)
    model = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
    )
    labels = pd.Series(model.fit_predict(x), index=valid.index)
    probabilities = pd.Series(model.predict_proba(x).max(axis=1), index=valid.index)
    return labels, probabilities, {"method": "gaussian_mixture", "vol_col": _vol_col(valid)}


def _quantile_fallback(valid: pd.DataFrame, n_states: int) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    vol_col = _vol_col(valid)
    compression_col = "atr_compression" if "atr_compression" in valid else valid.columns[0]
    trend_col = "trend_persistence" if "trend_persistence" in valid else valid.columns[0]
    score = (
        valid[vol_col].rank(pct=True)
        + (1.0 - valid[compression_col].rank(pct=True))
        + valid[trend_col].rank(pct=True)
    )
    labels = pd.qcut(score.rank(method="first"), q=n_states, labels=False, duplicates="drop")
    labels = pd.Series(labels, index=valid.index).astype(float)
    probabilities = pd.Series(1.0, index=valid.index)
    return labels, probabilities, {"method": "quantile_fallback", "vol_col": vol_col}


def _order_regimes_by_volatility(
    labels: pd.Series,
    valid: pd.DataFrame,
    vol_col: str | None,
) -> pd.Series:
    if labels.empty:
        return labels
    ranking_col = vol_col or valid.columns[0]
    ranking = valid.assign(_label=labels).groupby("_label")[ranking_col].median().sort_values()
    mapper = {old_label: new_label for new_label, old_label in enumerate(ranking.index)}
    return labels.map(mapper).astype(int)


def _vol_col(valid: pd.DataFrame) -> str:
    for col in ("realized_vol_24", "realized_vol_12", "realized_vol_60"):
        if col in valid:
            return col
    return valid.columns[0]
