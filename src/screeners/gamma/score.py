"""Cross-sectional scoring for the gamma screener.

Takes a list of FeatureRow and produces a list of ScoredRow, with:
    - Hard filters applied (liquidity, quality, bounds)
    - Three sub-scores (short, thirty, term) as cross-sectional z-scores of
      the component signals, oriented so that HIGHER = more underpriced gamma.
    - An earnings penalty.
    - A combined score and dense integer rank.

Sign conventions (higher score = more underpriced gamma):
    score_short      : z(-iv7_rv10_ratio)               # IV cheap vs RV = good
    score_thirty     : z(-iv30_rv20_ratio) + z(-iv_rank_252) / 2
    score_term       : z(iv30 - iv90)                   # contango = front cheap
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import numpy as np

from common.logging import get_logger

from .config import GammaScreenerConfig
from .signals import FeatureRow

logger = get_logger("gamma_screener_score")


@dataclass(frozen=True)
class ScoredRow:
    as_of_date: date
    symbol: str
    spot: float
    iv7: Optional[float]
    iv30: Optional[float]
    iv60: Optional[float]
    iv90: Optional[float]
    rv_cc10: Optional[float]
    rv_cc20: Optional[float]
    rv_yz20: Optional[float]
    rv_yz60: Optional[float]
    iv30_rv20_ratio: Optional[float]
    iv7_rv10_ratio: Optional[float]
    term_30_90: Optional[float]
    term_7_30: Optional[float]
    skew_25d_30: Optional[float]
    butterfly_25d_30: Optional[float]
    iv_rank_252: Optional[float]
    bid_ask_pct: Optional[float]
    stock_adv_usd: Optional[float]
    options_adv_usd: Optional[float]
    earnings_in_window: bool
    score_short: Optional[float]
    score_thirty: Optional[float]
    score_term: Optional[float]
    score_combined: Optional[float]
    rank_combined: Optional[int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _zscore(vals: np.ndarray) -> np.ndarray:
    """Robust z-score that handles NaN by returning NaN in the same slot."""
    mask = np.isfinite(vals)
    if mask.sum() < 3:
        return np.full_like(vals, np.nan, dtype=float)
    mu = float(np.mean(vals[mask]))
    sd = float(np.std(vals[mask], ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return np.full_like(vals, 0.0, dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
    out[mask] = (vals[mask] - mu) / sd
    return out


def _passes_filters(row: FeatureRow, cfg: GammaScreenerConfig) -> bool:
    if row.spot < cfg.min_spot_price:
        return False
    if row.iv30 is None or not (cfg.min_iv <= row.iv30 <= cfg.max_iv):
        return False
    if row.bid_ask_pct is not None and row.bid_ask_pct > cfg.max_bid_ask_pct:
        return False
    if row.stock_adv_usd is not None and row.stock_adv_usd < cfg.min_stock_adv_usd:
        return False
    # Need at least the 30d IV/RV ratio to score
    if row.iv30_rv20_ratio is None:
        return False
    return True


def rank_universe(
    features: list[FeatureRow],
    cfg: GammaScreenerConfig,
) -> list[ScoredRow]:
    """Apply filters, compute cross-sectional scores, and rank.

    Rows that fail filters are NOT dropped — they are returned with
    all score fields set to None and rank set to None. This keeps
    the downstream DB table complete for reproducibility.
    """
    if not features:
        return []

    eligible_idx = [i for i, r in enumerate(features) if _passes_filters(r, cfg)]
    if not eligible_idx:
        logger.warning("No symbols passed filters")
        return [_row_with_no_score(r) for r in features]

    def _col(attr: str) -> np.ndarray:
        out = np.array([
            getattr(features[i], attr) if getattr(features[i], attr) is not None else np.nan
            for i in eligible_idx
        ], dtype=float)
        return out

    iv7_rv10 = _col("iv7_rv10_ratio")
    iv30_rv20 = _col("iv30_rv20_ratio")
    term_30_90 = _col("term_30_90")
    iv_rank = _col("iv_rank_252")

    # All metrics are signed so HIGHER = cheaper gamma.
    # IV/RV ratios: lower = cheaper. Negate so higher = cheaper.
    z_short = _zscore(-iv7_rv10)
    z_thirty_rv = _zscore(-iv30_rv20)

    # IV rank: lower percentile = cheaper IV historically. Negate.
    if np.isfinite(iv_rank).sum() >= 3:
        z_rank = _zscore(-iv_rank)
        z_thirty = np.where(
            np.isfinite(z_rank),
            0.67 * z_thirty_rv + 0.33 * z_rank,
            z_thirty_rv,
        )
    else:
        z_thirty = z_thirty_rv  # not enough history yet

    # Term: iv30 - iv90 > 0 = contango. Higher spread = front richer, NOT what we want.
    # For "cheap gamma" we want BACKWARDATION at the short end (front < back), so
    # term_30_90 should be NEGATIVE = front cheap. Negate to score.
    z_term = _zscore(-term_30_90)

    earnings = np.array(
        [1.0 if features[i].earnings_in_window else 0.0 for i in eligible_idx]
    )

    combined = (
        cfg.weight_short * np.nan_to_num(z_short, nan=0.0)
        + cfg.weight_thirty * np.nan_to_num(z_thirty, nan=0.0)
        + cfg.weight_term * np.nan_to_num(z_term, nan=0.0)
        - cfg.weight_earnings_penalty * earnings
    )

    # Sort descending, assign dense 1-based ranks
    order = np.argsort(-combined, kind="stable")
    rank_of = np.empty_like(order)
    for rnk, pos in enumerate(order, start=1):
        rank_of[pos] = rnk

    scored_eligible: dict[int, ScoredRow] = {}
    for k, i in enumerate(eligible_idx):
        r = features[i]
        scored_eligible[i] = ScoredRow(
            **_feature_fields(r),
            score_short=_f(z_short[k]),
            score_thirty=_f(z_thirty[k]),
            score_term=_f(z_term[k]),
            score_combined=float(combined[k]),
            rank_combined=int(rank_of[k]),
        )

    # Preserve input order; eligible rows get scores, ineligible ones get Nones
    out: list[ScoredRow] = []
    for i, r in enumerate(features):
        if i in scored_eligible:
            out.append(scored_eligible[i])
        else:
            out.append(_row_with_no_score(r))
    return out


def _feature_fields(r: FeatureRow) -> dict[str, object]:
    """FeatureRow → dict keyed for ScoredRow constructor (all non-score fields)."""
    fields = {
        "as_of_date": r.as_of_date,
        "symbol": r.symbol,
        "spot": r.spot,
        "iv7": r.iv7, "iv30": r.iv30, "iv60": r.iv60, "iv90": r.iv90,
        "rv_cc10": r.rv_cc10, "rv_cc20": r.rv_cc20,
        "rv_yz20": r.rv_yz20, "rv_yz60": r.rv_yz60,
        "iv30_rv20_ratio": r.iv30_rv20_ratio,
        "iv7_rv10_ratio": r.iv7_rv10_ratio,
        "term_30_90": r.term_30_90,
        "term_7_30": r.term_7_30,
        "skew_25d_30": r.skew_25d_30,
        "butterfly_25d_30": r.butterfly_25d_30,
        "iv_rank_252": r.iv_rank_252,
        "bid_ask_pct": r.bid_ask_pct,
        "stock_adv_usd": r.stock_adv_usd,
        "options_adv_usd": r.options_adv_usd,
        "earnings_in_window": r.earnings_in_window,
    }
    return fields


def _row_with_no_score(r: FeatureRow) -> ScoredRow:
    return ScoredRow(
        **_feature_fields(r),
        score_short=None, score_thirty=None, score_term=None,
        score_combined=None, rank_combined=None,
    )


def _f(x: float) -> Optional[float]:
    """Turn NaN into None so Polars / DuckDB treat it as NULL."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x)
