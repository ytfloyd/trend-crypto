"""Causal μ-drivers and vol-expansion event labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Data availability (documented in Stage 1 result card):
# - OIBUILD: no futures OI time series in lake → volume-accumulation proxy only
# - POSEXT: no CFTC/COT loader → omitted (not proxied)
# - GAMMA: no clean dealer-gamma series → omitted


def build_causal_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Append strictly causal feature columns known at each bar's close."""
    out = daily.sort_values("ts").reset_index(drop=True).copy()
    close = pd.to_numeric(out["c"], errors="coerce")
    volume = pd.to_numeric(out["v"], errors="coerce").fillna(0.0)
    log_ret = np.log(close / close.shift(1))

    tf = str(daily.get("timeframe", pd.Series(["1d"])).iloc[-1]).lower()
    if tf == "4h":
        short_lb, rank_lb, ewma_span = 12, 120, 60
    else:
        short_lb, rank_lb, ewma_span = 5, 60, 20

    rv_short = log_ret.rolling(short_lb).std()
    rv_rank = rv_short.rolling(rank_lb, min_periods=rank_lb // 2).apply(
        lambda s: float(pd.Series(s).rank(pct=True).iloc[-1]) if len(s) else np.nan,
        raw=False,
    )
    out["VOLCOMP"] = rv_rank
    out["VOLCOMP_z"] = _zscore(rv_rank, rank_lb)

    vol_ma = volume.rolling(short_lb).mean()
    vol_growth = np.log(vol_ma / vol_ma.shift(short_lb))
    range_pct = (out["h"] - out["l"]) / close
    range_tight = range_pct / range_pct.rolling(rank_lb).median()
    out["OIBUILD_PROXY"] = _zscore(vol_growth - range_tight, rank_lb)

    sigma_ewma = log_ret.pow(2).ewm(span=ewma_span, adjust=False).mean().shift(1).pow(0.5)
    out["sigma_ewma_prior"] = sigma_ewma
    out["log_return"] = log_ret

    return out


def extract_expansion_events(
    frame: pd.DataFrame,
    *,
    kappa: float = 3.0,
) -> pd.DataFrame:
    """Label bars where |r_t| > κ·σ_t with σ known at prior close."""
    out = frame.copy()
    event = out["log_return"].abs() > (kappa * out["sigma_ewma_prior"])
    out["is_event"] = event.fillna(False)
    out["kappa"] = kappa
    return out


def _zscore(series: pd.Series, lookback: int) -> pd.Series:
    mu = series.rolling(lookback, min_periods=lookback // 2).mean()
    sd = series.rolling(lookback, min_periods=lookback // 2).std()
    return (series - mu) / sd.replace(0.0, np.nan)
