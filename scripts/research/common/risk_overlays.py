"""
Shared risk overlay utilities.

Used by all paper-recreation and multi-frequency research packages.
Provides portfolio-level risk management functions that operate on
wide-format weight DataFrames (index=ts, columns=symbols).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import simple_backtest
from .data import ANN_FACTOR


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------
def apply_position_limit_wide(
    weights: pd.DataFrame,
    max_wt: float,
) -> pd.DataFrame:
    """Cap individual weights in a wide-format weight matrix.

    Iteratively caps and redistributes excess proportionally until
    no single weight exceeds ``max_wt`` as a fraction of row total.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format: index=ts, columns=symbols, values=target weights.
    max_wt : float
        Maximum allowed weight as a fraction of row total (e.g. 0.15).
    """
    w = weights.copy()
    for _ in range(10):
        row_sum = w.sum(axis=1).replace(0, np.nan)
        pct = w.div(row_sum, axis=0).fillna(0)
        over = pct > max_wt
        if not over.any().any():
            break
        w = w.where(~over, pct.clip(upper=max_wt).mul(row_sum, axis=0))
        new_sum = w.sum(axis=1).replace(0, np.nan)
        scale = (row_sum / new_sum).fillna(1.0)
        w = w.mul(scale, axis=0)
    return w


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------
def apply_vol_targeting(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_target: float = 0.20,
    lookback: int = 42,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale portfolio weights so realized vol ≈ ``vol_target``.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix.
    returns_wide : pd.DataFrame
        Wide-format return matrix (same column universe).
    vol_target : float
        Target annualized volatility (e.g. 0.20 for 20%).
    lookback : int
        Rolling window for realized portfolio vol estimate.
    max_leverage : float
        Cap on the scaling factor to prevent excessive leverage.
    """
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    w_held = w.shift(1).fillna(0.0)
    port_ret = (w_held * r).sum(axis=1)
    realized_vol = (
        port_ret.rolling(lookback, min_periods=max(10, lookback // 2)).std()
        * np.sqrt(ANN_FACTOR)
    )
    scalar = (vol_target / realized_vol).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return w.mul(scalar, axis=0)


# ---------------------------------------------------------------------------
# Drawdown control
# ---------------------------------------------------------------------------
def apply_dd_control(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    dd_threshold: float = 0.30,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Linearly scale down exposure as drawdown approaches threshold.

    At 0% DD: full weight.
    At dd_threshold: 50% weight.
    At 2 × dd_threshold: 0% weight (fully in cash).

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix.
    returns_wide : pd.DataFrame
        Wide-format return matrix.
    dd_threshold : float
        Drawdown level at which exposure is halved (e.g. 0.30 for 30%).
    cost_bps : float
        Transaction cost for the preliminary backtest (for DD calc).
    """
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    bt = simple_backtest(w, r, cost_bps=cost_bps)
    bt["ts"] = pd.to_datetime(bt["ts"])
    eq = bt.set_index("ts")["portfolio_equity"]
    dd = eq / eq.cummax() - 1.0
    scale = (1.0 + dd / (2.0 * dd_threshold)).clip(0.0, 1.0)
    return w.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# Trailing stop-loss
# ---------------------------------------------------------------------------
def apply_trailing_stop(
    base_weights: pd.DataFrame,
    close_wide: pd.DataFrame,
    stop_pct: float,
) -> pd.DataFrame:
    """Zero out asset weight when price drops >stop_pct from peak while held.

    After being stopped out, the asset can re-enter at the next non-zero
    base weight (i.e., at the next rebalance that selects it).

    Parameters
    ----------
    base_weights : pd.DataFrame
        Wide-format weight matrix (pre-overlay).
    close_wide : pd.DataFrame
        Wide-format close price matrix.
    stop_pct : float
        Stop-loss trigger as a fraction (e.g. 0.15 for 15%).
    """
    w = base_weights.copy()
    common = w.columns.intersection(close_wide.columns)
    w = w[common]
    cls = close_wide[common].reindex(w.index).ffill()

    stopped: dict[str, bool] = {c: False for c in common}
    peak: dict[str, float] = {c: 0.0 for c in common}
    sym_to_col = {s: j for j, s in enumerate(w.columns)}

    for i, dt in enumerate(w.index):
        for sym in common:
            col_idx = sym_to_col[sym]
            base_wt = base_weights.at[dt, sym] if sym in base_weights.columns else 0.0
            price = cls.iloc[i, col_idx] if not np.isnan(cls.iloc[i, col_idx]) else 0.0

            if base_wt > 0 and not stopped[sym]:
                peak[sym] = max(peak[sym], price) if peak[sym] > 0 else price
                if peak[sym] > 0 and price < peak[sym] * (1 - stop_pct):
                    stopped[sym] = True
                    w.iloc[i, col_idx] = 0.0
            elif base_wt > 0 and stopped[sym]:
                if i > 0:
                    prev_base = (
                        base_weights.at[w.index[i - 1], sym]
                        if sym in base_weights.columns else 0.0
                    )
                    if prev_base == 0.0 or base_wt != prev_base:
                        stopped[sym] = False
                        peak[sym] = price
                    else:
                        w.iloc[i, col_idx] = 0.0
                else:
                    w.iloc[i, col_idx] = 0.0
            else:
                stopped[sym] = False
                peak[sym] = 0.0

    # Re-normalize so weights sum to original exposure on each day
    orig_sum = base_weights[common].sum(axis=1).replace(0, np.nan)
    new_sum = w.sum(axis=1).replace(0, np.nan)
    scale = (orig_sum / new_sum).fillna(0.0).clip(upper=2.0)
    w = w.multiply(scale, axis=0)
    return w


# ---------------------------------------------------------------------------
# Predicted-vol concentration
# ---------------------------------------------------------------------------
def apply_vol_concentration(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    high_wide: pd.DataFrame | None = None,
    low_wide: pd.DataFrame | None = None,
    open_wide: pd.DataFrame | None = None,
    vol_windows: tuple[int, ...] = (5, 10, 20),
    train_window: int = 252,
    refit_every: int = 21,
    floor: float = 0.10,
) -> pd.DataFrame:
    """Tilt portfolio weights toward assets with the highest predicted vol.

    Uses a lightweight cross-sectional vol-prediction model (Ridge regression
    on rolling realized-vol features) to rank assets by expected next-day
    absolute return.  Weights are multiplied by the predicted-vol percentile
    rank, then rescaled to preserve gross exposure.

    The prediction model achieves ~0.21 cross-sectional Spearman IC in crypto
    (see notebook 09).  The overlay does not change trade direction — it only
    concentrates capital where moves are expected to be largest.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix (index=ts, columns=symbols).
    returns_wide : pd.DataFrame
        Wide-format return matrix.
    close_wide, high_wide, low_wide, open_wide : pd.DataFrame
        Wide-format price matrices.  If high/low/open are provided,
        Garman-Klass vol is added as a feature (more efficient estimator).
    vol_windows : tuple[int, ...]
        Rolling windows for close-to-close realized vol features.
    train_window : int
        Number of days of trailing data for fitting the Ridge model.
    refit_every : int
        Re-estimate model coefficients every N days.
    floor : float
        Minimum multiplier — prevents zeroing out positions entirely.
        0.10 means the least-volatile name keeps at least 10% of its
        original weight.
    """
    from sklearn.linear_model import Ridge

    common = sorted(weights.columns.intersection(returns_wide.columns)
                    .intersection(close_wide.columns))
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index)
    c = close_wide[common].reindex(w.index)

    log_ret = np.log(c / c.shift(1))
    abs_ret_1d = log_ret.abs()
    target = abs_ret_1d.shift(-1)

    # Features per symbol: multi-window realized vol + extras
    feat_frames = {}
    for win in vol_windows:
        feat_frames[f"rvol_{win}"] = log_ret.rolling(win).std()

    if vol_windows:
        short_w, long_w = min(vol_windows), max(vol_windows)
        feat_frames["vol_compress"] = (
            feat_frames[f"rvol_{short_w}"]
            / feat_frames[f"rvol_{long_w}"].replace(0, np.nan)
        )
        feat_frames["vov"] = feat_frames[f"rvol_{short_w}"].rolling(20).std()

    feat_frames["abs_ret_5d"] = abs_ret_1d.rolling(5).mean()

    if (high_wide is not None and low_wide is not None
            and open_wide is not None):
        h = high_wide[common].reindex(w.index)
        l = low_wide[common].reindex(w.index)
        o = open_wide[common].reindex(w.index)
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        feat_frames["gk_vol"] = np.sqrt(gk_var.clip(lower=0).rolling(10).mean() * ANN_FACTOR)
        feat_frames["range_5d"] = log_hl.rolling(5).mean()

    # BTC vol as a market-wide feature (broadcast)
    btc_col = "BTC-USD" if "BTC-USD" in common else common[0]
    btc_vol = log_ret[btc_col].rolling(10).std()

    # Rolling prediction per symbol
    pred_vol = pd.DataFrame(np.nan, index=w.index, columns=common)

    for sym in common:
        feat_cols = {}
        for fname, fdf in feat_frames.items():
            if sym in fdf.columns:
                feat_cols[fname] = fdf[sym]
        feat_cols["btc_vol"] = btc_vol

        feat_df = pd.DataFrame(feat_cols)
        tgt = target[sym] if sym in target.columns else None
        if tgt is None:
            continue

        joined = feat_df.join(tgt.rename("target")).dropna()
        if len(joined) < train_window + 60:
            continue

        X_all = joined.drop("target", axis=1)
        y_all = joined["target"]

        coefs = None
        for i_dt, dt in enumerate(w.index):
            if dt not in X_all.index:
                continue
            loc = X_all.index.get_loc(dt)
            if loc < train_window:
                continue

            if i_dt % refit_every == 0 or coefs is None:
                X_train = X_all.iloc[loc - train_window:loc]
                y_train = y_all.iloc[loc - train_window:loc]
                mu = X_train.mean()
                sd = X_train.std().replace(0, 1)
                X_norm = (X_train - mu) / sd
                model = Ridge(alpha=1.0)
                model.fit(X_norm, y_train)
                coefs = model.coef_
                intercept = model.intercept_
                feat_mu, feat_sd = mu, sd

            x_today = (X_all.loc[dt] - feat_mu) / feat_sd
            pred_vol.loc[dt, sym] = max(intercept + np.dot(coefs, x_today.values), 0)

    # Rank cross-sectionally and use as multiplier
    rank = pred_vol.rank(axis=1, pct=True).clip(lower=floor)

    w_tilted = w * rank.reindex_like(w).fillna(1.0)

    # Rescale to preserve original gross exposure
    orig_gross = w.abs().sum(axis=1)
    new_gross = w_tilted.abs().sum(axis=1).clip(lower=1e-10)
    scale = orig_gross / new_gross
    w_tilted = w_tilted.mul(scale, axis=0).fillna(0.0)

    return w_tilted
