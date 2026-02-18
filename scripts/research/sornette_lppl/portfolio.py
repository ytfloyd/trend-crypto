"""
Portfolio construction for the "Jumpers" strategy.

Builds a long-only portfolio of crypto assets exhibiting explosive
upside signatures detected by the LPPL bubble indicator.

Strategy logic
--------------
1. Every ``rebalance_every`` days, re-rank all assets by LPPL signal.
2. Go long top-K with strongest signal (bubble_rider or antibubble_reversal).
3. Weight by signal strength × inverse-volatility (optional).
4. Apply volatility-target overlay to the portfolio.
5. Hold until next rebalance (or emergency exit if signal collapses mid-period).

The backtest uses close-to-close returns and accounts for transaction costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_portfolio_weights(
    signals_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    top_k: int = 10,
    rebalance_every: int = 5,
    min_signal: float = 0.05,
    ivol_weight: bool = True,
    vol_lookback: int = 20,
    regime_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Construct time-series of portfolio weights.

    Parameters
    ----------
    signals_panel : pd.DataFrame
        Output of ``compute_signals``, long-format with columns
        [symbol, ts, signal, signal_type].
    returns_wide : pd.DataFrame
        Wide-format daily returns (index=ts, columns=symbols).
    top_k : int
        Maximum number of holdings.
    rebalance_every : int
        Rebalance period in trading days.
    min_signal : float
        Minimum signal threshold.
    ivol_weight : bool
        If True, weight = signal × inverse-vol; else signal only.
    vol_lookback : int
        Lookback for volatility estimation (days).
    regime_mask : pd.Series | None
        Boolean series (index=date). When False, go to cash.

    Returns
    -------
    pd.DataFrame
        Wide-format weights (index=ts, columns=symbols).
    """
    all_dates = returns_wide.index.sort_values()
    sig_dates = sorted(signals_panel["ts"].unique())

    # Pivot signals to wide: date x symbol → signal
    sig_wide = signals_panel.pivot_table(
        index="ts", columns="symbol", values="signal", aggfunc="last"
    ).reindex(all_dates).ffill()

    # Realised vol
    vol = returns_wide.rolling(vol_lookback, min_periods=vol_lookback // 2).std()

    weights = pd.DataFrame(0.0, index=all_dates, columns=returns_wide.columns)
    last_w = pd.Series(0.0, index=returns_wide.columns)
    day_counter = 0

    for dt in all_dates:
        day_counter += 1

        # Regime filter: go to cash in bear markets
        if regime_mask is not None and dt in regime_mask.index and not regime_mask.loc[dt]:
            last_w[:] = 0.0
            weights.loc[dt] = 0.0
            continue

        if day_counter % rebalance_every == 1 or rebalance_every == 1:
            # Re-rank
            if dt not in sig_wide.index:
                weights.loc[dt] = last_w
                continue

            s = sig_wide.loc[dt].dropna()
            s = s[s > min_signal].sort_values(ascending=False).head(top_k)

            if len(s) == 0:
                last_w[:] = 0.0
                weights.loc[dt] = 0.0
                continue

            if ivol_weight and dt in vol.index:
                v = vol.loc[dt].reindex(s.index).fillna(vol.loc[dt].median())
                v = v.clip(lower=1e-6)
                raw_w = s * (1.0 / v)
            else:
                raw_w = s.copy()

            raw_w = raw_w / raw_w.sum()
            last_w = pd.Series(0.0, index=returns_wide.columns)
            last_w.update(raw_w)

        weights.loc[dt] = last_w

    return weights


def backtest_portfolio(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_target: float | None = 0.40,
    vol_lookback: int = 20,
    tc_bps: float = 20.0,
    cash_rate: float = 0.04,
) -> pd.DataFrame:
    """Backtest the jumpers portfolio.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weights (same index/columns as returns_wide).
    returns_wide : pd.DataFrame
        Daily asset returns.
    vol_target : float | None
        Annualised vol target for the portfolio (None = no overlay).
    vol_lookback : int
        Rolling vol window.
    tc_bps : float
        One-way transaction cost in basis points.
    cash_rate : float
        Annual cash rate (earned on un-invested capital).

    Returns
    -------
    pd.DataFrame
        Columns: [date, gross_ret, net_ret, turnover, leverage,
                  n_holdings, cum_ret, equity].
    """
    ann = 365.0
    tc = tc_bps / 10_000.0
    daily_cash = (1 + cash_rate) ** (1 / ann) - 1

    aligned_idx = weights.index.intersection(returns_wide.index)
    W = weights.loc[aligned_idx]
    R = returns_wide.loc[aligned_idx]

    records = []
    prev_w = pd.Series(0.0, index=W.columns)
    port_rets = []

    for dt in aligned_idx:
        w = W.loc[dt]
        r = R.loc[dt].fillna(0.0)

        # Turnover
        delta = (w - prev_w).abs().sum() / 2.0
        cost = delta * tc

        # Gross portfolio return
        gross_ret = (w * r).sum()

        # Vol-targeting overlay
        leverage = 1.0
        if vol_target is not None and len(port_rets) >= vol_lookback:
            realised_vol = np.std(port_rets[-vol_lookback:]) * np.sqrt(ann)
            if realised_vol > 1e-6:
                leverage = min(vol_target / realised_vol, 2.0)

        # Net return
        cash_frac = max(1.0 - w.abs().sum(), 0.0)
        net_ret = leverage * gross_ret - cost + cash_frac * daily_cash

        port_rets.append(net_ret)
        records.append({
            "date": dt,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": delta,
            "leverage": leverage,
            "n_holdings": (w.abs() > 1e-6).sum(),
        })

        # Drift weights by returns for next period's turnover calc
        drifted = w * (1 + r)
        total = drifted.sum()
        prev_w = drifted / total if abs(total) > 1e-10 else drifted

    result = pd.DataFrame(records)
    if not result.empty:
        result["cum_ret"] = (1 + result["net_ret"]).cumprod()
        result["equity"] = result["cum_ret"]
    return result


def performance_summary(bt: pd.DataFrame, ann_factor: float = 365.0) -> dict:
    """Compute summary statistics from backtest results."""
    if bt.empty or len(bt) < 10:
        return {"error": "insufficient data"}

    total_days = len(bt)
    total_years = total_days / ann_factor
    cum = bt["cum_ret"].iloc[-1]
    cagr = cum ** (1 / total_years) - 1 if total_years > 0 else 0.0

    daily_rets = bt["net_ret"]
    vol = daily_rets.std() * np.sqrt(ann_factor)
    sharpe = cagr / vol if vol > 1e-6 else 0.0

    cum_series = bt["cum_ret"]
    drawdowns = cum_series / cum_series.cummax() - 1
    max_dd = drawdowns.min()

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    avg_holdings = bt["n_holdings"].mean()
    avg_turnover = bt["turnover"].mean()
    avg_leverage = bt["leverage"].mean()

    return {
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "total_return": cum - 1,
        "avg_holdings": avg_holdings,
        "avg_daily_turnover": avg_turnover,
        "avg_leverage": avg_leverage,
        "n_days": total_days,
    }
