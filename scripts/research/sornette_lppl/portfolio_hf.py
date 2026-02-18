"""
Hourly portfolio construction with tc-based exit timing.

Unlike the daily portfolio which rebalances on a fixed schedule,
the hourly portfolio uses event-driven entry/exit:

  ENTRY:  Fast layer triggers + minimum signal threshold
  EXIT:   (a) LPPLS tc < exit_horizon hours, OR
          (b) trailing stop triggered, OR
          (c) max holding period exceeded, OR
          (d) signal dropped below threshold at next rebalance

This is where LPPLS adds genuine value over a pure convexity screen:
the tc estimate provides a principled exit before the crash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_hourly_portfolio(
    signals: pd.DataFrame,
    returns_wide: pd.DataFrame,
    *,
    top_k: int = 10,
    rebalance_every_hours: int = 6,
    min_signal: float = 0.05,
    tc_exit_hours: float = 4.0,
    max_hold_hours: int = 168,
    trailing_stop_pct: float = 0.15,
    ivol_weight: bool = True,
    vol_lookback: int = 48,
    regime_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build hourly portfolio weights with tc-based exits.

    Parameters
    ----------
    signals : pd.DataFrame
        Output of scan_rolling with columns [symbol, ts, se_score,
        lppl_confirmed, tc_hours, bubble_conf].
    returns_wide : pd.DataFrame
        Hourly returns (index=ts, columns=symbols).
    top_k : int
        Max concurrent holdings.
    rebalance_every_hours : int
        Rebalance (re-rank) interval.
    min_signal : float
        Minimum se_score to enter.
    tc_exit_hours : float
        Exit when LPPLS tc estimate < this many hours.
    max_hold_hours : int
        Maximum holding period (force exit).
    trailing_stop_pct : float
        Exit if position drops this much from peak.
    ivol_weight : bool
        Weight by signal × inverse-vol.
    vol_lookback : int
        Hours for vol estimation.
    regime_mask : pd.Series | None
        Boolean (index=ts). False = cash.

    Returns
    -------
    (weights, trades) — weights is wide-format, trades is a log.
    """
    all_hours = returns_wide.index.sort_values()
    symbols = returns_wide.columns

    # Pivot signals to wide
    if not signals.empty:
        sig_wide = signals.pivot_table(
            index="ts", columns="symbol", values="se_score", aggfunc="last"
        ).reindex(all_hours).ffill(limit=rebalance_every_hours)

        # tc estimates wide
        tc_wide = signals[signals["lppl_confirmed"]].pivot_table(
            index="ts", columns="symbol", values="tc_hours", aggfunc="last"
        ).reindex(all_hours).ffill(limit=rebalance_every_hours * 2)
    else:
        sig_wide = pd.DataFrame(0.0, index=all_hours, columns=symbols)
        tc_wide = pd.DataFrame(np.nan, index=all_hours, columns=symbols)

    vol = returns_wide.rolling(vol_lookback, min_periods=vol_lookback // 2).std()

    weights = pd.DataFrame(0.0, index=all_hours, columns=symbols)
    holdings: dict[str, dict] = {}  # symbol -> {entry_hour, entry_price_idx, peak_cum}
    trades = []
    hour_counter = 0

    for dt in all_hours:
        hour_counter += 1

        # Regime filter
        if regime_mask is not None and dt in regime_mask.index and not regime_mask.loc[dt]:
            # Close everything
            for sym in list(holdings.keys()):
                trades.append({"ts": dt, "symbol": sym, "action": "exit_regime", "hours_held": 0})
            holdings.clear()
            continue

        r = returns_wide.loc[dt].fillna(0.0)

        # Update tracking for existing holdings
        for sym in list(holdings.keys()):
            h = holdings[sym]
            h["hours_held"] = h.get("hours_held", 0) + 1
            h["cum_ret"] = (1 + h.get("cum_ret", 0)) * (1 + r.get(sym, 0)) - 1
            h["peak_cum"] = max(h.get("peak_cum", 0), h["cum_ret"])

            # EXIT RULE 1: tc approaching (the LPPLS edge)
            tc = tc_wide.loc[dt].get(sym, np.nan) if dt in tc_wide.index else np.nan
            if not np.isnan(tc) and tc < tc_exit_hours:
                trades.append({
                    "ts": dt, "symbol": sym, "action": "exit_tc",
                    "hours_held": h["hours_held"], "cum_ret": h["cum_ret"],
                    "tc_at_exit": tc,
                })
                del holdings[sym]
                continue

            # EXIT RULE 2: trailing stop
            drawdown_from_peak = h["peak_cum"] - h["cum_ret"]
            if h["peak_cum"] > 0.02 and drawdown_from_peak > trailing_stop_pct:
                trades.append({
                    "ts": dt, "symbol": sym, "action": "exit_stop",
                    "hours_held": h["hours_held"], "cum_ret": h["cum_ret"],
                })
                del holdings[sym]
                continue

            # EXIT RULE 3: max holding period
            if h["hours_held"] >= max_hold_hours:
                trades.append({
                    "ts": dt, "symbol": sym, "action": "exit_maxhold",
                    "hours_held": h["hours_held"], "cum_ret": h["cum_ret"],
                })
                del holdings[sym]
                continue

        # ENTRY: every rebalance_every_hours, re-rank and add new positions
        if hour_counter % rebalance_every_hours == 0:
            if dt in sig_wide.index:
                s = sig_wide.loc[dt].dropna()
                s = s[s > min_signal]
                # Exclude already-held symbols
                s = s.drop(labels=[sym for sym in holdings if sym in s.index], errors="ignore")
                # Fill remaining slots
                n_open = top_k - len(holdings)
                if n_open > 0 and len(s) > 0:
                    new_entries = s.sort_values(ascending=False).head(n_open)
                    for sym in new_entries.index:
                        holdings[sym] = {
                            "entry_hour": dt,
                            "hours_held": 0,
                            "cum_ret": 0.0,
                            "peak_cum": 0.0,
                        }
                        trades.append({
                            "ts": dt, "symbol": sym, "action": "entry",
                            "signal": new_entries[sym],
                        })

        # Compute weights from current holdings
        if holdings:
            held = list(holdings.keys())
            if ivol_weight and dt in vol.index:
                v = vol.loc[dt].reindex(held).fillna(vol.loc[dt].median()).clip(lower=1e-6)
                raw_w = 1.0 / v
            else:
                raw_w = pd.Series(1.0, index=held)
            raw_w = raw_w / raw_w.sum()
            for sym in held:
                if sym in raw_w.index:
                    weights.loc[dt, sym] = raw_w[sym]

    trades_df = pd.DataFrame(trades)
    return weights, trades_df


def backtest_hourly(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    tc_bps: float = 30.0,
    cash_rate: float = 0.04,
) -> pd.DataFrame:
    """Backtest the hourly portfolio.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format hourly weights.
    returns_wide : pd.DataFrame
        Hourly asset returns.
    tc_bps : float
        One-way transaction cost in bps (wider for HF).
    cash_rate : float
        Annual cash rate.
    """
    ann = 365.0 * 24
    tc = tc_bps / 10_000.0
    daily_cash = (1 + cash_rate) ** (1 / ann) - 1

    aligned = weights.index.intersection(returns_wide.index)
    W = weights.loc[aligned]
    R = returns_wide.loc[aligned]

    records = []
    prev_w = pd.Series(0.0, index=W.columns)

    for dt in aligned:
        w = W.loc[dt]
        r = R.loc[dt].fillna(0.0)

        delta = (w - prev_w).abs().sum() / 2.0
        cost = delta * tc
        gross_ret = (w * r).sum()
        cash_frac = max(1.0 - w.abs().sum(), 0.0)
        net_ret = gross_ret - cost + cash_frac * daily_cash

        records.append({
            "ts": dt,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": delta,
            "n_holdings": (w.abs() > 1e-6).sum(),
        })

        drifted = w * (1 + r)
        total = drifted.sum()
        prev_w = drifted / total if abs(total) > 1e-10 else drifted

    result = pd.DataFrame(records)
    if not result.empty:
        result["cum_ret"] = (1 + result["net_ret"]).cumprod()
    return result


def hourly_performance_summary(bt: pd.DataFrame) -> dict:
    """Performance stats for hourly backtest."""
    if bt.empty or len(bt) < 100:
        return {"error": "insufficient data"}

    ann = 365.0 * 24
    n_hours = len(bt)
    n_years = n_hours / ann
    cum = bt["cum_ret"].iloc[-1]
    cagr = cum ** (1 / n_years) - 1 if n_years > 0 else 0

    vol = bt["net_ret"].std() * np.sqrt(ann)
    sharpe = cagr / vol if vol > 1e-6 else 0

    dd = bt["cum_ret"] / bt["cum_ret"].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0

    return {
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "total_return": cum - 1,
        "avg_holdings": bt["n_holdings"].mean(),
        "avg_hourly_turnover": bt["turnover"].mean(),
        "n_hours": n_hours,
        "n_days": n_hours / 24,
    }
