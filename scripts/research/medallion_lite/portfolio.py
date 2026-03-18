"""
Portfolio construction — event-driven holdings with factor-timed entry/exit.

At 30 bps crypto costs, continuous rebalancing destroys returns.
Instead we use the cross-sectional factor model for SELECTION
and the ensemble regime for TIMING, but hold positions
event-style (enter, hold, exit) to minimise turnover.

Entry:  composite score > entry_threshold AND regime > regime_entry_min
Exit:   (a) score drops below exit_threshold (factor degradation)
        (b) regime drops below regime_exit_min (market deterioration)
        (c) trailing stop (momentum breakdown)
        (d) max holding period (time decay)

This is the LPPLS/Donchian architecture with a better signal generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ANN_FACTOR = 8760.0


def build_factor_portfolio(
    composite_score: pd.DataFrame,
    returns_wide: pd.DataFrame,
    regime_score: pd.Series,
    *,
    entry_threshold: float = 0.65,
    exit_score_threshold: float = 0.40,
    regime_entry_min: float = 0.45,
    regime_exit_min: float = 0.15,
    max_hold_hours: int = 336,
    trailing_stop_pct: float = 0.15,
    rebalance_every_hours: int = 24,
    max_positions: int = 25,
    ivol_weight: bool = True,
    vol_lookback: int = 168,
    max_weight: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build portfolio with factor-timed entries and multi-rule exits.

    Returns
    -------
    weights : pd.DataFrame  (ts × symbol)
    trades  : pd.DataFrame  (trade log)
    """
    all_hours = returns_wide.index.sort_values()
    symbols = returns_wide.columns

    score_smooth = composite_score.ewm(span=72, min_periods=24).mean()

    vol = (
        returns_wide.rolling(vol_lookback, min_periods=vol_lookback // 4)
        .std()
        * np.sqrt(ANN_FACTOR)
    )

    weights = pd.DataFrame(0.0, index=all_hours, columns=symbols)
    holdings: dict[str, dict] = {}
    trades: list[dict] = []
    hour_counter = 0

    for dt in all_hours:
        hour_counter += 1

        rs = float(regime_score.get(dt, 0.0)) if dt in regime_score.index else 0.0
        r = returns_wide.loc[dt].fillna(0.0)

        # ── Update existing holdings ──────────────────────────────────
        for sym in list(holdings.keys()):
            h = holdings[sym]
            h["hours_held"] = h.get("hours_held", 0) + 1
            h["cum_ret"] = (1 + h.get("cum_ret", 0)) * (1 + r.get(sym, 0)) - 1
            h["peak_cum"] = max(h.get("peak_cum", 0), h["cum_ret"])

            # EXIT 1: regime collapse
            if rs < regime_exit_min:
                trades.append(dict(
                    ts=dt, symbol=sym, action="exit_regime",
                    hours_held=h["hours_held"], cum_ret=h["cum_ret"],
                ))
                del holdings[sym]
                continue

            # EXIT 2: trailing stop
            dd = h["peak_cum"] - h["cum_ret"]
            if h["peak_cum"] > 0.02 and dd > trailing_stop_pct:
                trades.append(dict(
                    ts=dt, symbol=sym, action="exit_stop",
                    hours_held=h["hours_held"], cum_ret=h["cum_ret"],
                ))
                del holdings[sym]
                continue

            # EXIT 3: max hold
            if h["hours_held"] >= max_hold_hours:
                trades.append(dict(
                    ts=dt, symbol=sym, action="exit_maxhold",
                    hours_held=h["hours_held"], cum_ret=h["cum_ret"],
                ))
                del holdings[sym]
                continue

            # EXIT 4: factor degradation (only at rebalance points)
            if hour_counter % rebalance_every_hours == 0:
                sc = (
                    score_smooth.loc[dt].get(sym, np.nan)
                    if dt in score_smooth.index
                    else np.nan
                )
                if not np.isnan(sc) and sc < exit_score_threshold:
                    trades.append(dict(
                        ts=dt, symbol=sym, action="exit_factor",
                        hours_held=h["hours_held"], cum_ret=h["cum_ret"],
                        score_at_exit=float(sc),
                    ))
                    del holdings[sym]
                    continue

        # ── Entry: at rebalance points, scan for new positions ────────
        if hour_counter % rebalance_every_hours == 0 and rs >= regime_entry_min:
            if dt in score_smooth.index:
                scores = score_smooth.loc[dt].dropna()
                candidates = scores[scores > entry_threshold]
                candidates = candidates.drop(
                    labels=[s for s in holdings if s in candidates.index],
                    errors="ignore",
                )

                n_open = max_positions - len(holdings)
                if n_open > 0 and len(candidates) > 0:
                    new_entries = candidates.sort_values(ascending=False).head(n_open)
                    for sym in new_entries.index:
                        holdings[sym] = dict(
                            entry_hour=dt,
                            hours_held=0,
                            cum_ret=0.0,
                            peak_cum=0.0,
                            entry_score=float(new_entries[sym]),
                        )
                        trades.append(dict(
                            ts=dt, symbol=sym, action="entry",
                            signal=float(new_entries[sym]),
                            regime_score=float(rs),
                        ))

        # ── Compute weights from current holdings ─────────────────────
        if holdings:
            held = list(holdings.keys())
            if ivol_weight and dt in vol.index:
                v = vol.loc[dt].reindex(held).fillna(vol.loc[dt].median()).clip(lower=0.1)
                raw_w = 1.0 / v
            else:
                raw_w = pd.Series(1.0, index=held)

            # Scale by regime probability (smooth exposure)
            regime_scale = max(rs, 0.20) if rs >= regime_exit_min else 0.0
            raw_w = raw_w * regime_scale

            total = raw_w.sum()
            if total > 0:
                raw_w = raw_w / total

            # Position cap
            raw_w = raw_w.clip(upper=max_weight)
            total = raw_w.sum()
            if total > 0:
                raw_w = raw_w / total

            for sym in held:
                if sym in raw_w.index:
                    weights.loc[dt, sym] = raw_w[sym]

    return weights, pd.DataFrame(trades)


def backtest_portfolio(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    tc_bps: float = 30.0,
    cash_rate: float = 0.04,
) -> pd.DataFrame:
    """Backtest weights vs hourly returns.

    Identical to sornette_lppl.portfolio_hf.backtest_hourly for fair comparison.
    """
    tc = tc_bps / 10_000.0
    hourly_cash = (1 + cash_rate) ** (1 / ANN_FACTOR) - 1

    aligned = weights.index.intersection(returns_wide.index)
    W = weights.loc[aligned]
    R = returns_wide.loc[aligned]

    records: list[dict] = []
    prev_w = pd.Series(0.0, index=W.columns)

    for dt in aligned:
        w = W.loc[dt]
        r = R.loc[dt].fillna(0.0)

        delta = (w - prev_w).abs().sum() / 2.0
        cost = delta * tc
        gross_ret = (w * r).sum()
        cash_frac = max(1.0 - w.abs().sum(), 0.0)
        net_ret = gross_ret - cost + cash_frac * hourly_cash

        records.append(dict(
            ts=dt, gross_ret=gross_ret, net_ret=net_ret,
            turnover=delta,
            n_holdings=int((w.abs() > 1e-6).sum()),
        ))

        drifted = w * (1 + r)
        total = drifted.sum()
        prev_w = drifted / total if abs(total) > 1e-10 else drifted

    result = pd.DataFrame(records)
    if not result.empty:
        result["cum_ret"] = (1 + result["net_ret"]).cumprod()
    return result


def performance_summary(bt: pd.DataFrame) -> dict:
    """Standard performance stats."""
    if bt.empty or len(bt) < 100:
        return {"error": "insufficient data"}

    n_hours = len(bt)
    n_years = n_hours / ANN_FACTOR
    cum = bt["cum_ret"].iloc[-1]
    cagr = cum ** (1 / n_years) - 1 if n_years > 0 else 0

    vol = bt["net_ret"].std() * np.sqrt(ANN_FACTOR)
    sharpe = (
        bt["net_ret"].mean() / bt["net_ret"].std() * np.sqrt(ANN_FACTOR)
        if bt["net_ret"].std() > 1e-12
        else 0
    )

    dd = bt["cum_ret"] / bt["cum_ret"].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0

    return dict(
        cagr=cagr,
        annual_vol=vol,
        sharpe=sharpe,
        max_dd=max_dd,
        calmar=calmar,
        total_return=cum - 1,
        avg_holdings=bt["n_holdings"].mean(),
        avg_hourly_turnover=bt["turnover"].mean(),
        n_hours=n_hours,
        n_days=n_hours / 24,
    )
