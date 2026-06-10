"""
Futures-only, NON vol-scaled backtester for the silver replicator.

Walks bar-by-bar from a signed contracts series and a bar dataframe:

    PnL[t]            = contracts[t-1] * (close[t] - close[t-1]) * QI_MULT
    commission[t]     = $2.50 * |contracts[t] - contracts[t-1]|

Returns a dict of headline performance stats and (optionally) the per-bar
equity curve / position diagnostics.

No vol-scaling — the simulator runs the model size unmodified.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

QI_MULT = 2_500.0       # $ per $1/oz move per QI mini contract
COMMISSION_PER_CONTRACT = 2.50


def simulate_futures_pnl(
    state: pd.Series,                  # signed state (-1/0/+1) -- unused but kept for API symmetry
    contracts: pd.Series,              # signed integer contracts per bar
    bars: pd.DataFrame,                # must contain 'c' column
    *,
    bars_per_year: float = 2.26 * 252,  # default 8H
    return_curve: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], pd.DataFrame]:
    """
    Bar-by-bar futures P&L simulation, no vol-scaling.

    Position is interpreted as "the model wants `contracts[t]` exposure into
    bar t -> t+1". To avoid look-ahead, the position that earns the return
    from bar t to bar t+1 is `contracts[t]` itself (set at the close of bar t,
    held until close of bar t+1). Equivalently in vector form:

        pnl[t] = contracts.shift(1)[t] * (close[t] - close[t-1]) * QI_MULT

    Commissions are charged on the bar where contracts change.
    """
    c = bars["c"].astype(float).reindex(contracts.index).ffill()
    ret_dollar = c.diff().fillna(0.0) * QI_MULT  # $ move per 1 QI contract per bar

    pos = contracts.astype(float)
    lag_pos = pos.shift(1).fillna(0.0)
    pnl = lag_pos * ret_dollar

    dpos = pos.diff().fillna(pos.iloc[0] if len(pos) else 0.0).abs()
    commission = dpos * COMMISSION_PER_CONTRACT
    net_pnl = pnl - commission
    equity = net_pnl.cumsum()

    total_pnl = float(equity.iloc[-1]) if len(equity) else 0.0
    # Drawdown
    running_max = equity.cummax()
    dd = (equity - running_max)
    max_drawdown_dollar = float(-dd.min()) if len(dd) else 0.0

    # Sharpe (annualized, on per-bar net pnl)
    if net_pnl.std() > 1e-9:
        sharpe = float(net_pnl.mean() / net_pnl.std() * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    # Trade-level stats
    # A "trade" = each |delta_contracts| event > 0.
    trade_events = int((dpos > 0).sum())

    # Longest hold (in bars) of a non-zero, signed position
    pos_arr = pos.to_numpy().astype(int)
    longest_hold = 0
    cur_sign = 0
    run_len = 0
    for v in pos_arr:
        s = int(np.sign(v))
        if s == cur_sign and s != 0:
            run_len += 1
        else:
            if cur_sign != 0:
                longest_hold = max(longest_hold, run_len)
            cur_sign = s
            run_len = 1 if s != 0 else 0
    if cur_sign != 0:
        longest_hold = max(longest_hold, run_len)

    total_dollar_traded = float((dpos * c).sum() * QI_MULT / QI_MULT)
    # ^ dpos * c = $/oz × |contracts|; multiplied by QI_MULT then divided cancels.
    # Equivalent simpler formulation:
    total_dollar_traded = float((dpos * c * QI_MULT).sum())

    out = {
        "total_pnl": total_pnl,
        "max_drawdown_$": max_drawdown_dollar,
        "sharpe_annualized": sharpe,
        "num_trades": trade_events,
        "longest_hold_bars": int(longest_hold),
        "total_$_traded": total_dollar_traded,
        "gross_pnl": float(pnl.sum()),
        "total_commission": float(commission.sum()),
    }

    if return_curve:
        curve = pd.DataFrame({
            "close": c,
            "contracts": pos,
            "lag_contracts": lag_pos,
            "pnl_bar": pnl,
            "commission": commission,
            "net_pnl_bar": net_pnl,
            "equity": equity,
        })
        return out, curve
    return out


if __name__ == "__main__":
    import pathlib
    art = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
    bars = pd.read_parquet(art / "si_front_month_8H.parquet").set_index("ts")
    n = len(bars)
    state = pd.Series([1] * n, index=bars.index)
    contracts = (state * 4).astype("int32")
    stats = simulate_futures_pnl(state, contracts, bars)
    print("smoke test (4 QI long, hold to expiry):", stats)
