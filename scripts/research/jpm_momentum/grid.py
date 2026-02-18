"""
Parameter sweep engine for JPM Momentum research.

Systematic grid scan to replicate the paper's Tables/Figures:
- Loop over: signal_type x lookback x rebalance_freq x position_sizing
- For each combination: run backtest, collect (Sharpe, CAGR, MaxDD, Turnover, Skewness)
- Output: results DataFrame + heatmaps (lookback vs. signal type)

Supports both crypto (ann_factor=365) and ETF (ann_factor=252) markets.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence

import pandas as pd

from .config import DEFAULT_LOOKBACKS, SIGNAL_TYPES
from .data import ANN_FACTOR_CRYPTO
from .signals import compute_signal
from .universe import filter_universe
from .weights import build_weights
from .backtest import simple_backtest
from .metrics import compute_metrics
from .data import compute_returns_wide


def run_grid(
    panel: pd.DataFrame,
    signal_types: Sequence[str] = SIGNAL_TYPES,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    weight_methods: Sequence[str] = ("equal",),
    cost_bps: float = 20.0,
    min_adv_usd: float = 1_000_000,
    min_history_days: int = 90,
    mode: str = "absolute",
    top_k: int | None = None,
    ann_factor: float = ANN_FACTOR_CRYPTO,
    return_method: str = "open_to_close",
) -> pd.DataFrame:
    """Run a full parameter grid sweep.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format daily panel (symbol, ts, open, high, low, close, volume).
    signal_types : sequence of str
        Signal types to sweep.
    lookbacks : sequence of int
        Lookback windows to sweep.
    weight_methods : sequence of str
        Weighting methods to sweep (``equal``, ``inv_vol``, ``risk_parity``).
    cost_bps : float
        Transaction cost in bps.
    min_adv_usd : float
        Universe ADV filter.
    min_history_days : int
        Universe listing age filter.
    mode : str
        ``absolute`` (TSMOM) or ``relative`` (XSMOM).
    top_k : int | None
        For relative mode: number of top assets to hold.
    ann_factor : float
        Annualisation factor (365 for crypto, 252 for ETFs).
    return_method : str
        ``"open_to_close"`` for crypto, ``"close_to_close"`` for ETFs.

    Returns
    -------
    pd.DataFrame
        One row per (signal_type, lookback, weight_method) with columns:
        signal_type, lookback, weight_method, cagr, vol, sharpe, sortino,
        calmar, max_dd, hit_rate, skewness, kurtosis, avg_turnover.
    """
    # Pre-compute universe and returns (shared across all grid points)
    panel_u = filter_universe(panel, min_adv_usd=min_adv_usd, min_history_days=min_history_days)
    returns_wide = compute_returns_wide(panel, method=return_method)

    rows = []
    combos = list(product(signal_types, lookbacks, weight_methods))
    print(f"[grid] Running {len(combos)} combinations ...")

    for i, (sig_type, lb, wm) in enumerate(combos):
        label = f"{sig_type}_L{lb}_{wm}"
        try:
            sig_panel = compute_signal(panel_u, signal_type=sig_type, lookback=lb)

            if mode == "absolute":
                selected = sig_panel[sig_panel["in_universe"]].copy()
                selected["selected"] = selected["signal"] > 0
            else:
                selected = sig_panel[sig_panel["in_universe"]].copy()
                if top_k is not None:
                    ranked = selected.groupby("ts")["signal"].rank(ascending=False)
                    selected["selected"] = ranked <= top_k
                else:
                    ranked = selected.groupby("ts")["signal"].rank(pct=True)
                    selected["selected"] = ranked >= 0.80

            mask = selected.pivot(index="ts", columns="symbol", values="selected").fillna(False)
            w = build_weights(mask, returns_wide, method=wm, ann_factor=ann_factor)
            bt = simple_backtest(w, returns_wide, cost_bps=cost_bps, ann_factor=ann_factor)

            eq = pd.Series(bt["portfolio_equity"].values, index=pd.to_datetime(bt["ts"]))
            m = compute_metrics(eq, ann_factor=ann_factor)
            m["signal_type"] = sig_type
            m["lookback"] = lb
            m["weight_method"] = wm
            m["label"] = label
            m["avg_turnover"] = float(bt["turnover"].mean())
            rows.append(m)

        except Exception as e:
            print(f"  [{i+1}/{len(combos)}] {label} FAILED: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(combos)}] done")

    print(f"[grid] Completed {len(rows)}/{len(combos)} combinations.")
    return pd.DataFrame(rows)
