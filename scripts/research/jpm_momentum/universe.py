"""
Universe filtering for JPM Momentum research.

Implements survivorship-bias-free dynamic universe selection adapted from the paper:
- Minimum listing age (e.g. 90 days of data before inclusion)
- Minimum ADV threshold (e.g. $1M/day rolling 20-day)
- Symbols enter/exit dynamically based on these criteria

Returns the set of eligible symbols per date via an ``in_universe`` boolean column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def filter_universe(
    panel: pd.DataFrame,
    min_adv_usd: float = 1_000_000,
    min_history_days: int = 90,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Apply dynamic universe filter: rolling ADV + minimum listing age.

    Adds boolean column ``in_universe`` to the panel.
    Symbols enter once they have ``min_history_days`` of data AND
    their rolling ``adv_window``-day average daily volume (in USD) exceeds
    ``min_adv_usd``.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel with columns: symbol, ts, open, high, low, close, volume.
    min_adv_usd : float
        Minimum average daily volume in USD.
    min_history_days : int
        Minimum number of trading days before a symbol becomes eligible.
    adv_window : int
        Rolling window for ADV calculation.

    Returns
    -------
    pd.DataFrame
        Copy of input with added ``in_universe`` boolean column.
    """
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        dollar_vol = g["close"] * g["volume"]
        adv = dollar_vol.rolling(adv_window, min_periods=adv_window).mean()
        day_count = np.arange(1, len(g) + 1)
        g["in_universe"] = (adv >= min_adv_usd) & (day_count >= min_history_days)
        return g

    out = df.groupby("symbol", group_keys=False).apply(_per_symbol)
    out["in_universe"] = out["in_universe"].fillna(False)
    return out


def get_eligible_symbols(
    panel_with_universe: pd.DataFrame,
    date: pd.Timestamp,
) -> list[str]:
    """Return the list of symbols eligible on a given date.

    Parameters
    ----------
    panel_with_universe : pd.DataFrame
        Panel with ``in_universe`` column (from :func:`filter_universe`).
    date : pd.Timestamp
        Date to query.

    Returns
    -------
    list[str]
        Sorted list of eligible symbol names.
    """
    mask = (panel_with_universe["ts"] == date) & panel_with_universe["in_universe"]
    return sorted(panel_with_universe.loc[mask, "symbol"].tolist())
