#!/usr/bin/env python3
"""
Strategy 1: Intraday Momentum Breakout (1h)

Donchian channel breakout adapted for 24/7 crypto markets.

Entry: price closes above N-bar high → long
Exit:  price closes below shorter-period low → flat
Sizing: inverse-vol weighted across assets, vol-targeted portfolio

This exploits crypto's strong intraday momentum: trends persist at
the hourly scale, and breakouts from consolidation ranges tend to
follow through. The asymmetric exit (tight trailing stop via shorter
lookback) creates the convex payoff profile — small losses, fat-tail
gains on trend continuation.
"""
from _common import *


def momentum_breakout_signals(
    panel: pd.DataFrame,
    entry_lookback: int = 48,    # 2 days of hourly bars
    exit_lookback: int = 24,     # 1 day
) -> pd.DataFrame:
    """Generate breakout signals for all assets."""
    signals = {}
    for sym, grp in panel[panel["in_universe"]].groupby("symbol"):
        g = grp.sort_values("ts").set_index("ts")
        if len(g) < entry_lookback * 2:
            continue

        high_channel = g["high"].rolling(entry_lookback).max()
        low_channel = g["low"].rolling(exit_lookback).min()

        # Position: 1 if above channel, 0 if below exit channel
        pos = pd.Series(0.0, index=g.index)
        in_trade = False
        for i in range(1, len(g)):
            if not in_trade:
                if g["close"].iloc[i] > high_channel.iloc[i - 1]:
                    in_trade = True
                    pos.iloc[i] = 1.0
            else:
                if g["close"].iloc[i] < low_channel.iloc[i - 1]:
                    in_trade = False
                    pos.iloc[i] = 0.0
                else:
                    pos.iloc[i] = 1.0

        signals[sym] = pos

    return pd.DataFrame(signals)


def inverse_vol_weight(
    signals: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 24 * 7,  # 1 week of hourly bars
) -> pd.DataFrame:
    """Weight active positions by inverse realized vol."""
    common = signals.columns.intersection(returns_wide.columns)
    vol = returns_wide[common].rolling(vol_lookback, min_periods=24).std()

    active = signals[common].reindex(returns_wide.index).fillna(0)
    raw_w = active / vol.replace(0, np.nan)
    raw_w = raw_w.fillna(0)

    # Normalize to sum to 1 when there are active positions
    row_sum = raw_w.sum(axis=1).replace(0, np.nan)
    weights = raw_w.div(row_sum, axis=0).fillna(0)
    return weights


def main():
    print("Loading 1h universe...")
    panel, returns_wide, close_wide = load_hourly_universe()
    print(f"  {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    # Sweep parameters
    configs = [
        {"entry": 24, "exit": 12, "label": "Breakout 24/12"},
        {"entry": 48, "exit": 24, "label": "Breakout 48/24"},
        {"entry": 72, "exit": 36, "label": "Breakout 72/36"},
        {"entry": 96, "exit": 48, "label": "Breakout 96/48"},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        signals = momentum_breakout_signals(panel, cfg["entry"], cfg["exit"])
        weights = inverse_vol_weight(signals, returns_wide)
        result = run_and_report(cfg["label"], weights, returns_wide, close_wide)
        all_results.append(result)

    print(f"\n\n{'='*70}")
    print("  MOMENTUM BREAKOUT SUMMARY")
    print(f"{'='*70}")
    print(format_metrics_table(all_results))


if __name__ == "__main__":
    main()
