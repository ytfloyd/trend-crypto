#!/usr/bin/env python3
"""
Strategy 4: CUSUM-Triggered Momentum (1h)

Uses the CUSUM filter from AFML to detect structural breaks in
intraday price series, then enters in the direction of the break.

Entry: CUSUM filter fires → enter in direction of the cumulative
       deviation that triggered the event
Exit:  Fixed holding period (24h) OR trailing stop at 1× vol

The CUSUM filter naturally adapts to market conditions:
  - Fires more often during volatile/trending periods
  - Stays quiet during low-vol consolidation
  → Natural regime filter built into the entry logic

Convexity: trailing stop preserves gains, fixed holding period
           limits time exposure. The CUSUM filter only triggers on
           significant moves, filtering out noise.
"""
from _common import *

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def cusum_momentum_signals(
    panel: pd.DataFrame,
    threshold_mult: float = 2.0,   # threshold = mult × hourly vol
    holding_period: int = 24,      # 1 day
    trail_stop_mult: float = 1.5,  # trailing stop at mult × vol
) -> pd.DataFrame:
    """CUSUM-triggered momentum entries with trailing stop exits."""
    signals = {}

    for sym, grp in panel[panel["in_universe"]].groupby("symbol"):
        g = grp.sort_values("ts").set_index("ts")
        if len(g) < 200:
            continue

        close = g["close"]
        ret = np.log(close / close.shift(1)).fillna(0)
        vol = ret.rolling(48).std()

        pos = pd.Series(0.0, index=g.index)
        s_pos, s_neg = 0.0, 0.0
        entry_bar = -holding_period - 1
        entry_price = 0.0
        peak_price = 0.0
        direction = 0

        for i in range(1, len(g)):
            v = vol.iloc[i]
            if pd.isna(v) or v <= 0:
                continue

            threshold = v * threshold_mult * close.iloc[i]

            # CUSUM accumulator
            diff = close.iloc[i] - close.iloc[i - 1]
            s_pos = max(0, s_pos + diff)
            s_neg = min(0, s_neg + diff)

            # Check if currently in a position
            bars_held = i - entry_bar

            if direction == 0:
                # Look for CUSUM trigger
                if s_pos > threshold:
                    direction = 1
                    entry_bar = i
                    entry_price = close.iloc[i]
                    peak_price = close.iloc[i]
                    pos.iloc[i] = 1.0
                    s_pos = 0.0
                elif s_neg < -threshold:
                    # Detected downward break — skip (long only for crypto)
                    s_neg = 0.0
            else:
                # In a long position
                peak_price = max(peak_price, close.iloc[i])
                trail_stop = peak_price * (1 - trail_stop_mult * v)

                # Exit conditions
                if bars_held >= holding_period:
                    pos.iloc[i] = 0.0
                    direction = 0
                elif close.iloc[i] < trail_stop:
                    pos.iloc[i] = 0.0
                    direction = 0
                else:
                    pos.iloc[i] = 1.0

        signals[sym] = pos

    return pd.DataFrame(signals)


def inverse_vol_weight(
    signals: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 24 * 7,
) -> pd.DataFrame:
    """Inverse-vol weight active positions."""
    common = signals.columns.intersection(returns_wide.columns)
    vol = returns_wide[common].rolling(vol_lookback, min_periods=24).std()

    active = signals[common].reindex(returns_wide.index).fillna(0)
    raw_w = active / vol.replace(0, np.nan)
    raw_w = raw_w.fillna(0)

    row_sum = raw_w.sum(axis=1).replace(0, np.nan)
    return raw_w.div(row_sum, axis=0).fillna(0)


def main():
    print("Loading 1h universe...")
    panel, returns_wide, close_wide = load_hourly_universe()
    print(f"  {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    configs = [
        {"thresh": 1.5, "hold": 12, "trail": 1.0, "label": "CUSUM 1.5x 12h T1.0"},
        {"thresh": 2.0, "hold": 24, "trail": 1.5, "label": "CUSUM 2.0x 24h T1.5"},
        {"thresh": 2.5, "hold": 48, "trail": 2.0, "label": "CUSUM 2.5x 48h T2.0"},
        {"thresh": 3.0, "hold": 72, "trail": 2.0, "label": "CUSUM 3.0x 72h T2.0"},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        signals = cusum_momentum_signals(
            panel,
            threshold_mult=cfg["thresh"],
            holding_period=cfg["hold"],
            trail_stop_mult=cfg["trail"],
        )
        weights = inverse_vol_weight(signals, returns_wide)
        result = run_and_report(cfg["label"], weights, returns_wide, close_wide)
        all_results.append(result)

    print(f"\n\n{'='*70}")
    print("  CUSUM MOMENTUM SUMMARY")
    print(f"{'='*70}")
    print(format_metrics_table(all_results))


if __name__ == "__main__":
    main()
