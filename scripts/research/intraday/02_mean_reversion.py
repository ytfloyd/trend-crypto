#!/usr/bin/env python3
"""
Strategy 2: Volatility-Filtered Mean Reversion (1h)

Bollinger band mean-reversion with a volatility regime filter.

Entry: price touches lower Bollinger Band during HIGH-vol regime
       (vol expansion creates temporary dislocations that revert)
Exit:  price reverts to mid-band (MA) OR hits tight stop-loss

Convexity comes from:
  - Tight asymmetric exits (stop < target)
  - Vol filter ensures we only trade when dislocations are genuine
  - Many small wins + rare but contained losses

This is a counter-trend strategy that works in ranging/choppy markets.
It complements the momentum breakout strategy (Strategy 1).
"""
from _common import *


def mean_reversion_signals(
    panel: pd.DataFrame,
    bb_window: int = 48,       # 2 days
    bb_std: float = 2.0,
    vol_window: int = 24 * 7,  # 1 week
    vol_percentile: float = 60,  # only trade when vol > this percentile
    stop_mult: float = 0.5,   # stop at 0.5× distance from entry to band
) -> pd.DataFrame:
    """Generate mean-reversion signals."""
    signals = {}
    for sym, grp in panel[panel["in_universe"]].groupby("symbol"):
        g = grp.sort_values("ts").set_index("ts")
        if len(g) < bb_window * 3:
            continue

        close = g["close"]
        ma = close.rolling(bb_window).mean()
        std = close.rolling(bb_window).std()
        upper = ma + bb_std * std
        lower = ma - bb_std * std

        # Vol regime: rolling realized vol
        ret = np.log(close / close.shift(1))
        realized_vol = ret.rolling(vol_window, min_periods=48).std()
        vol_threshold = realized_vol.expanding().quantile(vol_percentile / 100)

        pos = pd.Series(0.0, index=g.index)
        entry_price = 0.0
        stop_price = 0.0

        for i in range(bb_window, len(g)):
            if pos.iloc[i - 1] == 0:
                # Entry: price below lower band AND high vol regime
                if (close.iloc[i] < lower.iloc[i] and
                    realized_vol.iloc[i] > vol_threshold.iloc[i]):
                    pos.iloc[i] = 1.0
                    entry_price = close.iloc[i]
                    band_dist = ma.iloc[i] - lower.iloc[i]
                    stop_price = entry_price - stop_mult * band_dist
            else:
                # Exit: hit mid-band (profit target) or stop
                if close.iloc[i] >= ma.iloc[i]:
                    pos.iloc[i] = 0.0  # take profit at mid-band
                elif close.iloc[i] <= stop_price:
                    pos.iloc[i] = 0.0  # stop loss
                else:
                    pos.iloc[i] = 1.0

        signals[sym] = pos

    return pd.DataFrame(signals)


def equal_weight_active(signals: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight all active positions."""
    active = signals.fillna(0)
    n_active = active.sum(axis=1).replace(0, np.nan)
    return active.div(n_active, axis=0).fillna(0)


def main():
    print("Loading 1h universe...")
    panel, returns_wide, close_wide = load_hourly_universe()
    print(f"  {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    configs = [
        {"bb_window": 24, "bb_std": 2.0, "vol_pct": 50, "label": "MR 24h BB2.0 V50"},
        {"bb_window": 48, "bb_std": 2.0, "vol_pct": 60, "label": "MR 48h BB2.0 V60"},
        {"bb_window": 48, "bb_std": 2.5, "vol_pct": 60, "label": "MR 48h BB2.5 V60"},
        {"bb_window": 72, "bb_std": 2.0, "vol_pct": 70, "label": "MR 72h BB2.0 V70"},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        signals = mean_reversion_signals(
            panel, bb_window=cfg["bb_window"], bb_std=cfg["bb_std"],
            vol_percentile=cfg["vol_pct"],
        )
        weights = equal_weight_active(signals)
        weights = weights.reindex(returns_wide.index).fillna(0)
        result = run_and_report(cfg["label"], weights, returns_wide, close_wide)
        all_results.append(result)

    print(f"\n\n{'='*70}")
    print("  MEAN REVERSION SUMMARY")
    print(f"{'='*70}")
    print(format_metrics_table(all_results))


if __name__ == "__main__":
    main()
