# Transtrend Crypto v1 (Spot Long-Only + ATR Stops) — Overview

## What changed vs v0
v1 adds per-symbol ATR trailing stops to reduce tail risk and drawdowns, while preserving the
multi-horizon breakout + MA filter core. The strategy remains spot long-only and maintains
Model-B timing (decision at close, execution next open, open→close returns).

## ATR stop overlay
Defaults:
- ATR window: 20 days
- ATR multiple (k): 3.0
- Cooldown: 5 days

Stop logic:
- ATR uses shifted inputs (ATR(t) uses data through t-1; no lookahead).
- Entry price = next open (execution price).
- Initial stop = entry_price − k * ATR(entry).
- Trailing stop = max_close_since_entry − k * ATR(entry) (ATR fixed at entry in v1).
- If close[t] <= stop_level[t], the symbol is forced to 0 weight from the next bar,
  and enters cooldown for N days.
- After cooldown, normal score-based entry is allowed again.

## Interaction with sizing
Stops are applied **before** final normalization:
1) compute raw weights from score / vol
2) set w_raw=0 for stopped symbols
3) re-normalize to gross/vol targets and apply danger gating

## Outputs for audit
- stop_levels.parquet: per symbol stop_level, cooldown, ATR entry
- stop_events.csv: per symbol stop hits with stop_level/close/atr_entry

## Roadmap (v2+)
- ATR updated dynamically (ATR(t) vs ATR(entry))
- Volatility-adjusted cooldown
- Multi-asset regime gating or per-asset stop multipliers
