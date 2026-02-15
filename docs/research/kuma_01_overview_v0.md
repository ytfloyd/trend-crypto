# kuma_01 v0 (Breakout + MA5/40 + Dynamic ATR30 Stop)

## Summary
kuma_01 is a long-only trend strategy that combines a 20-day breakout with an MA(5/40) trend filter, uses a dynamic ATR(30) trailing stop, and sizes positions inverse to 31-day realized volatility. Allocation is long-only with residual cash when no assets qualify.

## Entry Criteria (decision at close t, using data through t-1)
- Breakout20: close[t] > max(close[t-1..t-20])
- Trend filter: SMA5(t) > SMA40(t), computed on close with shift(1)
- Entry signal: breakout20 AND trend_ok

## Stop Logic (dynamic ATR30)
- ATR window: 30-day True Range rolling mean, shifted by 1
- Highest close since entry tracked per symbol
- Stop level: highest_close_since_entry(t) - 3 * ATR30(t)
- Stop evaluated at close(t); if hit, exit next bar (t+1)

## Position Sizing
- Eligible assets: entry_signal == True AND stop_block == False AND vol31 finite
- vol31: rolling 31-day std of close-to-close returns, shifted by 1
- Weights proportional to 1/vol31 and normalized to gross exposure = 1.0
- If no eligible assets, weights are 0 and capital sits in cash

## Execution & Returns
- Model-B timing: decide at close(t), execute at open(t+1)
- PnL uses open-to-close returns
- Costs: turnover_one_sided * cost_bps / 10,000

## Universe
- Uses `kuma_live_universe` via `scripts/research/universes.py` with dynamic pruning by recent data
