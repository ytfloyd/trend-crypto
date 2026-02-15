# Transtrend Crypto MA(5/40) Top-K (Long-Only)

## Overview
This strategy applies the MA(5/40) long-only trend signal on a dynamically pruned `kuma_live_universe` and then selects the top-K symbols by recent performance. It is designed to reduce correlation and drawdown by focusing capital on the strongest trending names.

## Signal Definition
Per asset, daily:
- Fast MA = 5-day simple moving average of close
- Slow MA = 40-day simple moving average of close
- Signal = 1 if Fast MA > Slow MA, else 0

Signals are computed without lookahead (shifted inputs so decision at close(t) uses data through t-1).

## Ranking & Top-K Selection
- Default ranking metric: 60-day close-to-close return
- Score uses shifted prices so ranking at time t only uses data through t-1
- For each day, select the top K symbols among those with signal = 1 and finite scores
- Allocate equal weight across selected symbols

## Portfolio Construction
- Long-only, equal-weight across the selected K symbols
- If no symbols qualify, hold cash (zero gross exposure)

## Execution & Costs
- Model-B timing: decide at close(t), execute at open(t+1)
- Returns are open-to-close
- Cost = turnover_one_sided * cost_bps / 10,000

## Default Parameters
- fast = 5, slow = 40
- k = 5
- rank_lookback_days = 60
- cost_bps = 20
