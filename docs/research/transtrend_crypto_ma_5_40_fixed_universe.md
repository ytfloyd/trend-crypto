# Transtrend Crypto MA(5/40) Fixed-Universe Baseline

## Overview
This baseline mirrors the simple MA(20/100) long-only baseline but shortens the trend horizon to MA(5/40). It is a clean, interpretable benchmark for apples-to-apples comparison with the MA(20/100) model.

## Fixed Universe (Spot USD)
BTC, ETH, ADA, BNB, XRP, SOL, DOT, DOGE, AVAX, UNI, LINK, LTC, ALGO, BCH, ATOM, ICP, XLM, FIL, TRX, VET, ETC, XTZ

## Signal Definition
Per asset, daily:
- Fast MA = 5-day simple moving average of close
- Slow MA = 40-day simple moving average of close
- Signal = 1 if Fast MA > Slow MA, else 0

Signals are computed without lookahead using shifted inputs (decision at close(t) uses data through t-1).

## Portfolio Weights
- Equal-weight across active longs: if N active, weight = 1/N
- If no active signals, weights are 0 (cash)
- Gross exposure is either 0 or 1

## Execution & Costs
- Decision at close, execution at next open (one-bar lag)
- Open-to-close returns per bar
- Transaction cost = turnover_one_sided * cost_bps / 10,000

## How It Differs from MA(20/100)
- MA(5/40) responds faster and is more sensitive to short-term trend changes
- MA(20/100) is slower and smoother with lower turnover
