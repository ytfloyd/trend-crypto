# Transtrend Crypto Simple L/S Baseline (Fixed Universe)

## Overview
This is a minimal, interpretable long/short trend baseline designed to isolate signal quality. There are no overlays, no risk controls, and no dynamic universe logic.

## Fixed Universe (Spot USD)
BTC, ETH, ADA, BNB, XRP, SOL, DOT, DOGE, AVAX, UNI, LINK, LTC, ALGO, BCH, ATOM, ICP, XLM, FIL, TRX, VET, ETC, XTZ

## Signal Definition
Per asset, daily:
- Fast MA = 20-day simple moving average of close
- Slow MA = 100-day simple moving average of close
- Signal = +1 if Fast MA > Slow MA, else -1

Signals are computed without lookahead using shifted inputs (decision at close(t) uses data through t-1).

## Portfolio Weights
- Equal-weight long/short: w_i = signal_i / N, where N is the number of assets with data that day
- Normalize so sum(|w|) = 1.0 each day
- Gross exposure = 1.0 (unless no data), net exposure varies with signals

## Execution & Costs
- Decision at close, execution at next open (one-bar lag)
- Open-to-close returns for each bar
- Transaction cost = turnover_one_sided * cost_bps / 10,000

## Outputs
The runner writes:
- `equity.csv`
- `weights_signal.parquet`
- `weights_held.parquet`
- `turnover.csv`
- `run_manifest.json`

Metrics and tearsheet are produced by separate helper scripts.
