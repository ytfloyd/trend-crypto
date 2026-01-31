# Transtrend Crypto Simple Baseline (Fixed Universe)

## Overview
This baseline is a clean, interpretable long-only trend system designed to isolate signal quality without risk overlays. It uses a fixed universe and a single MA crossover rule, with equal-weight sizing across active signals.

## Fixed Universe (Spot USD)
BTC, ETH, ADA, BNB, XRP, SOL, DOT, DOGE, AVAX, UNI, LINK, LTC, ALGO, BCH, ATOM, ICP, XLM, FIL, TRX, VET, ETC, XTZ

## Signal Definition
Per asset, daily:
- Fast MA = 20-day simple moving average of close
- Slow MA = 100-day simple moving average of close
- Signal = 1 if Fast MA > Slow MA, else 0

## Position Sizing
- Equal weight across all assets with signal == 1
- If no signals are active, the portfolio is entirely in cash

## Execution & Costs
- Decision at close, execution at next open (one-bar lag)
- Default cost is 0 bps (configurable in the runner)

## Outputs
The runner writes:
- `equity.csv`
- `weights_signal.parquet`
- `weights_held.parquet`
- `turnover.csv`
- `run_manifest.json`

Metrics and tearsheet are produced by separate helper scripts.
