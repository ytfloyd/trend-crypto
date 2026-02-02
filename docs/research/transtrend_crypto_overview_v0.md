# Transtrend Crypto v0 (Spot Long-Only) — Research Overview

## Signal design
Transtrend v0 is a multi-horizon breakout + MA filter model. For each symbol and day:
- Breakout: close[t] > max(close[t-1..t-lookback])
- MA filter: SMA_fast(t-1) > SMA_slow(t-1)
- Horizon signal = breakout AND MA filter
- Composite score = mean of horizon signals in [0, 1]

Horizons (v0):
- fast: (breakout=10, SMA=2/20)
- mid:  (breakout=20, SMA=5/40)
- slow: (breakout=50, SMA=10/200)

All lookbacks use shift(1) to prevent same-bar leakage.

## Sizing
Long-only weights are inverse-volatility scaled:
- w_raw = score / max(vol_ann, vol_floor)
- Normalize to gross_target = min(max_gross, 1 - cash_buffer)
- Diagonal vol targeting: port_vol ≈ sqrt(sum((w_i * vol_i)^2))
- Scale down to hit target_vol_annual (never scale up above 1.0)
- Cap gross at max_gross

## Danger gating
A BTC-USD proxy drives portfolio-level risk-off gating. Danger is flagged if ANY:
- 20d annualized BTC vol > threshold
- 20d BTC drawdown < threshold
- 5d BTC return < threshold

If danger is True, gross exposure is scaled to `danger_gross`. If BTC data is missing, danger is False.

## Execution timing
Model-B timing is enforced:
- Decide at close(t)
- Execute at open(t+1)
- Earn open-to-close returns

## Outputs
Artifacts are deterministic and include:
- weights_signal.parquet
- weights_held.parquet
- equity.csv
- turnover.csv
- run_manifest.json
- metrics CSV and a tear sheet PDF

## v1 roadmap
- ATR-based stops
- Short sleeve (perps)
- Funding-aware returns
- Cross-asset regime filters
