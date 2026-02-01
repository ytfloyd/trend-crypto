# Transtrend Crypto v2 (Multi-Speed, Spot Long-Only)

## Overview
Transtrend Crypto v2 extends the single-speed trend sleeve into a two-sleeve portfolio with distinct time horizons. The slow sleeve targets long-horizon trend persistence and lower turnover, while the fast sleeve reacts to shorter-term momentum and provides tactical responsiveness. Both sleeves are spot long-only and share the same risk stack (vol targeting, danger gating, and trading costs).

## Sleeve Design
- Slow sleeve horizon: breakout=50, fast_ma=10, slow_ma=200
- Fast sleeve horizon: breakout=10, fast_ma=2, slow_ma=20

Each sleeve uses a single horizon to keep signal interpretation clean and avoid blending inside the sleeve.

## Combination Method (Return Mix)
Sleeves are run independently to produce their own portfolio returns, equity curves, and turnover series. The combined portfolio is built as a weighted return mix:

```
combined_ret[t] = w_fast * fast_ret[t] + w_slow * slow_ret[t]
combined_equity = cumprod(1 + combined_ret)
```

Turnover is approximated as the weighted sum of sleeve turnovers. The danger flag is combined via logical OR across sleeves.

## Why Multi-Speed Improves Convexity
A multi-speed design blends two different reaction horizons:
- The slow sleeve captures sustained trend moves with lower noise and lower turnover.
- The fast sleeve reacts earlier to regime changes, improving responsiveness during sharp trend shifts.

The return-mix approach creates a convexity benefit: the combined portfolio can participate earlier in emerging trends while maintaining a stable core exposure, often improving drawdown resilience without materially increasing costs.

## Default Risk Budgets
- Slow sleeve: 70%
- Fast sleeve: 30%

These weights prioritize stability while still allowing the fast sleeve to contribute during rapid transitions.
