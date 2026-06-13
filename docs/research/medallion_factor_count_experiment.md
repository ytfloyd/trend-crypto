# Medallion Lite — factor-count experiment (5 vs 100 TA-Lib factors)

**Question.** Does expanding the cross-sectional composite from the 5 hand-chosen factors to a
~100-indicator TA-Lib factor zoo improve out-of-sample performance?

**Method.** Same point-in-time top-100 universe, survivorship-free, within-universe ranking, same
param-frozen walk-forward and 30 bps costs as the validated baseline. Composite = **equal-weight**
mean of cross-sectional percentile ranks; each factor oriented by an **economic-convention prior**
(a fixed +/- sign, *not* a return-fitted IC sign) — so weights are never fit on returns. The 100
factors are TA-Lib momentum / overlap-ratio / volatility / volume / statistic / cycle indicators
across several lookbacks (deliberately collinear — that is part of the test).

Harness: `scripts/research/k2_atlas/run_medallion_factors.py`.

## Result 1 — the naïve 100-factor "win" is a concentration artifact

| Composite (as-is) | dispersion | avg holdings | WF-OOS Sortino | CAGR | MaxDD |
|---|---:|---:|---:|---:|---:|
| 5-factor | 0.188 | 21.8 | 2.64 | 154% | −37% |
| 100-factor | 0.146 | **12.4** | **5.95** | **1497%** | **−55%** |
| 5 + 100 combined | 0.144 | 19.9 | 2.99 | 213% | −39% |

A walk-forward Sortino of **5.95** with a **1497% CAGR and −55% drawdown** is not better alpha — it
is an unstable, concentrated book. Averaging 100 mostly-collinear ranks **compresses** the composite
toward the mean (dispersion 0.188 → 0.146), so far fewer names clear the fixed 0.65 entry threshold;
the book holds **~12 names instead of ~22**, piles into a lucky handful, and produces an explosive
but fragile path. The −55% drawdown is the tell.

## Result 2 — apples-to-apples (equal selectivity) shows a small, real lift

Re-ranking each composite to uniform dispersion forces the **same number of holdings (~22)** at a
given threshold, so any remaining difference is signal, not concentration.

| Composite (re-ranked) | avg holdings | WF-OOS Sortino | Sharpe | CAGR | MaxDD | +vol-target |
|---|---:|---:|---:|---:|---:|---:|
| 5-factor | 22.0 | 2.52 | 1.92 | 143% | −36% | 2.59 |
| **100-factor** | 21.7 | **2.85** | 2.17 | 184% | −41% | 3.09 |
| 5 + 100 combined | 22.0 | 2.81 | 2.14 | 175% | −35% | 2.94 |
| *BTC buy & hold* | — | *1.78* | *1.15* | *54%* | *−50%* | — |

Equalized, the 100-factor Sortino **drops from 5.95 to 2.85** — a modest improvement over the
5-factor baseline (2.52), with a slightly *worse* drawdown (−41% vs −36%). The combined 5+100 is
similar (2.81). The vol-targeted figures move together (3.09 vs 2.59).

## Findings

1. **Most of the apparent benefit was a backtest artifact**, not signal. Without the holdings and
   re-rank diagnostics, one would have "discovered" a fake Sortino-6 / 1500%-CAGR strategy — the
   exact multiple-testing / overfitting trap the discipline (QF-21) exists to catch.
2. **A disciplined 100-factor expansion adds only a small, fragile lift** (+~0.3 Sortino,
   equal-selectivity), and it is *not* 100 independent bets — the indicators are highly collinear
   momentum/trend variants, so the effective new information is far less than the count implies.
   Net of the added complexity, overfitting surface, and worse drawdown, it does not clear the bar
   for adoption on this evidence.
3. **Breadth still dominates factor count.** Widening the universe (top-50 → top-100) added ~1.0 to
   honest Sortino; adding 95 factors adds ~0.3 at best. The opportunity set, not the indicator
   count, is the dependable lever for this strategy.

## Recommended follow-ups (not yet done)
- **Disciplined factor selection**, not a flat average: per-fold IC screen on the train window
  (orient + drop weak/redundant factors), then deflated-Sharpe / PBO multiple-testing correction.
  This is the principled way to ask whether *any* subset of the zoo beats the 5-factor model.
- **Decorrelation** (cluster the 100 factors, keep one representative per cluster) to test how many
  *independent* factors actually exist.

**Provenance:** `scripts/research/k2_atlas/run_medallion_factors.py`,
`coinbase_crypto_ohlcv_lake.duckdb`, TA-Lib 0.6.8, 30 bps, 2021-01..2026-06, OOS 2023+,
top-100 point-in-time universe.
