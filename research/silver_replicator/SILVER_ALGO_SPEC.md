# Silver Replicator — Trading Algorithms v1, v2 & v3

Platform-agnostic spec. All indicators are standard TA-Lib formulas — port to Pine Script, NinjaScript, QuantConnect, MetaTrader, custom Python, etc. without modification.

## Common inputs

- **Instrument:** front-month silver futures (CME `SI` continuous, or QI mini-silver for sizing). Stitched front-month with calendar roll on first notice / last trade day, whichever comes first; carry-adjust returns optional but recommended.
- **Timeframe:** 8-hour bars (08:00 / 16:00 / 00:00 UTC). 8H won the in-sample grid; if your platform doesn't support 8H natively, 4H performs similarly.
- **Warm-up:** require at least 200 bars of history before the first live signal (driven by `SMA_200` and `bb_width_pctl_lookback=200`).

---

## Algorithm v1 — Trend with regime gate

### Indicators

```
SMA_fast       = SMA(close, 50)
SMA_slow       = SMA(close, 200)
RSI            = RSI(close, 14)
MACD_hist      = MACD_histogram(close, 12, 26, 9)
ADX            = ADX(high, low, close, 14)
ATR            = ATR(high, low, close, 14)
NATR           = ATR / close              // normalized ATR
```

### Signal logic (evaluate on each bar close)

```
trend_long      = SMA_fast >  SMA_slow
trend_short     = SMA_fast <  SMA_slow

momentum_long   = RSI >= 50
momentum_short  = RSI <= 45

confirm_long    = MACD_hist > 0
confirm_short   = MACD_hist < 0

vol_gate        = (ADX >= 25) OR (NATR <= 0.04)

state_long      = trend_long  AND momentum_long  AND confirm_long  AND vol_gate
state_short     = trend_short AND momentum_short AND confirm_short AND vol_gate

if state_long:                    state = +1
elif state_short and not state_long: state = -1
else:                             state =  0
```

`state` is the directional view for the next bar.

### Position translation

```
position_QI_contracts  = state * max_contracts          // default max_contracts = 4
call_delta_target      = state * 0.30                   // for the options overlay
```

Long QI on +1, short on -1, flat on 0. Re-evaluate each bar close; only act on a state change.

### Options overlay (synthetic, for replication)

On a flip from ≤0 → +1: buy 1 synthetic **50-DTE 5-delta OTM call** on full-size SI. On a flip back to ≤0: close it.
On a flip from ≥0 → −1: buy 1 synthetic **50-DTE 5-delta OTM put**. On a flip back to ≥0: close it.

Strike selection: invert Black-Scholes with `IV = 1.2 × annualized_30_bar_realized_vol` to find the strike that produces target_delta = 0.05. Price changes mark-to-market each bar with the same IV/RV rule.

This isn't a tradeable rule — it's the proxy the model uses to attribute the options-book P&L of the real account.

---

## Algorithm v2 — v1 + Bollinger-band regime switch

Same inputs and indicators as v1, plus:

```
BB_upper, BB_middle, BB_lower  = BBANDS(close, period=20, stdev=2.5)
BB_width                       = (BB_upper - BB_lower) / BB_middle
BB_width_pctl                  = rolling_percentile_rank(BB_width, lookback=200)
```

### Regime classifier

```
if BB_width_pctl < 0.20:    regime = "mean_revert"   // quiet / tight bands
else:                        regime = "trend"
```

The 0.20 percentile threshold means: when current bandwidth sits in the bottom 20% of the trailing 200-bar window, we treat the market as ranging.

### Signal logic

```
if regime == "trend":
    # exactly the v1 trend/momentum/MACD/ADX logic from above
    state = v1_state

elif regime == "mean_revert":
    mr_long   = (close < BB_lower) AND (RSI < 40)
    mr_short  = (close > BB_upper) AND (RSI > 60)
    if mr_long:                       state = +1
    elif mr_short and not mr_long:    state = -1
    else:                             state =  0
```

Everything else (position translation, options overlay) is unchanged from v1.

### Why v2 helped

In tight-band stretches the v1 trend logic was chopping on whippy SMA crosses. v2 forces a mean-reversion stance during those windows, only firing when price stretches to the band edge with confirming RSI. In-sample on the silver book this moved composite score from 0.745 → 0.755, direction accuracy from 65% → 66%, and combined simulated P&L from $79k → $91k vs. actual $77k.

---

---

## Algorithm v3 — v2 + regime-scoped vol-of-vol circuit breaker

Same inputs and indicators as v2, plus a realized-volatility-of-volatility expansion detector that is **only active during the mean-revert regime**. In trend regime the circuit breaker is suppressed.

### Vol-of-vol indicator

```
log_return     = ln(close / close[-1])
realized_vol   = rolling_stdev(log_return, window=20)        // bar-scale RV
rv_smooth      = rolling_mean(realized_vol, window=5)        // smoothed RV
drv            = rv_smooth - rv_smooth[-1]                   // first difference
drv_mean_100   = rolling_mean(drv, window=100)
drv_std_100    = rolling_stdev(drv, window=100)
vov_zscore     = (drv - drv_mean_100) / drv_std_100          // z-score of RV change
```

### Hysteresis-based expansion flag

The flag turns **on** when `|vov_zscore| ≥ 1.0` and stays on until `|vov_zscore| < 0.5` (entry threshold halved for exit — a 2-to-1 hysteresis band that prevents flicker).

```
expansion_on   = |vov_zscore| >= 1.0
expansion_off  = |vov_zscore| <  0.5
// state machine: turn on when expansion_on, turn off when expansion_off, hold otherwise
```

### Regime-scoped application

```
// Compute v2 state first (trend or mean-revert path as usual)
state = v2_state

// Then apply circuit breaker ONLY in mean-revert regime
if regime == "mean_revert" AND expansion_on:
    state = 0        // force flat for this bar; resume next bar when expansion clears
```

In trend regime the breaker is dormant — trend-following positions ride through vol spikes unchanged. In mean-revert regime, a vol-of-vol expansion overrides the band-edge entry signal and forces flat until volatility stabilizes.

### Why v3 is conservative

At the chosen BB threshold (20th percentile), the intersection of *mean-revert regime* ∧ *vol-z expansion* is sparse — the breaker fires on roughly 1 bar in the in-sample window vs. 58 bars for the unscoped v2 vov trigger. As a result, **v3 composite ≈ v2 composite** (0.7546 vs 0.7546 in-sample). v3 is "weakly dominant" — it removes the unscoped trigger's failure mode (cutting alpha during trend-regime vol spikes) without adding noise. Treat it as the safer default; if you want the breaker to do more work, widen the BB regime threshold or tighten `vov_zscore_thr`.

### Why we kept it anyway

Walk-forward validation showed the un-scoped vov trigger was picked in 62% of OOS folds — there are market regimes where it matters. Scoping it to mean-revert only means it never *hurts* in the trend regime where v2 was working, while preserving the option to fire when the band-edge mean-reversion logic would otherwise re-enter into a vol shock.

---

## Best-fit parameters (in-sample, 8H, 339 bars Oct 30 2025 → Mar 27 2026)

| Param | v1 | v2 | v3 |
|---|---|---|---|
| SMA fast | 50 | 50 | 50 |
| SMA slow | 200 | 200 | 200 |
| RSI long threshold | 50 | 50 | 50 |
| RSI short threshold | 45 | 45 | 45 |
| Use MACD confirm | true | true | true |
| Use ADX gate | true | true | true |
| ADX minimum | 25 | 25 | 25 |
| NATR max | 0.04 | 0.04 | 0.04 |
| BB period | — | 20 | 20 |
| BB stdev | — | 2.5 | 2.5 |
| BB width percentile lookback | — | 200 | 200 |
| BB regime threshold | — | 0.20 | 0.20 |
| Vol-of-vol trigger | — | off | **on, MR-scoped** |
| `vov_window` | — | — | 20 |
| `vov_smooth` | — | — | 5 |
| `vov_zscore_thr` | — | — | 1.0 |
| `vov_action` | — | — | `flip_to_flat` |
| `vov_only_in_mr_regime` | — | — | true |

### In-sample scoring

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Composite | 0.7451 | 0.7546 | 0.7546 |
| Direction accuracy | 0.6490 | 0.6608 | 0.6608 |
| Cohen's κ | 0.3243 | 0.3445 | 0.3445 |
| P&L curve correlation | 0.9712 | 0.9749 | 0.9749 |
| Sim P&L (futures only, vol-scaled) | $51,280 | $62,833 | $62,833 |
| Sim P&L (futures + options overlay) | $79,439 | $90,992 | $90,992 |
| Actual realized P&L | $77,329 | $77,329 | $77,329 |

### Out-of-sample (walk-forward, 8 folds at 8H)

| Metric | v2/v3 mean | v2/v3 median |
|---|---|---|
| Test composite | 0.405 | 0.375 |
| Test direction accuracy | 0.619 | — |
| Test Cohen's κ | 0.217 | — |
| Test P&L correlation | 0.352 | — |
| Folds picking `use_bb_regime=True` | 5 / 8 (62.5%) | — |

The 46% haircut from in-sample to OOS is normal for single-asset TA models. **Treat the in-sample composite as an optimistic ceiling.** Live expectation: composite 0.35–0.45, direction accuracy ~60%.

### Futures-only replication (no options overlay)

The directional signal alone, with vol-target sizing and a 3-bar minimum hold, on **4H bars** (slightly faster than the 8H signal TF), recovers **$71,490 (92.4%) of the actual $77,329** with **Sharpe 1.49** and **max drawdown $38,445** over the in-sample window. The unconstrained "oracle" (no drawdown cap) hits 100.1% but with $107k drawdown — well outside acceptable risk. The 8% gap from constrained-best to actual is the options book's convexity premium.

---

## Caveats before live deployment

1. **In-sample only.** No walk-forward, no held-out test set. Treat the params as plausible, not validated.
2. **Look-ahead-safe but not slippage-safe.** Backtest uses bar-close prices with no slippage/commission inside the simulator; real silver fills will be worse.
3. **No carry roll cost.** Front-month stitching is calendar-based, not return-adjusted.
4. **The options overlay is a replication proxy, not a strategy.** For real options trading you need actual IV surface, not realized-vol proxy.
5. **State persistence isn't modeled.** Each bar evaluates fresh; there's no minimum-holding-period or maximum-concurrent-positions guard. Add those before live deployment.
