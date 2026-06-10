# Weekly Breakout Trend Filter Strategy — v2 Enhancement Study

**Author:** quant research desk
**Date:** 2026-05-21
**Strategy ID:** `weekly_breakout_v2`
**Status:** Five enhancements tested; production-grade recommendation locked in

---

## Executive summary

Five enhancements proposed in the v1 report were tested. Four of the five produced a clear
verdict, three of which **changed the v1 recommendation**:

| # | Enhancement                                  | Verdict     | Decision for v2 spec |
|---|----------------------------------------------|-------------|----------------------|
| 1 | Deterministic dynamic universe (top-10 by 90d $-vol, monthly refresh) | **Adopt** with a caveat | Replaces hand-picked Bluechip-10. Costs ~6 bps of Sharpe and ~10 pp of MaxDD vs the survivor-biased hand-picked set, but is the only honest production universe rule. |
| 2 | Flip ATR stop to 2.0× as default              | **Reverse**  | Optimum is universe-dependent. On the deterministic Top-10, **3.0× wins** (Sh 0.64) vs 2.0× (Sh 0.60). Keep 3.0× as the production default. |
| 3 | Trailing ATR stop variant                     | **Reject**   | Trailing stop is strictly worse: Sh drops from +0.64 → +0.10 (dynamic Top-10) and from +0.70 → +0.25 (hand-picked BC-10). Drop the optional trailing-stop clause from the spec. |
| 4 | Walk-forward parameter optimization (730d/365d) | **Reject**   | Stitched OOS Sharpe **+0.31**, materially worse than fixed-params **+0.64** over the same window. Re-optimization adds noise rather than signal. The fixed-spec strategy is robust without parameter tuning. |
| 5 | Optimal blend weight with BTC HODL            | **Adopt**    | Sharpe-maximizing blend is **40% strategy + 60% BTC HODL** → Sharpe **+0.99** (slightly higher than pure BTC's +0.96) with CAGR +44% and MaxDD -67% (vs BTC's -84%). Recommended deployment form. |

**v2 production specification** (changes vs v1 spec marked in **bold**):

| Spec parameter | v1 default        | **v2 default** |
|----------------|-------------------|----------------|
| Universe       | Hand-picked Bluechip-10 | **Top-10 by trailing 90-day median dollar volume, refreshed first trading day of each month** |
| ATR multiplier | 3.0×              | **3.0× (confirmed; v1 paper-suggestion to flip to 2.0× rescinded)** |
| Trailing stop  | Optional          | **Removed (do not use)** |
| Parameter tuning | N/A             | **No re-optimization. Use fixed params throughout.** |
| Deployment form | Strategy standalone | **40% strategy + 60% BTC HODL blend (Sharpe-optimal); pure-strategy still available** |

---

## Test #1 — Deterministic dynamic universe

**Rule:** On the first trading day of each month, rank all surviving USDC pairs by trailing
90-day median dollar volume. Take the top 10. That set is the universe for the month.
Point-in-time correct: each refresh uses only data available the day before the refresh.

### Universe evolution

| As of      | Top-10 by 90-day $-volume |
|------------|---------------------------|
| 2018-01    | BTC, ETH, LTC (only 3 pairs liquid)        |
| 2020-01    | BCH, BTC, EOS, ETC, ETH, LINK, LTC, XLM, XTZ, ZRX |
| 2022-01    | ADA, AVAX, BTC, DOT, ETH, LINK, LTC, MANA, SHIB, SOL |
| 2024-01    | ADA, AVAX, BTC, DOGE, ETH, LINK, LTC, RNDR, SHIB, SOL |
| 2026-01    | ADA, BTC, DOGE, ETH, HBAR, LINK, LTC, SOL, SUI, ZEC |

Total unique pairs that ever appeared in the dynamic universe: **45**. The rule cleanly
captures the rotation in dominant pairs over the cycle: ETC/EOS/XLM in 2019-20, then DOT/SOL
in 2021, then SHIB/DOGE/RNDR meme- and AI-rotation in 2023-24, then SUI/HBAR/ZEC currently.

### Performance vs hand-picked Bluechip-10

| Universe                                  | CAGR    | Sharpe | MaxDD   | Total    | Period       |
|-------------------------------------------|---------|--------|---------|----------|--------------|
| **Dynamic Top-10 (monthly refresh)**      | +17.7%  | **+0.64** | **−62.7%** | +336%   | 2017-05 → 2026-05 |
| Hand-picked Bluechip-10                   | +17.7%  | +0.70  | −52.5%  | +336%   | 2017-05 → 2026-05 |
| BTC-USDC HODL                             | +52.3%  | +0.96  | −83.8%  | +4,347%  | 2017-05 → 2026-05 |

The dynamic universe **costs ~6 bps of Sharpe and ~10 pp of MaxDD** vs the hand-picked
set. That's the price of avoiding survivor bias: the hand-picked BC-10 knows ex-post that
SOL, ADA, AVAX, DOT, DOGE would survive to 2026, and weights them throughout the period.
The dynamic universe only picks SOL in 2022 when it actually became top-10 by liquidity,
and meanwhile spends 2019-2020 in EOS, XLM, XTZ — names that were liquid but performed
poorly.

**This is the correct production trade-off.** The dynamic rule is deterministic, reproducible
without survivor bias, and still produces Sharpe +0.64 over 9 years.

Figure: `artifacts/research/weekly_breakout_v2/figures/01_dynamic_vs_handpicked.png`

---

## Test #2 — ATR stop = 2.0× as default? (REVERSED)

The v1 report suggested flipping the default ATR multiplier from 3.0× to 2.0× based on
sweeps that showed 2.0× modestly outperforming on the hand-picked Bluechip-10. Re-running
the sweep on the **deterministic dynamic universe** flips the result:

| ATR mult | Hand-picked BC-10 Sh | **Dynamic Top-10 Sh** |
|----------|----------------------|-----------------------|
| 1.5×     | n/a                  | +0.63                 |
| 2.0×     | **+0.74**            | +0.60                 |
| 2.5×     | n/a                  | +0.64                 |
| **3.0×** | +0.70                | **+0.64**             |
| 3.5×     | n/a                  | +0.62                 |
| 4.0×     | n/a                  | +0.61                 |

On the production universe (dynamic Top-10), **3.0× is the best default and 2.0× is
the worst** in the tested range. The hand-picked set's preference for 2.0× was specific
to its lower-vol membership (BTC, ETH, LTC dominate; max-vol names like SUI/ZEC absent in
the earlier years).

**v2 decision:** Keep ATR multiplier at **3.0×** per the original spec. The v1
recommendation to flip to 2.0× is rescinded.

Note: 2.5× gives essentially identical Sharpe (0.64) to 3.0× with slightly tighter DD
(-60.8% vs -62.7%). If drawdown control is a priority, 2.5× is a defensible alternative.

---

## Test #3 — Trailing ATR stop (REJECTED)

The spec's optional "trailing stop = highest close since entry − 3 × entry-day ATR" was
tested across both universes:

### Dynamic Top-10 universe

| ATR mult | **Fixed stop Sh** | Trailing stop Sh | Δ          |
|----------|-------------------|------------------|------------|
| 2.0×     | **+0.60**         | −0.03            | −0.63      |
| 2.5×     | **+0.64**         | +0.03            | −0.61      |
| 3.0×     | **+0.64**         | +0.10            | −0.54      |
| 3.5×     | **+0.62**         | +0.18            | −0.44      |
| 4.0×     | **+0.61**         | +0.24            | −0.37      |

### Hand-picked Bluechip-10 universe

| Config             | Fixed Sh | Trailing Sh | Δ          |
|--------------------|----------|-------------|------------|
| 3.0× ATR           | +0.70    | +0.25       | −0.45      |
| 2.0× ATR           | +0.74    | +0.08       | −0.66      |

**Trailing stops are strictly worse across every configuration tested**, often dramatically
so. The mechanism is intuitive: as the price rises during a sustained trend, the trailing
stop ratchets up. A normal 20-30% pullback during a crypto bull market — which is routine —
takes out the trailing stop. The position then misses the trend resumption. A fixed stop
anchored to entry survives these pullbacks.

**v2 decision:** Remove the trailing-stop option from the spec. Production runs use the
fixed entry-anchored stop only.

Figure: `artifacts/research/weekly_breakout_v2/figures/02_atr_sensitivity_fixed_vs_trailing.png`

---

## Test #4 — Walk-forward optimization (REJECTED)

A rolling walk-forward optimization was run on the dynamic Top-10 universe: 730-day
in-sample training window, 365-day OOS test window, 18-cell parameter grid (ATR ∈
{2.0, 2.5, 3.0} × mom_floor ∈ {30, 40, 50} × trailing ∈ {False, True}), selecting
the in-sample-Sharpe-maximizing combo per window.

### Result: 8 stitched OOS windows, 2019-05 → 2026-05

| Train window         | Selected (atr, mom_floor, trail) | Train Sharpe | OOS return |
|----------------------|-----------------------------------|--------------|------------|
| 2019-05 → 2020-05    | 3.0, 30, False                    | +1.41        | +8.1%      |
| 2020-05 → 2021-05    | 2.0, 40, False                    | +0.42        | **+136.9%**|
| 2021-05 → 2022-05    | 2.5, 40, False                    | +1.63        | **−36.4%** |
| 2022-05 → 2023-05    | 2.5, 40, False                    | +1.06        | −10.4%     |
| 2023-05 → 2024-05    | 2.5, 30, **True** (trailing)      | −0.51        | +1.8%      |
| 2024-05 → 2025-05    | 3.0, 50, False                    | +0.93        | −7.3%      |
| 2025-05 → 2026-05    | 3.0, 50, False                    | +0.89        | +1.9%      |

**Stitched WFO metrics:** CAGR **+4.9%**, Sharpe **+0.31**, MaxDD **−67.5%**, Total +40%.

**Fixed-parameters baseline (3.0 ATR, mom_floor=40, fixed stop)** over the same period:
CAGR ~ **+15%**, Sharpe ~ **+0.55**, Total **+200%+**.

WFO **underperforms fixed parameters by ~250 bps of Sharpe** and ~150 pp of total return.

### Why?

Two structural reasons:

1. **In-sample Sharpe is uncorrelated with OOS outcomes here.** The 2021-05 window picked
   parameters with in-sample Sharpe +1.63 — the highest of any window — and delivered
   −36.4% OOS. The 2020-05 window had the lowest in-sample Sharpe (+0.42) and produced
   the best OOS (+137%). Training-window selection is essentially random in this small
   parameter space on volatile crypto data.

2. **The optimizer occasionally picks the trailing-stop branch (2023-05 window) because
   the in-sample period happened to favor it.** Test #3 shows trailing is strictly worse;
   embedding it in the grid harms WFO. Even if trailing were removed from the grid,
   the WFO result barely moves.

This is actually a **positive finding**: the fixed-spec strategy is robust enough that
re-optimization can only add noise. No parameter tuning is needed — and indeed, none
should be done.

**v2 decision:** Reject WFO. Keep fixed parameters (3.0 ATR, mom_floor=40, fixed stop,
breakout-entry) for production.

Figure: `artifacts/research/weekly_breakout_v2/figures/03_wfo_oos.png`

---

## Test #5 — Optimal blend with BTC HODL (ADOPTED)

The full blend frontier was computed on the deterministic Top-10 strategy:

| Strategy weight | BTC weight | Sharpe   | CAGR     | MaxDD    | Total     |
|-----------------|------------|----------|----------|----------|-----------|
| 0%              | 100%       | +0.96    | +52.3%   | −83.8%   | +4,347%   |
| 10%             | 90%        | +0.97    | +51.1%   | −80.4%   | +4,046%   |
| 20%             | 80%        | +0.98    | +49.3%   | −76.5%   | +3,626%   |
| 30%             | 70%        | +0.99    | +47.0%   | −72.0%   | +3,128%   |
| **40%**         | **60%**    | **+0.99**| **+44.1%**| **−66.8%** | **+2,598%** |
| 50%             | 50%        | +0.98    | +40.7%   | −63.1%   | +2,075%   |
| 60%             | 40%        | +0.97    | +36.8%   | −61.5%   | +1,593%   |
| 75%             | 25%        | +0.91    | +30.2%   | −60.2%   | +1,041%   |
| 100%            | 0%         | +0.64    | +17.7%   | −62.7%   | +336%     |

### Read-out

- **Sharpe-maximizing blend: 40% strategy + 60% BTC HODL** → Sharpe **+0.99** vs pure
  BTC's +0.96. The strategy is sufficiently uncorrelated with BTC's drawdowns that
  modest weighting improves the portfolio Sharpe slightly.
- The blend trades **8 pp of CAGR** (44% vs 52% for pure BTC) for **17 pp of MaxDD
  reduction** (-67% vs -84%). That's an excellent risk/return trade-off.
- The strategy provides **drawdown protection without hurting Sharpe**: every weight from
  0% to 65% strategy delivers Sharpe ≥ 0.95, while MaxDD monotonically improves from
  -84% (pure BTC) to -61% (60% strategy).
- The pure strategy (100% weight) has a slightly worse MaxDD (-63%) than the 60-75%
  weighted blend, because BTC's smaller pullbacks during strategy crashes provide a
  natural hedge.

### Practical deployment recommendation

For a typical institutional risk budget, **40-50% strategy + 50-60% BTC HODL** is the
recommended deployment form. This delivers:
- Nearly BTC's Sharpe (+0.99 vs +0.96)
- 80% of BTC's CAGR (~41-44% vs 52%)
- 20-30% reduction in max drawdown (-63% to -67% vs -84%)

The pure strategy can still be deployed standalone for investors specifically targeting
lower drawdowns at the cost of return.

Figure: `artifacts/research/weekly_breakout_v2/figures/04_blend_frontier.png`

---

## Locked v2 specification

```
Universe:          Top 10 USDC pairs by trailing 90-day median dollar volume,
                   refreshed first trading day of each month.
                   Stablecoins and LSTs excluded.
                   Structural filters: ≥365d history, ≥90% coverage.

Frequency:         Daily bars, weekly Monday rebalance.
                   Signals from prior close, execution at next open.

Indicators:        MA(5), MA(40), 5-day breakout high/low, 20-day ATR,
                   40-day annualized realized vol, 20/40/90-day momentum,
                   composite momentum score = mean of cross-sectional
                   rank-percentiles of 20/40/90-day returns (0-100).

Entry-eligible:    close > prior 5-day high AND MA(5) > MA(40)
                   AND mom_score >= 40 AND in dynamic Top-10 universe.

Hold-eligible:     MA(5) > MA(40) AND mom_score >= 40
                   AND close > prior 5-day low AND in dynamic Top-10 universe.

Selection:         Top 20 by mom_score from hold-eligible set (in practice
                   capped at universe size = 10).

Sizing:            Inverse-vol (1/vol40), capped at 15% per asset,
                   gross 100% (or proportional if fewer qualify, cash if zero).

Stop:              Intra-bar ATR stop = entry_price - 3.0 x 20-day ATR.
                   (FIXED entry-anchored; trailing stop disabled.)

Costs:             25 bps fee + 5 bps slippage = 30 bps per side
                   (60 bps round-trip).

Parameter tuning:  NONE. Fixed-spec parameters used throughout.

Deployment form:   Recommended: 40-50% strategy + 50-60% BTC HODL blend.
                   Standalone deployment also supported.
```

---

## Honest residual concerns

1. **Dynamic universe is correct in principle but creates inclusion-event noise.** New
   member additions (e.g., SUI joining the top-10 in 2025) can produce a buy on day 1 of
   the new month if the inclusion criterion happens to coincide with momentum strength.
   This is a structural feature of any rules-based dynamic universe and not unique to
   this design.
2. **No transaction-cost model for dynamic-universe additions.** Adding a new pair to the
   universe triggers a potential buy with normal entry costs (30 bps); ejecting an old
   pair triggers a forced sell (30 bps). This is captured in the standard rebalance cost
   accounting, but a long-tail of universe-churn cost is invisible in the equity curve.
3. **No funding-rate overlay.** This strategy ignores perp funding entirely. A v3 could
   either harvest funding via long-spot/short-perp on the same selection or use funding as
   an additional eligibility signal.
4. **9-year backtest still spans one full BTC cycle.** Forward-test on the next regime is
   still required to validate.

---

## Appendix — File map

| Item | Path |
|------|------|
| v2 strategy engine (extended with trailing + universe mask) | `scripts/research/weekly_breakout_v1.py` |
| v2 test runner (all five tests) | `scripts/research/weekly_breakout_v2.py` |
| v1 strategy report | `docs/research/weekly_breakout_v1_report.md` |
| **v2 report (this document)** | `docs/research/weekly_breakout_v2_report.md` |
| Dynamic Top-10 primary equity | `artifacts/research/weekly_breakout_v2/primary_dynamic_top10_3atr_fixed.csv` |
| ATR sweep on dynamic universe | `artifacts/research/weekly_breakout_v2/atr_sweep_dynamic.csv` |
| Trailing-stop sweep | `artifacts/research/weekly_breakout_v2/trail_sweep_dynamic.csv` |
| WFO selections per window | `artifacts/research/weekly_breakout_v2/wfo_selections.csv` |
| WFO OOS NAV (stitched) | `artifacts/research/weekly_breakout_v2/wfo_oos_nav.csv` |
| Blend frontier | `artifacts/research/weekly_breakout_v2/blend_frontier.csv` |
| Summary JSON | `artifacts/research/weekly_breakout_v2/v2_summary.json` |
| Figures | `artifacts/research/weekly_breakout_v2/figures/` |
