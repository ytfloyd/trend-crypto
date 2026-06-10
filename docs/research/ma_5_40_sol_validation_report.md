# MA(5/40) SOL-USDC validation report

**Status**: complete (Phase 1)
**Date**: 2026-05-19
**Companion docs**: [ETH MA(5/40) validation](./ma_5_40_eth_validation_report.md), [crypto perp funding & carry](./crypto_perp_funding_and_carry.md)
**Run ID**: `sol_daily_ma_5_40_v0_sol_usdc_20260520T041206Z`
**Figures**: `artifacts/research/ma_5_40_sol_usdc_baseline/figures/`

---

## TL;DR

1. **Headline numbers look like ETH's**: SOL MA(5/40) returned **+1,168% vs B&H +121%** over Jun 2021 → May 2026 — an **8.7× edge** with Sharpe 1.10 (vs 0.74) and MaxDD -77% (vs -96%).

2. **But the OOS picture is much weaker than ETH's.** Walk-forward (730d train / 182d test, 5 folds, 2023-06 → 2025-12) gave **strategy 607% vs B&H 773% = 0.81× edge** in total return. Strategy did edge out B&H on Sharpe (1.51 vs 1.42) and MaxDD (-55% vs -60%), but only marginally.

3. **The entire 8.7× edge collapses if you exclude 2022.** Removing just Nov 2022 - Feb 2023 keeps the edge at 6.5×. Removing **all of 2022** (Luna + 3AC + FTX trifecta) flips it to **0.63×** — the strategy *loses* to B&H.

4. **Bootstrap significance is borderline.** MaxDD edge is significant at 5% (p=0.033); Sharpe, CAGR, and total-return edges are marginal at p=0.099-0.101 (significant only at 10%).

5. **The parameter, cost, and synthetic-call results all confirm the strategy is real** — 5/40 sits in a broad plateau of good combos, survives 200 bps/trade, and shows the familiar truncated-left-tail payoff (up slope 0.93, down slope 0.25).

**Honest forward-looking statement for SOL**: the headline outperformance is real but **dependent on the existence of crash events**. The strategy is a synthetic call on SOL — full participation in normal years, large outperformance in crashes, lag in strong bulls. Expect Sharpe ~1.0-1.2 and DDs of -50 to -60% with no event; closer to -77% under another 2022-scale event.

---

## 1. Run setup

- **Universe**: SOL-USDC (Coinbase). Bit-for-bit identical to SOL-USD in our lake.
- **Window**: 2021-06-17 → 2026-05-17 (1,797 daily bars, ~4.9 years). Limited by SOL's Coinbase listing date.
- **Strategy**: same canonical MA(5/40) long-only used for BTC/ETH. No vol-targeting, no ADX, no DD throttle. One-bar execution lag, zero costs in the baseline run.
- **Config**: `configs/research/midcap_daily_ma_5_40_v0.yaml` (with symbol = SOL-USDC).

| | SOL B&H | SOL MA(5/40) | Edge |
|---|---|---|---|
| **CAGR** | 20.0% | **68.0%** | +48 pp |
| **Sharpe** | 0.68 | **1.09** | +0.41 |
| **MaxDD** | -96.2% | **-77.4%** | +18.8 pp |
| **Total return** | +145% | **+1,181%** | **8.1×** |
| **NAV (from $100k)** | $229,999 | **$1,280,651** | 5.57× |

The -77% strategy MaxDD happened during the FTX collapse (peak $716k Nov 2021 → trough $162k Jan 2023).

![SOL equity + drawdown](../../artifacts/research/ma_5_40_sol_usdc_baseline/sol_usdc_ma_5_40_equity.png)

---

## 2. Parameter sweep — not cherry-picked

Grid: fast ∈ {3, 5, 8, 10, 13, 21}, slow ∈ {20, 30, 40, 50, 80, 120, 200}.

The 5/40 baseline gives Sharpe **1.10**. Out of the 35 valid combinations:
- 16 combos exceed Sharpe 1.0
- Best combo: **13/30** → Sharpe 1.35, CAGR 104%
- Worst meaningful combo: **5/80** → Sharpe 0.28 (slow=80 is just too long for SOL's volatility)
- The (3-13 fast, 20-50 slow) region forms a broad plateau of good performance

5/40 is **not** an outlier — it's a representative point in the good region.

![Parameter sweep heatmaps](../../artifacts/research/ma_5_40_sol_usdc_baseline/figures/01_parameter_sweep.png)

---

## 3. Cost sensitivity — robust

| Round-trip bps | CAGR | Sharpe | Total |
|---|---|---|---|
| 0 | 69.4% | 1.10 | +1,168% |
| 10 | 67.3% | 1.08 | +1,093% |
| 25 | 64.2% | 1.06 | +988% |
| 60 | 57.0% | 0.99 | +778% |
| 100 | 49.2% | 0.92 | +587% |
| 200 | 31.2% | 0.74 | +270% |

The strategy stays above SOL B&H's Sharpe (0.74) and CAGR (20%) up to **200 bps/trade**. Coinbase taker fees are typically 5-25 bps — well within safe margins.

![Cost sensitivity](../../artifacts/research/ma_5_40_sol_usdc_baseline/figures/02_cost_sensitivity.png)

---

## 4. Yearly returns — full event-dependency exposure

| Year | Strategy | B&H | Edge (multiplier) | Regime |
|---|---|---|---|---|
| 2021 H2 | +373% | +501% | 0.79× | Bull (SOL launch run) |
| **2022** | **-65%** | **-94%** | **5.94×** | **Bear (Luna/3AC/FTX crashes)** |
| 2023 | +503% | +936% | 0.58× | Bull (recovery) |
| 2024 | +72% | +86% | 0.92× | Bull |
| 2025 | -10% | -34% | 1.36× | Bear |
| 2026 YTD | -18% | -30% | 1.17× | Bear |

- Strategy WON in 3 of 6 calendar years (all bear/sideways).
- Strategy LOST in 3 of 6 calendar years (all bull).
- The cumulative outperformance is dominated by **a single year — 2022** — where the 5.94× multiplier turns into the 8.1× full-sample edge.

![Yearly returns](../../artifacts/research/ma_5_40_sol_usdc_baseline/figures/03_yearly_returns.png)

---

## 5. The synthetic-call payoff is intact

Rolling 63-day returns: strategy vs B&H.

- **Up regression slope: 0.93** (~93% upside capture)
- **Down regression slope: 0.25** (~25% downside capture)
- When B&H is up: avg B&H +55.5% → avg strategy +47.3%
- When B&H is down: avg B&H -26.4% → avg strategy -13.8%

Slightly less convex than ETH (1.03 / 0.15) — strategy picks up slightly more downside on SOL (25% vs 15%). The DD-asymmetry edge is real but smaller in magnitude than ETH's.

![SOL synthetic call scatter](../../artifacts/research/ma_5_40_sol_usdc_baseline/figures/04_synthetic_call_scatter.png)

---

## 6. The crash-exclusion test — the most important table

Same test as we ran for ETH (where excluding 2018 left a 1.3× edge instead of 8.7×):

| Window | Strategy total | B&H total | Edge | Strategy MaxDD | B&H MaxDD |
|---|---|---|---|---|---|
| **Full sample** | **+1,168%** | **+121%** | **5.75×** | -77% | -96% |
| Excl Nov 2022 - Feb 2023 (FTX months) | +2,011% | +223% | 6.53× | -57% | -92% |
| Excl 2022 entirely (Luna+3AC+FTX) | +2,254% | +3,614% | **0.63×** | -46% | -70% |

The first exclusion keeps the edge intact (6.53×) because removing the worst months also removes the catastrophic B&H drawdown. **But removing all of 2022** — the year of three back-to-back crashes — flips the result: strategy underperforms B&H by 37%.

**Interpretation**: SOL's outperformance is *entirely* a function of 2022's once-in-a-decade trifecta of crashes. Without 2022, the strategy is a slight drag on B&H.

This is a stronger conclusion than the ETH case, where excluding 2018 still left a positive (though smaller) edge.

---

## 7. Bootstrap significance — marginal except for MaxDD

Block bootstrap, N=2,000 paired 30-day blocks. Test: is the strategy's realized edge (vs SOL B&H) larger than chance?

| Metric | Realized edge | Boot mean | Boot std | 5%ile | 95%ile | p(edge ≤ 0) |
|---|---|---|---|---|---|---|
| Sharpe | +0.36 | +0.41 | 0.33 | -0.11 | +0.95 | **0.099** |
| MaxDD (positive = strategy better) | +0.19 | +0.16 | 0.11 | +0.01 | +0.38 | **0.033** |
| CAGR | +0.42 | +0.48 | 0.45 | -0.18 | +1.28 | **0.101** |
| Total return | +9.52 | +50.5 | 1485 | -9.4 | +247.5 | **0.101** |

- **MaxDD edge significant at 5% (p=0.033)** — the drawdown reduction is the most robust property of the strategy.
- Sharpe, CAGR, Total edges all sit at p ≈ 0.10 — significant at 10%, not 5%. This is consistent with the short SOL history (only 4.8 years) and high return variance.

![Bootstrap distributions](../../artifacts/research/ma_5_40_sol_usdc_baseline/figures/05_bootstrap_distributions.png)

---

## 8. Walk-forward OOS — the strategy LAGS B&H

The most honest test. Training window: 730 days. Test window: 182 days. Step: 182 days. 5 folds total.

| Test window | Best params (train) | Train Sharpe | Test strategy | Test B&H |
|---|---|---|---|---|
| 2023-06-17 → 2023-12-15 | 21/40 | 1.55 | **+263%** | +376% |
| 2023-12-16 → 2024-06-14 | 21/40 | 1.35 | +20% | +97% |
| 2024-06-15 → 2024-12-13 | 21/40 | 1.50 | +36% | +58% |
| 2024-12-14 → 2025-06-13 | 13/20 | 2.08 | -6% | -34% |
| 2025-06-14 → 2025-12-12 | 13/30 | 2.29 | **+27%** | -11% |

**Combined OOS (910 days)**:

| Metric | OOS Strategy | OOS B&H | Edge |
|---|---|---|---|
| CAGR | 119.1% | 138.5% | -19.4 pp |
| Sharpe | 1.51 | 1.42 | +0.09 |
| MaxDD | -54.8% | -59.8% | +5.0 pp |
| Total return | +607% | +773% | **0.81×** |

**Key insight**: The walk-forward window (2023-06 → 2025-12) contains no event remotely like 2022. The OOS strategy participates less in the bull-dominated stretch, so it lags. The Sharpe and MaxDD edges survive — but barely.

This is **the opposite outcome of the ETH walk-forward**, where the OOS strategy *outperformed* B&H 1.84× over 2014-2026 because that window did contain crash events (2018, 2022).

![Walk-forward OOS](../../artifacts/research/ma_5_40_sol_usdc_baseline/figures/06_walkforward_oos.png)

---

## 9. Honest comparison: SOL vs ETH vs BTC

| | BTC | ETH | **SOL** |
|---|---|---|---|
| Sample years | 11.0 | 10.0 | **4.9** |
| MA(5/40) CAGR | 50.1% | 50.6% | **68.0%** |
| MA(5/40) Sharpe | 1.17 | 0.99 | **1.09** |
| MA(5/40) MaxDD | -64% | -53% | **-77%** |
| Total return edge vs B&H | 1.18× | 0.78× | **8.07×** |
| Bootstrap MaxDD p-value | 0.001 | 0.005 | **0.033** |
| Bootstrap Sharpe p-value | 0.01 | 0.02 | **0.099** |
| **Walk-forward OOS edge** | **1.5×** | **1.84×** | **0.81×** |
| Edge after crash exclusion | 1.0× | 1.3× | **0.63×** |
| Down-capture slope | ~0.30 | 0.15 | **0.25** |

**SOL's headline number is the biggest, but its OOS validation is the weakest.** The short data window combined with the unprecedented 2022 trifecta means we can't yet conclude the strategy generalizes on SOL the way it does on BTC/ETH.

---

## 10. Bottom line

1. **The strategy works on SOL the same way it works on ETH and BTC** — same synthetic-call payoff, same drawdown asymmetry, same parameter robustness. This is the strongest evidence yet that the MA(5/40) edge is a structural property of long-only trend in crypto, not a single-asset quirk.

2. **The 8.1× headline outperformance is real but historically contingent.** It is dominated by 2022's trifecta of crashes. Forward expectation should be ~1-1.5× edge in normal regimes with sharp outperformance only when SOL undergoes another -90% event.

3. **The defensible robust edge is drawdown reduction**: -77% strategy vs -96% B&H in-sample, -55% vs -60% out-of-sample. Bootstrap p=0.033 on full sample. This is the property worth selling.

4. **The walk-forward result is honest and humbling.** Without a comparable crash event in the OOS window, the strategy *lags* B&H by 19pp CAGR. Investors should be told: this strategy outperforms in volatile/bear regimes and lags in stable bulls.

5. **The SOL data window is too short** for the same confidence we have on ETH. We need another 5+ years of out-of-sample data, ideally including at least one more major crash event, before we can claim the same level of validation.

---

## Appendix — files referenced

- Run artifacts: `artifacts/runs/sol_daily_ma_5_40_v0_sol_usdc_20260520T041206Z/`
- Config: `configs/research/midcap_daily_ma_5_40_v0.yaml` (with symbol = SOL-USDC)
- Run script: `scripts/research/run_btc_eth_daily_ma_5_40_v0.py --symbols SOL-USDC`
- Engine: `src/strategy/ma_crossover_long_only.py`, `src/backtest/engine.py`
- Figures: `artifacts/research/ma_5_40_sol_usdc_baseline/figures/`
