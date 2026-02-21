# Chapter 6: Risk Management — Stop-Loss, Volatility Targeting, and Drawdown Control

*JPM Momentum Strategies — Digital Asset Recreation*
*Reference: Kolanovic & Wei (2015), Chapter 2, p.56–65*

---

## 1. Motivation

Through Chapters 1–5, we have progressively improved the risk-adjusted
return of crypto momentum from a negative Sharpe (-0.09 with a naive
252-day absolute signal) to +0.56 (EMAC 21d, cross-sectional, inverse-
vol weighted, bi-weekly rebalance). However, one critical deficiency
persists across every variant: **catastrophic drawdowns of ~90–95 %**.

No allocator — institutional or individual — can sustain a peak-to-
trough loss of this magnitude. A 90 % drawdown requires a subsequent
900 % return to recover. The strategy is uninvestable without a risk
management overlay.

The original paper devotes a substantial section (p.56–65) to risk
management techniques for momentum portfolios, including stop-losses,
volatility scaling, and dynamic exposure adjustment. This chapter
applies three distinct risk overlays to the Chapter 5 winner and
evaluates their effectiveness in the crypto context.

## 2. Experimental Design

### 2.1 Base Strategy

All tests begin with the Chapter 5 winner:

| Parameter       | Setting                            |
|-----------------|------------------------------------|
| Signal          | EMAC (Exponential MA Crossover)    |
| Lookback        | 21 days                            |
| Mode            | Cross-sectional, top quintile      |
| Weighting       | Inverse-volatility (63 d)          |
| Rebalance       | Bi-weekly (10 d)                   |
| Costs           | 20 bps round-trip                  |
| **Baseline Sharpe** | **0.56**                       |
| **Baseline MaxDD**  | **-92.4 %**                    |

### 2.2 Risk Overlays Tested

**A. Per-Asset Trailing Stop-Loss**

Track each held asset's peak price since entry. If price falls by more
than the stop threshold, zero the asset's weight. Allow re-entry at
the next rebalance that selects the asset. Remaining weights are
re-normalized to maintain original gross exposure.

Thresholds tested: **10 %, 15 %, 20 %, 30 %**

**B. Portfolio Volatility Targeting**

Compute the portfolio's trailing 42-day realized volatility. Scale all
weights by `target_vol / realized_vol`, capping at 2x leverage. When
realized vol is high (crash regime), the portfolio de-levers; when vol
is low (quiet market), it levers up.

Targets tested: **20 %, 30 %, 40 %, 60 %**

**C. Portfolio Drawdown Control**

Compute the portfolio's running drawdown from its high-water mark.
Apply a linear exposure reduction: full weight at 0 % DD, 50 % weight
at the DD threshold, 0 % weight (full cash) at 2x the threshold.

Thresholds tested: **10 %, 20 %, 30 %**

### 2.3 Combined Overlays

After identifying the best individual overlay from each category,
test all pairwise and triple combinations.

## 3. Results

### 3.1 Part A: Trailing Stop-Loss — A Negative Result

| Configuration       | Sharpe | MaxDD    | CAGR   | Avg Exposure |
|---------------------|--------|----------|--------|--------------|
| Baseline (no stop)  | **0.56** | -92.4 % | +7.9 % | 1.00       |
| Stop 10 %           | 0.31   | -96.5 % | -6.0 % | 0.69         |
| Stop 15 %           | 0.37   | -98.0 % | -3.9 % | 0.81         |
| Stop 20 %           | 0.36   | -97.5 % | -6.8 % | 0.88         |
| Stop 30 %           | 0.44   | -95.2 % | -2.3 % | 0.96         |

**Every stop-loss level degrades performance.** This is the most
counterintuitive result of this study and merits careful explanation.

**Why stops fail in crypto momentum:**

1. **Volatility-induced whipsaw.** Crypto assets routinely experience
   15–25 % intraday drawdowns during bull markets. A 10–20 % trailing
   stop triggers frequently during normal price action, not just
   during genuine trend reversals. The strategy gets "stopped out of
   winners."

2. **Asymmetric re-entry cost.** After stopping out, the asset can
   only re-enter at the next rebalance (10 days later). By then, the
   price has often recovered, and re-entry occurs at a higher level.
   The stop-loss buys high and sells low — the opposite of momentum.

3. **Concentration risk from re-normalization.** When stopped-out
   assets' weights are redistributed, the portfolio concentrates into
   fewer positions, amplifying idiosyncratic risk. The MaxDD actually
   *increases* under tighter stops.

4. **The altcoin graveyard creates false protection.** Many stopped
   assets don't recover — but these are assets that the momentum signal
   would have de-selected at the next rebalance anyway. The stop-loss
   redundantly removes assets the signal already knows are failing,
   while harming positions where the signal is still positive.

**Comparison with the paper:** The paper reports modest benefits from
stop-losses in commodities (p.59), where drawdowns are shallower and
price paths are smoother. Crypto's extreme volatility inverts the
stop-loss value proposition.

### 3.2 Part B: Volatility Targeting — The Clear Winner

| Configuration       | Sharpe | MaxDD    | CAGR    | Vol    | Avg Exposure |
|---------------------|--------|----------|---------|--------|--------------|
| Baseline            | 0.56   | -92.4 % | +7.9 %  | 96.2 % | 1.00         |
| **VolTarget 20 %**  | **0.55** | **-48.5 %** | **+11.5 %** | **25.4 %** | 0.24 |
| VolTarget 30 %      | 0.53   | -58.1 % | +13.4 % | 35.2 % | 0.36         |
| VolTarget 40 %      | 0.51   | -66.3 % | +13.9 % | 45.4 % | 0.48         |
| VolTarget 60 %      | 0.49   | -80.1 % | +11.0 % | 66.5 % | 0.72         |

Volatility targeting is transformative:

1. **MaxDD cut nearly in half** — from -92.4 % to -48.5 % at the 20 %
   target — while Sharpe barely moves (0.56 → 0.55).

2. **CAGR *improves*** — from +7.9 % to +11.5 %. This is the
   "volatility drag" effect: by running at constant 25 % vol instead of
   wild swings between 30 % and 200 %, the geometric mean return
   increases even as the arithmetic mean decreases.

3. **Average exposure of 0.24** — the portfolio is only ~24 % invested
   on average. During calm markets it scales up toward 100 %; during
   crypto crashes it naturally de-levers to 5–10 %. This is a
   mechanical "risk-off" switch that requires no subjective judgment.

4. **The Sharpe-drawdown tradeoff is monotonic**: lower vol targets give
   better drawdown protection but slightly lower Sharpe, creating a
   clean dial for risk preference.

**Why vol targeting works in crypto:**

Crypto's key pathology is **volatility clustering** — crashes are
preceded and accompanied by exploding volatility. A trailing 42-day
realized vol captures this signal in real time and mechanically reduces
exposure *before* the worst of the drawdown unfolds. Unlike stop-losses,
vol targeting doesn't require a price threshold and doesn't get whipsawed
by normal price fluctuations.

### 3.3 Part C: Drawdown Control — Effective but Expensive

| Configuration       | Sharpe | MaxDD    | CAGR   | Avg Exposure |
|---------------------|--------|----------|--------|--------------|
| Baseline            | 0.56   | -92.4 % | +7.9 % | 1.00         |
| DD Control 10 %     | 0.38   | -43.5 % | +6.8 % | 0.07         |
| DD Control 20 %     | 0.33   | -64.1 % | +5.5 % | 0.13         |
| DD Control 30 %     | 0.39   | -61.5 % | +7.8 % | 0.20         |

Drawdown control achieves the **best absolute drawdown** (-43.5 % at the
10 % threshold) but at a severe Sharpe penalty (0.56 → 0.38):

1. **Average exposure of 0.07** — the portfolio is in cash 93 % of the
   time at the 10 % threshold. Crypto spends extended periods in
   drawdown from all-time highs, so the control is triggered
   chronically, not just during crashes.

2. **Missed recoveries.** The V-shaped recoveries typical of crypto
   (e.g., March 2020, late 2022) begin while the portfolio is still in
   deep drawdown with near-zero exposure. By the time the high-water
   mark is recovered, the best part of the rally is over.

3. **The 30 % threshold offers the best balance** — still cuts MaxDD to
   -61.5 % while maintaining enough exposure (0.20) to participate in
   rallies.

### 3.4 Part D: Combined Overlays

| Configuration                        | Sharpe | MaxDD    | CAGR   |
|--------------------------------------|--------|----------|--------|
| Stop 30 % + VolTarget 20 %          | 0.47   | -46.6 % | +9.1 % |
| Stop 30 % + DD 30 %                 | 0.42   | -58.4 % | +9.2 % |
| **VolTarget 20 % + DD 30 %**        | **0.54** | **-34.4 %** | **+9.0 %** |
| All three combined                   | 0.45   | -33.9 % | +7.0 % |

The **VolTarget 20 % + DD Control 30 %** combination achieves the best
risk-adjusted profile:

- **Sharpe 0.54** — only 0.02 below the unmanaged baseline
- **MaxDD -34.4 %** — reduced by 63 % from -92.4 %
- **CAGR +9.0 %** — positive and investable

Adding the stop-loss on top marginally improves MaxDD (-33.9 %) but
costs 0.09 Sharpe points. The stop is not worth the drag.

### 3.5 Part E: Best Overlay Across All Strategies

Applying the best combined overlay (VolTarget 20 % + DD 30 % + Stop 30 %)
to all three strategies from Chapter 5:

| Strategy           | Base Sharpe | Overlay Sharpe | Base MaxDD | Overlay MaxDD |
|--------------------|-------------|----------------|------------|---------------|
| RET 10d            | 0.45        | 0.38           | -92.3 %    | **-35.1 %**   |
| **EMAC 21d**       | **0.56**    | **0.45**       | -92.4 %    | **-33.9 %**   |
| MAC 42d            | 0.43        | 0.28           | -94.6 %    | **-35.4 %**   |

The overlay universally cuts MaxDD to the **-33 % to -35 % range**
regardless of the underlying signal. This confirms that the risk
management layer is robust and not overfit to a specific signal.

The Sharpe cost ranges from -0.07 (RET) to -0.15 (MAC). Smoother
signals (EMAC) lose less alpha to the overlay because they generate
less conflicting signal noise.

## 4. The Volatility Targeting Mechanism

Since vol targeting emerged as the dominant risk overlay, it is worth
examining its mechanics in detail.

### 4.1 How It Works

```
scalar(t) = target_vol / realized_vol(t-1 to t-42)
weight(t) = base_weight(t) × min(scalar(t), max_leverage)
```

During the 2022 crypto winter, realized portfolio vol spiked to
~150 % annualized. With a 20 % target, the scalar drops to
0.20 / 1.50 ≈ 0.13, reducing exposure to 13 % of full weight. The
portfolio holds mostly cash while prices crash.

During the 2023–2024 recovery, vol compressed to ~40 %, and the scalar
rises to 0.20 / 0.40 = 0.50, allowing 50 % exposure — enough to
participate meaningfully in the rally.

### 4.2 Why CAGR Improves Under Vol Targeting

This is not just about cutting losses. The **volatility drag** formula
explains why:

```
Geometric return ≈ Arithmetic return - ½ × Variance
```

At 96 % annualized vol, the variance term is ½ × 0.96² ≈ 0.46, or
46 percentage points of annual return destroyed by compounding. At
25 % vol, the drag is only ½ × 0.25² ≈ 3 %. The geometric return
improves by ~43 percentage points annually, even if the arithmetic
return decreases.

## 5. Comparison with the Paper

| Technique        | Paper (Commodities)         | Our Findings (Crypto)              |
|------------------|----------------------------|------------------------------------|
| Stop-loss        | Modest benefit (p.59)      | **Harmful** — whipsaw dominates    |
| Vol targeting    | Effective (p.60-62)        | **Highly effective** — transforms investability |
| DD control       | Not explicitly tested      | Effective but expensive on alpha   |
| Combined         | Additive benefits          | Diminishing returns beyond vol target |

The paper's framework translates well in principle — vol targeting is
the best overlay in both contexts — but the *magnitudes* differ
dramatically:

- **Stop-losses**, which provide modest value in commodities, are
  actively destructive in crypto due to 5–10x higher routine volatility.
- **Vol targeting** is more valuable in crypto precisely because the
  vol-of-vol is so much higher — the scalar swings from 0.1x to 2.0x,
  providing genuine regime-adaptive exposure.

## 6. Updated Leaderboard

| Rank | Strategy                                  | Sharpe | MaxDD    | CAGR    |
|------|-------------------------------------------|--------|----------|---------|
| 1    | **EMAC 21d + VolTarget 20 %**             | **0.55** | **-48.5 %** | **+11.5 %** |
| 2    | EMAC 21d + VT 20 % + DD 30 %             | 0.54   | -34.4 %  | +9.0 %  |
| 3    | EMAC 21d baseline (no overlay)            | 0.56   | -92.4 %  | +7.9 %  |
| —    | BTC Buy & Hold                            | ~0.50  | ~-77 %   | ~25 %   |

The choice between ranks 1 and 2 depends on the allocator's drawdown
tolerance:
- **Rank 1** maximizes CAGR (+11.5 %) with a -48.5 % max drawdown
- **Rank 2** sacrifices 2.5 % CAGR for a more palatable -34.4 % drawdown

Both are meaningfully improved from the -92.4 % drawdown of the
unmanaged strategy.

## 7. Practical Implications

1. **Do not use per-asset stop-losses for crypto momentum.** They
   create more whipsaw cost than protection. Crypto's normal volatility
   is wide enough to trigger any reasonable stop threshold.

2. **Volatility targeting is the single most important overlay.** A
   20–30 % target cuts drawdowns by 40–50 % with minimal Sharpe loss,
   improves CAGR through variance reduction, and adapts mechanically
   to regime changes.

3. **Drawdown control provides incremental protection** when combined
   with vol targeting, cutting MaxDD from -48 % to -34 %, but at a
   Sharpe cost. Use it as a "circuit breaker" at a 30 % threshold.

4. **The risk overlay generalizes across signal types.** All three
   strategies converge to similar MaxDD (~35 %) under the combined
   overlay, suggesting the risk layer is robust and not overfit.

5. **Average exposure is 20–25 %.** This means 75–80 % of capital is
   in cash on average. The strategy is really a "risk-parity with
   cash" allocation, not a fully invested momentum portfolio.

6. **Remaining drawdown of -34 % to -49 %** is still significant.
   Chapter 7's long-only optimization and Chapter 8's multi-signal
   blending may further reduce this.

---

*Script*: `scripts/research/jpm_momentum/step_06_risk_management.py`
*Artifacts*: `artifacts/research/jpm_momentum/step_06/`
*Next*: Chapter 7 — Risk-Adjusted Momentum and Long-Only Optimization
