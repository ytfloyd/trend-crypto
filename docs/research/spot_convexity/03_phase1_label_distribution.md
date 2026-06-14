# Phase 1 — Unconditional R-multiple base rate

**What:** the labeler wired to the Coinbase daily lake (point-in-time top-100 universe), applied to
**every eligible (symbol, signal-date)** — "enter long on every name, every day" with the default
stop+trail. No selection signal. This is the **floor any signal must beat** and a sanity check that
the labeler behaves on real data. Harness: `make_labels.py`. Params: ATR(14) stop ×2, trailing ×3,
max-hold 60d, 30 bps round-trip cost (in R). Sample: **166,646 trades** (1,415 right-censored/excluded),
2021–2026.

## Base-rate distribution (net of 30 bps)

| Metric | Value | Read |
|---|--:|---|
| Mean R | **−0.08** | unconditional expectancy slightly negative (✓ no edge from indiscriminate entry) |
| Median R | −0.63 | most trades lose small; mean ≫ median ⇒ **right-skewed / convex** |
| Win rate | 32% | low, as expected for a trend/convexity payoff |
| Avg winner | +1.44R | |
| Avg loser | −0.80R | payoff ratio ≈ 1.8 : 1 (but 68% losers ⇒ net negative) |
| Stop-out-loss rate | 68% | |
| % ≥ +1R / +2R / +3R | 16% / 8% / 4% | the right tail exists but is thin unconditionally |
| Touched +1R / +2R (MFE) | 45% / 22% | many trades reach +1R then trail back below it |
| Mean MFE / MAE | +1.38R / −0.84R | |
| Median bars held | 14 | |
| Exit mix | trail 57% · intraday 34% · time 4% · gap 4% | |

## Two findings worth flagging

1. **The convex payoff shape is real and the machinery works.** Mean (−0.08) ≫ median (−0.63), 32%
   win rate, winners ~1.8× losers — exactly the bounded-left / open-right profile the framework is
   built to exploit. The job of a *signal* (Phase 2+) is to push mean-R net above 0 by raising the
   win rate and/or the right-tail frequency above these base rates.

2. **The −1R floor essentially holds on crypto daily; gap leak is negligible.** Genuine
   realized-worse-than-−1R = **0.01%** (12 trades). *Process note:* a first pass reported 12.8% here —
   a **floating-point artifact** (at-stop exits computing to −1.0000000001 via catastrophic
   cancellation), not real losses. Fixed by rounding R-multiples to 6 dp in the labeler; unit tests
   still green. This matches the design-doc expectation that gap-through-stop risk is more an equities
   concern than a 24/7-crypto one — but the labeler models it regardless, so an equities instantiation
   will surface it.

## The bar for Phase 2

A selection signal is only interesting if, OOS and net of costs, it beats this base rate:
**mean-R > −0.08**, ideally **> 0** with a higher right-tail frequency than 16% (≥+1R) / 8% (≥+2R).
The transparent `spot_convexity_score` is the first attempt; ML models must then beat *it*.

The trailing-give-back gap — **45% touch +1R but only 16% close ≥ +1R** — is a concrete tuning lever
(trail multiplier / horizon / partial profit-taking), to be explored in the Phase 2 stop/trail/
horizon sensitivity sweep.

**Provenance:** `scripts/research/spot_convexity/{data,features,labeler,make_labels}.py`;
labels in `artifacts/spot_convexity/labels_unconditional.csv` (gitignored). Costs in R; survivorship-
free point-in-time top-100.
