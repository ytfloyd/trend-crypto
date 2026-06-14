# Spot-Only Convexity Framework — Research Design Document

**Sleeve:** `research/spot-convexity` · **Asset universe:** crypto spot (Coinbase USD pairs, daily) ·
**Status:** foundation (design + primitives) — empirical phase not yet run.

This is Deliverable #1 of the research brief. It fixes definitions, the payoff model, what the
framework is and is not, the outcome metric, and the validation contract — *before* any empirical
result, so the rest of the sleeve is pre-registered against it.

---

## 1. What "spot convexity" means here

**Spot convexity = a positively skewed per-trade payoff distribution produced by asymmetric trade
management, not by an option contract.** We hold the underlying spot asset long; the asymmetry comes
entirely from *how the trade is managed*:

- **Downside is truncated** by a predefined hard stop and an upward-only trailing stop → losses are
  bounded near −1R (plus gap/slippage).
- **Upside is left open** → winners ride trend / breakout continuation / right-tail moves to large
  positive R.

The payoff per trade is therefore convex in the underlying's forward path: bounded left tail, open
right tail. The edge is **setup selection** — choosing entries where the right-tail potential is
large *relative to the stop distance*, and where the path is clean enough that the trailing stop
captures trend instead of getting whipsawed out.

### Why this is NOT options convexity
| | Options convexity | Spot convexity (this framework) |
|---|---|---|
| Source of asymmetry | Contract payoff (Γ, vega) | Trade-management rules (stops) |
| Cost of convexity | Premium / theta decay (paid upfront, bleeds) | Stop-outs + slippage (paid on losers) |
| Left tail | Capped at premium | Capped at ≈ −1R, **but** gap-through-stop can exceed it |
| Right tail | Unbounded (leveraged) | Bounded by realized spot path × position, no leverage |
| Capacity / borrow | Option liquidity, IV | Spot liquidity, slippage on entry/exit |
| Key risk | IV mean-reversion, decay | Whipsaw, false breakout, gap-through-stop |

The critical practical difference: an option *guarantees* the left-tail cap; a spot stop only
*approximates* it. **Gap-through-stop risk is the spot framework's analogue of theta** — the place
where the clean theoretical payoff leaks. The framework must model it explicitly (see §4) or it will
systematically overstate the left-tail truncation.

---

## 2. How hard + trailing stops create the asymmetric profile

A trade is defined by three numbers known at the signal date: entry, initial stop, and a trailing
rule. Initial risk per unit `R = entry − initial_stop` (set from ATR, so it scales with the asset's
own noise). All outcomes are then measured in **R-multiples** = (exit − entry) / R.

- **Initial hard stop** caps the loss at −1R *if* it fills at the stop price.
- **Upward-only trailing stop** converts unrealized gains into a rising floor: once price advances,
  the worst-case outcome improves from −1R toward break-even and then positive, while the upside
  stays open until trend breaks or the horizon ends.
- The result is a distribution with a **spike near −1R** (whipsaw/false-breakout stop-outs), a mass
  near 0R (trailed-out scratches), and a **long right tail** (trend captures). Positive expectancy
  requires the right-tail mass × magnitude to outweigh the −1R spike after costs and slippage.

This is the LPPLS/Donchian trend-with-stops payoff shape — routed in this repo to the **convexity
pipeline** (gates on the Composite Convexity Score, not Sharpe), consistent with CLAUDE.md.

---

## 3. Market behaviors to capture vs failure modes to avoid

**Capture** (the ideal setup, all causally measurable from OHLCV):
positive trend state · improving/accelerating momentum · breakout or re-acceleration ·
compression→expansion · acceptable stop distance vs noise · manageable gap risk · favorable
upside/downside asymmetry · clean path quality · sufficient liquidity.

**Avoid** (each is mapped to features/filters in the taxonomy doc and §9 of the brief):
false breakouts · trend exhaustion / late entries · stops inside normal noise · excessive gap risk ·
poor liquidity · vol spikes without continuation · high momentum + low path efficiency ·
compression resolving *downward* · crowded/correlated trades · regime shifts.

---

## 4. Outcome measurement — the stop-aware R-multiple target

The target is **realized R-multiple after realistic trade management**, NOT next-period return.
Full construction in `03_target_methodology.md`; the executable primitive is
`scripts/research/spot_convexity/labeler.py` (unit-tested). Non-negotiable execution rules:

1. Entry at the **next executable price** (next bar open) — no same-bar look-ahead.
2. Initial risk from **signal-date information only** (ATR at signal date).
3. **Gap-through-stop modeled explicitly:** if a bar opens beyond the stop, exit at the **open**
   (realized loss can exceed −1R); only if it opens above the stop and the intraday low breaches it
   do we model the exit *at* the stop.
4. Trailing stop **never moves down**.
5. Time stop at max horizon (exit at close).
6. Trades without enough forward history are **labeled incomplete** (right-censored), not silently
   filled.

Recorded per trade: realized R; exit reason (gap-stop / intraday-stop / trailing / time); stop-out
and "stopped-before-+1R" labels; MFE and MAE in R; time-to-stop; time-to-peak; reached-+1R / +2R
flags; positive-convexity label (R ≥ +2R). The research evaluates the **full distribution**, not the
mean alone.

---

## 5. Validation contract (pre-registered)

Strict point-in-time discipline, mirroring the hard-won standards from the medallion sleeve:

- **No future information in features**; **no full-sample normalization**; **no leakage** from the
  target's forward window into feature windows (features end at signal date; the label starts at
  entry = signal+1).
- **Rolling / expanding-window training only**; **walk-forward** OOS by time; secondary OOS by
  asset group / regime.
- **Costs, slippage, and gap-through-stop included** in every reported number; capacity / liquidity
  constraints considered.
- A **transparent baseline score** is built first; any ML model must beat it OOS or we investigate
  features / target / validation / costs rather than shipping the model.
- **Pre-register, then run.** Hypotheses (brief §5) and acceptance gates are fixed before fitting.

Evaluation metrics (full set in the roadmap): mean & median R, win rate, stop-out rate, avg winner/
loser in R, right-tail frequency, % above +1/+2/+3R, expected value per trade, drawdown, turnover,
holding period, time-to-stop/peak, realized-vs-planned slippage. Plus a **deflated/PBO** pass on any
selected configuration, as in the medallion validation.

---

## 6. Crypto-spot instantiation notes (universe choice)

Reusing the Coinbase OHLCV lake (`bars_1d_usd_universe_clean`, point-in-time top-N by trailing ADV).
Two crypto-specific adaptations the brief's equity-flavored language needs:

- **"Overnight gap"** → crypto trades 24/7, so daily bars have small open-vs-prior-close gaps; the
  more material gap risk is **intraday gap-through-stop within the daily bar**. The labeler models
  both (open-gap exit and intraday-breach exit); gap-risk *features* use the daily open-to-prior-
  close jump distribution as the proxy.
- **"Sector / theme crowding"** → no clean GICS sectors; we use **return-correlation clustering**
  (and known L1/L2/DeFi/meme tags where available) as the crowding proxy in the portfolio phase.
- Liquidity uses dollar-ADV and an **Amihud** `|ret|/dollar_volume` illiquidity proxy, both causal.

These are documented so the equity instantiation (the in-progress yfinance data) can later swap in
true overnight gaps and sectors without changing the framework.

---

## 7. What "done" looks like (acceptance, from the brief)

The sleeve answers: positive OOS expected R? right tail vs stop-outs+slippage? which feature groups
matter? does the stop-aware target beat a forward-return target? robust across time/assets/regimes?
sensitivity to stop / trail / horizon? survival after costs + realistic stop execution? implementable
within liquidity/capacity/drawdown? main false-positive regimes? required risk controls?

Mapped to concrete deliverables and status in `00_sleeve_plan.md`.
