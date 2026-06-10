# The Stop-Cost Hierarchy

### What nine years of Coinbase data say about ATR stops in crypto trend-following

---

## Abstract

We backtested a long-only weekly-rebalanced crypto trend-following strategy
across the ten largest Coinbase USDC-quoted spot pairs over the nine-year
period 2017-05-15 to 2026-05-19. Holding entry, sizing, universe, rebalance
cadence, and transaction costs fixed, we varied only the exit-stop geometry
across three specifications: no ATR stop, a fixed 3×ATR stop anchored to
entry, and a trailing 3×ATR stop ratcheted off the highest close since
entry. We observe a clean monotonic ordering on Sharpe (0.74 → 0.71 → 0.25),
on CAGR (+19.3% → +17.8% → +3.0%), and on the ratio of CAGR-foregone to
max-drawdown-protection-purchased (44 bps/pp → 296 bps/pp). The fixed stop
is a defensible trade-off; the trailing stop is not. We document the
mechanism — stop geometry interacting with the volatility-to-trend ratio of
crypto bull markets — and show that the result is robust to breakout-window
choice and consistent with the strategy's failure mode on individual case
studies. The piece is primarily about stop design; the strategy itself
underperforms BTC HODL on absolute return over this window and we address
that comparison explicitly in §3.

---

## 1. A Bitcoin trade in two paths

On October 12, 2020, both versions of our crypto trend-following strategy
bought Bitcoin at $11,374. Same signal, same entry price, same position
sizing logic. Five months later one version sold at $45,232: **+214% on a
single trade**. The other version, ostensibly the smarter one, produced
this sequence:

```
2020-10-12  BUY  BTC @ $11,374
2020-11-07  SELL BTC @ $14,722   (stop hit, +29%)
2020-12-14  BUY  BTC @ $19,167
2021-01-04  SELL BTC @ $30,064   (stop hit, +57%)
2021-01-04  BUY  BTC @ $33,083   [re-entered immediately at higher level]
2021-01-10  SELL BTC @ $34,526   (stop hit, +4%)
2021-02-15  BUY  BTC @ $48,667   [re-entered much higher]
2021-02-22  SELL BTC @ $47,631   (stop hit, −2%)
2021-02-22  BUY  BTC @ $57,489   [re-entered at all-time-high]
2021-02-23  SELL BTC @ $46,328   (stop hit, −19% next day)
```

Compounded across the five round trips: **+67%**. A 3.2× shortfall on the
same Bitcoin signal, in the same five-month window. Five times the
transaction costs, four adverse re-entries at higher levels, and — as we
will document below — no meaningful improvement in downside.

![Same Bitcoin signal, two realized paths](IMAGE_TO_BE_UPLOADED_01_btc_two_paths.png)

The shaded blue region is the entire trade the fixed-stop variant held:
entered once, sold once, captured the full move. Each red triangle is a
trailing-stop forced exit, followed by a fresh entry at a worse price. The
arithmetic of the second path is unforgiving even though every individual
trade in it was nominally a "win."

The only difference between the two backtests was whether the ATR stop was
anchored to the entry price or trailed the highest close since entry. Every
other parameter — universe, signal, sizing, costs, rebalance cadence — was
identical.

The same divergence shows up on the very first trade of the backtest. Both
variants buy ETH on 2017-05-15 at $89.42. On 2017-05-25 the trailing
variant is stopped out at $170.54 on a routine pullback after ETH had run
up; the fixed variant keeps holding. The trailing variant sits in cash for
eleven days while the fixed variant compounds, then re-enters ETH on
2017-06-05 at $245.48 — 44% above its own stop-out — and watches the asset
keep running to $343 by mid-June. The equity-curve gap that compounds over
the next nine years opens on trade number one for the same reason it opens
on the 2020 Bitcoin trade.

## 2. The seductive logic

Textbook descriptions of trailing stops sound reasonable, and we believed
them for some time:

> "A fixed stop protects you from the trade going wrong at entry. But as
> the trade moves your way, you have unrealized profit to protect. A
> trailing stop ratchets the exit up to lock in those gains. It is the
> fixed stop that upgrades itself as the trade succeeds."

Stated that way, the trailing stop is the obviously-better version of the
fixed stop. The pitch implies a strict improvement: same downside
protection at entry, plus additional upside protection as the position runs.
When we wrote the spec for our weekly-rebalanced crypto strategy, we
included a trailing-stop variant as an explicit option specifically because
we expected it to dominate the fixed version on risk-adjusted measures.

The result was the opposite, and lopsided enough that we re-ran it from
scratch.

## 3. Methodology, and a note on the benchmark

### The strategy

We use a deliberately simple trend-following specification. We did not tune
it for absolute return — we built it as a clean, replicable research
vehicle for isolating the effect of design choices like stop geometry.

The bluechip universe restriction is not cosmetic. Extending the same
strategy to the full set of Coinbase USDC-quoted spot pairs produces
categorically different behavior: the strategy becomes anti-edge on the
long tail of smaller alts, losing the majority of capital regardless of
stop choice. We treat universe definition as a foundational design
decision separate from the stop-design question this piece studies, and
leave the universe sensitivity to a companion note.

> **Backtest setup.** Long-only weekly-rebalanced strategy on Coinbase
> USDC-quoted bluechip spot pairs (BTC, ETH, SOL, ADA, XRP, DOGE, AVAX,
> LINK, DOT, LTC). Daily bars; signals computed at prior-day close;
> rebalance and execution at next Monday's open. Entry filter:
> `close > prior 5-day high AND MA(5) > MA(40) AND composite momentum
> score ≥ 40`. The composite momentum score is the mean of three
> cross-sectional rank-percentiles (20-, 40-, and 90-day returns), scaled
> 0–100. Selection takes the top 20 by composite score from the eligible
> set, inverse-vol sized and capped at 15% per asset, gross 100%. 20-day
> ATR; stop checked daily against the intraday low; fill at the stop price
> on hit. 30 bps per side (25 bps fee + 5 bps slippage). All variants
> share the entry, sizing, universe, and rebalance logic — only the stop
> type differs.

### Where this strategy sits versus simply holding Bitcoin

The first question any reader should ask is whether a Sharpe-0.71 crypto
trend strategy is worth studying at all when one could have held BTC and
done better in absolute terms. We agree the question is fair and answer it
directly.

| Metric                | Fixed 3×ATR | BTC HODL  |
|---|---:|---:|
| Annualized return     | +17.8%      | **+52.2%**    |
| Annualized volatility | 29.4%       | 69.3%     |
| Sharpe                | 0.71        | **0.95**      |
| Sortino               | 0.63        | **1.33**      |
| Max drawdown          | **-52.5%**      | -83.8%    |
| Calmar ratio          | 0.34        | **0.62**      |
| Days spent below -50% | **125**         | 1,331     |
| Total return (9y)     | +339%       | **+4,321%**   |

BTC HODL beats the strategy on annualized return, Sharpe, Sortino, and
Calmar over this window. The strategy beats BTC on max drawdown and
crushes BTC on the persistence of deep drawdowns — 125 days spent more
than 50% underwater versus 1,331 days for BTC HODL.

For a research vehicle, two observations matter. First, a Sharpe of 0.71 in
crypto over nine years that includes the 2018 bear, the 2020 boom, the 2022
crash, and the 2024 rally is a respectable institutional number; it is
broadly consistent with multi-decade trend-following studies on traditional
asset classes and well above the median Sharpe of crypto quant strategies
we have surveyed. Second — and this is the load-bearing point for what
follows — the strategy's mechanical, rules-based exit structure is what
allows us to *isolate* the marginal contribution of stop design, which is
the actual subject of this piece. The contribution we are evaluating is
not "strategy versus benchmark." It is "stop choice within strategy."

With that caveat stated explicitly, we turn to the central result.

## 4. The stop-cost hierarchy

We ran three exit specifications on the same signal:

- **No stop** — the strategy exits only via its rebalance logic (MA
  crossover flips negative, momentum score drops below the floor, asset
  falls out of the top-20 selection, or breakdown below the prior 5-day
  low).
- **Fixed 3×ATR stop** — anchored to entry; never moves; daily intra-bar
  check against the low.
- **Trailing 3×ATR stop** — anchored to the highest close since entry;
  ratchets up; otherwise identical machinery.

![The stop-cost hierarchy](IMAGE_TO_BE_UPLOADED_10_stop_cost_hierarchy.png)

The pattern is monotone. Each move from "no stop" → "fixed" → "trailing"
trades a small reduction in max drawdown for a much larger reduction in
return.

We make this quantitative by asking what each stop choice "costs" in
foregone CAGR per percentage point of max-drawdown reduction it purchases,
benchmarking off the no-stop variant:

| Variant         | CAGR    | Max DD   | CAGR cost vs no-stop | DD protection vs no-stop | Cost-per-pp |
|---|---:|---:|---:|---:|---:|
| No stop         | +19.3%  | -55.9%   | —                    | —                        | —           |
| Fixed 3×ATR     | +17.8%  | -52.5%   | 150 bps              | +3.4 pp                  | **44 bps per pp** |
| Trailing 3×ATR  | +3.0%   | -50.4%   | 1,628 bps            | +5.5 pp                  | **296 bps per pp** |

The fixed-stop variant gives up 44 basis points of annualized return for
each percentage point of max-drawdown protection it buys. That is a
defensible trade for a strategy held in a portfolio with other return
sources. The trailing-stop variant gives up 296 basis points per
percentage point — **roughly 6.7× the cost** of the fixed stop for the same
unit of risk reduction. That is not a defensible trade in any institutional
context we can construct.

![Four-column performance comparison](IMAGE_TO_BE_UPLOADED_02_scorecard.png)

The full performance scorecard puts the BTC HODL benchmark next to the
three strategy variants. The internal ordering — no-stop best, fixed close
behind, trailing far worse — is what this piece is about.

![Nine-year equity curves](IMAGE_TO_BE_UPLOADED_03_equity_curve.png)

The equity-curve view confirms that the divergence is not a single-trade or
single-regime artifact. The gap between the fixed and trailing variants
opens on the first ETH trade in May 2017 and widens through every
subsequent bull cycle. The no-stop line tracks slightly above the fixed
line for almost the entire period — confirming that the fixed stop is
modestly costly in absolute terms, but only modestly. The BTC HODL line
ends roughly an order of magnitude above the strategy's terminal NAV but
with a path through a -84% drawdown that no leveraged or capital-constrained
allocator could have stomached in real time.

![Drawdown comparison](IMAGE_TO_BE_UPLOADED_04_drawdown_comparison.png)

Rolling drawdown is the second half of the indictment of the trailing
variant. Both strategy variants spend most of their time underwater by
similar amounts. Both bottom near -50% in the 2022 crypto winter. The
trailing stop gave up two-thirds of the compound return in exchange for
one-twentieth of the drawdown protection it advertised.

## 5. Mechanism: why the trailing stop fails

The result is mechanical, not statistical. The clearest way to see it is to
look at where each stop sits relative to current price as a trade runs up.

![Why the fixed stop gets wider as the trade wins](IMAGE_TO_BE_UPLOADED_05_stop_geometry.png)

At the moment of entry, the two stops are identical. Both sit at
`entry − 3×ATR`, which for a crypto bluechip with about 5% daily ATR is
roughly 15% below the entry price.

After the trade is up 50%:

- **Fixed stop**: still at `entry − 3×ATR`. The current price is now 50%
  above entry, so the stop sits roughly 43% below the current price. The
  safety net got wider by standing still while price rose.
- **Trailing stop**: now at `highest_close − 3×ATR`. The highest close is
  50% above entry, so the stop has ratcheted up to match — and the cushion
  is still only about 10% below current price. The safety net stayed
  exactly the same width even as the trade succeeded.

The next question is empirical: how often does a successful crypto trend
have a 15%-or-more pullback during its run? The answer, in our data, is
*constantly*. The 2020-2021 Bitcoin run contained at least five intra-trend
corrections of 20-30%. The 2024 ETH rally had multiple 20%+ pullbacks. The
late-2023 Solana move from $24 to $120 contained a single-week pullback of
nearly 25% halfway up. These are not trend reversals — they are
mean-reverting noise around persistent uptrends, the texture of crypto bull
markets.

A fixed stop, by the time a trade is up 50–100%, can absorb a 30%+ pullback
without exiting. A position up 200% can absorb a 60% pullback. The trailing
stop cannot. Every routine pullback trips it, and the strategy is back in
cash watching the trend resume without it.

## 6. A clean case study: Solana, October 2023

We pick one trade out of the universe to show the dynamic at single-asset
resolution. Both variants entered SOL on the same week in early October
2023.

![SOL Oct 2023: fixed stop survives the pullback, trailing stop does not](IMAGE_TO_BE_UPLOADED_06_sol_centerpiece.png)

The navy horizontal line is the fixed stop at $20.89. It never moves for
the life of the trade.

The orange staircase is the trailing stop. It ratchets up each time SOL
prints a new closing high. By early November it has climbed to roughly $38
— visibly "locking in" the gains the textbook promised.

SOL then has a routine pullback. The trailing stop fires at $38.19. The
strategy banks +60% and is in cash.

SOL then goes from $40 to $120 over the next two months — the shaded region
on the chart. The trailing variant misses essentially all of it. The
fixed-stop variant just kept holding. It exited at $101.70 on a normal
rebalance signal in January: **+256% on one trade**. When the trailing
variant eventually re-entered SOL at $112.52, chasing higher, it caught only
the tail of the move and gave most of that back to noise.

Same signal. Same asset. Same nine-week window. One variant captured a 4×
move. The other captured roughly half of it and then paid slippage and
re-entry risk to lock in less.

## 7. Aggregate mechanical evidence

The SOL trade is one example. Three patterns show up across the full
nine-year backtest that explain the headline result.

### 7.1 The trailing stop chops off the right tail

![Per-trade return distribution](IMAGE_TO_BE_UPLOADED_07_return_distribution.png)

Two features of the distributions matter. First, the left tails are
**identical**: losing trades are the same size in either configuration,
because both stops were positioned at `entry − 3×ATR` at the moment of
entry. The trailing stop cannot protect more on the downside than the fixed
stop does — they start at the same level.

Second, the right tail is cut in half. The fixed-stop variant has a best
trade of +256% and a clear cluster of trades above +100%. The trailing-stop
variant has a best trade of +95% and almost nothing in the +100% bucket.
The right tail is exactly where trend-following expectancy lives. Anything
that systematically clips it does disproportionate damage even when median
trades and left tails are unchanged.

### 7.2 The trailing stop displaces the rebalance signal

![Exit composition](IMAGE_TO_BE_UPLOADED_08_stop_hit_composition.png)

The fixed-stop variant exits 72% of trades via its rebalance signal — the
strategy's actual edge — and only 28% via a stop hit. The stop is the
emergency brake.

The trailing-stop variant inverts this. **70% of trades exit at the stop**,
30% on signal. The stop has become the primary exit logic. The original
strategy has been replaced by a different strategy with the same entry
rule. Whatever risk-adjusted edge the underlying signal contains is being
filtered through a stop machinery designed around a much shorter holding
horizon than the strategy actually targets.

### 7.3 The trailing stop forces worse re-entry prices

![BTC re-entry tax](IMAGE_TO_BE_UPLOADED_09_reentry_tax.png)

This is the part most discussions of trailing stops omit. Once a stop fires
in a persistent uptrend, the only way back into the trade is a fresh upside
breakout — which by construction prints at a *higher* price than the level
where the trailing stop just fired.

The BTC sequence from §1, plotted as exit-prices on the left versus
re-entry prices on the right, shows the pattern with no ambiguity. Average
re-entry tax across the five round trips is +22% per cycle; the compounded
chain penalty across the five legs is +168% — meaning the strategy
re-entered at prices 168% above where the chain started, on the same
underlying trend, in five months.

This is the deepest reason the trailing variant kills the strategy. It does
not just clip the right tail and shift the exit mix. It converts a
trend-following strategy into a churn engine that systematically buys high
and sells slightly-less-high across the same uptrend it was designed to
capture.

## 8. Robustness

### 8.1 Breakout window

A reasonable question is whether the 5-day breakout filter is too tight and
whether the result depends on it. We re-ran the fixed-stop variant with
breakout windows of 5, 10, and 20 days, holding everything else constant.

![Breakout-window robustness](IMAGE_TO_BE_UPLOADED_11_breakout_sweep.png)

| Breakout window | Sharpe | CAGR  | Max DD | Total return |
|---|---:|---:|---:|---:|
| **5-day**       | **0.71**   | **+17.8%** | -52.5% | **+339%**        |
| 10-day          | 0.60   | +14.5% | -54.3% | +238%        |
| 20-day          | 0.65   | +16.0% | -54.2% | +281%        |

The 5-day spec dominates on both Sharpe and absolute return. Crypto trends
move quickly; wider windows enter later, give up the first leg of the
move, and reduce the proportion of the trend captured. The spec we
analyzed in §1–§7 is not a tuned outlier — it is the strongest member of
the natural neighborhood. This is a meaningful robustness check because if
the result *had* depended on a precisely-chosen breakout window, the stop
comparison would have been less informative.

### 8.2 The no-stop variant as the upper bound

Within the strategy family, the no-stop variant is the appropriate upper
bound for what the entry signal can deliver before any drawdown
machinery is applied. The fact that no-stop only beats the fixed-stop
variant by 150 bps of annualized return (Sharpe 0.74 vs 0.71) is what
makes the fixed-stop trade defensible: drawdown protection is being
bought at a reasonable price. The fact that the trailing variant
underperforms the no-stop variant by **1,628 bps** of annualized return
for an additional 2.1 percentage points of drawdown protection vs fixed
is what makes the trailing trade indefensible.

## 9. When trailing stops are warranted

The trailing stop is not a stupid construct in the abstract. It is wrong
for this problem and right for others. Trailing stops make structural
sense in two regimes:

- **Short-horizon momentum.** If the entry is a 1-3 day signal expecting
  the move to last 5-10 days, a tight trailing stop is appropriate. The
  alpha lives in small consistent moves rather than rare multi-month
  trends. You actively *do not* want to ride through a 25% pullback,
  because that pullback is a large fraction of your expected total move.
  Stop distance should be small relative to the trend being captured.

- **Markets with smooth, monotonic trends.** Institutional FX trends often
  unfold with low intraday volatility. Some commodity trends behave
  similarly. In those environments a 3×ATR cushion above the recent high
  is a reasonable "this trend is broken" signal, because the markets
  themselves do not typically pull back 15%+ during healthy trends.

Crypto bluechip trend-following has neither property. Trades are designed
to last weeks to months. Crypto bull markets routinely have 20-30%
pullbacks. The trailing stop's geometry is wrong for the
volatility-to-trend ratio it is being applied to.

The general principle worth taking away: **the right stop distance is a
function of the volatility-to-trend ratio of the moves the strategy is
trying to capture, not a universal rule.** When trying to capture moves
much larger than typical pullbacks, the stop should widen as the trade
succeeds. When trying to capture moves comparable to typical pullbacks,
the stop should follow along. Imposing the wrong geometry on the wrong
problem — because the wrong geometry sounds cleverer in the abstract —
costs measurable money.

## 10. Conclusion

The headline result is a clean monotone ordering of CAGR (None > Fixed >
Trailing) against an almost flat ordering of max drawdown
(None < Fixed < Trailing, separated by only 5.5 percentage points). The
fixed stop buys drawdown protection at 44 basis points per percentage
point of reduction. The trailing stop buys it at 296 basis points — almost
seven times as expensive. The cost difference is not statistical noise:
it is mechanical, and it follows directly from how the two stops behave as
a trend runs.

Two implications worth carrying away from this. The first is specific:
practitioners running trend-following strategies on crypto with weekly or
longer holding horizons should consider treating the trailing stop as a
contraindicated design choice for this problem rather than as a generic
upgrade. The second is more general: plausible-sounding rules deserve no
benefit of the doubt against backtest data, especially when the rule's
premise depends on an implicit model of market behavior that the data can
directly contradict. Trailing stops "lock in gains." That is the premise.
Locking in gains only matters if those gains would otherwise be lost. The
empirical question — would they have been lost? — is directly answerable,
and the answer for crypto bluechip trend-following over this window is
no: on average, the gains the trailing stop locks in are gains the trade
would have kept anyway, while the cost of locking them in (forced exits,
chased re-entries at higher prices, multiplied transaction costs, a clipped
right tail, an inverted exit mix) is large.

We ran the experiment because the question seemed genuinely open. We
recommend that any team running a trend strategy run the equivalent
experiment on its own signal. The backtest is straightforward. If the
fixed-stop variant beats the trailing variant by more than 30 bps of
Sharpe, the geometry of the underlying market's pullbacks is probably
similar to what is in our data. If the trailing variant wins, that is also
useful information — the edge may be shorter-horizon than assumed, and
sizing, holding period, and re-entry assumptions should be re-examined to
match.

The honest work is figuring out which kind of edge the strategy actually
has, and matching the stop geometry to the shape of the trend it is trying
to capture.

---

*This piece is one of a series on a weekly-rebalanced crypto trend-following
strategy across the Coinbase USDC bluechip universe. The backtest engine,
parameter sweeps, and diagnostic scripts that produced every chart above are
available on request. Nine years of daily data; 30 bps per side cost
assumption; 100% long-only gross exposure.*

*Next in the series: a rebalance-frequency study (daily vs weekly vs
biweekly vs monthly), and why walk-forward parameter optimization made our
Sharpe **worse** — and what that says about the way most quant strategies
are tuned.*
