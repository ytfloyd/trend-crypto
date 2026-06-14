# Spot-Convexity Feature Taxonomy

Deliverable #2. Eleven feature groups (A–K). Each lists the **intuition**, **concrete causal
features** computable from daily OHLCV (+ dollar volume), the **key research question**, and the
**failure mode(s)** it guards against. Every feature is **backward-looking only** and evaluated at
the *signal date*; the label window starts at entry (signal+1). No full-sample normalization — any
cross-sectional ranking or z-scoring is done per-date.

Convention: `ret_n = close/close[-n]-1`; `ATR_n` = Wilder ATR; MAs are simple unless noted; all
"percentile" features are **trailing-window** percentiles (e.g. rank of today within the last 252d).

---

## A. Returns & Momentum — *has it started moving, and is it accelerating?*
- `ret_1, ret_3, ret_5, ret_21, ret_63` (multi-horizon returns)
- `mom_st = ret_21`, `mom_mt = ret_63`, `mom_lt = ret_126`
- `mom_accel = ret_21 − ret_63/3` (short vs annualized-pace medium) and `Δmom_21 = ret_21 − ret_21[-21]`
- `ret_strength_ratio = ret_21 / |ret_63|` (recent vs prior trend strength)
- **Q:** transitioning from dormant/neutral into directional movement? **Guards:** late entry (if
  only long-horizon momentum is extreme), dormancy.

## B. Trend State — *is the move with or against the bigger trend?*
- `dist_ma50 = close/SMA50−1`, `dist_ma100`, `dist_ma200`
- `slope_ma50 = SMA50/SMA50[-21]−1` (and ma100 slope)
- `trend_stack = 1[close>SMA50>SMA100>SMA200]` (stack alignment, 0/1 or graded)
- `above_lt_filter = 1[close>SMA200]`
- `trend_stability = fraction of last 63d with close>SMA50`
- **Q:** already in a favorable regime, or fighting a weak/negative backdrop? **Guards:** counter-
  trend entries, trend exhaustion (extreme `dist_ma200`).

## C. Breakout & Range Position — *is price escaping a range / nearing highs?*
- `donchian20 = (close − max(high,20))` flag; same for `donchian55`, `donchian100`
- `pos_in_donchian_n = (close − min(low,n))/(max(high,n)−min(low,n))`
- `dist_252high = close/max(high,252)−1` (≤0; near 0 = at highs)
- `dist_recent_low = close/min(low,21)−1`
- `range_width_n = (max(high,n)−min(low,n))/close`
- `breakout_from_compression = donchian20 AND (atr_pctile_low prior)` (see E)
- **Q:** entering a regime where upside continuation is plausible? **Guards:** false breakout (pair
  with K volume + I path quality), buying into resistance.

## D. Realized Volatility & ATR — *constructive expansion vs just noisy?*
- `rv_10, rv_21, rv_63` (annualized stdev of daily returns)
- `rv_ratio_short_long = rv_10/rv_63`
- `atr_pct = ATR_14/close`
- `atr_expansion = ATR_14/ATR_14[-21]`
- `rv_regime = rv_21 percentile over 252d`
- **Q:** volatility expanding constructively, or just unstable? **Guards:** vol-spike-without-
  continuation (combine with A/B direction), unstable noise.

## E. Compression → Expansion — *emerging from quiet into directional expansion?*
- `bb_width = (SMA20 + 2·std20 − (SMA20 − 2·std20))/SMA20`
- `donchian_width_pctile` (trailing percentile of `range_width_n`)
- `atr_pctile = ATR_14 percentile over 252d`
- `rv_pctile = rv_21 percentile over 252d`
- `compression_score = 1 − min(bb_width_pctile, atr_pctile)` (high when quiet)
- `range_expansion = range_width_5 / range_width_21`, `vol_expansion = rv_5/rv_21`
- **Q:** moving from quiet conditions into expansion? **Guards:** compression resolving *downward*
  (must combine with B trend sign + C breakout direction), chop.

## F. Stop Viability — *will the stop survive noise, and is reward big vs risk?*
- `dist_to_support = close/min(low, 20)−1` (and swing-low variant)
- `support_in_atr = (close − support)/ATR_14`
- `stop_pct_1x/2x/3x = (k·ATR_14)/close` (planned stop size at k∈{1,2,3})
- `stop_beyond_support = 1[(close − k·ATR_14) < support]` (stop sits *under* logical support)
- `hist_right_tail = trailing 252d 90th/95th-pctile of forward-max-favorable move` *(computed only
  from data ending at signal date — see leakage note)*
- `right_tail_to_stop = hist_right_tail / stop_pct` (the core asymmetry feature)
- **Q:** stop outside ordinary noise, and reward worth the risk? **Guards:** stop-inside-noise,
  poor reward:risk.

## G. Gap & Stop-Slippage Risk — *could realized loss exceed planned loss?*
- `gap = open/close[-1]−1`; `down_gap_freq_63 = mean(1[gap < −x%])` over 63d
- `worst_gap_21, worst_gap_63, worst_gap_252` (min daily gap)
- `gap_vs_stop = |worst_gap_63| / stop_pct` (gap risk relative to planned stop)
- `slippage_proxy = mean(|gap|) / atr_pct`
- **Q:** can the stop be executed near its level, or will price gap through it? **Guards:** the
  spot-convexity left-tail leak (this is the framework's "theta").

## H. Upside/Downside Asymmetry — *is up-move more favorable than down?*
- `up_vol = std(ret | ret>0)`, `down_vol = std(ret | ret<0)` over 63d
- `downside_to_upside_vol = down_vol/up_vol`
- `avg_up_day, avg_down_day`, `up_down_mag_ratio = avg_up/|avg_down|`
- `pct_positive_days_63`, `pos_return_persistence = autocorr(ret_1, lag1) over 63d`
- **Q:** favorable upside participation vs downside pressure? **Guards:** assets that grind down /
  spike up then bleed.

## I. Path Quality & Whipsaw — *clean trend vs stop-churning chop?*
- `efficiency_ratio_21 = |close−close[-21]| / Σ|ret_1|` (Kaufman ER; 1=clean, 0=chop)
- `choppiness_14` (Choppiness Index)
- `ma_cross_freq_63 = count(SMA10×SMA30 crosses)/63`
- `whipsaw_score = 1 − efficiency_ratio` (or normalized cross freq)
- `trend_smoothness = 1 − std(ret_1)/|mean(ret_1)|`-style over 21d
- `directional_persistence = max run length of same-sign days / 21`
- **Q:** path clean enough for a trailing stop? **Guards:** high-momentum-but-choppy traps,
  repeated stop-outs.

## J. Pullback-Entry Conditions — *constructive trend, tighter invalidation?*
- `pullback_from_high = close/max(high,21)−1` (small negative = shallow pullback)
- `dist_ma10, dist_ma20` (pullback toward short/medium MA)
- `near_trend_support = 1[|dist_ma50| < 0.5·atr_pct]`
- `pullback_setup = above_lt_filter AND (−deep < pullback_from_high < −shallow)`
- `shallow_vs_deep = pullback_from_high vs −2·atr_pct` (shallow pullback vs trend damage)
- **Q:** favorable long entry with a tighter/logical invalidation? **Guards:** late/extended-
  breakout entries (pullback offers better R-profile), buying deep trend damage.

## K. Volume & Liquidity Confirmation — *is the move supported & tradable?*
- `vol_ratio = volume / SMA(volume,20)`; `vol_trend = SMA(vol,5)/SMA(vol,20)`
- `breakout_vol_confirm = donchian20 AND vol_ratio>1.5`
- `dollar_vol = close·volume`; `adv20_usd` (trailing) — universe + capacity
- `amihud = mean(|ret_1|/dollar_vol, 21d)` (illiquidity)
- `liq_adj_vol = rv_21 · sqrt(amihud)`-style (liquidity-adjusted noise)
- `range_expansion_with_volume = range_expansion AND vol_ratio>1`
- **Q:** participation-supported, and realistically enter/exitable? **Guards:** unconfirmed
  breakouts, illiquid names where stop execution is fiction.

---

## Leakage notes (critical)
- All features use windows ending **at the signal date**. The label uses bars from **entry
  (signal+1) forward** — disjoint windows, no overlap.
- `hist_right_tail` (group F) is the one feature that references *forward-max* moves; it must be
  computed only over a trailing window whose forward outcomes are **fully realized before the signal
  date** (i.e., a lagged historical statistic), never including the current trade's forward window.
- Cross-sectional features (ranks, percentiles across assets) are computed **per date** over the
  point-in-time universe only.
- ATR used for the stop is the **signal-date** ATR (the same value the label's initial risk uses),
  so features and target share a consistent, causal risk unit.

## Mapping to the baseline score (Deliverable #6)
The transparent `spot_convexity_score` (see `baseline_score.py`) rewards A (mom accel), B (trend +
slope), C (breakout), E (compression→expansion), F (`right_tail_to_stop`), I (efficiency); and
penalizes I (whipsaw), G (gap risk), F (stop-inside-noise), K (illiquidity), H (downside-vol
dominance). It is the benchmark every model must beat OOS.
