# Alpha Testing Framework - User Guide

## Overview

This framework converts trading heuristics into testable mathematical signals and measures their predictive power using **Information Coefficient (IC)** analysis.

## Philosophy

### What is an Alpha?

- **Heuristic (❌)**: "Buy when momentum is strong" → Trading on vibes
- **Alpha (✅)**: `(Price - MA_50) / σ_20` → Testable, measurable signal

### Why IC Matters

**IC (Information Coefficient) = Correlation(Signal_today, Return_tomorrow)**

| IC Value | Interpretation | Action |
|----------|---------------|--------|
| **IC > 0.05** | Excellent edge | Trade this signal! |
| **IC ~ 0.02-0.05** | Tradable | Viable with good execution |
| **IC ~ 0.00-0.02** | Weak | Barely profitable |
| **IC < 0.00** | Wrong direction | Flip the signal (multiply by -1) |

## Quick Start

### 1. Run IC Tests

```bash
cd /path/to/trend_crypto
source venv_trend_crypto/bin/activate

python scripts/research/test_trend_alphas.py \
  --db ../data/market.duckdb \
  --out artifacts/research/alpha_tests
```

This will:
- Load crypto data (BTC, ETH, SOL, etc.) from 2023-2025
- Compute 11 different trend/momentum alphas
- Calculate IC for 1-day, 5-day, and 10-day forward returns
- Generate visualizations and quantile spreads
- Save all results to `artifacts/research/alpha_tests/`

### 2. Review Results

Check the output files:

```bash
ls artifacts/research/alpha_tests/

# Key files:
# - ic_results_1d.csv       # IC rankings for 1-day returns
# - ic_results_5d.csv        # IC rankings for 5-day returns
# - alphas_raw.csv           # Raw alpha values (use for further analysis)
# - ic_chart_1d.png          # Visual IC rankings
# - ic_vs_tstat.png          # Quality map (IC vs statistical significance)
# - quantiles_alpha_*.csv    # Spread analysis for top alphas
```

### 3. Interpret Results

**Example output:**

```
IC Results (1-day forward return):
alpha                           ic      ic_mean   ic_std   t_stat   n_days
alpha_vol_adj_momentum      0.0423    0.0418    0.120     4.32      689
alpha_ema_fan              0.0385    0.0381    0.115     3.91      689
alpha_acceleration         0.0198    0.0195    0.122     2.05      689
alpha_momentum            -0.0112   -0.0108    0.118    -1.15      689
```

**Interpretation:**
- `alpha_vol_adj_momentum`: **IC = 0.0423** → Excellent! Use this signal.
- `alpha_ema_fan`: **IC = 0.0385** → Good signal, tradable.
- `alpha_momentum`: **IC = -0.0112** → Flip the signal (multiply by -1).

## Available Alphas

### Trend/Momentum

| Alpha | Heuristic | Formula |
|-------|-----------|---------|
| `momentum` | "Buy recent winners" | `(Price_t - Price_{t-20}) / Price_{t-20}` |
| `vol_adj_momentum` | "Sharpe-style momentum" | `Returns / Volatility` |
| `ema_fan` | "Steep trend is bullish" | `(EMA_10 - EMA_50) / (EMA_50 * σ)` |
| `acceleration` | "Momentum of momentum" | `Δ(Returns)` over time |

### Mean Reversion

| Alpha | Heuristic | Formula |
|-------|-----------|---------|
| `mean_reversion` | "Buy dips" | `-1 * (Price - MA_20) / MA_20` |
| `rsi_divergence` | "Buy oversold" | `50 - RSI` |

### Volatility

| Alpha | Heuristic | Formula |
|-------|-----------|---------|
| `vol_compression` | "Squeeze before breakout" | `-1 * (Vol_current / Vol_avg)` |
| `vol_breakout` | "Buy volatility spikes" | `(Vol_current / Vol_avg) - threshold` |

### Volume

| Alpha | Heuristic | Formula |
|-------|-----------|---------|
| `volume_momentum` | "Follow the volume" | `(Volume - MA_volume) / MA_volume` |
| `price_volume_trend` | "Volume confirms price" | `Corr(Price_Δ, Volume)` |

### Relative Strength

| Alpha | Heuristic | Formula |
|-------|-----------|---------|
| `relative_strength` | "Outperform BTC" | `Returns_alt / Returns_BTC` |

## Advanced Usage

### Test Your Own Alpha

Edit `alpha_engine_v0.py` and add a new method:

```python
def alpha_my_custom_signal(self, window: int = 20) -> pd.Series:
    """
    My Custom Signal

    Heuristic: "Your trading idea here"
    Math: Your formula
    """
    close = self.df['close']

    # Your logic here
    signal = ...  # Calculate your signal

    return self.cs_rank(signal)  # Cross-sectional rank
```

Then add it to the `available_alphas` dict in `get_alphas()`.

### Python API Usage

```python
from alpha_engine_v0 import AlphaEngine, load_crypto_data_from_duckdb

# Load data
df = load_crypto_data_from_duckdb(
    db_path='../data/market.duckdb',
    table='bars_1d_usd_universe_clean_adv10m',
    start='2023-01-01',
    end='2025-01-01'
)

# Compute alphas
engine = AlphaEngine(df)
alpha_df = engine.get_alphas()

# IC analysis
ic_results = AlphaEngine.compute_ic(alpha_df, forward_return_col='forward_return_1d')
print(ic_results)

# Quantile spread (top signal only)
best_alpha = ic_results.iloc[0]['alpha']
quantiles = AlphaEngine.quantile_analysis(alpha_df, alpha_col=best_alpha)
print(quantiles)
```

### Backtest Integration

Once you identify high-IC alphas, integrate them into your backtest:

1. **Option A: Add to `alphas101_lib_v0.py`**
   - Copy your best alphas to the existing alpha library
   - Use the same IC selection workflow as the 101 Alphas strategy

2. **Option B: Create New Strategy Class**
   - Build a standalone strategy that uses your top 3-5 alphas
   - Use `src/strategy/base.py` as a template
   - Run through the standard backtest engine

## Quantile Spread Analysis

After identifying high-IC alphas, check if the signal creates **monotonic returns** across quintiles:

```
Quantile Spread Analysis: alpha_vol_adj_momentum

quantile  avg_return  std_return  n_obs  spread_vs_q1
   0       -0.0024     0.0891     1378      0.0000
   1        0.0003     0.0856     1378      0.0027
   2        0.0012     0.0823     1378      0.0036
   3        0.0021     0.0798     1378      0.0045
   4        0.0039     0.0765     1378      0.0063

Spread (Q4 - Q0): +0.63% per day
```

**Good Signal:** Q4 > Q3 > Q2 > Q1 > Q0 (monotonic)
**Bad Signal:** Random ordering (not predictive)

## Tips for Alpha Discovery

### 1. Start Simple
- Test basic momentum/mean-reversion first
- Add complexity only if IC improves

### 2. Normalize Everything
- Use `cs_rank()` to make signals comparable across assets
- Use `z_score()` for time-series normalization

### 3. Check Stability
- IC should be consistent over time (low `ic_std`)
- T-stat > 3 means the signal is statistically significant

### 4. Test Multiple Horizons
- Some alphas work better at 1-day vs 10-day horizons
- Trend signals often have higher IC at longer horizons

### 5. Combine Signals
- Average your top 3-5 alphas for ensemble effect
- Use IC as weights: `combined_signal = Σ(IC_i * alpha_i)`

## Troubleshooting

### "IC is near zero for everything"
- Market regime might not favor your signals
- Try different time periods (2017-2020 vs 2023-2025)
- Increase universe size (more symbols → better cross-sectional signal)

### "IC is negative but t-stat is high"
- Signal is strong but inverted → multiply by -1
- This is common for mean-reversion signals

### "High IC but low quantile spread"
- Signal might be weak in practice despite good correlation
- Check for outliers skewing the IC calculation

## References

- **Kakushadze (2016)**: "101 Formulaic Alphas" - Original alpha library
- **Information Coefficient**: Standard quant finance metric for signal quality
- **Cross-sectional ranking**: Relative strength signals across assets at each timestamp

## Next Steps

After identifying your best alphas:

1. **Generate tearsheets** using the backtest engine
2. **Test transaction costs** (see `alphas101_tca_v0.py`)
3. **Add to existing ensemble** (see `run_101_alphas_ensemble_v0.py`)
4. **Monitor live IC** to detect signal decay

---

**Questions?** Check `alpha_engine_v0.py` source code for implementation details.
