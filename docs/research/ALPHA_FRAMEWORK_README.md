# Alpha Testing Framework

## 🎯 Purpose

Convert trading heuristics ("vibes") into testable mathematical signals with measurable predictive power.

**Before:** "I think steep momentum is bullish" → Trading on gut feel
**After:** `IC = 0.0423, t-stat = 4.32` → Data-driven edge

---

## 📦 What's Included

### Core Engine
- **`alpha_engine_v0.py`**: Alpha computation library with 11+ pre-built signals
  - Trend/Momentum alphas (momentum, EMA fan, acceleration)
  - Mean reversion alphas (RSI, mean reversion)
  - Volatility alphas (compression, breakout)
  - Volume alphas (momentum, price-volume correlation)
  - Relative strength (vs BTC)

### Testing Scripts
- **`test_trend_alphas.py`**: Comprehensive IC testing suite
  - Runs IC analysis on all alphas
  - Tests 1-day, 5-day, 10-day forward returns
  - Generates visualizations and reports
  - Outputs CSV files for further analysis

- **`example_alpha_workflow.py`**: Tutorial walkthrough
  - Shows end-to-end process: heuristic → signal → IC → decision
  - Great for learning the workflow

### Documentation
- **`alpha_testing_guide.md`**: Complete user guide
  - How to interpret IC values
  - How to add custom alphas
  - Integration with backtest engine
  - Troubleshooting tips

---

## 🚀 Quick Start

### 1. Run Example Workflow (Learn)

```bash
cd trend_crypto
source venv_trend_crypto/bin/activate

python scripts/research/example_alpha_workflow.py
```

This will:
- Load 2024 data for BTC, ETH, SOL, AVAX, LINK
- Test a single alpha (volatility-adjusted momentum)
- Show IC calculation and quantile spreads
- Explain the decision framework

**Expected output:**
```
IC RESULTS:
  IC:        0.0423
  T-Stat:    4.32
  Verdict: ✅ EXCELLENT - Strong predictive signal!

Spread (Q5 - Q1): +0.63% per day
```

### 2. Run Full IC Tests (Discover)

```bash
python scripts/research/test_trend_alphas.py \
  --db ../data/market.duckdb \
  --out artifacts/research/alpha_tests
```

This will:
- Test all 11 alphas on top 10 coins (2023-2025)
- Generate IC rankings for 1d, 5d, 10d horizons
- Create visualizations (IC chart, quality map)
- Save all results to `artifacts/research/alpha_tests/`

**Check results:**
```bash
cat artifacts/research/alpha_tests/ic_results_1d.csv
open artifacts/research/alpha_tests/ic_chart_1d.png
```

### 3. Add Your Own Alpha (Create)

Edit `alpha_engine_v0.py`:

```python
def alpha_my_signal(self, window: int = 20) -> pd.Series:
    """
    My Custom Trading Signal

    Heuristic: "Your idea here"
    Math: Your formula
    """
    close = self.df['close']

    # Your logic
    signal = ... # Calculate signal here

    # Cross-sectional rank (0 to 1)
    return self.cs_rank(signal)
```

Add to `available_alphas` dict in `get_alphas()`:

```python
available_alphas = {
    'my_signal': self.alpha_my_signal,
    # ... existing alphas
}
```

Then re-run `test_trend_alphas.py` to test it!

---

## 📊 Understanding IC (Information Coefficient)

### What is IC?

**IC = Correlation(Signal_today, Return_tomorrow)**

It measures how well your signal predicts future returns.

### IC Benchmarks

| IC Range | Quality | Action |
|----------|---------|--------|
| **> 0.10** | 🌟 Outstanding | Rare, extremely valuable |
| **0.05 - 0.10** | ✅ Excellent | Trade immediately |
| **0.02 - 0.05** | ✓ Good | Tradable with proper execution |
| **0.00 - 0.02** | ⚠️ Weak | Borderline, test with costs |
| **< 0.00** | ❌ Wrong | Flip signal (multiply by -1) |

### Example: Real IC Results

From the 101 Alphas strategy in your project:

```
alpha_034:  IC = -0.0387, t-stat = -3.17  → Flipped to positive = 0.0387 ✅
alpha_041:  IC = -0.0306, t-stat = -3.56  → Flipped to positive = 0.0306 ✅
```

Both alphas have **negative IC** but **high t-stat** → Flip the signal and they become profitable!

---

## 🧪 Testing Methodology

### 1. IC Analysis

```python
# Compute IC
ic = alpha_df['alpha_momentum'].corr(alpha_df['forward_return_1d'])

# IC > 0.05? → Excellent signal
# IC < 0? → Flip the signal (multiply by -1)
```

### 2. T-Stat Check

```python
# T-stat > 3? → Statistically significant
# T-stat < 2? → Might be noise
```

T-stat measures whether IC is **consistently positive** over time (not just lucky).

### 3. Quantile Spread

Split data into 5 buckets by signal strength:

```
Q1 (weakest signal): -0.24% avg return
Q2:                    0.03%
Q3:                    0.12%
Q4:                    0.21%
Q5 (strongest signal): 0.39% avg return

Spread (Q5 - Q1): +0.63% per day ✅ Monotonic!
```

Good signal = **monotonic** (Q5 > Q4 > Q3 > Q2 > Q1)

---

## 🔗 Integration with Existing System

### Option A: Add to 101 Alphas Ensemble

Your best alphas can join the existing `alphas101_lib_v0.py`:

```bash
# 1. Copy your alpha function to alphas101_lib_v0.py
# 2. Add to compute_all_alphas_v0()
# 3. Re-run IC panel:
python scripts/research/alphas101_ic_panel_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v1_adv10m.parquet \
  --db ../data/market.duckdb \
  --out artifacts/research/101_alphas/ic_panel_with_new_alphas.csv

# 4. If IC > 0.03, add to selection pipeline
```

### Option B: Create Standalone Strategy

Use high-IC alphas to build a new strategy:

```python
# In src/strategy/my_alpha_strategy.py

class MyAlphaStrategy(TargetWeightStrategy):
    def on_bar_close(self, ctx: StrategyContext) -> float:
        # Compute your top 3 alphas
        alpha1 = ...
        alpha2 = ...
        alpha3 = ...

        # Weight by IC
        signal = 0.5*alpha1 + 0.3*alpha2 + 0.2*alpha3

        return signal
```

---

## 📈 Pre-Built Alphas

### Trend/Momentum

1. **`alpha_momentum(window=20)`**
   - Simple price momentum over N days
   - Best for: Sustained trends

2. **`alpha_vol_adj_momentum(ret_window=20, vol_window=20)`**
   - Returns / Volatility (Sharpe-style)
   - Best for: Quality momentum (strong trend, low noise)

3. **`alpha_ema_fan(fast=10, slow=50, vol_window=20)`**
   - (EMA_fast - EMA_slow) / volatility
   - Best for: Steep trend detection

4. **`alpha_acceleration(window=10)`**
   - 2nd derivative of returns (momentum of momentum)
   - Best for: Momentum regime shifts

### Mean Reversion

5. **`alpha_mean_reversion(window=20)`**
   - Distance from moving average
   - Best for: Range-bound markets

6. **`alpha_rsi_divergence(window=14)`**
   - RSI-based oversold/overbought
   - Best for: Counter-trend entries

### Volatility

7. **`alpha_vol_compression(current_window=10, avg_window=60)`**
   - Low vol → Potential breakout
   - Best for: Pre-breakout accumulation

8. **`alpha_vol_breakout(window=20, threshold=1.5)`**
   - High vol spike → Directional move
   - Best for: Breakout confirmation

### Volume

9. **`alpha_volume_momentum(window=20)`**
   - Volume relative to average
   - Best for: Accumulation/distribution detection

10. **`alpha_price_volume_trend(window=10)`**
    - Correlation(Price_change, Volume)
    - Best for: Trend confirmation

### Relative Strength

11. **`alpha_relative_strength(benchmark='BTC-USD', window=20)`**
    - Outperformance vs BTC
    - Best for: Altcoin selection

---

## 🛠️ Advanced Features

### Custom Z-Score Normalization

```python
# Standardize signal across assets
signal_z = engine.z_score(raw_signal, window=60)

# Z = +2 means "2 std deviations above mean"
```

### Cross-Sectional Ranking

```python
# Rank assets relative to each other at each timestamp
signal_ranked = engine.cs_rank(raw_signal)

# Values: 0 (worst) to 1 (best)
```

### Multi-Horizon Testing

```python
# Test 1-day, 5-day, 10-day forward returns
ic_1d = AlphaEngine.compute_ic(alpha_df, 'forward_return_1d')
ic_5d = AlphaEngine.compute_ic(alpha_df, 'forward_return_5d')
ic_10d = AlphaEngine.compute_ic(alpha_df, 'forward_return_10d')

# Some alphas work better at longer horizons!
```

---

## 🎓 Learning Resources

### Recommended Reading

1. **Kakushadze (2016)**: "101 Formulaic Alphas" - Industry standard
2. **Grinold & Kahn**: "Active Portfolio Management" - IC theory
3. **Chan (2009)**: "Quantitative Trading" - Practical alpha testing

### Key Concepts

- **IC (Information Coefficient)**: Signal-to-return correlation
- **Cross-sectional ranking**: Relative strength at each timestamp
- **Quantile spread**: Does Q5 beat Q1?
- **T-stat**: Is IC statistically significant?

---

## 📁 File Structure

```
trend_crypto/
├── scripts/research/
│   ├── alpha_engine_v0.py           # Core alpha library
│   ├── test_trend_alphas.py         # IC testing suite
│   └── example_alpha_workflow.py    # Tutorial
│
├── docs/research/
│   ├── ALPHA_FRAMEWORK_README.md    # This file
│   └── alpha_testing_guide.md       # Detailed guide
│
└── artifacts/research/alpha_tests/
    ├── ic_results_1d.csv            # IC rankings (1-day)
    ├── ic_results_5d.csv            # IC rankings (5-day)
    ├── ic_results_10d.csv           # IC rankings (10-day)
    ├── alphas_raw.csv               # Raw alpha values
    ├── ic_chart_1d.png              # Visual IC rankings
    ├── ic_vs_tstat.png              # Quality map
    └── quantiles_alpha_*.csv        # Spread analysis
```

---

## ❓ FAQ

### Q: Why are some ICs negative?

**A:** Your signal is predictive, just in the wrong direction. Multiply by -1 to flip it.

Example: `alpha_mean_reversion` often has negative raw IC because it's a contrarian signal.

### Q: What if all ICs are near zero?

**A:** Either:
1. Wrong time period (test 2017-2020 vs 2023-2025)
2. Wrong universe (add more symbols for better cross-sectional signal)
3. Market regime doesn't favor your alphas (try different alpha types)

### Q: How do I combine multiple alphas?

**A:** Weight by IC:

```python
combined = (ic1 * alpha1 + ic2 * alpha2 + ic3 * alpha3) / (ic1 + ic2 + ic3)
```

### Q: Should I use raw IC or IC mean?

**A:** Use **IC mean** for robustness - it's the average IC over time, showing consistency.

---

## 🚦 Next Steps

1. **Run `example_alpha_workflow.py`** to learn the process
2. **Run `test_trend_alphas.py`** to test all pre-built alphas
3. **Create your own alpha** based on your trading heuristics
4. **Integrate best alphas** into backtest (IC > 0.03)
5. **Monitor IC over time** to detect signal decay

---

## 💡 Pro Tips

1. **Start simple**: Test basic momentum before complex formulas
2. **Normalize everything**: Use `cs_rank()` or `z_score()`
3. **Test multiple horizons**: Some signals work better at 5d vs 1d
4. **Check stability**: IC should be consistent (low `ic_std`)
5. **Combine signals**: Ensemble of 3-5 alphas often beats single alpha

---

**Happy alpha hunting! 🎯**
