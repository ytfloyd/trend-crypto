# Alpha Testing Quick Reference Card

## 🚀 Commands

```bash
# Learn (5 min)
python scripts/research/example_alpha_workflow.py

# Test all alphas (10 min)
python scripts/research/test_trend_alphas.py --db ../data/market.duckdb

# View results
cat artifacts/research/alpha_tests/ic_results_1d.csv
```

## 📊 IC Interpretation

| IC Value | Quality | Action |
|----------|---------|--------|
| > 0.10 | 🌟 Outstanding | Trade immediately |
| 0.05-0.10 | ✅ Excellent | Production ready |
| 0.02-0.05 | ✓ Good | Tradable |
| 0.00-0.02 | ⚠️ Weak | Borderline |
| < 0.00 | ❌ Wrong | Flip signal (×-1) |

## 🧬 Add Your Own Alpha

```python
# In alpha_engine_v0.py
def alpha_my_signal(self, window: int = 20) -> pd.Series:
    """Your trading idea"""
    close = self.df['close']
    signal = ... # Your formula
    return self.cs_rank(signal)

# Add to available_alphas dict
```

## 📈 Python API

```python
from alpha_engine_v0 import AlphaEngine, load_crypto_data_from_duckdb

# Load data
df = load_crypto_data_from_duckdb('../data/market.duckdb')

# Compute alphas
engine = AlphaEngine(df)
alphas = engine.get_alphas()

# IC analysis
ic = AlphaEngine.compute_ic(alphas)
print(ic)
```

## 🎯 Decision Framework

```
IC > 0.03 AND t-stat > 3 AND Q5-Q1 spread > 0
   → 🚀 TRADE IT!

IC < 0.02 OR t-stat < 2
   → ⚠️ DON'T TRADE
```

## 📚 Docs

- **Quick Start**: `docs/research/ALPHA_FRAMEWORK_README.md`
- **Full Guide**: `docs/research/alpha_testing_guide.md`
- **Summary**: `ALPHA_FRAMEWORK_SUMMARY.md`
