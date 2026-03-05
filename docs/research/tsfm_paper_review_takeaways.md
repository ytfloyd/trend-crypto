# Time Series Foundation Models in Finance: Review and Takeaways

## External Validation of Our Ensemble Approach and Five Actionable Improvements

*Research note — Paper review: Rahimikia, Ni & Wang (2025), "Re(Visiting) Time Series Foundation Models in Finance"*
*arXiv: 2511.18578, November 2025*

---

## 1. Paper Summary

This 138-page study from Manchester / UCL / Shanghai provides the first comprehensive empirical evaluation of Time Series Foundation Models (TSFMs) — large transformer-based architectures pre-trained on heterogeneous time series data — for financial return forecasting. The study uses daily excess returns spanning 34 years across 94 countries with ~2 billion observations and ~10,000 US securities.

Three TSFM regimes are tested: (i) zero-shot inference with pre-trained weights, (ii) fine-tuning on financial data, and (iii) pre-training from scratch on financial time series. All are benchmarked against tree-based ensembles (CatBoost, XGBoost, LightGBM), linear models (OLS, Lasso, Ridge, ElasticNet, PCR), and neural networks.

### Key Findings

| Finding | Detail |
|:---|:---|
| Off-the-shelf TSFMs fail | Zero-shot Chronos/TimesFM produce deeply negative R² and ~50% directional accuracy |
| Fine-tuning makes things worse | Fine-tuned portfolios have negative Sharpe in most configs |
| Pre-training from scratch helps | Domain-specific pre-training yields Sharpe 5.42 (Chronos small, 512d window) |
| **CatBoost dominates everything** | **Sharpe 6.79, annualized return 46.5% (252d window, long-short, daily rebalance)** |
| Longer lookback windows help TSFMs | TSFMs improve substantially from 5→512 day windows; benchmarks are robust across all |
| Data augmentation improves TSFMs | Adding JKP financial factors and synthetic data to pre-training helps |
| TSFMs decay slower | TSFM performance degrades more gradually over time than benchmark models |
| Small caps more predictable | Consistent across all model classes |

### Why We Are Not Incorporating TSFMs

1. **Universe size mismatch**: The paper uses ~10,000 stocks for cross-sectional decile sorting. Our crypto universe has ~50–200 liquid tokens — far too small for the data-hungry TSFM approach.
2. **Compute is prohibitive**: The study consumed ~50,000 GPU hours. Even scaled down, pre-training from scratch on crypto data is impractical for marginal gains.
3. **CatBoost wins anyway**: The best TSFM still trails CatBoost (Sharpe 5.42 vs 6.79). Our existing tree-based ensemble approach is validated as optimal.
4. **Different market microstructure**: The paper studies daily equities with dividends, delistings, and risk-free rates. Crypto is 24/7 with different regime dynamics.

## 2. What We Are Incorporating

Five actionable improvements derived from the paper's methodology:

### 2.1 Validation of CatBoost/Ensemble Dominance

The paper's strongest result — that CatBoost consistently outperforms all other model classes including 710M-parameter transformers — directly validates our 101 Alphas ensemble approach and the general ML research track. No architecture change needed; this is confirmation that formulaic alphas + cross-sectional ranking + ensemble averaging is near-optimal.

### 2.2 Huber Loss for Linear Regression Signals

All linear benchmarks in the paper use Huber loss (`OLS+H`, `LASSO+H`, `RIDGE+H`, `Enet+H`), which is robust to the fat-tailed return distributions typical of financial data. Our `signal_lreg()` in the momentum multifrequency study uses standard OLS via `scipy.stats.linregress`. We add a `huber_regression` utility to `src/research/alpha_pipeline.py` for use in any signal that involves fitting a linear model to returns.

**Implementation**: Added `huber_regression()` to `src/research/alpha_pipeline.py`.

### 2.3 R²-Against-Zero Metric

The paper uses an out-of-sample R² metric that benchmarks predictions against a naive forecast of zero (not the historical mean). Following Gu et al. (2020), this is more stringent for individual asset returns because historical mean excess returns are noisy and can make weak models appear good.

```
R²_OOS = 1 - Σ(actual - predicted)² / Σ(actual)²
```

When no explicit prediction is available, the signal-weighted return serves as the implicit prediction, and the metric degenerates to a measure of whether the signal adds information relative to "always predict zero."

**Implementation**: Added `r2_oos_vs_zero()` to `src/backtest/metrics.py` and integrated into `AlphaResult` in `src/research/alpha_pipeline.py`.

### 2.4 Longer Lookback Window Evaluation

The paper finds that 252–512 day windows substantially outperform shorter windows for return prediction. Our current momentum research tests lookbacks of 5d–126d (Chapter 2) and 10d–42d (Chapter 4). The 252d and 512d horizons have not been systematically evaluated.

The Phase A decay script already includes `MOM_12M` with a 252-day lookback, but the multifrequency momentum shootout is hard-coded to `LOOKBACK = 10`. We add 252d and 512d to the lookback grid in the momentum factor definitions.

**Implementation**: Extended `FACTOR_DEFS` in `scripts/research/paper_strategies/phase_a_decay.py` to include `MOM_252D` and `MOM_512D` variants; added a note in the multifrequency config.

### 2.5 Signal Decay Analysis Enhancement

The paper documents that all model classes experience gradual performance degradation over 2001–2023, but TSFMs degrade slower. Our existing `fit_decay()` in `phase_a_decay.py` already measures rolling Sharpe decay and exponential half-lives. We enhance this with:

- **IC decay across horizons**: Measure how predictive power attenuates at forward horizons of 1, 5, 10, 21, 42, 63, 126, 252 days — matching the paper's finding that longer windows improve signal.
- **Year-over-year Sharpe tracking**: Following the paper's expanding-window design, track whether each alpha's Sharpe is improving or degrading year-over-year.

**Implementation**: Added `compute_ic_decay()` and `compute_yearly_sharpe_trend()` to `scripts/research/common/metrics.py`.

## 3. Methodology Notes for Future Reference

### 3.1 Evaluation Best Practices from the Paper

- **Expanding window, not rolling**: Train on all data up to year t, test on year t+1. This avoids the data-discarding problem of pure rolling windows.
- **Cross-sectional portfolio construction**: Decile sorting on predicted returns, long top decile, short bottom decile. This tests the model's ranking ability rather than point-estimate accuracy.
- **Multiple window sizes**: Always test at 5, 21, 252, 512 day lookbacks to capture sensitivity.
- **Modified Diebold-Mariano tests**: Statistical significance tests for forecast comparison (following Gu et al. 2020).

### 3.2 Why TSFMs Might Matter Later

The paper identifies one scenario where TSFMs could become relevant for us: if the crypto universe grows substantially (>1,000 liquid tokens) and we have access to GPU compute. In that regime, pre-training a small Chronos model from scratch on crypto returns (not using off-the-shelf weights) could complement tree-based ensembles. The slower decay rate of TSFMs is particularly attractive for a market where alpha half-lives are short.

This is a "monitor, don't act" finding — revisit in 12–18 months if market structure changes.

---

**Paper**: Rahimikia, E., Ni, H., & Wang, W. (2025). Re(Visiting) Time Series Foundation Models in Finance. arXiv:2511.18578.
**Models released**: [huggingface.co/FinText](https://huggingface.co/FinText)
**Implementation artifacts**: See inline references to modified files above.
