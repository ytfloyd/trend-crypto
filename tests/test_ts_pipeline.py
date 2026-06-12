"""Tests for the time-series alpha pipeline.

Covers:
  - Helper functions (per-asset IC, pooled t-stat, autocorrelation)
  - All 7 pipeline stages
  - Vol-targeted portfolio construction
  - Pipeline integration test
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_pipeline.types import StageVerdict
from src.ts_pipeline.types import TSCandidate, TSGateConfig
from src.ts_pipeline.portfolio import vol_targeted_backtest
from src.ts_pipeline.stages import (
    _per_asset_ic,
    _pooled_tstat,
    _signal_autocorrelation,
    stage_ts_ic,
    stage_persistence,
    stage_horizon_profile,
    stage_portfolio_backtest,
    stage_blend_diversification,
)
from src.ts_pipeline.pipeline import TSAlphaPipeline


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

def _make_universe(
    n_dates: int = 500,
    n_symbols: int = 20,
    drift: float = 0.0005,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic crypto universe with known trend."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="D")
    symbols = [f"SYM_{i:03d}" for i in range(n_symbols)]

    close_data = {}
    for sym in symbols:
        returns = drift + rng.normal(0, 0.03, size=n_dates)
        prices = 100.0 * np.cumprod(1 + returns)
        close_data[sym] = prices

    close = pd.DataFrame(close_data, index=dates)
    high = close * (1 + rng.uniform(0, 0.01, size=close.shape))
    low = close * (1 - rng.uniform(0, 0.01, size=close.shape))
    volume = pd.DataFrame(
        rng.uniform(1e6, 1e8, size=close.shape),
        index=dates, columns=symbols,
    )
    returns = close.pct_change()

    return close, high, low, volume, returns


def _oracle_forecast(close: pd.DataFrame) -> pd.DataFrame:
    """Perfect-foresight signal: forecast = sign(next-day return) × 10."""
    fwd = close.pct_change().shift(-1)
    return np.sign(fwd) * 10.0


def _persistent_forecast(close: pd.DataFrame) -> pd.DataFrame:
    """Slow EWMAC-like signal with high autocorrelation."""
    fast = close.ewm(span=16).mean()
    slow = close.ewm(span=64).mean()
    vol = close.pct_change().rolling(64).std().replace(0, np.nan)
    raw = (fast - slow) / vol
    valid = raw.values[np.isfinite(raw.values)]
    med = np.nanmedian(np.abs(valid)) if len(valid) > 10 else 1.0
    if med > 1e-12:
        raw = raw * (10.0 / med)
    return raw.clip(-20, 20)


def _noisy_forecast(close: pd.DataFrame, seed: int = 77) -> pd.DataFrame:
    """Pure noise signal — zero IC, low autocorrelation."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 10, size=close.shape),
        index=close.index, columns=close.columns,
    )


# ═══════════════════════════════════════════════════════════════════════
# HELPER TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestPerAssetIC:
    def test_oracle_high_ic(self):
        close, _, _, _, returns = _make_universe(300, 15)
        fc = _oracle_forecast(close)
        ics = _per_asset_ic(fc, returns, horizon=1)
        assert len(ics) > 5
        assert ics.median() > 0.3, "Oracle should have very high IC"

    def test_noise_near_zero(self):
        close, _, _, _, returns = _make_universe(300, 15)
        fc = _noisy_forecast(close)
        ics = _per_asset_ic(fc, returns, horizon=1)
        assert abs(ics.median()) < 0.10, "Noise should have near-zero IC"

    def test_insufficient_data(self):
        close, _, _, _, returns = _make_universe(10, 3)
        fc = _persistent_forecast(close)
        ics = _per_asset_ic(fc, returns, horizon=1)
        assert len(ics) == 0


class TestPooledTstat:
    def test_strong_signal(self):
        ics = pd.Series([0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.04, 0.05,
                          0.06, 0.05, 0.04, 0.07, 0.06, 0.05, 0.04])
        t = _pooled_tstat(ics)
        assert t > 5.0

    def test_zero_signal(self):
        rng = np.random.default_rng(1)
        ics = pd.Series(rng.normal(0, 0.05, size=20))
        t = _pooled_tstat(ics)
        assert abs(t) < 3.0

    def test_insufficient(self):
        assert _pooled_tstat(pd.Series([0.05, 0.06])) == 0.0


class TestSignalAutocorrelation:
    def test_persistent_signal(self):
        close, _, _, _, _ = _make_universe(500, 10)
        fc = _persistent_forecast(close)
        ac = _signal_autocorrelation(fc, lag=1)
        assert ac.median() > 0.90, "EWMAC-like signal should be highly persistent"

    def test_noisy_signal(self):
        close, _, _, _, _ = _make_universe(500, 10)
        fc = _noisy_forecast(close)
        ac = _signal_autocorrelation(fc, lag=1)
        assert ac.median() < 0.10, "Noise should have near-zero autocorrelation"


# ═══════════════════════════════════════════════════════════════════════
# STAGE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestStage1TSIC:
    def test_oracle_passes(self):
        close, _, _, _, returns = _make_universe(400, 20)
        fc = _oracle_forecast(close)
        config = TSGateConfig(min_abs_tstat=2.0, min_abs_median_ic=0.01)
        result = stage_ts_ic(fc, returns, config)
        assert result.verdict == StageVerdict.PASS

    def test_noise_fails(self):
        close, _, _, _, returns = _make_universe(400, 20)
        fc = _noisy_forecast(close)
        config = TSGateConfig(min_abs_tstat=2.0, min_abs_median_ic=0.01)
        result = stage_ts_ic(fc, returns, config)
        assert result.verdict in (StageVerdict.FAIL, StageVerdict.SKIP)


class TestStage2Persistence:
    def test_persistent_passes(self):
        close, _, _, _, _ = _make_universe(500, 15)
        fc = _persistent_forecast(close)
        config = TSGateConfig(min_median_autocorr=0.80)
        result = stage_persistence(fc, config)
        assert result.verdict == StageVerdict.PASS

    def test_noisy_fails(self):
        close, _, _, _, _ = _make_universe(500, 15)
        fc = _noisy_forecast(close)
        config = TSGateConfig(min_median_autocorr=0.80)
        result = stage_persistence(fc, config)
        assert result.verdict == StageVerdict.FAIL


class TestStage3HorizonProfile:
    def test_oracle_passes(self):
        close, _, _, _, returns = _make_universe(400, 20)
        fc = _oracle_forecast(close)
        config = TSGateConfig(
            ic_horizons=[1, 2, 5, 10],
            min_positive_horizons=2,
            require_h1_positive=True,
        )
        result = stage_horizon_profile(fc, returns, config)
        assert result.verdict == StageVerdict.PASS


class TestStage4PortfolioBacktest:
    def test_oracle_passes(self):
        close, _, _, _, returns = _make_universe(400, 20)
        fc = _oracle_forecast(close)
        config = TSGateConfig(
            min_net_sharpe=0.3, vol_lookback=30,
            cost_bps=20.0, max_weight=0.40,
        )
        result = stage_portfolio_backtest(fc, returns, config)
        assert result.verdict == StageVerdict.PASS
        assert result.metrics["net_sharpe"] > 1.0

    def test_noise_fails(self):
        close, _, _, _, returns = _make_universe(400, 20)
        fc = _noisy_forecast(close)
        config = TSGateConfig(min_net_sharpe=0.3, vol_lookback=30)
        result = stage_portfolio_backtest(fc, returns, config)
        assert result.verdict in (StageVerdict.FAIL, StageVerdict.SKIP)


class TestStage7BlendDiversification:
    def test_no_approved(self):
        close, _, _, _, returns = _make_universe(200, 10)
        fc = _persistent_forecast(close)
        config = TSGateConfig()
        result = stage_blend_diversification(fc, {}, returns, config, "test")
        assert result.verdict == StageVerdict.PASS
        assert result.metrics["n_approved"] == 0

    def test_with_approved(self):
        close, _, _, _, returns = _make_universe(200, 10)
        fc1 = _persistent_forecast(close)
        fc2 = _noisy_forecast(close)
        approved = {"signal_a": fc1}
        config = TSGateConfig()
        result = stage_blend_diversification(fc2, approved, returns, config, "test")
        assert result.verdict == StageVerdict.PASS
        assert "most_correlated" in result.metrics


# ═══════════════════════════════════════════════════════════════════════
# PORTFOLIO TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestVolTargetedBacktest:
    def test_basic_output(self):
        close, _, _, _, returns = _make_universe(300, 10)
        fc = _persistent_forecast(close)
        result = vol_targeted_backtest(fc, returns, vol_lookback=30)
        assert len(result.daily_returns) > 0
        assert np.isfinite(result.net_sharpe)
        assert result.max_drawdown <= 0

    def test_oracle_profitable(self):
        close, _, _, _, returns = _make_universe(300, 10)
        fc = _oracle_forecast(close)
        result = vol_targeted_backtest(fc, returns, vol_lookback=30)
        assert result.net_sharpe > 1.0, "Oracle should be very profitable"

    def test_weights_capped(self):
        close, _, _, _, returns = _make_universe(300, 10)
        fc = _persistent_forecast(close) * 10  # amplify
        result = vol_targeted_backtest(
            fc, returns, max_weight=0.20, vol_lookback=30,
        )
        assert result.weights.abs().max().max() <= 0.20 + 1e-10

    def test_gross_leverage_capped(self):
        close, _, _, _, returns = _make_universe(300, 10)
        fc = _persistent_forecast(close) * 10
        result = vol_targeted_backtest(
            fc, returns, max_gross_leverage=1.5, vol_lookback=30,
        )
        gross = result.weights.abs().sum(axis=1)
        assert gross.max() <= 1.5 + 1e-10

    def test_empty_on_insufficient_data(self):
        close, _, _, _, returns = _make_universe(20, 2)
        fc = _persistent_forecast(close)
        result = vol_targeted_backtest(fc, returns, vol_lookback=30)
        assert result.daily_returns.empty


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    def test_full_pipeline_with_oracle(self):
        close, high, low, volume, returns = _make_universe(500, 25, seed=123)

        def oracle_fn(c, h, l, v, r):
            return _oracle_forecast(c)

        candidate = TSCandidate(
            name="oracle_test",
            family="test",
            description="Perfect foresight",
            compute_fn=oracle_fn,
        )

        # Use relaxed config for synthetic data
        config = TSGateConfig(
            min_abs_tstat=1.0,
            min_abs_median_ic=0.005,
            min_median_autocorr=-1.0,  # oracle flips sign daily → negative AC
            min_positive_horizons=1,
            require_h1_positive=True,
            min_net_sharpe=0.1,
            vol_lookback=30,
            max_pbo=0.99,  # relaxed for synthetic data
            min_deflated_sharpe_pval=0.01,
        )

        pipeline = TSAlphaPipeline(close, high, low, volume, returns, config)
        report = pipeline.evaluate(candidate)

        assert len(report.stages) >= 4  # should get through at least 4 stages
        assert report.stages[0].verdict == StageVerdict.PASS  # IC

    def test_noise_rejected_early(self):
        close, high, low, volume, returns = _make_universe(400, 20)

        def noise_fn(c, h, l, v, r):
            return _noisy_forecast(c)

        candidate = TSCandidate(
            name="noise_test",
            family="test",
            compute_fn=noise_fn,
        )

        config = TSGateConfig()
        pipeline = TSAlphaPipeline(close, high, low, volume, returns, config)
        report = pipeline.evaluate(candidate)

        assert report.final_verdict == StageVerdict.FAIL
        assert len(report.stages) <= 3  # should fail at IC or persistence

    def test_batch_evaluation(self):
        close, high, low, volume, returns = _make_universe(300, 15)

        candidates = [
            TSCandidate(
                name="test_ewmac",
                family="ewmac",
                compute_fn=lambda c, h, l, v, r: _persistent_forecast(c),
            ),
            TSCandidate(
                name="test_noise",
                family="test",
                compute_fn=lambda c, h, l, v, r: _noisy_forecast(c),
            ),
        ]

        config = TSGateConfig(min_median_autocorr=0.50)
        pipeline = TSAlphaPipeline(close, high, low, volume, returns, config)
        reports = pipeline.evaluate_batch(candidates)

        assert len(reports) == 2
        summary = pipeline.summary_df()
        assert len(summary) == 2
        assert "alpha" in summary.columns
        assert "verdict" in summary.columns

    def test_report_summary_dict(self):
        close, high, low, volume, returns = _make_universe(300, 15)

        candidate = TSCandidate(
            name="test_signal",
            family="ewmac",
            compute_fn=lambda c, h, l, v, r: _persistent_forecast(c),
        )

        config = TSGateConfig()
        pipeline = TSAlphaPipeline(close, high, low, volume, returns, config)
        report = pipeline.evaluate(candidate)

        d = report.summary_dict()
        assert d["alpha"] == "test_signal"
        assert d["family"] == "ewmac"
        assert d["verdict"] in ("PASS", "FAIL")
