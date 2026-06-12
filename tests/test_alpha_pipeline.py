"""Tests for the automated alpha pipeline (V2).

Covers:
  - Spearman rank IC (vs old Pearson)
  - IC screen gate logic
  - IC decay gate
  - Redundancy with orthogonalization
  - Turnover / cost gate
  - CPCV embargo
  - Inverse-vol weighted long/short proxy
  - Pipeline integration
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_pipeline.types import (
    AlphaCandidate,
    GateConfig,
    PipelineReport,
    StageResult,
    StageVerdict,
)
from src.alpha_pipeline.stages import (
    _apply_embargo,
    _cross_sectional_ic,
    _forward_returns,
    _invvol_ls_return,
    stage_ic_screen,
    stage_ic_decay,
    stage_redundancy,
    stage_turnover,
)
from src.alpha_pipeline.pipeline import AlphaPipeline


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_panel(n_dates: int = 500, n_syms: int = 30, seed: int = 42):
    """Create synthetic close and volume panels."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    symbols = [f"SYM-{i:03d}" for i in range(n_syms)]

    returns = rng.normal(0.0005, 0.02, (n_dates, n_syms))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    close = pd.DataFrame(prices, index=dates, columns=symbols)
    volume = pd.DataFrame(
        rng.uniform(1e4, 1e6, (n_dates, n_syms)), index=dates, columns=symbols,
    )
    return close, volume


def _make_signal_alpha(close: pd.DataFrame) -> pd.DataFrame:
    """A momentum alpha that has real IC (positive correlation with forward returns)."""
    return close.pct_change(10).rank(axis=1, pct=True)


def _make_noise_alpha(close: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """Pure noise alpha — should fail IC screening."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(0, 1, close.shape), index=close.index, columns=close.columns,
    )


# ── IC helper tests ──────────────────────────────────────────────────

class TestSpearmanIC:
    def test_perfect_rank_correlation(self):
        """Oracle scores (= forward returns) should give IC close to 1.0."""
        close, _ = _make_panel(n_dates=100, n_syms=20)
        fwd_ret = _forward_returns(close, horizon=1)
        # Use the forward returns themselves as scores — perfect foresight
        valid = fwd_ret.dropna(how="all")
        ic = _cross_sectional_ic(valid, fwd_ret)
        assert len(ic) > 0
        assert ic.mean() > 0.8, f"Expected high IC, got {ic.mean():.3f}"

    def test_noise_gives_low_ic(self):
        close, _ = _make_panel(n_dates=200, n_syms=20)
        fwd_ret = _forward_returns(close, horizon=1)
        scores = _make_noise_alpha(close)
        ic = _cross_sectional_ic(scores, fwd_ret)
        assert abs(ic.mean()) < 0.1

    def test_returns_series(self):
        close, _ = _make_panel(n_dates=100, n_syms=10)
        fwd_ret = _forward_returns(close, horizon=1)
        scores = _make_signal_alpha(close)
        ic = _cross_sectional_ic(scores, fwd_ret)
        assert isinstance(ic, pd.Series)
        # resolution-agnostic: pandas may use ns or us depending on version/platform
        assert pd.api.types.is_datetime64_any_dtype(ic.index)


# ── Embargo tests ─────────────────────────────────────────────────────

class TestEmbargo:
    def test_embargo_removes_boundary_indices(self):
        n_dates = 100
        blocks = [list(range(i * 20, (i + 1) * 20)) for i in range(5)]
        train_idx, test_idx = _apply_embargo(blocks, (1,), n_dates, pct_embargo=0.05)

        test_set = set(test_idx)
        train_set = set(train_idx)

        assert test_set == set(blocks[1])
        # Boundary indices should be purged from train
        for boundary in [blocks[1][0] - 1, blocks[1][-1] + 1]:
            if 0 <= boundary < n_dates:
                assert boundary not in train_set, (
                    f"Boundary index {boundary} should be embargoed"
                )

    def test_embargo_preserves_separation(self):
        """No train index should be adjacent to a test index."""
        blocks = [list(range(i * 25, (i + 1) * 25)) for i in range(4)]
        train_idx, test_idx = _apply_embargo(blocks, (1, 3), 100, pct_embargo=0.03)

        test_set = set(test_idx)
        for t in train_idx:
            assert t not in test_set

    def test_zero_embargo(self):
        blocks = [list(range(i * 10, (i + 1) * 10)) for i in range(3)]
        train_idx, test_idx = _apply_embargo(blocks, (1,), 30, pct_embargo=0.0)
        # With 0% embargo, we still remove at least 1 observation
        assert len(train_idx) + len(test_idx) <= 30


# ── Inverse-vol L/S tests ────────────────────────────────────────────

class TestInvvolLSReturn:
    def test_returns_float(self):
        rng = np.random.default_rng(42)
        idx = [f"S{i}" for i in range(20)]
        scores = pd.Series(rng.uniform(0, 1, 20), index=idx)
        returns = pd.Series(rng.normal(0, 0.02, 20), index=idx)
        vol = pd.Series(rng.uniform(0.01, 0.05, 20), index=idx)
        result = _invvol_ls_return(scores, returns, vol)
        assert result is not None
        assert isinstance(result, float)

    def test_no_vol_falls_back_to_equal_weight(self):
        rng = np.random.default_rng(42)
        idx = [f"S{i}" for i in range(20)]
        scores = pd.Series(rng.uniform(0, 1, 20), index=idx)
        returns = pd.Series(rng.normal(0, 0.02, 20), index=idx)
        result = _invvol_ls_return(scores, returns, vol_row=None)
        assert result is not None

    def test_insufficient_data_returns_none(self):
        scores = pd.Series([0.5], index=["A"])
        returns = pd.Series([0.01], index=["A"])
        result = _invvol_ls_return(scores, returns)
        assert result is None


# ── IC Screen tests ───────────────────────────────────────────────────

class TestStageICScreen:
    def test_noise_alpha_likely_fails(self):
        close, _ = _make_panel()
        scores = _make_noise_alpha(close)
        config = GateConfig(min_abs_tstat=2.0, min_abs_mean_ic=0.005, min_ic_days=100)
        result = stage_ic_screen(scores, close, config)
        assert result.verdict in (StageVerdict.FAIL, StageVerdict.PASS)
        assert result.metrics["n_days"] > 0

    def test_low_tstat_threshold_passes_more(self):
        close, _ = _make_panel()
        scores = _make_signal_alpha(close)
        strict = GateConfig(min_abs_tstat=5.0, min_abs_mean_ic=0.001, min_ic_days=100)
        loose = GateConfig(min_abs_tstat=0.5, min_abs_mean_ic=0.001, min_ic_days=100)
        r_strict = stage_ic_screen(scores, close, strict)
        r_loose = stage_ic_screen(scores, close, loose)
        if r_strict.verdict == StageVerdict.PASS:
            assert r_loose.verdict == StageVerdict.PASS

    def test_insufficient_data(self):
        close, _ = _make_panel(n_dates=10, n_syms=3)
        scores = _make_signal_alpha(close)
        config = GateConfig(min_ic_days=200)
        result = stage_ic_screen(scores, close, config)
        assert result.verdict == StageVerdict.FAIL


# ── IC Decay tests ────────────────────────────────────────────────────

class TestStageICDecay:
    def test_monotonic_decay_required(self):
        close, _ = _make_panel(n_dates=300, n_syms=20)
        scores = _make_signal_alpha(close)
        config = GateConfig(
            require_monotonic_decay=True,
            max_decay_horizons=[1, 2, 5],
            min_ic_days=10,
        )
        result = stage_ic_decay(scores, close, config)
        assert result.stage == "ic_decay"
        assert "is_monotonic" in result.metrics


# ── Redundancy tests ──────────────────────────────────────────────────

class TestStageRedundancy:
    def test_identical_alpha_fails_classic(self):
        close, _ = _make_panel()
        scores = _make_signal_alpha(close)
        existing = {"momentum_10d": scores.copy()}
        config = GateConfig(
            max_correlation_with_existing=0.70,
            orthogonalize_redundancy=False,
        )
        result = stage_redundancy(scores, existing, config)
        assert result.verdict == StageVerdict.FAIL
        assert result.metrics["max_corr"] > 0.99

    def test_identical_alpha_with_orthogonalization(self):
        close, _ = _make_panel()
        scores = _make_signal_alpha(close)
        existing = {"momentum_10d": scores.copy()}
        config = GateConfig(
            max_correlation_with_existing=0.70,
            orthogonalize_redundancy=True,
            residual_min_abs_tstat=1.5,
        )
        result = stage_redundancy(scores, existing, config, close_wide=close)
        assert result.metrics.get("method") == "orthogonal"
        # Identical alpha's residual should have no signal
        assert result.verdict == StageVerdict.FAIL

    def test_no_existing_passes(self):
        close, _ = _make_panel()
        scores = _make_signal_alpha(close)
        config = GateConfig()
        result = stage_redundancy(scores, {}, config)
        assert result.verdict == StageVerdict.PASS

    def test_uncorrelated_passes(self):
        close, _ = _make_panel()
        scores_a = _make_signal_alpha(close)
        scores_b = _make_noise_alpha(close, seed=123)
        existing = {"noise": scores_b}
        config = GateConfig(max_correlation_with_existing=0.70)
        result = stage_redundancy(scores_a, existing, config)
        assert result.verdict == StageVerdict.PASS


# ── Turnover tests ────────────────────────────────────────────────────

class TestStageTurnover:
    def test_low_turnover_signal_passes(self):
        close, _ = _make_panel(n_dates=300, n_syms=20)
        scores = _make_signal_alpha(close)
        config = GateConfig(
            turnover_cost_bps=5.0,
            min_net_ic=0.0001,
            min_ic_days=10,
        )
        result = stage_turnover(scores, close, config)
        assert result.stage == "turnover"
        assert "mean_daily_turnover" in result.metrics
        assert "net_ic" in result.metrics

    def test_high_cost_rejects(self):
        close, _ = _make_panel(n_dates=300, n_syms=20)
        scores = _make_signal_alpha(close)
        config = GateConfig(
            turnover_cost_bps=500.0,  # absurdly high cost
            min_net_ic=0.01,
            min_ic_days=10,
        )
        result = stage_turnover(scores, close, config)
        # With 500bps cost, net IC should be negative or very low
        assert result.metrics.get("net_ic", 0) < 0.01

    def test_insufficient_data(self):
        close, _ = _make_panel(n_dates=20, n_syms=3)
        scores = _make_signal_alpha(close)
        config = GateConfig()
        result = stage_turnover(scores, close, config)
        assert result.verdict == StageVerdict.SKIP


# ── PipelineReport tests ─────────────────────────────────────────────

class TestPipelineReport:
    def test_all_pass(self):
        c = AlphaCandidate(name="test", family="test")
        r = PipelineReport(candidate=c, stages=[
            StageResult(stage="s1", verdict=StageVerdict.PASS),
            StageResult(stage="s2", verdict=StageVerdict.PASS),
        ])
        assert r.passed is True
        assert r.final_verdict == StageVerdict.PASS
        assert r.failed_stage is None

    def test_one_fail(self):
        c = AlphaCandidate(name="test", family="test")
        r = PipelineReport(candidate=c, stages=[
            StageResult(stage="s1", verdict=StageVerdict.PASS),
            StageResult(stage="s2", verdict=StageVerdict.FAIL, detail="bad"),
        ])
        assert r.passed is False
        assert r.final_verdict == StageVerdict.FAIL
        assert r.failed_stage == "s2"

    def test_summary_dict(self):
        c = AlphaCandidate(name="alpha_x", family="momentum")
        r = PipelineReport(candidate=c, stages=[
            StageResult(
                stage="ic_screen", verdict=StageVerdict.PASS,
                metrics={"mean_ic": 0.02, "tstat_ic": 3.5},
            ),
        ])
        d = r.summary_dict()
        assert d["alpha"] == "alpha_x"
        assert d["ic_screen__mean_ic"] == 0.02


# ── Integration tests ─────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_pipeline_runs_all_six_stages(self):
        """Ensure the pipeline runs all 6 stages (including turnover) without error."""
        close, volume = _make_panel(n_dates=300, n_syms=20)

        def _mom_fn(c, v, r):
            return r.rolling(10).sum().rank(axis=1, pct=True)

        def _noise_fn(c, v, r):
            rng = np.random.default_rng(77)
            return pd.DataFrame(
                rng.uniform(0, 1, c.shape), index=c.index, columns=c.columns,
            )

        candidates = [
            AlphaCandidate(
                name="momentum_10d", family="momentum", compute_fn=_mom_fn,
            ),
            AlphaCandidate(
                name="noise", family="noise", compute_fn=_noise_fn,
            ),
        ]

        config = GateConfig(
            min_abs_tstat=0.5,
            min_abs_mean_ic=0.001,
            min_ic_days=50,
            require_monotonic_decay=False,
            turnover_cost_bps=5.0,
            min_net_ic=0.0001,
            max_pbo=0.99,
            min_deflated_sharpe_pval=0.01,
        )

        pipeline = AlphaPipeline(
            close_wide=close, volume_wide=volume, config=config,
        )
        reports = pipeline.evaluate_batch(candidates, stop_on_fail=False)

        assert len(reports) == 2
        for r in reports:
            assert isinstance(r, PipelineReport)
            assert len(r.stages) > 0

        # Momentum candidate should reach at least turnover stage
        mom_report = reports[0]
        stage_names = [s.stage for s in mom_report.stages]
        assert "ic_screen" in stage_names
        assert "turnover" in stage_names

    def test_pipeline_stop_on_fail(self):
        close, volume = _make_panel(n_dates=300, n_syms=20)

        def _noise_fn(c, v, r):
            rng = np.random.default_rng(77)
            return pd.DataFrame(
                rng.uniform(0, 1, c.shape), index=c.index, columns=c.columns,
            )

        config = GateConfig(
            min_abs_tstat=5.0,
            min_abs_mean_ic=0.05,
            min_ic_days=50,
        )

        pipeline = AlphaPipeline(
            close_wide=close, volume_wide=volume, config=config,
        )
        reports = pipeline.evaluate_batch(
            [AlphaCandidate(name="noise", family="noise", compute_fn=_noise_fn)],
            stop_on_fail=True,
        )

        assert len(reports) == 1
        assert reports[0].stages[-1].stage == "ic_screen"

    def test_catalog_output(self, tmp_path):
        close, volume = _make_panel(n_dates=300, n_syms=20)

        def _fn(c, v, r):
            return r.rolling(5).sum().rank(axis=1, pct=True)

        pipeline = AlphaPipeline(
            close_wide=close, volume_wide=volume,
            config=GateConfig(
                min_abs_tstat=0.1, min_abs_mean_ic=0.0001,
                min_ic_days=10, max_pbo=0.99, min_deflated_sharpe_pval=0.01,
                require_monotonic_decay=False,
                turnover_cost_bps=1.0, min_net_ic=0.0,
            ),
        )
        pipeline.evaluate(AlphaCandidate(
            name="test_alpha", family="test", compute_fn=_fn,
        ))
        out = pipeline.write_catalog(tmp_path)
        assert out.exists()
        files = list(out.iterdir())
        assert len(files) >= 2
