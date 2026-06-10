"""Synthetic demo of the convexity pipeline end-to-end.

Run as:
    python -m convexity_pipeline.demo

Creates two synthetic candidates - one designed to be convex (passes),
one designed to be linear/concave (kills early). Confirms scorecard
emits sensible results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .runner import ConvexityPipelineRunner
from .types import (
    BacktestResult,
    Candidate,
    Hypothesis,
    PayoffShape,
    Track,
)


def _make_returns(n: int, mean: float, sigma: float,
                  skew_strength: float = 0.0, seed: int = 0) -> pd.Series:
    """Generate a returns series with controllable skew (using mixture)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(mean, sigma, n)
    if skew_strength != 0:
        boost = rng.exponential(abs(skew_strength) * sigma, n)
        boost *= np.sign(skew_strength)
        base += boost - np.mean(boost)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(base, index=idx)


def _make_backtest(name: str, n: int, mean: float, sigma: float,
                   skew_strength: float = 0.0,
                   convexity_boost: bool = True,
                   underlying_corr: float = 0.0,
                   seed: int = 0) -> BacktestResult:
    """Synthesize a BacktestResult.

    convexity_boost=True: pad alpha returns on top-decile underlying moves
                          (creates positive convexity beta).
    """
    rng = np.random.default_rng(seed)
    underlying = pd.Series(
        rng.normal(0.0005, 0.012, n),
        index=pd.date_range("2010-01-01", periods=n, freq="B"),
    )
    alpha = _make_returns(n, mean, sigma, skew_strength=skew_strength, seed=seed + 1)
    alpha = alpha.reindex(underlying.index)

    if convexity_boost:
        # On top-decile |underlying| days, alpha aligns with sign(underlying)
        abs_u = underlying.abs()
        thr = abs_u.quantile(0.90)
        top = abs_u >= thr
        alpha[top] = alpha[top] + 0.015 * np.sign(underlying[top])

    if underlying_corr != 0:
        alpha = alpha + underlying_corr * underlying

    equity = (1 + alpha).cumprod()
    # Synthesize trades as runs of same-sign returns
    sign = np.sign(alpha)
    grp = (sign != sign.shift()).cumsum()
    trade_pnls = alpha.groupby(grp).sum()
    trade_durs = alpha.groupby(grp).size()

    return BacktestResult(
        alpha_returns=alpha,
        underlying_returns=underlying,
        equity=equity,
        trade_pnls=trade_pnls,
        trade_durations=trade_durs,
        per_instrument={
            "SYM_A": BacktestResult(
                alpha_returns=alpha * 1.0,
                underlying_returns=underlying,
                equity=(1 + alpha).cumprod(),
                trade_pnls=trade_pnls,
                trade_durations=trade_durs,
            ),
            "SYM_B": BacktestResult(
                alpha_returns=alpha * 0.7,
                underlying_returns=underlying,
                equity=(1 + alpha * 0.7).cumprod(),
                trade_pnls=trade_pnls * 0.7,
                trade_durations=trade_durs,
            ),
        },
        meta={"name": name, "n": n, "convexity_boost": convexity_boost},
    )


def make_synthetic_convex_candidate() -> Candidate:
    """A synthetic alpha designed to look convex (should pass S1)."""
    hyp = Hypothesis(
        name="synthetic_convex_trend",
        statement="On 60-min bars, when ATR-channel breakout fires after 3DC compression, forward 1-5 day returns are positively skewed.",
        rationale="Volatility compression resolves directionally via stops being run; trend confirms continuation.",
        expected_payoff_shape=PayoffShape.CONVEX,
        convexity_track=Track.TREND,
        horizon_bars=30,
        universe=["SYM_A", "SYM_B"],
        bar_frequency="60min",
        params={"atr_len": 13, "mult": 3, "compression_bars": 3},
        cost_assumptions={"commission_bps": 1.0, "spread_bps": 1.0},
        expected_metrics={
            "skew": (0.3, 1.5),
            "calmar": (0.7, 1.5),
            "tail_capture": (0.30, 0.55),
        },
        falsification_criteria=[
            "aggregate_skew < 0",
            "convexity_beta b <= 0",
            "ccs_oos / ccs_is < 0.5",
        ],
        blowup_scenario="prolonged range market - many small stop-outs without compensating tail",
        blowup_mitigation="cap monthly losses; halve allocation after 2 consecutive losing months",
        researcher="demo",
        registration_date="2024-12-01",
        source_reference="TASC V42:04 Ehlers UltimateSmoother + V42:08 Unger ChannelTestBench",
    )
    def _signal_fn(*args: object, **kwargs: object) -> pd.Series:  # placeholder
        return pd.Series(dtype=float)
    return Candidate(
        registry_id="DEMO-001",
        hypothesis=hyp,
        signal_fn=_signal_fn,
    )


def make_synthetic_linear_candidate() -> Candidate:
    """A synthetic alpha registered as linear - should be killed at S0."""
    hyp = Hypothesis(
        name="synthetic_linear_carry",
        statement="Long Treasury carry produces steady positive expected return.",
        rationale="Risk premium for duration exposure.",
        expected_payoff_shape=PayoffShape.LINEAR,
        convexity_track=Track.NA,
        horizon_bars=20,
        universe=["TLT"],
        bar_frequency="1d",
        researcher="demo",
        registration_date="2024-12-01",
    )
    def _signal_fn(*args: object, **kwargs: object) -> pd.Series:
        return pd.Series(dtype=float)
    return Candidate(
        registry_id="DEMO-002",
        hypothesis=hyp,
        signal_fn=_signal_fn,
    )


def make_synthetic_negative_skew_candidate() -> Candidate:
    """A synthetic alpha that claims convex but actually negative skew - killed at S1."""
    hyp = Hypothesis(
        name="synthetic_fake_convex",
        statement="Selling premium when vol high - claimed convex.",
        rationale="Compression resolves with vol crush - long stops",
        expected_payoff_shape=PayoffShape.CONVEX,
        convexity_track=Track.VOL_EXPANSION,
        horizon_bars=20,
        universe=["SYM_A", "SYM_B"],
        bar_frequency="1d",
        researcher="demo",
        registration_date="2024-12-01",
    )
    def _signal_fn(*args: object, **kwargs: object) -> pd.Series:
        return pd.Series(dtype=float)
    return Candidate(
        registry_id="DEMO-003",
        hypothesis=hyp,
        signal_fn=_signal_fn,
    )


# Backtest fn returns appropriate synthetic data per candidate
def synthetic_backtest_fn(candidate: Candidate, variant: str) -> BacktestResult:
    """Synthetic backtest generator. Returns convex or concave data per candidate."""
    seed_base = abs(hash(candidate.registry_id)) % 100000
    seed_offset = abs(hash(variant)) % 1000

    if candidate.registry_id == "DEMO-001":
        # Convex, should pass
        if variant.startswith("cost_2x"):
            return _make_backtest("c2x", 2500, 0.0001, 0.012,
                                   skew_strength=0.6,
                                   convexity_boost=True,
                                   seed=seed_base + seed_offset)
        return _make_backtest("conv", 2500, 0.0006, 0.012,
                               skew_strength=0.8, convexity_boost=True,
                               seed=seed_base + seed_offset)
    elif candidate.registry_id == "DEMO-003":
        # Negative skew - should fail S1
        return _make_backtest("neg", 2500, 0.0003, 0.012,
                               skew_strength=-0.6, convexity_boost=False,
                               seed=seed_base + seed_offset)
    else:
        return _make_backtest("def", 2500, 0.0002, 0.012,
                               seed=seed_base + seed_offset)


def main() -> None:
    runner = ConvexityPipelineRunner()
    candidates = [
        make_synthetic_convex_candidate(),
        make_synthetic_linear_candidate(),
        make_synthetic_negative_skew_candidate(),
    ]
    scorecard, per_cand = runner.run_cohort(
        candidates,
        synthetic_backtest_fn,
        oos_fold_count=8,
        perturbation_count=6,
        look_ahead_signoff=True,
    )

    print("\n=== SCORECARD ===")
    cols = ["registry_id", "name", "track", "shape",
            "final_stage", "passed_all_run",
            "s1_ccs", "s1_skew", "s1_calmar",
            "s1_convexity_beta_b", "s1_tail_capture",
            "s3_ccs_oos_to_is_ratio"]
    cols = [c for c in cols if c in scorecard.columns]
    print(scorecard[cols].to_string(index=False))

    print("\n=== KILL REASONS ===")
    for rid, results in per_cand.items():
        for r in results:
            if r.kill_reasons:
                print(f"  {rid} @ {r.stage.value}: {r.kill_reasons}")

    print("\n=== FINAL STAGES ===")
    for rid, results in per_cand.items():
        final = results[-1] if results else None
        print(f"  {rid}: reached {final.stage.value if final else 'NONE'}, "
              f"passed={final.passed if final else False}")


if __name__ == "__main__":
    main()
