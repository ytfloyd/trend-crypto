"""Transparent spot-convexity baseline score (Deliverable #6).

A simple, equal-weight, sign-explicit benchmark. Every model in the empirical phase must beat
this OUT OF SAMPLE; if it can't, we investigate features/target/validation/costs rather than
shipping the model. Components are causal trailing z-scores (no full-sample stats), so the score
is comparable across assets and dates without full-sample normalization.

score = + momentum acceleration   + trend state    + trend slope    + breakout
        + compression->expansion  + right-tail/stop + path efficiency
        - whipsaw  - gap risk  - normal-noise stop risk  - illiquidity  - downside-vol dominance

This is a RESEARCH BENCHMARK, not a production trading rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import _tz, compute_features

# (feature column, sign). +1 reward, -1 penalty. Equal weight by construction.
COMPONENTS = [
    ("mom_accel", +1),
    ("dist_ma50", +1),            # trend state (above MA)
    ("slope_ma50", +1),
    ("trend_stack", +1),
    ("donchian20", +1),           # breakout
    ("compression_score", +1),
    ("range_expansion", +1),
    ("right_tail_to_stop", +1),   # reward asymmetry
    ("efficiency_ratio_21", +1),  # clean path
    ("whipsaw", -1),
    ("gap_vs_stop", -1),
    ("support_in_atr", -1),       # penalize when stop sits INSIDE noise (small support distance);
                                  #   note: small support_in_atr => stop inside support => higher risk,
                                  #   so we penalize LOW values via the z-score sign below
    ("amihud", -1),               # illiquidity penalty
    ("downside_to_upside_vol", -1),
]


def spot_convexity_score(df: pd.DataFrame, *, stop_mult: float = 2.0) -> pd.DataFrame:
    """Return features + the transparent baseline score for one asset's OHLCV."""
    f = compute_features(df, stop_mult=stop_mult)
    score = pd.Series(0.0, index=f.index)
    n_used = pd.Series(0.0, index=f.index)
    for col, sign in COMPONENTS:
        if col not in f:
            continue
        # support_in_atr: a LARGER value is SAFER (stop beyond support), so reward it (flip the
        # listed -1 to a reward on the z-score). Encode by treating it as +1 here.
        eff_sign = +1 if col == "support_in_atr" else sign
        z = _tz(f[col]).fillna(0.0)
        score = score + eff_sign * z
        n_used = n_used + f[col].notna().astype(float)
    f["spot_convexity_score"] = score
    f["score_coverage"] = n_used / len(COMPONENTS)   # fraction of components available
    return f


def _smoke() -> None:
    # synthetic random-walk-with-drift OHLCV (no lake dependency); the lake wiring is the next phase
    rs = np.random.RandomState(7)
    n = 400
    rets = rs.normal(0.001, 0.03, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rs.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rs.normal(0, 0.01, n)))
    open_ = close * (1 + rs.normal(0, 0.005, n))
    vol = np.abs(rs.normal(1e6, 2e5, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    out = spot_convexity_score(df)
    tail = out[["spot_convexity_score", "score_coverage"]].dropna().tail(3)
    print("baseline score smoke (synthetic):")
    print(tail.to_string())
    print(f"score range: [{out['spot_convexity_score'].min():.2f}, {out['spot_convexity_score'].max():.2f}]")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _smoke()
