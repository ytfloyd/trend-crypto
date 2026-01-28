import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from portfolio.hrp import HierarchicalRiskParity


def test_hrp_allocates_to_diversifier():
    rng = np.random.default_rng(7)
    n = 500
    base = rng.normal(0, 0.01, size=n)
    r1 = base + rng.normal(0, 0.002, size=n)
    r2 = base + rng.normal(0, 0.002, size=n)
    r3 = rng.normal(0, 0.01, size=n)

    returns = pd.DataFrame(
        {
            "alpha_a": r1,
            "alpha_b": r2,
            "alpha_c": r3,
        }
    )

    weights = HierarchicalRiskParity.allocate(returns)

    assert np.isclose(weights.sum(), 1.0)
    assert weights["alpha_c"] > 0.2
