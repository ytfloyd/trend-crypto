import pandas as pd

from scripts.research.quant_momentum_usdc import (
    ResearchConfig,
    StrategySpec,
    build_selection_mask,
    compute_features,
)


def test_compute_features_uses_skip_window_for_momentum_and_fip():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    close = pd.DataFrame({"AAA-USDC": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=idx)
    volume = pd.DataFrame({"AAA-USDC": [10_000.0] * len(idx)}, index=idx)
    cfg = ResearchConfig(liquidity_window=2, min_dollar_volume=1.0)
    spec = StrategySpec("test", lookback_days=3, skip_days=1)

    features = compute_features(close, volume, spec, cfg)

    assert abs(features["momentum"].loc[idx[-1], "AAA-USDC"] - 0.331) < 1e-12
    assert abs(features["path_quality"].loc[idx[-1], "AAA-USDC"] - 1.0) < 1e-12
    assert bool(features["eligible"].loc[idx[-1], "AAA-USDC"])


def test_selection_ranks_momentum_pool_by_path_quality():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31"])
    cols = ["AAA-USDC", "BBB-USDC", "CCC-USDC"]
    momentum = pd.DataFrame(
        [[0.0, 0.0, 0.0], [0.50, 0.40, 0.30]],
        index=idx,
        columns=cols,
    )
    path_quality = pd.DataFrame(
        [[0.0, 0.0, 0.0], [0.20, 0.90, 1.00]],
        index=idx,
        columns=cols,
    )
    eligible = pd.DataFrame(True, index=idx, columns=cols)
    cfg = ResearchConfig(top_momentum=2, final_positions=1)

    selected = build_selection_mask(momentum, path_quality, eligible, cfg)

    assert not bool(selected.loc[idx[-1], "AAA-USDC"])
    assert bool(selected.loc[idx[-1], "BBB-USDC"])
    assert not bool(selected.loc[idx[-1], "CCC-USDC"])
