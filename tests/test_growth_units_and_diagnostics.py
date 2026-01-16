import pytest

pytest.importorskip("duckdb")

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.research.alpha_ensemble_v15_growth_lib_v0 import GrowthSleeveConfig, _apply_vol_scalar  # noqa: E402
from scripts.research.run_alpha_ensemble_v15_growth_backtest_v0 import _write_diagnostics  # noqa: E402


def test_percent_returns_hard_fail():
    weights = pd.Series([0.5], index=["BTC-USD"])
    # percent-like returns (~120%)
    returns_window = pd.DataFrame({"BTC-USD": np.full(30, 1.2)})
    cfg = GrowthSleeveConfig(allow_percent_returns_for_debug=False)
    try:
        _apply_vol_scalar(weights, returns_window, cfg)
        assert False, "Expected ValueError for percent-unit returns"
    except ValueError:
        pass


def test_percent_returns_allowed_when_flag_set():
    weights = pd.Series([0.5], index=["BTC-USD"])
    returns_window = pd.DataFrame({"BTC-USD": np.full(30, 1.2)})
    cfg = GrowthSleeveConfig(allow_percent_returns_for_debug=True)
    _apply_vol_scalar(weights, returns_window, cfg)  # should not raise


def test_write_diagnostics_schema(tmp_path: Path):
    path = tmp_path / "diag.csv"
    diag_rows = [
        {
            "date": pd.Timestamp("2024-01-01"),
            "universe_n": 2,
            "eligible_n": 2,
            "active_n": 1,
            "gross_exposure": 0.5,
            "net_exposure": 0.5,
            "vol_scalar": 1.2,
            "expected_vol_ann": 0.1,
            "target_vol": 0.2,
            "max_scalar": 1.5,
            "pre_cluster_gross": 0.6,
            "post_cluster_gross": 0.5,
            "cluster_scale": 0.83,
            "n_capped_single_name": 0,
            "n_capped_cluster": 0,
            "turnover": 0.05,
        }
    ]
    df = _write_diagnostics(diag_rows, path)
    assert path.exists()
    required = {
        "date",
        "universe_n",
        "eligible_n",
        "active_n",
        "gross_exposure",
        "net_exposure",
        "vol_scalar",
        "expected_vol_ann",
        "target_vol",
        "max_scalar",
        "pre_cluster_gross",
        "post_cluster_gross",
        "cluster_scale",
        "n_capped_single_name",
        "n_capped_cluster",
        "turnover",
    }
    assert required.issubset(set(df.columns))
