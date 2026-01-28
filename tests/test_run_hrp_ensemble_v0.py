import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("scipy")

from scripts.run_hrp_ensemble import run_hrp_ensemble


def _write_spread_returns(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=50, freq="D")
    spread = rng.normal(0, 0.01, size=len(ts))
    df = pl.DataFrame({"ts": ts, "spread_ret": spread})
    df.write_parquet(path)


def test_run_hrp_ensemble(tmp_path: Path):
    survivors_csv = tmp_path / "gatekeeper_survivors.csv"
    survivors_csv.write_text("alpha\nalpha_a\nalpha_b\nalpha_c\n", encoding="utf-8")

    survivor_dir = tmp_path / "survivors"
    for i, name in enumerate(["alpha_a", "alpha_b", "alpha_c"]):
        alpha_dir = survivor_dir / name
        alpha_dir.mkdir(parents=True, exist_ok=True)
        _write_spread_returns(alpha_dir / f"{name}.spread_returns.parquet", seed=100 + i)

    output_dir = tmp_path / "out"
    result = run_hrp_ensemble(
        survivors_csv=str(survivors_csv),
        survivor_dir=str(survivor_dir),
        output_dir=str(output_dir),
        min_survivors=2,
        method="hrp",
        title="Test HRP Report",
    )

    weights_path = output_dir / "hrp_weights.csv"
    assert weights_path.exists()
    weights = pd.read_csv(weights_path)
    assert np.isclose(weights["weight"].sum(), 1.0)

    meta = json.loads((output_dir / "hrp_meta.json").read_text())
    assert meta["n_survivors"] == 3
    assert (output_dir / "hrp_dendrogram.png").exists()
    assert (output_dir / "hrp_report.md").exists()
