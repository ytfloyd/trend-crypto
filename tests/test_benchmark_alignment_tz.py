import pandas as pd

from scripts.research.tearsheet_common_v0 import load_benchmark_equity


def test_benchmark_alignment_tz_aware_strategy(tmp_path):
    strategy_index = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    bench = pd.DataFrame(
        {
            "ts": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "equity": [1.0, 1.1, 1.05, 1.2, 1.3],
        }
    )
    path = tmp_path / "bench.csv"
    bench.to_csv(path, index=False)

    aligned = load_benchmark_equity(str(path), strategy_index)
    assert aligned.isna().sum() == 0
    assert not (aligned == 0).all()
    assert aligned.iloc[0] == 1.0
