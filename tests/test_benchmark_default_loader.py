import pandas as pd

from scripts.research.tearsheet_common_v0 import build_benchmark_comparison_table


def test_benchmark_table_without_label():
    stats = {"cagr": 0.1, "vol": 0.2, "sharpe": 1.0, "max_dd": -0.3}
    table = build_benchmark_comparison_table("Strategy", stats, benchmark_label=None, benchmark_eq=None)
    assert table.shape[1] == 2
