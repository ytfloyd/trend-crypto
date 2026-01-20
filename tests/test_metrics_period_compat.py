"""Test metrics CSV compatibility (with and without period column)."""
import pandas as pd
import pytest

from scripts.research.tearsheet_common_v0 import load_strategy_stats_from_metrics


def test_load_metrics_with_period_column(tmp_path):
    """Standard case: metrics CSV has period column."""
    metrics = pd.DataFrame({
        "period": ["full", "2020", "2021"],
        "cagr": [0.15, 0.20, 0.10],
        "vol": [0.20, 0.25, 0.18],
        "sharpe": [0.75, 0.80, 0.55],
        "max_dd": [-0.10, -0.12, -0.08],
    })
    csv_path = tmp_path / "metrics_with_period.csv"
    metrics.to_csv(csv_path, index=False)
    
    stats = load_strategy_stats_from_metrics(str(csv_path))
    assert stats["cagr"] == 0.15
    assert stats["sharpe"] == 0.75


def test_load_metrics_without_period_column_single_row(tmp_path):
    """MA baseline case: metrics CSV has no period column, single row."""
    metrics = pd.DataFrame({
        "symbol": ["BTC-USD"],
        "start": ["2020-01-01"],
        "end": ["2021-12-31"],
        "total_return": [1.5],
        "cagr": [0.15],
        "sharpe": [0.75],
        "max_dd": [-0.10],
        "vol": [0.20],
    })
    csv_path = tmp_path / "metrics_no_period.csv"
    metrics.to_csv(csv_path, index=False)
    
    stats = load_strategy_stats_from_metrics(str(csv_path))
    assert stats["cagr"] == 0.15
    assert stats["sharpe"] == 0.75
    assert stats["max_dd"] == -0.10


def test_load_metrics_without_period_column_multiple_rows_fails(tmp_path):
    """Error case: no period column and multiple rows."""
    metrics = pd.DataFrame({
        "symbol": ["BTC-USD", "ETH-USD"],
        "cagr": [0.15, 0.20],
        "vol": [0.20, 0.25],
        "sharpe": [0.75, 0.80],
        "max_dd": [-0.10, -0.12],
    })
    csv_path = tmp_path / "metrics_multi_no_period.csv"
    metrics.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="missing 'period' column and has 2 rows"):
        load_strategy_stats_from_metrics(str(csv_path))


def test_load_metrics_all_optional_fields(tmp_path):
    """Verify optional fields (sortino, calmar, etc.) are handled."""
    metrics = pd.DataFrame({
        "period": ["full"],
        "cagr": [0.15],
        "vol": [0.20],
        "sharpe": [0.75],
        "sortino": [0.85],
        "calmar": [1.5],
        "max_dd": [-0.10],
        "avg_dd": [-0.05],
        "hit_ratio": [0.6],
        "expectancy": [0.01],
        "n_days": [365],
    })
    csv_path = tmp_path / "metrics_full.csv"
    metrics.to_csv(csv_path, index=False)
    
    stats = load_strategy_stats_from_metrics(str(csv_path))
    assert stats["sortino"] == 0.85
    assert stats["calmar"] == 1.5
    assert stats["avg_dd"] == -0.05
    assert stats["hit_ratio"] == 0.6
    assert stats["expectancy"] == 0.01
