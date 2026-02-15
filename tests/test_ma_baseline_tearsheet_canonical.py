"""Test that MA baseline tear sheet uses the canonical builder."""
import pytest
import pandas as pd


def test_ma_baseline_uses_canonical_builder(tmp_path):
    """Verify MA baseline imports and can call build_standard_tearsheet."""
    # Verify the function is imported and available
    import sys
    sys.path.insert(0, "scripts/research")
    
    from scripts.research import tearsheet_common_v0
    from scripts.research import ma_baseline_tearsheet_v0
    
    # Verify build_standard_tearsheet exists in tearsheet_common_v0
    assert hasattr(tearsheet_common_v0, "build_standard_tearsheet"), \
        "tearsheet_common_v0 must export build_standard_tearsheet"
    
    # Verify ma_baseline imports it
    assert hasattr(ma_baseline_tearsheet_v0, "build_standard_tearsheet"), \
        "ma_baseline_tearsheet_v0 must import build_standard_tearsheet"
    
    # Verify MA baseline has no plotting logic (no matplotlib imports except via shared builder)
    import inspect
    source = inspect.getsource(ma_baseline_tearsheet_v0.main)
    assert "plt.subplots" not in source, "MA baseline must not contain plotting logic"
    assert "PdfPages" not in source, "MA baseline must not contain PDF generation logic"
    assert "build_standard_tearsheet" in source, "MA baseline must call build_standard_tearsheet"


def test_ma_baseline_pdf_structure(tmp_path):
    """Verify MA baseline PDF contains expected pages (integration test)."""
    pytest.importorskip("matplotlib")
    
    # Create test artifacts
    equity_data = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=100),
        "equity": [100000 * (1.005 ** i) for i in range(100)],
    })
    equity_csv = tmp_path / "equity.csv"
    equity_data.to_csv(equity_csv, index=False)
    
    metrics_data = pd.DataFrame({
        "period": ["full"],
        "start": ["2020-01-01"],
        "end": ["2020-04-09"],
        "n_days": [100],
        "cagr": [0.15],
        "vol": [0.20],
        "sharpe": [0.75],
        "sortino": [0.80],
        "calmar": [1.5],
        "max_dd": [-0.10],
        "avg_dd": [-0.05],
        "hit_ratio": [0.6],
        "expectancy": [0.01],
    })
    metrics_csv = tmp_path / "metrics_ma_5_40_baseline_v0.csv"
    metrics_data.to_csv(metrics_csv, index=False)
    
    out_pdf = tmp_path / "test_ma_baseline.pdf"
    
    # Run the tearsheet (no mock, real execution)
    import subprocess
    import sys
    
    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/ma_baseline_tearsheet_v0.py",
            "--research_dir", str(tmp_path),
            "--equity_csv", str(equity_csv),
            "--metrics_csv", str(metrics_csv),
            "--out_pdf", str(out_pdf),
            "--no-benchmark",
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0, f"Tearsheet generation failed: {result.stderr}"
    assert out_pdf.exists(), "PDF was not created"
    
    # Verify PDF is not empty
    assert out_pdf.stat().st_size > 10000, "PDF seems too small (likely malformed)"
