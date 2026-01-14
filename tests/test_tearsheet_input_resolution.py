import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.research.tearsheet_common_v0 import resolve_tearsheet_inputs  # noqa: E402


def test_resolve_single_candidate(tmp_path: Path):
    eq = tmp_path / "ensemble_equity_v0.csv"
    met = tmp_path / "metrics_101_ensemble_filtered_v1.csv"
    eq.write_text("ts,equity\n2020-01-01,1.0\n")
    met.write_text("period,cagr\nfull,0.1\n")
    e, m, manifest = resolve_tearsheet_inputs(research_dir=str(tmp_path))
    assert e == str(eq)
    assert m == str(met)
    assert manifest is None


def test_resolve_ambiguous_metrics(tmp_path: Path):
    eq = tmp_path / "ensemble_equity_v0.csv"
    eq.write_text("ts,equity\n2020-01-01,1.0\n")
    m1 = tmp_path / "metrics_101_ensemble_filtered_v1.csv"
    m2 = tmp_path / "metrics_alt.csv"
    m1.write_text("period,cagr\nfull,0.1\n")
    m2.write_text("period,cagr\nfull,0.2\n")
    try:
        resolve_tearsheet_inputs(research_dir=str(tmp_path))
        assert False, "Expected ValueError due to multiple metrics"
    except ValueError as e:
        msg = str(e)
        assert "metrics" in msg.lower()
        assert "metrics_101_ensemble_filtered_v1.csv" in msg or "metrics_alt.csv" in msg


def test_resolve_explicit_paths(tmp_path: Path):
    eq = tmp_path / "ensemble_equity_v0.csv"
    met = tmp_path / "metrics_101_ensemble_filtered_v1.csv"
    eq.write_text("ts,equity\n2020-01-01,1.0\n")
    met.write_text("period,cagr\nfull,0.1\n")
    e, m, manifest = resolve_tearsheet_inputs(equity_csv=str(eq), metrics_csv=str(met))
    assert e == str(eq)
    assert m == str(met)
    assert manifest is None
