import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import scripts.research.strategy_registry_v0 as reg  # noqa: E402


def test_validate_passes(monkeypatch):
    monkeypatch.chdir(ROOT)
    reg.REGISTRY_PATH = "docs/research/strategy_registry_v0.json"
    reg.SCHEMA_PATH = "docs/research/strategy_registry_v0.schema.json"
    reg.cmd_validate(argparse.Namespace())  # should not raise


def test_validate_fails_on_missing_fields(tmp_path, monkeypatch, capsys):
    broken = {
        "strategies": [
            {"id": "s1", "family": "fam", "version": "v0", "status": "canonical"}  # missing required fields
        ]
    }
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(broken))
    schema_path = ROOT / "docs/research/strategy_registry_v0.schema.json"

    reg.REGISTRY_PATH = str(reg_path)
    reg.SCHEMA_PATH = str(schema_path)

    with pytest.raises(SystemExit) as exc:
        reg.cmd_validate(argparse.Namespace())
    assert exc.value.code == 1
    captured = capsys.readouterr().out.lower()
    assert "missing" in captured
    assert "status=canonical" in captured or "canonical" in captured
