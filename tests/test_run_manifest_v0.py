from pathlib import Path
import json
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.research.run_manifest_v0 import (  # noqa: E402
    fingerprint_file,
    write_run_manifest,
    get_git_info,
    load_run_manifest,
    update_run_manifest,
)


def test_fingerprint_file_and_manifest(tmp_path: Path):
    # fingerprint missing file
    missing = fingerprint_file(tmp_path / "missing.txt")
    assert missing["exists"] is False

    # fingerprint real file
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc123")
    fp = fingerprint_file(f, with_hash=True)
    assert fp["exists"] is True
    assert fp["size"] == 6
    assert "sha256" in fp

    # write manifest and read
    out = tmp_path / "manifest.json"
    payload = {"a": 1, "fp": fp}
    write_run_manifest(out, payload)
    loaded = json.loads(out.read_text())
    assert loaded["a"] == 1
    assert loaded["fp"]["size"] == 6


def test_get_git_info_handles_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="git")

    monkeypatch.setattr(subprocess, "check_output", boom)
    info = get_git_info()
    assert info["git_branch"] is None
    assert info["git_sha"] is None


def test_update_run_manifest(tmp_path: Path):
    out = tmp_path / "manifest.json"
    write_run_manifest(out, {"a": 1})
    update_run_manifest(out, {"b": 2})
    loaded = load_run_manifest(out)
    assert loaded["a"] == 1
    assert loaded["b"] == 2


def test_update_creates_when_missing(tmp_path: Path):
    out = tmp_path / "missing_manifest.json"
    # Should create new manifest when missing
    update_run_manifest(out, {"foo": "bar"})
    loaded = load_run_manifest(out)
    assert loaded["foo"] == "bar"
