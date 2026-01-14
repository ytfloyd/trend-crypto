from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.research.run_manifest_v0 import fingerprint_file, write_run_manifest  # noqa: E402


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
