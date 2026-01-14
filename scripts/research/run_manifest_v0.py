#!/usr/bin/env python
from __future__ import annotations

"""
Lightweight run manifest helper for research runs.

Provides:
- get_git_info(): branch/SHA (graceful if git not available)
- fingerprint_file(path): existence, size, mtime, optional sha256
- write_run_manifest(out_path, manifest): atomic-ish JSON writer
- build_base_manifest(): consistent base keys
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional


def get_git_info(repo_root: Optional[Path] = None) -> dict:
    repo_root = Path(repo_root) if repo_root else Path(".")
    info = {"git_branch": None, "git_sha": None}
    try:
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
            .decode()
            .strip()
        )
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode()
            .strip()
        )
        info["git_branch"] = branch
        info["git_sha"] = sha
    except Exception:
        # git not available (e.g., zipped tree); leave as None
        pass
    return info


def fingerprint_file(path: str | Path, with_hash: bool = False) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    fp = {
        "path": str(p),
        "exists": True,
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }
    if with_hash and p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        fp["sha256"] = h.hexdigest()
    return fp


def hash_config_blob(obj: dict) -> str:
    """Stable hash of a JSON-serializable config dict."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def write_run_manifest(out_path: str | Path, manifest: dict) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    tmp_path.replace(out_path)
    return out_path


def load_run_manifest(path: str | Path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def update_run_manifest(path: str | Path, patch: dict) -> Optional[Path]:
    """
    Best-effort manifest patcher: loads if present, merges shallowly, writes back.
    """
    base = load_run_manifest(path) or {}
    base.update(patch)
    try:
        return write_run_manifest(path, base)
    except Exception:
        return None


def build_base_manifest(strategy_id: str, argv: list[str], repo_root: Optional[Path] = None) -> dict:
    ts = int(time.time())
    base = {
        "strategy_id": strategy_id,
        "timestamp_utc": ts,
        "command": " ".join(argv),
    }
    base.update(get_git_info(repo_root))
    # run_id uses ts + short sha if available
    sha = base.get("git_sha")
    short_sha = sha[:8] if sha else "nogit"
    base["run_id"] = f"{ts}-{short_sha}"
    return base


__all__ = ["get_git_info", "fingerprint_file", "write_run_manifest", "build_base_manifest", "hash_config_blob", "load_run_manifest", "update_run_manifest"]
