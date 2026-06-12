"""DEPRECATED import location.

``ts_pipeline`` moved to ``pipelines.time_series``. This shim re-exports it (and
aliases its submodules) so existing ``ts_pipeline`` / ``src.ts_pipeline``
imports keep working with zero behavior change. New code should import from
``pipelines.time_series``.
See docs/RESEARCH_PIPELINE_REORGANIZATION.md.
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

_TARGET = "pipelines.time_series"
_SUBMODULES = ("types", "stages", "pipeline", "portfolio")

_pkg = _importlib.import_module(_TARGET)
for _name in _SUBMODULES:
    _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"{_TARGET}.{_name}")

_names = getattr(_pkg, "__all__", None) or [n for n in dir(_pkg) if not n.startswith("_")]
globals().update({n: getattr(_pkg, n) for n in _names})
__all__ = list(_names)
