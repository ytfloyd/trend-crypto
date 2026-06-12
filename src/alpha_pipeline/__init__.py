"""DEPRECATED import location.

``alpha_pipeline`` moved to ``pipelines.cross_sectional``. This shim re-exports
it (and aliases its submodules) so existing ``alpha_pipeline`` /
``src.alpha_pipeline`` imports keep working with zero behavior change. New code
should import from ``pipelines.cross_sectional``.
See docs/RESEARCH_PIPELINE_REORGANIZATION.md.
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

_TARGET = "pipelines.cross_sectional"
_SUBMODULES = ("types", "stages", "pipeline")

_pkg = _importlib.import_module(_TARGET)
for _name in _SUBMODULES:
    _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"{_TARGET}.{_name}")

_names = getattr(_pkg, "__all__", None) or [n for n in dir(_pkg) if not n.startswith("_")]
globals().update({n: getattr(_pkg, n) for n in _names})
__all__ = list(_names)
