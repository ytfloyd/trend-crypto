"""DEPRECATED import location.

``convexity_pipeline`` moved to ``pipelines.convexity``. This shim re-exports it
(and aliases its submodules, including the ``adapters`` subpackage) so existing
``convexity_pipeline`` / ``src.convexity_pipeline`` imports keep working with
zero behavior change. New code should import from ``pipelines.convexity``.
See docs/RESEARCH_PIPELINE_REORGANIZATION.md.
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

_TARGET = "pipelines.convexity"
_SUBMODULES = (
    "types", "stages", "runner", "metrics", "thresholds", "demo",
    "adapters", "adapters.data_provider", "adapters.existing_engine_adapter",
)

_pkg = _importlib.import_module(_TARGET)
for _name in _SUBMODULES:
    _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"{_TARGET}.{_name}")

_names = getattr(_pkg, "__all__", None) or [n for n in dir(_pkg) if not n.startswith("_")]
globals().update({n: getattr(_pkg, n) for n in _names})
__all__ = list(_names)
