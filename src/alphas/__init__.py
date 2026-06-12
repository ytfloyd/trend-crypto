"""DEPRECATED import location.

The formulaic-alpha DSL moved to ``signals.alphas``. This shim re-exports it
(and aliases its submodules) so existing ``alphas`` / ``src.alphas`` imports keep
working with zero behavior change. New code should import from ``signals.alphas``.
See docs/RESEARCH_PIPELINE_REORGANIZATION.md.
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

_TARGET = "signals.alphas"
_SUBMODULES = ("parser", "primitives", "compiler", "factory", "adapters", "signal_processor")

_pkg = _importlib.import_module(_TARGET)
for _name in _SUBMODULES:
    _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"{_TARGET}.{_name}")

_names = getattr(_pkg, "__all__", None) or [n for n in dir(_pkg) if not n.startswith("_")]
globals().update({n: getattr(_pkg, n) for n in _names})
__all__ = list(_names)
