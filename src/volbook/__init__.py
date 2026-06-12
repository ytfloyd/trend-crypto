"""DEPRECATED import location.

``volbook`` moved to ``domains.volbook``. This shim re-exports it (and aliases
its submodules) so existing ``volbook`` / ``src.volbook`` imports keep working
with zero behavior change. New code should import from ``domains.volbook``.
See docs/RESEARCH_PIPELINE_REORGANIZATION.md.

Submodule aliasing is tolerant: a submodule that needs an optional dependency
(ib_insync, databento, TA-Lib) and can't import is simply not pre-aliased here,
matching the pre-move behavior (it would have failed on import then too).
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

_TARGET = "domains.volbook"
_SUBMODULES = (
    "bundle", "continuous", "contracts", "datalake", "signals", "indicators",
    "ibkr_client", "databento_loader", "canvas_writer", "html_writer", "cli",
)

_pkg = _importlib.import_module(_TARGET)
for _name in _SUBMODULES:
    try:
        _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"{_TARGET}.{_name}")
    except Exception:  # noqa: BLE001 — optional-dep submodule; import on demand later
        pass

_names = getattr(_pkg, "__all__", None) or [n for n in dir(_pkg) if not n.startswith("_")]
globals().update({n: getattr(_pkg, n) for n in _names})
__all__ = list(_names)
