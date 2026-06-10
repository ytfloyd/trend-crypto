#!/usr/bin/env python3
"""Daily-only wrapper for the volbook futures universe refresh."""
from __future__ import annotations

import sys
from typing import Sequence

from scripts.volbook.refresh_core_futures import main as refresh_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--daily-only" not in args:
        args.insert(0, "--daily-only")
    return refresh_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
