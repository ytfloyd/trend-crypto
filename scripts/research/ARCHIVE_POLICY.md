# scripts/research — archival policy

Goal: retire bespoke one-off scripts into `scripts/research/_archive/` as their
research is ported to the registry-driven runner (`python -m research`, see
[registry/README.md](../../registry/README.md)). This is **incremental**, not a
one-shot mass move — `scripts/research/` is entangled with the rest of the repo.

## Why this is not a bulk delete

A recon (2026-06) found `scripts/research/` is **not** 293 disposable one-offs:

- **122 top-level scripts + 21 subpackages.**
- **10+ tests import these scripts directly** — e.g. `tests/test_transtrend_crypto_ma_5_40_topk.py`, `test_registry_list_rendering.py`, `test_tearsheet_input_resolution.py`, `test_logreg_leakage.py`, `test_run_manifest_v0.py`. Moving them breaks the suite.
- **CI** runs `scripts/research/strategy_registry_v0.py validate`.
- The **README** documents ~15+ of these scripts as the active workflow.
- It contains **flagship strategy packages** (`medallion_lite`, `sornette_lppl`, …) the README tracks by Sharpe — not throwaway code.
- Archived scripts lose their `from common import …` resolution (they rely on
  `scripts/research/` being `sys.path[0]` when run directly), so moving them
  deeper makes them non-runnable until repathed — acceptable only once retired.

## Protected — never archive without a replacement

- `scripts/research/common/` — the shims re-exporting `src/core/*` (imported everywhere).
- `scripts/research/strategy_registry_v0.py` — used by CI.
- Anything imported by a file under `tests/`.
- Anything referenced in `README.md` or `.github/`.
- Maintained strategy subpackages (`medallion_lite`, `sornette_lppl`, `jpm_momentum`, `tsmom`, …) until ported to the registry.

## Archival procedure (per script / small batch)

1. Confirm the script is **not** referenced by `tests/`, `.github/`, `README.md`, cron, or another script:
   `grep -rn "<name>" tests .github README.md scripts`
2. Prefer porting its alpha to a `registry/alphas/<id>.yaml` + a `signals.*` function first.
3. `git mv scripts/research/<script>.py scripts/research/_archive/<script>.py` (recoverable).
4. Run `make test` (or the relevant tests) to confirm nothing broke.

## Blocked until

End-to-end execution works for the strategies in question — i.e. enough
`signals.*` modules + market data exist that `python -m research run --execute`
reproduces what the scripts produce. Until then these scripts are the working
research and stay put. Tracked as reorg task #7.
