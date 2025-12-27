# V2.5 Incubation Launch Checklist (Monday 00:00 UTC)

## Pre-flight (T-60m)
- [ ] Confirm git tag/commit pinned
- [ ] `python -m pytest` passes
- [ ] `python -m ruff check .` passes
- [ ] Configs present in `deployments/v2_5_incubation/`
- [ ] Secrets/env configured outside repo

## Launch (T-0)
- [ ] Start BTC sleeve job
- [ ] Start ETH sleeve job
- [ ] Confirm first signal timestamp aligns with UTC day boundary
- [ ] Confirm TWAP schedule (1 hour)

## Post-launch (T+60m)
- [ ] Confirm fills/logs written
- [ ] Confirm daily reconciliation job scheduled
- [ ] Confirm -15% DD alert route works
