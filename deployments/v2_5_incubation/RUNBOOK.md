# V2.5 Incubation Runbook (Ops)

## Version Pin
- Tag: v2.5-incubation-pack2 (see below after tagging)
- Base tag: v2.5-incubation-pack
- Commit: (filled automatically by git describe / rev-parse)

## What runs
Two sleeves (spot, long-only) with:
- 5/40 MA chassis
- inverse-vol sizing targeting TV=60%
- ADX entry-only filter
- rebalance deadband + max step
- drawdown throttle (floor enforced)
- cash yield credit (4% APY)
- turnover-based costs, close-to-close marking

Configs (from this directory):
- btc_daily_ma_5_40_v25_tv60_cash_yield.yaml
- eth_daily_ma_5_40_v25_tv60_cash_yield.yaml

## Install (clean venv)
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

## Sanity checks
python -m pytest
python -m ruff check .

## Launch
- Start: Monday 00:00 UTC
- Execution: 1-hour TWAP
- Monitoring: daily PnL reconciliation vs backtest MTM

## Fire drill
Trigger: live drawdown <= -15%
Action: Slack alert to desk + incident note.
