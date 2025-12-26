# V2.5 Incubation Deployment (Core Trend Book)

## Source of Truth (Pin Everything)
- Repo: git@github.com:ytfloyd/trend-crypto.git
- Tag: v2.5-incubation-pack (use this for deployment; v2.5-incubation is legacy)
- Commit: f4cda1f206332a1ebbe294d12c04d37c449f9c4d

## Environment
- Python 3.12 (venv_trend_crypto)
- Install: `python -m pip install -e ".[dev]"`
- Sanity: `python -m pytest` ; `python -m ruff check .`

## Strategy Configs (Incubation Pack)
Use the copies in this deployment pack:
- BTC: deployments/v2_5_incubation/btc_daily_ma_5_40_v25_tv60_cash_yield.yaml
- ETH: deployments/v2_5_incubation/eth_daily_ma_5_40_v25_tv60_cash_yield.yaml

Assumptions baked into configs:
- Daily bars; long-only spot (no leverage/funding)
- Backtest costs: fee_bps=10, slippage_bps=10
- Cash yield: 4% APY on uninvested cash
- Target vol: 60% via strategy sizing logic

## Methodology Note
- Marking is close-to-close MTM; this is not a claim of fills at the close.
- Execution realism is modeled via turnover * (fee_bps + slippage_bps); live trading uses TWAP.

## Turnover Semantics
- Combined 50/50 book is a return-mix of sleeve NAVs; no extra 50/50 rebalance unless explicitly implemented.
- Turnover reported comes from sleeve-level executed turnover already embedded in each sleeve run.

## Reproducibility
- Sleeves:
  - `python scripts/run_backtest.py --config deployments/v2_5_incubation/btc_daily_ma_5_40_v25_tv60_cash_yield.yaml`
  - `python scripts/run_backtest.py --config deployments/v2_5_incubation/eth_daily_ma_5_40_v25_tv60_cash_yield.yaml`
- Combined 50/50 return-mix:
  - `python scripts/build_combined_portfolio_50_50.py --run_a artifacts/runs/<BTC_RUN_ID> --run_b artifacts/runs/<ETH_RUN_ID> --out_dir artifacts/compare/v25_combined_tv60_cash_yield --initial_nav 100000`
- Tear sheet PDF:
  - `python scripts/generate_tearsheet_pdf.py --run_btc artifacts/runs/<BTC_RUN_ID> --run_eth artifacts/runs/<ETH_RUN_ID> --combined_dir artifacts/compare/v25_combined_tv60_cash_yield --benchmark_btc_bh artifacts/runs/btc_daily_buy_and_hold_20251223T164253Z --rf_apy 0.04 --roll_corr_days 90 --out_pdf artifacts/tearsheets/core_trend_book_v25.pdf`

## Included Deliverables (Shipped)
- Tear sheet: deployments/v2_5_incubation/core_trend_book_v25.pdf
- Combined summary: deployments/v2_5_incubation/combined_summary.json

## Monday Launch Checklist
- Pre-open (before Monday 00:00 UTC):
  - Checkout tag v2.5-incubation-pack
  - `python -m pytest` ; `python -m ruff check .`
  - Confirm configs present in deployments/v2_5_incubation/
  - Confirm paper-trading env has market data + scheduling
- At open:
  - Start signal generation + TWAP scheduling
  - Log: target weights, executed weights, fills, slippage vs ref
- Daily:
  - Reconcile sleeve PnL + combined PnL vs backtest MTM; monitor slippage variance
- Alerts:
  - -15% drawdown from HWM -> immediate Slack alert
