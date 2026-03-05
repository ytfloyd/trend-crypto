# JPM Momentum Strategies: Crypto Adaptation
# Kolanovic & Wei (2015) "Momentum Strategies Across Asset Classes"
#
# Package structure:
#   config.py      - Strategy configuration dataclasses
#   data.py        - Data loading from market.duckdb bars_1d view
#   universe.py    - Dynamic universe filtering (ADV, listing age)
#   signals.py     - Momentum signal implementations (RET, MAC, EMAC, BRK, LREG, RADJ)
#   weights.py     - Position sizing (equal-weight, inverse-vol, risk-parity)
#   risk.py        - Risk overlays (vol targeting, stop-loss, mean-reversion)
#   backtest.py    - Portfolio simulation engine
#   metrics.py     - Thin wrapper around existing compute_metrics()
#   grid.py        - Lookback / signal parameter sweep engine
