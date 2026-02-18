# Sornette LPPL Bubble Detection & "Jumpers" Portfolio
#
# Applies the Log-Periodic Power Law Singularity (LPPLS) model from
# Sornette (2003) / Filimonov & Sornette (2013) to detect explosive
# upside moves in digital assets and build portfolios of "jumpers".
#
# Package structure:
#   lppl.py             - LPPL model: formula, linearised fitting, quality metrics
#   bubble_indicator.py - Multi-window bubble/anti-bubble confidence scoring
#   signals.py          - Trading signals: early bubble, anti-bubble reversal
#   portfolio.py        - Portfolio construction for jumpers
#   data.py             - Data loading (crypto from market.duckdb)
#   run_bubble_scan.py  - Scan all tokens for active bubble signatures
#   run_portfolio.py    - Backtest the jumpers portfolio
